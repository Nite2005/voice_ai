import asyncio
import time
import re
import base64
import struct
import audioop
from typing import Dict, Optional, List, Tuple
from collections import deque
from datetime import datetime as dt

import httpx
from fastapi import WebSocket, WebSocketDisconnect
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

from models import SessionLocal, Conversation, WebhookConfig
from config import (
    logger, DEEPGRAM_API_KEY, DEEPGRAM_VOICE, SILENCE_THRESHOLD_SEC,
    INTERRUPT_ENABLED, INTERRUPT_MIN_ENERGY, INTERRUPT_BASELINE_FACTOR,
    INTERRUPT_MIN_SPEECH_MS, INTERRUPT_DEBOUNCE_MS
)
from agents import send_webhook, send_webhook_and_get_response

# Deepgram client setup
deepgram_config = DeepgramClientOptions(options={"keepalive": "true", "timeout": "60"})
deepgram = DeepgramClient(DEEPGRAM_API_KEY, config=deepgram_config)


# ----------------------------
# WebSocket Connection Manager
# ----------------------------
class WSConn:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.stream_sid: Optional[str] = None
        self.inbound_ulaw_buffer: bytearray = bytearray()
        self.is_responding: bool = False
        self.last_transcript: str = ""
        self.stream_ready: bool = False
        self.speech_detected: bool = False
        self.currently_speaking: bool = False
        self.interrupt_requested: bool = False
        self.conversation_history: List[Dict[str, str]] = []
        self.call_phase: str = "CALL_START"
        self.last_intent: Optional[str] = None
        
        # Agent and call data
        self.agent_id: Optional[str] = None
        self.agent_config: Optional[Dict] = None
        self.dynamic_variables: Optional[Dict] = None
        self.custom_first_message: Optional[str] = None
        self.custom_voice_id: Optional[str] = None
        self.custom_model: Optional[str] = None
        self.conversation_id: Optional[str] = None

        # Streaming STT
        self.deepgram_live = None
        self.stt_transcript_buffer: str = ""
        self.stt_is_final: bool = False
        self.last_speech_time: float = 0
        self.silence_start: Optional[float] = None

        # Streaming TTS
        self.tts_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.tts_task: Optional[asyncio.Task] = None

        # Smart voice-based interrupt detection
        self.user_speech_detected: bool = False
        self.speech_start_time: Optional[float] = None
        self.speech_energy_buffer: deque = deque(maxlen=50)
        self.last_interrupt_time: float = 0
        self.interrupt_debounce: float = INTERRUPT_DEBOUNCE_MS / 1000.0
        self.baseline_energy: float = INTERRUPT_MIN_ENERGY * 0.5
        self.background_samples: deque = deque(maxlen=50)

        # For smarter interrupt gating
        self.last_interim_text: str = ""
        self.last_interim_time: float = 0.0
        self.last_interim_conf: float = 0.0
        self.last_tts_send_time: float = 0.0

        # Pending action confirmation
        self.pending_action: Optional[dict] = None

        # Speech validation
        self.false_speech_check_time: Optional[float] = None

        # VAD validation fields
        self.vad_triggered_time: Optional[float] = None
        self.vad_validation_threshold: float = 0.3
        self.vad_validated: bool = False
        self.vad_timeout: float = 5.0
        self.energy_drop_time: Optional[float] = None
        self.last_valid_speech_energy: float = 0.0

        # Resampler state
        self.resampler_state = None
        self.resampler_initialized: bool = False


class ConnectionManager:
    def __init__(self):
        self._conns: Dict[str, WSConn] = {}
        self.pending_call_data: Dict[str, Dict] = {}

    async def connect(self, call_sid: str, ws: WebSocket):
        self._conns[call_sid] = WSConn(ws)

    async def disconnect(self, call_sid: str):
        conn = self._conns.pop(call_sid, None)
        if conn:
            if conn.deepgram_live:
                try:
                    conn.deepgram_live.finish()
                except:
                    pass

            if conn.tts_task and not conn.tts_task.done():
                conn.tts_task.cancel()

            try:
                await conn.ws.close()
            except Exception:
                pass

    def get(self, call_sid: str) -> Optional[WSConn]:
        return self._conns.get(call_sid)

    async def send_media_chunk(self, call_sid: str, stream_sid: str, raw_mulaw_bytes: bytes):
        conn = self.get(call_sid)
        if not conn or not conn.ws or not conn.stream_ready:
            return False

        if conn.interrupt_requested:
            return False

        if not raw_mulaw_bytes or len(raw_mulaw_bytes) == 0:
            return False

        if not stream_sid or stream_sid != conn.stream_sid:
            logger.warning(f"Invalid stream_sid: {stream_sid} vs {conn.stream_sid}")
            return False

        payload = base64.b64encode(raw_mulaw_bytes).decode("utf-8")

        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": payload
            }
        }

        try:
            await conn.ws.send_json(msg)
            return True
        except Exception as e:
            return False

    def set_pending_call_data(self, call_sid: str, data: dict):
        self.pending_call_data[call_sid] = data

    def get_pending_call_data(self, call_sid: str) -> Optional[dict]:
        return self.pending_call_data.get(call_sid)


manager = ConnectionManager()


# ----------------------------
# Helper Functions
# ----------------------------
def calculate_audio_energy(mulaw_bytes: bytes) -> int:
    """Calculate RMS energy of audio chunk"""
    if not mulaw_bytes or len(mulaw_bytes) < 160:
        return 0
    try:
        pcm = audioop.ulaw2lin(mulaw_bytes, 2)
        return audioop.rms(pcm, 2)
    except Exception:
        return 0


def update_baseline(conn: WSConn, energy: int):
    """Update background noise baseline with improved adaptivity"""
    if not conn.currently_speaking:
        if energy < max(conn.baseline_energy * 2, 600):
            conn.background_samples.append(energy)
            if len(conn.background_samples) >= 20:
                recent_samples = list(conn.background_samples)[-20:]
                sorted_samples = sorted(recent_samples)
                weighted_median = sorted_samples[len(sorted_samples) // 2]
                conn.baseline_energy = (conn.baseline_energy * 0.7) + (weighted_median * 0.3)


async def handle_interrupt(call_sid: str):
    """Handle user interruption with complete cleanup"""
    conn = manager.get(call_sid)
    if not conn:
        return

    logger.info("🛑 INTERRUPT - Stopping playback and clearing buffers")
    conn.interrupt_requested = True

    cleared = 0
    while not conn.tts_queue.empty():
        try:
            conn.tts_queue.get_nowait()
            conn.tts_queue.task_done()
            cleared += 1
        except:
            break

    try:
        if conn.stream_sid:
            await conn.ws.send_json({
                "event": "clear",
                "streamSid": conn.stream_sid
            })
    except:
        pass

    old_buffer = conn.stt_transcript_buffer
    conn.stt_transcript_buffer = ""
    conn.stt_is_final = False
    conn.last_transcript = ""
    conn.currently_speaking = False
    conn.is_responding = False
    conn.speech_energy_buffer.clear()
    conn.speech_start_time = None
    conn.user_speech_detected = False
    conn.last_speech_time = 0
    conn.silence_start = None
    conn.last_interim_text = ""
    conn.last_interim_time = 0.0
    conn.last_interim_conf = 0.0

    buffer_preview = old_buffer[:50] if old_buffer else "(empty)"

# logger.info(
#     f"✅ Interrupt handled:\n"
#     f"   Cleared TTS items: {cleared}\n"
#     f"   Cleared STT buffer: '{buffer_preview}'\n"
#     "   Ready for new input"
# )



# ----------------------------
# TTS Functions
# ----------------------------
async def stream_tts_worker(call_sid: str):
    """Optimized TTS - Fast first response + smooth playback + no clicks"""
    conn = manager.get(call_sid)
    if not conn:
        return

    try:
        while True:
            text = await conn.tts_queue.get()
            if text is None:
                conn.tts_queue.task_done()
                break
            conn.tts_queue.task_done()

            if not text or not text.strip():
                continue

            if conn.interrupt_requested:
                logger.info("🛑 Skipping batch due to interrupt")
                while not conn.tts_queue.empty():
                    try:
                        conn.tts_queue.get_nowait()
                        conn.tts_queue.task_done()
                    except:
                        break
                conn.currently_speaking = False
                conn.interrupt_requested = False
                break

            logger.info(f"🎵 TTS sentence ({len(text)} chars): '{text[:80]}...'")
            t_start = time.time()
            conn.currently_speaking = True
            conn.speech_energy_buffer.clear()
            conn.speech_start_time = None
            is_first_chunk = True
            audio_chunks_buffer = []

            try:
                url = "https://api.deepgram.com/v1/speak"
                headers = {
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {"text": text}
                
                voice_to_use = DEEPGRAM_VOICE
                if conn.custom_voice_id and str(conn.custom_voice_id).strip():
                    voice_to_use = conn.custom_voice_id
                elif conn.agent_config and conn.agent_config.get("voice_id"):
                    voice_to_use = conn.agent_config["voice_id"]
                
                logger.info(f"🎤 TTS Voice: {voice_to_use} for text: '{text[:50]}...'")
                
                params = {
                    "model": voice_to_use,
                    "encoding": "linear16",
                    "sample_rate": "16000"
                }

                interrupted = False
                chunk_count = 0

                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream("POST", url, json=payload, headers=headers, params=params) as response:
                        response.raise_for_status()

                        async for audio_chunk in response.aiter_bytes(chunk_size=3200):
                            if conn.interrupt_requested:
                                logger.info(f"🛑 TTS interrupted at chunk {chunk_count}")
                                interrupted = True
                                break

                            if len(audio_chunk) == 0:
                                continue

                            try:
                                if conn.resampler_state is None:
                                    _, conn.resampler_state = audioop.ratecv(
                                        b'\x00' * 160, 2, 1, 16000, 8000, None
                                    )

                                pcm_8k, conn.resampler_state = audioop.ratecv(
                                    audio_chunk, 2, 1, 16000, 8000, conn.resampler_state
                                )

                                if is_first_chunk and len(pcm_8k) >= 320:
                                    samples = list(struct.unpack(f'<{len(pcm_8k)//2}h', pcm_8k))
                                    fade_samples = min(160, len(samples))
                                    for i in range(fade_samples):
                                        fade_factor = (i + 1) / fade_samples
                                        samples[i] = int(samples[i] * fade_factor)
                                    pcm_8k = struct.pack(f'<{len(samples)}h', *samples)
                                    is_first_chunk = False

                                audio_chunks_buffer.append(pcm_8k)
                                
                                while len(audio_chunks_buffer) > 2:
                                    chunk_to_convert = audio_chunks_buffer.pop(0)
                                    mulaw = audioop.lin2ulaw(chunk_to_convert, 2)

                                    for i in range(0, len(mulaw), 160):
                                        if conn.interrupt_requested:
                                            interrupted = True
                                            break

                                        chunk_to_send = mulaw[i:i+160]
                                        if len(chunk_to_send) < 160:
                                            chunk_to_send += b'\xff' * (160 - len(chunk_to_send))

                                        success = await manager.send_media_chunk(call_sid, conn.stream_sid, chunk_to_send)
                                        if not success:
                                            interrupted = True
                                            break

                                        conn.last_tts_send_time = time.time()
                                        chunk_count += 1
                                        await asyncio.sleep(0.018)

                                    if interrupted:
                                        break

                            except Exception as e:
                                continue
                
                if not interrupted and audio_chunks_buffer:
                    for idx, chunk_to_convert in enumerate(audio_chunks_buffer):
                        is_last_chunk = (idx == len(audio_chunks_buffer) - 1)
                        
                        if is_last_chunk and len(chunk_to_convert) >= 320:
                            try:
                                samples = list(struct.unpack(f'<{len(chunk_to_convert)//2}h', chunk_to_convert))
                                fade_samples = min(160, len(samples))
                                start_idx = len(samples) - fade_samples
                                for i in range(fade_samples):
                                    fade_factor = 1.0 - ((i + 1) / fade_samples)
                                    samples[start_idx + i] = int(samples[start_idx + i] * fade_factor)
                                chunk_to_convert = struct.pack(f'<{len(samples)}h', *samples)
                            except Exception as e:
                                logger.warning(f"⚠️ Fade-out failed: {e}")
                        
                        mulaw = audioop.lin2ulaw(chunk_to_convert, 2)
                        
                        for i in range(0, len(mulaw), 160):
                            if conn.interrupt_requested:
                                interrupted = True
                                break

                            chunk_to_send = mulaw[i:i+160]
                            if len(chunk_to_send) < 160:
                                chunk_to_send += b'\xff' * (160 - len(chunk_to_send))

                            success = await manager.send_media_chunk(call_sid, conn.stream_sid, chunk_to_send)
                            if not success:
                                interrupted = True
                                break

                            conn.last_tts_send_time = time.time()
                            chunk_count += 1
                            await asyncio.sleep(0.018)

                        if interrupted:
                            break
                    
                    audio_chunks_buffer.clear()

                t_end = time.time()

                if interrupted:
                    await handle_interrupt(call_sid)
                    while not conn.tts_queue.empty():
                        try:
                            conn.tts_queue.get_nowait()
                            conn.tts_queue.task_done()
                        except:
                            break
                else:
                    logger.info(
                        f"✅ Sentence completed in {(t_end - t_start)*1000:.0f}ms "
                        f"({chunk_count} chunks, {len(text) / (t_end - t_start) if (t_end - t_start) > 0 else 0:.1f} chars/sec)"
                    )

            except Exception as e:
                if "resampler" in str(e).lower() or "audio" in str(e).lower():
                    conn.resampler_state = None

            if conn.tts_queue.empty():
                conn.currently_speaking = False
                conn.interrupt_requested = False
                conn.speech_energy_buffer.clear()
                conn.speech_start_time = None
                conn.user_speech_detected = False

    except asyncio.CancelledError:
        pass
    except Exception as e:
        pass
    finally:
        conn.currently_speaking = False
        conn.interrupt_requested = False


async def speak_text_streaming(call_sid: str, text: str):
    """Queue text with smart sentence splitting"""
    conn = manager.get(call_sid)
    if not conn or not conn.stream_sid:
        return

    try:
        if conn.stream_sid:
            await conn.ws.send_json({
                "event": "clear",
                "streamSid": conn.stream_sid
            })
    except:
        pass

    conn.currently_speaking = True
    conn.interrupt_requested = False
    conn.speech_energy_buffer.clear()
    conn.user_speech_detected = False

    sentences = []
    current = ""
    for char in text:
        current += char
        if char in '.!?' and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    for sentence in sentences:
        if sentence:
            try:
                await asyncio.wait_for(conn.tts_queue.put(sentence), timeout=2.0)
            except asyncio.TimeoutError:
                break
            except Exception as e:
                break

    await conn.tts_queue.join()
    conn.currently_speaking = False


# ----------------------------
# STT Functions
# ----------------------------
async def setup_streaming_stt(call_sid: str):
    """Setup Deepgram streaming STT with improved VAD"""
    conn = manager.get(call_sid)
    if not conn:
        return

    try:
        dg_connection = deepgram.listen.live.v("1")

        def on_message(self, result, **kwargs):
            try:
                if not result or not result.channel:
                    return
                alt = result.channel.alternatives[0]
                transcript = alt.transcript
                if not transcript:
                    return

                is_final = result.is_final
                now = time.time()

                logger.info(f"🎤 STT {'FINAL' if is_final else 'interim'}: '{transcript}'")
                conn.last_speech_time = now

                if is_final:
                    current_buffer = conn.stt_transcript_buffer.strip()
                    if current_buffer:
                        if (not current_buffer.endswith((".", "!", "?")) and len(transcript) > 3):
                            conn.stt_transcript_buffer += " " + transcript
                            logger.info(f"➡️ Appending to sentence: '{transcript}'")
                        else:
                            conn.stt_transcript_buffer = transcript
                            logger.info(f"📝 New sentence: '{transcript}'")
                    else:
                        conn.stt_transcript_buffer = transcript
                    conn.stt_is_final = True
                    logger.info(f"📋 Complete buffer: '{conn.stt_transcript_buffer.strip()}'")
                else:
                    conn.last_interim_time = now
                    conn.last_interim_text = transcript
                    if not conn.stt_transcript_buffer or not conn.stt_is_final:
                        conn.stt_transcript_buffer = transcript
                        logger.info(f"📝 Interim as buffer: '{transcript}'")

            except Exception as e:
                pass

        def on_open(self, open, **kwargs):
            pass

        def on_error(self, error, **kwargs):
            pass

        def on_close(self, close_msg, **kwargs):
            pass

        def on_speech_started(self, speech_started, **kwargs):
            conn.vad_triggered_time = time.time()
            conn.user_speech_detected = True
            conn.speech_start_time = time.time()
            logger.info("🎤 VAD: Speech trigger (needs validation)")

        def on_utterance_end(self, utterance_end, **kwargs):
            now = time.time()
            if conn.last_interim_time and (now - conn.last_interim_time) < 0.2:
                logger.info("⏸️ UtteranceEnd ignored - recent interim detected")
                return

            if conn.user_speech_detected:
                logger.info("✅ UtteranceEnd - clearing VAD (Deepgram confirmed)")
                conn.user_speech_detected = False
                conn.speech_start_time = None
                conn.vad_triggered_time = None
                conn.vad_validated = False
                conn.energy_drop_time = None

            conn.last_speech_time = now
            logger.info(f"⏱️ UtteranceEnd - last_speech_time: {now}")

        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)

        options = LiveOptions(
            model=os.getenv("DEEPGRAM_STT_MODEL", "nova-2"),
            language="en-US",
            smart_format=True,
            interim_results=True,
            vad_events=True,
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            endpointing=UTTERANCE_END_MS,
        )

        start_ok = False
        try:
            start_ok = dg_connection.start(options)
        except Exception as e:
            pass

        if not start_ok:
            fallback = LiveOptions(
                model=os.getenv("DEEPGRAM_STT_FALLBACK_MODEL", "nova-2-general"),
                encoding="mulaw",
                sample_rate=8000,
                interim_results=True,
            )
            try:
                start_ok = dg_connection.start(fallback)
            except Exception as e2:
                return

        if start_ok:
            conn.deepgram_live = dg_connection
            logger.info("✅ Streaming STT initialized")
        else:
            logger.error("❌ Deepgram start() returned False (model/options/API key)")

    except Exception as e:
        pass