import os
import io
import json
import uuid
import base64
import asyncio
import wave
import audioop
import hashlib
import time
import re
import struct
from typing import Dict, Optional, List, Tuple
from collections import deque

from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
from fastapi.responses import Response, PlainTextResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient
from gpu_detection_llm import detect_gpu,public_ws_host,end_call_tool,transfer_call_tool,clean_markdown_for_tts,detect_intent,detect_confirmation_response,parse_llm_response,call_webhook_tool,execute_detected_tool,query_rag_streaming,calculate_audio_energy
from database import get_db,Agent,Conversation,WebhookConfig,PhoneNumber,KnowledgeBase,AgentTool
from models import AgentCreate,AgentUpdate,OutboundCallRequest,WebhookResponse,WebhookCreate,ToolCreate,CallRequest
from tts_stt import setup_streaming_stt

# PyTorch (for GPU detection)
import torch

# RAG stack
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

import httpx

# Deepgram SDK
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    SpeakOptions,
)

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Text, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from datetime import datetime as dt
from database import SessionLocal
load_dotenv()
# ----------------------------
# Environment and configuration
# ----------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_VOICE = os.getenv("DEEPGRAM_VOICE", "aura-2-thalia-en")
DATA_FILE = os.getenv("DATA_FILE", "./data/data.json")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mixtral:8x7b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "384"))
TOP_K = int(os.getenv("TOP_K", "3"))

# 🎯 SMART INTERRUPT SETTINGS - ALL CONFIGURABLE
INTERRUPT_ENABLED = os.getenv("INTERRUPT_ENABLED", "true").lower() == "true"
INTERRUPT_MIN_ENERGY = int(os.getenv("INTERRUPT_MIN_ENERGY", "1000"))
INTERRUPT_DEBOUNCE_MS = int(os.getenv("INTERRUPT_DEBOUNCE_MS", "1000"))
INTERRUPT_BASELINE_FACTOR = float(
    os.getenv("INTERRUPT_BASELINE_FACTOR", "3.5"))
INTERRUPT_MIN_SPEECH_MS = int(os.getenv("INTERRUPT_MIN_SPEECH_MS", "300"))
INTERRUPT_REQUIRE_TEXT = os.getenv(
    "INTERRUPT_REQUIRE_TEXT", "false").lower() == "true"

# âœ… SILENCE DETECTION (matches Deepgram utterance_end_ms)
SILENCE_THRESHOLD_SEC = float(os.getenv("SILENCE_THRESHOLD_SEC", "0.8"))
UTTERANCE_END_MS = int(SILENCE_THRESHOLD_SEC * 1000)

REQUIRE_ENV = [TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
               TWILIO_PHONE_NUMBER, PUBLIC_URL, DEEPGRAM_API_KEY]
if not all(REQUIRE_ENV):
    raise RuntimeError(
        "Missing required env: TWILIO_*, PUBLIC_URL, DEEPGRAM_API_KEY")

# JWT Secret for signed URLs
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")

# API Key Authentication
API_KEY_HEADER = APIKeyHeader(name="xi-api-key", auto_error=False)
API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []

# Webhook Events
WEBHOOK_EVENTS = [
    "call.initiated",
    "call.started", 
    "call.ended",
    "call.failed",
    "transcript.partial",
    "transcript.final",
    "agent.response",
    "tool.called",
    "user.interrupted"
]

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
LOG_FILE = os.getenv("LOG_FILE", "server.log")

_logger = logging.getLogger("new")
_logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
_logger.addHandler(_ch)

try:
    _fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=2)
    _fh.setFormatter(_fmt)
    _logger.addHandler(_fh)
except Exception:
    pass


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
        # 🧠 CALL AWARENESS (ElevenLabs-style)
        self.call_phase: str = "CALL_START"
        self.last_intent: Optional[str] = None

        
        # ✨ NEW: Agent and call data
        self.agent_id: Optional[str] = None
        self.agent_config: Optional[Dict] = None
        self.dynamic_variables: Optional[Dict] = None
        self.custom_first_message: Optional[str] = None

        self.custom_voice_id: Optional[str] = None
        self.custom_model: Optional[str] = None
        self.conversation_id: Optional[str] = None  # For DB tracking

        # Streaming STT
        self.deepgram_live = None
        self.stt_transcript_buffer: str = ""
        self.stt_is_final: bool = False
        self.last_speech_time: float = 0
        self.silence_start: Optional[float] = None

        # Streaming TTS
        # Limit queue size to prevent memory issues
        self.tts_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.tts_task: Optional[asyncio.Task] = None

        # 🎯 SMART VOICE-BASED INTERRUPT DETECTION
        self.user_speech_detected: bool = False
        self.speech_start_time: Optional[float] = None
        self.speech_energy_buffer: deque = deque(maxlen=50)
        self.last_interrupt_time: float = 0
        self.interrupt_debounce: float = INTERRUPT_DEBOUNCE_MS / 1000.0

        # ✅ Baseline starts at 50% of threshold
        self.baseline_energy: float = INTERRUPT_MIN_ENERGY * 0.5
        self.background_samples: deque = deque(maxlen=50)

        # For smarter interrupt gating
        self.last_interim_text: str = ""
        self.last_interim_time: float = 0.0
        self.last_interim_conf: float = 0.0
        self.last_tts_send_time: float = 0.0

        # ✨ Pending action confirmation
        self.pending_action: Optional[dict] = None

        # 🔧 ADD THIS: Speech validation to prevent false positives
        self.false_speech_check_time: Optional[float] = None

        # VAD validation fields
        self.vad_triggered_time: Optional[float] = None
        self.vad_validation_threshold: float = 0.3
        self.vad_validated: bool = False
        self.vad_timeout: float = 5.0
        self.energy_drop_time: Optional[float] = None
        self.last_valid_speech_energy: float = 0.0

        # 🔧 CRITICAL FIX: Session-level resampler state (prevents clicks between responses)
        self.resampler_state = None
        self.resampler_initialized: bool = False


class ConnectionManager:
    def __init__(self):
        self._conns: Dict[str, WSConn] = {}

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

        # ✅ Validate payload is not empty
        if not raw_mulaw_bytes or len(raw_mulaw_bytes) == 0:
            return False

        # ✅ Validate stream_sid
        if not stream_sid or stream_sid != conn.stream_sid:
            _logger.warning(f"Invalid stream_sid: {stream_sid} vs {conn.stream_sid}")
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


manager = ConnectionManager()

# ✨ Store for passing call data from API to WebSocket
# Key: call_sid, Value: {agent_id, dynamic_variables, overrides}
pending_call_data: Dict[str, Dict] = {}


async def save_conversation_transcript(call_sid: str, conn: WSConn):
    """
    Save conversation transcript to database
    
    ✨ ELEVENLABS-COMPATIBLE: Always saves transcript, even if empty
    """
    _logger.info(f"💾 save_conversation_transcript called for {call_sid}")
    _logger.info(f"   - conn exists: {bool(conn)}")
    _logger.info(f"   - conversation_history length: {len(conn.conversation_history) if conn else 0}")
    
    if not conn:
        _logger.warning(f"⚠️ No connection found for {call_sid} - cannot save transcript")
        return
    
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == call_sid
        ).first()
        
        if conversation:
            # Build transcript text
            transcript_lines = []
            for entry in conn.conversation_history:
                user_text = entry.get('user', '')
                assistant_text = entry.get('assistant', '')
                transcript_lines.append(f"User: {user_text}")
                transcript_lines.append(f"Assistant: {assistant_text}")
            
            # ✨ Save transcript even if empty (to show call happened)
            conversation.transcript = "\n".join(transcript_lines) if transcript_lines else "[No conversation - call ended early]"
            conversation.status = "completed"
            conversation.ended_at = dt.utcnow()
            
            # Calculate duration
            if conversation.started_at:
                duration = (conversation.ended_at - conversation.started_at).total_seconds()
                conversation.duration_secs = int(duration)
            
            db.commit()
            _logger.info(f"✅ Saved transcript for {call_sid}")
            _logger.info(f"   - Exchanges: {len(conn.conversation_history)}")
            _logger.info(f"   - Duration: {conversation.duration_secs}s")
            _logger.info(f"   - Transcript length: {len(conversation.transcript)} chars")
        else:
            _logger.warning(f"⚠️ Conversation record not found in DB for {call_sid}")
    except Exception as e:
        _logger.error(f"❌ Failed to save transcript: {e}")
        import traceback
        _logger.error(traceback.format_exc())
        db.rollback()
    finally:
        db.close()

async def handle_call_end(call_sid: str, reason: str):
    """
    Handle call ending - save data and send webhooks
    
    ✨ ELEVENLABS-COMPATIBLE: Always saves transcript and sends webhooks
    """
    conn = manager.get(call_sid)
    
    # Save transcript
    if conn:
        await save_conversation_transcript(call_sid, conn)
    
    db = SessionLocal()
    try:
        # Update conversation
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == call_sid
        ).first()
        
        if conversation:
            conversation.ended_reason = reason
            conversation.status = "completed"
            if not conversation.ended_at:
                conversation.ended_at = dt.utcnow()
            
            # Calculate duration if not already set
            if conversation.started_at and not conversation.duration_secs:
                duration = (conversation.ended_at - conversation.started_at).total_seconds()
                conversation.duration_secs = int(duration)
            
            db.commit()
            
            # Extract direction from call_metadata
            call_direction = "outbound"
            if conversation.call_metadata and isinstance(conversation.call_metadata, dict):
                call_direction = conversation.call_metadata.get("direction", "outbound")
            
            # ✨ ALWAYS send webhooks (like ElevenLabs)
            webhooks = db.query(WebhookConfig).filter(
                WebhookConfig.is_active == True
            ).all()
            
            for webhook in webhooks:
                should_send = False
                if webhook.agent_id is None:
                    should_send = True  # Global webhook
                elif conversation.agent_id and webhook.agent_id == conversation.agent_id:
                    should_send = True  # Agent-specific webhook
                
                if should_send and ("call.ended" in webhook.events or not webhook.events):
                    await send_webhook(
                        webhook.webhook_url,
                        "call.ended",
                        {
                            "conversation_id": call_sid,
                            "agent_id": conversation.agent_id,
                            "duration_secs": conversation.duration_secs,
                            "ended_reason": reason,
                            "transcript": conversation.transcript,
                            "phone_number": conversation.phone_number,
                            "direction": call_direction,
                            "dynamic_variables": conversation.dynamic_variables,
                            "status": "completed"
                        }
                    )
            
            _logger.info(f"✅ Call ended: {call_sid} - reason: {reason} - duration: {conversation.duration_secs}s")
        else:
            _logger.warning(f"⚠️ Conversation not found for call end: {call_sid}")
    except Exception as e:
        _logger.error(f"❌ Failed to handle call end: {e}")
    finally:
        db.close()

def update_baseline(conn: WSConn, energy: int):
    """Update background noise baseline with improved adaptivity"""
    if not conn.currently_speaking:
        if energy < max(conn.baseline_energy * 2, 600):
            conn.background_samples.append(energy)
            if len(conn.background_samples) >= 20:
                recent_samples = list(conn.background_samples)[-20:]
                sorted_samples = sorted(recent_samples)
                weighted_median = sorted_samples[len(sorted_samples) // 2]
                conn.baseline_energy = (
                    conn.baseline_energy * 0.7) + (weighted_median * 0.3)


# ----------------------------
# âš¡ Interrupt Handler
# ----------------------------

async def handle_interrupt(call_sid: str):
    """Handle user interruption with complete cleanup"""
    conn = manager.get(call_sid)
    if not conn:
        return

    _logger.info("ðŸ›‘ INTERRUPT - Stopping playback and clearing buffers")

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
        if conn.stream_sid:  # ✅ Validate stream_sid exists
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

    _logger.info(
        "âœ… Interrupt handled:\n"
        "   Cleared TTS items: %d\n"
        "   Cleared STT buffer: '%s'\n"
        "   Ready for new input",
        cleared, old_buffer[:50] if old_buffer else "(empty)"
    )

async def stream_tts_worker(call_sid: str):
    """âš¡ OPTIMIZED TTS - Fast first response + smooth playback + no clicks"""
    conn = manager.get(call_sid)
    if not conn:
        return

    # Single resampler for entire session (critical for smoothness)
    # persistent_resampler_state = None

    try:
        while True:
            # âœ… SINGLE SENTENCE: Process one sentence at a time
            text = await conn.tts_queue.get()

            if text is None:
                conn.tts_queue.task_done()
                break

            conn.tts_queue.task_done()

            if not text or not text.strip():
                continue

            if conn.interrupt_requested:
                _logger.info("ðŸ›‘ Skipping batch due to interrupt")
                while not conn.tts_queue.empty():
                    try:
                        conn.tts_queue.get_nowait()
                        conn.tts_queue.task_done()
                    except:
                        break
                conn.currently_speaking = False
                conn.interrupt_requested = False
                # persistent_resampler_state = None
                break

            _logger.info("ðŸŽ¤ TTS sentence (%d chars): '%s...'",
                         len(text), text[:80])

            t_start = time.time()
            conn.currently_speaking = True
            conn.speech_energy_buffer.clear()
            conn.speech_start_time = None
            is_first_chunk = True  # Track first chunk of sentence
            audio_chunks_buffer = []  # Buffer to apply fade-out to last chunk

            try:
                url = "https://api.deepgram.com/v1/speak"
                headers = {
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {"text": text}
                
                # ✨ Use custom voice if provided, otherwise agent default, otherwise env default
                voice_to_use = DEEPGRAM_VOICE  # Default from env
                voice_source = "env_default"
                
                # 🔍 DEBUG: Log raw values for debugging
                _logger.debug(f"🔍 TTS Voice Debug - conn.custom_voice_id: '{conn.custom_voice_id}'")
                _logger.debug(f"🔍 TTS Voice Debug - conn.agent_config: {conn.agent_config}")
                
                if conn.custom_voice_id and str(conn.custom_voice_id).strip():
                    voice_to_use = conn.custom_voice_id
                    voice_source = "api_override"
                elif conn.agent_config and conn.agent_config.get("voice_id"):
                    voice_to_use = conn.agent_config["voice_id"]
                    voice_source = "agent_config"
                
                # Log voice selection for EVERY sentence (to debug first message issue)
                _logger.info(f"🎤 TTS Voice: {voice_to_use} (source: {voice_source}) for text: '{text[:50]}...'")
                
                params = {
                    "model": voice_to_use,
                    "encoding": "linear16",
                    "sample_rate": "16000"
                }

                interrupted = False
                chunk_count = 0

                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream("POST", url, json=payload,
                                             headers=headers, params=params) as response:
                        response.raise_for_status()

                        async for audio_chunk in response.aiter_bytes(chunk_size=3200):
                            if conn.interrupt_requested:
                                _logger.info(
                                    "ðŸ›' TTS interrupted at chunk %d", chunk_count)
                                interrupted = True
                                break

                            if len(audio_chunk) == 0:
                                continue

                            try:
                                # ✅ CRITICAL: Ensure resampler is initialized before first chunk
                                if conn.resampler_state is None:
                                    # Initialize resampler with silence
                                    _, conn.resampler_state = audioop.ratecv(
                                        b'\x00' * 160, 2, 1, 16000, 8000, None
                                    )

                                # ✅ CRITICAL: Reuse same resampler state across all sentences
                                pcm_8k, conn.resampler_state = audioop.ratecv(
                                    audio_chunk, 2, 1, 16000, 8000,
                                    conn.resampler_state
                                )

                                # ✅ FIX: Apply fade-in to first chunk to prevent clicks
                                if is_first_chunk and len(pcm_8k) >= 320:
                                    # Convert to list for manipulation
                                    samples = list(struct.unpack(
                                        f'<{len(pcm_8k)//2}h', pcm_8k))

                                    # Apply fade-in to first 160 samples (20ms at 8kHz)
                                    fade_samples = min(160, len(samples))
                                    for i in range(fade_samples):
                                        fade_factor = (i + 1) / fade_samples
                                        samples[i] = int(
                                            samples[i] * fade_factor)

                                    # Repack
                                    pcm_8k = struct.pack(
                                        f'<{len(samples)}h', *samples)
                                    is_first_chunk = False

                                # Buffer the chunk for potential fade-out processing
                                audio_chunks_buffer.append(pcm_8k)
                                
                                # Convert and send buffered chunks (keep last 2 for fade-out)
                                while len(audio_chunks_buffer) > 2:
                                    chunk_to_convert = audio_chunks_buffer.pop(0)
                                    mulaw = audioop.lin2ulaw(chunk_to_convert, 2)

                                    for i in range(0, len(mulaw), 160):
                                        if conn.interrupt_requested:
                                            interrupted = True
                                            break

                                        chunk_to_send = mulaw[i:i+160]
                                        if len(chunk_to_send) < 160:
                                            chunk_to_send += b'\xff' * \
                                                (160 - len(chunk_to_send))

                                        success = await manager.send_media_chunk(
                                            call_sid, conn.stream_sid, chunk_to_send
                                        )
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
                
                # ✅ Process remaining buffered chunks with fade-out on the last one
                if not interrupted and audio_chunks_buffer:
                    for idx, chunk_to_convert in enumerate(audio_chunks_buffer):
                        is_last_chunk = (idx == len(audio_chunks_buffer) - 1)
                        
                        # Apply fade-out to last chunk to prevent clicks between sentences
                        if is_last_chunk and len(chunk_to_convert) >= 320:
                            try:
                                samples = list(struct.unpack(
                                    f'<{len(chunk_to_convert)//2}h', chunk_to_convert))
                                
                                # Apply fade-out to last 160 samples (20ms at 8kHz)
                                fade_samples = min(160, len(samples))
                                start_idx = len(samples) - fade_samples
                                for i in range(fade_samples):
                                    fade_factor = 1.0 - ((i + 1) / fade_samples)
                                    samples[start_idx + i] = int(
                                        samples[start_idx + i] * fade_factor)
                                
                                chunk_to_convert = struct.pack(
                                    f'<{len(samples)}h', *samples)
                            except Exception as e:
                                _logger.warning(f"⚠️ Fade-out failed: {e}")
                        
                        mulaw = audioop.lin2ulaw(chunk_to_convert, 2)
                        
                        for i in range(0, len(mulaw), 160):
                            if conn.interrupt_requested:
                                interrupted = True
                                break

                            chunk_to_send = mulaw[i:i+160]
                            if len(chunk_to_send) < 160:
                                chunk_to_send += b'\xff' * \
                                    (160 - len(chunk_to_send))

                            success = await manager.send_media_chunk(
                                call_sid, conn.stream_sid, chunk_to_send
                            )
                            if not success:
                                interrupted = True
                                break

                            conn.last_tts_send_time = time.time()
                            chunk_count += 1
                            await asyncio.sleep(0.018)

                        if interrupted:
                            break
                    
                    # Clear buffer after processing
                    audio_chunks_buffer.clear()

                t_end = time.time()

                if interrupted:
                    await handle_interrupt(call_sid)
                    # Keep resampler state - don't reset on interrupt
                    while not conn.tts_queue.empty():
                        try:
                            conn.tts_queue.get_nowait()
                            conn.tts_queue.task_done()
                        except:
                            break
                else:
                    _logger.info("âœ… Sentence completed in %.0fms (%d chunks, %.1f chars/sec)",
                                 (t_end - t_start)*1000, chunk_count,
                                 len(text) / (t_end - t_start) if (t_end - t_start) > 0 else 0)

            except Exception as e:
                # ✅ Only reset resampler on serious conversion errors
                if "resampler" in str(e).lower() or "audio" in str(e).lower():
                    conn.resampler_state = None

            # Only clear state when truly done
            if conn.tts_queue.empty():
                conn.currently_speaking = False
                conn.interrupt_requested = False
                conn.speech_energy_buffer.clear()
                conn.speech_start_time = None
                conn.user_speech_detected = False
                # Keep resampler for next turn

    except asyncio.CancelledError:
        pass
    except Exception as e:
        pass
    finally:
        conn.currently_speaking = False
        conn.interrupt_requested = False


async def speak_text_streaming(call_sid: str, text: str):
    """âš¡ Queue text with smart sentence splitting"""
    conn = manager.get(call_sid)
    if not conn or not conn.stream_sid:
        return

    try:
        if conn.stream_sid:  # ✅ Validate stream_sid exists
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

    # âœ… Split into sentences for queue
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in '.!?' and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    # Queue all sentences (worker will batch them automatically)
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

# âš¡ STREAMING STT WITH IMPROVED VAD - Deepgram live + final-guard


async def setup_streaming_stt(call_sid: str):
    """âš¡ Setup Deepgram streaming STT with improved VAD"""
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

                _logger.info("ðŸŽ™ï¸ STT %s: '%s'",
                             "FINAL" if is_final else "interim", transcript)

                # âœ… Always update speech time when we receive text
                conn.last_speech_time = now

                if is_final:
                    # ========================================
                    # âœ… FINAL RESULT - ALWAYS ACCUMULATE
                    # ========================================
                    current_buffer = conn.stt_transcript_buffer.strip()

                    if current_buffer:
                        # Check if this continues the current thought
                        if (not current_buffer.endswith((".", "!", "?")) and
                                len(transcript) > 3):
                            # Continue the sentence
                            conn.stt_transcript_buffer += " " + transcript
                            _logger.info(
                                f"âž• Appending to sentence: '{transcript}'")
                        else:
                            # New thought or refinement
                            conn.stt_transcript_buffer = transcript
                            _logger.info(f"ðŸ”„ New sentence: '{transcript}'")
                    else:
                        # First content
                        conn.stt_transcript_buffer = transcript

                    # Mark that we have FINAL text
                    conn.stt_is_final = True

                    _logger.info(
                        f"ðŸ“ Complete buffer: '{conn.stt_transcript_buffer.strip()}'")

                else:
                    # ========================================
                    # âœ… INTERIM RESULT - TRACK BUT DON'T OVERWRITE
                    # ========================================

                    # Track interim time for activity detection
                    conn.last_interim_time = now
                    conn.last_interim_text = transcript

                    # Only use interim if we have no FINAL content yet
                    if not conn.stt_transcript_buffer or not conn.stt_is_final:
                        conn.stt_transcript_buffer = transcript
                        _logger.info(f"ðŸ“ Interim as buffer: '{transcript}'")

            except Exception as e:
                pass

        def on_open(self, open, **kwargs):
            pass

        def on_error(self, error, **kwargs):
            pass

        def on_close(self, close_msg, **kwargs):
            pass

        def on_speech_started(self, speech_started, **kwargs):
            """âœ… FIXED: Mark VAD trigger but require validation"""
            conn.vad_triggered_time = time.time()
            conn.user_speech_detected = True  # Tentatively set
            conn.speech_start_time = time.time()
            _logger.info("ðŸŽ¤ VAD: Speech trigger (needs validation)")

        def on_utterance_end(self, utterance_end, **kwargs):
            """âœ… FIXED: Clear VAD when Deepgram confirms utterance ended"""
            now = time.time()

            # Check if we got interim text very recently (within 200ms)
            if conn.last_interim_time and (now - conn.last_interim_time) < 0.2:
                _logger.info(
                    "â­ï¸ UtteranceEnd ignored - recent interim detected")
                return

            # âœ… Clear VAD state when Deepgram confirms end
            if conn.user_speech_detected:
                _logger.info(
                    "âœ… UtteranceEnd - clearing VAD (Deepgram confirmed)")
                conn.user_speech_detected = False
                conn.speech_start_time = None
                conn.vad_triggered_time = None
                conn.vad_validated = False
                conn.energy_drop_time = None

            conn.last_speech_time = now
            _logger.info(f"ðŸ•’ UtteranceEnd - last_speech_time: {now}")

        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(
            LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd,
                         on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)

        # Minimal, safe options for Twilio mu-law 8k (works on deepgram-sdk 3.2)
        options = LiveOptions(
            model=os.getenv("DEEPGRAM_STT_MODEL", "nova-2"),
            language="en-US",
            smart_format=True,
            interim_results=True,
            vad_events=True,
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            # If you want Deepgram to emit UtteranceEnd reliably, try enabling endpointing:
            # uncomment to try (if your project supports it)
            endpointing=UTTERANCE_END_MS,
        )

        # start() is synchronous and returns bool in SDK 3.2
        start_ok = False
        try:
            start_ok = dg_connection.start(options)
        except Exception as e:
            pass

        if not start_ok:
            fallback = LiveOptions(
                model=os.getenv("DEEPGRAM_STT_FALLBACK_MODEL",
                                "nova-2-general"),
                encoding="mulaw",
                sample_rate=8000,
                interim_results=True,
                # utterance_end_ms=UTTERANCE_END_MS,  # optional legacy param if endpointing not supported
            )
            try:
                start_ok = dg_connection.start(fallback)
            except Exception as e2:
                return

        if start_ok:
            conn.deepgram_live = dg_connection
            _logger.info("âœ… Streaming STT initialized")
        else:
            _logger.error(
                "âŒ Deepgram start() returned False (model/options/API key)")

    except Exception as e:
        pass