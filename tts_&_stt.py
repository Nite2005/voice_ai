# text to speech(tts) & speech to text(stt)

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

load_dotenv()

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



# ----------------------------
# âš¡ STREAMING TTS
# ----------------------------


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
