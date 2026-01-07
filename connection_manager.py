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