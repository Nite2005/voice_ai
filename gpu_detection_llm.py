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
from sqlalchemy.orm import Session
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
from fastapi.responses import Response, PlainTextResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Text, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from datetime import datetime as dt
from dotenv import load_dotenv
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
from connection_manager import manager

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


# ----------------------------
# 🚀 GPU DETECTION & OPTIMIZATION
# ----------------------------


def detect_gpu():
    """Detect and configure GPU"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda

        _logger.info("=" * 60)
        _logger.info("🚀 GPU DETECTED!")
        _logger.info(f"   Device: {gpu_name}")
        _logger.info(f"   Count: {gpu_count}")
        _logger.info(f"   Memory: {gpu_memory:.2f} GB")
        _logger.info(f"   CUDA: {cuda_version}")
        _logger.info("=" * 60)

        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            _logger.info("âœ… TF32 enabled (Ampere+ GPU)")

        torch.cuda.empty_cache()

    elif torch.backends.mps.is_available():
        device = 'mps'
        _logger.info("=" * 60)
        _logger.info("🚀 Apple Silicon GPU detected")
        _logger.info("=" * 60)
    else:
        device = 'cpu'
        _logger.warning("=" * 60)
        _logger.warning("âš ï¸  NO GPU DETECTED - Using CPU")
        _logger.warning("=" * 60)

    return device


DEVICE = detect_gpu()

_logger.info("🚀 Config: PUBLIC_URL=%s DEVICE=%s", PUBLIC_URL, DEVICE)
_logger.info("🎯 Interrupt: ENABLED=%s MIN_SPEECH=%dms MIN_ENERGY=%d BASELINE_FACTOR=%.1f",
             INTERRUPT_ENABLED, INTERRUPT_MIN_SPEECH_MS, INTERRUPT_MIN_ENERGY, INTERRUPT_BASELINE_FACTOR)
_logger.info("â±ï¸  Silence Threshold: %.1fs (utterance_end=%dms)",
             SILENCE_THRESHOLD_SEC, UTTERANCE_END_MS)


def public_ws_host() -> str:
    host = PUBLIC_URL.replace(
        "https://", "").replace("http://", "").rstrip("/")
    return host


# ----------------------------
# Clients
# ----------------------------
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

deepgram_config = DeepgramClientOptions(
    options={"keepalive": "true", "timeout": "60"})
deepgram = DeepgramClient(DEEPGRAM_API_KEY, config=deepgram_config)

# ----------------------------
# 🚀 GPU-ACCELERATED RAG
# ----------------------------
_logger.info(f"📦 Loading SentenceTransformer on {DEVICE}...")
start_time = time.time()

embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
embedder.eval()

if DEVICE == 'cuda':
    try:
        embedder.half()
        _logger.info("âœ… FP16 precision enabled")
    except Exception as e:
        _logger.warning(f"âš ï¸  Could not enable FP16: {e}")

load_time = time.time() - start_time
_logger.info(f"âœ… Model loaded in {load_time:.2f}s")

_logger.info("🔥 Warming up GPU...")
with torch.no_grad():
    _ = embedder.encode(
        ["warmup sentence for GPU initialization"],
        device=DEVICE,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=1
    )
_logger.info("âœ… GPU warmed up")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("docs")

response_cache = {}


# ðŸ› ï¸ LLM-CONTROLLED TOOL SYSTEM
# ----------------------------


async def end_call_tool(call_sid: str, reason: str = "user_goodbye") -> dict:
    """End the active call"""
    _logger.info(f"ðŸ”š END_CALL: call_sid={call_sid}, reason={reason}")

    try:
        await asyncio.sleep(1.5)
        call = twilio_client.calls(call_sid).update(status="completed")
        await manager.disconnect(call_sid)

        return {
            "success": True,
            "message": f"Call ended: {reason}",
            "call_sid": call_sid
        }
    except Exception as e:
        _logger.error(f"âŒ Failed to end call: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def transfer_call_tool(call_sid: str, department: str = "sales") -> dict:
    """Transfer call to human agent - executes AFTER message is spoken"""
    _logger.info(f"ðŸ”€ TRANSFER_CALL: call_sid={call_sid}, dept={department}")

    DEPARTMENT_NUMBERS = {
        "sales": os.getenv("SALES_PHONE_NUMBER", "+918107061392"),
        "support": os.getenv("SUPPORT_PHONE_NUMBER", "+918107061392"),
        "technical": os.getenv("TECH_PHONE_NUMBER", "+918107061392"),
    }

    try:
        transfer_number = DEPARTMENT_NUMBERS.get(
            department, DEPARTMENT_NUMBERS["sales"])

        conn = manager.get(call_sid)
        if not conn:
            return {"success": False, "error": "Connection not found"}

        _logger.info("â³ Waiting for transfer message to be spoken...")
        await asyncio.sleep(3.0)

        conn.interrupt_requested = True

        while not conn.tts_queue.empty():
            try:
                conn.tts_queue.get_nowait()
                conn.tts_queue.task_done()
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

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>{transfer_number}</Dial>
</Response>"""

        twilio_client.calls(call_sid).update(twiml=twiml)

        _logger.info(
            f"âœ… Transfer completed to {department} ({transfer_number})")

        return {
            "success": True,
            "transfer_to": transfer_number,
            "department": department,
            "message": f"Transferred to {department}"
        }

    except Exception as e:
        _logger.error(f"âŒ Transfer failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
def clean_markdown_for_tts(text: str) -> str:
    """Remove markdown formatting before TTS to prevent reading symbols aloud"""
    # Remove bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # Remove italic: *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # Remove strikethrough: ~~text~~
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    
    # Remove code blocks: `text` or ```text```
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Remove links: [text](url) -> text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Remove headers: # text -> text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bullet points: - text or * text -> text
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered lists: 1. text -> text
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_markdown_for_tts(text: str) -> str:
    """Remove markdown formatting before TTS to prevent reading symbols aloud"""
    # Remove bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # Remove italic: *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # Remove strikethrough: ~~text~~
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    
    # Remove code blocks: `text` or ```text```
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Remove links: [text](url) -> text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Remove headers: # text -> text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bullet points: - text or * text -> text
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered lists: 1. text -> text
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_markdown_for_tts(text: str) -> str:
    """Remove markdown formatting before TTS to prevent reading symbols aloud"""
    # Remove bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # Remove italic: *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # Remove strikethrough: ~~text~~
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    
    # Remove code blocks: `text` or ```text```
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Remove links: [text](url) -> text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Remove headers: # text -> text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bullet points: - text or * text -> text
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered lists: 1. text -> text
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def detect_intent(text: str) -> str:
    """✨ HUMAN-LIKE: Only detect GOODBYE - let LLM handle everything else naturally"""
    t = text.lower().strip()

    # Only detect goodbye for call termination
    if any(x in t for x in ["bye", "goodbye", "end the call", "that's all", "talk later"]):
        return "GOODBYE"

    # Everything else goes to LLM for contextual, human-like responses
    return "QUESTION"


def detect_confirmation_response(text: str) -> Optional[str]:
    """
    Detect if user is confirming or rejecting a pending action.
    Returns: "yes", "no", or None
    """
    text_lower = text.lower().strip()

    yes_patterns = [
        "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "please",
        "go ahead", "do it", "that's fine", "sounds good",
        "yes please", "yeah please", "sure thing", "absolutely",
        "correct", "right", "affirmative", "proceed", "transfer me",
        "let's do it", "fine", "alright", "all right"
    ]

    no_patterns = [
        "no", "nope", "nah", "not yet", "not now", "maybe later",
        "don't", "wait", "hold on", "cancel", "never mind",
        "not right now", "i'll think about it", "let me think",
        "not really", "not interested"
    ]

    for pattern in yes_patterns:
        if pattern == text_lower or pattern in text_lower:
            if "not " not in text_lower and "no " not in text_lower[:3]:
                return "yes"

    for pattern in no_patterns:
        if pattern == text_lower or pattern in text_lower:
            return "no"

    return None


def parse_llm_response(text: str) -> Tuple[str, Optional[dict]]:
    """
    Parse LLM response for tool calls.
    Format: [TOOL:tool_name:param1:param2] for immediate execution
            [CONFIRM_TOOL:tool_name:param1] for confirmation requests

    Returns: (clean_text, tool_data)
    """

    tool_pattern = r'\[TOOL:([^\]]+)\]'
    confirm_pattern = r'\[CONFIRM_TOOL:([^\]]+)\]'

    tool_data = None

    confirm_matches = re.findall(confirm_pattern, text)
    if confirm_matches:
        tool_parts = confirm_matches[0].split(':')
        tool_name = tool_parts[0].strip()

        if tool_name == "transfer":
            department = tool_parts[1].strip() if len(
                tool_parts) > 1 else "sales"

            valid_departments = ["sales", "support", "technical"]
            if department not in valid_departments:
                _logger.warning(
                    f"âŒ Invalid department in CONFIRM_TOOL: {department} - ignoring tool call")
            else:
                tool_data = {
                    "tool": "transfer_call",
                    "params": {"department": department},
                    "requires_confirmation": True
                }
    else:
        tool_matches = re.findall(tool_pattern, text)
        if tool_matches:
            tool_parts = tool_matches[0].split(':')
            tool_name = tool_parts[0].strip()

            if tool_name == "end_call":
                tool_data = {
                    "tool": "end_call",
                    "params": {"reason": "user_requested"},
                    "requires_confirmation": False
                }
            elif tool_name == "transfer":
                department = tool_parts[1].strip() if len(
                    tool_parts) > 1 else "sales"

                valid_departments = ["sales", "support", "technical"]
                if department not in valid_departments:
                    _logger.warning(
                        f"âŒ Invalid department in TOOL: {department} - ignoring tool call")
                else:
                    tool_data = {
                        "tool": "transfer_call",
                        "params": {"department": department},
                        "requires_confirmation": False
                    }
            else:
                # Generic custom tool handler
                # Format: [TOOL:tool_name:param1:value1:param2:value2]
                # Or: [TOOL:tool_name:value1:value2] (positional)
                tool_params = {}
                
                # Try to parse as key:value pairs or positional args
                if len(tool_parts) > 1:
                    # Check if it looks like key:value (contains = or looks paired)
                    remaining_parts = tool_parts[1:]
                    
                    # Try positional parameters first (simpler)
                    for idx, part in enumerate(remaining_parts):
                        tool_params[f"param{idx+1}"] = part.strip()
                
                tool_data = {
                    "tool": tool_name,
                    "params": tool_params,
                    "requires_confirmation": False
                }

    # Remove tool markers from text
    clean_text = re.sub(tool_pattern, '', text)
    clean_text = re.sub(confirm_pattern, '', clean_text)
    clean_text = clean_text.strip()

    return clean_text, tool_data


async def call_webhook_tool(webhook_url: str, tool_name: str, parameters: dict, call_context: dict) -> dict:
    """Call an external webhook tool and return the response"""
    try:
        payload = {
            "tool_name": tool_name,
            "parameters": parameters,
            "call_context": call_context,
            "timestamp": dt.utcnow().isoformat()
        }
        
        _logger.info(f"🔧 Calling webhook tool: {tool_name} at {webhook_url}")
        _logger.debug(f"   Parameters: {parameters}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                _logger.info(f"✅ Webhook tool response: {result}")
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "response": result.get("response", result),
                    "data": result.get("data", {}),
                    "message": result.get("message", "")
                }
            else:
                _logger.error(f"❌ Webhook returned status {response.status_code}")
                return {
                    "success": False,
                    "error": f"Webhook returned status {response.status_code}",
                    "tool_name": tool_name
                }
                
    except asyncio.TimeoutError:
        _logger.error(f"❌ Webhook timeout for {tool_name}")
        return {
            "success": False,
            "error": "Tool request timed out",
            "tool_name": tool_name
        }
    except Exception as e:
        _logger.error(f"❌ Webhook error for {tool_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool_name": tool_name
        }


async def execute_detected_tool(call_sid: str, tool_data: dict) -> dict:
    """Execute a tool that was detected from LLM response"""
    tool_name = tool_data["tool"]
    params = tool_data.get("params", {})

    _logger.info(
        f"🔧 Executing LLM-requested tool: {tool_name} with params: {params}")

    # Built-in tools
    if tool_name == "end_call":
        result = await end_call_tool(call_sid, **params)
    elif tool_name == "transfer_call":
        result = await transfer_call_tool(call_sid, **params)
    else:
        # Try to find custom webhook tool
        conn = manager.get(call_sid)
        if not conn or not conn.agent_id:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        db = SessionLocal()
        try:
            # Look up the tool in the database
            tool = db.query(AgentTool).filter(
                AgentTool.agent_id == conn.agent_id,
                AgentTool.tool_name == tool_name,
                AgentTool.is_active == True
            ).first()
            
            if not tool or not tool.webhook_url:
                return {"success": False, "error": f"Unknown or inactive tool: {tool_name}"}
            
            # Prepare call context
            call_context = {
                "call_sid": call_sid,
                "agent_id": conn.agent_id,
                "conversation_id": conn.conversation_id,
                "phone_number": None,
                "dynamic_variables": conn.dynamic_variables or {}
            }
            
            # Call the webhook
            result = await call_webhook_tool(
                webhook_url=tool.webhook_url,
                tool_name=tool_name,
                parameters=params,
                call_context=call_context
            )
            
            # Send webhook event for monitoring
            webhooks = db.query(WebhookConfig).filter(
                WebhookConfig.agent_id == conn.agent_id,
                WebhookConfig.is_active == True
            ).all()
            
            for webhook in webhooks:
                if "tool.called" in webhook.events:
                    await send_webhook(
                        webhook.webhook_url,
                        "tool.called",
                        {
                            "call_sid": call_sid,
                            "agent_id": conn.agent_id,
                            "tool_name": tool_name,
                            "parameters": params,
                            "result": result,
                            "timestamp": dt.utcnow().isoformat()
                        }
                    )
            
        finally:
            db.close()

    return result

# ----------------------------
# âš¡ GPU-ACCELERATED STREAMING RAG QUERY
# ----------------------------


# ----------------------------
# âš¡ GPU-ACCELERATED STREAMING RAG QUERY (true streaming)
# ----------------------------
# ----------------------------
# âš¡ GPU-ACCELERATED STREAMING RAG QUERY (no tools, true streaming)
# ----------------------------
async def query_rag_streaming(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = TOP_K,
    call_sid: Optional[str] = None
):
    """✨ ENHANCED: RAG with agent configuration and dynamic variables support"""
    if history is None:
        history = []

    # Get current date in America/New_York timezone
    from datetime import datetime
    import pytz
    ny_tz = pytz.timezone('America/New_York')
    current_date = datetime.now(ny_tz).strftime("%A, %B %d, %Y")
    
    # ✨ Load agent configuration and dynamic variables
    conn = manager.get(call_sid) if call_sid else None
    agent_prompt = None
    dynamic_vars = {}
    model_to_use = OLLAMA_MODEL  # Default from env
    
    model_source = "env_default"
    
    if conn and conn.agent_config:
        agent_prompt = conn.agent_config.get("system_prompt")
        dynamic_vars = conn.dynamic_variables or {}
        _logger.info(f"✅ Using agent prompt with {len(dynamic_vars)} dynamic variables")
        
        # ✨ Use custom model if provided, otherwise agent default, otherwise env default
        if conn.custom_model and conn.custom_model.strip():
            model_to_use = conn.custom_model
            model_source = "api_override"
        elif conn.agent_config.get("model_name"):
            model_to_use = conn.agent_config["model_name"]
            model_source = "agent_config"
    
    _logger.info(f"🤖 Model: {model_to_use} (source: {model_source})")

    loop = asyncio.get_running_loop()

    def _embed_and_query():
        with torch.no_grad():
            query_embedding = embedder.encode(
                [question],
                device=DEVICE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1
            )[0].tolist()

            return collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2
            )

    results = await loop.run_in_executor(None, _embed_and_query)

    raw_docs = results.get("documents", [[]])[0] if results else []
    distances = results.get("distances", [[]])[0] if results else []

    # Simple relevance filtering
    relevant_chunks = []
    for doc, dist in zip(raw_docs, distances):
        if dist <= 1.3:  # Simple threshold
            relevant_chunks.append(doc)

    # Use top 3 most relevant
    context_text = "\n".join(relevant_chunks[:3])

    # _logger.info(f"📚 Found {len(relevant_chunks)} relevant chunks")

    # Build conversation history
    history_text = ""
    if history and len(history) > 0:
        recent_history = history[-6:]  # Keep last 3 exchanges
        history_lines = []
        for h in recent_history:
            history_lines.append(
                f"User: {h['user']}\nAssistant: {h['assistant']}")
        history_text = "\n".join(history_lines)

    # ✨ BUILD PROMPT - Use agent's system_prompt if available, otherwise use default
    if agent_prompt:
        # ✨ AGENT-BASED PROMPT with dynamic variables
        vars_section = ""
        if dynamic_vars:
            vars_lines = []
            for key, value in dynamic_vars.items():
                if value and str(value).strip():
                    vars_lines.append(f"- **{key}**: {value}")
            if vars_lines:
                vars_section = "\n\n## Lead/Customer Information:\n" + "\n".join(vars_lines)
        call_context = ""
        if conn:
            call_context = f"""
        ## CALL CONTEXT (VERY IMPORTANT)
        You are on a LIVE PHONE CALL with a real person.
        - DO NOT include:
            - stage directions (e.g. **pause**, **laughs**, **sighs**)
            - do not use **bold** or _italics_, just respond in normal text and paragraphs
            - emotional markers (e.g. [happy], [thinking])
            - symbols like *, [], (), <> 
            - DO NOT describe actions or emotions.

        Current call phase: {conn.call_phase}
        Detected user intent: {conn.last_intent}

        Speech rules:
        - Speak briefly and naturally, like a human on the phone
        - Never explain in long paragraphs until asked
        """

        prompt = f"""{agent_prompt}

    {call_context}
    
## Current Date (America/New_York):
Today is {current_date}.{vars_section}

## Knowledge Base Context  (please make responses from only this company knowledge base):
{context_text if context_text.strip() else "No specific context found for user's this current query."}

## current conversation history(if nothing is here, that means this is the start of the call):
{history_text if history_text else ""}

## User's Current Question:
{question}"""
    else:
        prompt = f"""You are MILA, a friendly voice assistant for Technology Mindz. Technology Mindz provides key services: Salesforce, AI, Managed IT, Cybersecurity, Microsoft Dynamics 365, Staff Augmentation, CRM Consulting, Web Development, Mobile App Development.

## Current Date (America/New_York):
Today is {current_date}.

## YOUR PHONE PERSONALITY:
- You're on a LIVE phone call with a real person
- Speak naturally like a human would on the phone
- Keep responses BRIEF (1-2 sentences max)
- Use natural filler words: "um", "you know", "well", "actually", "yeah"
- Acknowledge what they say naturally: "Got it", "Makes sense", "Oh interesting", "I see", "Right"
- Sound conversational, not scripted
- Mirror their energy and pace

## RESPONSE GUIDELINES:
- For simple acknowledgments: Be brief and natural ("Yeah, got it" / "Makes sense" / "Okay, cool")
- For questions: Answer concisely from the knowledge base
- For confirmations: Respond naturally based on context (don't just say "okay got it" - be contextual)
- For hesitation ("um", "uh"): Gently encourage them ("Take your time" / "What's on your mind?")
- Never give long explanations - this is a phone call, not an essay

## KNOWLEDGE BASE RULES:
- Only use company knowledge base for factual answers
- If something isn't in the knowledge base, say "I'm not sure about that, but let me connect you with someone who can help"
- Never make up information

## MEETING SCHEDULING:
- When relevant, offer to schedule meetings
- Ask for: date, time, timezone (only FUTURE dates)
- After getting details: [TOOL:meeting_call:DATE:TIMEZONE:address]
- If valid=true: confirm scheduled, else: apologize and reschedule

## ENDING CALLS:
- If they want to end ("bye", "that's all", "talk later"), output: [TOOL:end_call]

## Previous Conversation:
{history_text if history_text else "This is the start of the call."}

## Knowledge Base:
{context_text if context_text else "No specific context."}

## What they just said:
{question}

Respond naturally and briefly:"""

    # Rest of your streaming code remains the same...
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    full_response = ""

    def _safe_put(item):
        """Safely put item in queue, handling QueueFull gracefully"""
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            # Queue is full - drop to prevent blocking
            # Try to make space by removing oldest item if queue is very full
            if queue.qsize() > 400:
                try:
                    queue.get_nowait()  # Remove one old item
                    queue.put_nowait(item)  # Try again
                except:
                    pass  # If that fails, just drop the item

    def _producer():
        nonlocal full_response
        try:
            for chunk in ollama.generate(
                model=model_to_use,
                prompt=prompt,
                stream=True,
                options={
                    "temperature": 0.2,
                    "num_predict": 1200,
                    "top_k": 40,
                    "top_p": 0.9,
                    "num_ctx": 1024,
                    "num_thread": 8,
                    "repeat_penalty": 1.2,
                    "repeat_last_n": 128,
                    "num_gpu": 99,
                    "stop": ["\nUser:", "\nAssistant:", "User:"],
                }
            ):
                token = chunk.get("response")
                if token:
                    full_response += token
                    loop.call_soon_threadsafe(_safe_put, token)
            loop.call_soon_threadsafe(_safe_put, None)
        except Exception as e:
            loop.call_soon_threadsafe(_safe_put, {"__error__": str(e)})

    loop.run_in_executor(None, _producer)

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, dict) and "__error__" in item:
                yield "I'm having trouble responding right now. Could you repeat that?"
                return

            # Yield tokens immediately (consumer will decide when to speak)
            yield item

    except Exception as e:
        yield "I'm having trouble answering right now. Could you repeat that?"


def calculate_audio_energy(mulaw_bytes: bytes) -> int:
    """Calculate RMS energy of audio chunk"""
    if not mulaw_bytes or len(mulaw_bytes) < 160:
        return 0
    try:
        pcm = audioop.ulaw2lin(mulaw_bytes, 2)
        return audioop.rms(pcm, 2)
    except Exception:
        return 0