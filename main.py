# main.py
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
from gpu_detection_llm import detect_gpu,public_ws_host,end_call_tool,transfer_call_tool,clean_markdown_for_tts,detect_intent,detect_confirmation_response,parse_llm_response,call_webhook_tool,execute_detected_tool,query_rag_streaming,calculate_audio_energy
from interrupt_detection import update_baseline
from gpu_detection_llm import twilio_client
from connection_manager import manager, handle_call_end,pending_call_data
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


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key - returns None if no API_KEYS configured (dev mode)"""
    # If no API keys configured, allow all requests (dev mode)
    if not API_KEYS or API_KEYS == ['']:
        return None
    
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

DEVICE = detect_gpu()

_logger.info("🚀 Config: PUBLIC_URL=%s DEVICE=%s", PUBLIC_URL, DEVICE)
_logger.info("🎯 Interrupt: ENABLED=%s MIN_SPEECH=%dms MIN_ENERGY=%d BASELINE_FACTOR=%.1f",
             INTERRUPT_ENABLED, INTERRUPT_MIN_SPEECH_MS, INTERRUPT_MIN_ENERGY, INTERRUPT_BASELINE_FACTOR)
_logger.info("â±ï¸  Silence Threshold: %.1fs (utterance_end=%dms)",
             SILENCE_THRESHOLD_SEC, UTTERANCE_END_MS)



# FastAPI app
# ----------------------------
app = FastAPI(
    title="AI Voice Call System - ElevenLabs Compatible",
    description="Self-hosted voice AI with agent management, webhooks, and dynamic variables",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# HELPER FUNCTIONS
# ================================

def generate_agent_id() -> str:
    """Generate unique agent ID"""
    return f"agent_{uuid.uuid4().hex[:16]}"


def generate_conversation_id() -> str:
    """Generate unique conversation ID"""
    return f"conv_{uuid.uuid4().hex[:16]}"


async def send_webhook(webhook_url: str, event: str, data: Dict):
    """Send webhook notification to registered webhook URLs (fire-and-forget)"""
    try:
        # Webhook URL must be absolute (http:// or https://)
        if not webhook_url.startswith(("http://", "https://")):
            _logger.error(f"❌ Invalid webhook URL: {webhook_url} - must start with http:// or https://")
            return False
        
        async with httpx.AsyncClient() as client:
            payload = {
                "event": event,
                "timestamp": dt.utcnow().isoformat(),
                "data": data
            }
            response = await client.post(
                webhook_url,
                json=payload,
                timeout=10
            )
            _logger.info(f"📤 Webhook sent: {event} to {webhook_url} (status: {response.status_code})")
            return response.status_code == 200
    except Exception as e:
        _logger.error(f"❌ Webhook failed: {event} to {webhook_url} - {e}")
        return False


async def send_webhook_and_get_response(webhook_url: str, event: str, data: Dict) -> Optional[Dict]:
    """Send webhook and wait for response data (for inbound call configuration)"""
    try:
        # Webhook URL must be absolute (http:// or https://)
        if not webhook_url.startswith(("http://", "https://")):
            _logger.error(f"❌ Invalid webhook URL: {webhook_url} - must start with http:// or https://")
            return None
        
        async with httpx.AsyncClient() as client:
            payload = {
                "event": event,
                "timestamp": dt.utcnow().isoformat(),
                "data": data
            }
            response = await client.post(
                webhook_url,
                json=payload,
                timeout=10
            )
            _logger.info(f"📤 Webhook sent: {event} to {webhook_url} (status: {response.status_code})")
            
            if response.status_code == 200:
                response_data = response.json()
                _logger.info(f"📥 Webhook response received: {list(response_data.keys())}")
                return response_data
            else:
                _logger.warning(f"⚠️ Webhook returned non-200 status: {response.status_code}")
                return None
    except Exception as e:
        _logger.error(f"❌ Webhook failed: {event} to {webhook_url} - {e}")
        return None


# ================================
# AGENT MANAGEMENT API
# ================================

@app.post("/v1/convai/agents", tags=["Agent Management"])
async def create_agent(
    agent: AgentCreate, 
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new agent with custom configuration
    
    Like ElevenLabs: Each agent has system prompt, voice, model settings
    """
    try:
        agent_id = generate_agent_id()
        
        db_agent = Agent(
            agent_id=agent_id,
            name=agent.name,
            system_prompt=agent.system_prompt,
            first_message=agent.first_message,
            voice_provider=agent.voice_provider,
            voice_id=agent.voice_id,
            model_provider=agent.model_provider,
            model_name=agent.model_name,
            interrupt_enabled=agent.interrupt_enabled,
            silence_threshold_sec=agent.silence_threshold_sec
        )
        
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)
        
        _logger.info(f"✅ Created agent: {agent_id} - {agent.name}")
        
        return {
            "success": True,
            "agent_id": agent_id,
            "name": agent.name,
            "created_at": db_agent.created_at.isoformat()
        }
    except Exception as e:
        db.rollback()
        _logger.error(f"❌ Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/convai/agents/{agent_id}", tags=["Agent Management"])
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """Get agent configuration"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "system_prompt": agent.system_prompt,
        "first_message": agent.first_message,
        "voice_provider": agent.voice_provider,
        "voice_id": agent.voice_id,
        "model_provider": agent.model_provider,
        "model_name": agent.model_name,
        "interrupt_enabled": agent.interrupt_enabled,
        "silence_threshold_sec": agent.silence_threshold_sec,
        "is_active": agent.is_active,
        "created_at": agent.created_at.isoformat(),
        "updated_at": agent.updated_at.isoformat()
    }


@app.get("/v1/convai/agents", tags=["Agent Management"])
async def list_agents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all agents"""
    agents = db.query(Agent).filter(Agent.is_active == True).offset(skip).limit(limit).all()
    
    return {
        "agents": [
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "voice_id": agent.voice_id,
                "model_name": agent.model_name,
                "created_at": agent.created_at.isoformat()
            }
            for agent in agents
        ],
        "total": len(agents)
    }


@app.patch("/v1/convai/agents/{agent_id}", tags=["Agent Management"])
async def update_agent(
    agent_id: str,
    updates: AgentUpdate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Update agent configuration"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Update only provided fields
    update_data = updates.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(agent, key, value)
    
    agent.updated_at = dt.utcnow()
    db.commit()
    db.refresh(agent)
    
    _logger.info(f"✅ Updated agent: {agent_id}")
    
    return {
        "success": True,
        "agent_id": agent_id,
        "updated_fields": list(update_data.keys())
    }


@app.delete("/v1/convai/agents/{agent_id}", tags=["Agent Management"])
async def delete_agent(
    agent_id: str, 
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete (deactivate) agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.is_active = False
    db.commit()
    
    _logger.info(f"✅ Deleted agent: {agent_id}")
    
    return {"success": True, "message": "Agent deleted"}


# ================================
# ELEVENLABS-COMPATIBLE CALL API
# ================================

@app.post("/v1/convai/twilio/outbound-call", tags=["Call Operations"])
async def initiate_outbound_call(
    request: OutboundCallRequest,
    db: Session = Depends(get_db)
):
    """
    ✨ ELEVENLABS-COMPATIBLE ENDPOINT
    
    Initiate outbound call with agent configuration and dynamic variables
    
    Request format (matches ElevenLabs):
    {
        "agent_id": "agent_xxx",
        "to_number": "+1234567890",
        "conversation_initiation_client_data": {
            "dynamic_variables": {
                "customer_name": "John",
                "company": "Acme Corp",
                ...
            },
            "conversation_config_override": {
                "tts": {"voice_id": "custom_voice"},
                "agent": {"prompt": {"llm": "custom_model"}}
            }
        }
    }
    """
    try:
        # Validate agent exists
        agent = db.query(Agent).filter(
            Agent.agent_id == request.agent_id,
            Agent.is_active == True
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent not found: {request.agent_id}")
        
        # Extract dynamic variables and overrides
        client_data = request.conversation_initiation_client_data or {}
        dynamic_variables = client_data.get("dynamic_variables", {})
        config_override = client_data.get("conversation_config_override", {})
        
        # 🔍 DEBUG: Log raw extraction
        _logger.info(f"🔍 API Debug - client_data keys: {list(client_data.keys())}")
        _logger.info(f"🔍 API Debug - config_override: {config_override}")
        _logger.info(f"🔍 API Debug - tts section: {config_override.get('tts', {})}")
        
        # Extract voice and model overrides
        custom_voice_id = config_override.get("tts", {}).get("voice_id")
        custom_model = config_override.get("agent", {}).get("prompt", {}).get("llm")
        custom_first_message = request.first_message or config_override.get("agent", {}).get("first_message")

        _logger.info(f"📞 Initiating call to {request.to_number} with agent {request.agent_id}")
        _logger.info(f"📊 Dynamic variables: {len(dynamic_variables)} fields")
        
        # 🔍 DEBUG: Log extracted values
        _logger.info(f"🔍 API Extracted - custom_voice_id: '{custom_voice_id}'")
        _logger.info(f"🔍 API Extracted - custom_model: '{custom_model}'")
        _logger.info(f"🔍 API Extracted - custom_first_message: '{custom_first_message[:50] if custom_first_message else None}...'")

        if custom_voice_id:
            _logger.info(f"🎤 Voice override: {custom_voice_id}")
        if custom_model:
            _logger.info(f"🤖 Model override: {custom_model}")
        
        # ✨ Look up phone number from database (priority order)
        phone_number_to_use = TWILIO_PHONE_NUMBER  # Default fallback from env
        
        # Priority 1: Use agent_phone_number_id from request (if provided)
        if request.agent_phone_number_id:
            phone_record = db.query(PhoneNumber).filter(
                PhoneNumber.id == request.agent_phone_number_id,
                PhoneNumber.is_active == True
            ).first()
            if phone_record:
                phone_number_to_use = phone_record.phone_number
                _logger.info(f"📞 Using phone number from database (ID: {request.agent_phone_number_id}): {phone_number_to_use}")
            else:
                _logger.warning(f"⚠️ Phone number ID '{request.agent_phone_number_id}' not found in database, using fallback")
        
        # Priority 2: Try to get phone number linked to agent
        if phone_number_to_use == TWILIO_PHONE_NUMBER and agent.agent_id:
            phone_record = db.query(PhoneNumber).filter(
                PhoneNumber.agent_id == agent.agent_id,
                PhoneNumber.is_active == True
            ).first()
            if phone_record:
                phone_number_to_use = phone_record.phone_number
                _logger.info(f"📞 Using agent's linked phone number: {phone_number_to_use}")
        
        # Priority 3: Use TWILIO_PHONE_NUMBER from env (already set as default)
        if phone_number_to_use == TWILIO_PHONE_NUMBER:
            _logger.info(f"📞 Using default phone number from env: {phone_number_to_use}")
        
        # Make Twilio call
        webhook_url = f"{PUBLIC_URL.rstrip('/')}/voice/outbound"
        status_callback_url = f"{PUBLIC_URL.rstrip('/')}/voice/status"
        
        call = twilio_client.calls.create(
            to=request.to_number,
            from_=phone_number_to_use,  # ✅ From database lookup or env fallback
            url=webhook_url,
            method="POST",
            status_callback=status_callback_url,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST"
        )
        
        call_sid = call.sid
        conversation_id = call_sid  # Use Twilio call_sid as conversation_id
        
        # Store call data for when WebSocket connects
        pending_call_data[call_sid] = {
            "agent_id": request.agent_id,
            "dynamic_variables": dynamic_variables,
            "custom_voice_id": custom_voice_id,
            "custom_model": custom_model,
            "custom_first_message": custom_first_message,
            "to_number": request.to_number,
            "enable_recording": request.enable_recording,
            "direction": "outbound"
        }
        
        _logger.info(f"💾 Stored call data for: {call_sid}")
        _logger.info(f"💾 - Agent ID: {request.agent_id}")
        _logger.info(f"💾 - Custom voice: {custom_voice_id}")
        _logger.info(f"💾 - Custom model: {custom_model}")
        _logger.info(f"💾 - Dynamic vars: {len(dynamic_variables)} fields")
        
        # Create conversation record in database
        conversation = Conversation(
            conversation_id=conversation_id,
            agent_id=request.agent_id,
            phone_number=request.to_number,
            status="initiated",
            dynamic_variables=dynamic_variables,
            call_metadata={"overrides": config_override,
            "custom_first_message": custom_first_message}
        )
        db.add(conversation)
        db.commit()
        
        _logger.info(f"✅ Call initiated: {conversation_id}")
        
        # Send webhook notification (if configured)
        webhooks = db.query(WebhookConfig).filter(
            WebhookConfig.is_active == True
        ).filter(
            (WebhookConfig.agent_id == request.agent_id) | (WebhookConfig.agent_id == None)
        ).all()
        
        for webhook in webhooks:
            if "call.initiated" in webhook.events or not webhook.events:
                await send_webhook(
                    webhook.webhook_url,
                    "call.initiated",
                    {
                        "conversation_id": conversation_id,
                        "agent_id": request.agent_id,
                        "to_number": request.to_number,
                        "status": "initiated"
                    }
                )
        
        # Return ElevenLabs-compatible response
        return {
            "conversation_id": conversation_id,
            "agent_id": request.agent_id,
            "status": "initiated",
            "phone_number": request.to_number,
            "first_message": custom_first_message or agent.first_message  
        }
        
    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"❌ Call initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# CONVERSATION RETRIEVAL API
# ================================

@app.get("/v1/convai/conversations/{conversation_id}", tags=["Conversations"])
async def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """
    ✨ ELEVENLABS-COMPATIBLE ENDPOINT
    
    Get conversation details including transcript
    """
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Extract direction from call_metadata
    call_direction = "outbound"
    if conversation.call_metadata and isinstance(conversation.call_metadata, dict):
        call_direction = conversation.call_metadata.get("direction", "outbound")
    
    # Get agent name if exists
    agent_name = None
    if conversation.agent_id:
        agent = db.query(Agent).filter(Agent.agent_id == conversation.agent_id).first()
        if agent:
            agent_name = agent.name
    
    return {
        "conversation_id": conversation.conversation_id,
        "agent_id": conversation.agent_id,
        "agent_name": agent_name,
        "status": conversation.status,
        "transcript": conversation.transcript,
        "started_at": conversation.started_at.isoformat() if conversation.started_at else None,
        "ended_at": conversation.ended_at.isoformat() if conversation.ended_at else None,
        "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
        "metadata": {
            "call_duration_secs": conversation.duration_secs,
            "termination_reason": conversation.ended_reason,
            "phone_number": conversation.phone_number,
            "direction": call_direction,
            "recording_url": conversation.recording_url
        },
        "analysis": {
            "transcript_length": len(conversation.transcript) if conversation.transcript else 0,
            "has_recording": bool(conversation.recording_url)
        },
        "dynamic_variables": conversation.dynamic_variables,
        "call_metadata": conversation.call_metadata
    }


@app.get("/v1/convai/conversations", tags=["Conversations"])
async def list_conversations(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    direction: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List conversations (optionally filtered by agent_id, status, direction)
    
    ✨ ELEVENLABS-COMPATIBLE: Supports filtering and pagination
    """
    query = db.query(Conversation)
    
    if agent_id:
        query = query.filter(Conversation.agent_id == agent_id)
    
    if status:
        query = query.filter(Conversation.status == status)
    
    conversations = query.order_by(Conversation.created_at.desc()).offset(skip).limit(limit).all()
    
    # Filter by direction if specified (direction is in call_metadata)
    if direction:
        conversations = [
            conv for conv in conversations
            if conv.call_metadata and isinstance(conv.call_metadata, dict) 
            and conv.call_metadata.get("direction") == direction
        ]
    
    # Get total count (without filters for pagination info)
    total_query = db.query(Conversation)
    if agent_id:
        total_query = total_query.filter(Conversation.agent_id == agent_id)
    if status:
        total_query = total_query.filter(Conversation.status == status)
    total_count = total_query.count()
    
    return {
        "conversations": [
            {
                "conversation_id": conv.conversation_id,
                "agent_id": conv.agent_id,
                "status": conv.status,
                "phone_number": conv.phone_number,
                "duration_secs": conv.duration_secs,
                "direction": conv.call_metadata.get("direction", "outbound") if conv.call_metadata and isinstance(conv.call_metadata, dict) else "outbound",
                "ended_reason": conv.ended_reason,
                "has_transcript": bool(conv.transcript),
                "has_recording": bool(conv.recording_url),
                "started_at": conv.started_at.isoformat() if conv.started_at else None,
                "ended_at": conv.ended_at.isoformat() if conv.ended_at else None,
                "created_at": conv.created_at.isoformat() if conv.created_at else None
            }
            for conv in conversations
        ],
        "total": total_count,
        "page_size": limit,
        "offset": skip
    }


# ================================
# WEBHOOK MANAGEMENT API
# ================================

@app.post("/v1/convai/webhooks", tags=["Webhooks"], response_model=WebhookResponse)
async def create_webhook(
    request: WebhookCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Register webhook for call events
    
    **Available Events:**
    - `call.initiated` - When a call is initiated
    - `call.started` - When a call connects
    - `call.ended` - When a call ends
    - `call.failed` - When a call fails
    - `transcript.partial` - Partial transcript updates
    - `transcript.final` - Final transcript
    - `agent.response` - Agent responds
    - `tool.called` - When a tool is called
    - `user.interrupted` - When user interrupts
    
    **Examples:**
    ```json
    {
      "webhook_url": "https://your-app.com/webhook",
      "events": ["call.started", "call.ended"],
      "agent_id": "agent_123"
    }
    ```
    
    Set `agent_id` to `null` for global webhooks (all agents).
    """
    try:
        # Validate webhook URL
        if not request.webhook_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400, 
                detail="Webhook URL must start with http:// or https://"
            )
        
        # Validate events
        if request.events:
            invalid_events = [e for e in request.events if e not in WEBHOOK_EVENTS]
            if invalid_events:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid events: {invalid_events}. Valid events: {WEBHOOK_EVENTS}"
                )
        
        # If agent_id provided, verify agent exists
        if request.agent_id:
            agent = db.query(Agent).filter(Agent.agent_id == request.agent_id).first()
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent not found: {request.agent_id}")
        
        webhook = WebhookConfig(
            agent_id=request.agent_id,
            webhook_url=request.webhook_url,
            events=request.events or WEBHOOK_EVENTS
        )
        
        db.add(webhook)
        db.commit()
        db.refresh(webhook)
        
        _logger.info(
            f"✅ Webhook registered: {request.webhook_url} "
            f"for agent: {request.agent_id or 'GLOBAL'} "
            f"with events: {request.events}"
        )
        
        return WebhookResponse(
            success=True,
            webhook_id=webhook.id,
            webhook_url=request.webhook_url,
            events=webhook.events,
            agent_id=request.agent_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"❌ Webhook creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/convai/webhooks", tags=["Webhooks"])
async def list_webhooks(
    agent_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all registered webhooks
    
    **Query Parameters:**
    - `agent_id` (optional): Filter by agent ID. Omit to see all webhooks.
    
    **Returns:**
    - List of webhooks with their configuration
    - Includes global webhooks (agent_id = null)
    """
    query = db.query(WebhookConfig).filter(WebhookConfig.is_active == True)
    
    if agent_id:
        query = query.filter(WebhookConfig.agent_id == agent_id)
    
    webhooks = query.all()
    
    return {
        "webhooks": [
            {
                "id": w.id,
                "agent_id": w.agent_id or "GLOBAL",
                "webhook_url": w.webhook_url,
                "events": w.events,
                "created_at": w.created_at.isoformat() if w.created_at else None
            }
            for w in webhooks
        ],
        "total": len(webhooks)
    }


@app.delete("/v1/convai/webhooks/{webhook_id}", tags=["Webhooks"])
async def delete_webhook(
    webhook_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Delete a webhook by ID
    
    **Path Parameters:**
    - `webhook_id`: The numeric ID of the webhook to delete
    
    **Returns:**
    - Success confirmation
    """
    webhook = db.query(WebhookConfig).filter(WebhookConfig.id == webhook_id).first()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook_url = webhook.webhook_url
    agent_id = webhook.agent_id
    
    webhook.is_active = False
    db.commit()
    
    _logger.info(f"✅ Webhook deleted: ID={webhook_id}, URL={webhook_url}, Agent={agent_id or 'GLOBAL'}")
    
    return {
        "success": True,
        "message": "Webhook deleted successfully",
        "webhook_id": webhook_id
    }


# ================================
# PHONE NUMBER MANAGEMENT API
# ================================

@app.post("/v1/convai/phone-numbers", tags=["Phone Numbers"])
async def register_phone_number(
    phone_number: str,
    agent_id: Optional[str] = None,
    label: Optional[str] = None,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Register a phone number and optionally link to agent"""
    # Check if phone number already exists
    existing = db.query(PhoneNumber).filter(
        PhoneNumber.phone_number == phone_number
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Phone number already registered")
    
    # Verify agent exists if provided
    if agent_id:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
    
    phone = PhoneNumber(
        id=f"pn_{uuid.uuid4().hex[:16]}",
        phone_number=phone_number,
        agent_id=agent_id,
        label=label
    )
    db.add(phone)
    db.commit()
    db.refresh(phone)
    
    _logger.info(f"✅ Registered phone number: {phone_number} -> agent: {agent_id}")
    
    return {
        "phone_number_id": phone.id,
        "phone_number": phone_number,
        "agent_id": agent_id,
        "label": label
    }


@app.get("/v1/convai/phone-numbers", tags=["Phone Numbers"])
async def list_phone_numbers(db: Session = Depends(get_db)):
    """List all registered phone numbers"""
    phones = db.query(PhoneNumber).filter(PhoneNumber.is_active == True).all()
    
    return {
        "phone_numbers": [
            {
                "id": p.id,
                "phone_number": p.phone_number,
                "agent_id": p.agent_id,
                "label": p.label,
                "provider": p.provider,
                "created_at": p.created_at.isoformat()
            }
            for p in phones
        ]
    }


@app.patch("/v1/convai/phone-numbers/{phone_id}", tags=["Phone Numbers"])
async def update_phone_number(
    phone_id: str,
    agent_id: Optional[str] = None,
    label: Optional[str] = None,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Update phone number configuration (link to different agent)"""
    phone = db.query(PhoneNumber).filter(PhoneNumber.id == phone_id).first()
    
    if not phone:
        raise HTTPException(status_code=404, detail="Phone number not found")
    
    if agent_id is not None:
        if agent_id:  # Not empty string
            agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
        phone.agent_id = agent_id if agent_id else None
    
    if label is not None:
        phone.label = label
    
    db.commit()
    
    return {"success": True, "phone_number_id": phone_id}


@app.delete("/v1/convai/phone-numbers/{phone_id}", tags=["Phone Numbers"])
async def delete_phone_number(
    phone_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a phone number"""
    phone = db.query(PhoneNumber).filter(PhoneNumber.id == phone_id).first()
    
    if not phone:
        raise HTTPException(status_code=404, detail="Phone number not found")
    
    phone.is_active = False
    db.commit()
    
    return {"success": True, "message": "Phone number deleted"}


# ================================
# KNOWLEDGE BASE PER AGENT API
# ================================

@app.post("/v1/convai/agents/{agent_id}/knowledge-base", tags=["Knowledge Base"])
async def add_knowledge(
    agent_id: str,
    content: str,
    metadata: Optional[Dict] = None,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Add knowledge to agent's knowledge base"""
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    doc_id = f"doc_{uuid.uuid4().hex[:16]}"
    
    # Add to database
    kb = KnowledgeBase(
        agent_id=agent_id,
        document_id=doc_id,
        content=content,
        kb_metadata=metadata
    )
    db.add(kb)
    db.commit()
    
    # Add to ChromaDB with agent prefix
    chunks = _chunk_text(content, CHUNK_SIZE, overlap=50)
    
    with torch.no_grad():
        embeddings = embedder.encode(
            chunks, 
            device=DEVICE, 
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
    
    # Use agent-specific collection
    agent_collection = chroma_client.get_or_create_collection(f"agent_{agent_id}")
    
    agent_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        metadatas=[{"agent_id": agent_id, "doc_id": doc_id} for _ in chunks]
    )
    
    _logger.info(f"✅ Added knowledge to agent {agent_id}: {len(chunks)} chunks")
    
    return {
        "document_id": doc_id,
        "agent_id": agent_id,
        "chunks_created": len(chunks)
    }


@app.get("/v1/convai/agents/{agent_id}/knowledge-base", tags=["Knowledge Base"])
async def list_agent_knowledge(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """List knowledge base documents for an agent"""
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    documents = db.query(KnowledgeBase).filter(
        KnowledgeBase.agent_id == agent_id
    ).all()
    
    return {
        "agent_id": agent_id,
        "documents": [
            {
                "document_id": doc.document_id,
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "metadata": doc.kb_metadata,
                "created_at": doc.created_at.isoformat()
            }
            for doc in documents
        ],
        "total": len(documents)
    }


@app.delete("/v1/convai/agents/{agent_id}/knowledge-base/{document_id}", tags=["Knowledge Base"])
async def delete_agent_knowledge(
    agent_id: str,
    document_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a knowledge base document"""
    doc = db.query(KnowledgeBase).filter(
        KnowledgeBase.agent_id == agent_id,
        KnowledgeBase.document_id == document_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from database
    db.delete(doc)
    db.commit()
    
    # Remove from ChromaDB
    try:
        agent_collection = chroma_client.get_or_create_collection(f"agent_{agent_id}")
        # Get all IDs that start with this document_id
        results = agent_collection.get(where={"doc_id": document_id})
        if results and results.get("ids"):
            agent_collection.delete(ids=results["ids"])
    except Exception as e:
        _logger.warning(f"⚠️ Could not delete from ChromaDB: {e}")
    
    return {"success": True, "message": "Document deleted"}


# ================================
# CUSTOM TOOLS PER AGENT API
# ================================

@app.post("/v1/convai/agents/{agent_id}/tools", tags=["Tools"])
async def add_agent_tool(
    agent_id: str,
    tool_data: ToolCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Add custom tool to agent
    
    Example request body:
    ```json
    {
        "tool_name": "weather",
        "description": "Get weather information for a location",
        "webhook_url": "https://your-api.com/weather",
        "parameters": {
            "location": {
                "type": "string",
                "required": true,
                "description": "City name"
            }
        }
    }
    ```
    """
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    tool = AgentTool(
        agent_id=agent_id,
        tool_name=tool_data.tool_name,
        description=tool_data.description,
        webhook_url=tool_data.webhook_url,
        parameters=tool_data.parameters or {}
    )
    db.add(tool)
    db.commit()
    db.refresh(tool)
    
    _logger.info(f"✅ Added tool '{tool_data.tool_name}' to agent {agent_id}")
    
    return {
        "success": True,
        "tool_id": tool.id,
        "tool_name": tool_data.tool_name,
        "agent_id": agent_id,
        "webhook_url": tool_data.webhook_url
    }


@app.get("/v1/convai/agents/{agent_id}/tools", tags=["Tools"])
async def list_agent_tools(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """List custom tools for an agent"""
    tools = db.query(AgentTool).filter(
        AgentTool.agent_id == agent_id,
        AgentTool.is_active == True
    ).all()
    
    return {
        "agent_id": agent_id,
        "tools": [
            {
                "id": t.id,
                "tool_name": t.tool_name,
                "description": t.description,
                "webhook_url": t.webhook_url,
                "parameters": t.parameters,
                "created_at": t.created_at.isoformat()
            }
            for t in tools
        ]
    }


@app.delete("/v1/convai/agents/{agent_id}/tools/{tool_id}", tags=["Tools"])
async def delete_agent_tool(
    agent_id: str,
    tool_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Delete a custom tool"""
    tool = db.query(AgentTool).filter(
        AgentTool.id == tool_id,
        AgentTool.agent_id == agent_id
    ).first()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    tool.is_active = False
    db.commit()
    
    return {"success": True, "message": "Tool deleted"}


# ================================
# CALL RECORDING API
# ================================

@app.post("/recording-callback", tags=["Recording"])
async def recording_callback(request: Request):
    """Handle recording completion from Twilio"""
    form = await request.form()
    call_sid = form.get("CallSid")
    recording_url = form.get("RecordingUrl")
    recording_sid = form.get("RecordingSid")
    recording_duration = form.get("RecordingDuration")
    
    _logger.info(f"🎙️ Recording completed: {call_sid} - {recording_url}")
    
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == call_sid
        ).first()
        
        if conversation:
            conversation.recording_url = recording_url
            # Store additional recording metadata
            if conversation.call_metadata:
                conversation.call_metadata["recording_sid"] = recording_sid
                conversation.call_metadata["recording_duration"] = recording_duration
            else:
                conversation.call_metadata = {
                    "recording_sid": recording_sid,
                    "recording_duration": recording_duration
                }
            db.commit()
            _logger.info(f"✅ Recording URL saved for {call_sid}")
    except Exception as e:
        _logger.error(f"❌ Failed to save recording URL: {e}")
    finally:
        db.close()
    
    return PlainTextResponse("OK")


@app.get("/v1/convai/conversations/{conversation_id}/recording", tags=["Recording"])
async def get_recording(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get recording URL for a conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if not conversation.recording_url:
        raise HTTPException(status_code=404, detail="No recording available")
    
    return {
        "conversation_id": conversation_id,
        "recording_url": conversation.recording_url,
        "recording_metadata": conversation.call_metadata
    }


# ================================
# SIGNED URL FOR WIDGETS (JWT)
# ================================

@app.get("/v1/convai/conversation/get-signed-url", tags=["Widgets"])
async def get_signed_url(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """Generate signed URL for embedding widget"""
    import jwt
    from datetime import timedelta
    
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    payload = {
        "agent_id": agent_id,
        "exp": dt.utcnow() + timedelta(hours=24),
        "iat": dt.utcnow()
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    
    return {
        "signed_url": f"{PUBLIC_URL}/widget?token={token}",
        "expires_in": 86400,  # 24 hours in seconds
        "agent_id": agent_id
    }


@app.get("/widget", tags=["Widgets"])
async def widget_page(
    token: str,
    db: Session = Depends(get_db)
):
    """Widget endpoint that validates JWT token"""
    import jwt
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        agent_id = payload.get("agent_id")
        
        # Verify agent exists
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "valid": True,
            "agent_id": agent_id,
            "agent_name": agent.name,
            "message": "Widget authentication successful"
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/test-end-call")
async def test_end_call(request: Request):
    """Test end call tool"""
    try:
        data = await request.json()
        call_sid = data.get("call_sid", "test_call_123")
        reason = data.get("reason", "test")

        result = await end_call_tool(call_sid, reason)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/test-transfer")
async def test_transfer(request: Request):
    """Test transfer tool"""
    try:
        data = await request.json()
        call_sid = data.get("call_sid", "test_call_123")
        department = data.get("department", "sales")

        result = await transfer_call_tool(call_sid, department)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/tools/status")
async def tools_status():
    """Check tool configuration"""
    return {
        "tools_available": ["end_call", "transfer_call"],
        "departments": {
            "sales": os.getenv("SALES_PHONE_NUMBER", "NOT_SET"),
            "support": os.getenv("SUPPORT_PHONE_NUMBER", "NOT_SET"),
            "technical": os.getenv("TECH_PHONE_NUMBER", "NOT_SET"),
        },
        "confirmation_system": "enabled",
        "transfer_requires_confirmation": True,
        "end_call_requires_confirmation": False,
        "silence_threshold_sec": SILENCE_THRESHOLD_SEC,
        "utterance_end_ms": UTTERANCE_END_MS


    }


@app.websocket("/media-stream")
async def media_ws(websocket: WebSocket):
    try:
        await websocket.accept()
    except RuntimeError as e:
        return

    async def send_heartbeat():
        while True:
            try:
                await asyncio.sleep(5)
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({"event": "heartbeat"})
            except Exception as e:
                break

    heartbeat_task = asyncio.create_task(send_heartbeat())

    current_call_sid: Optional[str] = None

    processing_task: Optional[asyncio.Task] = None

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except RuntimeError as e:
                break
            except Exception as e:
                break

            event = data.get("event")

            if event == "start":
                start_info = data.get("start", {})
                current_call_sid = start_info.get("callSid")
                stream_sid = start_info.get("streamSid")

                if not current_call_sid:
                    break

                await manager.connect(current_call_sid, websocket)
                conn = manager.get(current_call_sid)
                if conn:
                    conn.stream_sid = stream_sid
                    conn.stream_ready = True
                    conn.conversation_id = current_call_sid

                    # ✨ Load agent configuration and call data
                    call_data = pending_call_data.get(current_call_sid, {})
                    agent_id = call_data.get("agent_id")
                    call_direction = call_data.get("direction", "outbound")
                    
                    _logger.info(f"🔍 WebSocket Debug - call_sid: {current_call_sid}")
                    _logger.info(f"🔍 Pending call data found: {bool(call_data)}")
                    _logger.info(f"🔍 Agent ID: {agent_id}")
                    _logger.info(f"🔍 Direction: {call_direction}")
                    _logger.info(f"🔍 Custom voice_id: {call_data.get('custom_voice_id')}")
                    _logger.info(f"🔍 Custom model: {call_data.get('custom_model')}")
                    
                    # ✨ ALWAYS load dynamic variables (like ElevenLabs)
                    conn.dynamic_variables = call_data.get("dynamic_variables", {})
                    conn.custom_voice_id = call_data.get("custom_voice_id")
                    conn.custom_model = call_data.get("custom_model")
                    conn.custom_first_message = call_data.get("custom_first_message")
                    
                    # ✨ Log all overrides for debugging
                    _logger.info(f"🔧 Overrides loaded:")
                    _logger.info(f"   - custom_voice_id: {conn.custom_voice_id or 'None (will use agent/default)'}")
                    _logger.info(f"   - custom_model: {conn.custom_model or 'None (will use agent/default)'}")
                    _logger.info(f"   - custom_first_message: {'Yes (' + conn.custom_first_message[:30] + '...)' if conn.custom_first_message else 'None (will use agent/default)'}")
                    
                    db = SessionLocal()
                    try:
                        # Load agent if specified
                        if agent_id:
                            agent = db.query(Agent).filter(
                                Agent.agent_id == agent_id
                            ).first()
                            
                            if agent:
                                conn.agent_id = agent_id
                                conn.agent_config = {
                                    "system_prompt": agent.system_prompt,
                                    "first_message": agent.first_message,
                                    "voice_id": agent.voice_id,
                                    "model_name": agent.model_name,
                                    "silence_threshold_sec": agent.silence_threshold_sec
                                }
                                
                                _logger.info(f"✅ Loaded agent: {agent_id} ({agent.name})")
                                _logger.info(f"📊 Dynamic variables: {len(conn.dynamic_variables)} fields")

                                if call_data.get("custom_first_message"):
                                    conn.agent_config["first_message"] = call_data["custom_first_message"]
                                    _logger.info(f"💬 Using custom first message: {call_data['custom_first_message'][:50]}...")
                            else:
                                _logger.warning(f"⚠️ Agent not found: {agent_id}")
                        else:
                            _logger.info("ℹ️ No agent specified, using default behavior")
                        
                        # ✨ ALWAYS update conversation status to "in-progress" (like ElevenLabs)
                        conversation = db.query(Conversation).filter(
                            Conversation.conversation_id == current_call_sid
                        ).first()
                        
                        if conversation:
                            conversation.status = "in-progress"
                            conversation.started_at = dt.utcnow()
                            db.commit()
                            _logger.info(f"✅ Conversation status updated to 'in-progress': {current_call_sid}")
                        else:
                            # Create conversation record if it doesn't exist (fallback)
                            _logger.warning(f"⚠️ Conversation not found, creating new record: {current_call_sid}")
                            # ✅ For inbound: use from_number (caller), for outbound: use to_number (recipient)
                            phone_for_record = call_data.get("from_number") if call_direction == "inbound" else call_data.get("to_number")
                            new_conversation = Conversation(
                                conversation_id=current_call_sid,
                                agent_id=agent_id,
                                phone_number=phone_for_record,
                                status="in-progress",
                                started_at=dt.utcnow(),
                                dynamic_variables=conn.dynamic_variables,
                                call_metadata={"direction": call_direction}
                            )
                            db.add(new_conversation)
                            db.commit()
                        
                        # ✨ ALWAYS send "call.started" webhook (like ElevenLabs)
                        webhooks = db.query(WebhookConfig).filter(
                            WebhookConfig.is_active == True
                        ).all()
                        
                        for webhook in webhooks:
                            should_send = False
                            if webhook.agent_id is None:
                                should_send = True  # Global webhook
                            elif agent_id and webhook.agent_id == agent_id:
                                should_send = True  # Agent-specific webhook
                            
                            if should_send and ("call.started" in webhook.events or not webhook.events):
                                # ✅ For inbound: send caller's number (from_number in call_data)
                                # ✅ For outbound: send recipient's number (to_number in call_data)
                                caller_phone = call_data.get("from_number") if call_direction == "inbound" else call_data.get("to_number")
                                
                                # ✅ For INBOUND calls: Wait for webhook response to get dynamic variables
                                if call_direction == "inbound":
                                    _logger.info(f"🔄 Sending call.started webhook to {webhook.webhook_url} and waiting for response...")
                                    webhook_response = await send_webhook_and_get_response(
                                        webhook.webhook_url,
                                        "call.started",
                                        {
                                            "conversation_id": current_call_sid,
                                            "agent_id": agent_id,
                                            "direction": call_direction,
                                            "status": "in-progress",
                                            "phone_number": caller_phone
                                        }
                                    )
                                    
                                    _logger.info(f"📥 Webhook response received: {webhook_response is not None}, has dynamic_variables: {webhook_response and 'dynamic_variables' in webhook_response if webhook_response else False}")
                                    
                                    # Apply dynamic variables from webhook response
                                    if webhook_response and "dynamic_variables" in webhook_response:
                                        response_vars = webhook_response["dynamic_variables"]
                                        _logger.info(f"📥 Applying {len(response_vars)} dynamic variables from webhook response")
                                        
                                        # Merge with existing dynamic variables
                                        if conn.dynamic_variables:
                                            conn.dynamic_variables.update(response_vars)
                                        else:
                                            conn.dynamic_variables = response_vars
                                        
                                        # Apply first_message if provided
                                        if "first_message" in response_vars:
                                            if conn.agent_config:
                                                conn.agent_config["first_message"] = response_vars["first_message"]
                                                _logger.info(f"✅ Applied first_message from webhook: '{response_vars['first_message'][:50]}...'")
                                            else:
                                                _logger.warning("⚠️ Cannot apply first_message - agent_config not loaded yet")
                                else:
                                    # For OUTBOUND calls: Fire-and-forget webhook
                                    asyncio.create_task(send_webhook(
                                        webhook.webhook_url,
                                        "call.started",
                                        {
                                            "conversation_id": current_call_sid,
                                            "agent_id": agent_id,
                                            "direction": call_direction,
                                            "status": "in-progress",
                                            "phone_number": caller_phone
                                        }
                                    ))
                    finally:
                        db.close()

                    # âœ… CRITICAL: Initialize resampler ONCE per connection
                    dummy_state = None
                    try:
                        _, dummy_state = audioop.ratecv(
                            b'\x00' * 3200, 2, 1, 16000, 8000, dummy_state
                        )
                        conn.resampler_state = dummy_state
                        conn.resampler_initialized = True
                        _logger.info("ðŸŽµ Resampler pre-initialized for this connection")
                    except Exception as e:
                        _logger.warning("Failed to pre-init resampler: %s", e)

                    await setup_streaming_stt(current_call_sid)
                    conn.tts_task = asyncio.create_task(
                        stream_tts_worker(current_call_sid))

                await asyncio.sleep(0.1)
                greeting = None

                # ✨ Use agent's first_message or default greeting
                if conn and conn.agent_config and conn.agent_config.get("first_message"):
                    # Replace {{variable}} placeholders in first_message
                    greeting = conn.agent_config["first_message"]
                    if conn.dynamic_variables:
                        for key, value in conn.dynamic_variables.items():
                            greeting = greeting.replace(f"{{{{{key}}}}}", str(value))
                else:
                    greeting = "hello there! this is default greeting from AI assistant. How can I help you today?"
                if conn and conn.dynamic_variables and greeting:
                    for key, value in conn.dynamic_variables.items():
                        greeting = greeting.replace(f"{{{{{key}}}}}", str(value))
                
                # 🔍 DEBUG: Verify overrides are still set before greeting
                _logger.info(f"🎯 BEFORE GREETING - conn.custom_voice_id: '{conn.custom_voice_id}'")
                _logger.info(f"🎯 BEFORE GREETING - conn.agent_config voice: '{conn.agent_config.get('voice_id') if conn.agent_config else None}'")
                
                await speak_text_streaming(current_call_sid, greeting)
                
                # ✨ CAPTURE GREETING IN TRANSCRIPT (like ElevenLabs)
                # This ensures we have a transcript even if user hangs up immediately
                if conn and greeting:
                    conn.conversation_history.append({
                        "user": "[Call Started]",
                        "assistant": greeting,
                        "timestamp": time.time()
                    })
                    _logger.info(f"✅ Greeting captured in conversation history")

            elif event == "media":
                if not current_call_sid:
                    continue

                media_data = data.get("media", {})
                payload_b64 = media_data.get("payload")

                if payload_b64:
                    try:
                        chunk = base64.b64decode(payload_b64)
                        conn = manager.get(current_call_sid)

                        if not conn:
                            continue

                        # Send to Deepgram
                        if conn.deepgram_live:
                            try:
                                conn.deepgram_live.send(chunk)
                            except Exception as e:
                                pass

                        energy = calculate_audio_energy(chunk)
                        update_baseline(conn, energy)

                        now = time.time()

                        # Calculate energy threshold
                        energy_threshold = max(
                            conn.baseline_energy * INTERRUPT_BASELINE_FACTOR,
                            INTERRUPT_MIN_ENERGY
                        )

                        # ========================================
                        # âœ… SMART VAD VALIDATION & TIMEOUT LOGIC
                        # ========================================

                        if conn.vad_triggered_time and conn.user_speech_detected:
                            time_since_vad = now - conn.vad_triggered_time

                            # Check if we're seeing actual speech energy
                            if energy >= energy_threshold:
                                # âœ… Real speech detected
                                conn.last_valid_speech_energy = energy
                                conn.energy_drop_time = None  # Reset drop timer

                                # Validate VAD after short period
                                if not conn.vad_validated and time_since_vad >= conn.vad_validation_threshold:
                                    conn.vad_validated = True
                                    _logger.info(
                                        f"âœ… VAD validated after {time_since_vad*1000:.0f}ms (energy: {energy})")

                                if not conn.speech_start_time:
                                    conn.speech_start_time = now

                            else:
                                # Low energy - but is it silence or just a pause?

                                if conn.vad_validated:
                                    # âœ… VAD was real - this is just low energy during speech (normal)
                                    # Track when energy dropped
                                    if conn.energy_drop_time is None:
                                        conn.energy_drop_time = now

                                    # Only clear VAD if energy stays low for extended period
                                    # AND we have FINAL or interim text (meaning Deepgram also thinks speech ended)
                                    low_energy_duration = now - conn.energy_drop_time

                                    if low_energy_duration >= 1.5:  # 1.5s of low energy
                                        # Check if Deepgram also stopped detecting speech
                                        time_since_last_text = now - conn.last_interim_time if conn.last_interim_time else 999

                                        if time_since_last_text > 1.0:  # No text for 1s
                                            _logger.info(
                                                f"âœ… VAD cleared naturally (low energy: {low_energy_duration:.1f}s, no text: {time_since_last_text:.1f}s)")
                                            conn.user_speech_detected = False
                                            conn.speech_start_time = None
                                            conn.vad_triggered_time = None
                                            conn.vad_validated = False
                                            conn.energy_drop_time = None
                                else:
                                    # âŒ VAD not validated yet - might be false positive
                                    # Give it 1s to validate (reduced from 3s)
                                    if time_since_vad >= 1.0:
                                        _logger.warning(
                                            f"âš ï¸ VAD timeout - false positive (duration: {time_since_vad:.1f}s)")
                                        conn.user_speech_detected = False
                                        conn.speech_start_time = None
                                        conn.vad_triggered_time = None
                                        conn.vad_validated = False
                                        conn.energy_drop_time = None

                        # ========================================
                        # âœ… INTERRUPT DETECTION (unchanged logic)
                        # ========================================

                        if conn.currently_speaking and conn.user_speech_detected and not conn.interrupt_requested:
                            # Only interrupt if VAD has been validated (real speech)
                            if conn.vad_validated and conn.speech_start_time:
                                user_speaking_duration = (
                                    now - conn.speech_start_time) * 1000.0

                                if user_speaking_duration < 500:
                                    continue

                                conn.speech_energy_buffer.append((now, energy))

                                vad_dur_ms = (
                                    now - conn.speech_start_time) * 1000.0
                                buf = list(conn.speech_energy_buffer)

                                window_ms = 300
                                cutoff_time = now - (window_ms / 1000.0)
                                recent_packets = [
                                    (t, e) for t, e in buf if t >= cutoff_time]

                                high_energy_count = sum(
                                    1 for _, e in recent_packets if e >= energy_threshold)
                                total_count = len(recent_packets)
                                energy_percentage = (
                                    high_energy_count / total_count * 100) if total_count > 0 else 0

                                peak_energy = max(
                                    (e for _, e in recent_packets), default=0)

                                time_since_last_interrupt = now - conn.last_interrupt_time
                                debounced = time_since_last_interrupt >= (
                                    INTERRUPT_DEBOUNCE_MS / 1000.0)

                                vad_ok = vad_dur_ms >= INTERRUPT_MIN_SPEECH_MS
                                energy_ok = energy_percentage >= 60 or peak_energy >= (
                                    conn.baseline_energy * INTERRUPT_BASELINE_FACTOR)
                                current_energy_ok = energy >= (
                                    energy_threshold * 0.8)

                                all_checks_pass = vad_ok and energy_ok and current_energy_ok and debounced

                                if all_checks_pass:
                                    conn.interrupt_requested = True
                                    conn.last_interrupt_time = now
                                    _logger.info(
                                        "ðŸ›‘ INTERRUPT! VAD: %.0fms | Energy: %.0f%% | Peak: %d | Threshold: %d",
                                        vad_dur_ms, energy_percentage, peak_energy, energy_threshold
                                    )

                        # ========================================
                        # âœ… PROCESS TRANSCRIPT
                        # ========================================

                        if not conn.currently_speaking and not conn.interrupt_requested:
                            if processing_task is None or processing_task.done():
                                processing_task = asyncio.create_task(
                                    process_streaming_transcript(
                                        current_call_sid)
                                )

                    except Exception as e:
                        pass

            elif event == "stop":
                break

    except WebSocketDisconnect:
        _logger.info(f"📞 WebSocket disconnected for call: {current_call_sid}")
    except Exception as e:
        _logger.error(f"❌ WebSocket error: {e}")
    finally:
        try:
            if processing_task and not processing_task.done():
                processing_task.cancel()
        except:
            pass

        try:
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
        except:
            pass

        # ✅ CRITICAL: Save transcript BEFORE disconnecting (in case /voice/status comes later)
        if current_call_sid:
            conn = manager.get(current_call_sid)
            if conn:
                _logger.info(f"💾 Saving transcript on WebSocket disconnect for: {current_call_sid}")
                _logger.info(f"   - conversation_history entries: {len(conn.conversation_history)}")
                await save_conversation_transcript(current_call_sid, conn)
            else:
                _logger.warning(f"⚠️ No connection found on WebSocket disconnect for: {current_call_sid}")
            
            try:
                await manager.disconnect(current_call_sid)
            except:
                pass

        try:
            await websocket.close()
        except:
            pass


@app.api_route("/", methods=["GET", "POST"])
async def index_page():
    return {
        "status": "ok",
        "message": "Twilio RAG Voice System - GPU + SMART VOICE INTERRUPTS + TRANSFER CONFIRMATION",
        "device": str(DEVICE),
        "features": [
            "âœ… Transfer requires user confirmation",
            "âœ… End call is immediate (no confirmation)",
            "âœ… Interrupts on real voice (configurable)",
            "âœ… GPU-accelerated RAG",
            "âœ… Streaming STT/TTS pipeline",
            "âœ… Smart conversation handling",
            f"âœ… {SILENCE_THRESHOLD_SEC}s silence before processing"
        ]
    }


@app.get("/gpu-status")
async def gpu_status():
    """🚀 GPU status"""
    status = {
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        try:
            status.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "memory": {
                    "total_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}",
                    "allocated_gb": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}",
                    "free_gb": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f}"
                },
            })
        except Exception as e:
            status["gpu_error"] = str(e)

    try:
        result = ollama.list()
        status["ollama"] = {
            "models": [m["name"] for m in result.get("models", [])],
            "current_model": OLLAMA_MODEL
        }
    except Exception as e:
        status["ollama"] = {"error": str(e)}

    status["embedding"] = {
        "model": EMBED_MODEL,
        "device": str(embedder.device),
    }

    return status


@app.post("/voice/outbound")
@app.get("/voice/outbound")
async def voice_outbound(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "")
    
    # Check if recording is enabled for this call
    call_data = pending_call_data.get(call_sid, {})
    enable_recording = call_data.get("enable_recording", False)
    
    response = VoiceResponse()
    
    # Enable recording if requested
    if enable_recording:
        response.record(
            recording_status_callback=f"{PUBLIC_URL}/recording-callback",
            recording_status_callback_method="POST",
            recording_status_callback_event="completed"
        )
        _logger.info(f"🎙️ Recording enabled for call: {call_sid}")
    
    response.say(" ")
    response.pause(length=0)

    connect = Connect()
    connect.stream(url=f"wss://{public_ws_host()}/media-stream")
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")


@app.post("/voice/inbound")
@app.get("/voice/inbound")
async def voice_inbound(request: Request):
    """
    Handle incoming calls - route to appropriate agent
    
    ✨ ELEVENLABS-COMPATIBLE: Always creates conversation record and tracks status
    """
    form = await request.form()
    from_number = form.get("From", "")
    to_number = form.get("To", "")
    call_sid = form.get("CallSid", "")
    
    _logger.info(f"📞 Inbound call: from={from_number}, to={to_number}, call_sid={call_sid}")
    
    db = SessionLocal()
    try:
        # Try to find agent linked to this phone number
        phone_record = db.query(PhoneNumber).filter(
            PhoneNumber.phone_number == to_number,
            PhoneNumber.is_active == True
        ).first()
        
        agent = None
        if phone_record and phone_record.agent_id:
            agent = db.query(Agent).filter(
                Agent.agent_id == phone_record.agent_id,
                Agent.is_active == True
            ).first()
        
        # Fallback to first active agent if no phone number mapping
        if not agent:
            agent = db.query(Agent).filter(Agent.is_active == True).first()
        
        # ✨ ALWAYS store call data for WebSocket (like ElevenLabs)
        pending_call_data[call_sid] = {
            "agent_id": agent.agent_id if agent else None,
            "dynamic_variables": {"caller_number": from_number},
            "custom_voice_id": None,
            "custom_model": None,
            "custom_first_message": None,
            "from_number": from_number,  # ✅ Caller's phone number
            "to_number": to_number,      # Agent's phone number
            "enable_recording": False,
            "direction": "inbound"
        }
        
        # ✨ ALWAYS create conversation record (like ElevenLabs)
        conversation = Conversation(
            conversation_id=call_sid,
            agent_id=agent.agent_id if agent else None,
            phone_number=from_number,
            status="initiated",
            dynamic_variables={"caller_number": from_number, "direction": "inbound"},
            call_metadata={"direction": "inbound", "to_number": to_number}
        )
        db.add(conversation)
        db.commit()
        
        if agent:
            _logger.info(f"✅ Inbound call routed to agent: {agent.agent_id} ({agent.name})")
        else:
            _logger.warning("⚠️ No active agent found for inbound call - using default behavior")
        
        # ✨ ALWAYS send webhooks (like ElevenLabs)
        webhooks = db.query(WebhookConfig).filter(
            WebhookConfig.is_active == True
        ).all()
        
        # Filter webhooks by agent_id or global (agent_id == None)
        for webhook in webhooks:
            should_send = False
            if webhook.agent_id is None:
                should_send = True  # Global webhook
            elif agent and webhook.agent_id == agent.agent_id:
                should_send = True  # Agent-specific webhook
            
            if should_send and ("call.initiated" in webhook.events or not webhook.events):
                asyncio.create_task(send_webhook(
                    webhook.webhook_url,
                    "call.initiated",
                    {
                        "conversation_id": call_sid,
                        "agent_id": agent.agent_id if agent else None,
                        "from_number": from_number,
                        "to_number": to_number,
                        "direction": "inbound",
                        "status": "initiated"
                    }
                ))
    finally:
        db.close()
    
    response = VoiceResponse()
    response.say(" ")
    response.pause(length=0)
    
    connect = Connect()
    connect.stream(url=f"wss://{public_ws_host()}/media-stream")
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")


@app.post("/make-call")
async def make_call(request: CallRequest):
    try:
        to_number = request.to_number

        _logger.info(f"📞 Starting outbound call: to={to_number}")  
        webhook = f"{PUBLIC_URL.rstrip('/')}/voice/outbound"
        status_callback_url = f"{PUBLIC_URL.rstrip('/')}/voice/status"

        call_sid = twilio_client.calls.create(
            to=to_number,
            from_=+15108963495,  # your fixed Twilio number
            url=webhook,
            method="POST",
            status_callback=status_callback_url,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST"
        )

        return {
            "success": True,
            "message": "Call initiated successfully",
            "call_sid": call_sid.sid,
            "to": to_number
        }

    except Exception as e:
        _logger.exception("âŒ Call initiation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/status")
async def voice_status(request: Request):
    """Enhanced voice status handler with transcript saving and webhooks"""
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")
    
    _logger.info(f"📞 Call status update: {call_sid} -> {call_status}")
    
    if call_status in {"completed", "failed", "busy", "no-answer", "canceled"} and call_sid:
        conn = manager.get(call_sid)
        
        # Save transcript before disconnecting
        if conn:
            await save_conversation_transcript(call_sid, conn)
        
        # Handle call end (update DB and send webhooks)
        await handle_call_end(call_sid, call_status)
        
        # Disconnect
        await manager.disconnect(call_sid)
        
        # Clean up pending call data
        if call_sid in pending_call_data:
            del pending_call_data[call_sid]

    return PlainTextResponse("OK")


@app.get("/health")
async def health():
    """Health check"""
    health_data = {
        "status": "ok",
        "mode": "GPU + FIXED_INTERRUPTS + CONFIRMATION + 1s_SILENCE",
        "device": str(DEVICE),
        "docs_count": collection.count(),
        "active_connections": len(manager._conns),
        "confirmation_enabled": True,
        "silence_threshold_sec": SILENCE_THRESHOLD_SEC,
        "interrupt_settings": {
            "enabled": INTERRUPT_ENABLED,
            "min_energy": INTERRUPT_MIN_ENERGY,
            "baseline_factor": INTERRUPT_BASELINE_FACTOR,
            "min_speech_ms": INTERRUPT_MIN_SPEECH_MS,
            "require_text": INTERRUPT_REQUIRE_TEXT
        }
    }

    if torch.cuda.is_available():
        health_data["gpu_memory_allocated_gb"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}"

    return health_data


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = 50) -> List[str]:
    """
    Advanced semantic chunking with overlap for context preservation
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text.strip())

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for i, sentence in enumerate(sentences):
        # If adding this sentence would exceed chunk size
        if current_chunk and len(current_chunk) + len(sentence) + 1 > size:
            chunks.append(current_chunk.strip())

            # Start new chunk with overlap from previous chunk
            if overlap > 0:
                # Take last few sentences from current chunk for overlap
                prev_sentences = current_chunk.split('. ')
                overlap_sentences = prev_sentences[-2:] if len(
                    prev_sentences) > 2 else prev_sentences[-1:]
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
            else:
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter and deduplicate
    final_chunks = []
    seen = set()
    for chunk in chunks:
        if len(chunk) < 25:
            continue
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:12]
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            final_chunks.append(chunk)

    _logger.info(
        f"ðŸ“ Created {len(final_chunks)} overlapping chunks (size: {size}, overlap: {overlap})")
    return final_chunks


def build_index_from_file(path: str = DATA_FILE) -> Tuple[int, int]:
    """Build ChromaDB index using contextual chunking"""

    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_FILE not found: {path}")

    _logger.info(f"ðŸ“– Reading data from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        _logger.warning("âš  DATA_FILE is empty.")
        return (0, 0)

    # Use the new chunking with overlap
    # 50 character overlap
    docs = _chunk_text(raw_text, CHUNK_SIZE, overlap=50)

    # Clear existing collection
    try:
        chroma_client.delete_collection("docs")
    except:
        pass
    collection = chroma_client.get_or_create_collection("docs")

    metadatas = []
    ids = []

    for i, doc in enumerate(docs):
        metadatas.append({
            "chunk_id": i,
            "length": len(doc),
            "word_count": len(doc.split()),
            "type": "contextual_chunk"
        })
        ids.append(f"ctx_{i}_{uuid.uuid4().hex[:8]}")

    total = len(docs)
    if total == 0:
        _logger.warning("âš  No valid chunks found.")
        return (0, 0)

    _logger.info(f"ðŸ”„ Generating {total} embeddings...")

    start = time.time()

    with torch.no_grad():
        embeddings = embedder.encode(
            docs,
            device=DEVICE,
            batch_size=64 if DEVICE == "cuda" else 32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        ).tolist()

    duration = time.time() - start
    _logger.info(f"âœ… Embeddings done in {duration:.2f}s")

    # Batch insertion
    CHROMA_MAX_BATCH = 5000

    for start_idx in range(0, total, CHROMA_MAX_BATCH):
        end_idx = start_idx + CHROMA_MAX_BATCH
        collection.add(
            documents=docs[start_idx:end_idx],
            embeddings=embeddings[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
            ids=ids[start_idx:end_idx]
        )

    _logger.info(f"âœ… Contextual index built: {total} meaningful chunks")
    return (total, total)

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["server", "build", "test"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9001, type=int)
    args = parser.parse_args()

    if args.mode == "build":
        docs, chunks = build_index_from_file(DATA_FILE)
        print(f"âœ… Built index: {docs} docs, {chunks} chunks")
        print(f"🚀 Device used: {DEVICE}")
    elif args.mode == "test":
        print(f"Test mode (device: {DEVICE}):")

        async def test_query():
            while True:
                q = input("> ").strip()
                if q.lower() in {"exit", "quit"}:
                    break
                result = ""
                print("Response: ", end="", flush=True)
                async for token in query_rag_streaming(q):
                    result += token
                    print(token, end="", flush=True)
                print("\n")

        asyncio.run(test_query())
    else:
        _logger.info("🚀 Starting server on %s:%s", args.host, args.port)
        _logger.info(f"🔥 GPU: {DEVICE}")
        _logger.info("âœ… Transfer confirmation: ENABLED")
        _logger.info(
            f"â±ï¸  Silence threshold: {SILENCE_THRESHOLD_SEC}s (utterance_end={UTTERANCE_END_MS}ms)")
        _logger.info(
            f"🎯 Interrupt: enabled={INTERRUPT_ENABLED}, min_speech={INTERRUPT_MIN_SPEECH_MS}ms, "
            f"min_energy={INTERRUPT_MIN_ENERGY}, baseline_factor={INTERRUPT_BASELINE_FACTOR}, "
            f"require_text={INTERRUPT_REQUIRE_TEXT}"
        )
        uvicorn.run("new:app",
                    host=args.host,
                    port=args.port,
                    reload=False,
                    timeout_keep_alive=60,
                    timeout_graceful_shutdown=30,
                    ws_ping_interval=10.0,    # Add WebSocket ping
                    ws_ping_timeout=10.0
                    )


