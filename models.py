# # ================================
# # PYDANTIC MODELS (API Requests)
# # ================================
# import os
# import io
# import json
# import uuid
# import base64
# import asyncio
# import wave
# import audioop
# import hashlib
# import time
# import re
# import struct
# from typing import Dict, Optional, List, Tuple
# from collections import deque
# from sqlalchemy.orm import Session
# import logging
# from logging.handlers import RotatingFileHandler
# from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
# from fastapi.responses import Response, PlainTextResponse
# from fastapi.security import APIKeyHeader
# from fastapi.middleware.cors import CORSMiddleware
# from twilio.twiml.voice_response import VoiceResponse, Connect
# from twilio.rest import Client as TwilioClient
# from pydantic import BaseModel, Field
# from sqlalchemy import create_engine, Column, String, Text, Integer, Float, Boolean, DateTime, JSON
# from sqlalchemy.orm import sessionmaker, Session, declarative_base
# from datetime import datetime as dt
# from dotenv import load_dotenv

# load_dotenv()

# class CallRequest(BaseModel):
#     to_number: str


# class AgentCreate(BaseModel):
#     name: str
#     system_prompt: str
#     first_message: Optional[str] = None
#     voice_provider: str = "deepgram"
#     voice_id: str = "aura-2-thalia-en"
#     model_provider: str = "ollama"
#     model_name: str = "mixtral:8x7b"
#     interrupt_enabled: bool = True
#     silence_threshold_sec: float = 0.8


# class AgentUpdate(BaseModel):
#     name: Optional[str] = None
#     system_prompt: Optional[str] = None
#     first_message: Optional[str] = None
#     voice_provider: Optional[str] = None
#     voice_id: Optional[str] = None
#     model_provider: Optional[str] = None
#     model_name: Optional[str] = None
#     interrupt_enabled: Optional[bool] = None
#     silence_threshold_sec: Optional[float] = None
#     is_active: Optional[bool] = None


# class OutboundCallRequest(BaseModel):
#     """ElevenLabs-compatible outbound call request"""
#     agent_id: str
#     agent_phone_number_id: Optional[str] = None  # For compatibility
#     to_number: str
#     first_message: Optional[str] = None  # Override agent's default

#     conversation_initiation_client_data: Optional[Dict] = Field(default_factory=dict)
#     enable_recording: bool = False  # Enable call recording


# class WebhookCreate(BaseModel):
#     """Create webhook for agent events"""
#     webhook_url: str = Field(..., description="URL to send webhook events to")
#     events: List[str] = Field(
#         default_factory=lambda: ["call.initiated", "call.started", "call.ended"],
#         description="List of events to subscribe to"
#     )
#     agent_id: Optional[str] = Field(None, description="Agent ID (null for global webhook)")


# class WebhookResponse(BaseModel):
#     """Webhook response"""
#     success: bool
#     webhook_id: int
#     webhook_url: str
#     events: List[str]
#     agent_id: Optional[str] = None


# class ToolCreate(BaseModel):
#     """Create custom tool for agent"""
#     tool_name: str = Field(..., description="Name of the tool (used in [TOOL:name:...] syntax)")
#     description: str = Field(..., description="Description of what the tool does")
#     webhook_url: str = Field(..., description="URL to call when tool is invoked")
#     parameters: Optional[Dict] = Field(
#         default_factory=dict,
#         description="Parameter schema for the tool",
#         example={
#             "param1": {"type": "string", "required": True, "description": "First parameter"},
#             "param2": {"type": "number", "required": False, "description": "Second parameter"}
#         }
#     )













import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime as dt

from sqlalchemy import create_engine, Column, String, Text, Integer, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from pydantic import BaseModel, Field

# ================================
# DATABASE MODELS
# ================================
Base = declarative_base()

class Agent(Base):
    """Agent configuration"""
    __tablename__ = "agents"
    
    agent_id = Column(String(100), primary_key=True)
    name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=False)
    first_message = Column(Text, nullable=True)
    
    # Voice settings
    voice_provider = Column(String(50), default="deepgram")
    voice_id = Column(String(100), default="aura-2-thalia-en")
    
    # Model settings
    model_provider = Column(String(50), default="ollama")
    model_name = Column(String(100), default="mixtral:8x7b")
    
    # Behavior settings
    interrupt_enabled = Column(Boolean, default=True)
    silence_threshold_sec = Column(Float, default=0.8)
    
    # Metadata
    created_at = Column(DateTime, default=dt.utcnow)
    updated_at = Column(DateTime, default=dt.utcnow, onupdate=dt.utcnow)
    user_id = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)


class Conversation(Base):
    """Store conversation history"""
    __tablename__ = "conversations"
    
    conversation_id = Column(String(100), primary_key=True)
    agent_id = Column(String(100), nullable=False)
    
    # Call details
    phone_number = Column(String(50), nullable=True)
    status = Column(String(50), default="initiated")
    
    # Transcript
    transcript = Column(Text, nullable=True)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    duration_secs = Column(Integer, default=0)
    
    # Metadata
    dynamic_variables = Column(JSON, nullable=True)
    call_metadata = Column(JSON, nullable=True)
    
    # Results
    ended_reason = Column(String(100), nullable=True)
    recording_url = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=dt.utcnow)


class WebhookConfig(Base):
    """Webhook configuration for call events"""
    __tablename__ = "webhook_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=True)
    
    webhook_url = Column(String(500), nullable=False)
    events = Column(JSON, default=list)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.utcnow)


class PhoneNumber(Base):
    """Phone numbers linked to agents"""
    __tablename__ = "phone_numbers"
    
    id = Column(String(100), primary_key=True)
    phone_number = Column(String(50), nullable=False, unique=True)
    agent_id = Column(String(100), nullable=True)
    provider = Column(String(50), default="twilio")
    label = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.utcnow)


class KnowledgeBase(Base):
    """Knowledge base documents per agent"""
    __tablename__ = "knowledge_bases"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=False)
    document_id = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    kb_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=dt.utcnow)


class AgentTool(Base):
    """Custom tools per agent"""
    __tablename__ = "agent_tools"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=False)
    tool_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    webhook_url = Column(String(500), nullable=True)
    parameters = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.utcnow)


# Database connection
DATABASE_URL = os.getenv("AGENT_DATABASE_URL", "sqlite:///./agents.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ================================
# PYDANTIC MODELS (API Requests)
# ================================

class CallRequest(BaseModel):
    to_number: str


class AgentCreate(BaseModel):
    name: str
    system_prompt: str
    first_message: Optional[str] = None
    voice_provider: str = "deepgram"
    voice_id: str = "aura-2-thalia-en"
    model_provider: str = "ollama"
    model_name: str = "mixtral:8x7b"
    interrupt_enabled: bool = True
    silence_threshold_sec: float = 0.8


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    first_message: Optional[str] = None
    voice_provider: Optional[str] = None
    voice_id: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    interrupt_enabled: Optional[bool] = None
    silence_threshold_sec: Optional[float] = None
    is_active: Optional[bool] = None


class OutboundCallRequest(BaseModel):
    """ElevenLabs-compatible outbound call request"""
    agent_id: str
    agent_phone_number_id: Optional[str] = None
    to_number: str
    first_message: Optional[str] = None

    conversation_initiation_client_data: Optional[Dict] = Field(default_factory=dict)
    enable_recording: bool = False


class WebhookCreate(BaseModel):
    """Create webhook for agent events"""
    webhook_url: str = Field(..., description="URL to send webhook events to")
    events: List[str] = Field(
        default_factory=lambda: ["call.initiated", "call.started", "call.ended"],
        description="List of events to subscribe to"
    )
    agent_id: Optional[str] = Field(None, description="Agent ID (null for global webhook)")


class WebhookResponse(BaseModel):
    """Webhook response"""
    success: bool
    webhook_id: int
    webhook_url: str
    events: List[str]
    agent_id: Optional[str] = None


class ToolCreate(BaseModel):
    """Create custom tool for agent"""
    tool_name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    webhook_url: str = Field(..., description="URL to call when tool is invoked")
    parameters: Optional[Dict] = Field(
        default_factory=dict,
        description="Parameter schema for the tool",
        example={
            "param1": {"type": "string", "required": True, "description": "First parameter"},
            "param2": {"type": "number", "required": False, "description": "Second parameter"}
        }
    )