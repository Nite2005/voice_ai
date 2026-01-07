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


# ================================
# DATABASE MODELS (Agent Management)
# ================================
Base = declarative_base()

class Agent(Base):
    """Agent configuration (like ElevenLabs Agent)"""
    __tablename__ = "agents"
    
    agent_id = Column(String(100), primary_key=True)
    name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=False)
    first_message = Column(Text, nullable=True)
    
    # Voice settings
    voice_provider = Column(String(50), default="deepgram")  # deepgram, elevenlabs, etc.
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
    user_id = Column(String(100), nullable=True)  # For multi-tenancy
    is_active = Column(Boolean, default=True)


class Conversation(Base):
    """Store conversation history (like ElevenLabs Conversation)"""
    __tablename__ = "conversations"
    
    conversation_id = Column(String(100), primary_key=True)  # Twilio call_sid
    agent_id = Column(String(100), nullable=False)
    
    # Call details
    phone_number = Column(String(50), nullable=True)
    status = Column(String(50), default="initiated")  # initiated, in-progress, completed, failed
    
    # Transcript
    transcript = Column(Text, nullable=True)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    duration_secs = Column(Integer, default=0)
    
    # Metadata (renamed to avoid SQLAlchemy reserved word)
    dynamic_variables = Column(JSON, nullable=True)  # Lead data
    call_metadata = Column(JSON, nullable=True)  # Call metadata (was 'metadata')
    
    # Results
    ended_reason = Column(String(100), nullable=True)
    recording_url = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=dt.utcnow)


class WebhookConfig(Base):
    """Webhook configuration for call events"""
    __tablename__ = "webhook_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=True)  # null = global webhook
    
    webhook_url = Column(String(500), nullable=False)
    events = Column(JSON, default=list)  # ["call.started", "call.ended", etc.]
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.utcnow)


class PhoneNumber(Base):
    """Phone numbers linked to agents"""
    __tablename__ = "phone_numbers"
    
    id = Column(String(100), primary_key=True)
    phone_number = Column(String(50), nullable=False, unique=True)
    agent_id = Column(String(100), nullable=True)  # Linked agent
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
    webhook_url = Column(String(500), nullable=True)  # External webhook for tool
    parameters = Column(JSON, default=dict)  # Tool parameters schema
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
