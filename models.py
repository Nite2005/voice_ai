# ================================
# PYDANTIC MODELS (API Requests)
# ================================
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

load_dotenv()

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
    agent_phone_number_id: Optional[str] = None  # For compatibility
    to_number: str
    first_message: Optional[str] = None  # Override agent's default

    conversation_initiation_client_data: Optional[Dict] = Field(default_factory=dict)
    enable_recording: bool = False  # Enable call recording


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
    tool_name: str = Field(..., description="Name of the tool (used in [TOOL:name:...] syntax)")
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

