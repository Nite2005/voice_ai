
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

from db.session import init_db
from db.session import get_db
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
# from connection_manager import ConnectionManager
# manager = ConnectionManager()
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
# âš¡ STREAMING PIPELINE
# ----------------------------
async def process_streaming_transcript(call_sid: str):
    """âœ… FIXED: Waits for COMPLETE final transcript + proper silence"""
    conn = manager.get(call_sid)
    if not conn or conn.is_responding:
        return

    if conn.interrupt_requested:
        _logger.debug("â­ï¸ Skipping - interrupt active")
        return

    now = time.time()

    # ========================================
    # âœ… CHECK 1: STUCK VAD TIMEOUT
    # ========================================
    if conn.user_speech_detected and conn.vad_triggered_time:
        vad_duration = now - conn.vad_triggered_time
        if vad_duration > conn.vad_timeout:
            _logger.warning(
                f"âš ï¸ Clearing stuck VAD (duration: {vad_duration:.1f}s)")
            conn.user_speech_detected = False
            conn.speech_start_time = None
            conn.vad_triggered_time = None
            conn.vad_validated = False

    # ========================================
    # âœ… CHECK 2: WAIT FOR USER TO FINISH SPEAKING
    # ========================================

    # Check if user is STILL speaking (recent interim text)
    if conn.last_interim_time and (now - conn.last_interim_time) < 0.5:
        _logger.debug(
            "â¸ï¸ User still adding to sentence (recent interim) - waiting...")
        return

    # If VAD says user is still speaking, wait
    if conn.user_speech_detected:
        _logger.debug("â¸ï¸ User still speaking (VAD active) - waiting...")
        return

    # ========================================
    # âœ… CHECK 3: MUST HAVE FINAL RESULT
    # ========================================

    # ðŸ”§ FIX: Must have at least ONE FINAL result before processing
    if not conn.stt_is_final:
        _logger.debug("â¸ï¸ Waiting for FINAL result...")
        return

    # ðŸ”§ FIX: Buffer must not be empty
    if not conn.stt_transcript_buffer or len(conn.stt_transcript_buffer.strip()) < 3:
        _logger.debug("â¸ï¸ Buffer empty or too short")
        return

    # ========================================
    # âœ… CHECK 4: ENFORCE SILENCE THRESHOLD
    # ========================================

    if not conn.last_speech_time:
        _logger.debug("â¸ï¸ No speech time recorded")
        return

    silence_elapsed = now - conn.last_speech_time

    # Enforce minimum silence
    if silence_elapsed < SILENCE_THRESHOLD_SEC:
        _logger.debug("â¸ï¸ Waiting for silence: %.2fs / %.1fs",
                      silence_elapsed, SILENCE_THRESHOLD_SEC)
        return

    # ========================================
    # âœ… CHECK 5: DOUBLE-CHECK FOR NEW SPEECH
    # ========================================

    # âœ… FIX: Set is_responding flag EARLY to prevent race condition
    conn.is_responding = True

    # Small delay to catch any last-moment speech
    await asyncio.sleep(0.05)

    # Re-check after delay
    final_silence = time.time() - conn.last_speech_time

    # If new speech arrived during our checks, abort
    if final_silence < SILENCE_THRESHOLD_SEC:
        _logger.debug(
            "â¸ï¸ New speech detected during threshold check - resetting")
        return

    # Check if new interim/final arrived during our checks
    if conn.last_interim_time and (time.time() - conn.last_interim_time) < 0.3:
        _logger.debug("â¸ï¸ New interim detected - waiting for completion")
        return

    # ========================================
    # âœ… ALL CHECKS PASSED - PROCESS NOW
    # ========================================

    _logger.info("âœ… %.1fs silence threshold met (%.2fs)",
                 SILENCE_THRESHOLD_SEC, final_silence)

    try:
        # Get the COMPLETE accumulated transcript
        text = conn.stt_transcript_buffer.strip()

        # 🧠 Detect user intent (ElevenLabs-style)
        intent = detect_intent(text)
        conn.last_intent = intent

        _logger.info(f"🎯 Detected intent: {intent} | text='{text}'")


        # ⚡ Handle simple intents WITHOUT calling LLM
        if intent == "CONFIRMATION":
            # ✅ FIX: Clear buffer BEFORE returning to prevent infinite loop
            conn.stt_transcript_buffer = ""
            conn.stt_is_final = False
            conn.last_interim_text = ""
            
            # ✅ FIX: Save to conversation history
            conn.conversation_history.append({
                "user": text,
                "assistant": "Okay, got it.",
                "timestamp": time.time()
            })
            
            await speak_text_streaming(call_sid, "Okay, got it.")
            conn.is_responding = False
            return

        if intent == "HESITATION":
            # ✅ FIX: Clear buffer BEFORE returning to prevent infinite loop
            conn.stt_transcript_buffer = ""
            conn.stt_is_final = False
            conn.last_interim_text = ""
            
            # ✅ FIX: Save to conversation history
            conn.conversation_history.append({
                "user": text,
                "assistant": "No worries, take your time.",
                "timestamp": time.time()
            })
            
            await speak_text_streaming(call_sid, "No worries, take your time.")
            conn.is_responding = False
            return

        if intent == "GOODBYE":
            # ✅ FIX: Clear buffer BEFORE ending call
            conn.stt_transcript_buffer = ""
            conn.stt_is_final = False
            conn.last_interim_text = ""
            
            # ✅ FIX: Save to conversation history
            conn.conversation_history.append({
                "user": text,
                "assistant": "Thanks for your time. Have a great day.",
                "timestamp": time.time()
            })
            
            await speak_text_streaming(call_sid, "Thanks for your time. Have a great day.")
            await end_call_tool(call_sid, "user_goodbye")
            return



        # 🧠 Update call phase
        if conn.call_phase == "CALL_START":
            conn.call_phase = "DISCOVERY"

        elif conn.call_phase == "DISCOVERY":
            if len(conn.conversation_history) >= 2:
                conn.call_phase = "ACTIVE"

        elif conn.call_phase == "ACTIVE":
            if any(w in text.lower() for w in ["bye", "thank", "that's all", "no more"]):
                conn.call_phase = "CLOSING"


        # One final interrupt check
        if conn.interrupt_requested:
            _logger.debug("â­ï¸ Interrupt detected - aborting")
            conn.stt_transcript_buffer = ""
            conn.stt_is_final = False
            conn.last_interim_text = ""
            return

        # âœ… CHECK: Handle pending action confirmation
        if conn.pending_action:
            _logger.info(
                "Pending action detected. Checking user response: '%s'", text)
            confirmation = detect_confirmation_response(text)

            if confirmation == "yes":
                _logger.info("User confirmed action: %s",
                             conn.pending_action.get("tool"))
                result = await execute_detected_tool(call_sid, conn.pending_action)
                _logger.info("Confirmed tool execution result: %s", result)
                conn.pending_action = None
                conn.stt_transcript_buffer = ""
                conn.stt_is_final = False
                conn.last_interim_text = ""
                return
            elif confirmation == "no":
                await speak_text_streaming(call_sid, "Understood, I've cancelled that request. How else can I help you?")
                conn.pending_action = None
                conn.stt_transcript_buffer = ""
                conn.stt_is_final = False
                conn.last_interim_text = ""
                return
            else:
                # ✅ FIX: Check if user changed topic (long response = new question)
                word_count = len(text.split())
                if word_count > 5:  # Long response suggests topic change
                    _logger.info("âœ… User changed topic (%d words) - clearing pending action", word_count)
                    conn.pending_action = None
                    # Continue to normal LLM processing below
                else:
                    # Still asking for confirmation
                    await speak_text_streaming(call_sid, "Could you please confirm yes or no?")
                    conn.stt_transcript_buffer = ""
                    conn.stt_is_final = False
                    conn.last_interim_text = ""
                    return

        # âœ… CRITICAL: Clear buffer AFTER getting text
        conn.stt_transcript_buffer = ""
        conn.stt_is_final = False
        conn.last_interim_text = ""

        if not text or len(text) < 3:
            _logger.warning(f"⚠️ Text too short or empty: '{text}' - skipping")
            conn.is_responding = False
            return

        # Input: User transcript
        _logger.info(f"📝 USER INPUT: '{text}'")
        print(f"INPUT: {text}")

        t_start = time.time()

        # Stream LLM response
        response_buffer = ""
        sentence_buffer = ""
        sentence_count = 0
        MAX_SENTENCES = 10

        async for token in query_rag_streaming(text, conn.conversation_history, call_sid=call_sid):
            if conn.interrupt_requested:
                _logger.info("â­ï¸ Generation interrupted")
                break

            token = re.sub(r'\[(?:TOOL|CONFIRM_TOOL):[^\]]+\]', '', token)

            response_buffer += token
            sentence_buffer += token

            # Flush on sentence end
            if sentence_buffer.rstrip().endswith(('.', '?', '!')):
                sentence = sentence_buffer.strip()

                # ✅ Only queue non-empty sentences (skip if only had tool tags)
                if sentence:
                    # ✅ Clean markdown before speaking
                    clean_sentence = clean_markdown_for_tts(sentence)
                    sentence_count += 1
                    _logger.info("🎯 Sentence %d: '%s'",
                                 sentence_count, clean_sentence)

                    # ✅ FIX: Handle queue full gracefully with backpressure
                    try:
                        await asyncio.wait_for(conn.tts_queue.put(clean_sentence), timeout=2.0)
                    except asyncio.TimeoutError:
                        # If queue is full, skip this sentence to prevent deadlock
                        if conn.interrupt_requested:
                            break
                    except Exception as e:
                        if conn.interrupt_requested:
                            break

                sentence_buffer = ""

                if sentence_count >= MAX_SENTENCES:
                    break

        # Send any remaining text
        if not conn.interrupt_requested and sentence_buffer.strip():
            final_sentence = sentence_buffer.strip()
            # ✅ Only queue if not just tool tags
            if final_sentence and not re.match(r'^\s*\[\w+:[^\]]*\]\s*$', final_sentence):
                # ✅ Clean markdown before speaking
                clean_final = clean_markdown_for_tts(final_sentence)
                _logger.info("🎯 Final: '%s'", clean_final)
                try:
                    await asyncio.wait_for(conn.tts_queue.put(clean_final), timeout=2.0)
                except asyncio.TimeoutError:
                    _logger.warning("TTS queue full, dropping final sentence")
                except Exception as e:
                    _logger.error(f"Error queuing final sentence: {e}")

        # ✨ CRITICAL FIX: Save to conversation history IMMEDIATELY after LLM generates response
        # This ensures transcript is saved even if user hangs up during TTS playback
        cleaned_response, tool_data = parse_llm_response(response_buffer)
        
        if not conn.interrupt_requested and response_buffer.strip():
            conn.conversation_history.append({
                "user": text,
                "assistant": cleaned_response,
                "timestamp": time.time()
            })
            _logger.info(f"✅ Added to conversation_history BEFORE TTS: user='{text[:50]}...', assistant='{cleaned_response[:50]}...'")
            _logger.info(f"   Total history entries: {len(conn.conversation_history)}")

            # Keep last 10 exchanges
            if len(conn.conversation_history) > 10:
                conn.conversation_history = conn.conversation_history[-10:]
        else:
            _logger.warning(f"⚠️ NOT added to history - interrupt: {conn.interrupt_requested}, response empty: {not response_buffer.strip()}")

        # âœ¨ Handle tool calls - execute and get AI natural response
        if tool_data:
            _logger.info("ðŸ§ Tool detected: %s - requires_confirmation: %s",
                         tool_data.get('tool'), tool_data.get('requires_confirmation'))

            if tool_data.get("requires_confirmation"):
                conn.pending_action = tool_data
                _logger.info("â³ Awaiting user confirmation for: %s",
                             tool_data.get("tool"))
            else:
                # âœ¨ Execute tool and get result
                _logger.info("âš¡ Executing tool immediately...")
                tool_result = await execute_detected_tool(call_sid, tool_data)
                _logger.info(f"ðŸ¦ Tool result: {tool_result}")
                
                # âœ¨ Feed tool result back to AI for natural response
                tool_context = f"""You just executed the tool '{tool_data['tool']}' with result:

Tool Result:
{tool_result.get('response', tool_result.get('data', tool_result))}

Generate a brief, natural response (1-2 sentences) to inform the user about this result. Be conversational."""
                
                _logger.info("ðŸ¤– Asking AI to generate natural response from tool result...")
                
                # Clear buffers for tool response
                ai_tool_response = ""
                tool_sentence_buffer = ""
                tool_sentence_count = 0
                
                # Get AI's natural response to the tool result
                async for token in query_rag_streaming(tool_context, conn.conversation_history, call_sid=call_sid):
                    if conn.interrupt_requested:
                        break
                    
                    ai_tool_response += token
                    tool_sentence_buffer += token
                    
                    # Flush on sentence end
                    if tool_sentence_buffer.rstrip().endswith(('.', '?', '!')):
                        sentence = tool_sentence_buffer.strip()
                        if sentence:
                            clean_sentence = clean_markdown_for_tts(sentence)
                            tool_sentence_count += 1
                            _logger.info("ðŸŽ¯ Tool response sentence %d: '%s'", tool_sentence_count, clean_sentence)
                            
                            try:
                                await asyncio.wait_for(conn.tts_queue.put(clean_sentence), timeout=2.0)
                            except asyncio.TimeoutError:
                                if conn.interrupt_requested:
                                    break
                        
                        tool_sentence_buffer = ""
                        if tool_sentence_count >= 3:  # Limit tool responses
                            break
                
                # Send any remaining text
                if not conn.interrupt_requested and tool_sentence_buffer.strip():
                    final_sentence = tool_sentence_buffer.strip()
                    if final_sentence:
                        clean_final = clean_markdown_for_tts(final_sentence)
                        try:
                            await asyncio.wait_for(conn.tts_queue.put(clean_final), timeout=2.0)
                        except:
                            pass
                
                # âœ¨ Update conversation history with tool execution
                if conn.conversation_history:
                    conn.conversation_history[-1]["tool_executed"] = tool_data['tool']
                    conn.conversation_history[-1]["tool_result"] = tool_result
                    conn.conversation_history[-1]["ai_response"] = ai_tool_response
                    _logger.info(f"âœ… Updated history with tool execution: {tool_data['tool']}")

        _logger.info("⏳ Waiting for TTS...")
        # Wait for TTS queue to empty (all sentences spoken)
        max_wait = 30.0
        wait_start = time.time()
        while not conn.tts_queue.empty() and (time.time() - wait_start) < max_wait:
            await asyncio.sleep(0.1)
        _logger.info("✅ TTS completed")

        t_end = time.time()
        _logger.info("âœ… TOTAL PROCESSING TIME: %.1fms",
                     (t_end - t_start) * 1000)

    except Exception as e:
        # ✨ FIX: Log errors instead of silently ignoring them!
        _logger.error(f"❌ ERROR in process_streaming_transcript: {e}")
        import traceback
        _logger.error(traceback.format_exc())
        
        # ✨ Still try to save what we have to conversation history
        if 'text' in locals() and 'response_buffer' in locals() and response_buffer:
            try:
                conn.conversation_history.append({
                    "user": text,
                    "assistant": f"[Error: {str(e)[:100]}] {response_buffer[:200]}",
                    "timestamp": time.time()
                })
                _logger.info(f"✅ Saved partial response to history despite error")
            except:
                pass
    finally:
        conn.is_responding = False
        if conn.interrupt_requested:
            conn.interrupt_requested = False
