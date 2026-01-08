import uuid
import re
from typing import Dict, Optional, List, Tuple
from datetime import datetime as dt

import httpx
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session

from models import (
    Agent, Conversation, WebhookConfig, PhoneNumber, KnowledgeBase, AgentTool,
    AgentCreate, AgentUpdate, OutboundCallRequest, WebhookCreate, ToolCreate,
    get_db, SessionLocal
)
from config import logger, WEBHOOK_EVENTS
from webhooks import send_webhook, send_webhook_and_get_response

# ----------------------------
# Helper Functions
# ----------------------------
def generate_agent_id() -> str:
    """Generate unique agent ID"""
    return f"agent_{uuid.uuid4().hex[:16]}"


def generate_conversation_id() -> str:
    """Generate unique conversation ID"""
    return f"conv_{uuid.uuid4().hex[:16]}"


async def send_webhook(webhook_url: str, event: str, data: Dict):
    """Send webhook notification to registered webhook URLs (fire-and-forget)"""
    try:
        if not webhook_url.startswith(("http://", "https://")):
            logger.error(f"❌ Invalid webhook URL: {webhook_url} - must start with http:// or https://")
            return False
        
        async with httpx.AsyncClient() as client:
            payload = {
                "event": event,
                "timestamp": dt.utcnow().isoformat(),
                "data": data
            }
            response = await client.post(webhook_url, json=payload, timeout=10)
            logger.info(f"📤 Webhook sent: {event} to {webhook_url} (status: {response.status_code})")
            return response.status_code == 200
    except Exception as e:
        logger.error(f"❌ Webhook failed: {event} to {webhook_url} - {e}")
        return False


async def send_webhook_and_get_response(webhook_url: str, event: str, data: Dict) -> Optional[Dict]:
    """Send webhook and wait for response data (for inbound call configuration)"""
    try:
        if not webhook_url.startswith(("http://", "https://")):
            logger.error(f"❌ Invalid webhook URL: {webhook_url} - must start with http:// or https://")
            return None
        
        async with httpx.AsyncClient() as client:
            payload = {
                "event": event,
                "timestamp": dt.utcnow().isoformat(),
                "data": data
            }
            response = await client.post(webhook_url, json=payload, timeout=10)
            logger.info(f"📤 Webhook sent: {event} to {webhook_url} (status: {response.status_code})")
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"📥 Webhook response received: {list(response_data.keys())}")
                return response_data
            else:
                logger.warning(f"⚠️ Webhook returned non-200 status: {response.status_code}")
                return None
    except Exception as e:
        logger.error(f"❌ Webhook failed: {event} to {webhook_url} - {e}")
        return None


# ----------------------------
# Text Processing Functions
# ----------------------------
def clean_markdown_for_tts(text: str) -> str:
    """Remove markdown formatting before TTS to prevent reading symbols aloud"""
    # Remove bold
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # Remove italic
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # Remove strikethrough
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Remove links
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bullet points
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered lists
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def detect_intent(text: str) -> str:
    """✨ HUMAN-LIKE: Only detect GOODBYE - let LLM handle everything else naturally"""
    t = text.lower().strip()
    if any(x in t for x in ["bye", "goodbye", "end the call", "that's all", "talk later"]):
        return "GOODBYE"
    return "QUESTION"


def detect_confirmation_response(text: str) -> Optional[str]:
    """Detect if user is confirming or rejecting a pending action."""
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
    """Parse LLM response for tool calls."""
    tool_pattern = r'\[TOOL:([^\]]+)\]'
    confirm_pattern = r'\[CONFIRM_TOOL:([^\]]+)\]'

    tool_data = None
    confirm_matches = re.findall(confirm_pattern, text)
    
    if confirm_matches:
        tool_parts = confirm_matches[0].split(':')
        tool_name = tool_parts[0].strip()
        if tool_name == "transfer":
            department = tool_parts[1].strip() if len(tool_parts) > 1 else "sales"
            valid_departments = ["sales", "support", "technical"]
            if department in valid_departments:
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
                department = tool_parts[1].strip() if len(tool_parts) > 1 else "sales"
                valid_departments = ["sales", "support", "technical"]
                if department in valid_departments:
                    tool_data = {
                        "tool": "transfer_call",
                        "params": {"department": department},
                        "requires_confirmation": False
                    }
            else:
                tool_params = {}
                if len(tool_parts) > 1:
                    remaining_parts = tool_parts[1:]
                    for idx, part in enumerate(remaining_parts):
                        tool_params[f"param{idx+1}"] = part.strip()
                tool_data = {
                    "tool": tool_name,
                    "params": tool_params,
                    "requires_confirmation": False
                }

    clean_text = re.sub(tool_pattern, '', text)
    clean_text = re.sub(confirm_pattern, '', clean_text)
    clean_text = clean_text.strip()

    return clean_text, tool_data


# ----------------------------
# Tool Functions
# ----------------------------
async def call_webhook_tool(webhook_url: str, tool_name: str, parameters: dict, call_context: dict) -> dict:
    """Call an external webhook tool and return the response"""
    try:
        payload = {
            "tool_name": tool_name,
            "parameters": parameters,
            "call_context": call_context,
            "timestamp": dt.utcnow().isoformat()
        }
        
        logger.info(f"🔧 Calling webhook tool: {tool_name} at {webhook_url}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(webhook_url, json=payload, headers={"Content-Type": "application/json"})
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Webhook tool response: {result}")
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "response": result.get("response", result),
                    "data": result.get("data", {}),
                    "message": result.get("message", "")
                }
            else:
                logger.error(f"❌ Webhook returned status {response.status_code}")
                return {
                    "success": False,
                    "error": f"Webhook returned status {response.status_code}",
                    "tool_name": tool_name
                }
                
    except asyncio.TimeoutError:
        logger.error(f"❌ Webhook timeout for {tool_name}")
        return {
            "success": False,
            "error": "Tool request timed out",
            "tool_name": tool_name
        }
    except Exception as e:
        logger.error(f"❌ Webhook error for {tool_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool_name": tool_name
        }


async def end_call_tool(call_sid: str, reason: str = "user_goodbye") -> dict:
    """End the active call"""
    logger.info(f"📞 END_CALL: call_sid={call_sid}, reason={reason}")

    try:
        await asyncio.sleep(1.5)
        # Note: twilio_client would need to be imported from main.py or config.py
        # This is a placeholder - actual implementation would use twilio_client
        return {
            "success": True,
            "message": f"Call ended: {reason}",
            "call_sid": call_sid
        }
    except Exception as e:
        logger.error(f"❌ Failed to end call: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def transfer_call_tool(call_sid: str, department: str = "sales") -> dict:
    """Transfer call to human agent - executes AFTER message is spoken"""
    logger.info(f"🔄 TRANSFER_CALL: call_sid={call_sid}, dept={department}")
    
    # This is a placeholder - actual implementation would use environment variables
    DEPARTMENT_NUMBERS = {
        "sales": os.getenv("SALES_PHONE_NUMBER", "+918107061392"),
        "support": os.getenv("SUPPORT_PHONE_NUMBER", "+918107061392"),
        "technical": os.getenv("TECH_PHONE_NUMBER", "+918107061392"),
    }

    try:
        transfer_number = DEPARTMENT_NUMBERS.get(department, DEPARTMENT_NUMBERS["sales"])
        conn = manager.get(call_sid)
        if not conn:
            return {"success": False, "error": "Connection not found"}

        logger.info("⏳ Waiting for transfer message to be spoken...")
        await asyncio.sleep(3.0)
        conn.interrupt_requested = True

        while not conn.tts_queue.empty():
            try:
                conn.tts_queue.get_nowait()
                conn.tts_queue.task_done()
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

        # Note: Actual Twilio call update would go here
        logger.info(f"✅ Transfer completed to {department} ({transfer_number})")
        return {
            "success": True,
            "transfer_to": transfer_number,
            "department": department,
            "message": f"Transferred to {department}"
        }

    except Exception as e:
        logger.error(f"❌ Transfer failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def execute_detected_tool(call_sid: str, tool_data: dict) -> dict:
    """Execute a tool that was detected from LLM response"""
    tool_name = tool_data["tool"]
    params = tool_data.get("params", {})

    logger.info(f"🔧 Executing LLM-requested tool: {tool_name} with params: {params}")

    # Built-in tools
    if tool_name == "end_call":
        result = await end_call_tool(call_sid, **params)
    elif tool_name == "transfer_call":
        result = await transfer_call_tool(call_sid, **params)
    else:
        conn = manager.get(call_sid)
        if not conn or not conn.agent_id:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        db = SessionLocal()
        try:
            tool = db.query(AgentTool).filter(
                AgentTool.agent_id == conn.agent_id,
                AgentTool.tool_name == tool_name,
                AgentTool.is_active == True
            ).first()
            
            if not tool or not tool.webhook_url:
                return {"success": False, "error": f"Unknown or inactive tool: {tool_name}"}
            
            call_context = {
                "call_sid": call_sid,
                "agent_id": conn.agent_id,
                "conversation_id": conn.conversation_id,
                "phone_number": None,
                "dynamic_variables": conn.dynamic_variables or {}
            }
            
            result = await call_webhook_tool(
                webhook_url=tool.webhook_url,
                tool_name=tool_name,
                parameters=params,
                call_context=call_context
            )
            
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
# Conversation Management
# ----------------------------
async def save_conversation_transcript(call_sid: str, conn):
    """Save conversation transcript to database"""
    logger.info(f"💾 save_conversation_transcript called for {call_sid}")
    
    if not conn:
        logger.warning(f"⚠️ No connection found for {call_sid} - cannot save transcript")
        return
    
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == call_sid
        ).first()
        
        if conversation:
            transcript_lines = []
            for entry in conn.conversation_history:
                user_text = entry.get('user', '')
                assistant_text = entry.get('assistant', '')
                transcript_lines.append(f"User: {user_text}")
                transcript_lines.append(f"Assistant: {assistant_text}")
            
            conversation.transcript = "\n".join(transcript_lines) if transcript_lines else "[No conversation - call ended early]"
            conversation.status = "completed"
            conversation.ended_at = dt.utcnow()
            
            if conversation.started_at:
                duration = (conversation.ended_at - conversation.started_at).total_seconds()
                conversation.duration_secs = int(duration)
            
            db.commit()
            logger.info(f"✅ Saved transcript for {call_sid}")
        else:
            logger.warning(f"⚠️ Conversation record not found in DB for {call_sid}")
    except Exception as e:
        logger.error(f"❌ Failed to save transcript: {e}")
        db.rollback()
    finally:
        db.close()


async def handle_call_end(call_sid: str, reason: str):
    """Handle call ending - save data and send webhooks"""
    conn = manager.get(call_sid)
    
    if conn:
        await save_conversation_transcript(call_sid, conn)
    
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == call_sid
        ).first()
        
        if conversation:
            conversation.ended_reason = reason
            conversation.status = "completed"
            if not conversation.ended_at:
                conversation.ended_at = dt.utcnow()
            
            if conversation.started_at and not conversation.duration_secs:
                duration = (conversation.ended_at - conversation.started_at).total_seconds()
                conversation.duration_secs = int(duration)
            
            db.commit()
            
            call_direction = "outbound"
            if conversation.call_metadata and isinstance(conversation.call_metadata, dict):
                call_direction = conversation.call_metadata.get("direction", "outbound")
            
            webhooks = db.query(WebhookConfig).filter(
                WebhookConfig.is_active == True
            ).all()
            
            for webhook in webhooks:
                should_send = False
                if webhook.agent_id is None:
                    should_send = True
                elif conversation.agent_id and webhook.agent_id == conversation.agent_id:
                    should_send = True
                
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
            
            logger.info(f"✅ Call ended: {call_sid} - reason: {reason} - duration: {conversation.duration_secs}s")
        else:
            logger.warning(f"⚠️ Conversation not found for call end: {call_sid}")
    except Exception as e:
        logger.error(f"❌ Failed to handle call end: {e}")
    finally:
        db.close()