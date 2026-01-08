# webhooks.py
import httpx
from datetime import datetime as dt

async def send_webhook(url: str, event: str, payload: dict):
    async with httpx.AsyncClient(timeout=5.0) as client:
        await client.post(url, json={
            "event": event,
            "payload": payload,
            "timestamp": dt.utcnow().isoformat()
        })

async def send_webhook_and_get_response(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=payload)
        return response.json()
