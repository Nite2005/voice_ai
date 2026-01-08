import os
import logging
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

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
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mixtral:8x7b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "384"))
TOP_K = int(os.getenv("TOP_K", "3"))

# 🎯 SMART INTERRUPT SETTINGS
INTERRUPT_ENABLED = os.getenv("INTERRUPT_ENABLED", "true").lower() == "true"
INTERRUPT_MIN_ENERGY = int(os.getenv("INTERRUPT_MIN_ENERGY", "1000"))
INTERRUPT_DEBOUNCE_MS = int(os.getenv("INTERRUPT_DEBOUNCE_MS", "1000"))
INTERRUPT_BASELINE_FACTOR = float(os.getenv("INTERRUPT_BASELINE_FACTOR", "3.5"))
INTERRUPT_MIN_SPEECH_MS = int(os.getenv("INTERRUPT_MIN_SPEECH_MS", "300"))
INTERRUPT_REQUIRE_TEXT = os.getenv("INTERRUPT_REQUIRE_TEXT", "false").lower() == "true"

# ✅ SILENCE DETECTION
SILENCE_THRESHOLD_SEC = float(os.getenv("SILENCE_THRESHOLD_SEC", "0.8"))
UTTERANCE_END_MS = int(SILENCE_THRESHOLD_SEC * 1000)

REQUIRE_ENV = [TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, PUBLIC_URL, DEEPGRAM_API_KEY]
if not all(REQUIRE_ENV):
    raise RuntimeError("Missing required env: TWILIO_*, PUBLIC_URL, DEEPGRAM_API_KEY")

# JWT Secret for signed URLs
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")

# API Key Authentication
API_KEY_HEADER_NAME = "xi-api-key"
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

# ----------------------------
# Logging
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
LOG_FILE = os.getenv("LOG_FILE", "server.log")

def setup_logger():
    """Setup and return logger instance"""
    logger = logging.getLogger("new")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=2)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    
    return logger

logger = setup_logger()

def public_ws_host() -> str:
    """Extract WebSocket host from public URL"""
    host = PUBLIC_URL.replace("https://", "").replace("http://", "").rstrip("/")
    return host