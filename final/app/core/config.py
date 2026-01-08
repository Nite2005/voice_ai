import os
import torch

# ==============================
# DEVICE / GPU
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# INTERRUPT CONFIG
# ==============================

INTERRUPT_ENABLED = True
INTERRUPT_MIN_ENERGY = 0.001
INTERRUPT_BASELINE_FACTOR = 1.5
INTERRUPT_MIN_SPEECH_MS = 200
INTERRUPT_REQUIRE_TEXT = True

# ==============================
# SILENCE / UTTERANCE
# ==============================

SILENCE_THRESHOLD_SEC = 1.0
UTTERANCE_END_MS = 1000

# ==============================
# RAG / CHUNKING
# ==============================

CHUNK_SIZE = 800

# ==============================
# FILES / URLS
# ==============================

DATA_FILE = os.getenv("DATA_FILE", "data.txt")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:9001")
