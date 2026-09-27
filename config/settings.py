import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
PROMPTS_DIR = BASE_DIR / "prompts"
LOGS_DIR = BASE_DIR / "logs"

# Ensure essential directories exist
for path in [DATA_DIR, VECTORSTORE_DIR, PROCESSED_DIR, METADATA_DIR, PROMPTS_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# LLM Keys & Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini")  # 'gemini' or 'openai'
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
NORMALIZE_EMBEDDINGS = True

# Chunking Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval Settings
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
