import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QUESTIONS_DIR = DATA_DIR / "questions"
INDEX_DIR = ROOT / "indexes"

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SEED = 42
