import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01  # Default RMS threshold, can be overridden by VAD calibration
SILENCE_DURATION = 1.5   # Seconds of silence before stopping recording
VAD_MIN_SPEECH_DURATION = 0.3  # Minimum speech duration in seconds

# Whisper ASR Settings
WHISPER_MODEL = "base.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# LLM Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")  # groq, ollama, or mock
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# TTS Settings
VOICE_MODEL_NAME = "en_US-amy-low"
VOICE_ONNX_FILENAME = "en_US-amy-low.onnx"
VOICE_CONFIG_FILENAME = "en_US-amy-low.onnx.json"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")

# Piper Model Download URLs
PIPER_MODEL_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/{VOICE_ONNX_FILENAME}"
PIPER_CONFIG_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/{VOICE_CONFIG_FILENAME}"
