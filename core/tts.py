import os
import logging
import requests
from typing import Generator, Tuple
import numpy as np
from piper import PiperVoice
import utils.config as config

logger = logging.getLogger(__name__)

class TextToSpeech:
    """Wrapper around Piper TTS engine with automatic model downloading."""
    
    def __init__(self):
        self.model_path = os.path.join(config.MODELS_DIR, config.VOICE_ONNX_FILENAME)
        self.config_path = os.path.join(config.MODELS_DIR, config.VOICE_CONFIG_FILENAME)
        
        # Ensure data/models directory exists
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        # Download models if not present
        self._ensure_models_exist()
        
        # Load the voice model
        logger.info("Loading Piper TTS voice model from: %s", self.model_path)
        try:
            self.voice = PiperVoice.load(
                model_path=self.model_path,
                config_path=self.config_path,
                use_cuda=False
            )
            logger.info("Piper TTS engine initialized successfully.")
        except Exception as e:
            logger.error("Failed to load Piper TTS model: %s", e)
            raise e

    def _ensure_models_exist(self):
        """Check if model and config files exist locally, download if missing."""
        if not os.path.exists(self.model_path):
            logger.info("Voice model file not found locally. Downloading %s...", config.VOICE_MODEL_NAME)
            self._download_file(config.PIPER_MODEL_URL, self.model_path)
            
        if not os.path.exists(self.config_path):
            logger.info("Voice config file not found locally. Downloading config...")
            self._download_file(config.PIPER_CONFIG_URL, self.config_path)

    def _download_file(self, url: str, dest_path: str):
        """Downloads a file with basic logging of progress."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            logger.info("Starting download: %s -> %s (Size: %.2f MB)", url, dest_path, total_size / (1024 * 1024))
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            # Log every 25% or so
                            if int(percent) % 25 == 0 and int(percent - (len(chunk)/total_size)*100) % 25 != 0:
                                logger.info("Download progress: %.1f%%", percent)
                                
            logger.info("Successfully downloaded file to %s", dest_path)
        except Exception as e:
            logger.error("Failed to download file from %s to %s: %s", url, dest_path, e)
            # Remove partial file if it exists
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise e

    def synthesize_stream(self, text: str) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Synthesize text into raw float32 audio chunks.
        
        :param text: Text string to convert to speech.
        :return: Generator yielding tuples of (audio_numpy_array, sample_rate).
        """
        if not text or not text.strip():
            return
            
        logger.info("Synthesizing speech for: '%s'", text)
        try:
            # Piper synthesize returns a generator of AudioChunk objects
            for chunk in self.voice.synthesize(text):
                yield chunk.audio_float_array, chunk.sample_rate
        except Exception as e:
            logger.error("Error during TTS synthesis: %s", e)
