import logging
import warnings
import numpy as np

# Suppress spurious NumPy 2.x floating-point matmul warnings from faster-whisper on macOS CPU
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

from faster_whisper import WhisperModel
import utils.config as config

logger = logging.getLogger(__name__)

class SpeechToText:
    """Wrapper around faster-whisper for speech transcription."""
    
    def __init__(self):
        logger.info("Initializing Speech-to-Text engine using Whisper model: %s", config.WHISPER_MODEL)
        try:
            self.model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE
            )
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise e

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe a 16kHz mono float32 numpy array.
        
        :param audio_data: 1D numpy float32 array of audio samples in [-1.0, 1.0].
        :return: Transcribed text.
        """
        if audio_data is None or len(audio_data) == 0:
            return ""
            
        logger.info("Transcribing audio chunk of size %d...", len(audio_data))
        try:
            # We use beam_size=5 for highly accurate transcription.
            # temperature=0.0 ensures deterministic decoding.
            # initial_prompt provides helpful contextual keywords to reduce spelling errors.
            # vad_filter=True helps filter out background noise/breathing.
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=5,
                temperature=0.0,
                initial_prompt="Open WhatsApp, Open Microsoft Teams, Open Google Chrome, Open Spotify, place a call.",
                vad_filter=True,
                language="en"
            )
            
            # segments is a generator, so we iterate over it to extract text
            text_segments = [segment.text for segment in segments]
            full_text = " ".join(text_segments).strip()
            
            logger.info("Transcription complete. Language detected: %s (prob: %.2f). Text: %s",
                        info.language, info.language_probability, full_text)
            return full_text
        except Exception as e:
            logger.error("Error during ASR transcription: %s", e)
            return ""
