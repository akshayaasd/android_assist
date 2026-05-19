import queue
import threading
import time
import logging
import numpy as np
import sounddevice as sd
import utils.config as config

logger = logging.getLogger(__name__)

class AudioManager:
    """Manages audio recording with VAD and asynchronous queue-based playback."""
    
    def __init__(self):
        self.playback_queue = queue.Queue()
        self.stop_playback = False
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        self.calibrated_threshold = config.SILENCE_THRESHOLD
        logger.info("Audio Manager initialized.")

    def _playback_worker(self):
        """Background worker thread to play synthesized audio chunks sequentially."""
        while not self.stop_playback:
            try:
                # Poll queue
                item = self.playback_queue.get(timeout=0.2)
                if item is None:
                    break
                audio_data, sample_rate = item
                
                # Play audio using sounddevice
                sd.play(audio_data, sample_rate)
                sd.wait()  # Block until current chunk is finished playing
                self.playback_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Error in audio playback worker: %s", e)

    def play(self, audio_data: np.ndarray, sample_rate: int):
        """Add an audio chunk to the playback queue."""
        if audio_data is not None and len(audio_data) > 0:
            self.playback_queue.put((audio_data, sample_rate))

    def wait_for_playback(self):
        """Blocks until all queued audio chunks are finished playing."""
        self.playback_queue.join()

    def stop(self):
        """Stop the audio playback worker thread."""
        self.stop_playback = True
        self.playback_queue.put(None)
        self.playback_thread.join(timeout=2.0)

    def calibrate(self, duration: float = 1.0) -> float:
        """
        Record ambient noise to calibrate the RMS silence threshold.
        
        :param duration: Seconds to record for calibration.
        :return: Calibrated RMS threshold.
        """
        logger.info("Calibrating background noise floor... Please remain silent.")
        print("\n[Calibrating microphone, please remain quiet for 1 second...]")
        
        chunk_size = 1024
        frames = []
        
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype='float32'
        ) as stream:
            num_chunks = int(config.SAMPLE_RATE * duration / chunk_size)
            for _ in range(num_chunks):
                data, _ = stream.read(chunk_size)
                frames.append(data.flatten())
                
        all_audio = np.concatenate(frames)
        # Compute RMS of segments
        segment_rms = []
        for i in range(0, len(all_audio), chunk_size):
            segment = all_audio[i:i+chunk_size]
            if len(segment) > 0:
                rms = np.sqrt(np.mean(segment**2))
                segment_rms.append(rms)
                
        avg_rms = np.mean(segment_rms) if segment_rms else 0.01
        # Set threshold to 3.0x noise floor or a minimum threshold (to avoid absolute quiet issues)
        self.calibrated_threshold = max(config.SILENCE_THRESHOLD, avg_rms * 3.0)
        logger.info("Calibration finished. Ambient RMS: %.4f, Selected Threshold: %.4f", avg_rms, self.calibrated_threshold)
        print(f"[Calibration done. Threshold: {self.calibrated_threshold:.4f}]")
        return self.calibrated_threshold

    def record_until_silence(self, timeout: float = 6.0) -> np.ndarray:
        """
        Record audio from the microphone.
        Starts recording immediately, waits for user to speak, and stops after silence is detected.
        
        :param timeout: Maximum seconds to wait for speech to start before aborting.
        :return: 1D numpy float32 array representing the recorded voice.
        """
        chunk_size = 1024  # ~64ms chunks at 16kHz
        chunk_duration = chunk_size / config.SAMPLE_RATE
        
        frames = []
        is_speaking = False
        silence_timer = 0.0
        start_time = time.time()
        
        logger.info("Starting recording stream...")
        print("\nListening... (Start speaking)", end="", flush=True)
        
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype='float32'
        ) as stream:
            while True:
                data, _ = stream.read(chunk_size)
                audio_chunk = data.flatten()
                frames.append(audio_chunk)
                
                # Compute RMS volume of this chunk
                rms = np.sqrt(np.mean(audio_chunk**2))
                
                if not is_speaking:
                    # Check if speech has started
                    if rms > self.calibrated_threshold:
                        is_speaking = True
                        print("\n[Speech detected...]", end="", flush=True)
                        logger.info("Speech detected. Transitioning to speaking state.")
                    else:
                        # Timeout if user doesn't say anything
                        if time.time() - start_time > timeout:
                            print("\n[Silence timeout. No speech detected.]")
                            logger.info("Recording timed out without speech start.")
                            return np.array([], dtype=np.float32)
                else:
                    # If speaking, check for silence
                    if rms < self.calibrated_threshold:
                        silence_timer += chunk_duration
                        if silence_timer >= config.SILENCE_DURATION:
                            print("\n[Silence detected. Processing...]")
                            logger.info("Silence duration limit reached. Stopping recording.")
                            break
                    else:
                        silence_timer = 0.0
                        # Print a small dot to indicate voice activity in console
                        print(".", end="", flush=True)
                        
        # Combine all frames
        audio_data = np.concatenate(frames)
        
        # Trim leading quiet frames to optimize whisper transcription
        # We find where RMS exceeds a noise threshold and start slightly before it
        step = 512
        start_index = 0
        for idx in range(0, len(audio_data), step):
            segment = audio_data[idx:idx+step]
            if len(segment) > 0 and np.sqrt(np.mean(segment**2)) > self.calibrated_threshold * 0.7:
                start_index = max(0, idx - step * 2)  # Keep a bit of prefix
                break
                
        return audio_data[start_index:]
