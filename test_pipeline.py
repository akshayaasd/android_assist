import os
import time
import numpy as np
from scipy.io.wavfile import write
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_pipeline")

from core.asr import SpeechToText
from core.llm import LLMClient
from core.tts import TextToSpeech
import utils.config as config

def test_llm():
    logger.info("--- Testing LLM Client ---")
    client = LLMClient()
    history = [{"role": "user", "content": "Say hello in exactly 5 words."}]
    stream = client.get_response_stream("You are a helpful assistant.", history)
    
    response = []
    for token in stream:
        response.append(token)
        print(token, end="", flush=True)
    print()
    
    full_text = "".join(response).strip()
    assert len(full_text) > 0, "LLM returned empty response"
    logger.info("LLM Client test passed successfully.\n")

def test_tts():
    logger.info("--- Testing TTS Engine ---")
    tts = TextToSpeech()
    text = "Hello! This is a test of the local Text to Speech voice engine."
    
    chunks = []
    sample_rate = 22050
    for chunk_array, sr in tts.synthesize_stream(text):
        chunks.append(chunk_array)
        sample_rate = sr
        
    assert len(chunks) > 0, "TTS yielded no audio chunks"
    
    # Combine chunks and save to verification file
    full_audio = np.concatenate(chunks)
    output_path = os.path.join(config.BASE_DIR, "test_tts_output.wav")
    
    # Convert float array to int16 for WAV file compliance
    int_audio = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)
    write(output_path, sample_rate, int_audio)
    
    assert os.path.exists(output_path), "Failed to save TTS output WAV"
    logger.info("TTS test passed. Saved output file to: %s\n", output_path)

def test_asr():
    logger.info("--- Testing ASR Engine ---")
    asr = SpeechToText()
    
    # Generate a 1-second synthetic sine wave at 440 Hz, sample rate 16000
    duration = 1.0
    t = np.linspace(0, duration, int(config.SAMPLE_RATE * duration), endpoint=False)
    # Silent audio or low frequency wave
    audio_data = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Transcribe the audio
    text = asr.transcribe(audio_data)
    logger.info("ASR test complete. Transcription of synthetic sound: '%s'", text)
    # Silence or pure sine wave shouldn't crash transcription
    logger.info("ASR test passed.\n")

def test_tools():
    logger.info("--- Testing Tool Registry & Action Parser ---")
    from core.tools import parse_action_tag, execute_action
    
    # Test case 1: Open app command parsing
    test_str = '[ACTION: open_app("WhatsApp")] Opening WhatsApp now.'
    action_name, action_arg, rest_text = parse_action_tag(test_str)
    assert action_name == "open_app", f"Expected open_app, got {action_name}"
    assert action_arg == "WhatsApp", f"Expected WhatsApp, got {action_arg}"
    assert rest_text == "Opening WhatsApp now.", f"Expected rest text, got {rest_text}"
    
    # Test case 2: Place call command parsing
    test_str_call = '[ACTION: place_call("12345")] Calling now.'
    action_name_call, action_arg_call, rest_text_call = parse_action_tag(test_str_call)
    assert action_name_call == "place_call", f"Expected place_call, got {action_name_call}"
    assert action_arg_call == "12345", f"Expected 12345, got {action_arg_call}"
    assert rest_text_call == "Calling now.", f"Expected rest text, got {rest_text_call}"
    
    # Test case 2.5: Close app command parsing
    test_str_close = '[ACTION: close_app("Chrome")] Closing Chrome now.'
    action_name_close, action_arg_close, rest_text_close = parse_action_tag(test_str_close)
    assert action_name_close == "close_app", f"Expected close_app, got {action_name_close}"
    assert action_arg_close == "Chrome", f"Expected Chrome, got {action_arg_close}"
    assert rest_text_close == "Closing Chrome now.", f"Expected rest text, got {rest_text_close}"
    
    # Test case 3: Non-action string
    normal_str = "Hello world."
    name, arg, cleaned = parse_action_tag(normal_str)
    assert name is None
    assert arg is None
    assert cleaned == "Hello world."
    
    # Test case 4: Dry-run execution
    res = execute_action("open_app", "invalid_app_name_test_xyz")
    assert "couldn't find or open" in res or "Generic launch failed" in res
    
    res_close = execute_action("close_app", "invalid_app_name_test_xyz")
    assert "Successfully closed native" in res_close or "Failed to close" in res_close
    
    logger.info("Tool Registry and Action Parser tests passed successfully.\n")

if __name__ == "__main__":
    logger.info("Starting voice assistant pipeline verification test...\n")
    try:
        test_llm()
        test_tts()
        test_asr()
        test_tools()
        logger.info("ALL PIPELINE TESTS PASSED!")
    except Exception as e:
        logger.error("Pipeline test failed: %s", e)
        exit(1)
