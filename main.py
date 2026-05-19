import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# Set third-party logs to warning to keep console clean
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("ctranslate2").setLevel(logging.WARNING)
logging.getLogger("onnxruntime").setLevel(logging.WARNING)

logger = logging.getLogger("voice_assistant")

from core.audio import AudioManager
from core.asr import SpeechToText
from core.llm import LLMClient
from core.tts import TextToSpeech
import utils.config as config

SYSTEM_PROMPT = (
    "You are a helpful, friendly, and concise local voice assistant named Luna. "
    "If the user asks you to open an application, open a website/URL (like GitHub, GitLab, Google, etc.), "
    "close/quit/exit an application, or make a call, you MUST prepend an action command to the beginning of your "
    "response in the exact format:\n"
    "- [ACTION: open_app(\"AppNameOrWebsite\")]\n"
    "- [ACTION: close_app(\"AppName\")]\n"
    "- [ACTION: place_call(\"PhoneOrContact\")]\n"
    "and then reply with a friendly spoken announcement. Do not mention the action tag in your spoken response. "
    "Otherwise, do not output any action tags.\n"
    "Your responses should be conversational, clear, and brief (ideally 1 to 3 sentences) "
    "as they will be spoken aloud to the user. Avoid using markdown lists, markdown bolding, "
    "or complex symbols in your text output."
)

def main():
    print("=" * 60)
    print("      Welcome to your Modular Local AI Voice Assistant")
    print("=" * 60)
    print(f"LLM Provider: {config.LLM_PROVIDER.upper()}")
    print(f"ASR Model:    {config.WHISPER_MODEL}")
    print(f"TTS Voice:    {config.VOICE_MODEL_NAME}")
    print("=" * 60)
    
    try:
        # Initialize components
        print("\n[Initializing voice assistant engines...]")
        audio_manager = AudioManager()
        asr = SpeechToText()
        llm = LLMClient()
        tts = TextToSpeech()
        
        # Calibrate background noise floor
        audio_manager.calibrate(duration=1.0)
        
        # Welcoming sequence
        welcome_text = "Hello! I am ready to assist you. How can I help you today?"
        print(f"\nAssistant: {welcome_text}")
        for audio_chunk, sample_rate in tts.synthesize_stream(welcome_text):
            audio_manager.play(audio_chunk, sample_rate)
        
        conversation_history = []
        
        while True:
            # Wait until current speech playback finishes before listening
            audio_manager.wait_for_playback()
            
            # Record user voice input
            audio_data = audio_manager.record_until_silence(timeout=6.0)
            
            if audio_data is None or len(audio_data) == 0:
                # No speech detected (timeout)
                continue
                
            # Transcribe audio
            user_text = asr.transcribe(audio_data)
            
            if not user_text:
                print("[Didn't catch that, please speak again.]")
                continue
                
            print(f"\nYou: {user_text}")
            
            # Exit conditions
            if any(exit_cmd in user_text.lower() for exit_cmd in ["goodbye", "exit assistant", "quit assistant"]):
                farewell = "Goodbye! Have a great day!"
                print(f"Assistant: {farewell}")
                for audio_chunk, sample_rate in tts.synthesize_stream(farewell):
                    audio_manager.play(audio_chunk, sample_rate)
                audio_manager.wait_for_playback()
                break
                
            # Append user utterance to conversation history
            conversation_history.append({"role": "user", "content": user_text})
            
            # Generate assistant stream
            print("Assistant: ", end="", flush=True)
            response_stream = llm.get_response_stream(SYSTEM_PROMPT, conversation_history)
            
            text_buffer = ""
            full_response_parts = []
            
            action_checked = False
            is_buffering_action = False
            
            import re
            
            for token in response_stream:
                full_response_parts.append(token)
                
                # Check if response begins with an action tag
                if not action_checked:
                    temp_text = "".join(full_response_parts).strip()
                    if temp_text.startswith("["):
                        is_buffering_action = True
                        if "]" in temp_text:
                            action_checked = True
                            is_buffering_action = False
                            
                            # Parse and execute action
                            accumulated_str = "".join(full_response_parts)
                            from core.tools import parse_action_tag, execute_action
                            action_name, action_arg, rest_text = parse_action_tag(accumulated_str)
                            
                            if action_name and action_arg:
                                logger.info("Detected action tag: %s(%s)", action_name, action_arg)
                                import threading
                                threading.Thread(
                                    target=execute_action,
                                    args=(action_name, action_arg),
                                    daemon=True
                                ).start()
                                
                            if rest_text:
                                print(rest_text, end="", flush=True)
                                text_buffer += rest_text
                        continue
                    else:
                        action_checked = True
                        is_buffering_action = False
                        # Flush the accumulated tokens to print & text buffer
                        accumulated = "".join(full_response_parts)
                        print(accumulated, end="", flush=True)
                        text_buffer += accumulated
                        continue
                        
                if is_buffering_action:
                    continue
                
                # Regular stream printing & buffering
                print(token, end="", flush=True)
                text_buffer += token
                
                # Check for sentence boundaries: punctuation followed by space or a newline
                # This prevents splitting on initials (like B.H.R.O.M.B. or U.S.A.)
                parts = re.split(r'(?<=[.!?])\s+|\n+', text_buffer)
                if len(parts) > 1:
                    for part in parts[:-1]:
                        sentence = part.strip()
                        if len(sentence) > 1:
                            for audio_chunk, sample_rate in tts.synthesize_stream(sentence):
                                audio_manager.play(audio_chunk, sample_rate)
                    text_buffer = parts[-1]
            
            # Process remaining text in buffer
            if text_buffer:
                sentence = text_buffer.strip()
                if len(sentence) > 1:
                    for audio_chunk, sample_rate in tts.synthesize_stream(sentence):
                        audio_manager.play(audio_chunk, sample_rate)
                        
            print() # Print newline after response ends
            
            full_assistant_response = "".join(full_response_parts).strip()
            
            # Save assistant response to conversation history
            conversation_history.append({"role": "assistant", "content": full_assistant_response})
            
            # Keep history under control (last 10 turns)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
                
    except KeyboardInterrupt:
        print("\nExiting voice assistant...")
    except Exception as e:
        logger.error("An error occurred in the assistant loop: %s", e)
    finally:
        # Cleanup
        if 'audio_manager' in locals():
            audio_manager.stop()
        print("\nAssistant stopped. Goodbye!")

if __name__ == "__main__":
    main()
