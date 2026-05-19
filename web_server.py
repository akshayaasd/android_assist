import asyncio
import logging
import threading
import re
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn

from core.audio import AudioManager
from core.asr import SpeechToText
from core.llm import LLMClient
from core.tts import TextToSpeech
from core.tools import execute_action, parse_action_tag
import utils.config as config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("web_server")

app = FastAPI()

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

# Websocket and status tracking state
active_websockets = set()
main_loop = None
current_status = "initializing"
chat_history = []
actions_log = []

async def broadcast_ws(message: dict):
    if not active_websockets:
        return
    payload = json.dumps(message)
    sockets = list(active_websockets)
    for ws in sockets:
        try:
            await ws.send_text(payload)
        except Exception:
            active_websockets.discard(ws)

def send_update_to_web(message: dict):
    # Keep track of history in case page is refreshed
    if message["type"] == "chat":
        chat_history.append({"sender": message["sender"], "text": message["text"]})
    elif message["type"] == "chat_start":
        chat_history.append({"sender": "luna", "text": ""})
    elif message["type"] == "token":
        if chat_history and chat_history[-1]["sender"] == "luna":
            chat_history[-1]["text"] += message["text"]
    elif message["type"] == "action":
        # Log actions
        actions_log.append(message)
        if len(actions_log) > 20:
            actions_log.pop(0)

    if active_websockets and main_loop:
        main_loop.call_soon_threadsafe(
            lambda: asyncio.create_task(broadcast_ws(message))
        )

def set_status(status: str):
    global current_status
    current_status = status
    send_update_to_web({"type": "status", "data": status})

def run_action_background(action_name: str, action_arg: str):
    """Runs a tool action asynchronously and sends results back to the UI."""
    send_update_to_web({"type": "action", "name": action_name, "arg": action_arg, "status": "running"})
    
    # Check if we are closing the assistant itself
    if action_name == "close_app" and action_arg.lower() in ["luna", "amy", "assistant", "voice assistant", "self", "web_server", "web server"]:
        import os
        logger.info("Self-shutdown action triggered via close_app action tag.")
        send_update_to_web({"type": "action", "name": action_name, "arg": action_arg, "status": "success", "result": "Shutting down local voice assistant server..."})
        # Wait for the farewell speech stream to finish playing
        time.sleep(5.0)
        os._exit(0)
        
    try:
        res = execute_action(action_name, action_arg)
        send_update_to_web({"type": "action", "name": action_name, "arg": action_arg, "status": "success", "result": res})
    except Exception as e:
        logger.error("Failed to run action: %s", e)
        send_update_to_web({"type": "action", "name": action_name, "arg": action_arg, "status": "failed", "result": str(e)})

def run_assistant_loop():
    logger.info("Starting background voice assistant thread...")
    try:
        set_status("initializing")
        audio_manager = AudioManager()
        asr = SpeechToText()
        llm = LLMClient()
        tts = TextToSpeech()
        
        # Calibrate background noise floor
        set_status("calibrating")
        audio_manager.calibrate(duration=1.0)
        
        # Welcoming sequence
        set_status("speaking")
        welcome_text = "Hello! I am Luna. I am ready to assist you. How can I help you today?"
        send_update_to_web({"type": "chat", "sender": "luna", "text": welcome_text})
        for audio_chunk, sample_rate in tts.synthesize_stream(welcome_text):
            audio_manager.play(audio_chunk, sample_rate)
            
        conversation_history = []
        set_status("idle")
        
        while True:
            audio_manager.wait_for_playback()
            set_status("listening")
            
            # Record user voice input
            audio_data = audio_manager.record_until_silence(timeout=6.0)
            if audio_data is None or len(audio_data) == 0:
                continue
                
            set_status("processing")
            user_text = asr.transcribe(audio_data)
            if not user_text:
                continue
                
            # Send user chat message to browser
            send_update_to_web({"type": "chat", "sender": "user", "text": user_text})
            
            # Exit conditions
            if any(exit_cmd in user_text.lower() for exit_cmd in ["goodbye", "exit assistant", "quit assistant", "close you", "quit you", "close assistant", "quit assistant", "stop assistant", "shut down", "exit luna", "close luna", "quit luna"]):
                set_status("speaking")
                farewell = "Goodbye! Have a great day!"
                send_update_to_web({"type": "chat", "sender": "luna", "text": farewell})
                for audio_chunk, sample_rate in tts.synthesize_stream(farewell):
                    audio_manager.play(audio_chunk, sample_rate)
                audio_manager.wait_for_playback()
                import os
                logger.info("Exiting voice assistant via voice exit command...")
                os._exit(0)
                
            conversation_history.append({"role": "user", "content": user_text})
            
            # Generate assistant stream response
            response_stream = llm.get_response_stream(SYSTEM_PROMPT, conversation_history)
            
            text_buffer = ""
            full_response_parts = []
            
            action_checked = False
            is_buffering_action = False
            
            # Initialize empty text bubble in frontend
            send_update_to_web({"type": "chat_start"})
            
            for token in response_stream:
                full_response_parts.append(token)
                
                # Check for action tags
                if not action_checked:
                    temp_text = "".join(full_response_parts).strip()
                    if temp_text.startswith("["):
                        is_buffering_action = True
                        if "]" in temp_text:
                            action_checked = True
                            is_buffering_action = False
                            
                            # Parse and execute action
                            accumulated_str = "".join(full_response_parts)
                            action_name, action_arg, rest_text = parse_action_tag(accumulated_str)
                            
                            if action_name and action_arg:
                                threading.Thread(
                                    target=run_action_background,
                                    args=(action_name, action_arg),
                                    daemon=True
                                ).start()
                                
                            if rest_text:
                                send_update_to_web({"type": "token", "text": rest_text})
                                text_buffer += rest_text
                        continue
                    else:
                        action_checked = True
                        is_buffering_action = False
                        accumulated = "".join(full_response_parts)
                        send_update_to_web({"type": "token", "text": accumulated})
                        text_buffer += accumulated
                        continue
                        
                if is_buffering_action:
                    continue
                
                # Regular stream token delivery to UI
                send_update_to_web({"type": "token", "text": token})
                text_buffer += token
                
                # Check for sentence boundaries to play to TTS immediately
                parts = re.split(r'(?<=[.!?])\s+|\n+', text_buffer)
                if len(parts) > 1:
                    for part in parts[:-1]:
                        sentence = part.strip()
                        if len(sentence) > 1:
                            set_status("speaking")
                            for audio_chunk, sample_rate in tts.synthesize_stream(sentence):
                                audio_manager.play(audio_chunk, sample_rate)
                    text_buffer = parts[-1]
                    set_status("processing")
            
            # Process remaining text in buffer
            if text_buffer:
                sentence = text_buffer.strip()
                if len(sentence) > 1:
                    set_status("speaking")
                    for audio_chunk, sample_rate in tts.synthesize_stream(sentence):
                        audio_manager.play(audio_chunk, sample_rate)
                        
            set_status("idle")
            
            full_assistant_response = "".join(full_response_parts).strip()
            _, _, cleaned_response = parse_action_tag(full_assistant_response)
            conversation_history.append({"role": "assistant", "content": cleaned_response})
            
            # Keep history context within limits
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
                
    except Exception as e:
        logger.error("Error in background voice loop: %s", e)
        set_status("error")
    finally:
        if 'audio_manager' in locals():
            audio_manager.stop()
        logger.info("Voice assistant loop terminated.")

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_event_loop()
    # Start the assistant pipeline background loop
    threading.Thread(target=run_assistant_loop, daemon=True).start()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    # Send current state and history on connection
    await websocket.send_text(json.dumps({"type": "status", "data": current_status}))
    await websocket.send_text(json.dumps({"type": "history", "data": chat_history}))
    if actions_log:
        for action_msg in actions_log:
            await websocket.send_text(json.dumps(action_msg))
            
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
    except Exception:
        active_websockets.discard(websocket)

# Mount the static folder at root
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
