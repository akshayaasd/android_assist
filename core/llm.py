import logging
import time
import requests
import json
from typing import Generator, List, Dict
import utils.config as config

logger = logging.getLogger(__name__)

class LLMClient:
    """Handles communications with LLM providers (Groq, Ollama, or Mock fallback)."""
    
    def __init__(self):
        self.provider = config.LLM_PROVIDER.lower()
        logger.info("Initializing LLM client with provider: %s", self.provider)
        
        # Check Groq setup
        if self.provider == "groq":
            if not config.GROQ_API_KEY:
                logger.warning("Groq API key is missing! Falling back to 'mock' provider.")
                self.provider = "mock"
            else:
                try:
                    from groq import Groq
                    self.groq_client = Groq(api_key=config.GROQ_API_KEY)
                    logger.info("Groq client initialized successfully.")
                except ImportError:
                    logger.error("groq package not installed! Falling back to 'mock' provider.")
                    self.provider = "mock"
        
        # Check Ollama setup
        if self.provider == "ollama":
            # Test connectivity to Ollama
            try:
                response = requests.get(config.OLLAMA_API_BASE, timeout=2)
                if response.status_code == 200:
                    logger.info("Local Ollama server detected.")
                else:
                    logger.warning("Ollama server returned status %d. Falling back to mock.", response.status_code)
                    self.provider = "mock"
            except requests.RequestException:
                logger.warning("Ollama server not running or unreachable at %s. Falling back to mock.", config.OLLAMA_API_BASE)
                self.provider = "mock"

    def get_response_stream(self, system_prompt: str, conversation_history: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Generate streaming tokens from the chosen LLM provider.
        
        :param system_prompt: The instruction for the AI assistant behavior.
        :param conversation_history: List of dictionary messages in format {'role': 'user'/'assistant', 'content': 'text'}.
        :return: Generator yielding text chunks as they arrive.
        """
        # Build messages payload
        messages = [{"role": "system", "content": system_prompt}] + conversation_history
        
        if self.provider == "groq":
            yield from self._stream_groq(messages)
        elif self.provider == "ollama":
            yield from self._stream_ollama(messages)
        else:
            yield from self._stream_mock(conversation_history[-1]["content"] if conversation_history else "")

    def _stream_groq(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        try:
            logger.info("Sending request to Groq using model %s...", config.GROQ_MODEL)
            completion = self.groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=512
            )
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.error("Error communicating with Groq: %s. Falling back to mock response.", e)
            yield from self._stream_mock(messages[-1]["content"])

    def _stream_ollama(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        try:
            logger.info("Sending request to Ollama at %s using model %s...", config.OLLAMA_API_BASE, config.OLLAMA_MODEL)
            payload = {
                "model": config.OLLAMA_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.7
                }
            }
            response = requests.post(
                f"{config.OLLAMA_API_BASE}/api/chat",
                json=payload,
                stream=True,
                timeout=10
            )
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
        except Exception as e:
            logger.error("Error communicating with Ollama: %s. Falling back to mock response.", e)
            yield from self._stream_mock(messages[-1]["content"])

    def _stream_mock(self, user_input: str) -> Generator[str, None, None]:
        """Provides a simple keyword-based mock agent when offline or keys are missing."""
        user_input_lower = user_input.lower().strip()
        
        # Rule-based conversational responses with action tags
        if "close" in user_input_lower or "quit" in user_input_lower:
            # Extract application name
            keyword = "close" if "close" in user_input_lower else "quit"
            parts = user_input_lower.split(keyword, 1)
            app_name = parts[1].strip().replace(".", "").replace("?", "").replace("!", "").replace("it", "").strip()
            if app_name:
                app_title = app_name.title()
                response = f'[ACTION: close_app("{app_title}")] Closing {app_title} now.'
            else:
                response = "Which application would you like me to close?"
        elif "open" in user_input_lower:
            # Extract application name
            parts = user_input_lower.split("open", 1)
            app_name = parts[1].strip().replace(".", "").replace("?", "").replace("!", "")
            if app_name:
                app_title = app_name.title()
                response = f'[ACTION: open_app("{app_title}")] Opening {app_title} now.'
            else:
                response = "Which application would you like me to open?"
        elif "call" in user_input_lower:
            # Extract contact or phone number
            parts = user_input_lower.split("call", 1)
            contact = parts[1].strip().replace(".", "").replace("?", "").replace("!", "")
            if contact:
                contact_title = contact.title()
                response = f'[ACTION: place_call("{contact_title}")] Initiating a call to {contact_title} now.'
            else:
                response = "Who would you like me to call?"
        elif any(greet in user_input_lower for greet in ["hello", "hi", "hey"]):
            response = "Hello there! I am your local voice assistant Luna. I can help you open apps like WhatsApp or Teams, place FaceTime calls, or chat. How can I help you today?"
        elif "how are you" in user_input_lower:
            response = "I'm doing great, thank you! Since I run completely locally on your Mac, I'm feeling lightweight and fast."
        elif "name" in user_input_lower:
            response = "My name is Luna. I'm a modular AI voice assistant."
        elif "time" in user_input_lower:
            current_time = time.strftime("%I:%M %p")
            response = f"Sure! The current local time is {current_time}."
        elif "weather" in user_input_lower:
            response = "I can't check the live weather since I'm running locally without live web integrations, but you can check your window or configure an API key to help me check!"
        elif any(bye in user_input_lower for bye in ["bye", "goodbye", "exit"]):
            response = "Goodbye! It was nice chatting with you. Talk to you soon!"
        else:
            response = f"You said: '{user_input}'. If you want to launch an application, say 'open WhatsApp' or 'open Teams'. To place a call, say 'call 12345'."
            
        # Stream the mock response token by token to simulate realistic latency
        words = response.split(" ")
        for word in words:
            yield word + " "
            time.sleep(0.08)  # Simulate conversational speed
