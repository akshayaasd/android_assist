================================================================================
                     LUNA VOICE ASSISTANT ARCHITECTURE & OPTIMIZATIONS
================================================================================

This document details the complete modular architecture of the Luna local voice
assistant and the precise optimizations implemented to enhance ASR (speech-to-text)
accuracy, eliminate TTS (text-to-speech) choppiness, and extend cross-platform
tool execution.

--------------------------------------------------------------------------------
1. PIPELINE ARCHITECTURE OVERVIEW
--------------------------------------------------------------------------------

Luna is structured around a real-time, event-driven voice pipeline running locally
on macOS:

    [ Microphone Input ] ---> Audio Manager (RMS VAD & Noise Calibration)
                                     |
                                     v
    [ numpy audio float32 ] -> ASR Engine (faster-whisper base.en)
                                     |
                                     v
    [ Transcribed text ] ----> LLM Orchestrator (Groq / llama-3.3-70b-versatile)
                                     |
                                     v
        +----------------------------+----------------------------+
        |                                                         |
        v                                                         v
    [ Streamed text response ]                               [ Action Tags ]
        |                                                         |
        v (Lookbehind splitting)                                  v (Background thread)
    [ TTS Engine (Piper-TTS) ]                               Tool Registry
        |                                                     (App Launcher / Closer)
        v                                                         |
    [ Audio chunks ]                                              v
        |                                            - AppleScript (Mac Apps)
        v                                            - URL schemes (whatsapp://)
    [ Speaker Playback ]                             - ADB Shell (Android S24)
                                                     - Google Chrome website open

---

Detailed Pipeline Sequence:
1. AudioManager (core/audio.py): Continuous recording stream calibrates noise
   thresholds and uses Root Mean Square (RMS) energy to detect speech start and
   silence transitions.
2. SpeechToText (core/asr.py): Transcribes audio to string.
3. LLMClient (core/llm.py): Generates streaming text. If the user requests an action,
   it prepends `[ACTION: action_name("arg")]` to the stream.
4. Orchestrator (main.py): Intercepts the Action tags and executes them in a daemon
   thread (so the agent can speak concurrently without waiting for the app to load).
5. TextToSpeech (core/tts.py): Synthesizes streamed sentences chunk-by-chunk and plays
   them back immediately to minimize latency.

--------------------------------------------------------------------------------
2. ASR (SPEECH-TO-TEXT) OPTIMIZATIONS
--------------------------------------------------------------------------------

* Problem:
  Originally, Whisper was configured with a beam size of 1 (greedy search) to
  optimize speed, but it suffered from high phonetic error rates (e.g. transcribing
  "Chrome" as "clone", "Google Chrome" as "clone", or hallucinating initials).

* Solutions Implemented:
  1. Beam Search Tuning: Increased `beam_size` from `1` to `5` in core/asr.py.
     This permits the model to explore multiple possible paths and select the most
     probable phrase sequence.
  2. Deterministic Temperature: Set `temperature=0.0` to force deterministic decoding
     based on the highest log probabilities.
  3. Prompt Injection: Provided an `initial_prompt` keyword guideline to the model:
     "Open WhatsApp, Open Microsoft Teams, Open Google Chrome, Open Spotify, place a call."
     This guides the search path toward correctly spelling applications and websites.

--------------------------------------------------------------------------------
3. TTS (TEXT-TO-SPEECH) OPTIMIZATIONS
--------------------------------------------------------------------------------

* Problem:
  To reduce latency, Luna processes text in real-time streaming mode. The original
  sentence splitter checked for the presence of periods `.` to detect boundaries.
  However, this caused initials and abbreviations (like `B.H.R.O.M.B.`, `U.S.A.`,
  or decimal numbers like `1.5`) to trigger premature sentence splits, resulting
  in highly fragmented, choppy voice outputs and missing syllables.

* Solutions Implemented:
  - Regex Lookbehind Chunker: Integrated a smart lookbehind regex in main.py:
    `re.split(r'(?<=[.!?])\s+|\n+', text_buffer)`
  - Preservation Mechanism: This ensures that punctuation only triggers a split
    if it is immediately followed by a space or a newline. Acronyms or periods inside
    middle-word abbreviation strings are preserved as single continuous blocks.

--------------------------------------------------------------------------------
4. TOOL REGISTRY & WINDOWS CLOSING CAPABILITIES
--------------------------------------------------------------------------------

* Problem:
  - Chrome website launches were opening only in the default system browser (which might
    be Safari or Firefox) rather than Chrome.
  - The assistant could not close applications when asked, instead simulating speech
    without performing the task.

* Solutions Implemented:
  1. Google Chrome Target: Added a parser in core/tools.py that checks if an app
     name maps to a known domain (like GitHub, GitLab, Google, YouTube) or a URL.
     It runs a subprocess `open -a "Google Chrome" URL` to open websites specifically
     in Chrome, falling back gracefully to the default system browser.
  2. App Closing Command (`close_app`):
     - macOS: Uses AppleScript gracefully:
       `osascript -e 'quit app "Google Chrome"'`
       This allows apps to close cleanly rather than forcing a raw `pkill`.
     - Android: If Samsung S24 is plugged in, stops package activity via ADB shell:
       `adb shell am force-stop <package>`
  3. System Prompt Refinement: Added guidelines to SYSTEM_PROMPT to force website
     requests (like "Open GitHub") to output action tags.

--------------------------------------------------------------------------------
5. MOBILE SCHEMES & ADB COMMAND DETAILS
--------------------------------------------------------------------------------

- Android Package Launching:
  When your Samsung S24 is connected via USB debugging (ADB), the assistant triggers
  applications by package name using the launcher category monkey:
  `adb shell monkey -p <package_name> -c android.intent.category.LAUNCHER 1`
  This starts the default launcher activity without needing class configurations.

- Phone Calls / Android Dialer:
  - Android: Launches the phone dialer with prefilled numbers:
    `adb shell am start -a android.intent.action.DIAL -d tel:<number>`
  - macOS: Falls back to FaceTime or tel URI schemes:
    `facetime://<number>` or `tel://<number>`

- Mobile Scheme Fallback Registry:
  If an app is not running locally on macOS, the registry executes standard mobile scheme URIs
  which macOS triggers to launch applications or falls back to web versions:
  - Teams: `msteams://` (Fallback: `https://teams.microsoft.com`)
  - WhatsApp: `whatsapp://` (Fallback: `https://web.whatsapp.com`)
  - Spotify: `spotify://` (Fallback: `https://open.spotify.com`)
