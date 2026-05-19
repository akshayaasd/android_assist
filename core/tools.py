import subprocess
import webbrowser
import logging
import re
import sys
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Common mappings for macOS native applications and fallback URLs
APP_REGISTRY = {
    "whatsapp": {
        "mac_name": "WhatsApp",
        "mobile_scheme": "whatsapp://",
        "url": "https://web.whatsapp.com"
    },
    "teams": {
        "mac_name": "Microsoft Teams",
        "mobile_scheme": "msteams://",
        "url": "https://teams.microsoft.com"
    },
    "chrome": {
        "mac_name": "Google Chrome",
        "mobile_scheme": "googlechrome://",
        "url": "https://google.com"
    },
    "spotify": {
        "mac_name": "Spotify",
        "mobile_scheme": "spotify://",
        "url": "https://open.spotify.com"
    },
    "notes": {
        "mac_name": "Notes",
        "mobile_scheme": "mobilenotes://",
        "url": None
    },
    "calendar": {
        "mac_name": "Calendar",
        "mobile_scheme": "calshow://",
        "url": None
    },
    "contacts": {
        "mac_name": "Contacts",
        "mobile_scheme": "contacts://",
        "url": None
    },
    "facetime": {
        "mac_name": "FaceTime",
        "mobile_scheme": "facetime://",
        "url": None
    }
}

# Android app package name mappings (tailored for Samsung/Google default apps)
ANDROID_APP_REGISTRY = {
    "whatsapp": "com.whatsapp",
    "teams": "com.microsoft.teams",
    "spotify": "com.spotify.music",
    "chrome": "com.android.chrome",
    "notes": "com.samsung.android.app.notes",       # Samsung Notes
    "calendar": "com.samsung.android.calendar",      # Samsung Calendar
    "contacts": "com.samsung.android.contacts",      # Samsung Contacts
    "dialer": "com.samsung.android.dialer",          # Samsung Dialer
}

def is_android_connected() -> bool:
    """Checks if an Android device is currently reachable via ADB."""
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=1.5)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Look for devices with state 'device' (excluding headers)
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "device":
                        return True
    except Exception:
        pass
    return False

def open_android_app(app_key: str) -> Optional[str]:
    """Attempts to launch an app on the connected Android phone via ADB monkey tool."""
    package_name = ANDROID_APP_REGISTRY.get(app_key)
    if not package_name:
        return None
        
    try:
        logger.info("ADB: Attempting to launch package: %s on Android...", package_name)
        # Use monkey tool to launch target package launcher activity
        cmd = ["adb", "shell", "monkey", "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully opened {app_key} on your Android phone."
    except Exception as e:
        logger.error("ADB: Failed to launch Android package %s: %s", package_name, e)
    return None

def dial_android_number(contact: str) -> Optional[str]:
    """Brings up the dialer with prefilled number on the connected Android phone via ADB."""
    clean_contact = re.sub(r"[^\d+]", "", contact) # keep digits and + sign
    if not clean_contact:
        return None
        
    try:
        logger.info("ADB: Prefilling call to %s on Android...", clean_contact)
        # Start DIAL intent (safer than CALL, doesn't require extra phone app privileges)
        cmd = ["adb", "shell", "am", "start", "-a", "android.intent.action.DIAL", "-d", f"tel:{clean_contact}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return f"Triggered phone dialer for {contact} on your Android phone."
    except Exception as e:
        logger.error("ADB: Failed to trigger Android dialer: %s", e)
    return None

def open_app(app_name: str) -> str:
    """Attempts to open an app. Prioritizes Android if connected, then macOS, then web."""
    app_key = app_name.lower().strip()
    
    # 1. Android Check
    if is_android_connected():
        android_res = open_android_app(app_key)
        if android_res:
            return android_res
            
    # 2. Native macOS application launcher (if on Mac)
    if app_key in APP_REGISTRY:
        reg = APP_REGISTRY[app_key]
        mac_name = reg["mac_name"]
        fallback_url = reg["url"]
        mobile_scheme = reg.get("mobile_scheme")
        
        try:
            logger.info("Attempting to launch native macOS app: %s", mac_name)
            result = subprocess.run(["open", "-a", mac_name], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Successfully opened native app: %s", mac_name)
                return f"Successfully opened native {mac_name} application."
        except Exception as e:
            logger.warning("Could not launch native app %s: %s", mac_name, e)
            
        # 3. Try custom URL scheme which directly triggers native apps on iOS/Android (and macOS if installed)
        if mobile_scheme:
            try:
                logger.info("Attempting custom URL scheme: %s", mobile_scheme)
                import platform
                if platform.system() == "Darwin":
                    result = subprocess.run(["open", mobile_scheme], capture_output=True, text=True)
                    if result.returncode == 0:
                        return f"Successfully opened {mac_name} via custom scheme."
                else:
                    webbrowser.open(mobile_scheme)
                    return f"Triggered {mac_name} mobile scheme: {mobile_scheme}"
            except Exception as e:
                logger.warning("Custom URL scheme launch failed: %s", e)
                
        # 4. Fall back to web URL in default browser
        if fallback_url:
            logger.info("App not found or failed to launch. Opening web version: %s", fallback_url)
            webbrowser.open(fallback_url)
            return f"Opened web version of {mac_name} in your browser."
            
    # Generic launch for other unspecified apps
    try:
        logger.info("Attempting generic launch for app: %s", app_name)
        result = subprocess.run(["open", "-a", app_name], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Opened {app_name}."
    except Exception as e:
        logger.warning("Generic launch failed for %s: %s", app_name, e)
        
    # Check if app_name resembles a URL, domain, or known website keyword
    known_websites = {
        "github": "https://github.com",
        "gitlab": "https://gitlab.com",
        "google": "https://google.com",
        "youtube": "https://youtube.com",
    }
    
    url = None
    app_lower = app_name.lower().strip()
    
    if app_name.startswith("http") or "." in app_name:
        url = app_name if app_name.startswith("http") else f"https://{app_name}"
    elif app_lower in known_websites:
        url = known_websites[app_lower]
        
    if url:
        # Prioritize opening in Google Chrome if requested/available
        try:
            logger.info("Attempting to open URL in Google Chrome: %s", url)
            import platform
            if platform.system() == "Darwin":
                result = subprocess.run(["open", "-a", "Google Chrome", url], capture_output=True, text=True)
                if result.returncode == 0:
                    return f"Successfully opened {url} in Google Chrome."
        except Exception as e:
            logger.warning("Failed to open URL in Google Chrome: %s", e)
            
        # Fallback to default browser
        webbrowser.open(url)
        return f"Opened link: {url}"
        
    if "://" in app_name:
        webbrowser.open(app_name)
        return f"Opened scheme: {app_name}"
        
    return f"Sorry, I couldn't find or open the application {app_name}."

def place_call(contact: str) -> str:
    """Initiates a call. Prioritizes Android if connected, then macOS protocols."""
    # 1. Android Check
    if is_android_connected():
        android_res = dial_android_number(contact)
        if android_res:
            return android_res
            
    clean_contact = contact.replace(" ", "").strip()
    logger.info("Initiating call request to: %s", clean_contact)
    
    # Try FaceTime audio/video call
    try:
        logger.info("Opening FaceTime call dialog...")
        subprocess.run(["open", f"facetime://{clean_contact}"])
        return f"Placing a FaceTime call to {contact}."
    except Exception as e:
        logger.error("Failed to launch FaceTime protocol: %s", e)
        
    # Fallback to tel protocol
    try:
        subprocess.run(["open", f"tel://{clean_contact}"])
        return f"Placing a call to {contact}."
    except Exception as e:
        logger.error("Failed to launch tel protocol: %s", e)
        
    return f"Failed to place a call to {contact}."

def close_app(app_name: str) -> str:
    """Attempts to close/quit an app. Prioritizes Android if connected, then macOS."""
    app_key = app_name.lower().strip()
    
    # 1. Android Check
    if is_android_connected():
        package_name = ANDROID_APP_REGISTRY.get(app_key)
        if package_name:
            try:
                logger.info("ADB: Attempting to force-stop package: %s on Android...", package_name)
                cmd = ["adb", "shell", "am", "force-stop", package_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return f"Successfully closed {app_name} on your Android phone."
            except Exception as e:
                logger.error("ADB: Failed to force-stop package %s: %s", package_name, e)
                
    # 2. macOS Application Graceful Quit using AppleScript
    if app_key in APP_REGISTRY:
        mac_name = APP_REGISTRY[app_key]["mac_name"]
    else:
        mac_name = app_name # Fallback to literal name
        
    try:
        logger.info("Attempting to quit macOS app: %s", mac_name)
        # AppleScript command: quit app "App Name"
        cmd = ["osascript", "-e", f'quit app "{mac_name}"']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Successfully quit native app: %s", mac_name)
            return f"Successfully closed native {mac_name} application."
    except Exception as e:
        logger.warning("Could not quit native app %s: %s", mac_name, e)
        
    return f"Failed to close {app_name}."

def parse_action_tag(response_text: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Parses action tags like [ACTION: open_app("WhatsApp")] from a text string.
    
    :return: A tuple of (action_name, action_arg, cleaned_text_without_action_tag)
    """
    # Regex matching [ACTION: tool_name("arg")] or [ACTION: tool_name('arg')] or [ACTION: tool_name(arg)]
    match = re.search(r'\[ACTION:\s*(\w+)\((.*?)\)\]', response_text)
    if match:
        action_name = match.group(1)
        action_arg = match.group(2).strip(' "\'')
        
        # Remove the tag from the text
        cleaned_text = response_text.replace(match.group(0), "").strip()
        return action_name, action_arg, cleaned_text
        
    return None, None, response_text

def execute_action(action_name: str, action_arg: str) -> str:
    """Executes the specified tool action."""
    logger.info("Executing action: %s with argument: %s", action_name, action_arg)
    if action_name == "open_app":
        return open_app(action_arg)
    elif action_name == "close_app":
        return close_app(action_arg)
    elif action_name == "place_call":
        return place_call(action_arg)
    else:
        return f"Unknown action: {action_name}"
