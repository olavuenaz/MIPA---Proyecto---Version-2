import re
import uuid
import logging
from datetime import datetime
from typing import Dict, Any

# Initialize logging
logger = logging.getLogger(__name__)

class InputModule:
    """
    Input Module (MPAI).
    
    Responsible for normalizing input text, generating unique session IDs, 
    and attaching UTC timestamps for downstream processing and auditing.
    """
    
    def __init__(self):
        """Initializes the InputModule."""
        logger.info("Input module initialized successfully.")

    def sanitize_text(self, raw_text: str) -> str:
        """
        Sanitizes the raw input text by removing potentially dangerous characters 
        (e.g., HTML/Scripting tags) and stripping trailing whitespaces.

        Args:
            raw_text (str): The raw text input from the user.

        Returns:
            str: The cleaned and sanitized text string.
        """
        if not raw_text:
            return ""
        
        clean_text = re.sub(r'[<>{}\[\]\\]', '', raw_text.strip())
        return clean_text

    def process_input(self, user_id: str, raw_text: str) -> Dict[str, Any]:
        """
        Processes the raw user input, sanitizes it, and constructs the standard 
        data payload with metadata (session ID, timestamp, and user ID).

        Args:
            user_id (str): The identifier of the user making the request.
            raw_text (str): The raw text input provided by the user.

        Returns:
            Dict[str, Any]: A dictionary containing 'meta' and 'payload' data.
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"
        clean_text = self.sanitize_text(raw_text)
        
        return {
            "meta": {
                "session_id": session_id,
                "user_id": user_id,
                "timestamp": timestamp
            },
            "payload": {
                "raw_text": raw_text,
                "clean_text": clean_text
            }
        }