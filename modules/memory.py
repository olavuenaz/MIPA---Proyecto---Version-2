import json
import os
import re
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# Initialize logging
logger = logging.getLogger(__name__)

class MemoryModule:
    """
    Memory and Induction Module (Hybrid RAG).
    
    Manages short-term memory (recent conversational turns), long-term memory 
    (vector-based semantic search), and persistent user profiles (OCEAN traits).
    """
    def __init__(self):
        """Initializes the MemoryModule, paths, and the embedding model."""
        self.profiles_path = "./data_memory/user_profiles.json"
        self.history_dir = "./data_memory/history"
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        os.makedirs(self.history_dir, exist_ok=True)
        self._init_db()
        
        logger.info("Memory module (Hybrid RAG) initialized successfully.")

    def _init_db(self) -> None:
        """Ensures the user profiles JSON database exists."""
        if not os.path.exists(self.profiles_path):
            with open(self.profiles_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _safe_load_json(self, path: str) -> Dict[str, Any]:
        """
        Safely loads a JSON file.
        
        Args:
            path (str): The file path to load.
            
        Returns:
            Dict[str, Any]: Parsed JSON data or an empty dictionary if an error occurs.
        """
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading JSON at %s: %s", path, str(e))
            return {}

    def _load_history_lines(self, user_id: str) -> List[str]:
        """
        Loads the raw conversation history lines for a specific user.
        
        Args:
            user_id (str): The unique identifier for the user.
            
        Returns:
            List[str]: A list of non-empty conversation lines.
        """
        safe_id = "".join([c for c in user_id if c.isalnum() or c in ('-', '_')])
        filepath = os.path.join(self.history_dir, f"{safe_id}.txt")
        
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if len(line.strip()) > 5]
        except Exception as e:
            logger.error("Error loading history for user %s: %s", safe_id, str(e))
            return []

    def _sanitize_text(self, text: str) -> str:
        """
        Cleans text by removing internal tags, brackets, and speaker prefixes.
        
        Args:
            text (str): The raw text to sanitize.
            
        Returns:
            str: The sanitized text.
        """
        if not text:
            return ""
        text = re.sub(r'<<.*?>>', '', text, flags=re.DOTALL)
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'^(User:|Usuario:|IA:|Assistant:|System:)', '', text, flags=re.IGNORECASE)
        return text.strip().strip('"').strip("'")

    def build_context(self, user_id: str, current_input_text: str) -> Dict[str, Any]:
        """
        Constructs the RAG context by combining short-term history and relevant 
        long-term memories using semantic similarity.
        
        Args:
            user_id (str): The active user's ID.
            current_input_text (str): The current prompt to find relevant past context for.
            
        Returns:
            Dict[str, Any]: A dictionary containing the user's OCEAN profile and RAG context list.
        """
        profiles = self._safe_load_json(self.profiles_path)
        user_profile = profiles.get(user_id, {
            "nombre": "Usuario",
            "ocean_traits": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5}
        })

        full_history = self._load_history_lines(user_id)
        short_term_window = 4
        
        recent_history = full_history[-short_term_window:] if full_history else []
        searchable_history = full_history[:-short_term_window] if len(full_history) > short_term_window else []

        relevant_memories = []
        if searchable_history and len(current_input_text.split()) > 2:
            try:
                query_vector = self.embedder.encode(current_input_text).reshape(1, -1)
                history_vectors = self.embedder.encode(searchable_history)
                similarities = cosine_similarity(query_vector, history_vectors)[0]
                
                top_k_indices = np.argsort(similarities)[-3:][::-1]
                for idx in top_k_indices:
                    if similarities[idx] > 0.60:
                        clean_mem = self._sanitize_text(searchable_history[idx])
                        relevant_memories.append(f"- {clean_mem}")
            except Exception as e:
                logger.error("RAG semantic search error: %s", str(e))

        rag_context = []
        if relevant_memories:
            rag_context.append("DATOS RECUPERADOS DEL PASADO:")
            rag_context.extend(relevant_memories)
            
        if recent_history:
            rag_context.append("CONVERSACIÓN ACTUAL:")
            rag_context.extend(recent_history)
        else:
            rag_context.append("(Inicio de conversación)")

        return {
            "user_profile_ocean": user_profile,
            "rag_context": rag_context
        }

    def save_interaction(self, user_id: str, input_text: str, response_text: str) -> None:
        """
        Saves the input and response into the user's long-term history file.
        
        Args:
            user_id (str): The active user's ID.
            input_text (str): The user's input message.
            response_text (str): The generated response.
        """
        safe_id = "".join([c for c in user_id if c.isalnum() or c in ('-', '_')])
        filepath = os.path.join(self.history_dir, f"{safe_id}.txt")
        
        clean_input = self._sanitize_text(input_text)
        clean_response = self._sanitize_text(response_text)
        
        if not clean_input or not clean_response:
            return
            
        if "Error" in clean_response or "FAIL" in clean_response:
            return
            
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"Usuario: {clean_input}\n")
                f.write(f"IA: {clean_response}\n")
        except Exception as e:
            logger.error("Failed to save interaction for %s: %s", safe_id, str(e))

    def _save_profile(self, user_id: str, traits: Dict[str, float]) -> None:
        """
        Saves or updates a user's OCEAN traits in the profiles database.
        
        Args:
            user_id (str): The active user's ID.
            traits (Dict[str, float]): The updated OCEAN traits dictionary.
        """
        profiles = self._safe_load_json(self.profiles_path)
        
        if user_id not in profiles:
            profiles[user_id] = {
                "nombre": "Usuario",
                "ocean_traits": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5}
            }
            
        profiles[user_id]["ocean_traits"] = traits
        
        try:
            with open(self.profiles_path, "w", encoding="utf-8") as f:
                json.dump(profiles, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error("Failed to save profile for %s: %s", user_id, str(e))