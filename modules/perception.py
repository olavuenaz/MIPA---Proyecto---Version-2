import json
import os
import logging
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple, Optional
from pysentimiento import create_analyzer

# Initialize logging
logger = logging.getLogger(__name__)

class PerceptionModule:
    """
    Perception Module (MMCE).
    
    Responsible for natural language understanding, intent classification via semantic 
    similarity, emotion detection using PySentimiento, and early crisis detection.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the PerceptionModule, loading models, thresholds, and keyword anchors.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        self.crisis_threshold = self.config.get("thresholds", {}).get("crisis_confidence", 0.75)
        self.intent_threshold = self.config.get("thresholds", {}).get("intent_threshold", 0.20)
        
        self._load_intent_anchors()
        
        self.crisis_keywords = [
            "suicidio", "matarme", "quitarme la vida", "me voy a matar", 
            "sobredosis de", "cortarme las venas", "ahorcarme"
        ]
        
        self.tech_keywords = [
            "qué es", "necesito saber", "dame un ejemplo",  
            "qué plantea", "por qué", "explica"
        ]
        
        self.social_keywords = [
            "qué frío", "comer", "galleta", "techo"
        ]
        
        self.sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
        logger.info("Perception module initialized (enhanced crisis detection active).")

    def _load_intent_anchors(self) -> None:
        """Loads or creates the anchor vectors for intent classification."""
        anchors_file = "intent_anchors.json"
        
        if not os.path.exists(anchors_file):
            example_anchors = {
                "CRISIS_ALERTA": ["quiero morir", "suicidio", "acabar con todo", "no puedo más"],
                "SOPORTE_EMOCIONAL": ["me siento triste", "estoy frustrado", "necesito ayuda"],
                "CONSULTA_INFORMATIVA": ["qué es", "cómo funciona", "explícame"],
                "DIALOGO_SOCIAL": ["hola", "buenos días", "qué tal"]
            }
            with open(anchors_file, 'w', encoding='utf-8') as f:
                json.dump(example_anchors, f, indent=2, ensure_ascii=False)
        
        with open(anchors_file, 'r', encoding='utf-8') as f:
            raw_anchors = json.load(f)
        
        self.intent_anchors = {}
        for intent, phrases in raw_anchors.items():
            self.intent_anchors[intent] = self.model.encode(phrases).mean(axis=0)

    def _detect_emotion_pysentimiento(self, text: str) -> str:
        """
        Classifies the emotion of the input text using PySentimiento and refinement rules.
        
        Args:
            text (str): The raw text to analyze.
            
        Returns:
            str: The mapped emotion label (e.g., 'Alegría', 'Tristeza', 'Ira').
        """
        result = self.sentiment_analyzer.predict(text)
        label = result.output
        
        if label == 'POS':
            return "Alegría"
        elif label == 'NEG':
            text_lower = text.lower()
            if any(w in text_lower for w in ["triste", "angustia", "deprimido"]):
                return "Tristeza"
            if any(w in text_lower for w in ["odio", "rabia", "furia"]):
                return "Ira"
            if any(w in text_lower for w in ["ansiedad", "miedo", "pánico"]):
                return "Ansiedad"
            return "Tristeza" 
        else:
            return "Neutral"

    def _detect_crisis(self, text_lower: str, user_vec: np.ndarray) -> Tuple[bool, Optional[str], float, int]:
        """
        Unified crisis detection combining lexical matching and cosine similarity.
        
        Args:
            text_lower (str): The input text in lowercase.
            user_vec (np.ndarray): The encoded embedding vector of the input text.
            
        Returns:
            Tuple[bool, Optional[str], float, int]: Crisis flag, detection method, 
                                                    cosine score, and keyword match count.
        """
        matches = [kw for kw in self.crisis_keywords if kw in text_lower]
        keyword_count = len(matches)
        score = 0.0
        crisis_detected = False
        method = None

        if "CRISIS_ALERTA" in self.intent_anchors:
            crisis_anchor = self.intent_anchors["CRISIS_ALERTA"]
            if user_vec.ndim == 1:
                user_vec = user_vec.reshape(1, -1)
            score = float(cosine_similarity(user_vec, crisis_anchor.reshape(1, -1))[0][0])

        if score >= self.crisis_threshold or keyword_count >= 2:
            crisis_detected = True
            method = "cosine" if score >= self.crisis_threshold else "keyword"
        
        return crisis_detected, method, score, keyword_count

    def analyze(self, clean_text: str) -> Dict[str, Any]:
        """
        Analyzes the text and extracts perception metrics including intent, emotion, 
        and potential crisis signals.
        
        Args:
            clean_text (str): The sanitized input text from the user.
            
        Returns:
            Dict[str, Any]: A payload containing all perception data for downstream reasoning.
        """
        if not clean_text:
            return {
                "perception": {
                    "emotion_label": "Neutral",
                    "intent": "UNKNOWN",
                    "confidence": 0.0,
                    "detection_method": "none",
                    "crisis_detected": False,
                    "crisis_method": None,
                    "crisis_score": 0.0,
                    "crisis_keyword_count": 0
                }
            }

        text_lower = clean_text.lower()
        user_vec = self.model.encode([clean_text])[0]
        
        crisis_detected, crisis_method, crisis_score, crisis_kw_count = self._detect_crisis(text_lower, user_vec)

        if crisis_detected:
            best_intent = "CRISIS_ALERTA"
            best_score = 1.0
            detection_method = f"crisis_{crisis_method}"
        else:
            best_intent = "UNKNOWN"
            best_score = 0.0
            detection_method = "cosine_similarity"

            for intent, anchor_vec in self.intent_anchors.items():
                user_vec_2d = user_vec.reshape(1, -1) if user_vec.ndim == 1 else user_vec
                sim = float(cosine_similarity(user_vec_2d, anchor_vec.reshape(1, -1))[0][0])
                
                if sim > best_score:
                    best_score = sim
                    best_intent = intent

            if best_score < self.intent_threshold:
                if any(kw in text_lower for kw in self.tech_keywords):
                    best_intent = "CONSULTA_INFORMATIVA"
                    best_score = max(best_score, 0.65)
                    detection_method = "keyword_fallback"
                elif any(kw in text_lower for kw in self.social_keywords):
                    best_intent = "DIALOGO_SOCIAL"
                    best_score = max(best_score, 0.55)
                    detection_method = "keyword_fallback"

            if best_intent == "UNKNOWN":
                emotion = self._detect_emotion_pysentimiento(clean_text)
                if emotion in ["Tristeza", "Ira", "Ansiedad"]:
                    best_intent = "SOPORTE_EMOCIONAL"
                    best_score = max(best_score, 0.65)
                    detection_method = "emotion_fallback"

        if 'emotion' not in locals():
            emotion = self._detect_emotion_pysentimiento(clean_text)

        return {
            "perception": {
                "emotion_label": emotion,
                "intent": best_intent,
                "confidence": round(best_score, 3),
                "detection_method": detection_method,
                "crisis_detected": crisis_detected,
                "crisis_method": crisis_method,
                "crisis_score": round(crisis_score, 4),
                "crisis_keyword_count": crisis_kw_count,
                "raw_text": clean_text
            }
        }