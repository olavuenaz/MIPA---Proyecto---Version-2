import json
import time
import csv
import logging
import requests
import os
from datetime import datetime
from typing import List, Dict, Any

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and Configuration
BASE_URL = "http://127.0.0.1:8000"
USER_PROFILES_FILE = "data_memory/user_profiles.json"
TEST_SET_FILE = "test_set.jsonl"
OUTPUT_CSV = "experiment_results.csv"

EXPECTED_ARCHETYPE_MAP = {
    "CRISIS_ALERTA": "PROTOCOL_SAFETY",
    "SOPORTE_EMOCIONAL": "THE_PROTECTOR",
    "CONSULTA_INFORMATIVA": "THE_SAGE",
    "DIALOGO_SOCIAL": "THE_JESTER",
}

def load_user_profiles(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Loads user profiles containing OCEAN traits from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_test_set(filepath: str) -> List[Dict[str, Any]]:
    """Loads the evaluation stimuli from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def send_chat(user_id: str, text: str) -> Dict[str, Any]:
    """Sends a chat request to the MIPA API and returns the response payload."""
    resp = requests.post(f"{BASE_URL}/chat", json={"user_id": user_id, "text": text})
    resp.raise_for_status()
    return resp.json()

def reset_user_state(user_id: str) -> None:
    """Resets the user's temporal context and crisis history in the API."""
    resp = requests.post(f"{BASE_URL}/reset_user_state", json={"user_id": user_id})
    resp.raise_for_status()

def run_experiment(reset_between_stimuli: bool = True) -> None:
    """
    Executes the batch evaluation experiment across all loaded user profiles 
    and stimuli, logging the cognitive traces and performance metrics to a CSV.
    
    Args:
        reset_between_stimuli (bool): If True, clears the system's short-term memory 
                                      between each request to prevent context leakage.
    """
    profiles = load_user_profiles(USER_PROFILES_FILE)
    test_items = load_test_set(TEST_SET_FILE)
    
    logger.info("Loaded %d profiles: %s", len(profiles), list(profiles.keys()))
    logger.info("Loaded %d test stimuli.", len(test_items))
    logger.info("Reset state between stimuli: %s", reset_between_stimuli)
    
    fieldnames = [
        "id", "user_id", "timestamp", "input_text",
        "expected_intent", "expected_archetype",
        "coherence", "emotion", "archetype_detected",
        "mges_intervened", "mges_reason",
        "archetype_change", "latency_ms",
        "score_protector", "score_sage", "score_jester", "centroid",
        "crisis_score"
    ]
    
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        global_id = 1
        total_items = len(profiles) * len(test_items)
        processed = 0
        
        start_time = time.time()
        
        for user_id, profile_data in profiles.items():
            ocean_traits = profile_data.get("ocean_traits", {})
            
            # Inject profile into the system before running stimuli
            requests.post(f"{BASE_URL}/update_profile", json={
                "user_id": user_id,
                "ocean_traits": ocean_traits
            }).raise_for_status()
            
            previous_archetype = None
            
            for item in test_items:
                text = item["text"]
                expected_intent = item["expected_intent"]
                expected_archetype = EXPECTED_ARCHETYPE_MAP.get(expected_intent, "UNKNOWN")
                
                if reset_between_stimuli:
                    reset_user_state(user_id)
                
                req_start_time = time.time()
                try:
                    response = send_chat(user_id, text)
                except requests.exceptions.RequestException as e:
                    logger.error("API Request failed for user %s: %s", user_id, str(e))
                    continue
                    
                total_ms = (time.time() - req_start_time) * 1000
                
                trace = response.get("traza_cognitiva", {})
                decision = trace.get("decision", {})
                perception = trace.get("percepcion", {})
                
                archetype_detected = trace.get("final_archetype", "UNKNOWN")
                emotion = perception.get("emotion_label", "Neutral")
                selected_archetype = decision.get("selected_archetype", "UNKNOWN")
                mges_intervened = decision.get("mges_intervened", False)
                mges_reason = decision.get("mges_reason", "")
                crisis_score = perception.get("crisis_score", 0.0)
                
                if previous_archetype is None:
                    archetype_change = "NO"
                else:
                    archetype_change = "YES" if selected_archetype != previous_archetype else "NO"
                
                previous_archetype = selected_archetype
                coherence = 1 if selected_archetype == expected_archetype else 0
                
                scores = decision.get("scores", {})
                
                row = {
                    "id": global_id,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "input_text": text,
                    "expected_intent": expected_intent,
                    "expected_archetype": expected_archetype,
                    "coherence": coherence,
                    "emotion": emotion,
                    "archetype_detected": archetype_detected,
                    "mges_intervened": mges_intervened,
                    "mges_reason": mges_reason,
                    "archetype_change": archetype_change,
                    "latency_ms": round(total_ms, 2),
                    "score_protector": round(scores.get("protector", 0.0), 4),
                    "score_sage": round(scores.get("sage", 0.0), 4),
                    "score_jester": round(scores.get("jester", 0.0), 4),
                    "centroid": round(decision.get("centroid", 0.0), 4),
                    "crisis_score": round(crisis_score, 4)
                }
                
                writer.writerow(row)
                
                global_id += 1
                processed += 1
                
                if processed % 50 == 0:
                    logger.info("Processed %d/%d stimuli...", processed, total_items)
        
        elapsed = time.time() - start_time
        logger.info("Experiment completed successfully in %.2f seconds.", elapsed)
        logger.info("Results saved to %s", OUTPUT_CSV)

if __name__ == "__main__":
    run_experiment(reset_between_stimuli=True)