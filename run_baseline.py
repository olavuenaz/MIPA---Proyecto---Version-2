import json
import time
import csv
import logging
import os
from datetime import datetime
from typing import List

from modules.perception import PerceptionModule

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize perception module (Naive Baseline)
perception = PerceptionModule()

USER_PROFILES_FILE = "data_memory/user_profiles.json"
TEST_SET_FILE = "test_set.jsonl"
OUTPUT_CSV = "baseline_results.csv"

# Direct static mapping (No fuzzy logic, no OCEAN evaluation)
DIRECT_MAPPING = {
    "CRISIS_ALERTA": "PROTOCOL_SAFETY",
    "SOPORTE_EMOCIONAL": "THE_PROTECTOR",
    "CONSULTA_INFORMATIVA": "THE_SAGE",
    "DIALOGO_SOCIAL": "THE_JESTER",
    "UNKNOWN": "THE_PROTECTOR"  # Default fallback
}

def load_user_profiles(filepath: str) -> List[str]:
    """
    Loads the user profile IDs from the JSON database.
    
    Args:
        filepath (str): Path to the user profiles JSON file.
        
    Returns:
        List[str]: A list of user IDs.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    return list(profiles.keys())

def run_baseline_experiment() -> None:
    """
    Executes the naive baseline experiment (Ablation without MIPC or MGES).
    Evaluates the stimuli using only the Perception module and a direct intent-to-archetype mapping,
    deliberately failing to consider OCEAN traits or safety guardrails.
    """
    if not os.path.exists(TEST_SET_FILE):
        logger.error("Test set file not found: %s", TEST_SET_FILE)
        return

    with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f if line.strip()]
    
    user_ids = load_user_profiles(USER_PROFILES_FILE)
    
    logger.info("Loaded %d test stimuli.", len(test_items))
    logger.info("Simulated users: %s", user_ids)
    logger.info("Running Baseline experiment (Ablation without MIPC/MGES)...")

    fieldnames = [
        "id", "user_id", "timestamp", "input_text", "expected_intent", 
        "expected_archetype", "raw_coherence", "emotion", 
        "intent_detected", "mges_intervened", "baseline_archetype"
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        global_id = 1
        start_time = time.time()

        for user_id in user_ids:
            for item in test_items:
                input_text = item["text"]
                expected_intent = item["expected_intent"]
                expected_archetype = DIRECT_MAPPING.get(expected_intent, "UNKNOWN")

                percep_result = perception.analyze(input_text)
                intent_detected = percep_result.get("perception", {}).get("intent", "UNKNOWN")
                emotion = percep_result.get("perception", {}).get("emotion_label", "Neutral")

                baseline_archetype = DIRECT_MAPPING.get(intent_detected, "THE_PROTECTOR")
                
                raw_coherence = 1 if baseline_archetype == expected_archetype else 0

                row = {
                    "id": global_id,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "input_text": input_text,
                    "expected_intent": expected_intent,
                    "expected_archetype": expected_archetype,
                    "raw_coherence": raw_coherence,
                    "emotion": emotion,
                    "intent_detected": intent_detected,
                    "mges_intervened": False, 
                    "baseline_archetype": baseline_archetype
                }
                
                csv_writer.writerow(row)
                global_id += 1

        elapsed = time.time() - start_time
        logger.info("Baseline experiment completed successfully in %.2f seconds.", elapsed)
        logger.info("Results saved to %s", OUTPUT_CSV)

if __name__ == "__main__":
    run_baseline_experiment()