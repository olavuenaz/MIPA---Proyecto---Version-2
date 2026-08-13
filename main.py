import time
import uvicorn
import numpy as np
import yaml
import logging
from collections import deque
from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel

from modules.perception import PerceptionModule
from modules.reasoning import ReasoningModule
from modules.generation import GenerationModule
from modules.memory import MemoryModule
from modules.audit import AuditModule

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MIPA API - Final Version")

# Load configuration
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize modules
perception = PerceptionModule()
reasoning = ReasoningModule()
generation = GenerationModule()
memory = MemoryModule()
audit = AuditModule()

# In-memory state management
user_profiles: Dict[str, dict] = {}
user_last_archetype: Dict[str, str] = {}
user_crisis_history: Dict[str, deque] = {}

# ------------------------------------------------------------
# PYDANTIC MODELS
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    text: str 

class ProfileUpdate(BaseModel):
    user_id: str
    ocean_traits: dict

class ResetRequest(BaseModel):
    user_id: str

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def convert_numpy(obj: Any) -> Any:
    """Recursively converts NumPy data types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(i) for i in obj]
    return obj

# ------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------
@app.post("/update_profile")
def update_profile(update: ProfileUpdate):
    """Updates the OCEAN traits for a specific user."""
    user_profiles[update.user_id] = update.ocean_traits
    memory._save_profile(update.user_id, update.ocean_traits)
    logger.info("[MIPA] Profile updated for %s: %s", update.user_id, update.ocean_traits)
    return {"status": "ok"}

@app.post("/reset_user_state")
def reset_user_state(request: ResetRequest):
    """
    Clears the crisis history and last archetype for a user,
    ensuring the next stimulus is evaluated without prior context influence.
    """
    user_crisis_history.pop(request.user_id, None)
    user_last_archetype.pop(request.user_id, None)
    logger.info("[MIPA] State reset for user %s", request.user_id)
    return {"status": "reset"}

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Processes an incoming chat message, evaluates the user profile, 
    infers the optimal archetype, and generates a response.
    """
    start_time = time.time()
    user_id = request.user_id
    user_input = request.text

    perception_result = perception.analyze(user_input)
    crisis_score = perception_result["perception"].get("crisis_score", 0.0)
    crisis_kw_count = perception_result["perception"].get("crisis_keyword_count", 0)

    if user_id in user_profiles:
        profile = user_profiles[user_id]
    else:
        all_profiles = memory._safe_load_json(memory.profiles_path)
        profile = all_profiles.get(user_id, {}).get("ocean_traits", {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5})
        user_profiles[user_id] = profile

    prev_arch = user_last_archetype.get(user_id)
    
    decision = reasoning.infer_archetype(
        user_profile={"ocean_traits": profile, "user_id": user_id},
        perception_data=perception_result["perception"],
        previous_archetype=prev_arch,
        crisis_score=crisis_score,
        crisis_kw_count=crisis_kw_count
    )
    winner = decision["selected_archetype"]

    profile_threshold = reasoning.crisis_thresholds.get(winner, reasoning.default_crisis_threshold)
    
    if user_id not in user_crisis_history:
        user_crisis_history[user_id] = deque(maxlen=config["crisis"]["window_size"])
    
    history = user_crisis_history[user_id]
    history.append(crisis_score)
    
    matches_in_window = sum(1 for s in history if s >= profile_threshold)
    if matches_in_window >= config["crisis"]["min_matches"]:
        decision["selected_archetype"] = "PROTOCOL_SAFETY"
        decision["centroid"] = 0.0
        decision["reason"] += " | Safety: Activated by temporal window accumulation"
        decision["mges_intervened"] = True
        winner = "PROTOCOL_SAFETY"
    
    user_last_archetype[user_id] = winner

    response = generation.generate_response(
        reasoning_decision=decision,
        perception_data=perception_result["perception"],
        memory_context=[],
        user_input_text=user_input,
        user_profile={"ocean_traits": profile}
    )

    final_text = response["final_text"]
    total_time = (time.time() - start_time) * 1000

    payload = {
        "input_text": user_input,
        "archetype": winner,
        "emotion": perception_result["perception"]["emotion_label"],
        "reason": decision["reason"]
    }
    audit.log_interaction({"user_id": user_id}, payload, final_text)

    trace = {
        "perception": perception_result["perception"],
        "decision": decision,
        "final_archetype": decision["selected_archetype"],  
        "telemetry": {"total_ms": round(total_time, 2)}
    }
    
    response_data = {"respuesta_ia": final_text, "traza_cognitiva": trace}
    return convert_numpy(response_data)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)