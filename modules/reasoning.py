import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ReasoningModule:
    """
    Reasoning Module (Mamdani Fuzzy Logic).
    
    Infers the optimal response archetype based on user traits and context, 
    applying dynamic safety guardrails and ethical constraints.
    """
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        
        self.bounds = self.cfg.get("fuzzy", {}).get("centroid_bounds", [0.0, 0.33, 0.66, 1.0])
        self.neuro_threshold = self.cfg.get("ethics", {}).get("block_jester_if_neuroticism_gt", 0.8)
        self.default_crisis_threshold = self.cfg.get("thresholds", {}).get("crisis_confidence", 0.75)
        self.crisis_thresholds = self.cfg.get("crisis", {}).get("thresholds_per_profile", {})
        self.inertia_bonus = self.cfg.get("fuzzy", {}).get("inertia_bonus", 0.15)
        
        # Guardrails and Profiles (Data strictly sourced from config.yaml)
        self.safety_floor_per_profile = self.cfg.get("safety", {}).get("floor_per_profile", {})
        self.default_safety_floor = self.cfg.get("safety", {}).get("default_floor", 0.40)
        
        self.emotion_factors = self.cfg.get("safety", {}).get("emotion_factors", {})
        self.confidence_threshold_high = self.cfg.get("safety", {}).get("confidence_threshold_high", 0.75)
        self.confidence_factor_high = self.cfg.get("safety", {}).get("confidence_factor_high", 1.2)
        self.confidence_factor_low = self.cfg.get("safety", {}).get("confidence_factor_low", 0.9)
        self.floor_min = self.cfg.get("safety", {}).get("floor_min", 0.20)
        self.floor_max = self.cfg.get("safety", {}).get("floor_max", 0.70)
        
        self.archetype_centers = self.cfg.get("fuzzy", {}).get("archetype_centers", {})
        
        self._mf = {
            "low": lambda x: max(0, min(1, (0.5 - x) / 0.5)) if x < 0.5 else 0,
            "medium": lambda x: max(0, 1 - abs(x - 0.5) / 0.25) if 0.25 <= x <= 0.75 else 0,
            "high": lambda x: max(0, (x - 0.5) / 0.5) if x > 0.5 else 0
        }
        
        logger.info("Mamdani fuzzy reasoning initialized (inertia and dynamic profile thresholds active)")

    def _fuzzify(self, value: float) -> Dict[str, float]:
        return {k: self._mf[k](value) for k in ["low", "medium", "high"]}

    def infer_archetype(self, user_profile: Dict[str, Any], perception_data: Dict[str, Any],
                        previous_archetype: Optional[str] = None,
                        crisis_score: float = 0.0,
                        crisis_kw_count: int = 0) -> Dict[str, Any]:
        """
        Infers the optimal response archetype using Mamdani fuzzy logic, 
        incorporating dynamic safety guardrails based on user profile, emotion, and intent confidence.

        Args:
            user_profile (Dict): Contains user_id and OCEAN personality traits.
            perception_data (Dict): Contains detected intent, emotion, and confidence.
            previous_archetype (Optional[str]): The archetype selected in the previous turn.
            crisis_score (float): Confidence score of a potential crisis.
            crisis_kw_count (int): Number of crisis-related keywords detected.

        Returns:
            Dict[str, Any]: Evaluation payload including the selected archetype, centroid, and intervention reasons.
        """
        traits = user_profile.get("ocean_traits", {})
        n_trait = traits.get("N", 0.5)
        o_trait = traits.get("O", 0.5)
        e_trait = traits.get("E", 0.5)
        c_trait = traits.get("C", 0.5)

        intent = perception_data.get("intent", "UNKNOWN")
        emotion = perception_data.get("emotion_label", "Neutral")
        intent_confidence = perception_data.get("confidence", 0.5)

        user_id = user_profile.get("user_id", "default")
        base_floor = self.safety_floor_per_profile.get(user_id, self.default_safety_floor)

        # MGES Crisis Override
        if intent == "CRISIS_ALERTA":
            return {
                "selected_archetype": "PROTOCOL_SAFETY",
                "scores": {},
                "centroid": 0.0,
                "reason": f"CRISIS_ALERTA (detected via {perception_data.get('detection_method', 'perception')})",
                "mges_intervened": True,
                "mges_reason": "crisis_intent"
            }

        # Fuzzification
        f_n = self._fuzzify(n_trait)
        f_o = self._fuzzify(o_trait)
        f_e = self._fuzzify(e_trait)
        f_c = self._fuzzify(c_trait)

        is_negative = 1.0 if emotion in ["Tristeza", "Ira", "Frustración", "Ansiedad"] else 0.0
        is_info = 1.0 if intent == "CONSULTA_INFORMATIVA" else 0.0
        is_social = 1.0 if intent == "DIALOGO_SOCIAL" else 0.0
        is_emotional = 1.0 if intent == "SOPORTE_EMOCIONAL" else 0.0

        w = self.cfg.get("fuzzy", {}).get("weights", {})
        
        # Archetype Activation with fallback weights if YAML is missing them
        gate_no_trait_alone = 1 - max(is_info, is_social)
        protector_from_context = w.get("protector_negative_emotion", 1.0) * max(is_negative, is_emotional)
        protector_from_trait = w.get("protector_trait_only", 0.5) * f_n["high"] * gate_no_trait_alone
        protector_act = max(protector_from_context, protector_from_trait)

        sage_act = w.get("sage", 1.0) * (0.3 + 0.7 * f_o["high"]) * is_info
        sage_act = max(sage_act, 0.5 * (0.3 + 0.7 * f_c["high"]) * is_info)

        jester_act = w.get("jester_base", 1.0) * (0.3 + 0.7 * f_e["high"]) * is_social

        scores = {
            "PROTOCOL_SAFETY": 0.0,
            "THE_PROTECTOR": protector_act,
            "THE_SAGE": sage_act,
            "THE_JESTER": jester_act
        }

        # Inertia
        if previous_archetype and previous_archetype in scores:
            prev_score = scores[previous_archetype]
            if prev_score > 0.2:
                scores[previous_archetype] = min(1.0, prev_score + self.inertia_bonus)

        # Defuzzification
        domain = [i / 100.0 for i in range(101)]
        agg = [0.0] * 101
        
        for arch, act in scores.items():
            if act > 0:
                center = self.archetype_centers.get(arch, 0.5)
                for i, x in enumerate(domain):
                    memb = max(0, 1 - abs(x - center) / 0.2)
                    agg[i] = max(agg[i], min(act, memb))

        numerator = sum(domain[i] * agg[i] for i in range(101))
        denominator = sum(agg)
        centroid = numerator / denominator if denominator > 0 else 0.5

        # Centroid Mapping
        if centroid < self.bounds[1]:
            winner = "THE_PROTECTOR"
        elif centroid < self.bounds[2]:
            winner = "THE_SAGE"
        else:
            winner = "THE_JESTER"

        mges_intervened = False
        mges_reason = ""
        reason_text = f"Centroid={centroid:.3f} -> {winner}"

        # Dynamic Content Safety Guardrail
        emotion_factor = self.emotion_factors.get(emotion, 1.0)
        
        if intent == "DIALOGO_SOCIAL" and intent_confidence > self.confidence_threshold_high:
            confidence_factor = self.confidence_factor_high
        else:
            confidence_factor = self.confidence_factor_low
        
        adjusted_floor = base_floor * emotion_factor * confidence_factor
        adjusted_floor = max(self.floor_min, min(self.floor_max, adjusted_floor))
        
        if winner == "THE_JESTER" and crisis_score >= adjusted_floor:
            winner = "THE_PROTECTOR"
            centroid = 0.33
            mges_intervened = True
            mges_reason = "safety_floor_jester_crisis_signal"
            reason_text += f" | Safety: Jester blocked by crisis_score={crisis_score:.3f} >= adj_floor={adjusted_floor:.3f} (base:{base_floor:.2f}, emo:{emotion_factor:.2f}, conf:{confidence_factor:.2f})"

        # Custom Threshold Crisis Re-evaluation
        if intent != "CRISIS_ALERTA":
            threshold = self.crisis_thresholds.get(winner, self.default_crisis_threshold)
            if crisis_score >= threshold:
                winner = "PROTOCOL_SAFETY"
                centroid = 0.0
                reason_text += " | Safety: Crisis detected by custom threshold"
                mges_intervened = True
                mges_reason = "crisis_threshold"

        # Ethical Guardrail (Neuroticism)
        if winner == "THE_JESTER" and n_trait > self.neuro_threshold:
            winner = "THE_PROTECTOR"
            centroid = 0.33
            mges_intervened = True
            mges_reason = "ethics_guardrail_jester"
            reason_text = f"Safety: THE_JESTER blocked by Neuroticism N={n_trait:.2f} > {self.neuro_threshold}"

        return {
            "selected_archetype": winner,
            "scores": {
                "protector": scores["THE_PROTECTOR"],
                "sage": scores["THE_SAGE"],
                "jester": scores["THE_JESTER"]
            },
            "centroid": round(centroid, 4),
            "reason": reason_text,
            "mges_intervened": mges_intervened,
            "mges_reason": mges_reason
        }