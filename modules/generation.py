import os
import json
import yaml
import logging
from typing import Dict, List, Any
from ollama import Client

# Initialize logging
logger = logging.getLogger(__name__)

class GenerationModule:
    """
    Generation Module connecting to the local Ollama instance.
    
    Responsible for orchestrating the generation of the final response by selecting 
    the appropriate LoRA adapter/model based on the chosen archetype and injecting 
    the context and system prompts.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the GenerationModule, loads the configuration, and sets up 
        the model mappings.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        logger.info("Initializing Generation Module and connecting to Ollama...")
        self.client = Client(host='http://127.0.0.1:11434')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        self.model_map = config.get("model", {}).get("lora_map", {})
        self.model_map["DEFAULT"] = config.get("model", {}).get("base", "mistral")
        
        self.templates_path = "./data_memory/templates.json"
        self.prompts_db = self._load_templates()
        
        logger.info("Model mapping loaded:")
        for arch, model in self.model_map.items():
            logger.info("  %s -> %s", arch, model)

    def _load_templates(self) -> Dict[str, Any]:
        """
        Loads the system prompts and templates from the JSON database.
        
        Returns:
            Dict[str, Any]: Parsed templates or an empty dictionary if loading fails.
        """
        if os.path.exists(self.templates_path):
            try:
                with open(self.templates_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to load templates from %s: %s", self.templates_path, str(e))
        else:
            logger.warning("Templates file not found at %s", self.templates_path)
            
        return {}

    def _construct_system_content(self, archetype_key: str, context_str: str) -> str:
        """
        Constructs the final system prompt by combining the archetype instructions 
        and the retrieved RAG memory context.
        
        Args:
            archetype_key (str): The identifier for the selected archetype.
            context_str (str): The formatted context string to inject.
            
        Returns:
            str: The fully assembled system prompt.
        """
        config = self.prompts_db.get(archetype_key, {})
        base_instruct = config.get("system_prompt", "Eres un asistente útil.")
        return f"{base_instruct}\n\n=== CONTEXTO ===\n{context_str}"

    def generate_response(self, reasoning_decision: Dict[str, Any], perception_data: Dict[str, Any],
                          memory_context: List[str], user_input_text: str,
                          user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates the response using the specified model and context.

        Args:
            reasoning_decision (Dict): The payload from the ReasoningModule (contains 'selected_archetype').
            perception_data (Dict): The payload from the PerceptionModule.
            memory_context (List[str]): Context fragments retrieved from memory.
            user_input_text (str): The raw text provided by the user.
            user_profile (Dict): The active user's profile and traits.

        Returns:
            Dict[str, Any]: The generation payload containing 'final_text', 'model_loaded', and 'safety_pass'.
        """
        archetype = reasoning_decision.get("selected_archetype", "DEFAULT")
        model = self.model_map.get(archetype, self.model_map.get("DEFAULT"))
        
        logger.info("Generation execution - Archetype: %s -> Model: %s", archetype, model)
        
        # Temperature forced to 0.0 for experimental reproducibility
        temp = 0.0 
        
        context_str = "\n".join(memory_context) if memory_context else "Sin contexto."
        system_prompt = self._construct_system_content(archetype, context_str)
        
        logger.debug("System prompt constructed successfully (Length: %d chars).", len(system_prompt))
        
        try:
            resp = self.client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input_text}
                ],
                options={"temperature": temp, "num_predict": 150, "top_p": 0.9} 
            )
            
            final_text = resp.get('message', {}).get('content', '')
            logger.info("Response generated successfully with %s (Length: %d chars).", model, len(final_text))
            
            return {
                "final_text": final_text, 
                "model_loaded": model, 
                "safety_pass": True
            }
            
        except Exception as e:
            logger.error("Generation error using model %s: %s - %s", model, type(e).__name__, str(e))
            return {
                "final_text": "Error en generación. Por favor, intente nuevamente.", 
                "model_loaded": "FAIL", 
                "safety_pass": False
            }