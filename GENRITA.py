import re
import torch
from abc import ABC, abstractmethod
import ast
import json
import logging
import ollama

from model import RoBERTaMultiTaskClassifier
from data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_json_objects(text):
    """
    Scans the input text for well-formed JSON objects and returns
    a list of the ones that successfully parse.
    """
    results = []
    n = len(text)

    i = 0
    while i < n:
        # Find the next opening brace
        if text[i] != '{':
            i += 1
            continue

        depth = 0
        in_string = False
        escape = False

        # Try to find matching closing brace
        for j in range(i, n):
            ch = text[j]

            if escape:
                # This character is escaped; skip special handling
                escape = False
            elif ch == '\\':
                # Next character is escaped iff we are in a string
                if in_string:
                    escape = True
            elif ch == '"' and not escape:
                # Toggle string mode
                in_string = not in_string
            elif not in_string:
                # Only count braces outside strings
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        # Potential end of JSON object
                        candidate = text[i:j+1]

                        # Optional cleanup: remove trailing commas
                        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

                        try:
                            parsed = ast.literal_eval(candidate)
                            results.append(parsed)
                            # Move i to the end of the object
                            i = j
                        except json.JSONDecodeError:
                            # Not valid JSON—ignore this chunk
                            pass
                        break

        i += 1

    return results

SENSE_CLASSES = {
    "Normal and neutral": 0, "Love and romantic": 1, "War and combat": 2,
    "Fantasy and mythology": 3, "Honor and respect": 4, "Drama and tragedy": 5,
    "City and Crowd": 6, "Mountain and the heights": 7, "Desert and dunes": 8,
    "Sea and tides": 9, "Forest and tress": 10,
}
AGE_CLASSES = {
    "ancient and old age": 0,
    "neutral and not special age (non-ancient, non technology)": 1,
    "technology modern age": 2,
}

SENSE_ID_TO_NAME = {v: k for k, v in SENSE_CLASSES.items()}
AGE_ID_TO_NAME = {v: k for k, v in AGE_CLASSES.items()}

# --- LLM Prompts ---

AGE_PROMPT_TEMPLATE = """
# System:
**Role**: You are an `age/temporal-period` classifier model.  
**Output**: The output should be a `dict` of the probability (preds) of belonging to a class. if no class is matched you will score on top to `"neutral and not special age (non-ancient, non technology)"`.

# samples:

**example 1**:
- "context": "Ivory pillars, worn smooth by centuries of wind and sand, lined the entrance to the temple. Moss crept along the carved reliefs, evidence of countless seasons washing over the stone. In the courtyard, a weathered statue of a long-forgotten king stood sentinel, his features blurred by time’s slow hand. Villagers spoke of rituals performed under the full moon, passed down through generations whose names were lost to history. Every incantation and drumbeat echoed an age when gods walked among mortals. Here, the air felt heavy with memory, and each breath tasted of antiquity—a world suspended between legend and the dust of ages past."
+ "preds": {{
    "ancient and old age": 0.87,
    "neutral and not special age (non-ancient, non technology)": 0.10,
    "technology modern age": 0.03
}}

---

**example 2**:
- "context": "In the quiet village, smoke curled from chimneys atop thatched cottages. Farmers guided ox-drawn plows through tilling fields, and children chased chickens along the dirt lanes. Lanterns flickered at dusk, casting warm circles of light onto cobblestones slick with morning dew. The blacksmith’s hammer rang out as he shaped iron by firelight, following techniques unchanged for centuries. No steam engines or electric wires marred the horizon—only the steady pulse of human hands at work. It was a time both simple and enduring, where every stone and seed bore the mark of traditions that spanned generations without the trappings of modern invention."
+ "preds": {{
    "ancient and old age": 0.02,
    "neutral and not special age (non-ancient, non technology)": 0.93,
    "technology modern age": 0.05
}}

---

**example 3**:
- "context": "Glass skyscrapers shimmered under the midday sun, their mirrored facades reflecting a tangle of highways below. Electric cars drifted silently past automated traffic signals, while delivery drones hummed overhead, weaving between towers. Pedestrians tapped at glowing screens strapped to their wrists, and robots in crisp uniforms guided visitors through sleek atria. In the distance, a maglev train glided along its elevated track with a barely audible hum. Data streams flowed invisibly through fiber-optic veins beneath the city, powering every facet of life. Here, progress raced forward at light speed, and the boundaries between human and machine blurred into a seamless modern tapestry."
+ "preds": {{
    "ancient and old age": 0.00,
    "neutral and not special age (non-ancient, non technology)": 0.05,
    "technology modern age": 0.95
}}

---

**example 4**:
- "context": "Stone tools lay scattered in the shallow excavation pit, their edges chipped from use. Paleolithic artists had etched crude images of bison and horses onto cave walls, their pigments faded but still visible in flickering torchlight. The air was thick with the scent of damp rock and the echo of dripping water from unseen chambers. In this place, human hands first shaped the world around them, long before empires or empires’ machines. Every fragment of bone and blade spoke of dawn eras when survival hinged on fire and stone—an age when time itself moved at the pace of the seasons and the turn of the moon."
+ "preds": {{
    "ancient and old age": 0. ninety two,
    "neutral and not special age (non-ancient, non technology)": 0.06,
    "technology modern age": 0.02
}}

---

**example 5**:
- "context": "Beneath the flicker of neon signs, crowds gathered in the labyrinthine alleys of the marketplace. Holographic displays hovered above vendor stalls, advertising fresh produce grown in vertical farms miles away. Autonomous carts glided between shoppers, offering samples of lab-grown meat and cultured dairy. Smartphones buzzed with augmented-reality overlays, guiding tourists toward street-food stalls and historical landmarks. In the distance, solar sail drones harvested energy from the sun, while wind turbines atop skyscrapers turned slowly in the breeze. This was a world where innovation and tradition coexisted, a testament to humanity’s march into a new epoch defined by silicon and steel."
+ "preds": {{
    "ancient and old age": 0.00,
    "neutral and not special age (non-ancient, non technology)": 0.04,
    "technology modern age": 0.96
}}

---
# All Age classes: {{
    "ancient and old age": 0,
    "neutral and not special age (non-ancient, non technology)": 1,
    "technology modern age": 2,
}}

# problem: Fill following Question Marks (?) with suitable classified Probability values (float between 0.0 and 1.0) for each class based on "context". your output structure must be exactly like and only contains following "preds": 

- "context": "{text_content}"
+ "preds": {{
    "ancient and old age": ?,
    "neutral and not special age (non-ancient, non technology)": ?,
    "technology modern age": ?
}}
"""

SENSE_PROMPT_TEMPLATE = """
# System:
**Role**: You are an `sense/genre/theme` classifier model.  
**Output**: The output should be a `dict` of the probability (preds) of belonging to a class.  if no class is matched you will score on top to `"Normal and neutral"`.

# samples:

**example 1**:
- "context": "When the morning sun slipped gently over the rooftops, Julia rose from her simple wooden bed and stretched with a contented sigh. The scent of coffee drifted in from the courtyard below, mingling with the faint aroma of freshly baked bread. Birds chirped in the garden, and distant church bells tolled softly, marking the start of another ordinary day. She paused at the window to gaze at the dew-kissed flowers lining the narrow cobblestone street, savoring the quiet rhythm of her small town. There were no grand adventures awaiting her—only the comforting patterns of daily life, from tending her potted herbs to sweeping the front porch—each moment a gentle reminder that tranquility often resides in the unremarkable."
+ "preds": {{
    "Normal and neutral": 0.84,
    "Love and romantic": 0.02,
    "War and combat": 0.00,
    "Fantasy and mythology": 0.00,
    "Honor and respect": 0.01,
    "Drama and tragedy": 0.01,
    "City and Crowd": 0.05,
    "Mountain and the heights": 0.00,
    "Desert and dunes": 0.00,
    "Sea and tides": 0.00,
    "Forest and tress": 0.07
}}

---

**example 2**:
- "context": "Evan’s heart pounded as he watched Lila step into the ballroom, her gown a cascade of ivory silk that caught the amber light. He recalled every shared glance in the library’s dim corner and each stolen conversation by the fountain at midnight. Now, as she drifted past the marble columns, laughter and music swirling around them, he felt an ache that was both thrilling and terrifying. Their fingers brushed as she accepted his outstretched hand, sending a spark of warmth through his chest. In that moment, the world contracted to the space between two souls, and nothing existed beyond the unspoken promise glittering in her deep, hazel eyes."
+ "preds": {{
    "Normal and neutral": 0.00,
    "Love and romantic": 0.92,
    "War and combat": 0.00,
    "Fantasy and mythology": 0.00,
    "Honor and respect": 0.02,
    "Drama and tragedy": 0.03,
    "City and Crowd": 0.01,
    "Mountain and the heights": 0.00,
    "Desert and dunes": 0.00,
    "Sea and tides": 0.00,
    "Forest and tress": 0.02
}}

---

**example 3**:
- "context": "Smoke curled above the shattered ramparts as the last defenders made their stand. Steel rang against steel in a ferocious clash that shook the blood-soaked ground. Captain Aric barked orders through the chaos, rallying his weary soldiers with unwavering resolve. The enemy’s war horn blared, a savage echo that promised death to all who stood in its path. Amid the flash of torches and the roar of trebuchets, friendships were tested and sacrifices made. Every swing of Aric’s blade carried the weight of his kingdom’s survival, and every fallen comrade steeled his determination. In this crucible of fire and steel, valor and desperation intertwined until only one force could claim victory."
+ "preds": {{
    "Normal and neutral": 0.00,
    "Love and romantic": 0.00,
    "War and combat": 0.88,
    "Fantasy and mythology": 0.05,
    "Honor and respect": 0.03,
    "Drama and tragedy": 0.02,
    "City and Crowd": 0.00,
    "Mountain and the heights": 0.00,
    "Desert and dunes": 0.00,
    "Sea and tides": 0.00,
    "Forest and tress": 0.02
}}

---

**example 4**:
- "context": "Elowen traced her fingers over the glowing sigils carved into the ancient oak door. Beyond it lay the realm of the Silver Fey, where moonlight danced on crystalline streams and starlight wove through the woven canopies. Legends spoke of a guardian serpent whose scales shimmered like opals; they said she alone could grant passage to those of pure heart. As Elowen stepped forward, the air hummed with arcane power, and the earth beneath her feet seemed to awaken. A soft voice whispered in her mind, reminding her of the prophecy she’d carried since birth. In that sacred moment, myth and reality blurred, and the boundaries of the known world shifted forever."
+ "preds": {{
    "Normal and neutral": 0.00,
    "Love and romantic": 0.00,
    "War and combat": 0.00,
    "Fantasy and mythology": 0.94,
    "Honor and respect": 0.01,
    "Drama and tragedy": 0.02,
    "City and Crowd": 0.00,
    "Mountain and the heights": 0.01,
    "Desert and dunes": 0.00,
    "Sea and tides": 0.00,
    "Forest and tress": 0.02
}}

---

**example 5**:
- "context": "The station thrummed with life as commuters poured through the wide glass doors, each hurrying toward trains that disappeared into tunnels below the city. Neon signs flickered overhead, advertising late-night cafes and underground jazz clubs. A street musician leaned against a pillar, coaxing melancholic notes from his saxophone that wove through the crowd like smoke. Delivery drones buzzed above, carrying parcels to high-rise apartments where windows glowed against the dusk. Somewhere in the swirl of faces, Anna felt both lost and exhilaratingly free—part of an ever-shifting tapestry of motion and ambition. Here, in the pulse of the metropolis, every stranger’s story became a fleeting echo in the grand narrative of urban life."
+ "preds": {{
    "Normal and neutral": 0.05,
    "Love and romantic": 0.01,
    "War and combat": 0.00,
    "Fantasy and mythology": 0.00,
    "Honor and respect": 0.01,
    "Drama and tragedy": 0.03,
    "City and Crowd": 0.85,
    "Mountain and the heights": 0.00,
    "Desert and dunes": 0.00,
    "Sea and tides": 0.00,
    "Forest and tress": 0.05
}}

---
# All Sense Classes: {{
     "Normal and neutral": 0, "Love and romantic": 1, "War and combat": 2,
     "Fantasy and mythology": 3, "Honor and respect": 4, "Drama and tragedy": 5,
     "City and Crowd": 6, "Mountain and the heights": 7, "Desert and dunes": 8,
     "Sea and tides": 9, "Forest and tress": 10,
}}

# problem: Fill following Question Marks (?) with suitable classified Probability values (float between 0.0 and 1.0) for each class based on "context". your output structure must be exactly like and only contains following "preds":

- "context": "{text_content}"
+ "preds": {{
    "Normal and neutral": ?,
    "Love and romantic": ?,
    "War and combat": ?,
    "Fantasy and mythology": ?,
    "Honor and respect": ?,
    "Drama and tragedy": ?,
    "City and Crowd": ?,
    "Mountain and the heights": ?,
    "Desert and dunes": ?,
    "Sea and tides": ?,
    "Forest and tress": ?
}}
"""

# --- classifiers ---

class GENRITADriver(ABC):
    """
    Abstract Base Class for a classifier. It defines the interface that the main
    pipeline will use, allowing for interchangeable backend implementations (NN, LLM, etc.).
    """

    @staticmethod
    def get_classifer(driver_type : str, driver_params : dict):
        if driver_type == 'nn':
            return NNDriver(checkpoint_path=driver_params['checkpoint_path'])
        elif driver_type == 'llm':
            return LLMDriver(ollama_model_name=driver_params['ollama_model_name'])
        else:
            raise ValueError(f"Invalid driver type specified: {driver_type}. Choose 'nn' or 'llm'.")

    @abstractmethod
    def classify(self, text: str, allowed_senses: set = None, allowed_ages: set = None) -> dict:
        """
        Processes a text string and returns classification predictions for sense and age.

        Args:
            text (str): The input text to classify.
            allowed_senses (set, optional): A set of sense class IDs to filter by.
            allowed_ages (set, optional): A set of age class IDs to filter by.

        Returns:
            dict: A dictionary containing 'sense_prediction' and 'age_prediction'.
        """
        pass

class NNDriver(GENRITADriver):
    """
    A classifier driver that uses the fine-tuned RoBERTa-based neural network model.
    """

    def __init__(self, checkpoint_path: str):
        logging.info("--- Initializing NN Driver ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_classifier(checkpoint_path)
        logging.info(f"NN Driver ready on device: {self.device}")

    def _load_classifier(self, checkpoint_path: str):
        """Loads the text classification model from a checkpoint."""
        try:
            logging.info(f"Loading Classification Model from: {checkpoint_path}")
            model = RoBERTaMultiTaskClassifier.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=self.device
            )
            model.freeze()
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Could not load classifier model: {e}")
            raise

    def _predict_with_probabilities(self, text: str):
        """Gets probability distributions from the classifier."""
        encoding = self.model.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.model.hparams.max_token_len,
            return_token_type_ids=False, padding="max_length", truncation=True, return_tensors='pt',
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            sense_logits, age_logits = self.model(input_ids, attention_mask)
        return torch.softmax(sense_logits, dim=1).squeeze(), torch.softmax(age_logits, dim=1).squeeze()

    def _get_best_allowed_prediction(self, probabilities: torch.Tensor, id_to_name_map: dict, allowed_ids: set = None):
        """Finds the highest-confidence prediction within the list of allowed class IDs."""
        if not allowed_ids:
            prob, idx_tensor = torch.max(probabilities, dim=0)
            idx = idx_tensor.item()
            return {"class_name": id_to_name_map.get(idx, "Unknown"), "class_id": idx, "confidence": prob.item()}

        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        for i, idx_tensor in enumerate(sorted_indices):
            idx = idx_tensor.item()
            if idx in allowed_ids:
                return {"class_name": id_to_name_map.get(idx, "Unknown"), "class_id": idx,
                        "confidence": sorted_probs[i].item()}
        return {"class_name": "No allowed class found", "class_id": -1, "confidence": 0.0}

    def classify(self, text: str, allowed_senses: set = None, allowed_ages: set = None) -> dict:
        sense_probs, age_probs = self._predict_with_probabilities(DataProcessor._clean_text(text, to_lower=True))
        sense_pred = self._get_best_allowed_prediction(sense_probs, SENSE_ID_TO_NAME, allowed_senses)
        age_pred = self._get_best_allowed_prediction(age_probs, AGE_ID_TO_NAME, allowed_ages)

        return {
            "sense_prediction": sense_pred,
            "age_prediction": age_pred
        }

class LLMDriver(GENRITADriver):
    """
    A classifier driver that uses a generative LLM (via Ollama) with few-shot prompting.
    This version uses ast.literal_eval for robust parsing of the LLM's output.
    """

    def __init__(self, ollama_model_name: str):
        logging.info("--- Initializing LLM Driver ---")
        if ollama is None:
            raise ImportError("The 'ollama' library is required. Please run 'pip install ollama'.")
        self.client = ollama.Client()
        self.model_name = ollama_model_name
        try:
            self.client.list()
            logging.info(f"LLM Driver connected to Ollama, using model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to connect to Ollama. Is the Ollama service running? Error: {e}")
            raise

    def _call_llm(self, prompt: str) -> dict:
        """
        Sends a prompt to the Ollama model and safely evaluates the Python dict-like string response.
        """
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt)
            parsed_dict = self.response_cleaner(response)
            if not isinstance(parsed_dict, dict):
                logging.warning(f"Parsed output is not a dictionary. Type: {type(parsed_dict)}. Output: {parsed_dict}")
                return {}

            return parsed_dict

        except Exception as e:
            logging.error(f"An error occurred while calling the LLM: {e}")
            return {}

    def response_cleaner(self, response: ollama.GenerateResponse):
        response_text = response['response'].strip()

        # Normalize smart quotes
        response_text = response_text.replace('“', '"').replace('”', '"')

        # Try to parse with ast.literal_eval (if it looks like a dict)
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')

        parsed_dict = {}

        if start_index != -1 and end_index != -1:
            try:
                parsed = extract_json_objects(response_text)[0]
            except Exception as e:
                logging.warning(f"Literal eval failed: {e}")
                logging.warning(f"Raw response: {response_text}")
                return {}

            # Flatten if response has nested "preds" dict
            if isinstance(parsed, dict) and 'preds' in parsed and isinstance(parsed['preds'], dict):
                parsed_dict = parsed['preds']
            else:
                parsed_dict = parsed

        else:
            # Fallback: line-by-line parsing
            lines = response_text.splitlines()
            for line in lines:
                match = re.match(r'^\s*"?([^"]+)"?\s*:\s*([0-9.]+)', line)
                if match:
                    key, value = match.groups()
                    try:
                        parsed_dict[key] = float(value)
                    except ValueError:
                        continue

            if not parsed_dict:
                logging.warning(f"LLM did not return a dictionary-like object. Response: {response_text}")
                return {}

        # Final cleaning step
        cleaned_dict = {}
        for key, value in parsed_dict.items():
            try:
                cleaned_dict[key] = float(value)
            except (ValueError, TypeError):
                continue
        return cleaned_dict

    def _get_best_prediction_from_llm_output(self, preds: dict, class_map: dict, id_to_name_map: dict,
                                             allowed_ids: set = None):
        """Finds the best prediction from the LLM's probability dictionary."""
        if not preds:
            return {"class_name": "LLM call failed", "class_id": -1, "confidence": 0.0}

        source_preds = preds
        if allowed_ids:
            allowed_preds = {
                name: prob for name, prob in preds.items()
                if class_map.get(name) in allowed_ids
            }
            if not allowed_preds:
                return {"class_name": "No allowed class found", "class_id": -1, "confidence": 0.0}
            source_preds = allowed_preds

        if not source_preds:
            return {"class_name": "No predictions available", "class_id": -1, "confidence": 0.0}

        best_class_name = max(source_preds, key=source_preds.get)
        confidence = source_preds[best_class_name]
        class_id = class_map.get(best_class_name, -1)

        return {"class_name": best_class_name, "class_id": class_id, "confidence": confidence}

    def classify(self, text: str, allowed_senses: set = None, allowed_ages: set = None) -> dict:
        # 1. Classify Age
        age_prompt = AGE_PROMPT_TEMPLATE.format(text_content=json.dumps(DataProcessor._clean_text(text, to_lower=False))[1:-1])
        age_preds_dict = self._call_llm(age_prompt)
        age_pred = self._get_best_prediction_from_llm_output(age_preds_dict, AGE_CLASSES, AGE_ID_TO_NAME, allowed_ages)

        # 2. Classify Sense
        sense_prompt = SENSE_PROMPT_TEMPLATE.format(text_content=json.dumps(text)[1:-1])
        sense_preds_dict = self._call_llm(sense_prompt)
        sense_pred = self._get_best_prediction_from_llm_output(sense_preds_dict, SENSE_CLASSES, SENSE_ID_TO_NAME,
                                                               allowed_senses)

        return {
            "sense_prediction": sense_pred,
            "age_prediction": age_pred
        }
