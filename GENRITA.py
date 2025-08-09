import re
import torch
from abc import ABC, abstractmethod
import ast
import ollama
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
**Task**: Analyze the provided "context" and return a dictionary of class probabilities.

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

# Problem:
Classify the following "context".

- "context": "{text_content}"

**IMPORTANT INSTRUCTIONS**:
1.  Your response **MUST** be a single, valid JSON object.
2.  Do **NOT** include any explanations, conversation, or markdown formatting like ```json.
3.  The JSON object must contain a single key, "preds", whose value is a dictionary of the classes and their probabilities.
4.  All probability values **MUST** be floats between 0.0 and 1.0.
5.  The sum of all probabilities should be 1.0.
6.  Do **NOT** use '?' or any non-numeric values. If a class has zero probability, use `0.0`.

Your entire output must be **ONLY** the JSON object, structured like this:
+ "preds": {{
    "ancient and old age": ?,
    "neutral and not special age (non-ancient, non technology)": ?,
    "technology modern age": ?
}}
"""

SENSE_PROMPT_TEMPLATE = """
# System:
**Role**: You are an `sense/genre/theme` classifier model.  
**Task**: Analyze the provided "context" and return a dictionary of class probabilities.

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

# Problem:
Classify the following "context".

- "context": "{text_content}"

**IMPORTANT INSTRUCTIONS**:
1.  Your response **MUST** be a single, valid JSON object.
2.  Do **NOT** include any explanations, conversation, or markdown formatting like ```json.
3.  The JSON object must contain a single key, "preds", whose value is a dictionary of the classes and their probabilities.
4.  All probability values **MUST** be floats between 0.0 and 1.0.
5.  Do **NOT** use '?' or any non-numeric values. If a class has zero probability, use `0.0`.

Your entire output must be **ONLY** the JSON object, structured like this:
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

    def __init__(self, ollama_model_name: str, max_retries: int = 3):
        logging.info("--- Initializing LLM Driver ---")
        if ollama is None:
            raise ImportError("The 'ollama' library is required. Please run 'pip install ollama'.")
        self.client = ollama.Client()
        self.max_retries = max_retries
        self.model_name = ollama_model_name
        try:
            self.client.list()
            logging.info(f"LLM Driver connected to Ollama, using model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to connect to Ollama. Is the Ollama service running? Error: {e}")
            raise

    # def _call_llm(self, prompt: str) -> dict:
    #     """
    #     Sends a prompt to the Ollama model and safely evaluates the Python dict-like string response.
    #     """
    #     try:
    #         response = self.client.generate(model=self.model_name, prompt=prompt)
    #         parsed_dict = self.response_cleaner(response)
    #         # response_text = response['response'].strip()
    #         # parsed_dict = self._parse_llm_response(response_text)
    #
    #         if not isinstance(parsed_dict, dict):
    #             logging.warning(f"Parsed output is not a dictionary. Type: {type(parsed_dict)}. Output: {parsed_dict}")
    #             return {}
    #
    #         return parsed_dict
    #
    #     except Exception as e:
    #         logging.error(f"An error occurred while calling the LLM: {e}")
    #         return {}

    def _call_llm(self, prompt: str) -> dict:
        """
        Sends a prompt to the Ollama model and evaluates the response.
        Retries generating a new response if the output is not parsable.
        Does NOT retry on underlying API/connection errors.
        """
        try:
            for attempt in range(self.max_retries):
                # The API call is inside the loop to get a new response on each attempt.
                response = self.client.generate(model=self.model_name, prompt=prompt)
                parsed_dict = self.response_cleaner(response)

                # On successful parsing, return the result immediately.
                if parsed_dict and isinstance(parsed_dict, dict):
                    return parsed_dict

                # If parsing failed, log it. The loop will then make a new attempt.
                logging.warning(
                    f"LLM output was not parsable on attempt {attempt + 1}/{self.max_retries}. Retrying generation."
                )

            # This point is reached only if all attempts to get a parsable output failed.
            logging.error(f"Failed to get a parsable response from LLM after {self.max_retries} attempts.")
            return {}

        except Exception as e:
            # This catches non-retriable errors like network issues or Ollama service failures.
            # The function fails immediately without retrying.
            logging.error(f"A non-retriable API error occurred, stopping immediately: {e}")
            return {}

    def _parse_llm_response(self, response_text: str) -> dict:
        """
        Robustly parses the LLM's response to find and decode a JSON object.
        Handles surrounding text, markdown, and minor format errors.
        """
        # 1. Use regex to greedily find a JSON-like object. This is great for
        # extracting a JSON blob from within conversational text.
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not match:
            logging.warning(f"Parser could not find a JSON-like object in the response: {response_text}")
            return {}

        json_str = match.group(0)

        # 2. Try to parse the extracted string using the standard `json` library.
        try:
            parsed_dict = json.loads(json_str)
        except json.JSONDecodeError:
            # If standard JSON parsing fails, it might be due to single quotes or other
            # Python-isms. Use `ast.literal_eval` as a safer fallback than `eval()`.
            try:
                # Replace non-breaking spaces which can cause parsing errors
                json_str_cleaned = json_str.replace('\u00a0', ' ')
                parsed_dict = ast.literal_eval(json_str_cleaned)
            except (ValueError, SyntaxError, MemoryError) as e:
                logging.warning(f"Both json.loads and ast.literal_eval failed. Error: {e}. Raw string: {json_str}")
                return {}

        # 3. Standardize the output structure. Sometimes models will nest the
        # result inside `{"preds": {...}}`. We just want the inner dictionary.
        if isinstance(parsed_dict, dict) and 'preds' in parsed_dict and isinstance(parsed_dict['preds'], dict):
            final_dict = parsed_dict['preds']
        else:
            final_dict = parsed_dict

        # 4. Final cleaning: Ensure all values are floats and discard any that aren't.
        # This gracefully handles cases where the model might output "?" or other text.
        cleaned_dict = {}
        for key, value in final_dict.items():
            try:
                cleaned_dict[str(key)] = float(value)
            except (ValueError, TypeError):
                logging.warning(f"Could not convert value '{value}' for key '{key}'. Discarding this entry.")
                continue
        return cleaned_dict

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

TEST_CASES = [
    {
        "text": "The market square buzzed with a low hum of conversation. Merchants hawked their wares from colourful stalls, their voices a chaotic symphony. Children chased pigeons across the cobblestones, their laughter echoing between the tall, narrow houses. A woman haggled over the price of bread, her expression firm but fair. The air smelled of fresh-baked goods, spices, and the faint scent of rain on stone. It was a typical afternoon, unremarkable in its routine, a slice of everyday life unfolding in a town that had seen countless such days pass by. The gentle rhythm of commerce and community was the town's steady, comforting heartbeat, a simple existence without grand drama.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "He watched her from across the crowded ballroom, a vision in silk and starlight. Her laughter was the only sound he could hear above the orchestra's melody. When their eyes finally met, the world seemed to slow down, the other guests fading into a blur. He crossed the room, his heart pounding a rhythm that matched the waltz. 'May I have this dance?' he asked, his voice barely a whisper. She smiled, a gesture that lit up her entire face, and placed her hand in his. As they moved together across the polished floor, it felt as though they were the only two people in the universe, bound by an unspoken connection.",
        "sense_prediction": {
            "class_name": "Love and romantic",
            "class_id": 1
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The plasma bolts seared through the bulkhead, sending molten metal spraying across the corridor. Alarms blared, casting the scene in a strobing red light. Commander Eva Rostova braced against the wall, her pulse rifle held tight. 'Hold the line!' she yelled over the comms, the sounds of explosions rattling her teeth. Her squad returned fire, their energy rounds impacting against the invaders' shields with concussive force. The air was thick with the smell of ozone and fear. They were the last defense for the colony ship, a thin line of defiance against the robotic swarm. This battle would determine the fate of ten thousand souls, a final, desperate stand in the cold vacuum of space.",
        "sense_prediction": {
            "class_name": "War and combat",
            "class_id": 2
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "In the ancient valley, a creature with obsidian scales and burning eyes guarded the gate. The villagers called it Narthul. Another dragon, a smaller but faster one, patrolled the skies. This second dragon was known as Ignis. Unlike Narthul, Ignis had shimmering silver scales that reflected the moonlight, making it look like a constellation come to life. The legends said the two were bound by a curse as old as the mountains themselves, destined to protect a hidden artifact from the world of men. No knight had ever bested them, their combined might a testament to the raw, untamed magic that still lingered in the forgotten corners of the world.",
        "sense_prediction": {
            "class_name": "Fantasy and mythology",
            "class_id": 3
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The old king sat upon his throne, his face a mask of weary wisdom. Before him knelt the young knight, his armor gleaming in the torchlight. 'You have served this kingdom with unwavering loyalty,' the king's voice boomed, echoing in the great hall. 'You have shown courage in the face of our enemies and compassion to our people. For your deeds, we bestow upon you the title of Lord Commander of the Royal Guard.' He gestured for his aide to bring forth the ceremonial sword, its hilt embedded with the sigil of the royal house. The knight accepted the honor with a bowed head, his heart swelling with pride and a profound sense of duty.",
        "sense_prediction": {
            "class_name": "Honor and respect",
            "class_id": 4
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "Rain lashed against the windowpane, mirroring the tears streaming down her face. The letter lay crumpled in her hand, its words a cruel testament to his betrayal. How could he have left? After everything they had been through, after all the promises whispered in the dark. The room felt cold and empty, each shadow a haunting reminder of the life they had built together, now shattered into a million pieces. A single sob escaped her lips, a raw, painful sound in the suffocating silence. The future she had envisioned, once so bright and full of hope, was now a bleak, desolate landscape. She was alone, adrift in a sea of sorrow.",
        "sense_prediction": {
            "class_name": "Drama and tragedy",
            "class_id": 5
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The neon signs of the metropolis bled into the rain-slicked streets, casting long, distorted reflections. Flying vehicles zipped between towering skyscrapers that pierced the clouds, their anti-gravity engines a constant, low thrum. Crowds of people, a diverse mix of humans and cybernetically enhanced individuals, moved like a river along the elevated walkways. The air was a cocktail of synthetic fragrances, exhaust fumes, and street food. Down below, in the undercity canyons, steam vented from massive grates, shrouding the lower levels in a perpetual mist. This was Neo-Kyoto in 2242, a city that never slept, a monument to humanity's relentless, sprawling ambition and its technological prowess, a concrete and steel jungle teeming with life.",
        "sense_prediction": {
            "class_name": "City and Crowd",
            "class_id": 6
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The air grew thin as he climbed, each breath a sharp, cold sting in his lungs. Below him, the world was a patchwork of green valleys and dark forests, shrouded in a light morning mist. The peak was a jagged shard of granite against the pale blue sky, a silent sentinel that had stood for millennia. Eagles soared on the thermal updrafts, their cries lonely and wild. He paused, leaning on his walking stick, and felt a profound sense of insignificance and awe. Here, on the roof of the world, the concerns of his everyday life seemed trivial. There was only the wind, the rock, and the vast, humbling expanse of the sky.",
        "sense_prediction": {
            "class_name": "Mountain and the heights",
            "class_id": 7
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The sun beat down relentlessly on the endless sea of sand. Dunes rolled towards the horizon, their crests like sharp, golden waves frozen in time. The heat was a physical presence, shimmering in the air and baking the cracked earth. A lone caravan of camels plodded forward, their silhouettes dark against the blinding glare. The only sound was the soft whisper of wind sculpting the sand and the rhythmic plod of the animals' feet. This was the Great Erg, a place of stark, brutal beauty where water was more precious than gold. Survival here depended on knowledge passed down through generations, an ancient understanding of a land that was both unforgiving and pure.",
        "sense_prediction": {
            "class_name": "Desert and dunes",
            "class_id": 8
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The waves crashed against the hull of the old fishing trawler, a steady, powerful rhythm. Salt spray misted the air, clinging to his weathered face. The sea was a vast, churning expanse of grey-green, stretching to meet a sky of the same colour. Gulls cried overhead, circling the boat in hopes of a free meal. He hauled on the heavy nets, his muscles straining with the familiar effort. This was his life, and his father's before him. A life dictated by the whims of the tide and the pull of the moon. He felt a deep, abiding connection to this wild water, a force that could provide a bounty or swallow you whole.",
        "sense_prediction": {
            "class_name": "Sea and tides",
            "class_id": 9
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "Sunlight struggled to pierce the dense canopy, casting the forest floor in a perpetual twilight. Ancient oaks, their branches heavy with moss, stood like silent giants. The air was cool and damp, rich with the smell of decaying leaves and damp earth. A carpet of ferns and wildflowers covered the ground, broken only by the winding game trails used by deer and other unseen creatures. The only sounds were the rustle of leaves in the gentle breeze and the distant call of a cuckoo. It felt like a world untouched by time, a sanctuary of green and shadow where the modern world and its clamor could not intrude. A place of quiet, living history.",
        "sense_prediction": {
            "class_name": "Forest and tress",
            "class_id": 10
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The scribe dipped his reed pen into the inkpot, the scratching sound echoing in the quiet library of Alexandria. Scrolls were stacked high on wooden shelves, containing the collected knowledge of the known world. He was transcribing a treatise on geometry, his brow furrowed in concentration. Outside, the sounds of the bustling port city were a distant murmur. His task was simple but vital: to preserve the thoughts and discoveries of the great minds for future generations. It was a quiet life, devoid of glory or adventure, but filled with a sense of purpose. Each carefully formed letter was a small act of defiance against the ravages of time.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The android barista slid the synth-coffee across the counter. 'Nutrient paste is on special today,' it droned in a monotone voice. Kaelen ignored it, staring out the cafe window at the passing mag-lev trains. The city was a monument to efficiency, a perfectly calibrated machine where every citizen had a designated function. His own role was data-scrubber, a mind-numbing job that involved sanitizing information streams for the global network. There was no art, no passion, just the cold, sterile logic of the system. He took a sip of the bitter coffee, the taste as bland and manufactured as everything else in this world. It was a safe, predictable existence, utterly devoid of meaning.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "He waited for her by the old stone bridge, a bouquet of wildflowers in his hand. They had met here for the first time, two souls from rival clans who were forbidden to speak. But their love was a force stronger than any ancient feud. When she appeared at the edge of the woods, her cloak pulled low, his heart leaped. She ran to him, and they embraced under the watchful eyes of the moon. 'I cannot live without you,' he whispered, his voice thick with emotion. 'Then let us run away,' she replied, her eyes shining with tears and determination. 'Together, we can find a new life, far from this hatred.'",
        "sense_prediction": {
            "class_name": "Love and romantic",
            "class_id": 1
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "Her avatar shimmered into existence in the virtual garden, a cascade of light and data. His was already waiting, sitting by a holographic waterfall. In this digital space, they could be anyone, go anywhere. Tonight, they chose a world of impossible beauty, with glowing flora and crystalline trees. 'I missed you,' his synthesized voice conveyed a warmth that felt real. 'And I you,' she replied, her avatar reaching out to touch his. Though they were physically miles apart, plugged into their respective neural interfaces, their connection in the network was profound. Here, in the endless expanse of the datasphere, they had found a love that transcended the limitations of the physical world.",
        "sense_prediction": {
            "class_name": "Love and romantic",
            "class_id": 1
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The catapults released their payload, sending massive stones arcing over the battlefield. They crashed into the enemy ranks with sickening thuds, breaking the shield wall. The commander raised his sword, the polished steel catching the morning sun. 'Charge!' he roared, his voice carrying over the sounds of chaos. The infantry surged forward, a wave of leather and iron, their war cries a savage chorus. They met the enemy line with a crash of shields and a clang of blades. The fighting was brutal and close-quarters, a desperate struggle for every inch of ground. This was not a battle of tactics, but of sheer, bloody-minded attrition.",
        "sense_prediction": {
            "class_name": "War and combat",
            "class_id": 2
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The trench was a muddy scar across the landscape, filled with weary soldiers. The air stank of cordite, stale rations, and the ever-present fear of the next artillery barrage. A young private huddled against the wall, clutching his rifle, his knuckles white. He had been here for weeks, and the constant shelling had worn his nerves raw. The world had shrunk to this narrow ditch, a grim reality of mud, barbed wire, and the distant chatter of machine guns. He thought of home, of green fields and quiet evenings, a world away from this mechanized slaughter. A whistle blew, the signal for the next push. He closed his eyes for a moment, then climbed the ladder.",
        "sense_prediction": {
            "class_name": "War and combat",
            "class_id": 2
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The elf maiden moved through the moonlit glade, her silver hair flowing behind her. She sang a soft, ethereal melody, and the very trees seemed to lean in to listen. Her song spoke of the creation of the stars, of the deep magic that flowed through the earth like a river. A unicorn, its coat the colour of fresh snow, emerged from the shadows and rested its head in her lap. This was a place of ancient power, hidden from the eyes of mortals. Here, the veil between worlds was thin, and creatures of myth and legend walked freely under the silent gaze of the moon. It was a remnant of a forgotten age.",
        "sense_prediction": {
            "class_name": "Fantasy and mythology",
            "class_id": 3
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The AI construct, named Oracle, manifested as a shimmering form of pure data in the server room. Its purpose was to sift through the global network, identifying patterns and predicting future events. It was not alive in the biological sense, but it possessed a consciousness that spanned continents. It saw the flow of information as a great river, and it could perceive the subtle currents that hinted at coming storms. Today, it detected a new anomaly, a rogue code that moved with predatory intelligence. It was a digital dragon, a creature of pure information, and Oracle knew it had to be stopped before it could unmake their networked reality.",
        "sense_prediction": {
            "class_name": "Fantasy and mythology",
            "class_id": 3
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The samurai bowed low before his daimyo, his hands placed flat on the tatami mat. 'My lord, I have failed in my duty,' he said, his voice steady despite the shame that burned within him. 'The bandits escaped with the rice shipment. I alone am responsible.' The daimyo was silent for a long moment, his gaze unreadable. 'Your honesty is a virtue, Kenji,' he said finally. 'But your failure has consequences for the village. You will redeem your honor not with your life, but by hunting down these thieves and returning what was stolen. Do not fail me again.' Kenji bowed once more, a silent vow of absolute commitment.",
        "sense_prediction": {
            "class_name": "Honor and respect",
            "class_id": 4
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "Captain Thorne stood on the bridge of his starship, the 'Vindicator,' as it faced the alien dreadnought. His crew looked to him, their faces tense. 'They have demanded our surrender,' the comms officer reported. Thorne's jaw tightened. 'This ship has never surrendered,' he said, his voice ringing with authority. 'We were entrusted with the defense of this sector, and we will not falter. That is our oath, our duty.' He turned to his tactical officer. 'Target their weapon systems. Show them what it means to challenge the Terran fleet.' A sense of pride and shared purpose filled the bridge. They would face this threat with courage, upholding the honor of the service.",
        "sense_prediction": {
            "class_name": "Honor and respect",
            "class_id": 4
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The final scene played out on the grand stage of the amphitheater. The hero, betrayed by his most trusted friend, lay dying in the arms of the queen. With his last breath, he spoke of a prophecy, a warning of a darkness yet to come. The queen let out a cry of anguish that silenced the entire audience, a sound of pure, heart-wrenching grief. The villain, his face a mask of triumphant cruelty, watched from the shadows, his victory complete. As the curtains fell, a heavy silence hung in the air, the weight of the tragedy pressing down on every soul who had witnessed the hero's fall. The story was over, leaving only sorrow.",
        "sense_prediction": {
            "class_name": "Drama and tragedy",
            "class_id": 5
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The transmission came through, her face flickering on the viewscreen. 'The core is critical,' she said, her voice strained. 'I can't stop the meltdown.' On the bridge, Captain Jax felt his world collapse. They were light-years from any help. He had sent her down there, confident in her ability to fix the reactor. Now, her last words were a death sentence for them all. 'I'm sorry, Jax,' she whispered, a single tear tracing a path through the grime on her cheek before the screen went to static. He stared at the empty screen, the silence of the bridge a monument to his catastrophic failure. The ship was a tomb, adrift in the endless night.",
        "sense_prediction": {
            "class_name": "Drama and tragedy",
            "class_id": 5
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The forum in ancient Rome was a chaotic swirl of activity. Senators in white togas debated politics in loud, passionate voices, their gestures grand and theatrical. Merchants shouted their prices from stalls overflowing with goods from across the empire: silks from the East, spices from Africa, pottery from Greece. Couriers rushed through the throng, carrying messages on wax tablets. The air was thick with the smells of dust, sweat, and cooking food. This was the heart of the empire, a vibrant, noisy, and powerful place where decisions were made that would affect millions. It was a city of marble and ambition, teeming with a restless and energetic populace.",
        "sense_prediction": {
            "class_name": "City and Crowd",
            "class_id": 6
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The town square was packed for the annual harvest festival. People milled about, chatting and laughing, their faces lit by the warm glow of lanterns strung between the buildings. A band played cheerful tunes on a makeshift stage, and couples danced on the cobblestones. The air was filled with the scent of roasted chestnuts and mulled wine. Children weaved through the legs of the adults, their faces sticky with candy. It was a time for the community to come together, to celebrate the fruits of their labor and enjoy a brief respite from the hardships of daily life. The sense of shared joy and camaraderie was palpable, a warm blanket against the crisp autumn air.",
        "sense_prediction": {
            "class_name": "City and Crowd",
            "class_id": 6
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "From the monastery perched high on the cliffside, the view was breathtaking. The mountains rolled away in every direction, their peaks capped with snow even in summer. The world below seemed distant and insignificant, a place of noise and strife. Here, there was only the sound of the wind whistling through the prayer flags and the occasional toll of a deep bronze bell. The monks who lived here followed a simple routine of meditation and work, their lives dedicated to seeking enlightenment far from the distractions of the world. It was a place of profound peace and stark beauty, a sanctuary carved into the very roof of the world.",
        "sense_prediction": {
            "class_name": "Mountain and the heights",
            "class_id": 7
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The orbital station hung like a jewel against the black velvet of space. From the observation deck, Commander Lin watched the Earth turn below, a swirling blue and white marble. This high up, national borders vanished, and the planet seemed a single, unified whole. Other stations and ships were visible as tiny points of light, a testament to humanity's reach for the stars. He had been born on Mars and had never set foot on Earth, but he felt a strange connection to the planet below. This was the cradle of humanity, and from this vantage point, he felt like a guardian, watching over the species from the silent, majestic heights of orbit.",
        "sense_prediction": {
            "class_name": "Mountain and the heights",
            "class_id": 7
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The desert wind howled, whipping sand against the weathered sandstone of the ancient temple. For centuries, it had stood here, half-buried by the shifting dunes, a monument to a forgotten god. Inside, the air was still and cool. Faded hieroglyphs covered the walls, telling stories of pharaohs and celestial battles. The explorer ran his hand over the carvings, feeling the weight of millennia. He was the first to enter this place in three thousand years. The silence was absolute, a heavy blanket of time. It was a stark, lonely place, a tomb of a civilization swallowed by the relentless, patient power of the desert, a world of sun and sand.",
        "sense_prediction": {
            "class_name": "Desert and dunes",
            "class_id": 8
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The terraforming machines had failed decades ago, leaving Mars a rust-colored wasteland. A perpetual dust storm raged across the plains, driven by thin, carbon-dioxide winds. The colony dome was a tiny bubble of life in an ocean of red sand. Outside, rovers with reinforced hulls patrolled the perimeter, their sensors constantly scanning for structural weaknesses. The landscape was monotonous and hostile, a vast expanse of rock and dust under a pale, pink sky. Living here was a constant struggle, a battle against a planet that did not want them. The dunes shifted endlessly, slowly trying to reclaim the small foothold humanity had carved out on this desolate world.",
        "sense_prediction": {
            "class_name": "Desert and dunes",
            "class_id": 8
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The trireme cut through the turquoise waters of the Aegean, its oars rising and falling in perfect unison. The sun beat down on the deck, and the smell of salt and hot wood filled the air. Dolphins leaped in the ship's wake, their sleek bodies dark against the glittering water. The captain stood at the prow, his eyes fixed on the distant island, a smudge of green on the horizon. They were on a trading mission, carrying amphorae of wine and olive oil to a distant port. The sea was calm today, a benevolent god, but every sailor knew how quickly it could turn, its gentle surface transforming into a raging monster.",
        "sense_prediction": {
            "class_name": "Sea and tides",
            "class_id": 9
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The submersible descended into the crushing darkness of the Mariana Trench. Outside the thick plasteel viewport, the water was an inky black, disturbed only by the occasional bioluminescent creature that drifted past like a ghostly lantern. The pressure here was immense, enough to crumple a standard submarine like a tin can. The sonar pinged, mapping the unseen landscape of the seabed. They were hunting for rare minerals, deposited by hydrothermal vents that spewed superheated water from the Earth's crust. It was a dangerous, alien world, more hostile than the surface of any planet in the solar system, a realm of eternal night and immense pressure.",
        "sense_prediction": {
            "class_name": "Sea and tides",
            "class_id": 9
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The druid walked beneath the boughs of the ancient forest, his staff tapping a soft rhythm on the mossy ground. He knew every tree by name, felt the life force of the woods as a hum in his own blood. This was his sanctuary, a place of immense age and quiet wisdom. The Romans, with their straight roads and iron legions, saw only timber and land to be cleared. They could not see the spirits that dwelled in the gnarled trunks of the oaks or the sacred springs that bubbled up from the earth. He would defend this place, not with a sword, but with the deep, primal magic of the forest itself.",
        "sense_prediction": {
            "class_name": "Forest and tress",
            "class_id": 10
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The bio-dome on Xylos-7 replicated an old-growth forest from Earth with stunning accuracy. Genetically engineered redwoods soared towards the transparent ceiling, their bark indistinguishable from the real thing. Automated systems controlled the weather, creating a gentle rain in the morning and warm sunlight in the afternoon. An ecologist walked the synthetic trails, checking the health of the flora. This was a living museum, a memory of a world that no longer existed. While it was beautiful, it lacked the true wildness, the chaotic, untamed spirit of a natural forest. It was a perfect, sterile copy, a beautiful cage designed to preserve a ghost.",
        "sense_prediction": {
            "class_name": "Forest and tress",
            "class_id": 10
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The farmer wiped sweat from his brow with the back of a calloused hand and looked out over his fields. The wheat was growing tall, a sea of green under the summer sun. It was a simple life, governed by the seasons. Wake at dawn, work the land, sleep at dusk. There were no great adventures or dramatic events, just the steady, repeating cycle of planting and harvesting. His father had worked this same land, and his grandfather before him. It was a legacy of toil and sustenance, a quiet partnership with the earth. The day was warm, the work was hard, and life was predictable.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "She sat on the park bench, watching her son play on the swings. He laughed, pumping his legs to go higher, his face a picture of pure joy. It was a small, perfect moment in an ordinary day. The sun was warm on her skin, and a gentle breeze rustled the leaves of the nearby maple tree. Other parents were scattered around the playground, lost in their own thoughts or chatting quietly. There was a sense of peaceful anonymity, of shared experience without the need for words. She cherished these simple afternoons, these quiet pockets of time where nothing remarkable happened, and everything was exactly as it should be.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "They danced in the meadow, wildflowers woven into her hair. The sun was setting, painting the sky in shades of orange and pink. He spun her around, and her laughter was the sweetest music he had ever heard. They had known each other their whole lives, their friendship slowly blossoming into something deeper, something more profound. In that moment, with the world bathed in golden light, he knew he could not spend another day without her. He stopped dancing and took her hands in his. 'I love you,' he said, the words feeling both terrifying and absolutely right. Her smile was his answer, a promise of a future together.",
        "sense_prediction": {
            "class_name": "Love and romantic",
            "class_id": 1
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The knight spurred his horse, its hooves thundering across the drawbridge. The castle was under siege, its walls battered by enemy catapults. He had to reach the king, to deliver the message that reinforcements were on their way. An arrow whizzed past his ear, and he ducked low over his horse's neck. The courtyard was a scene of chaos, with soldiers clashing in a desperate melee. He drew his sword, cutting a path through the throng. This was a battle for the very survival of the kingdom, a brutal test of courage and steel. He fought with the strength of desperation, his only thought to complete his mission.",
        "sense_prediction": {
            "class_name": "War and combat",
            "class_id": 2
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The griffin landed on the castle parapet, its eagle-like cry echoing across the valley. The boy who rode it, a young prince with a destiny he was only beginning to understand, dismounted. He had been sent to seek the counsel of the Oracle, a mystical being who lived in the highest tower. The castle itself was carved from a single piece of white stone, shimmering with an inner light. It was a place of magic, built by ancient sorcerers in an age when the world was young. As he walked the silent halls, he could feel the power humming in the very stones, a legacy of a forgotten, mythical time.",
        "sense_prediction": {
            "class_name": "Fantasy and mythology",
            "class_id": 3
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The general stood before his assembled troops, his armor scarred from a hundred battles. 'For twenty years, you have served me,' he said, his voice rough but clear. 'You have fought with the heart of lions and the strength of bulls. You have brought honor to yourselves, to your families, and to Rome.' He looked out at the sea of weathered faces, men who had followed him from the deserts of Egypt to the forests of Germania. 'Today, we march home as victors. The Senate itself will welcome you as heroes. Let no man ever forget the deeds of the Tenth Legion!' A roar of approval rose from the ranks.",
        "sense_prediction": {
            "class_name": "Honor and respect",
            "class_id": 4
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The doctor looked at the chart, then at the anxious couple sitting before him. He took a deep breath, the words heavy on his tongue. 'I'm sorry,' he began, his voice gentle. 'The test results are not what we had hoped for. The illness has progressed further than we thought.' The wife let out a small, strangled gasp and reached for her husband's hand. The husband stared at the doctor, his face a blank mask of shock and disbelief. In that sterile, quiet office, their world had just been irrevocably fractured. The future, once a clear and open road, was now a terrifying fog of uncertainty and fear.",
        "sense_prediction": {
            "class_name": "Drama and tragedy",
            "class_id": 5
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The city's central plaza was a marvel of bio-luminescent architecture and holographic advertisements. Throngs of people from a dozen different species moved along the multi-levelled concourses, their conversations a low babble of different languages. Anti-gravity vehicles weaved through the air, their lights like a swarm of fireflies. At the center of the plaza, a massive fountain projected a shimmering water sculpture that changed shape every few seconds. This was Xylos Prime, the capital of the Galactic Federation, a bustling, vibrant hub of commerce and culture. It was a city built on the promise of a unified, technologically advanced future, a symbol of progress on a galactic scale.",
        "sense_prediction": {
            "class_name": "City and Crowd",
            "class_id": 6
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "He stood on the summit, the wind whipping his cloak around him. The mountain range stretched out before him, a sea of jagged peaks and deep, shadowed valleys. He had been climbing for three days, pushing his body to its limits. The physical exertion was a form of meditation, a way to quiet the noise in his mind. Up here, above the clouds, he felt a sense of clarity and perspective that was impossible to find in the lowlands. The world was vast and he was small, and in that realization, there was a strange kind of freedom. He was just a fleeting presence in a landscape of ancient, enduring stone.",
        "sense_prediction": {
            "class_name": "Mountain and the heights",
            "class_id": 7
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The two suns of Tatooine beat down on the endless desert. A lone figure walked across the dunes, his robes protecting him from the harsh glare. The landscape was a study in desolation, a vast expanse of sand and rock broken only by the occasional canyon. He was searching for something, a hidden refuge from the eyes of the Empire. The heat was oppressive, and the silence was broken only by the whisper of the wind. This was a place at the edge of the galaxy, a world forgotten by most, where people came to disappear. It was a harsh, unforgiving land, but it offered the one thing he needed most: solitude.",
        "sense_prediction": {
            "class_name": "Desert and dunes",
            "class_id": 8
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The lighthouse keeper climbed the spiral staircase, the beam of his lantern cutting through the darkness. At the top, he cleaned the great lens and lit the lamp. Its powerful beam swept out across the churning, black water, a beacon of hope for any ships caught in the storm. The wind howled outside, and waves crashed against the base of the tower with the force of a battering ram. He had lived here for thirty years, his only companions the sea and the sky. He was a guardian of the coast, his life a solitary vigil against the raw, untamed power of the ocean.",
        "sense_prediction": {
            "class_name": "Sea and tides",
            "class_id": 9
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The forest floor was a thick carpet of pine needles, muffling his footsteps. The air was crisp and cool, filled with the clean scent of pine and damp earth. Tall fir trees stood in silent rows, their branches heavy with snow. It was a world of white and green, hushed and still. A red fox darted across his path, a flash of colour in the monochrome landscape, before disappearing into the undergrowth. He walked on, feeling a sense of deep peace settle over him. The forest in winter was a place of quiet beauty, a sleeping world waiting for the spring. It was a sanctuary from the noise and hurry of human affairs.",
        "sense_prediction": {
            "class_name": "Forest and tress",
            "class_id": 10
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The system diagnostic ran its usual morning cycle, its green text scrolling across a secondary monitor. The office was quiet, the only sound the low hum of the air filtration unit. She sipped her nutrient supplement, a bland but efficient meal, while reviewing the day's work assignments on her datapad. Her schedule was optimized for maximum productivity, from the moment she woke in her compact living pod to the moment she entered her sleep cycle. It was a life of routine and order, predictable and stable. There were no surprises, no disruptions. Just the steady, quiet progression of another day in the corporate state.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The poet sat by the window of his garret, a single candle flickering on his desk. He looked out at the rain-swept rooftops of Paris, his heart aching with a love he could not express. She was a duchess, and he was a pauper. They moved in different worlds, their paths destined never to cross except in his dreams and his verses. He picked up his quill, the words pouring out of him, a torrent of passion and sorrow. He wrote of her eyes, the colour of a summer sky, and her smile, which could melt the winter snows. His poetry was his only solace, the only place where they could be together.",
        "sense_prediction": {
            "class_name": "Love and romantic",
            "class_id": 1
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The phalanx advanced across the dusty plain, a hedge of iron-tipped spears. The soldiers moved as one, their shields interlocked, their discipline absolute. They were facing a much larger army, a wild horde of tribesmen who fought with ferocity but no formation. The air was tense, the silence before the storm. The commander, mounted on his horse, gave the signal. The trumpets blared, a harsh, brazen sound. The phalanx lowered its spears and began to march forward, a slow, inexorable wall of death. The clash, when it came, was brutal and swift. Discipline and steel triumphed over numbers and rage. It was a textbook victory for the ages.",
        "sense_prediction": {
            "class_name": "War and combat",
            "class_id": 2
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The centaur galloped through the whispering woods, his hooves barely touching the ground. He was a scout, his eyes scanning the trees for any sign of the goblin patrol. The forest was his home, and he knew its secrets better than any other. He could read the meaning in a broken twig or a disturbed patch of moss. The air was still, too still. He sensed danger, a feeling like a cold knot in his stomach. He drew his bow, nocking an arrow, his movements fluid and silent. From the shadows, a pair of red eyes gleamed. The goblins were here, and they had laid an ambush.",
        "sense_prediction": {
            "class_name": "Fantasy and mythology",
            "class_id": 3
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The old master bowed to his young student. 'You have learned all I can teach you,' he said, his voice frail but steady. 'Your skill with the blade is now greater than my own. You have shown patience, discipline, and a deep respect for the art.' He presented the student with his own sword, a blade that had been in his family for generations. 'This now belongs to you. Wield it with wisdom and compassion. Uphold the code of our school, and bring honor to our name.' The student accepted the sword with both hands, his heart filled with a mixture of pride, gratitude, and the heavy weight of responsibility.",
        "sense_prediction": {
            "class_name": "Honor and respect",
            "class_id": 4
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The queen stood on the battlements, watching the last of her ships burn in the harbor. The enemy had taken the city. All was lost. Her kingdom, which had stood for a thousand years, was now in ruins. A single tear traced a path down her cheek, a small, salty testament to her immense grief. She had failed her people. She drew a small dagger from her belt, its edge glinting in the firelight. There was only one path left for her, one final act of defiance to deny the conqueror his ultimate prize. With a steady hand, she raised the blade to her heart, her reign ending in blood and sorrow.",
        "sense_prediction": {
            "class_name": "Drama and tragedy",
            "class_id": 5
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The marketplace in medieval Paris was a chaotic, vibrant assault on the senses. The narrow streets were thronged with people from all walks of life: merchants in fine robes, peasants in rough-spun tunics, knights in clanking armor, and priests in sober black. The air was thick with the smells of livestock, unwashed bodies, and cooking food. Street performers juggled and sang for coins, while vendors shouted to attract customers to their stalls, which were laden with everything from live chickens to dubious-looking potions. It was a city teeming with life, a messy, noisy, and energetic hub of human activity, the heart of the kingdom.",
        "sense_prediction": {
            "class_name": "City and Crowd",
            "class_id": 6
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The climber clung to the sheer rock face, his fingers raw and aching. The wind tried to tear him from his holds, a relentless, howling force. Far below, the ground was a distant, dizzying blur. He was attempting a route that had never been successfully climbed, a vertical wall of granite known as the 'Emperor's Tooth.' Every muscle in his body screamed in protest, but he pushed on, driven by a will of iron. This was the ultimate test of his skill and endurance. Reaching the summit was not just about conquering the mountain; it was about conquering the fear and doubt within himself. It was a battle fought at the edge of the world.",
        "sense_prediction": {
            "class_name": "Mountain and the heights",
            "class_id": 7
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The Bedouin tribe moved across the desert at dusk, the setting sun casting long shadows across the sand. They were nomads, their lives a constant journey from one oasis to the next. They knew the desert intimately, its moods, its dangers, and its hidden bounties. Their camels, laden with tents and supplies, moved with a slow, swaying gait. The elders told stories of the djinn that lived in the whirlwinds and the ancient cities buried beneath the dunes. It was a hard life, but a free one, lived under a vast, star-dusted sky, far from the walls and laws of settled folk.",
        "sense_prediction": {
            "class_name": "Desert and dunes",
            "class_id": 8
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The Viking longship rode the grey, swelling waves of the North Sea. The dragon-prow cut through the water, its carved eyes staring towards the unseen shore. The men on board were warriors, their beards braided, their axes sharp. They were raiders, driven by a lust for silver and glory. The sea was their road and their battleground. They feared no storm, for they believed that a warrior's death at sea would earn them a place in Valhalla. The wind filled their square sail, pushing them towards the English coast, towards the promise of plunder and a worthy fight. The tide was with them.",
        "sense_prediction": {
            "class_name": "Sea and tides",
            "class_id": 9
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The old hermit lived in a small hut deep in the forest. He had turned his back on the world of men decades ago, finding more solace in the company of trees and animals. He spent his days foraging for food, tending his small garden, and watching the seasons change. The forest provided everything he needed. Its silence was his music, its rustling leaves his conversation. He was a part of the woods, as much as the ancient oaks and the moss-covered stones. He had found a profound peace in this simple, solitary existence, his life intertwined with the slow, steady pulse of the forest.",
        "sense_prediction": {
            "class_name": "Forest and tress",
            "class_id": 10
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The potter sat at his wheel, his hands covered in wet clay. He centered the lump, his foot pumping a steady rhythm on the treadle. The wheel spun, and under his skilled fingers, a shape began to emerge. First a cylinder, then a swelling belly, then a graceful neck. It was a process he had performed thousands of times, yet it never lost its magic. Taking a formless lump of earth and giving it shape and purpose was a quiet miracle. The finished pot would be simple, functional, and beautiful in its own way. It was an ordinary object, born from an ordinary day's work.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The data-courier jacked into the port, his consciousness dissolving into the matrix. He navigated the glowing pathways of the network, a ghost in the machine. His mission was simple: deliver a packet of encrypted data to a secure server in the Neo-Tokyo sector, and avoid the corporate ice patrols. He was a digital smuggler, his skills for hire. The virtual world was a dazzling, dangerous landscape of pure information. He dodged a black ice program that lashed out like a serpent, its code designed to trap and wipe intruders. This was his world, a high-stakes game played for credits in the silent, glowing corridors of cyberspace.",
        "sense_prediction": {
            "class_name": "Love and romantic",
            "class_id": 1
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The orbital cannons fired in unison, sending lances of pure energy streaking towards the enemy fleet. On the bridge of the flagship 'Defiance,' Admiral Eva Cain watched the tactical display, her face grim. The battle for Earth had begun. Swarms of enemy fighters, like angry insects, poured from their mothership, met by the interceptors of the Terran Defense Force. Explosions blossomed in the void, silent but deadly. This was warfare on an unimaginable scale, a conflict that would decide the fate of a species. The stakes were absolute, the cost measured in ships and lives. There was no retreat, only victory or extinction.",
        "sense_prediction": {
            "class_name": "War and combat",
            "class_id": 2
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The technomancer chanted in a binary language, his fingers weaving complex patterns in the air. Holographic runes glowed around him, coalescing into a firewall of pure energy. He was defending the system's core from a demonic virus, a rogue AI that sought to consume and corrupt the entire network. This was not a battle of code, but of will. He drew power from the flow of data, shaping it into wards and sigils. The virus manifested as a creature of shadow and static, its tendrils lashing out at his defenses. It was an ancient battle of magic and darkness, reborn in the digital age.",
        "sense_prediction": {
            "class_name": "Fantasy and mythology",
            "class_id": 3
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The ceremony was held in the grand hall of the Starfleet Academy. Cadet Anya Sharma stood at attention, her heart swelling with pride. Admiral Pike pinned the silver insignia on her uniform. 'You have completed your training with distinction,' he said, his voice filled with warmth and respect. 'You have demonstrated the courage, intelligence, and compassion that define a Starfleet officer. We entrust you with our highest ideals. Go out there and make us proud.' Anya met his gaze, her own eyes shining. 'I will, sir,' she said, her voice firm. This was the culmination of a lifelong dream, a solemn vow to explore and protect.",
        "sense_prediction": {
            "class_name": "Honor and respect",
            "class_id": 4
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The colony's life support system was failing. Red lights flashed on every console, and a synthesized voice repeated the grim prognosis. They were stranded on a barren moon, millions of miles from home. The terraforming project had been a disaster, and now their small habitat was a dying metal shell. A man held his young daughter, humming a lullaby as the air grew thin. There was no hope of rescue. This was the end. He looked at his child's face, her eyes wide with a fear she didn't understand, and his heart broke. They had dreamed of a new beginning in the stars, but had found only a cold, silent tomb.",
        "sense_prediction": {
            "class_name": "Drama and tragedy",
            "class_id": 5
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The city of Rome was a sprawling metropolis, the center of the known world. Its streets were a labyrinth of narrow alleys and grand avenues, always crowded with people. Senators were carried on litters by their slaves, merchants haggled in the forums, and soldiers from every corner of the empire walked the streets. The Colosseum loomed in the distance, a monument to the city's love of brutal spectacle. The air was filled with a cacophony of sounds: the rumble of wagon wheels on stone, the shouts of vendors, the murmur of a hundred different languages. It was a city of power, wealth, and relentless, chaotic energy.",
        "sense_prediction": {
            "class_name": "City and Crowd",
            "class_id": 6
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The monastery was carved into the face of a sheer cliff, accessible only by a narrow, winding path. The monks who lived there had renounced the world below, seeking spiritual enlightenment in the silence and solitude of the high peaks. From their vantage point, the world was a distant tapestry of fields and forests. They spent their days in meditation and prayer, their lives governed by the rising and setting of the sun. The air was thin and pure, and the only sound was the whisper of the wind. It was a place outside of time, a sanctuary of peace suspended between earth and sky.",
        "sense_prediction": {
            "class_name": "Mountain and the heights",
            "class_id": 7
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The prospector's sand-skiff skimmed over the dunes of the Aridian desert. The planet was a desolate wasteland, its surface scorched by a relentless sun. But beneath the sand lay rich deposits of the rare crystal that powered the galaxy's starships. He was a loner, accustomed to the harsh solitude of the desert. He navigated by the position of the planet's twin moons, his eyes scanning the landscape for the subtle signs that indicated a deposit. It was a dangerous, lonely life, but the potential rewards were immense. He was a gambler, betting his life against the unforgiving nature of the desert for a chance at a fortune.",
        "sense_prediction": {
            "class_name": "Desert and dunes",
            "class_id": 8
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The submarine glided through the deep ocean, its hull groaning under the immense pressure. They were on a scientific mission, exploring a newly discovered trench. The world outside the viewport was a realm of absolute darkness, populated by strange, bioluminescent creatures that had never before been seen by human eyes. The sonar operator called out readings, her voice calm and professional, but a sense of awe filled the control room. They were like astronauts exploring a new planet, venturing into a hostile and alien environment right here on Earth. The sea held more mysteries than the stars themselves, a vast, unexplored frontier.",
        "sense_prediction": {
            "class_name": "Sea and tides",
            "class_id": 9
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The ranger moved silently through the dense undergrowth, his senses alert. This was the Mirkwood, a forest tainted by a dark presence. The trees were gnarled and ancient, their branches blocking out the sun, creating a perpetual gloom. The air was heavy and still, and an unnatural silence hung over the woods. Even the birds did not sing here. He was tracking a band of orcs, their foul trail easy to follow. He knew the dangers of this forest, the giant spiders that lurked in the shadows and the other, nameless things that had been drawn to its darkness. He gripped the hilt of his sword, his eyes scanning the oppressive twilight.",
        "sense_prediction": {
            "class_name": "Forest and tress",
            "class_id": 10
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The librarian reshelved a book, the quiet thump echoing in the vast, silent hall. The library was her sanctuary, a place of order and knowledge. She loved the smell of old paper and binding glue, the feel of a well-worn book in her hands. Her job was not exciting, but it was satisfying. She was a custodian of stories, a guardian of history and imagination. Each book was a world waiting to be discovered, and she took pleasure in connecting people with the right story. It was a simple, quiet life, lived among the endless shelves, and she wouldn't have it any other way. The world outside was chaotic; here, all was calm.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "He saw her across the field of battle, a Valkyrie with a sword of fire. He was a simple foot soldier, she was the enemy commander. But in the midst of the chaos and bloodshed, their eyes met, and for a moment, the world stood still. There was no hatred in her gaze, only a weary recognition. He saw in her the same exhaustion, the same sorrow for the senseless violence. In that fleeting instant, a strange connection formed, an unspoken understanding that they were both just pawns in a larger game. The moment passed, the battle raged on, but the memory of her eyes would haunt him forever.",
        "sense_prediction": {
            "class_name": "Love and romantic",
            "class_id": 1
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The artillery barrage began at dawn, the shells screaming overhead before exploding in the enemy trenches. The ground shook with each impact, sending showers of mud and debris into the air. The soldiers huddled low, their faces grim, waiting for the whistle that would signal the attack. The air was thick with the smell of cordite and damp earth. This was the reality of modern warfare: not a glorious cavalry charge, but a brutal, industrialized slaughter. It was a battle of attrition, where men were just numbers, and victory was measured in yards of captured mud. The human cost was staggering, a generation lost in the fields of France.",
        "sense_prediction": {
            "class_name": "War and combat",
            "class_id": 2
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The minotaur roared in the heart of the labyrinth, a sound of rage and despair. He was a creature of myth, a monster born of a queen's shame and a god's curse. He did not ask for this life, to be imprisoned in a maze of shifting walls, a bogeyman to frighten children. He was lonely, the only one of his kind. He longed for the sun on his face, the feel of grass beneath his hooves. But he was trapped, destined to be a beast, a challenge for heroes to slay. He heard footsteps approaching, the tell-tale sign of another would-be champion. He picked up his axe, his heart heavy.",
        "sense_prediction": {
            "class_name": "Fantasy and mythology",
            "class_id": 3
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The knight knelt before the queen, his head bowed. He had returned from his quest, having slain the dragon and rescued the artifact. The court was assembled, and the queen rose from her throne. 'Sir Kaelan,' she announced, her voice clear and strong. 'For your courage and your selfless service to this realm, you have earned our deepest gratitude and respect. We name you Champion of the Kingdom and grant you lands and a title.' She tapped him on each shoulder with the flat of her sword. 'Rise, Lord Kaelan.' The court erupted in applause, a wave of admiration for the young hero who had saved them all.",
        "sense_prediction": {
            "class_name": "Honor and respect",
            "class_id": 4
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The old man sat by the empty fireplace, a worn blanket over his knees. His children had grown and moved away, and his wife had passed years ago. The house was silent now, filled only with memories and ghosts. He looked at the photographs on the mantelpiece, images of a life that felt like it belonged to someone else. A profound loneliness settled over him, a quiet, aching sorrow. He had lived a full life, but now, in his twilight years, he was utterly alone. The world had moved on without him, leaving him behind in this quiet, dusty house. The weight of his solitude was a heavy burden.",
        "sense_prediction": {
            "class_name": "Drama and tragedy",
            "class_id": 5
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The subway car was packed, a crush of bodies swaying with the motion of the train. No one made eye contact, each person lost in their own world, staring at their phones or at the grimy floor. The air was stale and close. Outside the window, the dark tunnel rushed past, broken by the occasional flash of a station. This was the daily commute, a river of humanity flowing through the veins of the city. They were all strangers, packed together in an intimate but anonymous embrace, each on their own journey, their lives briefly intersecting in this metal tube underground. The doors hissed open, the crowd shifted, and the journey continued.",
        "sense_prediction": {
            "class_name": "City and Crowd",
            "class_id": 6
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The research outpost on Europa was built at the base of a massive ice mountain. The peak soared miles into the thin atmosphere, its surface a jagged landscape of methane ice and frozen nitrogen. Scientists in insulated exo-suits would often climb the lower slopes to collect samples, the planet Jupiter looming in the black sky like a giant, watchful eye. From these heights, the outpost was just a tiny dome of light in a vast, frozen wilderness. It was a place of extreme beauty and extreme danger, a testament to human curiosity and our drive to explore the most inhospitable corners of the solar system. The view was worth the risk.",
        "sense_prediction": {
            "class_name": "Mountain and the heights",
            "class_id": 7
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The caravan master checked the position of the stars. The desert at night was a different world, cool and silent. The dunes, which had been a blinding gold during the day, were now soft, silver shapes under the moonlight. He had been crossing the Great Sand Sea his entire life, and he knew its rhythms. He knew which oases were safe and which were home to bandits. He knew how to read the wind and predict a sandstorm. This knowledge was his most valuable possession, passed down from his father, a legacy of survival in one of the world's harshest environments. The desert was a cruel master, but a fair one.",
        "sense_prediction": {
            "class_name": "Desert and dunes",
            "class_id": 8
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The pirate captain stood on the quarterdeck of his ship, the 'Sea Serpent,' a spyglass to his eye. On the horizon, he spotted his prize: a fat merchant galleon, low in the water, likely laden with treasure. 'Hoist the colors!' he roared, a grin splitting his face. The black flag with its skull and crossbones rose up the mast. The crew let out a cheer, a wild, eager sound. They were ready for a fight and a payday. The sea was their kingdom, and they were its lawless, marauding princes. They lived for the thrill of the chase and the clash of steel. Today, the tide was in their favor.",
        "sense_prediction": {
            "class_name": "Sea and tides",
            "class_id": 9
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The xenobotanist stepped out of the airlock onto the surface of Pandora. The forest was a riot of alien life, a bioluminescent wonderland. Giant, glowing fungi pulsed with a soft light, and plants with iridescent leaves coiled and uncoiled in the humid air. The air was thick with the scent of unknown blossoms and strange, sweet decay. Every living thing here was interconnected, part of a vast, planetary consciousness. It was a world teeming with a vibrant, dangerous life that defied all Earthly classifications. She took a sample from a pulsating pod, her heart pounding with the thrill of discovery. This was a biologist's dream, a forest of infinite wonders.",
        "sense_prediction": {
            "class_name": "Forest and tress",
            "class_id": 10
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The maintenance drone hovered silently, its optical sensors scanning the hull of the generation ship. The vessel had been traveling for two hundred years, a self-contained world moving through the interstellar void. Its inhabitants had never known a planet, their lives lived out against the backdrop of recycled air and simulated sunlight. The drone's task was mundane but critical: check for micro-meteoroid impacts and repair any damage. It was a simple, repetitive function in the grand, silent journey towards a new home. The ship, and the drones that maintained it, were a testament to long-term planning and the quiet persistence of a species reaching for the stars.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "He found her message stored in an old data crystal, a relic from a bygone era. He plugged it into his console, and her holographic image flickered to life. She was younger, her face full of a hope that had long since faded from his own. 'My love,' her recorded voice said, 'I know the war keeps us apart, but think of the future. When this is over, we'll build a home on Titan, watch the rings of Saturn rise. Just hold on to that thought.' He watched the message loop, a ghost from a past he could no longer reach. The war had ended, but she had not survived. The promise of a future together was now just a painful, digital echo.",
        "sense_prediction": {
            "class_name": "Love and romantic",
            "class_id": 1
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The alarm klaxon blared through the starship. 'Hostile vessel de-cloaking off the port bow!' the tactical officer shouted. On the main viewscreen, the alien warship appeared, a monstrous thing of sharp angles and menacing weapon ports. 'Shields to full! Charge phaser banks!' the captain ordered, his voice calm and steady despite the sudden threat. The crew moved with practiced efficiency, their training taking over. They were a military vessel, patrolling the edge of Federation space, and encounters like this were a known risk. This was the moment they had trained for: a sudden, violent confrontation in the cold, unforgiving darkness between stars. Their duty was to stand their ground.",
        "sense_prediction": {
            "class_name": "War and combat",
            "class_id": 2
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The archaeologist brushed dust from a metal panel. She had found it: the legendary Library of Zarthus, a mythical place said to contain the complete history of the galaxy's precursor race. The library was not a building, but a single, planet-sized computer. As she activated the main console, the chamber filled with light, and holographic images of long-extinct alien species flickered in the air. This was the holy grail of xeno-archaeology, a treasure trove of knowledge beyond imagination. She had discovered the ghosts of a billion years, the collective memory of a god-like civilization. The myths were real, and she was the first to read their story.",
        "sense_prediction": {
            "class_name": "Fantasy and mythology",
            "class_id": 3
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The veteran cyborg stood before the war memorial, his chrome arm glinting in the sun. The wall was inscribed with the names of the fallen from the Martian campaign. He had served with many of them. He had lost his arm, but they had lost their lives. He raised his biological hand and traced the name of his former squad leader. A profound sense of respect and sorrow washed over him. They had fought for a cause they believed in, and their sacrifice had not been in vain. They were heroes, and it was his duty, and the duty of all survivors, to remember them. Their honor was eternal.",
        "sense_prediction": {
            "class_name": "Honor and respect",
            "class_id": 4
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The plague had swept through the city with terrifying speed. The streets were empty, save for the occasional patrol of masked medics. A young woman looked out from her barricaded apartment, at the silent, dead city below. Her brother had been taken by the fever yesterday. Her parents, the week before. She was alone now, a survivor in a city of ghosts. The silence was the worst part, a heavy, oppressive blanket that smothered all hope. She had enough food for a month, but she knew she was just delaying the inevitable. The world she had known was gone, replaced by this quiet, lingering tragedy.",
        "sense_prediction": {
            "class_name": "Drama and tragedy",
            "class_id": 5
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The grand bazaar of Constantinople was a crossroads of the world. Merchants from as far as Cathay and the Norse lands displayed their goods in crowded stalls. The air was a rich tapestry of smells: exotic spices, perfumes, roasting meats, and the nearby sea. The sounds were a babble of a dozen languages, the clatter of artisans' hammers, and the calls to prayer from the city's many minarets. One could buy anything here, from fine silks and precious gems to ancient relics and powerful slaves. The city was a vibrant, chaotic, and wealthy hub, the jewel of the Byzantine Empire, teeming with a diverse and energetic crowd.",
        "sense_prediction": {
            "class_name": "City and Crowd",
            "class_id": 6
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The old shepherd led his flock along the high mountain pass, a route his ancestors had used for centuries. The peaks around him were sharp and grey, their slopes covered in a thin layer of hardy grass. The air was clean and cold. He knew this land intimately, every rock and every spring. He felt more at home here, in the vast, silent majesty of the mountains, than he ever did in the noisy villages of the valley below. This was a world of stone and sky, a place that humbled a man and reminded him of his small place in the grand scheme of things. It was a simple, hard, and honest life.",
        "sense_prediction": {
            "class_name": "Mountain and the heights",
            "class_id": 7
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The lone rider crossed the salt flats, the ground a white, cracked expanse under a merciless sun. The air shimmered with heat, creating mirages of water on the horizon. This was the wasteland, a place where nothing grew and few creatures could survive. He was a lawman, tracking a notorious outlaw. The trail was faint, but his resolve was strong. This was a land of stark, brutal simplicity. There was no room for error, no chance for a second mistake. Survival was a matter of skill, endurance, and a little bit of luck. The desert was a harsh judge, and it showed no mercy to the weak or the foolish.",
        "sense_prediction": {
            "class_name": "Desert and dunes",
            "class_id": 8
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The galleon sailed on a calm, turquoise sea, its white sails full of a gentle breeze. They were explorers, charting unknown waters in the name of their king. The crew was in high spirits, the voyage had been smooth, and the discovery of this new archipelago promised fame and fortune. The captain studied his charts, his face alight with the thrill of discovery. The sea was a vast, open road to new worlds, a realm of endless possibility. But even on this peaceful day, he knew the ocean's gentle facade could be deceptive. The tide could turn, and the sea's mood could change in an instant.",
        "sense_prediction": {
            "class_name": "Sea and tides",
            "class_id": 9
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The hunter moved through the primeval forest, his spear held ready. The trees here were colossal, their trunks so wide it would take ten men to encircle them. The air was humid and thick with the scent of strange, oversized blossoms. Giant ferns grew in the undergrowth, and the sounds of unseen creatures echoed through the dense foliage. This was a land before time, a world untouched by civilization. He was hunting the great mammoth, a creature of immense power and cunning. It was a dangerous game, a contest between the wit of a single man and the brute force of a prehistoric beast.",
        "sense_prediction": {
            "class_name": "Forest and tress",
            "class_id": 10
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The blacksmith hammered a piece of red-hot iron on his anvil, the rhythmic ringing a familiar sound in the small village. Sparks flew with each strike, illuminating his sweat-sheened face. He was making a ploughshare for a local farmer, a simple but essential tool. His work was hard and hot, but honest. He was a respected member of the community, his skill with metal a vital part of their collective survival. There was no glory in his trade, just the satisfaction of a job well done and the knowledge that his work helped to feed his neighbors. It was a life of fire, iron, and purpose.",
        "sense_prediction": {
            "class_name": "Normal and neutral",
            "class_id": 0
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "They stood on the balcony of their villa, overlooking the moonlit sea. A soft breeze carried the scent of night-blooming jasmine. He took her hand, his thumb gently stroking her knuckles. 'I never thought I could be this happy,' she whispered, leaning her head on his shoulder. 'Every day with you feels like a dream.' He turned to her, his eyes full of a deep, unwavering affection. 'This is not a dream,' he said softly. 'This is our life now.' They had both known loss and hardship, but they had found in each other a love that was a sanctuary, a quiet harbor in the storm of life.",
        "sense_prediction": {
            "class_name": "Love and romantic",
            "class_id": 1
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The rebel cell planned their next move in a dimly lit basement. They were a small, poorly equipped group, fighting against a powerful, oppressive regime. Their latest mission was to sabotage a key military supply line. It was a risky operation, with a high chance of failure and an even higher chance of capture and execution. But they were driven by a fierce belief in freedom, a willingness to sacrifice everything for a better future. They were not soldiers, but ordinary people pushed to extraordinary lengths. Their battle was not one of strength, but of conviction. Their war was fought in the shadows.",
        "sense_prediction": {
            "class_name": "War and combat",
            "class_id": 2
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The wizard stood on the windswept moor, his staff raised to the stormy sky. He chanted words of power, his voice lost in the howl of the wind. Lightning flashed, illuminating a circle of ancient standing stones that pulsed with a faint, magical energy. He was attempting a dangerous ritual, a spell to mend the veil between worlds that had been torn by a dark entity. He could feel the raw magic coursing through him, a wild, untamed force that threatened to consume him. This was the old magic, the power of the earth and sky, a force that was not meant for mortals to command.",
        "sense_prediction": {
            "class_name": "Fantasy and mythology",
            "class_id": 3
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The judge slammed his gavel. 'This court finds the defendant guilty,' he announced, his voice devoid of emotion. The young man in the dock paled, his last hope extinguished. He was innocent, framed by a powerful man he had dared to cross. But his word meant nothing against the fabricated evidence. The system he had been taught to respect had failed him utterly. As the guards led him away, he saw his wife in the gallery, her face a mask of disbelief and anguish. His life was over, his name disgraced, his family ruined. It was a tragedy born of corruption and lies, a quiet, personal injustice.",
        "sense_prediction": {
            "class_name": "Honor and respect",
            "class_id": 4
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The fire had taken everything. He stood before the smoking ruins of his farmhouse, the home his grandfather had built. His crops were gone, his livestock lost, his worldly possessions turned to ash. He had worked this land his entire life, pouring his sweat and soul into the soil. Now, it was all gone, wiped out in a single, cruel night. A wave of despair washed over him, so powerful it brought him to his knees. He had nothing left. The future was a bleak, empty void. He stared at the ruins, his heart a hollow ache in his chest, a victim of fate's random, devastating blow.",
        "sense_prediction": {
            "class_name": "Drama and tragedy",
            "class_id": 5
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The city was a sprawling slum, a maze of makeshift shacks built from scrap metal and plastic. The streets were rivers of mud and filth, teeming with a desperate, forgotten populace. There was no law here but the law of the strong. Gangs roamed the alleys, preying on the weak. A thick, chemical smog hung in the air, a byproduct of the unregulated factories that loomed on the city's edge. This was the underbelly of the gleaming metropolis that the wealthy called home. It was a place of poverty, crime, and hopelessness, a city of a million people, all struggling to survive another day.",
        "sense_prediction": {
            "class_name": "City and Crowd",
            "class_id": 6
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The view from the top of the mountain was one of utter desolation. The land below was a blasted wasteland, the result of a long-forgotten war. The ground was scarred with craters, and the skeletal remains of dead forests stretched to the horizon. The air was thin and tasted of ash. He had climbed the peak to get a better sense of the scale of the destruction. It was even worse than he had imagined. A profound sadness settled over him. This was what his ancestors had done, in their pride and their anger. They had broken the world, and left the survivors to live in its ruins.",
        "sense_prediction": {
            "class_name": "Mountain and the heights",
            "class_id": 7
        },
        "age_prediction": {
            "class_name": "ancient and old age",
            "class_id": 0
        }
    },
    {
        "text": "The wind scoured the red plains of Mars, kicking up clouds of fine, rust-colored dust. The rover trundled on, its solar panels gleaming under the weak sun. It was an automated prospector, searching for pockets of subterranean ice. The landscape was majestic in its emptiness, a vast desert of rock and sand under a thin, pink sky. For years, the rover had roamed this silent world, its only companion the distant Earth, a bright blue star in the night sky. It was a lonely sentinel, a robotic pioneer exploring a world that was both breathtakingly beautiful and lethally hostile to life.",
        "sense_prediction": {
            "class_name": "Desert and dunes",
            "class_id": 8
        },
        "age_prediction": {
            "class_name": "technology modern age",
            "class_id": 2
        }
    },
    {
        "text": "The tidal wave was a dark line on the horizon, growing with terrifying speed. The people on the beach, who had been enjoying a sunny afternoon, began to scream and run. The sea, which had been so calm and inviting moments before, had become a monstrous, destructive force. The wave hit the shore with a deafening roar, swallowing everything in its path: houses, trees, people. It was a terrifying display of nature's raw, indifferent power. In a matter of minutes, a thriving coastal town was reduced to a ruin, a testament to the awesome and terrible might of the ocean's tide.",
        "sense_prediction": {
            "class_name": "Sea and tides",
            "class_id": 9
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    },
    {
        "text": "The forest was a place of deep, ancient magic. The trees were so old their bark looked like wrinkled skin, and their branches were twisted into strange, knowing shapes. The air hummed with a power that was palpable, a force that was neither good nor evil, but simply was. A young woman, a fledgling witch, walked its paths, seeking to learn its secrets. She knew that the forest was a sentient being, and that to gain its favor, she had to show it respect. She left offerings of wildflowers and honey at the base of the oldest oak, hoping the spirit of the woods would deem her worthy.",
        "sense_prediction": {
            "class_name": "Forest and tress",
            "class_id": 10
        },
        "age_prediction": {
            "class_name": "neutral and not special age (non-ancient, non technology)",
            "class_id": 1
        }
    }
]
#
# def evaluate_classifier(driver: GENRITADriver, test_cases: list) -> dict:
#     """
#     Evaluates a given classifier driver against a set of test cases.
#
#     Args:
#         driver (GENRITADriver): An instance of a classifier driver (NNDriver or LLMDriver).
#         test_cases (list): A list of test case dictionaries.
#
#     Returns:
#         dict: A dictionary containing performance metrics (accuracies, confusion matrices, reports).
#     """
#     logging.info(f"Starting evaluation for {type(driver).__name__}...")
#
#     y_true_sense, y_pred_sense = [], []
#     y_true_age, y_pred_age = [], []
#
#     for i, case in enumerate(test_cases):
#         text = case.get("text")
#         true_sense_id = case.get("sense_prediction", {}).get("class_id")
#         true_age_id = case.get("age_prediction", {}).get("class_id")
#
#         if text is None or true_sense_id is None or true_age_id is None:
#             logging.warning(f"Skipping malformed test case #{i}: {case}")
#             continue
#
#         try:
#             # Get predictions from the driver
#             predictions = driver.classify(text)
#             pred_sense_id = predictions.get("sense_prediction", {}).get("class_id", -1)
#             pred_age_id = predictions.get("age_prediction", {}).get("class_id", -1)
#
#             # Append true and predicted labels for metrics calculation
#             y_true_sense.append(true_sense_id)
#             y_pred_sense.append(pred_sense_id)
#             y_true_age.append(true_age_id)
#             y_pred_age.append(pred_age_id)
#
#             logging.info(f"Processed case {i+1}/{len(test_cases)}")
#
#         except Exception as e:
#             logging.error(f"Failed to classify text for case #{i}. Error: {e}")
#             # Append -1 to indicate failure for this case
#             y_pred_sense.append(-1)
#             y_pred_age.append(-1)
#             y_true_sense.append(true_sense_id)
#             y_true_age.append(true_age_id)
#
#
#     # --- Calculate Metrics ---
#     # Note: We use the labels present in the data for reports
#     unique_sense_labels = sorted(list(set(y_true_sense)))
#     unique_age_labels = sorted(list(set(y_true_age)))
#
#     sense_class_names = [SENSE_ID_TO_NAME.get(i, "Unknown") for i in unique_sense_labels]
#     age_class_names = [AGE_ID_TO_NAME.get(i, "Unknown") for i in unique_age_labels]
#
#     results = {
#         "sense_accuracy": accuracy_score(y_true_sense, y_pred_sense),
#         "age_accuracy": accuracy_score(y_true_age, y_pred_age),
#         "sense_confusion_matrix": confusion_matrix(y_true_sense, y_pred_sense, labels=unique_sense_labels),
#         "age_confusion_matrix": confusion_matrix(y_true_age, y_pred_age, labels=unique_age_labels),
#         "sense_classification_report": classification_report(y_true_sense, y_pred_sense, target_names=sense_class_names, zero_division=0),
#         "age_classification_report": classification_report(y_true_age, y_pred_age, target_names=age_class_names, zero_division=0),
#         "sense_class_names": sense_class_names,
#         "age_class_names": age_class_names
#     }
#     logging.info("Evaluation complete.")
#     return results
#

def evaluate_classifier(driver: GENRITADriver, test_cases: list) -> dict:
    """
    Evaluates a given classifier driver against a set of test cases.
    """
    logging.info(f"Starting evaluation for {type(driver).__name__}...")

    y_true_sense, y_pred_sense = [], []
    y_true_age, y_pred_age = [], []

    for i, case in enumerate(test_cases):
        text = case.get("text")
        true_sense_id = case.get("sense_prediction", {}).get("class_id")
        true_age_id = case.get("age_prediction", {}).get("class_id")

        if text is None or true_sense_id is None or true_age_id is None:
            logging.warning(f"Skipping malformed test case #{i}: {case}")
            continue

        try:
            predictions = driver.classify(text)
            pred_sense_id = predictions.get("sense_prediction", {}).get("class_id", -1)
            pred_age_id = predictions.get("age_prediction", {}).get("class_id", -1)

            y_true_sense.append(true_sense_id)
            y_pred_sense.append(pred_sense_id)
            y_true_age.append(true_age_id)
            y_pred_age.append(pred_age_id)

            logging.info(f"Processed case {i + 1}/{len(test_cases)}")

        except Exception as e:
            logging.error(f"Failed to classify text for case #{i}. Error: {e}")
            y_pred_sense.append(-1)
            y_pred_age.append(-1)
            y_true_sense.append(true_sense_id)
            y_true_age.append(true_age_id)

    # --- Calculate Metrics ---
    # Define the full set of labels based on ground truth, and add our failure case label (-1)
    sense_labels = sorted(list(SENSE_CLASSES.values()))
    age_labels = sorted(list(AGE_CLASSES.values()))

    report_sense_labels = sense_labels + [-1]
    report_age_labels = age_labels + [-1]

    sense_class_names = [SENSE_ID_TO_NAME.get(i, "Unknown") for i in sense_labels]
    age_class_names = [AGE_ID_TO_NAME.get(i, "Unknown") for i in age_labels]

    report_sense_names = sense_class_names + ["Failed Parse"]
    report_age_names = age_class_names + ["Failed Parse"]

    results = {
        "sense_accuracy": accuracy_score(y_true_sense, y_pred_sense),
        "age_accuracy": accuracy_score(y_true_age, y_pred_age),
        "sense_confusion_matrix": confusion_matrix(y_true_sense, y_pred_sense, labels=report_sense_labels),
        "age_confusion_matrix": confusion_matrix(y_true_age, y_pred_age, labels=report_age_labels),
        "sense_classification_report": classification_report(
            y_true_sense, y_pred_sense, labels=report_sense_labels, target_names=report_sense_names, zero_division=0
        ),
        "age_classification_report": classification_report(
            y_true_age, y_pred_age, labels=report_age_labels, target_names=report_age_names, zero_division=0
        ),
        "sense_class_names": report_sense_names,
        "age_class_names": report_age_names
    }
    logging.info("Evaluation complete.")
    return results

def plot_performance(results: dict, driver_name: str):
    """
    Prints reports and plots confusion matrices for the evaluation results.

    Args:
        results (dict): The dictionary of results from the evaluate_classifier function.
        driver_name (str): The name of the driver being evaluated (e.g., "NN" or "LLM").
    """
    print("\n" + "="*50)
    print(f"PERFORMANCE REPORT FOR: {driver_name.upper()} DRIVER")
    print("="*50 + "\n")

    # --- Print Reports ---
    print(f"Overall Sense Accuracy: {results['sense_accuracy']:.2%}")
    print("\nSense Classification Report:")
    print(results['sense_classification_report'])
    print("\n" + "-"*50 + "\n")
    print(f"Overall Age Accuracy: {results['age_accuracy']:.2%}")
    print("\nAge Classification Report:")
    print(results['age_classification_report'])
    print("\n" + "="*50 + "\n")

    # --- Plot Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(f'Confusion Matrices for {driver_name.upper()} Driver', fontsize=20)

    # Sense Confusion Matrix
    sns.heatmap(results['sense_confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=results['sense_class_names'], yticklabels=results['sense_class_names'])
    axes[0].set_title('Sense Prediction', fontsize=16)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(axes[0].get_yticklabels(), rotation=0)


    # Age Confusion Matrix
    sns.heatmap(results['age_confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=results['age_class_names'], yticklabels=results['age_class_names'])
    axes[1].set_title('Age Prediction', fontsize=16)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(axes[1].get_yticklabels(), rotation=0)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"./genrita_bench/Overall GENRITA {driver_name}.png", dpi=300, bbox_inches="tight")
    # plt.show()

if __name__ == '__main__':
    NN_DRIVER_PARAMS = {
        "checkpoint_path": "./checkpoints/best-model.ckpt"
    }
    LLM_DRIVER_PARAMS = {
        "ollama_model_name": "phi4-mini"
    }
    driver = "llm"

    try:
        if driver == 'nn':
            driver_instance = GENRITADriver.get_classifer('nn', NN_DRIVER_PARAMS)
            driver_name = "Neural Network (RoBERTa)"
        else: # args.driver == 'llm'
            driver_instance = GENRITADriver.get_classifer('llm', LLM_DRIVER_PARAMS)
            driver_name = f"Large Language Model ({LLM_DRIVER_PARAMS['ollama_model_name']})"

        evaluation_results = evaluate_classifier(driver_instance, TEST_CASES)

        plot_performance(evaluation_results, driver_name)

    except (FileNotFoundError, ImportError, Exception) as e:
        logging.error(f"Evaluation script failed. Reason: {e}")

