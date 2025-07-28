import argparse
import json
import re
from pathlib import Path
import torch
from tqdm import tqdm

from model import RoBERTaMultiTaskClassifier
from data_processor import DataProcessor


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


class DocumentProcessor:
    """
    An object-oriented class to process text documents, respecting the fixed structure of other project files.
    """

    def __init__(self, checkpoint_path: str, confidence_threshold: float = 0.0, allowed_senses: list[int] = None,
                 allowed_ages: list[int] = None):
        """
        Initializes the processor, loads the model, and sets processing parameters.
        """
        print(f"--- Initializing DocumentProcessor from: {checkpoint_path} ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)

        # Reverse maps for decoding predictions
        self.sense_id_to_name = {v: k for k, v in SENSE_CLASSES.items()}
        self.age_id_to_name = {v: k for k, v in AGE_CLASSES.items()}

        # Store processing parameters
        self.confidence_threshold = confidence_threshold

        # Convert allowed IDs to a set for efficient lookup
        self.allowed_sense_ids = set(allowed_senses) if allowed_senses else None
        self.allowed_age_ids = set(allowed_ages) if allowed_ages else None

        print(f"--- Processor ready on device: {self.device} ---")
        if self.allowed_sense_ids: print(f"Filtering for Sense IDs: {self.allowed_sense_ids}")
        if self.allowed_age_ids: print(f"Filtering for Age IDs: {self.allowed_age_ids}")
        print(f"Confidence threshold set to: {self.confidence_threshold}")

    def _load_model(self, checkpoint_path: str):
        """Internal method to load the model and move it to the correct device."""
        try:
            model = RoBERTaMultiTaskClassifier.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=self.device
            )
            model.freeze()
            model.eval()
            return model
        except FileNotFoundError:
            print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
            raise
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading the model: {e}")
            raise

    def _predict_with_probabilities(self, text: str):
        """
        A custom prediction method that returns full probability distributions.
        This is necessary because the original model.predict does not, and we cannot change it.
        """
        # Step 1: Clean text using the original, unchanged DataProcessor
        cleaned_text = DataProcessor._clean_text(text)

        # Step 2: Tokenize using the model's tokenizer
        encoding = self.model.tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.model.hparams.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Step 3: Perform a forward pass to get logits
        with torch.no_grad():
            sense_logits, age_logits = self.model(input_ids, attention_mask)

        # Step 4: Convert logits to probabilities
        sense_probs = torch.softmax(sense_logits, dim=1).squeeze()
        age_probs = torch.softmax(age_logits, dim=1).squeeze()

        return sense_probs, age_probs

    def _get_best_allowed_prediction(self, probabilities: torch.Tensor, id_to_name_map: dict, allowed_ids: set = None):
        """Finds the highest-confidence prediction within the list of allowed class IDs."""
        # If no filter is applied, return the top prediction
        if not allowed_ids:
            top_prob, top_id_tensor = torch.max(probabilities, dim=0)
            top_id = top_id_tensor.item()
            return {
                "class_name": id_to_name_map.get(top_id, "Unknown"),
                "class_id": top_id,
                "confidence": top_prob.item()
            }

        # Sort probabilities to find the best allowed one
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

        for i, idx_tensor in enumerate(sorted_indices):
            idx = idx_tensor.item()
            if idx in allowed_ids:
                return {
                    "class_name": id_to_name_map.get(idx, "Unknown"),
                    "class_id": idx,
                    "confidence": sorted_probs[i].item()
                }

        # Fallback if no allowed class is found
        return {"class_name": "No allowed class found", "class_id": -1, "confidence": 0.0}

    @staticmethod
    def _chunk_text(text: str, target_words: int = 128) -> list[str]:
        """Splits text into chunks of ~target_words, without breaking sentences."""
        text = text.replace('\n', ' ').strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences: return []
        chunks, current_chunk_sentences = [], []
        for sentence in sentences:
            if not sentence: continue
            current_chunk_sentences.append(sentence)
            temp_chunk_text = " ".join(current_chunk_sentences)
            if len(temp_chunk_text.split()) >= target_words:
                chunks.append(temp_chunk_text)
                current_chunk_sentences = []
        if current_chunk_sentences: chunks.append(" ".join(current_chunk_sentences))
        return chunks

    def process_text_content(self, text_content: str, title: str = "Untitled") -> dict:
        """Processes a raw text string and returns the analysis as a dictionary."""
        print(f"Processing document titled: '{title}'")
        paragraphs = self._chunk_text(text_content)
        print(f"Split text into {len(paragraphs)} paragraphs.")

        all_results = []
        last_successful_prediction = None

        for i, text_paragraph in enumerate(tqdm(paragraphs, desc=f"Analyzing paragraphs for '{title}'")):
            sense_probs, age_probs = self._predict_with_probabilities(text_paragraph)

            sense_pred = self._get_best_allowed_prediction(sense_probs, self.sense_id_to_name, self.allowed_sense_ids)
            age_pred = self._get_best_allowed_prediction(age_probs, self.age_id_to_name, self.allowed_age_ids)

            # Apply confidence threshold logic (for paragraphs after the first one)
            if i > 0 and last_successful_prediction:
                if sense_pred["confidence"] < self.confidence_threshold:
                    sense_pred = last_successful_prediction["sense_prediction"]
                if age_pred["confidence"] < self.confidence_threshold:
                    age_pred = last_successful_prediction["age_prediction"]

            # Construct the final result for this paragraph (without cleaned_text)
            final_result = {
                "text": text_paragraph,
                "sense_prediction": sense_pred,
                "age_prediction": age_pred
            }
            all_results.append(final_result)
            last_successful_prediction = final_result

        return {'title': title, 'paragraphs': all_results}

    @staticmethod
    def save_to_json(data: dict, output_file_path: str):
        """Saves a dictionary to a JSON file."""
        if not data:
            print("No data to save. JSON file not created.")
            return
        print(f"Saving results to: {output_file_path}")
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("File saved successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process a text file with a RoBERTa model, with filtering and thresholding."
    )
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best-checkpoint-epoch=02-val_loss=0.90.ckpt", help="Path to the trained model .ckpt file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .txt file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the output .json file.")
    parser.add_argument("--title", type=str, default=None, help="Document title (uses filename if not provided).")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Confidence threshold to fallback to previous paragraph's prediction.")
    parser.add_argument("--allowed_senses", type=str, default=None,
                        help="Comma-separated list of allowed sense class IDs (e.g., '2,3').")
    parser.add_argument("--allowed_ages", type=str, default=None,
                        help="Comma-separated list of allowed age class IDs (e.g., '0,1').")
    args = parser.parse_args()

    try:
        # Convert comma-separated ID strings to lists of integers
        allowed_senses_ids = [int(id_str) for id_str in args.allowed_senses.split(',')] if args.allowed_senses else None
        allowed_ages_ids = [int(id_str) for id_str in args.allowed_ages.split(',')] if args.allowed_ages else None

        processor = DocumentProcessor(
            checkpoint_path=args.checkpoint_path,
            confidence_threshold=args.threshold,
            allowed_senses=allowed_senses_ids,
            allowed_ages=allowed_ages_ids
        )

        text_content = Path(args.input_file).read_text(encoding='utf-8')
        results = processor.process_text_content(
            text_content,
            title=args.title or Path(args.input_file).stem
        )

        processor.save_to_json(data=results, output_file_path=args.output_file)

        print("\n--- Process finished successfully ---")

    except Exception as e:
        print(f"\nERROR: A critical error occurred: {e}")
