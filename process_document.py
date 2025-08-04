import argparse
import re
from pathlib import Path
from tqdm import tqdm
import json
import logging

from GENRITA import GENRITADriver
from ROAST import ROASTDriver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GRPipeline:
    """
    An object-oriented class to process text documents, integrating a configurable
    classifier (NN or LLM) and the ROAST instance extractor.
    """

    def __init__(self, driver_type: str, driver_params: dict, roast_params: dict, processing_params: dict):
        logging.info(f"--- Initializing Genrita Pipeline with '{driver_type.upper()}' driver ---")

        self.classifier = GENRITADriver.get_classifer(driver_type, driver_params)

        self.target_abstracts = roast_params.get('target_abstracts')
        if self.target_abstracts:
            logging.info(f"Loading ROAST Model from: {roast_params['roast_model_path']}")

            self.extractor = ROASTDriver.get_extractor(
                driver_type=driver_type,
                model_name_or_path=roast_params['roast_model_path'],
                score_threshold=roast_params.get('roast_score_threshold', 0.55)
            )

            logging.info(f"ROAST will extract instances for: {list(self.target_abstracts.keys())}")
        else:
            self.extractor = None
            logging.info("ROAST extractor not configured.")

        self.confidence_threshold = processing_params.get('confidence_threshold', 0.0)
        self.allowed_sense_ids = set(processing_params.get('allowed_senses')) if processing_params.get(
            'allowed_senses') else None
        self.allowed_age_ids = set(processing_params.get('allowed_ages')) if processing_params.get(
            'allowed_ages') else None

        logging.info(f"--- Pipeline ready ---")
        if self.allowed_sense_ids: logging.info(f"Filtering for Sense IDs: {self.allowed_sense_ids}")
        if self.allowed_age_ids: logging.info(f"Filtering for Age IDs: {self.allowed_age_ids}")
        logging.info(f"Confidence threshold set to: {self.confidence_threshold}")

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

            if len(" ".join(current_chunk_sentences).split()) >= target_words:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def process_text_content(self, text_content: str, title: str = "Untitled") -> dict:
        """Processes a raw text string and returns the full analysis."""
        logging.info(f"Processing document titled: '{title}'")

        paragraphs = self._chunk_text(text_content)
        if not paragraphs:
            logging.warning("Input text was empty or could not be split into paragraphs.")
            return {'title': title, 'paragraphs': []}
        logging.info(f"Split text into {len(paragraphs)} paragraphs.")

        all_results = []
        last_successful_prediction = None

        for text_paragraph in tqdm(paragraphs, desc=f"Analyzing paragraphs for '{title}'"):

            classification_result = self.classifier.classify(
                text_paragraph, self.allowed_sense_ids, self.allowed_age_ids
            )
            sense_pred = classification_result['sense_prediction']
            age_pred = classification_result['age_prediction']

            if last_successful_prediction:
                if sense_pred["confidence"] < self.confidence_threshold:
                    sense_pred = last_successful_prediction["sense_prediction"]
                if age_pred["confidence"] < self.confidence_threshold:
                    age_pred = last_successful_prediction["age_prediction"]

            entities = []
            if self.extractor and self.target_abstracts:
                for abstract, explanation in self.target_abstracts.items():
                    try:
                        instances = self.extractor.extract(text_paragraph, abstract.strip(), explanation.strip())
                        for span, score, start, end in instances:
                            entities.append(
                                {"type": abstract, "sample": span, "start_pos": start, "end_pos": end, "score": score})
                    except Exception as e:
                        logging.warning(f"ROAST failed for abstract '{abstract}' on a paragraph. Error: {e}")

            final_result = {
                "text": text_paragraph,
                "sense_prediction": sense_pred,
                "age_prediction": age_pred,
                "entities": entities
            }
            all_results.append(final_result)

            if sense_pred['class_id'] != -1 and age_pred['class_id'] != -1:
                last_successful_prediction = final_result

        return {'title': title, 'paragraphs': all_results}

    @staticmethod
    def save_to_json(data: dict, output_file_path: str):
        """Saves a dictionary to a JSON file."""
        if not data or not data.get('paragraphs'):
            logging.warning("No data to save. JSON file not created.")
            return
        logging.info(f"Saving results to: {output_file_path}")
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info("File saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Genrita Pipeline: Process a text file with a configurable classifier (NN or LLM) and instance extraction.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Driver Selection ---
    parser.add_argument("--driver", type=str, required=True, choices=['nn', 'llm'],
                        help="The classification driver to use.\n"
                             " 'nn': Use the fine-tuned RoBERTa model.\n"
                             " 'llm': Use a generative LLM via Ollama.")

    # --- File Arguments ---
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .txt file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the output .json file.")
    parser.add_argument("--title", type=str, default=None, help="Document title (uses filename if not provided).")

    # --- Processing Parameters ---
    proc_group = parser.add_argument_group('Classification Processing Parameters')
    proc_group.add_argument("--threshold", type=float, default=0.9, help="Confidence threshold for classification.")
    proc_group.add_argument("--allowed_senses", type=str, default=None,
                            help="Comma-separated list of allowed sense class IDs (e.g., '2,3').")
    proc_group.add_argument("--allowed_ages", type=str, default=None,
                            help="Comma-separated list of allowed age class IDs (e.g., '0,1').")

    # --- NN Driver Arguments ---
    nn_group = parser.add_argument_group('NN Driver Arguments (required if --driver=nn)')
    nn_group.add_argument("--checkpoint_path", type=str, default="checkpoints/best-model.ckpt",
                          help="Path to the classifier .ckpt file.")

    # --- LLM Driver Arguments ---
    llm_group = parser.add_argument_group('LLM Driver Arguments (required if --driver=llm)')
    llm_group.add_argument("--ollama_model", type=str, default="gemma3:1b",
                           help="Name of the model to use with Ollama (e.g., 'mistral', 'gemma3:12b').")

    # --- ROAST Arguments ---
    roast_group = parser.add_argument_group('ROAST Instance Extractor Arguments (optional)')
    roast_group.add_argument("--roast_model_path", type=str, default="QA_RoBERTA_SQUADv2",
                             help="Path to the ROAST (extractive QA) models (e.g. 'QA_RoBERTA_SQUADv2', 'QA_XLM_RoBERTA'")
    roast_group.add_argument("--target_abstracts", type=str, default=None,
                             help="Quoted, comma-separated list of abstract concepts to extract.\n"
                                  "Format: 'concept1:explanation1,concept2:explanation2'\n"
                                  "Example: 'dragon:a mythical beast,magic spell:an arcane incantation'")

    args = parser.parse_args()

    # --- Argument Validation and Setup ---
    driver_params = {}
    if args.driver == 'nn':
        if not args.checkpoint_path:
            parser.error("--checkpoint_path is required when --driver=nn")
        driver_params['checkpoint_path'] = args.checkpoint_path
    elif args.driver == 'llm':
        if not args.ollama_model:
            parser.error("--ollama_model is required when --driver=llm")
        driver_params['ollama_model_name'] = args.ollama_model

    target_abstracts_dict = {}
    if args.target_abstracts:
        try:
            for abs_exp in args.target_abstracts.split(','):
                parts = abs_exp.split(":", 1)
                if len(parts) == 2:
                    target_abstracts_dict[parts[0].strip()] = parts[1].strip()
                else:
                    logging.warning(f"Skipping malformed abstract: '{abs_exp}'. Expected 'concept:explanation' format.")
        except Exception as e:
            parser.error(f"Could not parse --target_abstracts. Error: {e}")

    roast_params = {
        'roast_model_path': args.roast_model_path,
        'target_abstracts': target_abstracts_dict if target_abstracts_dict else None
    }

    processing_params = {
        'confidence_threshold': args.threshold,
        'allowed_senses': [int(id_str) for id_str in args.allowed_senses.split(',')] if args.allowed_senses else None,
        'allowed_ages': [int(id_str) for id_str in args.allowed_ages.split(',')] if args.allowed_ages else None
    }

    try:
        pipeline = GRPipeline(
            driver_type=args.driver,
            driver_params=driver_params,
            roast_params=roast_params,
            processing_params=processing_params
        )

        input_path = Path(args.input_file)
        text_content = input_path.read_text(encoding='utf-8')
        results = pipeline.process_text_content(
            text_content,
            title=args.title or input_path.stem
        )
        pipeline.save_to_json(data=results, output_file_path=args.output_file)

        logging.info("\n--- Process finished successfully ---")

    except Exception as e:
        logging.critical(f"\n--- A critical error occurred: {e} ---", exc_info=True)
