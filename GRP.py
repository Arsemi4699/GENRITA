import re
from logging import Logger
from typing import Optional
from tqdm import tqdm
from GENRITA import GENRITADriver, SENSE_ID_TO_NAME, AGE_ID_TO_NAME
from ROAST import ROASTDriver
import gc
import torch
from data_processor import DataProcessor
from thefuzz import fuzz
from post_processor import ContextualizedPostprocessor

# --- GRPipeline Class (Adapted for API) ---
class GRPipeline:
    """
    Processes text documents, integrating configurable classifiers and extractors,
    and applies a robust post-processing pipeline for sense classification.
    """

    def __init__(
        self,
        classifier_driver_type: str,
        classifier_params: dict,
        extractor_driver_type: Optional[str],
        roast_params: dict,
        processing_params: dict,
        logger: Logger,
    ):
        self.logger = logger
        self.logger.info(f"--- Initializing GRPipeline ---")
        self.logger.info(f"Classifier Driver: '{classifier_driver_type.upper()}'")
        self.classifier = GENRITADriver.get_classifer(
            classifier_driver_type, classifier_params
        )
        self.logger.info("Classifier initialized successfully.")

        self.target_abstracts = roast_params.get("target_abstracts")
        if self.target_abstracts and extractor_driver_type:
            self.logger.info(f"Extractor Driver: '{extractor_driver_type.upper()}'")
            self.logger.info(
                f"Loading ROAST Model from: {roast_params['roast_model_path']}"
            )
            self.extractor = ROASTDriver.get_extractor(
                driver_type=extractor_driver_type,
                model_name_or_path=roast_params["roast_model_path"],
                score_threshold=roast_params.get("roast_score_threshold", 0.55),
            )
            self.logger.info(
                f"ROAST will extract instances for: {list(self.target_abstracts.keys())}"
            )
        else:
            self.extractor = None
            self.logger.info(
                "ROAST extractor not configured or extractor driver not specified."
            )

        self.confidence_threshold = processing_params.get("confidence_threshold", 0.0)
        self.allowed_sense_ids = (
            set(processing_params.get("allowed_senses"))
            if processing_params.get("allowed_senses")
            else None
        )
        self.allowed_age_ids = (
            set(processing_params.get("allowed_ages"))
            if processing_params.get("allowed_ages")
            else None
        )

        self.run_postprocessor = True
        self.postprocessor = ContextualizedPostprocessor(
            logger=self.logger,
            sense_neutral_id=0,
            sense_centroids_path="centroids_embeds/sense_centroids.npy",
            sense_entropy_thr=1.1,
            age_neutral_id=1,
            age_centroids_path="centroids_embeds/age_centroids.npy",
            age_entropy_thr=0.5
        )

        self.canonical_entity_bank = {}
        self.logger.info(f"--- Pipeline ready ---")

    @staticmethod
    def _chunk_text(
            text: str,
            target_words: int = 128,
            min_paragraph_words: int = 48,
            min_block_words: int = 30,
    ) -> list[str]:
        """
        Split text into semantically meaningful chunks.
        Returns:
            list[str]: A list of text chunks.
        """
        text = text.strip()
        if not text:
            return []

        chunks = []
        current_chunk_sentences = []

        # Split text into blocks with 3+ newlines
        blocks = re.split(r"\n{3,}", text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            block_word_count = len(block.split())
            treat_as_strict_boundary = block_word_count >= min_block_words

            # Split block into paragraphs
            paragraphs = [p.strip() for p in block.split("\n") if p.strip()]
            for paragraph in paragraphs:
                para_word_count = len(paragraph.split())

                # Large paragraph → standalone chunk
                if para_word_count >= min_paragraph_words:
                    if current_chunk_sentences:
                        chunks.append(" ".join(current_chunk_sentences))
                        current_chunk_sentences = []
                    chunks.append(paragraph)
                    continue

                # Split paragraph into sentences
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    if not sentence:
                        continue
                    current_chunk_sentences.append(sentence)
                    if len(" ".join(current_chunk_sentences).split()) >= target_words:
                        chunks.append(" ".join(current_chunk_sentences))
                        current_chunk_sentences = []

            # At the end of a "hard boundary" block → flush current sentences
            if treat_as_strict_boundary and current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []

        # Flush remaining sentences
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _unify_entities(self, found_entities: list) -> list:
        if not found_entities:
            return []
        entity_to_group_id = {}
        next_group_id_counter = 0
        for i, entity in enumerate(found_entities):
            abstract_type = entity["type"]
            current_sample = entity["sample"]
            if abstract_type not in self.canonical_entity_bank:
                self.canonical_entity_bank[abstract_type] = {}
            preprocessed_current = DataProcessor._preprocess_entity_text(current_sample)
            best_match_group_id = -1
            highest_score = 0
            for group_id, canonical_sample in self.canonical_entity_bank[
                abstract_type
            ].items():
                preprocessed_canonical = DataProcessor._preprocess_entity_text(
                    canonical_sample
                )
                ratio_score = fuzz.ratio(preprocessed_current, preprocessed_canonical)
                partial_ratio_score = fuzz.partial_ratio(
                    preprocessed_current, preprocessed_canonical
                )
                combined_score = (ratio_score + partial_ratio_score) / 2
                if combined_score > highest_score:
                    highest_score = combined_score
                    best_match_group_id = group_id
            SIMILARITY_THRESHOLD = 60
            if highest_score >= SIMILARITY_THRESHOLD:
                entity_to_group_id[i] = best_match_group_id
                canonical_sample = self.canonical_entity_bank[abstract_type][
                    best_match_group_id
                ]
                if len(current_sample) > len(canonical_sample):
                    self.canonical_entity_bank[abstract_type][
                        best_match_group_id
                    ] = current_sample
            else:
                while next_group_id_counter in self.canonical_entity_bank.get(
                    abstract_type, {}
                ):
                    next_group_id_counter += 1
                new_id = next_group_id_counter
                entity_to_group_id[i] = new_id
                self.canonical_entity_bank[abstract_type][new_id] = current_sample
                next_group_id_counter += 1
        unified_results = []
        for i, entity in enumerate(found_entities):
            final_entity = entity.copy()
            group_id = entity_to_group_id[i]
            abstract_type = final_entity["type"]
            final_entity["sample"] = self.canonical_entity_bank[abstract_type][group_id]
            unified_results.append(final_entity)
        return unified_results

    def process_text_content(self, text_content: str, title: str = "Untitled") -> dict:
        self.logger.info(f"Processing document titled: '{title}'")
        paragraphs = self._chunk_text(text_content)
        if not paragraphs:
            self.logger.warning("Input text was empty or could not be chunked.")
            return {"title": title, "paragraphs": []}
        self.logger.info(f"Split text into {len(paragraphs)} paragraphs.")

        # --- Phase 1: Data Collection for Both Tasks ---
        self.logger.info("Phase 1: Collecting raw model outputs for Sense and Age...")
        raw_outputs = []
        for text_paragraph in tqdm(paragraphs, desc=f"Phase 1/3: Analyzing paragraphs for '{title}'"):
            # The classify method returns final predictions and raw probabilities for both tasks
            _, _, sense_probs_np, age_probs_np = self.classifier.classify(
                text_paragraph, self.allowed_sense_ids, self.allowed_age_ids
            )
            # We store the raw probabilities needed for the multi-task post-processor
            raw_outputs.append({
                "text": text_paragraph,
                "sense_probs": sense_probs_np,
                "age_probs": age_probs_np,  # Now storing raw age probabilities as well
                "raw_entities": []
            })

        # --- Phase 1b: Entity Extraction (Preserved from your original code) ---
        self.logger.info("Phase 1b: Extracting entities and building canonical bank...")
        for output in tqdm(raw_outputs, desc=f"Phase 2/3: Extracting entities for '{title}'"):
            text_paragraph = output["text"]
            raw_entities = []
            if self.extractor and self.target_abstracts:
                for abstract, explanation in self.target_abstracts.items():
                    try:
                        instances = self.extractor.extract(
                            text_paragraph, abstract.strip(), explanation.strip()
                        )
                        for span, score, start, end in instances:
                            raw_entities.append({
                                "type": abstract,
                                "sample": span,
                                "start_pos": start,
                                "end_pos": end,
                                "score": score,
                            })
                    except Exception as e:
                        self.logger.warning(f"ROAST failed for abstract '{abstract}'. Error: {e}")
            output["raw_entities"] = raw_entities
            self._unify_entities(raw_entities)

        # --- Phase 2: Multi-Task Post-Processing ---
        self.logger.info("Phase 2: Applying multi-task post-processing pipeline...")
        # The post-processor now returns final IDs and probabilities for both tasks
        final_sense_ids, final_age_ids, final_sense_probs, final_age_probs = self.postprocessor.process_book(
            raw_outputs)

        # --- Phase 3: Final Assembly ---
        self.logger.info("Phase 3: Assembling final results...")
        final_results = []
        for i, result in enumerate(tqdm(raw_outputs, desc=f"Phase 3/3: Assembling final results for '{title}'")):
            # 1. Assemble SENSE prediction using post-processed results
            final_sense_id = final_sense_ids[i]
            sense_confidence = final_sense_probs[i, final_sense_id]
            sense_pred = {
                "class_name": SENSE_ID_TO_NAME.get(final_sense_id, "Unknown"),
                "class_id": int(final_sense_id),
                "confidence": float(sense_confidence)
            }

            # 2. Assemble AGE prediction using post-processed results (old fallback logic is now removed)
            final_age_id = final_age_ids[i]
            age_confidence = final_age_probs[i, final_age_id]
            age_pred = {
                "class_name": AGE_ID_TO_NAME.get(final_age_id, "Unknown"),
                "class_id": int(final_age_id),
                "confidence": float(age_confidence)
            }

            # 3. Unify entities for the final output
            unified_entities = self._unify_entities(result["raw_entities"])

            final_paragraph_result = {
                "text": result["text"],
                "sense_prediction": sense_pred,
                "age_prediction": age_pred,
                "entities": unified_entities,
            }
            final_results.append(final_paragraph_result)

        final_unified_instances = {k: list(v.values()) for k, v in self.canonical_entity_bank.items()}

        return {
            "title": title,
            "paragraphs": final_results,
            "canonical_entity_bank": final_unified_instances,
        }

    def cleanup(self, genrita_driver, roast_driver):
        self.logger.info(f"--- Cleaning up resources for GRPipeline instance ---")

        if genrita_driver == "nn":
            self.logger.info("Releasing classifier resources...")
            if hasattr(self.classifier, "model"):
                del self.classifier.model
            del self.classifier

        if roast_driver == "nn":
            self.logger.info("Releasing extractor resources...")
            if hasattr(self.extractor, "reader"):
                del self.extractor.reader
            del self.extractor

        self.logger.info("Releasing postprocessor resources...")
        if hasattr(self.postprocessor, "sense_centroids"):
            del self.postprocessor.sense_centroids
        if hasattr(self.postprocessor, "age_centroids"):
            del self.postprocessor.age_centroids
        if hasattr(self.postprocessor, "embedder"):
            del self.postprocessor.embedder
        del self.postprocessor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared.")

        gc.collect()
        self.logger.info("--- Cleanup complete ---")
