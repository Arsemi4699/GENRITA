import re
from logging import Logger
from typing import Optional
from tqdm import tqdm
from GENRITA import GENRITADriver
from ROAST import ROASTDriver
import gc
import torch
from data_processor import DataProcessor
from thefuzz import fuzz

# --- GRPipeline Class (Adapted for API) ---
class GRPipeline:
    """
    Processes text documents, integrating configurable classifiers and extractors.
    Adapted for use within the FastAPI application.
    """

    def __init__(
        self,
        classifier_driver_type: str,
        classifier_params: dict,
        extractor_driver_type: Optional[str],
        roast_params: dict,
        processing_params: dict,
        logger: Logger
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
            self.logger.info(f"Loading ROAST Model from: {roast_params['roast_model_path']}")
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

        self.canonical_entity_bank = {}

        self.logger.info(f"--- Pipeline ready ---")
        if self.allowed_sense_ids:
            self.logger.info(f"Filtering for Sense IDs: {self.allowed_sense_ids}")
        if self.allowed_age_ids:
            self.logger.info(f"Filtering for Age IDs: {self.allowed_age_ids}")
        self.logger.info(f"Confidence threshold set to: {self.confidence_threshold}")

    @staticmethod
    def _chunk_text(text: str, target_words: int = 128) -> list[str]:
        text = text.replace("\n", " ").strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return []
        chunks, current_chunk_sentences = [], []
        for sentence in sentences:
            if not sentence:
                continue
            current_chunk_sentences.append(sentence)
            if len(" ".join(current_chunk_sentences).split()) >= target_words:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        return chunks

    def _unify_entities(self, found_entities: list) -> list:
        """
        یکپارچه‌سازی موجودیت‌های استخراج شده با استفاده از یک رویکرد بهینه تک-حلقه‌ای
        که مستقیماً بانک متعارف کلاس را آپدیت می‌کند.
        """
        if not found_entities:
            return []

        # دیکشنری برای نگهداری ID گروه هر موجودیت در این پاراگراف
        entity_to_group_id = {}
        next_group_id_counter = 0

        # --- حلقه اصلی (شناسایی گروه‌ها و به‌روزرسانی بانک) ---
        for i, entity in enumerate(found_entities):
            abstract_type = entity["type"]
            current_sample = entity["sample"]

            # اگر بانک برای این abstract وجود ندارد، آن را بساز
            if abstract_type not in self.canonical_entity_bank:
                self.canonical_entity_bank[abstract_type] = {}

            preprocessed_current = DataProcessor._preprocess_entity_text(current_sample)

            best_match_group_id = -1
            highest_score = 0

            # مقایسه با نماینده‌های متعارف موجود در بانک اصلی کلاس
            for group_id, canonical_sample in self.canonical_entity_bank[abstract_type].items():
                preprocessed_canonical = DataProcessor._preprocess_entity_text(canonical_sample)

                ratio_score = fuzz.ratio(preprocessed_current, preprocessed_canonical)
                partial_ratio_score = fuzz.partial_ratio(preprocessed_current, preprocessed_canonical)
                combined_score = (ratio_score + partial_ratio_score) / 2

                if combined_score > highest_score:
                    highest_score = combined_score
                    best_match_group_id = group_id

            SIMILARITY_THRESHOLD = 60
            if highest_score >= SIMILARITY_THRESHOLD:
                # یک تطابق پیدا شد. ID گروه را به موجودیت فعلی اختصاص بده.
                entity_to_group_id[i] = best_match_group_id

                # بررسی اینکه آیا نمونه فعلی کامل‌تر است تا نماینده گروه را در بانک اصلی آپدیت کنیم؟
                canonical_sample = self.canonical_entity_bank[abstract_type][best_match_group_id]
                if len(current_sample) > len(canonical_sample):
                    self.canonical_entity_bank[abstract_type][best_match_group_id] = current_sample
            else:
                # یک گروه جدید است.
                # پیدا کردن یک ID جدید که در بانک وجود نداشته باشد
                while next_group_id_counter in self.canonical_entity_bank.get(abstract_type, {}):
                    next_group_id_counter += 1

                new_id = next_group_id_counter
                entity_to_group_id[i] = new_id
                self.canonical_entity_bank[abstract_type][new_id] = current_sample
                next_group_id_counter += 1

        # --- حلقه نهایی (یکپارچه‌سازی خروجی بر اساس بانک نهایی) ---
        unified_results = []
        for i, entity in enumerate(found_entities):
            final_entity = entity.copy()
            group_id = entity_to_group_id[i]
            abstract_type = final_entity["type"]

            # نمونه را با نماینده نهایی و متعارف گروهش از بانک اصلی جایگزین کن
            final_entity["sample"] = self.canonical_entity_bank[abstract_type][group_id]
            unified_results.append(final_entity)

        return unified_results

    def process_text_content(self, text_content: str, title: str = "Untitled") -> dict:
        self.logger.info(f"Processing document titled: '{title}'")
        paragraphs = self._chunk_text(text_content)
        if not paragraphs:
            self.logger.warning(
                "Input text was empty or could not be split into paragraphs."
            )
            return {"title": title, "paragraphs": []}
        self.logger.info(f"Split text into {len(paragraphs)} paragraphs.")

        # --- مرحله اول: جمع‌آوری داده‌ها و ساخت بانک متعارف ---
        self.logger.info("Pass 1: Collecting data and building canonical entity bank...")
        intermediate_results = []
        last_successful_prediction = None

        for text_paragraph in tqdm(
            paragraphs, desc=f"Pass 1/2: Analyzing paragraphs for '{title}'"
        ):
            # بخش طبقه‌بندی بدون تغییر باقی می‌ماند
            classification_result = self.classifier.classify(
                text_paragraph, self.allowed_sense_ids, self.allowed_age_ids
            )
            sense_pred = classification_result["sense_prediction"]
            age_pred = classification_result["age_prediction"]

            if last_successful_prediction:
                if sense_pred["confidence"] < self.confidence_threshold:
                    sense_pred = last_successful_prediction["sense_prediction"]
                if age_pred["confidence"] < self.confidence_threshold:
                    age_pred = last_successful_prediction["age_prediction"]

            # استخراج موجودیت‌های خام
            raw_entities = []
            if self.extractor and self.target_abstracts:
                for abstract, explanation in self.target_abstracts.items():
                    try:
                        instances = self.extractor.extract(
                            text_paragraph, abstract.strip(), explanation.strip()
                        )
                        for span, score, start, end in instances:
                            raw_entities.append(
                                {
                                    "type": abstract,
                                    "sample": span,
                                    "start_pos": start,
                                    "end_pos": end,
                                    "score": score,
                                }
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"ROAST failed for abstract '{abstract}' on a paragraph. Error: {e}"
                        )

            # در این مرحله، _unify_entities فقط برای آپدیت کردن بانک استفاده می‌شود
            self._unify_entities(raw_entities)

            # ذخیره نتایج میانی به همراه موجودیت‌های خام
            intermediate_result = {
                "text": text_paragraph,
                "sense_prediction": sense_pred,
                "age_prediction": age_pred,
                "raw_entities": raw_entities,  # مهم: موجودیت‌های خام را نگه می‌داریم
            }
            intermediate_results.append(intermediate_result)

            if sense_pred["class_id"] != -1 and age_pred["class_id"] != -1:
                last_successful_prediction = intermediate_result

        # --- مرحله دوم: یکپارچه‌سازی نهایی خروجی ---
        self.logger.info("Pass 2: Finalizing and unifying paragraph entities...")
        final_results = []
        for result in tqdm(
            intermediate_results, desc=f"Pass 2/2: Unifying results for '{title}'"
        ):
            # حالا با استفاده از بانک کامل شده، خروجی را یکپارچه می‌کنیم
            unified_entities = self._unify_entities(result["raw_entities"])

            final_paragraph_result = {
                "text": result["text"],
                "sense_prediction": result["sense_prediction"],
                "age_prediction": result["age_prediction"],
                "entities": unified_entities,
            }
            final_results.append(final_paragraph_result)

        final_unified_instances = { k: list() for k, _ in self.canonical_entity_bank.items() }
        for abst, grp in self.canonical_entity_bank.items():
            for _, inst in grp.items():
                final_unified_instances[abst].append(inst)

        # بازگرداندن نتایج نهایی و بانک کامل شده
        return {
            "title": title,
            "paragraphs": final_results,
            "canonical_entity_bank": final_unified_instances
        }

    def cleanup(self, genrita_driver, roast_driver):
        """
        Releases model resources from memory (RAM and VRAM) to prevent memory leaks.
        """
        self.logger.info(f"--- Cleaning up resources for GRPipeline instance ---")

        # پاک‌سازی مدل طبقه‌بندی (Classifier)
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

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared.")
        gc.collect()

        self.logger.info("--- Cleanup complete ---")

