import numpy as np
import re
import logging
import torch
from sentence_transformers import SentenceTransformer


class ClassificationPostRefinement:
    """
    Final multi-task architecture: Implements a Confidence-Weighted Rank Combination
    for BOTH sense and age classifications.
    """

    def __init__(
        self,
        logger,
        # --- Sense Parameters ---
        sense_neutral_id: int = 0,
        sense_centroids_path: str = "sense_centroids.npy",
        sense_entropy_thr: float = 1.1,
        # --- Age Parameters ---
        age_neutral_id: int = 1,
        age_centroids_path: str = "age_centroids.npy",
        age_entropy_thr: float = 0.5,  # Age has fewer classes, so entropy is naturally lower
    ):
        self.logger = logger if logger else logging.getLogger(__name__)

        # --- Sense Configuration ---
        self.sense_neutral_id = sense_neutral_id
        self.sense_entropy_thr = sense_entropy_thr

        # --- Age Configuration ---
        self.age_neutral_id = age_neutral_id
        self.age_entropy_thr = age_entropy_thr

        try:
            self.logger.info(f"Loading sense centroids from: {sense_centroids_path}")
            self.sense_centroids = np.load(sense_centroids_path)

            self.logger.info(f"Loading age centroids from: {age_centroids_path}")
            self.age_centroids = np.load(age_centroids_path)

            embedder_model_name = "sentence-transformers/all-mpnet-base-v2"
            self.logger.info(f"Loading sentence embedding model: {embedder_model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedder = SentenceTransformer(embedder_model_name, device=device)

        except FileNotFoundError as e:
            self.logger.error(
                f"A required model file was not found: {e}. The postprocessor cannot function."
            )
            raise

        self.logger.info("ClassificationPostRefinement (Multi-Task Strategy) is ready.")

    def _get_ranks(self, scores_matrix: np.ndarray) -> np.ndarray:
        return np.argsort(-scores_matrix, axis=1).argsort(axis=1)

    def _calculate_entropy_margin(
        self, probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        log_probs = np.log(probs + 1e-9)
        entropy = -np.sum(probs * log_probs, axis=1)
        sorted_probs = -np.sort(-probs, axis=1)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return entropy, margin

    def process_book(
        self, raw_outputs: list[dict]
    ) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
        if not raw_outputs:
            return [], [], np.array([]), np.array([])

        self.logger.info("--- Starting Multi-Task Post-processing ---")

        # --- STEP 1: Get all inputs in batches ---
        all_sense_probs = np.vstack([out["sense_probs"] for out in raw_outputs])
        all_age_probs = np.vstack([out["age_probs"] for out in raw_outputs])
        texts = [out["text"] for out in raw_outputs]
        n_paragraphs = len(texts)

        # 1a. Get RoBERTa's confidence metrics for both tasks
        sense_entropy, sense_margin = self._calculate_entropy_margin(all_sense_probs)
        age_entropy, age_margin = self._calculate_entropy_margin(all_age_probs)

        # 1b. Get Similarity Gate's scores for both tasks
        self.logger.info(f"Embedding {n_paragraphs} paragraphs...")
        text_embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

        # Sense Similarities
        sense_sims = np.dot(text_embeddings, self.sense_centroids.T) / (
            np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            * np.linalg.norm(self.sense_centroids, axis=1).T
        )

        # Age Similarities
        age_sims = np.dot(text_embeddings, self.age_centroids.T) / (
            np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            * np.linalg.norm(self.age_centroids, axis=1).T
        )

        # --- STEP 2: Get Ranks from both models for both tasks ---
        roberta_sense_ranks = self._get_ranks(all_sense_probs)
        similarity_sense_ranks = self._get_ranks(sense_sims)

        roberta_age_ranks = self._get_ranks(all_age_probs)
        similarity_age_ranks = self._get_ranks(age_sims)

        final_sense_labels = []
        final_age_labels = []
        self.logger.info(
            "Applying confidence-weighted rank combination to each paragraph for both tasks..."
        )

        for i in range(n_paragraphs):
            # --- Process SENSE ---
            if sense_entropy[i] > self.sense_entropy_thr:
                final_sense_labels.append(self.sense_neutral_id)
            else:
                weight_r = 1.0 + sense_margin[i]
                weight_s = 1.0
                rank_sum = (weight_r * roberta_sense_ranks[i]) + (
                    weight_s * similarity_sense_ranks[i]
                )
                winner = np.argmin(rank_sum)
                final_sense_labels.append(winner)

            # --- Process AGE ---
            if age_entropy[i] > self.age_entropy_thr:
                final_age_labels.append(self.age_neutral_id)
            else:
                weight_r = 1.0 + age_margin[i]
                weight_s = 1.0
                rank_sum = (weight_r * roberta_age_ranks[i]) + (
                    weight_s * similarity_age_ranks[i]
                )
                winner = np.argmin(rank_sum)
                final_age_labels.append(winner)

        self.logger.info("--- Multi-Task Post-processing Finished ---")
        return final_sense_labels, final_age_labels, all_sense_probs, all_age_probs
