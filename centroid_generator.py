import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score
from GENRITA import TEST_CASES


# --- Model and File Configuration ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TRAINING_CSV_PATH = "data/cleaned_data_merged.csv"

# --- Task-Specific Configurations ---
SENSE_CONFIG = {
    "task_name": "Sense",
    "class_id_col": "sense_class_id",
    "prediction_key": "sense_prediction",
    "centroids_path": "sense_centroids.npy",
    "class_names": [
        "Normal and neutral",
        "Love and romantic",
        "War and combat",
        "Fantasy and mythology",
        "Honor and respect",
        "Drama and tragedy",
        "City and Crowd",
        "Mountain and the heights",
        "Desert and dunes",
        "Sea and tides",
        "Forest and tress",
    ],
}

AGE_CONFIG = {
    "task_name": "Age",
    "class_id_col": "age_class_id",
    "prediction_key": "age_prediction",
    "centroids_path": "age_centroids.npy",
    "class_names": [
        "ancient and old age",
        "neutral and not special age (non-ancient, non technology)",
        "technology modern age",
    ],
}


def create_and_save_centroids(
    df: pd.DataFrame, embedder: SentenceTransformer, class_id_col: str, output_path: str
) -> np.ndarray:
    """
    Loads data, computes embeddings, calculates class centroids for a given task,
    and saves them to a file.
    """
    print(f"--- Starting Centroid Creation for '{class_id_col}' ---")

    # Compute Embeddings if not already present
    if "embedding" not in df.columns:
        print("Embedding texts for centroid calculation...")
        embeddings = embedder.encode(
            df["text"].tolist(), convert_to_numpy=True, show_progress_bar=True
        )
        df["embedding"] = list(embeddings)
    else:
        print("Embeddings already exist, skipping re-computation.")
        embeddings = np.vstack(df["embedding"].values)

    # Compute Centroids
    print(f"Calculating centroids for each class in '{class_id_col}'...")
    unique_class_ids = sorted(df[class_id_col].unique())
    num_classes = len(unique_class_ids)
    embedding_size = embeddings.shape[1]
    centroids = np.zeros((num_classes, embedding_size))

    for class_id in unique_class_ids:
        class_embeddings = np.vstack(
            df[df[class_id_col] == class_id]["embedding"].values
        )
        centroids[class_id] = np.mean(class_embeddings, axis=0)

    print(f"Centroids created with shape: {centroids.shape}")

    # Save Centroids
    np.save(output_path, centroids)
    print(f"Centroids saved to {output_path}")
    print(f"--- Centroid Creation for '{class_id_col}' Finished ---\n")

    return centroids, df


def evaluate_with_centroids(
    test_cases: list, config: dict, embedder: SentenceTransformer, centroids: np.ndarray
):
    """
    Evaluates the centroid-based classifier on a set of test cases for a given task.
    """
    task_name = config["task_name"]
    class_names = config["class_names"]
    prediction_key = config["prediction_key"]

    print(f"--- Starting Evaluation for: {task_name} Classification ---")

    texts = [case["text"] for case in test_cases]
    true_labels = [case[prediction_key]["class_id"] for case in test_cases]

    # Get embeddings for test cases
    print(f"Embedding {len(texts)} test cases for {task_name}...")
    test_embeddings = embedder.encode(
        texts, convert_to_numpy=True, show_progress_bar=True
    )

    # Predict labels based on closest centroid
    pred_labels = []
    for vec in test_embeddings:
        sims = cosine_similarity([vec], centroids)[0]
        pred_class = int(np.argmax(sims))
        pred_labels.append(pred_class)

    print("\n" + "=" * 50)
    print(f"           EVALUATION RESULTS FOR: {task_name.upper()}")
    print("=" * 50)
    print(f"\nAccuracy: {accuracy_score(true_labels, pred_labels):.2%}")
    print("\nClassification Report:\n")
    print(
        classification_report(
            true_labels, pred_labels, target_names=class_names, digits=2
        )
    )
    print("=" * 50)
    print(f"--- Evaluation for {task_name} Finished ---\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load the embedding model once
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    main_embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # Load and preprocess data once
    print(f"Loading and validating training data from {TRAINING_CSV_PATH}...")
    main_df = pd.read_csv(TRAINING_CSV_PATH)
    assert {"text", "sense_class_id", "age_class_id"}.issubset(
        main_df.columns
    ), "CSV missing required columns"

    # --- Process SENSE classification ---
    sense_centroids, df_with_embeddings = create_and_save_centroids(
        df=main_df,
        embedder=main_embedder,
        class_id_col=SENSE_CONFIG["class_id_col"],
        output_path=SENSE_CONFIG["centroids_path"],
    )
    if len(TEST_CASES) > 0:
        evaluate_with_centroids(
            test_cases=TEST_CASES,
            config=SENSE_CONFIG,
            embedder=main_embedder,
            centroids=sense_centroids,
        )

    # --- Process AGE classification ---
    age_centroids, _ = create_and_save_centroids(
        df=df_with_embeddings,
        embedder=main_embedder,
        class_id_col=AGE_CONFIG["class_id_col"],
        output_path=AGE_CONFIG["centroids_path"],
    )
    if len(TEST_CASES) > 0:
        evaluate_with_centroids(
            test_cases=TEST_CASES,
            config=AGE_CONFIG,
            embedder=main_embedder,
            centroids=age_centroids,
        )
