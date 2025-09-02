import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


GT_PATH = "./docs/test_story_gt.csv"
JOB_ID = "3bba7424-e14c-4d04-b70c-4a08dcf70f8a"
PR_PATH = f"./output_results/{JOB_ID}.json"


def calculate_and_print_metrics(y_true, y_pred, target_names, model_name=""):
    """
    Calculates and prints classification metrics.

    Args:
        y_true (list): List of true labels.
        y_pred (list): List of predicted labels.
        target_names (list): List of class names for the report.
        model_name (str): Name of the model/task for printing.
    """
    print(f"--- METRICS FOR: {model_name} ---")

    # --- Accuracy ---
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # --- Classification Report (Precision, Recall, F1-Score) ---
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    # --- Confusion Matrix ---
    # Ensure labels are consistent for the confusion matrix
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[target_names[i] for i in labels],
        columns=[target_names[i] for i in labels],
    )

    print("\nConfusion Matrix:")
    print(cm_df)
    print("-" * (25 + len(model_name)))


def main():
    """
    Main function to load data and calculate metrics.
    """
    try:
        gt_df = pd.read_csv(GT_PATH)

        with open(PR_PATH, "r") as f:
            predictions_data = json.load(f)

        paragraph_preds = predictions_data.get("paragraphs", [])

        if len(paragraph_preds) != len(gt_df):
            print(
                f"Warning: Number of predictions ({len(paragraph_preds)}) does not match ground truth ({len(gt_df)})."
            )
            min_len = min(len(paragraph_preds), len(gt_df))
            gt_df = gt_df.head(min_len)
            paragraph_preds = paragraph_preds[:min_len]

        # --- Prepare Lists for Metrics ---
        # Age Prediction
        y_true_age = gt_df["age_class_id"].tolist()
        y_pred_age = [p["age_prediction"]["class_id"] for p in paragraph_preds]

        # Sense Prediction
        y_true_sense = gt_df["sense_class_id"].tolist()
        y_pred_sense = [p["sense_prediction"]["class_id"] for p in paragraph_preds]

        age_target_names = (
            gt_df.drop_duplicates(subset=["age_class_id"])
            .sort_values("age_class_id")
            .set_index("age_class_id")["age_class_name"]
            .to_dict()
        )
        sense_target_names = (
            gt_df.drop_duplicates(subset=["sense_class_id"])
            .sort_values("sense_class_id")
            .set_index("sense_class_id")["sense_class_name"]
            .to_dict()
        )

        age_target_names_list = [
            age_target_names[i] for i in sorted(age_target_names.keys())
        ]
        sense_target_names_list = [
            sense_target_names[i] for i in sorted(sense_target_names.keys())
        ]

        # --- Calculate and Print Metrics ---
        calculate_and_print_metrics(
            y_true_age, y_pred_age, age_target_names_list, "Age Classification"
        )
        print("\n\n")
        calculate_and_print_metrics(
            y_true_sense, y_pred_sense, sense_target_names_list, "Sense Classification"
        )

    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please make sure 'ground_truth.csv' and 'predictions.json' are in the same directory."
        )
    except (KeyError, IndexError) as e:
        print(
            f"Error processing the data files: {e}. Please check if the file formats are correct."
        )


if __name__ == "__main__":
    main()
