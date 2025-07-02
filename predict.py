import argparse
from model import RoBERTaMultiTaskClassifier
import json

# Define the class mappings to decode predictions
# You should get these from your initial data processing script
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
# Create reverse maps for decoding
SENSE_ID_TO_NAME = {v: k for k, v in SENSE_CLASSES.items()}
AGE_ID_TO_NAME = {v: k for k, v in AGE_CLASSES.items()}


def predict_text(checkpoint_path: str, text: str):
    """
    Loads a model from a checkpoint and predicts on a given text.
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # Load the model from the checkpoint
    trained_model = RoBERTaMultiTaskClassifier.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    trained_model.freeze()  # Freeze weights for faster inference

    print(f"Predicting on text: '{text}'")
    prediction = trained_model.predict(
        text,
        sense_id_map=SENSE_ID_TO_NAME,
        age_id_map=AGE_ID_TO_NAME
    )

    print("\n--- Prediction Result ---")
    # Use json.dumps for pretty printing the dictionary
    print(json.dumps(prediction, indent=4))
    print("-------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using a fine-tuned RoBERTa model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model .ckpt file."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The text string to classify."
    )
    args = parser.parse_args()

    predict_text(args.checkpoint_path, args.text)
