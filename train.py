import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse  # Import argparse

from dataset import TextDataModule
from model import RoBERTaMultiTaskClassifier

# ... (Configurations remain the same) ...
MODEL_NAME = 'roberta-base'
CLEANED_DATA_PATH = 'cleaned_data_merged.csv'
BATCH_SIZE = 32
MAX_TOKEN_COUNT = 128
N_EPOCHS = 5
PATIENCE = 3
LEARNING_RATE = 2e-5
RANDOM_STATE = 42
SENSE_CLASSES_COUNT = 11
AGE_CLASSES_COUNT = 3

pl.seed_everything(RANDOM_STATE)


def main(args):
    """Main function to run the training pipeline."""
    torch.set_float32_matmul_precision('high')
    CLEANED_DATA_PATH = args.dataset_path
    print("Initializing DataModule...")
    data_module = TextDataModule(
        data_path=CLEANED_DATA_PATH, batch_size=BATCH_SIZE, max_token_len=MAX_TOKEN_COUNT,
        model_name=MODEL_NAME, random_state=RANDOM_STATE
    )
    data_module.setup()

    steps_per_epoch = len(data_module.train_dataloader())
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = int(total_training_steps * 0.1)

    print("Initializing Model...")
    model = RoBERTaMultiTaskClassifier(
        model_name=MODEL_NAME, n_sense_classes=SENSE_CLASSES_COUNT, n_age_classes=AGE_CLASSES_COUNT,
        learning_rate=LEARNING_RATE, n_training_steps=total_training_steps,
        n_warmup_steps=warmup_steps, max_token_len=MAX_TOKEN_COUNT
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=True)

    print("Initializing Trainer...")
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback], max_epochs=N_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, log_every_n_steps=10
    )

    if not args.test_only:
        # UPDATED: Use the checkpoint path from args
        print("Starting training...")
        model.roberta.train()
        trainer.fit(model, datamodule=data_module, ckpt_path=args.checkpoint_path)

    print("Starting testing...")
    # Test the best model found during this run, or the one specified if training was skipped
    test_ckpt_path = 'best' if not args.checkpoint_path else args.checkpoint_path
    trainer.test(model, datamodule=data_module, ckpt_path=test_ckpt_path)


if __name__ == '__main__':
    # NEW: Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to a specific .ckpt file to resume training from.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to a specific .csv file to resume training on this dataset.'
    )
    parser.add_argument(
        '--test_only',
        action="store_true",
        help='Set to avoid training steps and leads to direct test.'
    )
    args = parser.parse_args()
    main(args)
