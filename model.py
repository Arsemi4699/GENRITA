import torch
from torch import nn
import pytorch_lightning as pl
from transformers import RobertaModel, get_linear_schedule_with_warmup, RobertaTokenizer
import torchmetrics
from torch.optim import AdamW

from data_processor import DataProcessor

class ClassificationHead(nn.Module):
    """A more advanced classification head with multiple layers and dropout."""

    def __init__(self, hidden_size: int, num_labels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.activation = nn.ReLU()

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RoBERTaMultiTaskClassifier(pl.LightningModule):
    """The main model class with advanced heads, comprehensive metrics, and a predict method."""

    def __init__(self, model_name: str, n_sense_classes: int, n_age_classes: int, learning_rate: float,
                 n_training_steps: int, n_warmup_steps: int, max_token_len: int = 128):
        super().__init__()
        self.save_hyperparameters()

        self.roberta = RobertaModel.from_pretrained(model_name, return_dict=True, local_files_only=True)
        # Tokenizer needs to be part of the model for easy prediction
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name, local_files_only=True)

        self.sense_classifier = ClassificationHead(self.roberta.config.hidden_size, n_sense_classes)
        self.age_classifier = ClassificationHead(self.roberta.config.hidden_size, n_age_classes)

        self.criterion = nn.CrossEntropyLoss()

        metric_collection = lambda num_classes, prefix: torchmetrics.MetricCollection({
            'acc': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            'f1': torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro'),
        }, prefix=prefix)

        self.train_sense_metrics = metric_collection(n_sense_classes, 'train_sense_')
        self.val_sense_metrics = metric_collection(n_sense_classes, 'val_sense_')
        self.test_sense_metrics = metric_collection(n_sense_classes, 'test_sense_')

        self.train_age_metrics = metric_collection(n_age_classes, 'train_age_')
        self.val_age_metrics = metric_collection(n_age_classes, 'val_age_')
        self.test_age_metrics = metric_collection(n_age_classes, 'test_age_')

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        sense_logits = self.sense_classifier(pooled_output)
        age_logits = self.age_classifier(pooled_output)
        return sense_logits, age_logits

    # ... (training, validation, test steps and configure_optimizers remain the same) ...
    # Omitted for brevity, they are identical to the previous version.
    def _shared_step(self, batch):
        sense_logits, age_logits = self(batch["input_ids"], batch["attention_mask"])
        loss_sense = self.criterion(sense_logits, batch["sense_labels"])
        loss_age = self.criterion(age_logits, batch["age_labels"])
        total_loss = loss_sense + loss_age
        return total_loss, sense_logits, age_logits

    def training_step(self, batch, batch_idx):
        total_loss, sense_logits, age_logits = self._shared_step(batch)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_sense_metrics.update(sense_logits, batch["sense_labels"])
        self.train_age_metrics.update(age_logits, batch["age_labels"])
        return total_loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_sense_metrics.compute())
        self.log_dict(self.train_age_metrics.compute())
        self.train_sense_metrics.reset()
        self.train_age_metrics.reset()

    def validation_step(self, batch, batch_idx):
        total_loss, sense_logits, age_logits = self._shared_step(batch)
        self.log("val_loss", total_loss, prog_bar=True, logger=True)
        self.val_sense_metrics.update(sense_logits, batch["sense_labels"])
        self.val_age_metrics.update(age_logits, batch["age_labels"])

    def on_validation_epoch_end(self):
        self.log_dict(self.val_sense_metrics.compute(), prog_bar=True)
        self.log_dict(self.val_age_metrics.compute(), prog_bar=True)
        self.val_sense_metrics.reset()
        self.val_age_metrics.reset()

    def test_step(self, batch, batch_idx):
        total_loss, sense_logits, age_logits = self._shared_step(batch)
        self.log("test_loss", total_loss, logger=True)
        self.test_sense_metrics.update(sense_logits, batch["sense_labels"])
        self.test_age_metrics.update(age_logits, batch["age_labels"])

    def on_test_epoch_end(self):
        self.log_dict(self.test_sense_metrics.compute())
        self.log_dict(self.test_age_metrics.compute())
        self.test_sense_metrics.reset()
        self.test_age_metrics.reset()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.n_warmup_steps,
            num_training_steps=self.hparams.n_training_steps
        )
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))

    def predict(self, text: str, sense_id_map: dict, age_id_map: dict):
        """
        Takes a raw text string and returns predictions for both tasks.
        """
        # Set model to evaluation mode
        self.eval()

        # 1. Preprocess the text using the same cleaner as in training
        cleaned_text = DataProcessor._clean_text(text)

        # 2. Tokenize the cleaned text
        encoding = self.tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.hparams.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 3. Perform inference
        with torch.no_grad():
            sense_logits, age_logits = self(input_ids, attention_mask)

        # 4. Get probabilities and predictions
        sense_probs = torch.softmax(sense_logits, dim=1)
        age_probs = torch.softmax(age_logits, dim=1)

        sense_pred_id = torch.argmax(sense_probs, dim=1).item()
        age_pred_id = torch.argmax(age_probs, dim=1).item()

        sense_pred_name = sense_id_map[sense_pred_id]
        age_pred_name = age_id_map[age_pred_id]

        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "sense_prediction": {
                "class_name": sense_pred_name,
                "class_id": sense_pred_id,
                "confidence": sense_probs.max().item()
            },
            "age_prediction": {
                "class_name": age_pred_name,
                "class_id": age_pred_id,
                "confidence": age_probs.max().item()
            }
        }
