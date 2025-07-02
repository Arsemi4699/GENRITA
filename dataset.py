import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    """
    Custom PyTorch Dataset for loading text and multi-task labels.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: RobertaTokenizer, max_token_len: int):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data.iloc[index]
        text = str(item.text)
        sense_label = item.sense_class_id
        age_label = item.age_class_id

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            sense_labels=torch.tensor(sense_label, dtype=torch.long),
            age_labels=torch.tensor(age_label, dtype=torch.long)
        )


class TextDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule to handle dataset loading and splitting.
    This version uses stratified splitting to maintain class distribution.
    """

    def __init__(self, data_path: str, batch_size: int, max_token_len: int, model_name: str, random_state: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.random_state = random_state
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.train_df, self.val_df, self.test_df = None, None, None

    def setup(self, stage=None):
        """Load data and perform stratified splitting."""
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            return

        # NEW: Using stratified splitting to preserve class distribution across sets.
        # We stratify by 'sense_class_id' as it has more classes.
        stratify_col = 'sense_class_id'

        # Check if stratification is possible
        if df[stratify_col].nunique() < 2:
            print("Warning: Cannot stratify with only one class. Using random split.")
            stratify_param = None
        else:
            stratify_param = df[stratify_col]

        train_val_df, self.test_df = train_test_split(
            df,
            test_size=0.1,
            random_state=self.random_state,
            stratify=stratify_param
        )

        if stratify_param is not None:
            stratify_param_val = train_val_df[stratify_col]
        else:
            stratify_param_val = None

        self.train_df, self.val_df = train_test_split(
            train_val_df,
            test_size=0.111,  # 0.111 * 0.9 ~= 0.1 of total
            random_state=self.random_state,
            stratify=stratify_param_val
        )

        print("Data loaded and split using stratification.")
        print(f"Total samples: {len(df)}")
        print(
            f"Train samples: {len(self.train_df)}, Val samples: {len(self.val_df)}, Test samples: {len(self.test_df)}")

    def train_dataloader(self):
        dataset = TextDataset(self.train_df, self.tokenizer, self.max_token_len)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        dataset = TextDataset(self.val_df, self.tokenizer, self.max_token_len)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        dataset = TextDataset(self.test_df, self.tokenizer, self.max_token_len)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)
