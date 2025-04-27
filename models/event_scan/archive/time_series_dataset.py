import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df,
        seq_len,
        is_train,
        ref_df,
        low_dim_columns,
        high_dim_columns,
        num_columns,
        emb_dim=64,
        preprocessor=None,  # ✅ Allow reuse of fitted preprocessor
    ):
        """
        Args:
            df (pd.DataFrame): Input data (train or test).
            seq_len (int): Sequence length for LSTM/CNN input.
            is_train (bool): Whether this is training data.
            ref_df (pd.DataFrame): Reference DataFrame (for shared categories).
            low_dim_columns (list): Low-dimensional categorical columns.
            high_dim_columns (list): High-dimensional categorical columns.
            num_columns (list): Numerical columns.
            emb_dim (int): Embedding dimension (only relevant for high-dim features).
            preprocessor (ColumnTransformer): Optional fitted preprocessor for test set.
        """
        self.seq_len = seq_len
        self.low_dim_columns = low_dim_columns
        self.high_dim_columns = high_dim_columns
        self.num_columns = num_columns

        self.emb_dims = [
            (len(ref_df[col].astype("category").cat.categories), emb_dim)
            for col in high_dim_columns
        ]

        # 1. One-hot encode low-dim + scale numerical
        if is_train:
            print("Fitting and transforming low-dimension features")
            self.preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "low",
                        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                        low_dim_columns,
                    ),
                    ("num", StandardScaler(), num_columns),
                ]
            )
            self.X_processed = self.preprocessor.fit_transform(df)
        else:
            print("Transforming low-dimension features using fitted preprocessor")
            self.preprocessor = preprocessor
            if self.preprocessor is None:
                raise ValueError("Must pass fitted preprocessor for test set.")
            self.X_processed = self.preprocessor.transform(df)

        # 2. Encode high-dim categorical with shared categories
        print("Encoding high-dimension categorical features")
        self.encoded_high_dim = []
        for col in high_dim_columns:
            categories = ref_df[col].astype("category").cat.categories
            df[col] = pd.Categorical(df[col], categories=categories).codes
            self.encoded_high_dim.append(df[col].values.reshape(-1, 1))

        self.encoded_high_dim = np.hstack(self.encoded_high_dim).astype(np.int64)

        self.num_cont_features = self.X_processed.shape[1]

        # 3. Combine all parts
        print("Combining features")
        self.data = np.concatenate([self.X_processed, self.encoded_high_dim], axis=1)

        # 4. Create sequences
        print("Creating sequences")
        self.samples = self.create_sequences(self.data, self.seq_len)

        if is_train:
            # Convert the full processed array to a DataFrame
            all_cols = self.preprocessor.get_feature_names_out().tolist() + [
                f"high_{col}" for col in self.high_dim_columns
            ]
            df_out = pd.DataFrame(self.data, columns=all_cols)
            df_out.attrs["num_cont_features"] = self.num_cont_features
            # Save to parquet
            df_out.to_parquet("preprocessed_train.parquet", index=False)
            print("Saved preprocessed dataset to 'preprocessed_train.parquet'")

    def create_sequences(self, data, seq_len):
        return [data[i : i + seq_len] for i in range(len(data) - seq_len + 1)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cont_len = self.num_cont_features
        cont_part = sample[:, :cont_len]
        cat_part = sample[:, cont_len:].astype(
            np.int64
        )  # Make sure it's int for embedding

        return {
            "cont": torch.tensor(cont_part, dtype=torch.float32),
            "cat": [
                torch.tensor(cat_part[:, i], dtype=torch.long)
                for i in range(cat_part.shape[1])
            ],
        }
