# backend/app/services/autoencoder_service.py
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Any, Dict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
import joblib

from app.config.config import get_settings
settings = get_settings()

def _get_ae_artifact_path(
    base_path: str, 
    dataset_id: int, 
    user_id: int, 
    artifact_name: str
) -> str:
    dir_path = os.path.join(base_path, f"user_{user_id}", f"dataset_{dataset_id}", "autoencoder")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, artifact_name)

class AutoencoderService:
    def __init__(
            self,
            dataset_id: int,
            user_id: int,
            input_dim: int,
            latent_dim: int = 16,
            epochs: int = 50,
            batch_size: int = 32,
            optimizer: str = 'adam',
            loss_function: str = 'mse'
        ):
            self.dataset_id = dataset_id
            self.user_id = user_id
            self.input_dim = input_dim
            self.latent_dim = min(latent_dim, input_dim)  # Ensure latent_dim <= input_dim
            self.epochs = epochs  # Kept for API compatibility
            self.batch_size = batch_size  # Kept for API compatibility
            self.optimizer = optimizer  # Kept for API compatibility
            self.loss = loss_function  # Kept for API compatibility
            
            self.scaler = StandardScaler()
            self.autoencoder = None  # Will be built in build_and_compile_model
            self.encoder = None

            self.ARTIFACT_BASE = settings.OUTLIER_ARTIFACTS_BASE_PATH 
            self.model_save_path = _get_ae_artifact_path(self.ARTIFACT_BASE, dataset_id, user_id, "autoencoder_model.keras")
            self.scaler_save_path = _get_ae_artifact_path(self.ARTIFACT_BASE, dataset_id, user_id, "scaler.joblib")

    def build_and_compile_model(self):
        """Build and compile a Keras autoencoder model."""
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(self.latent_dim, activation="relu")(input_layer)
        # Decoder
        decoded = Dense(self.input_dim, activation="linear")(encoded)
        # Autoencoder
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=self.optimizer, loss=self.loss)
        self.autoencoder = autoencoder
        # Encoder for feature extraction
        self.encoder = Model(input_layer, encoded)

    def train_model(self, X_train: pd.DataFrame, X_val: pd.DataFrame, return_history: bool = False):
        if self.autoencoder is None:
            self.build_and_compile_model()

        # Keep only numeric columns for training
        orig_cols = X_train.columns.tolist()
        X_train = X_train.select_dtypes(include=["number"])
        X_val = X_val[X_train.columns] if not X_val.empty else X_val
        dropped_cols = list(set(orig_cols) - set(X_train.columns))
        if dropped_cols:
            print(f"AE_SERVICE WARNING: Dropping non-numeric columns for autoencoder training: {dropped_cols}")

        # Rebuild the model if input_dim changed after dropping columns
        actual_input_dim = X_train.shape[1]
        if self.input_dim != actual_input_dim:
            print(f"AE_SERVICE: Adjusting input_dim from {self.input_dim} to {actual_input_dim} due to dropped columns.")
            self.input_dim = actual_input_dim
            self.latent_dim = min(self.latent_dim, self.input_dim)
            self.build_and_compile_model()

        print(f"AE_SERVICE: Starting Autoencoder training for dataset {self.dataset_id}, user {self.user_id}...")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        history = self.autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(X_val_scaled, X_val_scaled),
            verbose=1
        )

        self.autoencoder.save(self.model_save_path)
        joblib.dump(self.scaler, self.scaler_save_path)
        print(f"AE_SERVICE: Autoencoder training finished. Model saved to: {self.model_save_path}")
        print(f"AE_SERVICE: Scaler saved to: {self.scaler_save_path}")

        if return_history:
            return history
        return self.model_save_path

    def extract_latent_features(self, data: pd.DataFrame) -> pd.DataFrame:
        from tensorflow.keras.models import load_model
        if self.autoencoder is None:
            self.autoencoder = load_model(self.model_save_path)
        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_save_path)
        X_scaled = self.scaler.transform(data)
        # Get encoded (latent) features
        encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.layers[1].output)
        latent_features = encoder.predict(X_scaled)
        columns = [f'latent_{i+1}' for i in range(self.latent_dim)]
        return pd.DataFrame(latent_features, columns=columns, index=data.index)

    def calculate_reconstruction_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-row reconstruction error using the trained Keras autoencoder.
        Returns a DataFrame with a single column 'reconstruction_error' indexed like input data.
        """
        from tensorflow.keras.models import load_model
        # Ensure model and scaler are loaded
        if self.autoencoder is None:
            if os.path.exists(self.model_save_path):
                print(f"AE_SERVICE: Loading autoencoder model from: {self.model_save_path}")
                self.autoencoder = load_model(self.model_save_path)
            else:
                raise ValueError("AE_SERVICE: Autoencoder model not found. Train the model first.")
        if self.scaler is None:
            if os.path.exists(self.scaler_save_path):
                print(f"AE_SERVICE: Loading scaler from: {self.scaler_save_path}")
                self.scaler = joblib.load(self.scaler_save_path)
            else:
                raise ValueError("AE_SERVICE: Scaler not found. Train the model to fit and save the scaler.")

        # Use numeric columns and align to scaler's expectation
        data_numeric = data.select_dtypes(include=["number"])
        X_scaled = self.scaler.transform(data_numeric)

        reconstructed = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        return pd.DataFrame({"reconstruction_error": mse}, index=data_numeric.index)

    def get_model_paths(self):
        """Get paths to saved model artifacts."""
        return {
            'autoencoder_model': self.model_save_path,
            'scaler': self.scaler_save_path
        }