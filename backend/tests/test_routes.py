# Add this to your FastAPI app (e.g., in a new file like test_routes.py)

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from app.services.outlier_detection.feature_extraction.autoencoder import AutoencoderService
from sklearn.metrics import mean_squared_error

router = APIRouter()

@router.post("/test-autoencoder")
async def test_autoencoder():
    try:
        # Create a sample dataset
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        X = np.random.rand(n_samples, n_features)
        
        # Add some anomalies
        X[-10:] += np.random.normal(0, 0.5, size=(10, n_features))
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        
        # Split into train/validation
        X_train = df.iloc[:800]
        X_val = df.iloc[800:]
        
        # Initialize and train autoencoder
        ae_service = AutoencoderService(
            dataset_id=999,  # test dataset ID
            user_id=1,       # test user ID
            input_dim=n_features,
            latent_dim=8,
            epochs=10,       # reduced for testing
            batch_size=32
        )
        
        # Build and train
        ae_service.build_and_compile_model()
        model_path = ae_service.train_model(X_train, X_val)
        
        # Extract features
        latent_features = ae_service.extract_latent_features(df)
        
        return {
            "status": "success",
            "model_path": model_path,
            "latent_features_shape": latent_features.shape,
            "sample_latent_features": latent_features.head().to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))