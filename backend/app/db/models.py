# backend/app/db/models.py
from enum import Enum as PyEnum # Renamed to avoid conflict if you use Enum from sqlalchemy elsewhere directly
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum as SQLEnum, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base # Assuming your Base is defined in db.database
from passlib.context import CryptContext
import secrets
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB

class FeatureTypeEnum(str, PyEnum):
    AUTOENCODER = "autoencoder"
    PCA = "pca"
    ISOMAP = "isomap"

class FeatureSet(Base):
    __tablename__ = "feature_sets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    name = Column(String(128), nullable=False)
    path = Column(String, nullable=False)
    feature_type = Column(SQLEnum(FeatureTypeEnum, name="feature_type_enum"), nullable=False)
    description = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="feature_sets")
    dataset = relationship("Dataset", back_populates="feature_sets")

    __table_args__ = (
        # Unique constraint: user cannot have two feature sets with the same name for the same dataset
        # (user_id, dataset_id, name) must be unique
        {
            'sqlite_autoincrement': True,
        },
    )

# Add relationship to User and Dataset below

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRole(str, PyEnum): # Use the renamed PyEnum
    ADMIN = "admin"
    USER = "user"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole, name="user_role_enum_models"), default=UserRole.USER) # Added unique name for SQLEnum
    company = Column(String(100), nullable=True) # Made nullable=True consistent with your previous schema
    phone_number = Column(String(20), nullable=True) # Made nullable=True
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    reset_token = Column(String, unique=True, nullable=True)

    # Relationship to datasets (one-to-many)
    datasets = relationship("Dataset", back_populates="user_owner", cascade="all, delete-orphan") # Changed back_populates for clarity
    
    # --- ADDED THIS RELATIONSHIP ---
    outlier_detection_runs = relationship("OutlierDetectionRun", back_populates="user", cascade="all, delete-orphan")
    transformation_runs = relationship("TransformationRun", back_populates="user", cascade="all, delete-orphan")
    feature_sets = relationship("FeatureSet", back_populates="user", cascade="all, delete-orphan")
    clustering_results = relationship("ClusteringResult", back_populates="user", cascade="all, delete-orphan")

    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)

    def set_password(self, password: str):
        self.hashed_password = pwd_context.hash(password)

    def generate_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)

    def clear_reset_token(self):
        self.reset_token = None

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_type = Column(String(10), nullable=False)
    file_path = Column(String, nullable=False)
    format = Column(String, nullable=True)
    missing_values = Column(JSONB, nullable=True)
    duplicates = Column(Integer, default=0)
    data_types = Column(JSONB, nullable=True)
    categorical_issues = Column(JSONB, nullable=True)
    summary_stats = Column(JSONB, nullable=True)
    analysis_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user_owner = relationship("User", back_populates="datasets")
    feature_sets = relationship("FeatureSet", back_populates="dataset", cascade="all, delete-orphan")
    outlier_detection_runs = relationship("OutlierDetectionRun", back_populates="dataset", cascade="all, delete-orphan")
    transformation_runs = relationship("TransformationRun", back_populates="dataset", cascade="all, delete-orphan")
    clustering_results = relationship("ClusteringResult", back_populates="dataset", cascade="all, delete-orphan")


class OutlierDetectionRun(Base):
    __tablename__ = "outlier_detection_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    task_id = Column(String, unique=True, index=True, nullable=True) # Task ID from background task system (if any)
    status = Column(String, default="queued", nullable=False) # e.g., queued, processing, completed, failed
    
    parameters_json = Column(JSONB, nullable=True, name="parameters") # Store input parameters as JSON

    # Paths to saved artifacts
    latent_features_path = Column(String, nullable=True)
    reconstruction_errors_path = Column(String, nullable=True)
    autoencoder_model_path = Column(String, nullable=True) # Path to the full AE model
    encoder_model_path = Column(String, nullable=True)     # Path to the standalone encoder model
    scaler_path = Column(String, nullable=True)            # Path to the scaler for original numerical data
    
    # ADDED: Path for scaler used on latent features before clustering
    clustering_latent_features_scaler_path = Column(String, nullable=True)
    
    cluster_labels_path = Column(String, nullable=True)
    scatter_plot_pca_path = Column(String, nullable=True) # Path to the main visualization
    outlier_distribution_path = Column(String, nullable=True) # Path to the outlier score distribution plot
    pca_plot_path = Column(String, nullable=True) # Path to the PCA visualization with outliers highlighted

    # Summary statistics from the run
    total_points_processed = Column(Integer, nullable=True)
    num_numerical_features_used = Column(Integer, nullable=True)
    num_clusters_found = Column(Integer, nullable=True)
    num_noise_points = Column(Integer, nullable=True)
    total_outliers_detected = Column(Integer, nullable=True)
    
    # Detailed results (e.g., list of dicts for each point: original_index, is_outlier, scores)
    outlier_results_json = Column(JSONB, nullable=True, name="outlier_results")

    # ADDED: Field to store evaluation metrics if ground truth was provided
    evaluation_metrics_json = Column(JSONB, nullable=True, name="evaluation_metrics")

    started_at = Column(DateTime, default=datetime.utcnow) # Changed to datetime.utcnow for consistency if not using server default
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String(1024), nullable=True) # Store error message if run failed

    # Relationships - ensure these back_populates match what's in User and Dataset models
    dataset = relationship("Dataset", back_populates="outlier_detection_runs")
    user = relationship("User", back_populates="outlier_detection_runs")

    def __repr__(self):
        return f"<OutlierDetectionRun(id={self.id}, dataset_id={self.dataset_id}, status='{self.status}')>"


class TransformationRun(Base):
    __tablename__ = "transformation_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Original file info
    original_filename = Column(String, nullable=False)
    original_file_path = Column(String, nullable=False)
    
    # Transformed file info
    transformed_filename = Column(String, nullable=False)
    transformed_file_path = Column(String, nullable=False)
    
    # Transformation configuration
    transformation_config = Column(JSONB, nullable=False)
    
    # Status and timestamps
    status = Column(String, default="completed", nullable=False)  # or 'failed'
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="transformation_runs")
    user = relationship("User", back_populates="transformation_runs")
    
    def __repr__(self):
        return f"<TransformationRun(id={self.id}, dataset_id={self.dataset_id}, status='{self.status}')>"


class ClusteringResult(Base):
    __tablename__ = "clustering_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)  # Nullable for direct file uploads
    algorithm = Column(String(50), nullable=False)  # dbscan, optics, denclue
    parameters = Column(JSONB, nullable=True)  # JSON of parameters
    result_path = Column(String, nullable=True)  # Path to the CSV file with results
    result_metadata = Column(JSONB, nullable=True)  # JSON of additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="clustering_results")
    dataset = relationship("Dataset", back_populates="clustering_results")
    
    def __repr__(self):
        return f"<ClusteringResult(id={self.id}, dataset_id={self.dataset_id}, algorithm='{self.algorithm}')>"
