from enum import Enum
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum as SQLEnum, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base
from passlib.context import CryptContext
import secrets
from datetime import datetime

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    company = Column(String(100))
    phone_number = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    reset_token = Column(String, unique=True, nullable=True)

    # Relationship to datasets (one-to-many)
    datasets = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")

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
    file_type = Column(String(10), nullable=False)  # csv, xlsx, etc.
    file_path = Column(String, nullable=False)
    format = Column(String, nullable=False)  # auto-detect, manual, etc.
    missing_values = Column(JSON, nullable=True)  # JSON object with missing value stats
    duplicates = Column(Integer, default=0)
    data_types = Column(JSON, nullable=True)  # JSON object with column data types
    categorical_issues = Column(JSON, nullable=True)  # JSON object with categorical issues
    summary_stats = Column(JSON, nullable=True)  # JSON object with summary statistics
    analysis_metadata = Column(JSON, nullable=True)  # Additional metadata from enhanced data quality analysis
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow) 
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    user = relationship("User", back_populates="datasets")