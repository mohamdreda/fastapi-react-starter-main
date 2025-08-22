"""
Clustering Result Model
"""
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..database import Base

class ClusteringResult(Base):
    __tablename__ = "clustering_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    algorithm = Column(String(50))  # dbscan, optics, denclue
    parameters = Column(Text)  # JSON string of parameters
    result_path = Column(String(255))  # Path to the CSV file with results
    metadata = Column(Text)  # JSON string of additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="clustering_results")
    dataset = relationship("Dataset", back_populates="clustering_results")
