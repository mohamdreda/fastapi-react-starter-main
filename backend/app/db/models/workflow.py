from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from ..database import Base
from uuid import uuid4
from datetime import datetime

class WorkflowTemplate(Base):
    __tablename__ = "workflow_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(128), nullable=False)
    version = Column(Integer, default=1)
    description = Column(String(256))
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    steps = Column(JSONB, nullable=False)  # ordered list of dicts
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User")
    __table_args__ = (UniqueConstraint("name", "version", "owner_id"),)

class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(UUID(as_uuid=True), ForeignKey("workflow_templates.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(20), default="queued")
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)

    template = relationship("WorkflowTemplate")
    dataset = relationship("Dataset")
    owner = relationship("User")
    steps = relationship("WorkflowStepRun", back_populates="run", cascade="all, delete-orphan", order_by="WorkflowStepRun.order")

class WorkflowStepRun(Base):
    __tablename__ = "workflow_step_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("workflow_runs.id"), nullable=False)
    order = Column(Integer, nullable=False)
    step = Column(String(64), nullable=False)
    substep = Column(String(64))
    algorithm = Column(String(64), nullable=False)
    params = Column(JSONB)
    status = Column(String(20), default="running")
    elapsed_ms = Column(Integer)
    metrics = Column(JSONB)
    visuals = Column(JSONB)  # list of plot refs
    error = Column(String(1024))

    run = relationship("WorkflowRun", back_populates="steps")

class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(128))
    description = Column(String(512))
    created_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)

    owner = relationship("User")
    steps = relationship(
        "SessionStep",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="SessionStep.order",
    )

class SessionStep(Base):
    __tablename__ = "session_steps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    order = Column(Integer, nullable=False)

    # Generic tool/run descriptor
    tool = Column(String(64), nullable=False)           # e.g., 'outliers', 'transformation'
    step = Column(String(64), nullable=False)           # e.g., 'detect', 'fit_transform'
    substep = Column(String(64))
    algorithm = Column(String(64))
    params = Column(JSONB)

    # Runtime and linkage to underlying run record
    status = Column(String(20), default="queued")
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)
    error = Column(String(1024))

    run_ref_type = Column(String(32))                   # e.g., 'outliers'
    run_ref_id = Column(String(64))                     # store as string to support int/UUID

    session = relationship("Session", back_populates="steps")
