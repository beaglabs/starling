"""Starlings: small, local decision models."""

from .remote import RemoteFactor, RemoteFactorResult
from .schemas import Candidate, Content, EvaluationResult, Question, SourceBlock
from .sdk import Classifier
from .trainer import EvaluationReport, FactorSpec, Trainer, TrainingConfig, TrainingResult

__version__ = "0.1.0"
__all__ = [
    "Classifier",
    "Trainer",
    "FactorSpec",
    "TrainingConfig",
    "TrainingResult",
    "EvaluationReport",
    "RemoteFactor",
    "RemoteFactorResult",
    "Candidate",
    "Content",
    "EvaluationResult",
    "Question",
    "SourceBlock",
]
