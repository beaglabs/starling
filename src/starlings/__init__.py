"""Starlings: small, local decision models."""

from .schemas import Candidate, Content, EvaluationResult, Question, SourceBlock
from .sdk import Classifier

__version__ = "0.1.0"
__all__ = ["Classifier", "Candidate", "Content", "EvaluationResult", "Question", "SourceBlock"]
