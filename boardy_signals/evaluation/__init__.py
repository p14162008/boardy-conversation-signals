"""
Evaluation framework for conversation signal detection
"""

from .metrics import MetricsCalculator
from .eval_set import EvaluationDataset
from .reporter import EvaluationReporter

__all__ = [
    "MetricsCalculator",
    "EvaluationDataset", 
    "EvaluationReporter",
]