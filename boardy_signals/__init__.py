"""
Boardy Conversation Quality Signals

An AI-powered system to detect match-seeking moments in conversations
and generate next best actions for dating app optimization.
"""

__version__ = "0.1.0"
__author__ = "Pranav Kunadharaju"
__email__ = "pranavkunadharaju@gmail.com"

from .config import Config
from .data.models import Conversation, Message, Signal, AnalysisResult

__all__ = [
    "Config",
    "Conversation", 
    "Message",
    "Signal",
    "AnalysisResult",
]