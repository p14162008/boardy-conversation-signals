"""
Configuration settings for Boardy Conversation Quality Signals
"""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Main configuration class with environment variable support"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    
    # System Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    data_path: str = Field(default="./data/transcripts.jsonl", env="DATA_PATH")
    eval_data_path: str = Field(default="./data/eval_set.jsonl", env="EVAL_DATA_PATH")
    output_path: str = Field(default="./output/results.json", env="OUTPUT_PATH")
    
    # Feature Extraction Settings
    chunk_size: int = Field(default=10, env="CHUNK_SIZE")
    min_message_length: int = Field(default=5, env="MIN_MESSAGE_LENGTH")
    max_message_length: int = Field(default=500, env="MAX_MESSAGE_LENGTH")
    
    # Signal Detection Thresholds
    min_confidence_threshold: float = Field(default=0.7, env="MIN_CONFIDENCE_THRESHOLD")
    sentiment_shift_threshold: float = Field(default=0.3, env="SENTIMENT_SHIFT_THRESHOLD")
    question_density_threshold: float = Field(default=0.2, env="QUESTION_DENSITY_THRESHOLD")
    commitment_language_threshold: float = Field(default=0.4, env="COMMITMENT_LANGUAGE_THRESHOLD")
    
    # Heuristic Weights
    heuristic_weights: Dict[str, float] = Field(default={
        "sentiment_shift": 0.25,
        "question_density": 0.20,
        "commitment_language": 0.30,
        "intent_keywords": 0.15,
        "response_time": 0.10
    })
    
    # LLM Settings
    max_tokens: int = Field(default=150, env="MAX_TOKENS")
    temperature: float = Field(default=0.3, env="TEMPERATURE")
    
    # Evaluation Settings
    eval_sample_size: int = Field(default=20, env="EVAL_SAMPLE_SIZE")
    precision_threshold: float = Field(default=0.7, env="PRECISION_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def update_config(**kwargs: Any) -> None:
    """Update configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)