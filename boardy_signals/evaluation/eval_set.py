"""
Evaluation dataset creation and management
"""

import json
import jsonlines
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..data.models import Conversation, Message, Signal, EvaluationSample
from ..config import get_config
from ..data.sample_data import generate_evaluation_samples

logger = logging.getLogger(__name__)


class EvaluationDataset:
    """Manage evaluation dataset with ground truth labels"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def create_evaluation_dataset(self) -> List[EvaluationSample]:
        """Create a comprehensive evaluation dataset"""
        
        # Generate base samples
        base_samples = generate_evaluation_samples()
        
        # Convert to EvaluationSample objects
        eval_samples = []
        
        for i, sample in enumerate(base_samples):
            try:
                # Parse messages
                messages = []
                for msg_data in sample["messages"]:
                    message = Message(
                        timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                        sender_id=msg_data["sender_id"],
                        content=msg_data["content"]
                    )
                    messages.append(message)
                
                # Parse ground truth signals
                ground_truth_signals = []
                for signal_type in sample["ground_truth"]["signals"]:
                    signal = Signal(
                        conversation_id=sample["conversation_id"],
                        timestamp=datetime.now(),
                        signal_type=signal_type,
                        confidence=0.9,  # Ground truth is high confidence
                        rationale=f"Ground truth: {signal_type}",
                        message_ids=[msg.id for msg in messages]
                    )
                    ground_truth_signals.append(signal)
                
                # Create evaluation sample
                eval_sample = EvaluationSample(
                    conversation_id=sample["conversation_id"],
                    message_chunk=messages,
                    ground_truth_signals=ground_truth_signals,
                    expected_next_action=sample["ground_truth"]["next_action"],
                    difficulty_level=self._determine_difficulty(messages, ground_truth_signals),
                    notes=f"Sample {i+1} from evaluation dataset"
                )
                
                eval_samples.append(eval_sample)
                
            except Exception as e:
                self.logger.warning(f"Failed to create evaluation sample {i}: {e}")
                continue
        
        self.logger.info(f"Created {len(eval_samples)} evaluation samples")
        return eval_samples
    
    def _determine_difficulty(self, messages: List[Message], signals: List[Signal]) -> str:
        """Determine difficulty level of evaluation sample"""
        
        # Easy: Clear, explicit signals
        if len(signals) > 0:
            explicit_keywords = ["meet", "date", "together", "love", "excited"]
            message_text = " ".join(msg.content.lower() for msg in messages)
            
            if any(keyword in message_text for keyword in explicit_keywords):
                return "easy"
        
        # Hard: Subtle or no signals
        if len(signals) == 0:
            return "hard"
        
        # Medium: Everything else
        return "medium"
    
    def load_evaluation_dataset(self, file_path: Optional[str] = None) -> List[EvaluationSample]:
        """Load evaluation dataset from file"""
        
        if file_path is None:
            file_path = self.config.eval_data_path
        
        try:
            eval_samples = []
            
            with jsonlines.open(file_path) as reader:
                for line in reader:
                    try:
                        sample = self._parse_evaluation_sample(line)
                        if sample:
                            eval_samples.append(sample)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse evaluation sample: {e}")
                        continue
            
            self.logger.info(f"Loaded {len(eval_samples)} evaluation samples from {file_path}")
            return eval_samples
            
        except FileNotFoundError:
            self.logger.warning(f"Evaluation file not found: {file_path}, creating new dataset")
            return self.create_evaluation_dataset()
        except Exception as e:
            self.logger.error(f"Error loading evaluation dataset: {e}")
            return []
    
    def _parse_evaluation_sample(self, data: Dict[str, Any]) -> Optional[EvaluationSample]:
        """Parse evaluation sample from dictionary"""
        try:
            # Parse messages
            messages = []
            for msg_data in data["messages"]:
                message = Message(
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    sender_id=msg_data["sender_id"],
                    content=msg_data["content"]
                )
                messages.append(message)
            
            # Parse ground truth signals
            ground_truth_signals = []
            for signal_type in data.get("ground_truth", {}).get("signals", []):
                signal = Signal(
                    conversation_id=data["conversation_id"],
                    timestamp=datetime.now(),
                    signal_type=signal_type,
                    confidence=0.9,
                    rationale=f"Ground truth: {signal_type}",
                    message_ids=[msg.id for msg in messages]
                )
                ground_truth_signals.append(signal)
            
            # Create evaluation sample
            return EvaluationSample(
                conversation_id=data["conversation_id"],
                message_chunk=messages,
                ground_truth_signals=ground_truth_signals,
                expected_next_action=data.get("ground_truth", {}).get("next_action", ""),
                difficulty_level=data.get("difficulty_level", "medium"),
                notes=data.get("notes", "")
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing evaluation sample: {e}")
            return None
    
    def save_evaluation_dataset(self, samples: List[EvaluationSample], file_path: Optional[str] = None):
        """Save evaluation dataset to file"""
        
        if file_path is None:
            file_path = self.config.eval_data_path
        
        try:
            with jsonlines.open(file_path, mode='w') as writer:
                for sample in samples:
                    data = {
                        "conversation_id": sample.conversation_id,
                        "messages": [
                            {
                                "timestamp": msg.timestamp.isoformat(),
                                "sender_id": msg.sender_id,
                                "content": msg.content
                            }
                            for msg in sample.message_chunk
                        ],
                        "ground_truth": {
                            "signals": [signal.signal_type for signal in sample.ground_truth_signals],
                            "next_action": sample.expected_next_action
                        },
                        "difficulty_level": sample.difficulty_level,
                        "notes": sample.notes
                    }
                    writer.write(data)
            
            self.logger.info(f"Saved {len(samples)} evaluation samples to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation dataset: {e}")
            raise
    
    def get_dataset_stats(self, samples: List[EvaluationSample]) -> Dict[str, Any]:
        """Get statistics about the evaluation dataset"""
        
        if not samples:
            return {}
        
        # Signal type distribution
        signal_types = {}
        for sample in samples:
            for signal in sample.ground_truth_signals:
                signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1
        
        # Difficulty distribution
        difficulty_dist = {}
        for sample in samples:
            difficulty_dist[sample.difficulty_level] = difficulty_dist.get(sample.difficulty_level, 0) + 1
        
        # Message count distribution
        message_counts = [len(sample.message_chunk) for sample in samples]
        
        # Signal count distribution
        signal_counts = [len(sample.ground_truth_signals) for sample in samples]
        
        stats = {
            "total_samples": len(samples),
            "signal_type_distribution": signal_types,
            "difficulty_distribution": difficulty_dist,
            "avg_messages_per_sample": sum(message_counts) / len(message_counts),
            "avg_signals_per_sample": sum(signal_counts) / len(signal_counts),
            "samples_with_signals": sum(1 for count in signal_counts if count > 0),
            "samples_without_signals": sum(1 for count in signal_counts if count == 0)
        }
        
        return stats
    
    def create_balanced_dataset(self, target_size: int = 20) -> List[EvaluationSample]:
        """Create a balanced evaluation dataset"""
        
        # Create base samples
        all_samples = self.create_evaluation_dataset()
        
        # Balance by difficulty and signal presence
        easy_samples = [s for s in all_samples if s.difficulty_level == "easy"]
        medium_samples = [s for s in all_samples if s.difficulty_level == "medium"]
        hard_samples = [s for s in all_samples if s.difficulty_level == "hard"]
        
        samples_with_signals = [s for s in all_samples if len(s.ground_truth_signals) > 0]
        samples_without_signals = [s for s in all_samples if len(s.ground_truth_signals) == 0]
        
        # Select balanced subset
        balanced_samples = []
        
        # Include samples with signals (70% of dataset)
        signal_count = int(target_size * 0.7)
        balanced_samples.extend(samples_with_signals[:signal_count])
        
        # Include samples without signals (30% of dataset)
        no_signal_count = target_size - len(balanced_samples)
        balanced_samples.extend(samples_without_signals[:no_signal_count])
        
        # Ensure we have the target size
        if len(balanced_samples) < target_size:
            remaining = target_size - len(balanced_samples)
            remaining_samples = [s for s in all_samples if s not in balanced_samples]
            balanced_samples.extend(remaining_samples[:remaining])
        
        self.logger.info(f"Created balanced dataset with {len(balanced_samples)} samples")
        return balanced_samples[:target_size]