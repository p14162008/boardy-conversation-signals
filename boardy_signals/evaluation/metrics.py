"""
Metrics calculation for evaluation
"""

from typing import List, Dict, Any, Tuple
import logging

from ..data.models import Signal, EvaluationResult
from ..config import get_config

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate precision, recall, and other evaluation metrics"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, 
                         predicted_signals: List[Signal], 
                         ground_truth_signals: List[Signal]) -> Dict[str, float]:
        """Calculate precision, recall, F1, and other metrics"""
        
        # Convert signals to comparable format
        predicted_set = self._signals_to_set(predicted_signals)
        ground_truth_set = self._signals_to_set(ground_truth_signals)
        
        # Calculate basic metrics
        true_positives = len(predicted_set.intersection(ground_truth_set))
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate accuracy
        total_signals = len(predicted_set.union(ground_truth_set))
        accuracy = true_positives / total_signals if total_signals > 0 else 0.0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "accuracy": round(accuracy, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_predicted": len(predicted_signals),
            "total_ground_truth": len(ground_truth_signals)
        }
    
    def calculate_signal_type_metrics(self, 
                                    predicted_signals: List[Signal], 
                                    ground_truth_signals: List[Signal]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each signal type"""
        
        signal_types = set()
        signal_types.update(signal.signal_type for signal in predicted_signals)
        signal_types.update(signal.signal_type for signal in ground_truth_signals)
        
        type_metrics = {}
        
        for signal_type in signal_types:
            # Filter signals by type
            pred_type = [s for s in predicted_signals if s.signal_type == signal_type]
            gt_type = [s for s in ground_truth_signals if s.signal_type == signal_type]
            
            # Calculate metrics for this type
            metrics = self.calculate_metrics(pred_type, gt_type)
            type_metrics[signal_type] = metrics
        
        return type_metrics
    
    def calculate_confidence_metrics(self, 
                                   predicted_signals: List[Signal], 
                                   ground_truth_signals: List[Signal]) -> Dict[str, Any]:
        """Calculate metrics based on confidence thresholds"""
        
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        confidence_metrics = {}
        
        for threshold in thresholds:
            # Filter predicted signals by confidence
            high_conf_pred = [s for s in predicted_signals if s.confidence >= threshold]
            
            # Calculate metrics at this threshold
            metrics = self.calculate_metrics(high_conf_pred, ground_truth_signals)
            confidence_metrics[f"threshold_{threshold}"] = metrics
        
        return confidence_metrics
    
    def calculate_conversation_level_metrics(self, 
                                           conversation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics at the conversation level"""
        
        total_conversations = len(conversation_results)
        correct_predictions = 0
        false_positives = 0
        false_negatives = 0
        
        for result in conversation_results:
            predicted_has_signals = result.get("predicted_has_signals", False)
            ground_truth_has_signals = result.get("ground_truth_has_signals", False)
            
            if predicted_has_signals and ground_truth_has_signals:
                correct_predictions += 1
            elif predicted_has_signals and not ground_truth_has_signals:
                false_positives += 1
            elif not predicted_has_signals and ground_truth_has_signals:
                false_negatives += 1
        
        # Calculate metrics
        precision = correct_predictions / (correct_predictions + false_positives) if (correct_predictions + false_positives) > 0 else 0.0
        recall = correct_predictions / (correct_predictions + false_negatives) if (correct_predictions + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = correct_predictions / total_conversations if total_conversations > 0 else 0.0
        
        return {
            "conversation_precision": round(precision, 3),
            "conversation_recall": round(recall, 3),
            "conversation_f1": round(f1_score, 3),
            "conversation_accuracy": round(accuracy, 3),
            "correct_conversations": correct_predictions,
            "false_positive_conversations": false_positives,
            "false_negative_conversations": false_negatives,
            "total_conversations": total_conversations
        }
    
    def calculate_next_action_metrics(self, 
                                    predicted_actions: List[str], 
                                    ground_truth_actions: List[str]) -> Dict[str, float]:
        """Calculate metrics for next best action predictions"""
        
        if len(predicted_actions) != len(ground_truth_actions):
            self.logger.warning("Mismatch in predicted and ground truth action counts")
            return {"action_accuracy": 0.0}
        
        # Simple exact match accuracy
        exact_matches = sum(1 for pred, gt in zip(predicted_actions, ground_truth_actions) if pred == gt)
        exact_accuracy = exact_matches / len(predicted_actions) if predicted_actions else 0.0
        
        # Semantic similarity (simplified - could use embeddings)
        semantic_matches = 0
        for pred, gt in zip(predicted_actions, ground_truth_actions):
            if self._actions_similar(pred, gt):
                semantic_matches += 1
        
        semantic_accuracy = semantic_matches / len(predicted_actions) if predicted_actions else 0.0
        
        return {
            "action_exact_accuracy": round(exact_accuracy, 3),
            "action_semantic_accuracy": round(semantic_accuracy, 3),
            "exact_matches": exact_matches,
            "semantic_matches": semantic_matches,
            "total_actions": len(predicted_actions)
        }
    
    def _signals_to_set(self, signals: List[Signal]) -> set:
        """Convert signals to a set for comparison"""
        signal_set = set()
        
        for signal in signals:
            # Create a comparable representation
            signal_key = (
                signal.signal_type,
                signal.timestamp.isoformat()[:16],  # Round to minute precision
                round(signal.confidence, 1)  # Round confidence to 1 decimal
            )
            signal_set.add(signal_key)
        
        return signal_set
    
    def _actions_similar(self, action1: str, action2: str) -> bool:
        """Check if two actions are semantically similar"""
        # Simple keyword-based similarity
        keywords1 = set(action1.lower().split())
        keywords2 = set(action2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity >= 0.5  # 50% keyword overlap
    
    def create_evaluation_result(self, 
                               conversation_results: List[Dict[str, Any]],
                               processing_time_ms: int) -> EvaluationResult:
        """Create a comprehensive evaluation result"""
        
        # Aggregate all signals
        all_predicted_signals = []
        all_ground_truth_signals = []
        all_predicted_actions = []
        all_ground_truth_actions = []
        
        for result in conversation_results:
            all_predicted_signals.extend(result.get("predicted_signals", []))
            all_ground_truth_signals.extend(result.get("ground_truth_signals", []))
            all_predicted_actions.append(result.get("predicted_action", ""))
            all_ground_truth_actions.append(result.get("ground_truth_action", ""))
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(all_predicted_signals, all_ground_truth_signals)
        signal_type_metrics = self.calculate_signal_type_metrics(all_predicted_signals, all_ground_truth_signals)
        conversation_metrics = self.calculate_conversation_level_metrics(conversation_results)
        action_metrics = self.calculate_next_action_metrics(all_predicted_actions, all_ground_truth_actions)
        
        # Create evaluation result
        eval_result = EvaluationResult(
            total_samples=len(conversation_results),
            correct_predictions=conversation_metrics["correct_conversations"],
            false_positives=conversation_metrics["false_positive_conversations"],
            false_negatives=conversation_metrics["false_negative_conversations"],
            precision=overall_metrics["precision"],
            recall=overall_metrics["recall"],
            f1_score=overall_metrics["f1_score"],
            signal_type_breakdown=signal_type_metrics,
            processing_time_ms=processing_time_ms
        )
        
        # Add additional metrics to metadata
        eval_result.metadata = {
            "overall_metrics": overall_metrics,
            "conversation_metrics": conversation_metrics,
            "action_metrics": action_metrics,
            "confidence_metrics": self.calculate_confidence_metrics(all_predicted_signals, all_ground_truth_signals)
        }
        
        return eval_result