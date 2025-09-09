"""
Main feature extraction pipeline
"""

import time
from typing import List, Dict, Any
from datetime import datetime
import logging

from ..data.models import Conversation, AnalysisResult, Signal
from ..config import get_config
from .heuristics import HeuristicAnalyzer
from .llm_analyzer import LLMAnalyzer
from .text_processing import TextProcessor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Main feature extraction and analysis pipeline"""
    
    def __init__(self):
        self.config = get_config()
        self.heuristic_analyzer = HeuristicAnalyzer()
        self.llm_analyzer = LLMAnalyzer()
        self.text_processor = TextProcessor()
        self.logger = logging.getLogger(__name__)
    
    def analyze_conversation(self, conversation: Conversation) -> AnalysisResult:
        """Analyze a single conversation and return results"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting analysis for conversation {conversation.id}")
            
            # Extract features using heuristics
            heuristic_signals = self.heuristic_analyzer.analyze_conversation(conversation)
            self.logger.info(f"Heuristic analysis found {len(heuristic_signals)} signals")
            
            # Extract features using LLM (if available)
            llm_signals = []
            if self.llm_analyzer.is_available():
                llm_signals = self.llm_analyzer.analyze_conversation(conversation)
                self.logger.info(f"LLM analysis found {len(llm_signals)} signals")
            else:
                self.logger.info("LLM analysis skipped (not available)")
            
            # Combine and deduplicate signals
            all_signals = self._combine_signals(heuristic_signals, llm_signals)
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(all_signals)
            
            # Generate next best action
            next_best_action = self._generate_next_best_action(conversation, all_signals)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create analysis result
            result = AnalysisResult(
                conversation_id=conversation.id,
                signals=all_signals,
                next_best_action=next_best_action,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                metadata={
                    "heuristic_signals": len(heuristic_signals),
                    "llm_signals": len(llm_signals),
                    "total_signals": len(all_signals),
                    "llm_available": self.llm_analyzer.is_available(),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Analysis completed for conversation {conversation.id} in {processing_time_ms}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation {conversation.id}: {e}")
            
            # Return empty result on error
            return AnalysisResult(
                conversation_id=conversation.id,
                signals=[],
                next_best_action="Error in analysis - suggest manual review",
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)}
            )
    
    def analyze_conversations(self, conversations: List[Conversation]) -> List[AnalysisResult]:
        """Analyze multiple conversations"""
        results = []
        
        self.logger.info(f"Starting analysis of {len(conversations)} conversations")
        
        for i, conversation in enumerate(conversations):
            self.logger.info(f"Processing conversation {i+1}/{len(conversations)}: {conversation.id}")
            
            result = self.analyze_conversation(conversation)
            results.append(result)
        
        self.logger.info(f"Completed analysis of {len(conversations)} conversations")
        return results
    
    def _combine_signals(self, heuristic_signals: List[Signal], llm_signals: List[Signal]) -> List[Signal]:
        """Combine and deduplicate signals from different sources"""
        combined_signals = []
        
        # Add heuristic signals
        for signal in heuristic_signals:
            signal.metadata["source"] = "heuristic"
            combined_signals.append(signal)
        
        # Add LLM signals (with deduplication)
        for llm_signal in llm_signals:
            # Check for similar signals from heuristics
            is_duplicate = False
            
            for heuristic_signal in heuristic_signals:
                if self._signals_similar(llm_signal, heuristic_signal):
                    # Merge signals - use higher confidence
                    if llm_signal.confidence > heuristic_signal.confidence:
                        heuristic_signal.confidence = llm_signal.confidence
                        heuristic_signal.rationale = f"Combined: {llm_signal.rationale}"
                        heuristic_signal.metadata["sources"] = ["heuristic", "llm"]
                    else:
                        heuristic_signal.metadata["sources"] = ["heuristic", "llm"]
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                llm_signal.metadata["source"] = "llm"
                combined_signals.append(llm_signal)
        
        # Filter by confidence threshold
        filtered_signals = [
            signal for signal in combined_signals 
            if signal.confidence >= self.config.min_confidence_threshold
        ]
        
        self.logger.info(f"Combined {len(heuristic_signals)} heuristic + {len(llm_signals)} LLM signals into {len(filtered_signals)} final signals")
        
        return filtered_signals
    
    def _signals_similar(self, signal1: Signal, signal2: Signal) -> bool:
        """Check if two signals are similar enough to be considered duplicates"""
        # Same signal type
        if signal1.signal_type != signal2.signal_type:
            return False
        
        # Similar timestamps (within 5 minutes)
        time_diff = abs((signal1.timestamp - signal2.timestamp).total_seconds())
        if time_diff > 300:  # 5 minutes
            return False
        
        # Similar confidence levels (within 0.3)
        confidence_diff = abs(signal1.confidence - signal2.confidence)
        if confidence_diff > 0.3:
            return False
        
        return True
    
    def _calculate_overall_confidence(self, signals: List[Signal]) -> float:
        """Calculate overall confidence score for the conversation"""
        if not signals:
            return 0.0
        
        # Weight different signal types
        weights = self.config.heuristic_weights
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = weights.get(signal.signal_type, 0.1)
            weighted_confidence += signal.confidence * weight
            total_weight += weight
        
        # Normalize by total weight
        overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # Boost confidence if multiple signal types are present
        unique_types = len(set(signal.signal_type for signal in signals))
        if unique_types > 1:
            overall_confidence *= (1 + (unique_types - 1) * 0.1)
        
        return min(1.0, overall_confidence)
    
    def _generate_next_best_action(self, conversation: Conversation, signals: List[Signal]) -> str:
        """Generate next best action based on analysis"""
        
        # Try LLM first if available
        if self.llm_analyzer.is_available():
            try:
                action = self.llm_analyzer.generate_next_best_action(conversation, signals)
                if action and len(action.strip()) > 0:
                    return action
            except Exception as e:
                self.logger.warning(f"LLM action generation failed: {e}")
        
        # Fall back to heuristic-based action generation
        return self._generate_heuristic_action(conversation, signals)
    
    def _generate_heuristic_action(self, conversation: Conversation, signals: List[Signal]) -> str:
        """Generate action using heuristic rules"""
        
        if not signals:
            return "Suggest an engaging question to spark conversation"
        
        # Find the highest confidence signal
        best_signal = max(signals, key=lambda s: s.confidence)
        
        # Generate action based on signal type and conversation context
        if best_signal.signal_type == "match_seeking":
            if best_signal.confidence > 0.8:
                return "Suggest a specific meetup time and location immediately"
            else:
                return "Encourage continued conversation and suggest meeting soon"
        
        elif best_signal.signal_type == "interest_escalation":
            return "Facilitate deeper conversation with personal questions"
        
        elif best_signal.signal_type == "commitment_language":
            return "Acknowledge their interest and suggest a meaningful first date"
        
        elif best_signal.signal_type == "question_asking":
            return "Encourage continued engagement and suggest meeting to continue the conversation"
        
        elif best_signal.signal_type == "sentiment_shift":
            return "Capitalize on positive momentum with a meetup suggestion"
        
        elif best_signal.signal_type == "response_urgency":
            return "Provide immediate response options or quick meetup opportunities"
        
        else:
            return "Suggest an engaging follow-up question to maintain interest"
    
    def get_extraction_stats(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """Get statistics about the extraction process"""
        if not results:
            return {}
        
        total_conversations = len(results)
        total_signals = sum(len(result.signals) for result in results)
        total_processing_time = sum(result.processing_time_ms for result in results)
        
        # Signal type breakdown
        signal_types = {}
        for result in results:
            for signal in result.signals:
                signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1
        
        # Confidence distribution
        confidences = [result.confidence_score for result in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # High confidence results
        high_confidence_count = sum(1 for conf in confidences if conf >= 0.7)
        
        stats = {
            "total_conversations": total_conversations,
            "total_signals": total_signals,
            "avg_signals_per_conversation": total_signals / total_conversations,
            "avg_processing_time_ms": total_processing_time / total_conversations,
            "avg_confidence": round(avg_confidence, 3),
            "high_confidence_results": high_confidence_count,
            "high_confidence_percentage": round(high_confidence_count / total_conversations * 100, 1),
            "signal_type_breakdown": signal_types,
            "llm_available": self.llm_analyzer.is_available()
        }
        
        return stats