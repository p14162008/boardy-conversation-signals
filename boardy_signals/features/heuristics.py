"""
Heuristic-based signal detection for conversation analysis
"""

from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging

from ..data.models import Conversation, Message, Signal
from ..config import get_config
from .text_processing import TextProcessor

logger = logging.getLogger(__name__)


class HeuristicAnalyzer:
    """Rule-based analyzer for detecting match-seeking signals"""
    
    def __init__(self):
        self.config = get_config()
        self.text_processor = TextProcessor()
        self.logger = logging.getLogger(__name__)
    
    def analyze_conversation(self, conversation: Conversation) -> List[Signal]:
        """Analyze conversation using heuristic rules"""
        signals = []
        
        try:
            # Analyze message chunks
            chunks = conversation.get_message_chunks(self.config.chunk_size)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_signals = self._analyze_chunk(conversation.id, chunk, chunk_idx)
                signals.extend(chunk_signals)
            
            # Analyze conversation-level patterns
            conversation_signals = self._analyze_conversation_patterns(conversation)
            signals.extend(conversation_signals)
            
            self.logger.info(f"Detected {len(signals)} signals in conversation {conversation.id}")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation {conversation.id}: {e}")
            return []
    
    def _analyze_chunk(self, conversation_id: str, chunk: List[Message], chunk_idx: int) -> List[Signal]:
        """Analyze a chunk of messages for signals"""
        signals = []
        
        if len(chunk) < 2:
            return signals
        
        # Extract text content
        messages_text = [msg.content for msg in chunk]
        timestamps = [msg.timestamp.isoformat() for msg in chunk]
        
        # Analyze sentiment shift
        sentiment_signal = self._detect_sentiment_shift(conversation_id, chunk, messages_text)
        if sentiment_signal:
            signals.append(sentiment_signal)
        
        # Analyze question density
        question_signal = self._detect_question_patterns(conversation_id, chunk, messages_text)
        if question_signal:
            signals.append(question_signal)
        
        # Analyze commitment language
        commitment_signal = self._detect_commitment_language(conversation_id, chunk, messages_text)
        if commitment_signal:
            signals.append(commitment_signal)
        
        # Analyze match-seeking intent
        intent_signal = self._detect_match_intent(conversation_id, chunk, messages_text)
        if intent_signal:
            signals.append(intent_signal)
        
        # Analyze response urgency
        urgency_signal = self._detect_response_urgency(conversation_id, chunk, timestamps)
        if urgency_signal:
            signals.append(urgency_signal)
        
        return signals
    
    def _detect_sentiment_shift(self, conversation_id: str, chunk: List[Message], messages_text: List[str]) -> Signal:
        """Detect positive sentiment shift indicating growing interest"""
        if len(messages_text) < 3:
            return None
        
        sentiments = []
        for text in messages_text:
            sentiment = self.text_processor.extract_sentiment(text)
            sentiments.append(sentiment['compound'])
        
        # Calculate sentiment trend
        if len(sentiments) >= 3:
            early_sentiment = sum(sentiments[:len(sentiments)//2]) / (len(sentiments)//2)
            late_sentiment = sum(sentiments[len(sentiments)//2:]) / (len(sentiments) - len(sentiments)//2)
            
            sentiment_shift = late_sentiment - early_sentiment
            
            if sentiment_shift >= self.config.sentiment_shift_threshold:
                confidence = min(0.95, sentiment_shift * 2)  # Scale to confidence
                
                return Signal(
                    conversation_id=conversation_id,
                    timestamp=chunk[-1].timestamp,
                    signal_type="sentiment_shift",
                    confidence=confidence,
                    rationale=f"Positive sentiment shift detected: {sentiment_shift:.2f} (threshold: {self.config.sentiment_shift_threshold})",
                    message_ids=[msg.id for msg in chunk],
                    metadata={
                        "sentiment_shift": sentiment_shift,
                        "early_sentiment": early_sentiment,
                        "late_sentiment": late_sentiment,
                        "chunk_size": len(chunk)
                    }
                )
        
        return None
    
    def _detect_question_patterns(self, conversation_id: str, chunk: List[Message], messages_text: List[str]) -> Signal:
        """Detect high question density indicating interest"""
        question_density = self.text_processor.calculate_question_density(messages_text)
        
        if question_density >= self.config.question_density_threshold:
            # Count total questions
            total_questions = sum(1 for text in messages_text 
                                if self.text_processor.detect_questions(text)[0])
            
            confidence = min(0.95, question_density * 3)  # Scale to confidence
            
            return Signal(
                conversation_id=conversation_id,
                timestamp=chunk[-1].timestamp,
                signal_type="question_asking",
                confidence=confidence,
                rationale=f"High question density detected: {question_density:.2f} (threshold: {self.config.question_density_threshold})",
                message_ids=[msg.id for msg in chunk],
                metadata={
                    "question_density": question_density,
                    "total_questions": total_questions,
                    "chunk_size": len(chunk)
                }
            )
        
        return None
    
    def _detect_commitment_language(self, conversation_id: str, chunk: List[Message], messages_text: List[str]) -> Signal:
        """Detect commitment language indicating serious interest"""
        commitment_scores = []
        commitment_messages = []
        
        for i, text in enumerate(messages_text):
            has_commitment, confidence = self.text_processor.detect_commitment_language(text)
            if has_commitment:
                commitment_scores.append(confidence)
                commitment_messages.append(chunk[i])
        
        if commitment_scores:
            avg_commitment = sum(commitment_scores) / len(commitment_scores)
            
            if avg_commitment >= self.config.commitment_language_threshold:
                confidence = min(0.95, avg_commitment * 2)
                
                return Signal(
                    conversation_id=conversation_id,
                    timestamp=chunk[-1].timestamp,
                    signal_type="commitment_language",
                    confidence=confidence,
                    rationale=f"Commitment language detected: {avg_commitment:.2f} (threshold: {self.config.commitment_language_threshold})",
                    message_ids=[msg.id for msg in commitment_messages],
                    metadata={
                        "commitment_score": avg_commitment,
                        "commitment_messages": len(commitment_messages),
                        "chunk_size": len(chunk)
                    }
                )
        
        return None
    
    def _detect_match_intent(self, conversation_id: str, chunk: List[Message], messages_text: List[str]) -> Signal:
        """Detect explicit match-seeking intent"""
        total_match_keywords = {}
        
        for text in messages_text:
            keywords = self.text_processor.extract_match_keywords(text)
            for category, count in keywords.items():
                total_match_keywords[category] = total_match_keywords.get(category, 0) + count
        
        # Calculate overall match intent score
        total_keywords = sum(total_match_keywords.values())
        if total_keywords > 0:
            # Weight different categories
            weighted_score = (
                total_match_keywords.get("meet", 0) * 0.3 +
                total_match_keywords.get("date", 0) * 0.4 +
                total_match_keywords.get("future", 0) * 0.2 +
                total_match_keywords.get("commitment", 0) * 0.1
            )
            
            # Normalize by chunk size
            normalized_score = weighted_score / len(messages_text)
            
            if normalized_score >= 0.3:  # Threshold for match intent
                confidence = min(0.95, normalized_score * 2)
                
                return Signal(
                    conversation_id=conversation_id,
                    timestamp=chunk[-1].timestamp,
                    signal_type="match_seeking",
                    confidence=confidence,
                    rationale=f"Match-seeking intent detected: {normalized_score:.2f} (keywords: {total_match_keywords})",
                    message_ids=[msg.id for msg in chunk],
                    metadata={
                        "match_keywords": total_match_keywords,
                        "normalized_score": normalized_score,
                        "chunk_size": len(chunk)
                    }
                )
        
        return None
    
    def _detect_response_urgency(self, conversation_id: str, chunk: List[Message], timestamps: List[str]) -> Signal:
        """Detect response urgency patterns"""
        if len(timestamps) < 2:
            return None
        
        # Analyze response times
        response_features = self.text_processor.extract_response_time_features(timestamps)
        avg_response_time = response_features.get("avg_response_time", 0)
        
        # Check for urgency in text
        urgency_scores = []
        for msg in chunk:
            has_urgency, confidence = self.text_processor.detect_urgency(msg.content)
            if has_urgency:
                urgency_scores.append(confidence)
        
        # Fast response time indicates urgency (under 5 minutes average)
        fast_response = avg_response_time < 5.0 and avg_response_time > 0
        
        # Urgency in text
        text_urgency = len(urgency_scores) > 0
        
        if fast_response or text_urgency:
            confidence = 0.0
            
            if fast_response:
                confidence += 0.4
            
            if text_urgency:
                confidence += 0.3
            
            if len(urgency_scores) > 0:
                confidence += min(0.3, sum(urgency_scores) / len(urgency_scores))
            
            return Signal(
                conversation_id=conversation_id,
                timestamp=chunk[-1].timestamp,
                signal_type="response_urgency",
                confidence=confidence,
                rationale=f"Response urgency detected: fast_response={fast_response}, text_urgency={text_urgency}",
                message_ids=[msg.id for msg in chunk],
                metadata={
                    "avg_response_time": avg_response_time,
                    "fast_response": fast_response,
                    "text_urgency": text_urgency,
                    "urgency_scores": urgency_scores
                }
            )
        
        return None
    
    def _analyze_conversation_patterns(self, conversation: Conversation) -> List[Signal]:
        """Analyze conversation-level patterns"""
        signals = []
        
        # Analyze overall conversation flow
        if conversation.message_count >= 10:
            # Check for escalating interest over time
            escalation_signal = self._detect_interest_escalation(conversation)
            if escalation_signal:
                signals.append(escalation_signal)
        
        return signals
    
    def _detect_interest_escalation(self, conversation: Conversation) -> Signal:
        """Detect escalating interest throughout conversation"""
        if len(conversation.messages) < 10:
            return None
        
        # Split conversation into thirds
        third_size = len(conversation.messages) // 3
        early_messages = conversation.messages[:third_size]
        late_messages = conversation.messages[-third_size:]
        
        # Analyze early vs late sentiment and engagement
        early_texts = [msg.content for msg in early_messages]
        late_texts = [msg.content for msg in late_messages]
        
        # Calculate engagement metrics
        early_engagement = self._calculate_engagement_score(early_texts)
        late_engagement = self._calculate_engagement_score(late_texts)
        
        engagement_increase = late_engagement - early_engagement
        
        if engagement_increase >= 0.2:  # Significant increase
            confidence = min(0.95, engagement_increase * 2)
            
            return Signal(
                conversation_id=conversation.id,
                timestamp=conversation.messages[-1].timestamp,
                signal_type="interest_escalation",
                confidence=confidence,
                rationale=f"Interest escalation detected: {engagement_increase:.2f} increase in engagement",
                message_ids=[msg.id for msg in late_messages],
                metadata={
                    "early_engagement": early_engagement,
                    "late_engagement": late_engagement,
                    "engagement_increase": engagement_increase
                }
            )
        
        return None
    
    def _calculate_engagement_score(self, messages: List[str]) -> float:
        """Calculate engagement score for a set of messages"""
        if not messages:
            return 0.0
        
        total_score = 0.0
        
        for text in messages:
            # Question density
            has_questions, _ = self.text_processor.detect_questions(text)
            if has_questions:
                total_score += 0.3
            
            # Match keywords
            keywords = self.text_processor.extract_match_keywords(text)
            keyword_score = sum(keywords.values()) * 0.1
            total_score += min(0.3, keyword_score)
            
            # Commitment language
            has_commitment, confidence = self.text_processor.detect_commitment_language(text)
            if has_commitment:
                total_score += confidence * 0.2
            
            # Sentiment
            sentiment = self.text_processor.extract_sentiment(text)
            if sentiment['compound'] > 0.1:
                total_score += 0.2
        
        return total_score / len(messages)
    
    def get_heuristic_confidence(self, signals: List[Signal]) -> float:
        """Calculate overall confidence based on heuristic signals"""
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
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0