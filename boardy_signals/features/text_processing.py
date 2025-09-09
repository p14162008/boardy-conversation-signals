"""
Text processing utilities for conversation analysis
"""

import re
import string
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class TextProcessor:
    """Advanced text processing for conversation analysis"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Match-seeking keywords and phrases
        self.match_keywords = {
            "meet": ["meet", "meeting", "meet up", "get together", "hang out"],
            "date": ["date", "dating", "go out", "dinner", "coffee", "lunch"],
            "future": ["sometime", "soon", "this week", "weekend", "tomorrow"],
            "commitment": ["love", "like", "interested", "attracted", "connection"],
            "excitement": ["excited", "can't wait", "looking forward", "amazing"],
            "personal": ["you", "your", "with you", "spend time", "together"]
        }
        
        # Question patterns
        self.question_patterns = [
            r'\?',  # Direct questions
            r'what\s+', r'when\s+', r'where\s+', r'why\s+', r'how\s+',  # Wh-questions
            r'are\s+you', r'do\s+you', r'would\s+you', r'could\s+you',  # Yes/no questions
            r'can\s+we', r'should\s+we', r'will\s+we'  # Suggestion questions
        ]
        
        # Commitment language patterns
        self.commitment_patterns = [
            r'\b(want|would like|would love|desire|wish)\b',
            r'\b(plan|planning|intend|intention)\b',
            r'\b(promise|commit|dedicated|serious)\b',
            r'\b(forever|always|never|forever)\b',
            r'\b(relationship|together|future|long-term)\b'
        ]
        
        # Response urgency indicators
        self.urgency_patterns = [
            r'\b(urgent|asap|immediately|right now|quickly)\b',
            r'\b(soon|fast|hurry|rush)\b',
            r'!{2,}',  # Multiple exclamation marks
            r'\b(please|pls)\b.*\b(soon|quick|fast)\b'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{3,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text
    
    def extract_sentiment(self, text: str) -> Dict[str, float]:
        """Extract sentiment scores using multiple methods"""
        if not text:
            return {"polarity": 0.0, "subjectivity": 0.0, "compound": 0.0}
        
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        compound = vader_scores['compound']
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "compound": compound,
            "positive": vader_scores['pos'],
            "negative": vader_scores['neg'],
            "neutral": vader_scores['neu']
        }
    
    def detect_questions(self, text: str) -> Tuple[bool, List[str]]:
        """Detect if text contains questions and extract them"""
        if not text:
            return False, []
        
        questions = []
        
        # Split by sentence
        sentences = re.split(r'[.!]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for question patterns
            for pattern in self.question_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    questions.append(sentence)
                    break
        
        return len(questions) > 0, questions
    
    def extract_match_keywords(self, text: str) -> Dict[str, int]:
        """Extract match-seeking keywords from text"""
        if not text:
            return {}
        
        text_lower = text.lower()
        keyword_counts = {}
        
        for category, keywords in self.match_keywords.items():
            count = 0
            for keyword in keywords:
                count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            keyword_counts[category] = count
        
        return keyword_counts
    
    def detect_commitment_language(self, text: str) -> Tuple[bool, float]:
        """Detect commitment language and return confidence score"""
        if not text:
            return False, 0.0
        
        text_lower = text.lower()
        matches = 0
        total_patterns = len(self.commitment_patterns)
        
        for pattern in self.commitment_patterns:
            if re.search(pattern, text_lower):
                matches += 1
        
        confidence = matches / total_patterns
        return matches > 0, confidence
    
    def detect_urgency(self, text: str) -> Tuple[bool, float]:
        """Detect urgency indicators in text"""
        if not text:
            return False, 0.0
        
        text_lower = text.lower()
        matches = 0
        total_patterns = len(self.urgency_patterns)
        
        for pattern in self.urgency_patterns:
            if re.search(pattern, text_lower):
                matches += 1
        
        confidence = matches / total_patterns
        return matches > 0, confidence
    
    def calculate_question_density(self, messages: List[str]) -> float:
        """Calculate question density across messages"""
        if not messages:
            return 0.0
        
        total_questions = 0
        total_sentences = 0
        
        for message in messages:
            if not message:
                continue
                
            # Count sentences
            sentences = re.split(r'[.!]+', message)
            total_sentences += len([s for s in sentences if s.strip()])
            
            # Count questions
            has_questions, _ = self.detect_questions(message)
            if has_questions:
                total_questions += 1
        
        return total_questions / len(messages) if messages else 0.0
    
    def analyze_emoji_usage(self, text: str) -> Dict[str, Any]:
        """Analyze emoji usage in text"""
        if not text:
            return {"count": 0, "types": [], "sentiment": 0.0}
        
        # Emoji patterns
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        emojis = emoji_pattern.findall(text)
        
        # Categorize emojis by sentiment
        positive_emojis = ['😊', '😍', '🥰', '😘', '❤️', '💕', '😄', '😁', '🤗']
        negative_emojis = ['😢', '😭', '😞', '😔', '😤', '😠', '😡', '💔']
        
        positive_count = sum(1 for emoji in emojis if emoji in positive_emojis)
        negative_count = sum(1 for emoji in emojis if emoji in negative_emojis)
        
        sentiment_score = (positive_count - negative_count) / len(emojis) if emojis else 0.0
        
        return {
            "count": len(emojis),
            "types": list(set(emojis)),
            "sentiment": sentiment_score,
            "positive_count": positive_count,
            "negative_count": negative_count
        }
    
    def extract_response_time_features(self, timestamps: List[str]) -> Dict[str, float]:
        """Extract response time features from message timestamps"""
        if len(timestamps) < 2:
            return {"avg_response_time": 0.0, "response_consistency": 0.0}
        
        from datetime import datetime
        
        response_times = []
        for i in range(1, len(timestamps)):
            try:
                prev_time = datetime.fromisoformat(timestamps[i-1])
                curr_time = datetime.fromisoformat(timestamps[i])
                response_time = (curr_time - prev_time).total_seconds() / 60  # minutes
                response_times.append(response_time)
            except:
                continue
        
        if not response_times:
            return {"avg_response_time": 0.0, "response_consistency": 0.0}
        
        avg_response_time = sum(response_times) / len(response_times)
        
        # Calculate consistency (lower std dev = more consistent)
        variance = sum((rt - avg_response_time) ** 2 for rt in response_times) / len(response_times)
        std_dev = variance ** 0.5
        consistency = max(0, 1 - (std_dev / avg_response_time)) if avg_response_time > 0 else 0
        
        return {
            "avg_response_time": avg_response_time,
            "response_consistency": consistency,
            "min_response_time": min(response_times),
            "max_response_time": max(response_times)
        }
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get comprehensive text statistics"""
        if not text:
            return {}
        
        # Basic stats
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        
        # Advanced stats
        has_questions, questions = self.detect_questions(text)
        match_keywords = self.extract_match_keywords(text)
        has_commitment, commitment_confidence = self.detect_commitment_language(text)
        has_urgency, urgency_confidence = self.detect_urgency(text)
        emoji_analysis = self.analyze_emoji_usage(text)
        sentiment = self.extract_sentiment(text)
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "has_questions": has_questions,
            "question_count": len(questions),
            "questions": questions,
            "match_keywords": match_keywords,
            "has_commitment": has_commitment,
            "commitment_confidence": commitment_confidence,
            "has_urgency": has_urgency,
            "urgency_confidence": urgency_confidence,
            "emoji_analysis": emoji_analysis,
            "sentiment": sentiment
        }