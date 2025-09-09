"""
LLM-based analysis for conversation signals
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

import openai
from openai import OpenAI

from ..data.models import Conversation, Message, Signal
from ..config import get_config
from .text_processing import TextProcessor

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """LLM-based analyzer for detecting match-seeking signals"""
    
    def __init__(self):
        self.config = get_config()
        self.text_processor = TextProcessor()
        self.client = None
        
        # Initialize OpenAI client if API key is available
        if self.config.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.config.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("No OpenAI API key provided, LLM analysis will be disabled")
    
    def analyze_conversation(self, conversation: Conversation) -> List[Signal]:
        """Analyze conversation using LLM"""
        if not self.client:
            logger.warning("LLM client not available, skipping LLM analysis")
            return []
        
        try:
            # Prepare conversation context
            context = self._prepare_conversation_context(conversation)
            
            # Analyze with LLM
            llm_signals = self._analyze_with_llm(context, conversation.id)
            
            logger.info(f"LLM detected {len(llm_signals)} signals in conversation {conversation.id}")
            return llm_signals
            
        except Exception as e:
            logger.error(f"Error in LLM analysis for conversation {conversation.id}: {e}")
            return []
    
    def _prepare_conversation_context(self, conversation: Conversation) -> str:
        """Prepare conversation context for LLM analysis"""
        context_parts = []
        
        # Add conversation metadata
        context_parts.append(f"Conversation ID: {conversation.id}")
        context_parts.append(f"Participants: {', '.join(conversation.participants)}")
        context_parts.append(f"Message count: {len(conversation.messages)}")
        context_parts.append(f"Duration: {conversation.duration_minutes:.1f} minutes" if conversation.duration_minutes else "Duration: Unknown")
        context_parts.append("")
        
        # Add messages with timestamps
        context_parts.append("Messages:")
        for i, message in enumerate(conversation.messages):
            timestamp = message.timestamp.strftime("%H:%M")
            context_parts.append(f"{i+1}. [{timestamp}] {message.sender_id}: {message.content}")
        
        return "\n".join(context_parts)
    
    def _analyze_with_llm(self, context: str, conversation_id: str) -> List[Signal]:
        """Analyze conversation context with LLM"""
        
        prompt = self._create_analysis_prompt(context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse LLM response
            llm_response = response.choices[0].message.content
            signals = self._parse_llm_response(llm_response, conversation_id)
            
            return signals
            
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return []
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM analysis"""
        return """You are an expert conversation analyst specializing in dating app interactions. Your task is to identify moments where users show interest in getting matched or meeting in person.

Analyze the conversation and identify signals of:
1. Match-seeking behavior (explicit requests to meet, date, or get together)
2. Interest escalation (increasing engagement, personal questions, future planning)
3. Commitment language (expressions of serious interest, relationship talk)
4. Question asking patterns (showing curiosity about the other person)
5. Sentiment shifts (positive emotional changes)
6. Response urgency (quick responses, immediate availability)

For each signal you detect, provide:
- Signal type
- Confidence level (0.0 to 1.0)
- Rationale explaining why this is a signal
- Relevant message numbers

Respond in JSON format with the following structure:
{
  "signals": [
    {
      "signal_type": "match_seeking|interest_escalation|commitment_language|question_asking|sentiment_shift|response_urgency",
      "confidence": 0.0-1.0,
      "rationale": "explanation",
      "message_numbers": [1, 2, 3]
    }
  ],
  "next_best_action": "suggested action for the dating app"
}"""
    
    def _create_analysis_prompt(self, context: str) -> str:
        """Create analysis prompt for LLM"""
        return f"""Please analyze this conversation for match-seeking signals:

{context}

Focus on identifying moments where either participant shows clear interest in:
- Meeting in person
- Going on a date
- Spending time together
- Building a relationship
- Making future plans together

Look for both explicit statements and subtle indicators of interest escalation.

Provide your analysis in the specified JSON format."""
    
    def _parse_llm_response(self, response: str, conversation_id: str) -> List[Signal]:
        """Parse LLM response into Signal objects"""
        signals = []
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in LLM response")
                return signals
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Parse signals
            for signal_data in data.get('signals', []):
                signal = self._create_signal_from_llm(signal_data, conversation_id)
                if signal:
                    signals.append(signal)
            
            logger.info(f"Parsed {len(signals)} signals from LLM response")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
        
        return signals
    
    def _create_signal_from_llm(self, signal_data: Dict[str, Any], conversation_id: str) -> Optional[Signal]:
        """Create Signal object from LLM response data"""
        try:
            signal_type = signal_data.get('signal_type', '')
            confidence = float(signal_data.get('confidence', 0.0))
            rationale = signal_data.get('rationale', '')
            message_numbers = signal_data.get('message_numbers', [])
            
            # Validate signal type
            valid_types = [
                'match_seeking', 'interest_escalation', 'commitment_language',
                'question_asking', 'sentiment_shift', 'response_urgency'
            ]
            
            if signal_type not in valid_types:
                logger.warning(f"Invalid signal type from LLM: {signal_type}")
                return None
            
            # Validate confidence
            if not 0.0 <= confidence <= 1.0:
                logger.warning(f"Invalid confidence from LLM: {confidence}")
                confidence = max(0.0, min(1.0, confidence))
            
            return Signal(
                conversation_id=conversation_id,
                timestamp=datetime.now(),  # LLM doesn't provide specific timestamps
                signal_type=signal_type,
                confidence=confidence,
                rationale=f"LLM Analysis: {rationale}",
                message_ids=[],  # Would need to map message numbers to IDs
                metadata={
                    "source": "llm",
                    "message_numbers": message_numbers,
                    "raw_confidence": confidence,
                    "raw_rationale": rationale
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating signal from LLM data: {e}")
            return None
    
    def generate_next_best_action(self, conversation: Conversation, signals: List[Signal]) -> str:
        """Generate next best action based on conversation and signals"""
        if not self.client:
            return self._generate_heuristic_action(signals)
        
        try:
            # Prepare context for action generation
            context = self._prepare_action_context(conversation, signals)
            
            prompt = f"""Based on this conversation analysis, suggest the next best action for a dating app to help these users connect:

{context}

Provide a specific, actionable recommendation that would help facilitate a successful match. Consider:
- The current level of interest shown
- The type of signals detected
- Appropriate timing and approach
- User preferences and conversation flow

Respond with a single, clear action recommendation (1-2 sentences max)."""
            
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a dating app optimization expert. Provide clear, actionable recommendations for facilitating user connections."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            action = response.choices[0].message.content.strip()
            logger.info(f"Generated next best action: {action}")
            return action
            
        except Exception as e:
            logger.error(f"Error generating next best action: {e}")
            return self._generate_heuristic_action(signals)
    
    def _prepare_action_context(self, conversation: Conversation, signals: List[Signal]) -> str:
        """Prepare context for action generation"""
        context_parts = []
        
        # Conversation summary
        context_parts.append(f"Conversation: {len(conversation.messages)} messages between {', '.join(conversation.participants)}")
        context_parts.append(f"Duration: {conversation.duration_minutes:.1f} minutes" if conversation.duration_minutes else "Duration: Unknown")
        context_parts.append("")
        
        # Recent messages
        context_parts.append("Recent messages:")
        recent_messages = conversation.messages[-5:]  # Last 5 messages
        for msg in recent_messages:
            context_parts.append(f"- {msg.sender_id}: {msg.content}")
        context_parts.append("")
        
        # Detected signals
        context_parts.append("Detected signals:")
        for signal in signals:
            context_parts.append(f"- {signal.signal_type}: {signal.confidence:.2f} confidence - {signal.rationale}")
        
        return "\n".join(context_parts)
    
    def _generate_heuristic_action(self, signals: List[Signal]) -> str:
        """Generate action using heuristic rules when LLM is unavailable"""
        if not signals:
            return "Suggest an engaging question to spark conversation"
        
        # Find highest confidence signal
        best_signal = max(signals, key=lambda s: s.confidence)
        
        action_map = {
            "match_seeking": "Suggest a specific meetup time and location",
            "interest_escalation": "Encourage continued conversation with a personal question",
            "commitment_language": "Facilitate a deeper connection with relationship-focused prompts",
            "question_asking": "Acknowledge their interest and suggest meeting to continue the conversation",
            "sentiment_shift": "Capitalize on positive momentum with a meetup suggestion",
            "response_urgency": "Provide immediate response options or quick meetup opportunities"
        }
        
        return action_map.get(best_signal.signal_type, "Suggest an engaging follow-up question")
    
    def is_available(self) -> bool:
        """Check if LLM analysis is available"""
        return self.client is not None