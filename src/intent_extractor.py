"""
Intent Extractor Module

This module extracts intent and slots from user prompts to enable
semantic-aware noise injection.
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .llm_client import LLMClient

# Import token tracker
try:
    from .token_tracker import record_tokens
except ImportError:
    def record_tokens(*args, **kwargs):
        pass


@dataclass
class ExtractedSemantics:
    """Represents extracted semantic information from a prompt."""
    intent: str                    # Core user intent (e.g., "write_email", "summarize")
    slots: Dict[str, Any]          # Key-value pairs of parameters
    original_text: str             # Original prompt text
    compressed_text: str           # Minimal representation of the request
    confidence: float = 1.0        # Extraction confidence score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedSemantics':
        """Create from dictionary."""
        return cls(**data)


class IntentSlotExtractor:
    """
    Extracts intent and slots from user prompts using LLM.

    This enables semantic-aware noise injection where we can:
    1. Preserve intent while adding surface noise (Layer 1)
    2. Remove/modify slots while keeping intent (Layer 2)
    3. Add ambiguity to intent itself (Layer 3)
    """

    DEFAULT_PROMPT_TEMPLATE = """You are an expert at analyzing user requests and extracting their core intent and parameters.

Given the following user prompt, extract:
1. **intent**: The core action the user wants (use snake_case, e.g., "write_email", "summarize_document", "debug_code")
2. **slots**: Key parameters/constraints mentioned (as key-value pairs)
3. **compressed_text**: A minimal, clean version of the request with just the essential information

User Prompt:
{text}

Respond in JSON format only:
{{
    "intent": "<core_intent>",
    "slots": {{
        "key1": "value1",
        "key2": "value2"
    }},
    "compressed_text": "<minimal request>"
}}

Examples of slot keys: topic, audience, format, length, tone, deadline, budget, quantity, style, constraints

JSON Response:"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the extractor.

        Args:
            llm_client: LLM client for extraction
            prompt_template: Custom prompt template (optional)
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.logger = logging.getLogger(__name__)
        
        # Get model info for token tracking
        self.model_name = getattr(llm_client, 'model', 'unknown')
        client_class = type(llm_client).__name__
        if 'OpenAI' in client_class:
            self.provider_name = 'openai'
        elif 'Anthropic' in client_class:
            self.provider_name = 'anthropic'
        else:
            self.provider_name = 'unknown'

    def extract(self, text: str) -> ExtractedSemantics:
        """
        Extract intent and slots from text.

        Args:
            text: User prompt text

        Returns:
            ExtractedSemantics object with intent, slots, and compressed text
        """
        try:
            # Build the extraction prompt
            prompt = self.prompt_template.format(text=text)

            # Call LLM with token tracking
            response, input_tokens, output_tokens = self.llm_client.generate_with_tokens(prompt)
            
            # Record token usage
            record_tokens(
                module='distractor',
                operation='intent_extraction',
                model=self.model_name,
                provider=self.provider_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={'text_length': len(text)}
            )

            # Parse JSON response
            result = self._parse_response(response)

            return ExtractedSemantics(
                intent=result.get('intent', 'unknown'),
                slots=result.get('slots', {}),
                original_text=text,
                compressed_text=result.get('compressed_text', text),
                confidence=1.0
            )

        except Exception as e:
            self.logger.warning(f"Failed to extract semantics: {e}. Using fallback.")
            return self._fallback_extraction(text)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response as JSON.

        Args:
            response: Raw LLM response

        Returns:
            Parsed dictionary
        """
        # Clean up response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith('```'):
            # Remove ```json and ``` markers
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise

    def _fallback_extraction(self, text: str) -> ExtractedSemantics:
        """
        Fallback extraction when LLM fails.

        Args:
            text: Original text

        Returns:
            Basic ExtractedSemantics with minimal info
        """
        # Simple heuristic extraction
        words = text.lower().split()

        # Try to identify intent from common verbs
        intent_verbs = {
            'write': 'write',
            'create': 'create',
            'make': 'create',
            'help': 'assist',
            'explain': 'explain',
            'summarize': 'summarize',
            'translate': 'translate',
            'fix': 'fix',
            'debug': 'debug',
            'review': 'review',
            'edit': 'edit',
            'improve': 'improve',
            'analyze': 'analyze',
            'find': 'search',
            'search': 'search',
            'generate': 'generate',
            'list': 'list',
            'compare': 'compare',
        }

        intent = 'general_request'
        for word in words:
            if word in intent_verbs:
                intent = intent_verbs[word]
                break

        return ExtractedSemantics(
            intent=intent,
            slots={},
            original_text=text,
            compressed_text=text,
            confidence=0.5
        )

    def extract_batch(self, texts: list) -> list:
        """
        Extract semantics from multiple texts.

        Args:
            texts: List of text strings

        Returns:
            List of ExtractedSemantics objects
        """
        return [self.extract(text) for text in texts]


def load_prompt_template(template_path: str) -> str:
    """
    Load prompt template from file.

    Args:
        template_path: Path to template file

    Returns:
        Template string
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()
