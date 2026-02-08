"""
Noise Generator Module

This module generates noisy versions of prompts using LLM based on
selected layer and strategies from the distractor strategy configuration.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .llm_client import LLMClient
from .intent_extractor import ExtractedSemantics

# Import token tracker
try:
    from .token_tracker import record_tokens
except ImportError:
    def record_tokens(*args, **kwargs):
        pass


@dataclass
class NoiseResult:
    """Result of noise generation."""
    original_text: str
    noisy_text: str
    layer: str                      # "surface_noise", "incomplete_info", "semantic_ambiguity"
    layer_index: int                # 1, 2, or 3
    applied_strategies: List[str]   # List of strategy names applied
    extracted_semantics: Optional[ExtractedSemantics] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_text': self.original_text,
            'noisy_text': self.noisy_text,
            'layer': self.layer,
            'layer_index': self.layer_index,
            'applied_strategies': self.applied_strategies,
            'extracted_semantics': self.extracted_semantics.to_dict() if self.extracted_semantics else None,
            'metadata': self.metadata
        }


# Layer descriptions for prompt generation
LAYER_DESCRIPTIONS = {
    'surface_noise': """Layer 1: Surface Noise (Intent/Slots Intact)
- The core meaning and all key information must be preserved
- Only the surface form/expression changes
- Make it sound like a real person typing casually
- Add realistic noise like typos, informal language, trailing sentences""",

    'incomplete_info': """Layer 2: Incomplete Information (Intent Clear, Slots Incomplete)
- The main intent/goal must remain clear
- Remove, obscure, or make vague some specific details/parameters
- The request should still be understandable but missing some specifics
- The assistant would need to ask clarifying questions or make assumptions""",

    'semantic_ambiguity': """Layer 3: Semantic Ambiguity (Intent Uncertain)
- The request should be ambiguous about what exactly is being asked
- Could include multiple intents, contradictions, or unclear goals
- The assistant would need to clarify what the user actually wants
- May include mid-sentence corrections or conflicting requirements"""
}

# Strategy descriptions for prompt generation
STRATEGY_DESCRIPTIONS = {
    # Layer 1: Surface Noise
    'colloquial_speech': "Convert to casual, conversational tone with contractions and informal phrasing. IMPORTANT: Do NOT overuse filler words like 'um', 'like', 'hey'. Vary sentence starters naturally. Keep it simple and direct.",
    'incomplete_sentence': "Occasionally use fragments or skip obvious words, but keep most of the message clear and complete",
    'typo_misspelling': "Add 1-2 realistic keyboard typos (teh, scheduel, definately) - don't overdo it",
    'word_segmentation': "Occasionally add extra/missing spaces or slight character issues",
    'punctuation_format': "Use casual punctuation - skip periods, use one emoji max if appropriate",
    'synonym_rewrite': "Replace some words with casual synonyms while keeping meaning clear",
    'emotional_attitude': "Add subtle emotional tone (need this soon, would appreciate it) - don't be dramatic",
    'negative_expression': "Express preferences using negation (don't make it too formal) when natural",
    'real_life_fragments': "Optionally add brief context (quick question, on mobile) - don't force it",

    # Layer 2: Incomplete Info
    'missing_slots': "Remove specific slot values like time, location, budget, quantity, or format constraints",
    'vague_slot_values': "Replace precise values with vague expressions (not too expensive, sometime soon, detailed enough)",
    'context_dependency': "Add references to non-existent context (like last time, the usual, same as before)",
    'unclear_priority': "Delegate decisions to assistant (whatever you think is best, you decide, up to you)",

    # Layer 3: Semantic Ambiguity
    'multi_intent': "Add another unrelated request using casual transitions (oh and also, btw, one more thing)",
    'intent_ambiguity': "Make the core request ambiguous (fix this, make it better, help with this)",
    'self_contradiction': "Include conflicting requirements (keep it short but detailed, simple but comprehensive)",
    'mind_changing': "Include mid-request corrections (actually wait, no actually, on second thought)"
}


class LLMNoiseGenerator:
    """
    Generates noisy versions of prompts using LLM.

    Takes extracted semantics, selected layer, and strategies,
    then uses LLM to generate a realistic noisy version.
    Supports persona-aware noise generation to ensure consistency with user profile.
    """

    DEFAULT_PROMPT_TEMPLATE = """You are an expert at transforming clean user prompts into realistic, natural versions that real users might actually type.

## User Persona
{persona}

## Persona-Aware Noise Generation
The noisy version MUST be consistent with the user persona above.
- Analyze the persona features and determine which ones are relevant to how this user would naturally express themselves
- Only apply noise patterns that match the persona's communication style
- Do NOT add noise elements that contradict the persona (e.g., don't add emoji for a formal persona, don't add internet slang for an older user)
- The noise should make the text more realistic for THIS specific persona, not a generic user

## Original Request Information
**Intent**: {intent}
**Key Information (Slots)**: {slots}
**Clean Version**: {compressed_text}

## Target Noise Profile
{layer_description}

## Strategies to Apply (apply ALL of these naturally)
{strategies_description}

## CRITICAL Guidelines (MUST FOLLOW)
- Apply ALL listed strategies naturally in a single coherent message
- Make it sound like a REAL person typing - natural and varied
- **IMPORTANT**: Ensure the noise matches the user persona above
- **AVOID** starting with "hey", "um", "so" every time - vary your sentence starters
- **AVOID** overusing filler words (like, um, uh) - use 0-1 max per message
- **AVOID** excessive punctuation (!!, ??) or too many emoji
- The output should be a single user message, not a conversation
- Do NOT include any explanations or meta-commentary
- Do NOT use quotation marks around the output
- Keep roughly the same length as the original

## Output
Generate ONLY the noisy version of the prompt. Nothing else:"""

    def __init__(
        self,
        llm_client: LLMClient,
        strategy_config: Dict[str, Any],
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the noise generator.

        Args:
            llm_client: LLM client for generation
            strategy_config: Strategy configuration from YAML
            prompt_template: Custom prompt template (optional)
        """
        self.llm_client = llm_client
        self.strategy_config = strategy_config
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

    def generate(
        self,
        semantics: ExtractedSemantics,
        layer: str,
        strategies: List[str],
        persona_features: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate noisy version of prompt.

        Args:
            semantics: Extracted semantic information
            layer: Layer name ("surface_noise", "incomplete_info", "semantic_ambiguity")
            strategies: List of strategy names to apply
            persona_features: Optional persona features for persona-aware noise generation

        Returns:
            Noisy version of the prompt
        """
        try:
            # Build the generation prompt
            prompt = self._build_prompt(semantics, layer, strategies, persona_features)

            # Call LLM with token tracking
            response, input_tokens, output_tokens = self.llm_client.generate_with_tokens(prompt)
            
            # Record token usage
            record_tokens(
                module='distractor',
                operation=f'noise_generation_{layer}',
                model=self.model_name,
                provider=self.provider_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={'layer': layer, 'strategies': strategies}
            )

            # Clean up response
            noisy_text = self._clean_response(response)

            return noisy_text

        except Exception as e:
            self.logger.error(f"Failed to generate noisy version: {e}")
            # Return original text as fallback
            return semantics.original_text

    def _build_prompt(
        self,
        semantics: ExtractedSemantics,
        layer: str,
        strategies: List[str],
        persona_features: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the LLM prompt for noise generation."""
        # Format persona
        if persona_features:
            persona_str = "\n".join(f"- {k}: {v}" for k, v in persona_features.items())
        else:
            persona_str = "(no specific persona constraints - generate generic noise)"

        # Format slots
        if semantics.slots:
            slots_str = "\n".join(f"- {k}: {v}" for k, v in semantics.slots.items())
        else:
            slots_str = "(none explicitly specified)"

        # Get layer description
        layer_description = LAYER_DESCRIPTIONS.get(layer, LAYER_DESCRIPTIONS['surface_noise'])

        # Build strategies description
        strategies_parts = []
        for i, strategy in enumerate(strategies, 1):
            desc = STRATEGY_DESCRIPTIONS.get(strategy, strategy)
            strategies_parts.append(f"{i}. **{strategy}**: {desc}")
        strategies_description = "\n".join(strategies_parts)

        # Build final prompt
        return self.prompt_template.format(
            persona=persona_str,
            intent=semantics.intent,
            slots=slots_str,
            compressed_text=semantics.compressed_text,
            layer_description=layer_description,
            strategies_description=strategies_description
        )

    def _clean_response(self, response: str) -> str:
        """Clean up LLM response."""
        # Remove leading/trailing whitespace
        response = response.strip()

        # Remove surrounding quotes if present
        if (response.startswith('"') and response.endswith('"')) or \
           (response.startswith("'") and response.endswith("'")):
            response = response[1:-1]

        # Remove markdown code blocks if present
        if response.startswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1] if lines[-1].startswith('```') else lines[1:])

        return response.strip()

    def generate_with_result(
        self,
        semantics: ExtractedSemantics,
        layer: str,
        layer_index: int,
        strategies: List[str],
        persona_features: Optional[Dict[str, Any]] = None
    ) -> NoiseResult:
        """
        Generate noisy version and return full result object.

        Args:
            semantics: Extracted semantic information
            layer: Layer name
            layer_index: Layer number (1, 2, or 3)
            strategies: List of strategy names to apply
            persona_features: Optional persona features for persona-aware noise generation

        Returns:
            NoiseResult object with all information
        """
        noisy_text = self.generate(semantics, layer, strategies, persona_features)

        return NoiseResult(
            original_text=semantics.original_text,
            noisy_text=noisy_text,
            layer=layer,
            layer_index=layer_index,
            applied_strategies=strategies,
            extracted_semantics=semantics
        )


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
