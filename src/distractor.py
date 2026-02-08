"""
Distractor Model Module

This module implements noise injection strategies to enhance model robustness
by creating noisy versions of user queries and feedback.

Includes:
- Legacy rule-based noise strategies (TypoNoise, WordSwapNoise, etc.)
- New semantic-aware three-layer distractor (SemanticDistractorModel)
"""

import random
import re
import logging
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .llm_client import LLMClient, create_llm_client
from .intent_extractor import IntentSlotExtractor, ExtractedSemantics
from .noise_generator import LLMNoiseGenerator, NoiseResult


@dataclass
class NoisyVersion:
    """Represents a noisy version of text."""
    original_text: str
    noisy_text: str
    noise_type: str
    noise_intensity: float
    metadata: Dict[str, Any]


class NoiseStrategy:
    """Base class for noise injection strategies."""
    
    def apply(self, text: str, intensity: float = 0.1) -> str:
        """
        Apply noise to text.
        
        Args:
            text: Original text
            intensity: Noise intensity (0-1)
            
        Returns:
            Noisy text
        """
        raise NotImplementedError


class TypoNoise(NoiseStrategy):
    """Inject realistic typos into text."""
    
    # Common typo patterns
    TYPO_PATTERNS = {
        'a': ['s', 'q'],
        'b': ['v', 'n'],
        'c': ['x', 'v'],
        'd': ['s', 'f'],
        'e': ['w', 'r'],
        'f': ['d', 'g'],
        'g': ['f', 'h'],
        'h': ['g', 'j'],
        'i': ['u', 'o'],
        'j': ['h', 'k'],
        'k': ['j', 'l'],
        'l': ['k', 'o'],
        'm': ['n', ','],
        'n': ['b', 'm'],
        'o': ['i', 'p'],
        'p': ['o', '['],
        'q': ['w', 'a'],
        'r': ['e', 't'],
        's': ['a', 'd'],
        't': ['r', 'y'],
        'u': ['y', 'i'],
        'v': ['c', 'b'],
        'w': ['q', 'e'],
        'x': ['z', 'c'],
        'y': ['t', 'u'],
        'z': ['x', 's'],
    }
    
    def apply(self, text: str, intensity: float = 0.1) -> str:
        """Inject typos."""
        words = text.split()
        num_typos = max(1, int(len(words) * intensity))
        
        for _ in range(num_typos):
            if not words:
                break
            
            # Pick random word
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]
            
            if len(word) <= 2:
                continue
            
            # Pick random character
            char_idx = random.randint(0, len(word) - 1)
            char = word[char_idx].lower()
            
            if char in self.TYPO_PATTERNS:
                replacements = self.TYPO_PATTERNS[char]
                new_char = random.choice(replacements)
                
                # Preserve case
                if word[char_idx].isupper():
                    new_char = new_char.upper()
                
                # Replace character
                word_list = list(word)
                word_list[char_idx] = new_char
                words[word_idx] = ''.join(word_list)
        
        return ' '.join(words)


class WordSwapNoise(NoiseStrategy):
    """Swap adjacent words."""
    
    def apply(self, text: str, num_swaps: int = 2) -> str:
        """Swap adjacent words."""
        words = text.split()
        
        if len(words) < 2:
            return text
        
        num_swaps = min(num_swaps, len(words) // 2)
        
        for _ in range(num_swaps):
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)


class WordDeletionNoise(NoiseStrategy):
    """Delete random words."""
    
    def apply(self, text: str, deletion_rate: float = 0.05) -> str:
        """Delete words."""
        words = text.split()
        num_deletions = max(1, int(len(words) * deletion_rate))
        
        if len(words) <= num_deletions + 2:  # Keep at least some words
            return text
        
        indices_to_delete = random.sample(range(len(words)), num_deletions)
        
        return ' '.join(word for i, word in enumerate(words) if i not in indices_to_delete)


class WordInsertionNoise(NoiseStrategy):
    """Insert filler words."""
    
    FILLER_WORDS = [
        'like', 'um', 'uh', 'basically', 'actually', 'you know',
        'I mean', 'sort of', 'kind of', 'just'
    ]
    
    def apply(self, text: str, insertion_rate: float = 0.05) -> str:
        """Insert filler words."""
        words = text.split()
        num_insertions = max(1, int(len(words) * insertion_rate))
        
        for _ in range(num_insertions):
            if not words:
                break
            
            idx = random.randint(0, len(words))
            filler = random.choice(self.FILLER_WORDS)
            words.insert(idx, filler)
        
        return ' '.join(words)


class SynonymReplacementNoise(NoiseStrategy):
    """Replace words with simple synonyms."""
    
    SIMPLE_SYNONYMS = {
        'big': ['large', 'huge', 'massive'],
        'small': ['tiny', 'little', 'mini'],
        'good': ['great', 'nice', 'fine'],
        'bad': ['poor', 'terrible', 'awful'],
        'fast': ['quick', 'rapid', 'swift'],
        'slow': ['sluggish', 'gradual'],
        'help': ['assist', 'aid', 'support'],
        'make': ['create', 'build', 'produce'],
        'get': ['obtain', 'acquire', 'receive'],
        'show': ['display', 'demonstrate', 'present'],
    }
    
    def apply(self, text: str, replacement_rate: float = 0.15) -> str:
        """Replace words with synonyms."""
        words = text.split()
        num_replacements = max(1, int(len(words) * replacement_rate))
        
        for _ in range(num_replacements):
            if not words:
                break
            
            # Try to find a word we can replace
            for _ in range(10):  # Max attempts
                word_idx = random.randint(0, len(words) - 1)
                word = words[word_idx].lower().strip('.,!?')
                
                if word in self.SIMPLE_SYNONYMS:
                    synonyms = self.SIMPLE_SYNONYMS[word]
                    replacement = random.choice(synonyms)
                    
                    # Preserve case
                    if words[word_idx][0].isupper():
                        replacement = replacement.capitalize()
                    
                    words[word_idx] = replacement
                    break
        
        return ' '.join(words)


class LLMParaphraseNoise(NoiseStrategy):
    """Use LLM to paraphrase text."""
    
    def __init__(self, llm_client: LLMClient, template_path: str):
        """Initialize with LLM client."""
        self.llm_client = llm_client
        self.template_path = template_path
        self.template = self._load_template()
    
    def _load_template(self) -> str:
        """Load template."""
        with open(self.template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def apply(self, text: str, intensity: float = 0.5) -> str:
        """Paraphrase using LLM."""
        try:
            prompt = self.template.replace('{text}', text)
            prompt = prompt.replace('{noise_type}', 'paraphrase with minor changes')
            
            noisy_text = self.llm_client.generate(prompt)
            return noisy_text
        except Exception as e:
            logging.error(f"Error in LLM paraphrase: {str(e)}")
            return text


class DistractorModel:
    """
    Main distractor model that applies various noise strategies.
    """
    
    def __init__(self, config: Dict[str, Any], llm_client: Optional[LLMClient] = None):
        """
        Initialize DistractorModel.
        
        Args:
            config: Configuration dictionary
            llm_client: Optional LLM client for paraphrase noise (deprecated, use config)
        """
        self.config = config
        self.distractor_config = config.get('distractor', {})
        self.enabled = self.distractor_config.get('enabled', True)
        
        # Initialize noise strategies
        self.strategies = {
            'typo': TypoNoise(),
            'word_swap': WordSwapNoise(),
            'word_deletion': WordDeletionNoise(),
            'word_insertion': WordInsertionNoise(),
            'synonym_replacement': SynonymReplacementNoise(),
        }
        
        # Initialize distractor LLM client from config (independent from other LLMs)
        llm_config = self.distractor_config.get('llm', {})
        if llm_config and 'template' in llm_config:
            # Create independent LLM client for distractor
            if llm_client is None:  # Only create if not provided
                # Support both max_tokens and max_completion_tokens
                distractor_max_tokens = llm_config.get('max_completion_tokens') or llm_config.get('max_tokens', 512)
                
                distractor_llm = create_llm_client({
                    'provider': llm_config.get('provider', 'openai'),
                    'api_key': llm_config.get('api_key', config.get('api', {}).get('api_key', '')),
                    'model': llm_config.get('model', 'gpt-3.5-turbo'),
                    'inference': {
                        'temperature': llm_config.get('temperature', 0.8),
                        'max_completion_tokens': distractor_max_tokens
                    }
                })
            else:
                distractor_llm = llm_client
            
            template_path = llm_config['template']
            self.strategies['paraphrase'] = LLMParaphraseNoise(distractor_llm, template_path)
        
        # Load noise strategy configurations
        self.noise_strategies_config = self.distractor_config.get('noise_strategies', [])
        self.apply_to_initial = self.distractor_config.get('apply_to', {}).get('initial_query', True)
        self.apply_to_followup = self.distractor_config.get('apply_to', {}).get('follow_up_feedback', True)
    
    def apply_noise(self, text: str, force: bool = False) -> List[NoisyVersion]:
        """
        Apply various noise strategies to text.
        
        Args:
            text: Original text
            force: Force noise application (ignore probabilities)
            
        Returns:
            List of NoisyVersion objects
        """
        if not self.enabled and not force:
            return []
        
        noisy_versions = []
        
        for strategy_config in self.noise_strategies_config:
            name = strategy_config['name']
            probability = strategy_config.get('probability', 1.0)
            
            # Decide whether to apply this strategy
            if not force and random.random() > probability:
                continue
            
            if name not in self.strategies:
                logging.warning(f"Noise strategy '{name}' not implemented")
                continue
            
            strategy = self.strategies[name]
            
            try:
                # Apply noise with strategy-specific parameters
                if name == 'typo':
                    intensity = strategy_config.get('intensity', 0.1)
                    noisy_text = strategy.apply(text, intensity)
                
                elif name == 'word_swap':
                    num_swaps = strategy_config.get('num_swaps', 2)
                    noisy_text = strategy.apply(text, num_swaps)
                
                elif name == 'word_deletion':
                    deletion_rate = strategy_config.get('deletion_rate', 0.05)
                    noisy_text = strategy.apply(text, deletion_rate)
                
                elif name == 'word_insertion':
                    insertion_rate = strategy_config.get('insertion_rate', 0.05)
                    noisy_text = strategy.apply(text, insertion_rate)
                
                elif name == 'synonym_replacement':
                    replacement_rate = strategy_config.get('replacement_rate', 0.15)
                    noisy_text = strategy.apply(text, replacement_rate)
                
                elif name == 'paraphrase':
                    if strategy_config.get('use_llm', True):
                        noisy_text = strategy.apply(text, 0.5)
                    else:
                        continue
                
                else:
                    noisy_text = strategy.apply(text)
                
                # Create noisy version object
                noisy_version = NoisyVersion(
                    original_text=text,
                    noisy_text=noisy_text,
                    noise_type=name,
                    noise_intensity=probability,
                    metadata=strategy_config
                )
                
                noisy_versions.append(noisy_version)
            
            except Exception as e:
                logging.error(f"Error applying {name} noise: {str(e)}")
        
        return noisy_versions
    
    def apply_to_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply noise to an interaction (initial query and follow-ups).
        
        Args:
            interaction: Interaction dictionary
            
        Returns:
            Enhanced interaction with noisy versions
        """
        result = interaction.copy()
        result['noisy_versions'] = {}
        
        # Apply noise to initial query
        if self.apply_to_initial:
            initial_query = interaction.get('initial_query', '')
            if initial_query:
                noisy_initial = self.apply_noise(initial_query)
                result['noisy_versions']['initial_query'] = [
                    {
                        'noisy_text': nv.noisy_text,
                        'noise_type': nv.noise_type,
                        'noise_intensity': nv.noise_intensity
                    }
                    for nv in noisy_initial
                ]
        
        # Apply noise to follow-up feedback
        if self.apply_to_followup:
            messages = interaction.get('messages', [])
            result['noisy_versions']['followups'] = []
            
            for msg in messages:
                if msg.get('role') == 'user' and msg.get('metadata', {}).get('is_followup'):
                    content = msg.get('content', '')
                    noisy_followups = self.apply_noise(content)
                    
                    result['noisy_versions']['followups'].append({
                        'original_message': msg,
                        'noisy_versions': [
                            {
                                'noisy_text': nv.noisy_text,
                                'noise_type': nv.noise_type,
                                'noise_intensity': nv.noise_intensity
                            }
                            for nv in noisy_followups
                        ]
                    })
        
        return result
    
    def apply_to_interactions_batch(self, interactions: List[Dict[str, Any]],
                                   show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Apply noise to multiple interactions.
        
        Args:
            interactions: List of interaction dictionaries
            show_progress: Whether to show progress
            
        Returns:
            List of enhanced interactions with noisy versions
        """
        enhanced_interactions = []
        
        for i, interaction in enumerate(interactions):
            enhanced = self.apply_to_interaction(interaction)
            enhanced_interactions.append(enhanced)
            
            if show_progress and (i + 1) % 50 == 0:
                logging.info(f"Applied noise to {i + 1}/{len(interactions)} interactions")
        
        return enhanced_interactions


# =============================================================================
# Semantic Distractor Model (New Three-Layer Architecture)
# =============================================================================

class SemanticDistractorModel:
    """
    Semantic-aware distractor model with three-layer noise injection.

    Layers:
    1. Surface Noise: Intent/slots intact, only surface form changes
    2. Incomplete Info: Intent clear, slots missing/vague
    3. Semantic Ambiguity: Intent uncertain or multiple intents

    Pipeline:
    1. Extract intent/slots from clean prompt
    2. Select one layer (weighted random)
    3. Sample strategies within the layer (mandatory + independent probability)
    4. Generate noisy version via LLM
    """

    # Layer name to index mapping
    LAYER_INDICES = {
        'surface_noise': 1,
        'incomplete_info': 2,
        'semantic_ambiguity': 3
    }

    def __init__(
        self,
        config: Dict[str, Any],
        strategy_config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize SemanticDistractorModel.

        Args:
            config: Main configuration dictionary
            strategy_config: Strategy configuration (if None, loads from strategy_path in config)
            llm_client: Optional LLM client (if None, creates from config)
        """
        self.config = config
        self.distractor_config = config.get('distractor', {})
        self.enabled = self.distractor_config.get('enabled', True)
        self.activation_probability = self.distractor_config.get('activation_probability', 1.0)
        self.logger = logging.getLogger(__name__)

        # Load strategy configuration
        if strategy_config is not None:
            self.strategy_config = strategy_config
        else:
            strategy_path = self.distractor_config.get('strategy_path', 'distractor_strategy.yaml')
            self.strategy_config = self._load_strategy_config(strategy_path)

        # Initialize LLM client
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = self._create_llm_client()

        # Initialize components
        self.extractor = IntentSlotExtractor(self.llm_client)
        self.generator = LLMNoiseGenerator(
            self.llm_client,
            self.strategy_config.get('strategies', {})
        )

        # Extract layer weights
        self.layer_weights = self.strategy_config.get('layer_weights', {
            'surface_noise': 0.5,
            'incomplete_info': 0.3,
            'semantic_ambiguity': 0.2
        })

        # Validate weights sum to 1.0
        total_weight = sum(self.layer_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Layer weights sum to {total_weight}, normalizing...")
            self.layer_weights = {k: v / total_weight for k, v in self.layer_weights.items()}

    def _load_strategy_config(self, strategy_path: str) -> Dict[str, Any]:
        """Load strategy configuration from YAML file."""
        try:
            with open(strategy_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Strategy config not found at {strategy_path}, using defaults")
            return self._get_default_strategy_config()

    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """Return default strategy configuration."""
        return {
            'layer_weights': {
                'surface_noise': 0.5,
                'incomplete_info': 0.3,
                'semantic_ambiguity': 0.2
            },
            'strategies': {
                'surface_noise': {
                    'colloquial_speech': {'mandatory': True},
                    'incomplete_sentence': {'mandatory': True},
                    'typo_misspelling': {'probability': 0.3},
                    'punctuation_format': {'probability': 0.5},
                    'emotional_attitude': {'probability': 0.5}
                },
                'incomplete_info': {
                    'missing_slots': {'probability': 0.5},
                    'vague_slot_values': {'probability': 0.6},
                    'unclear_priority': {'probability': 0.7}
                },
                'semantic_ambiguity': {
                    'multi_intent': {'probability': 0.6},
                    'self_contradiction': {'probability': 0.7},
                    'mind_changing': {'probability': 0.4}
                }
            }
        }

    def _create_llm_client(self) -> LLMClient:
        """Create LLM client from configuration."""
        llm_config = self.strategy_config.get('llm', {})
        
        # Support both max_tokens and max_completion_tokens
        max_tokens_value = llm_config.get('max_completion_tokens') or llm_config.get('max_tokens', 1024)

        return create_llm_client({
            'provider': llm_config.get('provider', self.config.get('api', {}).get('provider', 'openai')),
            'api_key': self.config.get('api', {}).get('api_key', ''),
            'model': llm_config.get('model', 'gpt-4o-mini'),
            'inference': {
                'temperature': llm_config.get('temperature', 0.9),
                'max_completion_tokens': max_tokens_value
            }
        })

    def _select_layer(self) -> str:
        """
        Select a layer based on weights (weighted random).

        Returns:
            Layer name
        """
        layers = list(self.layer_weights.keys())
        weights = list(self.layer_weights.values())

        return random.choices(layers, weights=weights, k=1)[0]

    def _sample_strategies(self, layer: str) -> List[str]:
        """
        Sample strategies for the selected layer.

        Mandatory strategies are always included.
        Optional strategies are independently sampled based on probability.

        Args:
            layer: Layer name

        Returns:
            List of strategy names to apply
        """
        layer_strategies = self.strategy_config.get('strategies', {}).get(layer, {})
        selected = []

        for strategy_name, strategy_config in layer_strategies.items():
            if strategy_config.get('mandatory', False):
                # Mandatory strategy - always include
                selected.append(strategy_name)
            else:
                # Optional strategy - sample based on probability
                probability = strategy_config.get('probability', 0.5)
                if random.random() < probability:
                    selected.append(strategy_name)

        # Ensure at least one strategy is selected
        if not selected and layer_strategies:
            # Pick one random strategy as fallback
            selected.append(random.choice(list(layer_strategies.keys())))

        return selected

    def apply_noise(
        self,
        text: str,
        force: bool = False,
        persona_features: Optional[Dict[str, Any]] = None
    ) -> NoiseResult:
        """
        Apply semantic-aware noise to text.

        Args:
            text: Original text
            force: Force noise application (ignore enabled flag and activation probability)
            persona_features: Optional persona features for persona-aware noise generation

        Returns:
            NoiseResult object with noisy version and metadata
        """
        if not self.enabled and not force:
            return NoiseResult(
                original_text=text,
                noisy_text=text,
                layer='none',
                layer_index=0,
                applied_strategies=[]
            )

        # Check activation probability (unless forced)
        if not force and random.random() > self.activation_probability:
            return NoiseResult(
                original_text=text,
                noisy_text=text,
                layer='skipped',
                layer_index=0,
                applied_strategies=[],
                metadata={'reason': 'activation_probability_check_failed'}
            )

        try:
            # Step 1: Extract intent/slots
            semantics = self.extractor.extract(text)

            # Step 2: Select layer
            layer = self._select_layer()
            layer_index = self.LAYER_INDICES.get(layer, 1)

            # Step 3: Sample strategies
            strategies = self._sample_strategies(layer)

            # Step 4: Generate noisy version (with persona awareness)
            result = self.generator.generate_with_result(
                semantics=semantics,
                layer=layer,
                layer_index=layer_index,
                strategies=strategies,
                persona_features=persona_features
            )

            return result

        except Exception as e:
            self.logger.error(f"Error applying semantic noise: {e}")
            return NoiseResult(
                original_text=text,
                noisy_text=text,
                layer='error',
                layer_index=0,
                applied_strategies=[],
                metadata={'error': str(e)}
            )

    def apply_noise_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
        persona_features: Optional[Dict[str, Any]] = None
    ) -> List[NoiseResult]:
        """
        Apply noise to multiple texts.

        Args:
            texts: List of text strings
            show_progress: Whether to show progress
            persona_features: Optional persona features for persona-aware noise generation

        Returns:
            List of NoiseResult objects
        """
        results = []

        for i, text in enumerate(texts):
            result = self.apply_noise(text, persona_features=persona_features)
            results.append(result)

            if show_progress and (i + 1) % 10 == 0:
                self.logger.info(f"Applied semantic noise to {i + 1}/{len(texts)} texts")

        return results

    def apply_to_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply semantic noise to an interaction.

        Args:
            interaction: Interaction dictionary with initial_query and messages

        Returns:
            Enhanced interaction with noisy versions
        """
        result = interaction.copy()
        result['semantic_noise'] = {}

        # Apply noise to initial query
        initial_query = interaction.get('initial_query', '')
        if initial_query:
            noise_result = self.apply_noise(initial_query)
            result['semantic_noise']['initial_query'] = noise_result.to_dict()

        # Apply noise to follow-up feedback
        messages = interaction.get('messages', [])
        result['semantic_noise']['followups'] = []

        for msg in messages:
            if msg.get('role') == 'user' and msg.get('metadata', {}).get('is_followup'):
                content = msg.get('content', '')
                noise_result = self.apply_noise(content)
                result['semantic_noise']['followups'].append({
                    'original_message': msg,
                    'noise_result': noise_result.to_dict()
                })

        return result


def create_distractor_model(
    config: Dict[str, Any],
    use_semantic: bool = True,
    llm_client: Optional[LLMClient] = None
) -> 'DistractorModel | SemanticDistractorModel':
    """
    Factory function to create appropriate distractor model.

    Args:
        config: Configuration dictionary
        use_semantic: Whether to use semantic distractor (True) or legacy (False)
        llm_client: Optional LLM client

    Returns:
        DistractorModel or SemanticDistractorModel instance
    """
    if use_semantic:
        return SemanticDistractorModel(config, llm_client=llm_client)
    else:
        return DistractorModel(config, llm_client=llm_client)
