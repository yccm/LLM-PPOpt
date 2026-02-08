"""
Interaction Generator Module

This module handles multi-turn dialogue generation between a user and an assistant,
collecting training signals based on user satisfaction.
"""

import json
import random
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pathlib import Path

from .llm_client import LLMClient, create_llm_client
from .distractor import DistractorModel, SemanticDistractorModel, create_distractor_model
from .noise_generator import NoiseResult
from .colored_logger import print_colored

# Import token tracker
try:
    from .token_tracker import record_tokens
except ImportError:
    def record_tokens(*args, **kwargs):
        pass


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result


@dataclass
class Interaction:
    """Represents a complete multi-turn interaction."""
    interaction_id: str
    persona_id: str
    persona_features: Dict[str, str]
    original_query: str  # Original query from dataset (before style transfer)
    initial_query: str   # Query after style transfer (used in conversation)
    messages: List[Message]
    num_turns: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'interaction_id': self.interaction_id,
            'persona_id': self.persona_id,
            'persona_features': self.persona_features,
            'original_query': self.original_query,
            'initial_query': self.initial_query,
            'messages': [msg.to_dict() for msg in self.messages],
            'num_turns': self.num_turns,
            'metadata': self.metadata
        }
    
    def get_trajectory(self) -> List[str]:
        """Get the trajectory of user prompts."""
        return [msg.content for msg in self.messages if msg.role == 'user']
    
    def get_full_conversation(self) -> List[Dict[str, str]]:
        """Get the full conversation as a list of messages."""
        return [{'role': msg.role, 'content': msg.content} for msg in self.messages]


class InteractionStorage:
    """
    Manages incremental storage of interactions — saves each interaction to disk
    immediately after it is generated (one JSON file per interaction + index).
    Saves in TrainingSample format (consistent with train_samples export).
    """

    def __init__(self, output_dir: str, collector=None):
        """
        Args:
            output_dir: Directory to store interaction files.
            collector: Optional TrainingDataCollector for converting to TrainingSample format.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collector = collector

        self.index_file = self.output_dir / 'index.json'
        self.index = self._load_index()
        self._lock = threading.Lock()

    def _load_index(self) -> Dict[str, Any]:
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'interactions': [], 'count': 0, 'last_updated': None}

    def _save_index(self):
        self.index['last_updated'] = datetime.now().isoformat()
        self.index['count'] = len(self.index['interactions'])
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    def save(self, interaction: 'Interaction') -> str:
        """
        Save a single interaction to disk immediately.

        If a TrainingDataCollector is available, converts to TrainingSample
        format first so the output is consistent with train_samples export.

        Thread-safe: uses a lock to protect index updates.

        Returns:
            Path to saved file.
        """
        filename = f"{interaction.interaction_id}.json"
        filepath = self.output_dir / filename

        # Convert to TrainingSample format if collector is available
        if self.collector is not None:
            interaction_dict = interaction.to_dict()
            sample = self.collector.interaction_to_training_sample(interaction_dict)
            data = sample.to_dict()
        else:
            data = interaction.to_dict()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        with self._lock:
            existing_ids = {e['interaction_id'] for e in self.index['interactions']}
            if interaction.interaction_id not in existing_ids:
                self.index['interactions'].append({
                    'interaction_id': interaction.interaction_id,
                    'persona_id': interaction.persona_id,
                    'filepath': str(filepath),
                    'num_turns': interaction.num_turns,
                    'created_at': interaction.metadata.get('created_at'),
                    'query_id': interaction.metadata.get('query_id'),
                })
                self._save_index()

        return str(filepath)

    def load(self, interaction_id: str) -> Dict[str, Any]:
        filepath = self.output_dir / f"{interaction_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Interaction {interaction_id} not found")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_all(self) -> List[Dict[str, Any]]:
        """Load all stored interactions as dicts."""
        results = []
        for entry in self.index['interactions']:
            try:
                data = self.load(entry['interaction_id'])
                results.append(data)
            except FileNotFoundError:
                logging.warning(f"Interaction {entry['interaction_id']} in index but file not found")
        return results


class AssistantModel:
    """
    Simulates an AI assistant responding to user queries.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        system_prompt: Optional[str] = None,
        model_pool: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize AssistantModel.

        Args:
            llm_client: Single LLM client for generating responses (legacy / simple mode)
            system_prompt: Optional system prompt for personalization
            model_pool: Optional model pool entries, each containing:
              - client: LLMClient (required)
              - weight: float (optional, default 1.0)
              - provider/model: optional metadata for logging
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt or "You are a helpful AI assistant. Answer based on what the user explicitly asks. Do not make assumptions about the user's background or expertise."
        self.model_pool = model_pool or []
        # Thread-local storage for locked client and model info (ensures thread-safe model locking in concurrent execution)
        self._thread_local = threading.local()
    
    def _format_conversation_history(self, messages: List[Message]) -> str:
        """Format conversation history for prompt."""
        if not messages:
            return "No previous conversation."
        
        history = []
        for msg in messages:
            role = "User" if msg.role == 'user' else "Assistant"
            history.append(f"{role}: {msg.content}")
        
        return '\n'.join(history)
    
    def respond(self, user_query: str, conversation_history: List[Message]) -> str:
        """
        Generate assistant response to user query.
        
        Args:
            user_query: Current user query
            conversation_history: Previous conversation messages
            
        Returns:
            Assistant response text
        """
        try:
            # Format conversation history
            history_text = self._format_conversation_history(conversation_history)

            # Build prompt directly (no external template file)
            prompt = (
                f"{self.system_prompt}\n\n"
                f"Conversation History:\n{history_text}\n\n"
                f"User Query:\n{user_query}\n\n"
                "Provide a helpful, accurate, and appropriate response based on the conversation context and the user's needs."
            )
            
            # Generate response with token tracking
            client = self._select_client()
            response, input_tokens, output_tokens = client.generate_with_tokens(prompt)
            
            # Record token usage
            model_info = self.last_model_info or {}
            record_tokens(
                module='interaction_generation',
                operation='assistant_response',
                model=model_info.get('model', 'unknown'),
                provider=model_info.get('provider', 'unknown'),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={'turn': len(conversation_history) // 2 + 1}
            )
            
            return response
        
        except Exception as e:
            logging.error(f"Error generating assistant response: {str(e)}")
            return None

    def lock_model(self) -> None:
        """
        Lock a model for the current conversation (thread-safe).
        Call this at the start of each conversation to ensure the same model is used throughout.
        Uses thread-local storage to support concurrent execution.
        """
        if self.model_pool:
            weights = [float(entry.get("weight", 1.0)) for entry in self.model_pool]
            chosen = random.choices(self.model_pool, weights=weights, k=1)[0]
            self._thread_local.locked_model_info = {
                "provider": chosen.get("provider"),
                "model": chosen.get("model"),
            }
            self._thread_local.locked_client = chosen.get("client")
            if not self._thread_local.locked_client:
                raise ValueError("Model pool entry missing 'client'")
        else:
            # Legacy single-client mode
            self._thread_local.locked_client = self.llm_client
            self._thread_local.locked_model_info = None

    def unlock_model(self) -> None:
        """
        Unlock the model after the conversation ends (thread-safe).
        Uses thread-local storage to support concurrent execution.
        """
        self._thread_local.locked_client = None
        self._thread_local.locked_model_info = None

    @property
    def last_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get the model info for the current thread (thread-safe).

        Returns:
            Model info dictionary or None
        """
        return getattr(self._thread_local, 'last_model_info', None)

    def _select_client(self) -> LLMClient:
        """
        Select an LLM client (thread-safe).

        - If a model is locked for the current thread's conversation, use it.
        - If a model pool is provided and no lock, sample by weight.
        - Otherwise fall back to the single configured client.
        """
        # Use locked client if available (ensures same model throughout conversation)
        # Access thread-local storage with getattr to handle uninitialized state
        locked_client = getattr(self._thread_local, 'locked_client', None)
        locked_model_info = getattr(self._thread_local, 'locked_model_info', None)

        if locked_client is not None:
            # Store in thread-local to avoid race conditions
            self._thread_local.last_model_info = locked_model_info
            return locked_client

        if self.model_pool:
            weights = [float(entry.get("weight", 1.0)) for entry in self.model_pool]
            chosen = random.choices(self.model_pool, weights=weights, k=1)[0]
            # Store in thread-local to avoid race conditions
            self._thread_local.last_model_info = {
                "provider": chosen.get("provider"),
                "model": chosen.get("model"),
            }
            client = chosen.get("client")
            if not client:
                raise ValueError("Model pool entry missing 'client'")
            return client

        # Legacy single-client mode
        self._thread_local.last_model_info = None
        if not self.llm_client:
            raise ValueError("AssistantModel has no llm_client and no model_pool")
        return self.llm_client


class UserFeedbackModel:
    """
    Simulates a user providing feedback based on their persona.
    """
    
    def __init__(self, llm_client: LLMClient, template_path: str,
                 feedback_probability: float = 0.6):
        """
        Initialize UserFeedbackModel.
        
        Args:
            llm_client: LLM client for generating feedback
            template_path: Path to feedback template
            feedback_probability: Probability of providing follow-up feedback
        """
        self.llm_client = llm_client
        self.template_path = template_path
        self.feedback_probability = feedback_probability
        self.template = self._load_template()
        
        # Get model info for token tracking
        self.model_name = getattr(llm_client, 'model', 'unknown')
        client_class = type(llm_client).__name__
        if 'OpenAI' in client_class:
            self.provider_name = 'openai'
        elif 'Anthropic' in client_class:
            self.provider_name = 'anthropic'
        else:
            self.provider_name = 'unknown'
    
    def _load_template(self) -> str:
        """Load the user feedback template."""
        with open(self.template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _format_persona(self, persona_features: Dict[str, str]) -> str:
        """Format persona features for prompt."""
        lines = []
        for dimension, value in persona_features.items():
            dimension_formatted = dimension.replace('_', ' ').title()
            lines.append(f"- {dimension_formatted}: {value}")
        return '\n'.join(lines)
    
    def _format_conversation_history(self, messages: List[Message]) -> str:
        """Format conversation history."""
        history = []
        for msg in messages:
            role = "User" if msg.role == 'user' else "Assistant"
            history.append(f"{role}: {msg.content}")
        return '\n'.join(history)
    
    def generate_feedback(self, persona_features: Dict[str, str],
                         conversation_history: List[Message],
                         assistant_response: str,
                         current_turn: int = 1,
                         target_turns: int = 3) -> Optional[str]:
        """
        Generate user feedback if needed.

        Args:
            persona_features: User persona features
            conversation_history: Previous conversation
            assistant_response: Latest assistant response
            current_turn: Current conversation turn number
            target_turns: Target number of turns for this conversation

        Returns:
            Feedback text or None if no follow-up
        """
        # No longer using random gating - let LLM decide satisfaction based on persona and turn progress
        try:
            # Format persona and conversation
            persona_text = self._format_persona(persona_features)
            history_text = self._format_conversation_history(conversation_history)

            # Fill template
            prompt = self.template.replace('{persona}', persona_text)
            prompt = prompt.replace('{conversation_history}', history_text)
            prompt = prompt.replace('{assistant_response}', assistant_response)
            prompt = prompt.replace('{current_turn}', str(current_turn))
            prompt = prompt.replace('{target_turns}', str(target_turns))
            
            # Generate feedback with token tracking
            response, input_tokens, output_tokens = self.llm_client.generate_with_tokens(prompt)
            
            # Record token usage
            record_tokens(
                module='interaction_generation',
                operation='user_feedback',
                model=self.model_name,
                provider=self.provider_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={'current_turn': current_turn, 'target_turns': target_turns}
            )

            # Parse JSON response
            try:
                # Clean response (remove markdown code blocks if present)
                clean_response = response.strip()
                if clean_response.startswith('```'):
                    clean_response = clean_response.split('\n', 1)[1]
                if clean_response.endswith('```'):
                    clean_response = clean_response.rsplit('```', 1)[0]
                clean_response = clean_response.strip()

                parsed = json.loads(clean_response)

                if parsed.get('satisfied', False):
                    return None

                feedback = parsed.get('feedback')
                if feedback and len(str(feedback).strip()) > 5:
                    return str(feedback)
                return None

            except json.JSONDecodeError:
                # Fallback: if not valid JSON, treat as raw feedback
                logging.warning(f"Failed to parse user feedback as JSON: {response[:100]}")
                if len(response.strip()) < 10:
                    return None
                return response
        
        except Exception as e:
            logging.error(f"Error generating user feedback: {str(e)}")
            return None  # No feedback on error


class InteractionGenerator:
    """
    Main class for generating multi-turn interactions with integrated distractor.
    """
    
    def __init__(self, config: Dict[str, Any],
                 distractor: Optional[Union[DistractorModel, SemanticDistractorModel]] = None):
        """
        Initialize InteractionGenerator.

        Args:
            config: Configuration dictionary
            distractor: Optional distractor model (DistractorModel or SemanticDistractorModel)
        """
        self.config = config
        self.interaction_config = config['interaction_generation']
        self.distractor = distractor  # Distractor for real-time noise injection
        
        # Initialize assistant model (supports either single model or weighted model pool)
        assistant_config = self.interaction_config['assistant_model']
        assistant_model_pool = assistant_config.get('model_pool')

        if assistant_model_pool:
            pool_entries: List[Dict[str, Any]] = []
            for entry in assistant_model_pool:
                provider = entry.get('provider', 'openai')
                model = entry.get('model', '')
                weight = float(entry.get('weight', 1.0))

                # Resolve API keys per provider (entry overrides top-level defaults)
                api_key = entry.get('api_key')
                if not api_key:
                    if provider == 'openrouter':
                        api_key = config.get('openrouter', {}).get('api_key', '')
                    else:
                        api_key = config.get('api', {}).get('api_key', '')

                # Resolve endpoint/base_url only for openrouter provider
                endpoint = entry.get('endpoint')
                base_url = entry.get('base_url')
                if provider == 'openrouter':
                    endpoint = endpoint or config.get('openrouter', {}).get('endpoint')
                    base_url = base_url or config.get('openrouter', {}).get('base_url')

                inference = entry.get('inference', {})
                client_cfg = {
                    'provider': provider,
                    'api_key': api_key,
                    'model': model,
                    'inference': inference,
                }
                if endpoint:
                    client_cfg['endpoint'] = endpoint
                if base_url:
                    client_cfg['base_url'] = base_url
                if entry.get('headers') or entry.get('default_headers'):
                    client_cfg['headers'] = entry.get('headers') or entry.get('default_headers')

                pool_entries.append({
                    'provider': provider,
                    'model': model,
                    'weight': weight,
                    'client': create_llm_client(client_cfg)
                })

            self.assistant_model = AssistantModel(
                llm_client=None,
                model_pool=pool_entries
            )
        else:
            # Legacy single-model config (backward compatible)
            # Support both max_tokens and max_completion_tokens
            assistant_max_tokens = assistant_config.get('max_completion_tokens') or assistant_config.get('max_tokens', 1024)
            
            assistant_client = create_llm_client({
                'provider': assistant_config['provider'],
                'api_key': assistant_config.get('api_key', config['api'].get('api_key', '')),
                'model': assistant_config['model'],
                'inference': {
                    'temperature': assistant_config['temperature'],
                    'max_completion_tokens': assistant_max_tokens
                }
            })
            self.assistant_model = AssistantModel(assistant_client)
        
        # Initialize user feedback model
        user_config = self.interaction_config['user_model']
        # Support both max_tokens and max_completion_tokens
        user_max_tokens = user_config.get('max_completion_tokens') or user_config.get('max_tokens', 256)
        
        user_client = create_llm_client({
            'provider': user_config['provider'],
            'api_key': user_config.get('api_key', config['api'].get('api_key', '')),
            'model': user_config['model'],
            'inference': {
                'temperature': user_config['temperature'],
                'max_completion_tokens': user_max_tokens
            }
        })
        self.user_model = UserFeedbackModel(
            user_client,
            user_config['system_prompt_template'],
            user_config['feedback_probability']
        )
        
        self.max_turns = self.interaction_config['max_turns']
        self.min_turns = self.interaction_config['min_turns']
    
    def generate_interaction(self, persona_id: str, persona_features: Dict[str, str],
                           initial_query: str, system_prompt: Optional[str] = None,
                           original_query: Optional[str] = None,
                           target_turns: int = 3,
                           query_id: Optional[str] = None) -> Optional[Interaction]:
        """
        Generate a complete multi-turn interaction with distractor applied in real-time.

        Args:
            persona_id: Persona identifier
            persona_features: User persona features
            initial_query: Initial user query (after style transfer)
            system_prompt: Optional personalized system prompt
            original_query: Original query from dataset (before style transfer)
            target_turns: Target number of conversation turns (3-4)

        Returns:
            Complete Interaction object with noisy versions
        """
        # If original_query not provided, use initial_query as original
        if original_query is None:
            original_query = initial_query

        # Note: We no longer pass persona system_prompt to assistant to avoid
        # role confusion (e.g., assistant thinking it's a "food planning assistant").
        # Assistant always uses the default generic prompt: "You are a helpful AI assistant."

        # Lock model for this conversation (ensures same model throughout all turns)
        self.assistant_model.lock_model()

        # Apply distractor to initial query
        current_query_clean = initial_query
        noisy_initial_queries = []
        semantic_noise_data = None  # For semantic distractor

        if self.distractor and self.distractor.enabled:
            # Check if using semantic distractor or legacy distractor
            if isinstance(self.distractor, SemanticDistractorModel):
                # Semantic distractor returns a single NoiseResult (persona-aware)
                noise_result = self.distractor.apply_noise(initial_query, persona_features=persona_features)
                noisy_initial_queries = [{
                    'noisy_text': noise_result.noisy_text,
                    'noise_type': noise_result.layer,
                    'noise_intensity': noise_result.layer_index,
                    'applied_strategies': noise_result.applied_strategies
                }]
                semantic_noise_data = {
                    'initial_query': noise_result.to_dict()
                }
                current_query = noise_result.noisy_text
                logging.debug(f"Using semantic noisy initial query (layer {noise_result.layer}): {current_query}")
            else:
                # Legacy distractor returns list of NoisyVersion
                noisy_versions = self.distractor.apply_noise(initial_query)
                noisy_initial_queries = [
                    {
                        'noisy_text': nv.noisy_text,
                        'noise_type': nv.noise_type,
                        'noise_intensity': nv.noise_intensity
                    }
                    for nv in noisy_versions
                ]
                # Use one of the noisy versions as the actual query
                if noisy_versions:
                    current_query = random.choice(noisy_versions).noisy_text
                    logging.debug(f"Using noisy initial query: {current_query}")
                else:
                    current_query = current_query_clean
        else:
            current_query = current_query_clean
        
        # Initialize conversation
        messages = []
        turn = 0
        noisy_followups = []  # Track noisy followup versions
        
        # Add initial user query (using noisy version)
        messages.append(Message(
            role='user',
            content=current_query,
            timestamp=datetime.now().isoformat(),
            metadata={
                'turn': turn,
                'is_initial': True,
                'clean_version': current_query_clean,
                'was_noised': current_query != current_query_clean
            }
        ))
        
        # Multi-turn dialogue
        while turn < self.max_turns:
            # Assistant responds (to the noisy query)
            conversation_so_far = messages[:-1]  # Exclude current query
            assistant_response = self.assistant_model.respond(
                current_query,
                conversation_so_far
            )
            
            # Check if assistant response is empty
            if not assistant_response or not assistant_response.strip():
                print_colored(
                    f"⚠️  Assistant returned empty response for persona {persona_id} at turn {turn}. "
                    f"Model: {(self.assistant_model.last_model_info or {}).get('model')}, "
                    f"Provider: {(self.assistant_model.last_model_info or {}).get('provider')}. "
                    f"Discarding this interaction.",
                    'WARNING'
                )
                return None  # Discard this interaction
            
            messages.append(Message(
                role='assistant',
                content=assistant_response,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'turn': turn,
                    'model': (self.assistant_model.last_model_info or {}).get('model'),
                    'provider': (self.assistant_model.last_model_info or {}).get('provider'),
                }
            ))
            
            turn += 1
            
            # Check if minimum turns reached
            if turn < self.min_turns:
                # Force continuation for minimum turns
                feedback_clean = self.user_model.generate_feedback(
                    persona_features,
                    messages,
                    assistant_response,
                    current_turn=turn,
                    target_turns=target_turns
                )
                if feedback_clean:
                    # Apply distractor to follow-up feedback
                    feedback_noisy_versions = []
                    if self.distractor and self.distractor.enabled:
                        if isinstance(self.distractor, SemanticDistractorModel):
                            # Semantic distractor returns a single NoiseResult (persona-aware)
                            noise_result = self.distractor.apply_noise(feedback_clean, persona_features=persona_features)
                            feedback_noisy_versions = [{
                                'noisy_text': noise_result.noisy_text,
                                'noise_type': noise_result.layer,
                                'noise_intensity': noise_result.layer_index,
                                'applied_strategies': noise_result.applied_strategies
                            }]
                            current_query = noise_result.noisy_text
                            noisy_followups.append({
                                'clean_feedback': feedback_clean,
                                'noisy_versions': feedback_noisy_versions,
                                'used_version': current_query,
                                'turn': turn,
                                'semantic_noise': noise_result.to_dict()
                            })
                        else:
                            # Legacy distractor returns list of NoisyVersion
                            noisy_versions = self.distractor.apply_noise(feedback_clean)
                            feedback_noisy_versions = [
                                {
                                    'noisy_text': nv.noisy_text,
                                    'noise_type': nv.noise_type,
                                    'noise_intensity': nv.noise_intensity
                                }
                                for nv in noisy_versions
                            ]
                            # Use one of the noisy versions
                            if noisy_versions:
                                current_query = random.choice(noisy_versions).noisy_text
                                noisy_followups.append({
                                    'clean_feedback': feedback_clean,
                                    'noisy_versions': feedback_noisy_versions,
                                    'used_version': current_query,
                                    'turn': turn
                                })
                            else:
                                current_query = feedback_clean
                    else:
                        current_query = feedback_clean
                    
                    messages.append(Message(
                        role='user',
                        content=current_query,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            'turn': turn,
                            'is_followup': True,
                            'clean_version': feedback_clean,
                            'was_noised': current_query != feedback_clean
                        }
                    ))
                else:
                    # No more feedback, end conversation
                    break
            else:
                # User decides whether to continue
                feedback_clean = self.user_model.generate_feedback(
                    persona_features,
                    messages,
                    assistant_response,
                    current_turn=turn,
                    target_turns=target_turns
                )
                
                if feedback_clean is None:
                    # No feedback, conversation ends
                    break
                else:
                    # Apply distractor to follow-up feedback
                    feedback_noisy_versions = []
                    if self.distractor and self.distractor.enabled:
                        if isinstance(self.distractor, SemanticDistractorModel):
                            # Semantic distractor returns a single NoiseResult (persona-aware)
                            noise_result = self.distractor.apply_noise(feedback_clean, persona_features=persona_features)
                            feedback_noisy_versions = [{
                                'noisy_text': noise_result.noisy_text,
                                'noise_type': noise_result.layer,
                                'noise_intensity': noise_result.layer_index,
                                'applied_strategies': noise_result.applied_strategies
                            }]
                            current_query = noise_result.noisy_text
                            noisy_followups.append({
                                'clean_feedback': feedback_clean,
                                'noisy_versions': feedback_noisy_versions,
                                'used_version': current_query,
                                'turn': turn,
                                'semantic_noise': noise_result.to_dict()
                            })
                        else:
                            # Legacy distractor returns list of NoisyVersion
                            noisy_versions = self.distractor.apply_noise(feedback_clean)
                            feedback_noisy_versions = [
                                {
                                    'noisy_text': nv.noisy_text,
                                    'noise_type': nv.noise_type,
                                    'noise_intensity': nv.noise_intensity
                                }
                                for nv in noisy_versions
                            ]
                            # Use one of the noisy versions
                            if noisy_versions:
                                current_query = random.choice(noisy_versions).noisy_text
                                noisy_followups.append({
                                    'clean_feedback': feedback_clean,
                                    'noisy_versions': feedback_noisy_versions,
                                    'used_version': current_query,
                                    'turn': turn
                                })
                            else:
                                current_query = feedback_clean
                    else:
                        current_query = feedback_clean
                    
                    # User has follow-up (using noisy version)
                    messages.append(Message(
                        role='user',
                        content=current_query,
                        timestamp=datetime.now().isoformat(),
                        metadata={
                            'turn': turn,
                            'is_followup': True,
                            'clean_version': feedback_clean,
                            'was_noised': current_query != feedback_clean
                        }
                    ))
        
        # Create interaction object with integrated noisy versions
        interaction_id = f"interaction_{persona_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Build noisy_versions structure
        noisy_versions_data = {}
        if noisy_initial_queries:
            noisy_versions_data['initial_query'] = {
                'clean_query': current_query_clean,
                'noisy_versions': noisy_initial_queries
            }
        if noisy_followups:
            noisy_versions_data['followups'] = noisy_followups
        
        interaction = Interaction(
            interaction_id=interaction_id,
            persona_id=persona_id,
            persona_features=persona_features,
            original_query=original_query,  # Original query from dataset
            initial_query=current_query_clean,  # Query after style transfer
            messages=messages,
            num_turns=turn,
            metadata={
                'created_at': datetime.now().isoformat(),
                'query_id': query_id,
                'system_prompt_used': system_prompt is not None,
                'total_messages': len(messages),
                'conversation_ended': turn < self.max_turns,
                'distractor_applied': self.distractor is not None and self.distractor.enabled,
                'distractor_type': 'semantic' if isinstance(self.distractor, SemanticDistractorModel) else 'legacy' if self.distractor else None,
                'noisy_versions': noisy_versions_data if noisy_versions_data else None,
                'semantic_noise': semantic_noise_data if semantic_noise_data else None,
                'assistant_model': self.assistant_model.last_model_info,  # Record which model was used
            }
        )

        # Unlock model after conversation ends
        self.assistant_model.unlock_model()

        return interaction
    
    def _generate_single_interaction(self, task: Dict[str, Any]) -> Optional[Interaction]:
        """
        Helper method to generate a single interaction (for concurrent execution).
        
        Args:
            task: Dictionary containing all parameters for generate_interaction
            
        Returns:
            Interaction object or None if discarded
        """
        return self.generate_interaction(
            persona_id=task['persona_id'],
            persona_features=task['persona_features'],
            initial_query=task['adapted_query'],
            system_prompt=task.get('system_prompt'),
            original_query=task['original_query'],
            target_turns=task['target_turns'],
            query_id=task.get('query_id')
        )
    
    def _generate_with_retry(self, task: Dict[str, Any], max_retries: int = 3, 
                            retry_delay: int = 2) -> Optional[Interaction]:
        """
        Generate a single interaction with retry logic.
        
        Args:
            task: Task dictionary with persona and query info
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            Interaction object or None if all retries failed
        """
        import time

        persona_index = task.get('persona_index')
        query_index = task.get('query_index')
        if persona_index is not None and query_index is not None:
            print_colored(f"Processing {persona_index}-{query_index}", 'INFO')

        last_error = None
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                interaction = self._generate_single_interaction(task)
                
                # Check if interaction is valid (not None and not empty)
                if interaction is not None:
                    if attempt > 0:
                        logging.info(
                            f"✓ Successfully generated interaction for persona {task['persona_id']} "
                            f"on attempt {attempt + 1}/{max_retries + 1}"
                        )
                    return interaction
                else:
                    last_error = "Empty or discarded interaction"
                    if attempt < max_retries:
                        logging.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for persona {task['persona_id']}: "
                            f"{last_error}. Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                    
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for persona {task['persona_id']}: "
                        f"{last_error}. Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logging.error(
                        f"All {max_retries + 1} attempts failed for persona {task['persona_id']}: {last_error}"
                    )
        
        # All retries exhausted
        return None
    
    def generate_interactions_batch(self, personas_with_queries: List[Dict[str, Any]],
                                   show_progress: bool = True,
                                   max_workers: int = 5,
                                   storage: Optional[InteractionStorage] = None) -> List[Interaction]:
        """
        Generate interactions for multiple personas with their queries (with concurrent support and retry logic).

        Args:
            personas_with_queries: List of dicts with persona and query info
            show_progress: Whether to show progress bar
            max_workers: Maximum number of concurrent workers (default: 5)
            storage: Optional InteractionStorage for saving each interaction immediately

        Returns:
            List of Interaction objects
        """
        all_interactions = []
        
        # Get retry configuration
        max_retries = self.interaction_config.get('max_retries', 3)
        retry_delay = self.interaction_config.get('retry_delay', 2)
        
        # Prepare all tasks (persona + query combinations)
        tasks = []
        persona_order = {}
        persona_query_counts = {}
        for item in personas_with_queries:
            persona_id = item['persona_id']
            persona_features = item['persona_features']
            system_prompt = item.get('system_prompt')
            queries = item.get('queries', [])
            if persona_id not in persona_order:
                persona_order[persona_id] = len(persona_order) + 1
            
            for query_data in queries:
                adapted_query = query_data.get('adapted_query', query_data.get('original_query'))
                original_query = query_data.get('original_query', adapted_query)
                target_turns = query_data.get('target_turns', 3)
                query_id = query_data.get('query_id')
                persona_query_counts[persona_id] = persona_query_counts.get(persona_id, 0) + 1
                
                tasks.append({
                    'persona_id': persona_id,
                    'persona_features': persona_features,
                    'adapted_query': adapted_query,
                    'system_prompt': system_prompt,
                    'original_query': original_query,
                    'target_turns': target_turns,
                    'query_id': query_id,
                    'persona_index': persona_order[persona_id],
                    'query_index': persona_query_counts[persona_id],
                })
        
        # Use ThreadPoolExecutor for concurrent generation
        total_tasks = len(tasks)
        retry_stats = {'total_retries': 0, 'failed_tasks': 0}
        
        print_colored(
            f"Generating {total_tasks} interactions with {max_workers} workers "
            f"(max {max_retries} retries per task)...", 
            'GENERATION'
        )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with retry logic
            future_to_task = {
                executor.submit(
                    self._generate_with_retry,
                    task,
                    max_retries,
                    retry_delay
                ): task for task in tasks
            }
            
            # Process completed tasks with progress bar
            if show_progress:
                pbar = tqdm(total=total_tasks, desc="Generating interactions", 
                           unit="interaction", ncols=100, colour='cyan')
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    interaction = future.result()
                    
                    if interaction is not None:
                        all_interactions.append(interaction)
                        # Save immediately if storage is provided
                        if storage is not None:
                            try:
                                storage.save(interaction)
                            except Exception as save_err:
                                logging.error(
                                    f"Failed to save interaction {interaction.interaction_id}: {save_err}"
                                )
                    else:
                        retry_stats['failed_tasks'] += 1
                        print_colored(
                            f"✗ Failed to generate interaction for persona {task['persona_id']} "
                            f"after {max_retries + 1} attempts",
                            'ERROR'
                        )
                    
                except Exception as e:
                    retry_stats['failed_tasks'] += 1
                    logging.error(
                        f"Unexpected error for persona {task['persona_id']}: {str(e)}"
                    )
                
                if show_progress:
                    pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        # Log statistics
        avg_turns = sum(inter.num_turns for inter in all_interactions) / len(all_interactions) if all_interactions else 0
        success_rate = (len(all_interactions) / total_tasks * 100) if total_tasks > 0 else 0
        
        logging.info(
            f"Generated {len(all_interactions)}/{total_tasks} interactions "
            f"(success rate: {success_rate:.1f}%, avg {avg_turns:.1f} turns per interaction)"
        )
        
        if retry_stats['failed_tasks'] > 0:
            logging.warning(
                f"{retry_stats['failed_tasks']} interactions failed after all retry attempts"
            )
        
        return all_interactions
