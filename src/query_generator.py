"""
Query Generator Module

This module handles loading user queries from datasets and adapting them
to match specific user personas through style transfer.
"""

import json
import re
import random
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Iterable
from pathlib import Path
import logging
from tqdm import tqdm

from .llm_client import LLMClient
from .colored_logger import print_colored

# Import token tracker
try:
    from .token_tracker import record_tokens
except ImportError:
    def record_tokens(*args, **kwargs):
        pass


# Length categories: index 0=very_short, 1=short, 2=medium, 3=long, 4=very_long
LENGTH_CATEGORIES = ['very_short', 'short', 'medium', 'long', 'very_long']


def sample_length_category(mean: float = 1.5, std: float = 1.0) -> str:
    """
    Sample a length category from a normal distribution.
    Default parameters favor short-to-medium lengths (mean=1.5 between 'short' and 'medium').

    Args:
        mean: Mean of the distribution (0=very_short, 1=short, 2=medium, 3=long, 4=very_long)
        std: Standard deviation (controls spread)

    Returns:
        Length category string
    """
    # Sample from normal distribution
    value = np.random.normal(mean, std)
    # Clip to valid range and round to nearest category index
    index = int(np.clip(np.round(value), 0, len(LENGTH_CATEGORIES) - 1))
    return LENGTH_CATEGORIES[index]


def sample_target_turns(mean: float = 3.0, std: float = 0.5, min_turns: int = 2, max_turns: int = 5) -> int:
    """
    Sample target conversation turns from a normal distribution.
    Default parameters favor 3 turns (mean=3.0).

    Args:
        mean: Mean of the distribution (default 3.0)
        std: Standard deviation (default 0.5, so ~68% will be 3, ~27% will be 2 or 4)
        min_turns: Minimum allowed turns (default 2)
        max_turns: Maximum allowed turns (default 5)

    Returns:
        Target number of conversation turns
    """
    # Sample from normal distribution
    value = np.random.normal(mean, std)
    # Clip to valid range and round to nearest integer
    turns = int(np.clip(np.round(value), min_turns, max_turns))
    return turns


# Valid domain and scenario values from persona.yaml
VALID_DOMAINS = [
    'general', 'writing', 'coding', 'data', 'product', 'design', 'marketing',
    'sales', 'customer_support', 'bizdev', 'ops', 'career', 'education',
    'finance', 'legal', 'health', 'travel', 'shopping', 'food', 'communication',
    'productivity', 'life_admin', 'research', 'news', 'entertainment', 'others'
]

VALID_SCENARIOS = [
    'in_a_rush', 'work_task', 'learning', 'decision_making', 'troubleshooting',
    'brainstorming', 'planning', 'casual', 'others'
]


class QueryDataset:
    """
    Manages loading and accessing user queries from various datasets.
    Ensures each query is only used once across all personas (no duplicates).
    """

    def __init__(self, dataset_paths: List[str], max_queries: Optional[int] = None):
        """
        Initialize QueryDataset.

        Args:
            dataset_paths: List of paths to query dataset files
            max_queries: Maximum number of queries to load (None = load all)
        """
        self.dataset_paths = dataset_paths
        self.max_queries = max_queries
        self.queries = []
        self._current_index = 0  # For round-robin distribution
        self._usage_count = {}  # Track how many times each query is used
        self._used_query_ids = set()  # Track which queries have been used (for no-duplicate mode)
        self._load_datasets()
    
    def _derive_query_id(self, query_text: str, metadata: Dict[str, Any], source_name: str, index: int) -> str:
        """Derive a stable query_id for a dataset entry."""
        metadata = metadata or {}
        for key in ['query_id', 'id', 'source_id', 'uid', 'uuid', 'qid']:
            val = metadata.get(key)
            if val:
                dataset = metadata.get('dataset') or metadata.get('source') or source_name
                return f"{dataset}:{val}"

        base = f"{source_name}:{index}:{query_text}"
        digest = hashlib.sha1(base.encode('utf-8')).hexdigest()[:16]
        return f"hash:{digest}"

    def _append_query(self, query_text: str, source_name: str, metadata: Dict[str, Any]):
        """Append a query entry with a stable query_id."""
        index = len(self.queries)
        query_id = self._derive_query_id(query_text, metadata, source_name, index)
        self.queries.append({
            'query_id': query_id,
            'text': query_text,
            'source': source_name,
            'metadata': metadata
        })

    def _load_datasets(self):
        """
        Load queries from all dataset files.
        Supports both JSON and JSONL formats.
        Respects max_queries limit.
        """
        for path in self.dataset_paths:
            # Check if we've reached max_queries limit
            if self.max_queries and len(self.queries) >= self.max_queries:
                break

            path_obj = Path(path)
            if not path_obj.exists():
                logging.warning(f"Dataset file not found: {path}")
                continue

            try:
                # Check if it's a JSONL file
                if path_obj.suffix.lower() == '.jsonl':
                    self._load_jsonl_file(path_obj)
                else:
                    self._load_json_file(path_obj)

                logging.info(f"Loaded {len(self.queries)} queries from {path}")

            except Exception as e:
                logging.error(f"Error loading dataset {path}: {str(e)}")

        # Trim to max_queries if exceeded
        if self.max_queries and len(self.queries) > self.max_queries:
            self.queries = self.queries[:self.max_queries]
            logging.info(f"Trimmed queries to max_queries limit: {self.max_queries}")

    def _load_jsonl_file(self, path_obj: Path):
        """Load queries from a JSONL file (one JSON object per line)."""
        with open(path_obj, 'r', encoding='utf-8') as f:
            for line in f:
                # Check max_queries limit
                if self.max_queries and len(self.queries) >= self.max_queries:
                    break

                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    query = self._extract_query(item)
                    if query:
                        self._append_query(query, path_obj.name, item)
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line in {path_obj}: {e}")

    def _load_json_file(self, path_obj: Path):
        """Load queries from a JSON file."""
        with open(path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different dataset formats
        if isinstance(data, list):
            for item in data:
                query = self._extract_query(item)
                if query:
                    metadata = item if isinstance(item, dict) else {}
                    self._append_query(query, path_obj.name, metadata)
        elif isinstance(data, dict):
            if 'queries' in data:
                for query in data['queries']:
                    if isinstance(query, str):
                        self._append_query(query, path_obj.name, {})
                    else:
                        text = self._extract_query(query)
                        if text:
                            metadata = query if isinstance(query, dict) else {}
                            self._append_query(text, path_obj.name, metadata)
    
    def _extract_query(self, item: Any) -> Optional[str]:
        """
        Extract query text from various dataset formats.
        
        Args:
            item: Dataset item (can be dict, string, etc.)
            
        Returns:
            Query text or None
        """
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            # Try common field names
            for field in ['instruction', 'query', 'question', 'prompt', 'text', 'input']:
                if field in item and item[field]:
                    return item[field]
            # For conversation format (like ShareGPT)
            if 'conversations' in item and len(item['conversations']) > 0:
                first_msg = item['conversations'][0]
                if isinstance(first_msg, dict) and 'value' in first_msg:
                    return first_msg['value']
        return None
    
    def sample_queries(self, n: int, strategy: str = "random") -> List[Dict[str, Any]]:
        """
        Sample n queries from the dataset. Each query can only be used once globally.

        Args:
            n: Number of queries to sample
            strategy: Sampling strategy ("random", "round_robin")

        Returns:
            List of sampled queries (no duplicates globally)
        """
        if not self.queries:
            logging.warning("No queries available in dataset")
            return []

        # Calculate available queries (not yet used)
        available_indices = [i for i, q in enumerate(self.queries) if q.get('query_id') not in self._used_query_ids]
        available_count = len(available_indices)

        if available_count == 0:
            logging.warning("All queries have been used. No more queries available.")
            return []

        if n > available_count:
            logging.warning(
                f"Requested {n} queries but only {available_count} available (unused). "
                f"Returning all available queries."
            )
            n = available_count

        if strategy == "round_robin":
            return self._sample_round_robin_no_duplicate(n)
        else:
            return self._sample_random_no_duplicate(n)

    def _sample_random_no_duplicate(self, n: int) -> List[Dict[str, Any]]:
        """
        Sample queries randomly without duplicates (globally).

        Args:
            n: Number of queries to sample

        Returns:
            List of sampled queries
        """
        # Get indices that haven't been used
        available_indices = [i for i, q in enumerate(self.queries) if q.get('query_id') not in self._used_query_ids]

        # Sample from available indices
        sampled_indices = random.sample(available_indices, n)

        sampled = []
        for idx in sampled_indices:
            sampled.append(self.queries[idx])
            qid = self.queries[idx].get('query_id')
            if qid:
                self._used_query_ids.add(qid)
            self._usage_count[idx] = self._usage_count.get(idx, 0) + 1

        return sampled

    def _sample_round_robin_no_duplicate(self, n: int) -> List[Dict[str, Any]]:
        """
        Sample queries using round-robin without duplicates (globally).

        Args:
            n: Number of queries to sample

        Returns:
            List of sampled queries
        """
        sampled = []

        for _ in range(n):
            # Find next unused query
            attempts = 0
            while self.queries[self._current_index].get('query_id') in self._used_query_ids:
                self._current_index = (self._current_index + 1) % len(self.queries)
                attempts += 1
                if attempts >= len(self.queries):
                    # All queries used
                    logging.warning("Round-robin: all queries have been used")
                    return sampled

            query = self.queries[self._current_index]
            sampled.append(query)

            # Mark as used
            qid = query.get('query_id')
            if qid:
                self._used_query_ids.add(qid)
            self._usage_count[self._current_index] = self._usage_count.get(self._current_index, 0) + 1

            # Move to next query
            self._current_index = (self._current_index + 1) % len(self.queries)

        return sampled

    def get_available_count(self) -> int:
        """Get number of queries that haven't been used yet."""
        return len([q for q in self.queries if q.get('query_id') not in self._used_query_ids])

    def reset_used(self):
        """Reset used queries tracking, allowing all queries to be sampled again."""
        self._used_query_ids.clear()
        self._current_index = 0
        logging.info(f"Reset query usage. {len(self.queries)} queries available.")

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get query usage statistics.

        Returns:
            Dictionary with usage stats
        """
        if not self._usage_count:
            return {'total_queries': len(self.queries), 'used': 0, 'unused': len(self.queries)}

        used = len(self._usage_count)
        counts = list(self._usage_count.values())
        return {
            'total_queries': len(self.queries),
            'used': used,
            'unused': len([q for q in self.queries if q.get('query_id') not in self._used_query_ids]),
            'min_usage': min(counts),
            'max_usage': max(counts),
            'avg_usage': sum(counts) / len(counts)
        }

    def reset_round_robin(self):
        """Reset round-robin index and clear used tracking."""
        self._current_index = 0
        self._used_query_ids.clear()

    def shuffle_queries(self):
        """Shuffle queries to randomize order. Also resets usage tracking."""
        random.shuffle(self.queries)
        self._current_index = 0
        self._used_query_ids.clear()

    def mark_used_query_ids(self, query_ids: Iterable[str]):
        """Mark a collection of query_ids as already used."""
        for qid in query_ids:
            if qid:
                self._used_query_ids.add(qid)
    
    def get_all_queries(self) -> List[Dict[str, Any]]:
        """
        Get all queries.
        
        Returns:
            List of all queries
        """
        return self.queries
    
    def __len__(self) -> int:
        return len(self.queries)


class QueryStyleTransfer:
    """
    Handles style transfer of queries to match user personas.
    Returns adapted query along with inferred domain and scenario.
    Supports two prompt strategies with random selection:
    - Full persona: Uses all persona dimensions with detailed guidance
    - Partial persona: LLM selects only relevant dimensions
    """

    def __init__(self, llm_client: LLMClient, template_path: str,
                 transfer_probability: float = 0.5,
                 partial_template_path: Optional[str] = None):
        """
        Initialize QueryStyleTransfer.

        Args:
            llm_client: LLM client for style transfer
            template_path: Path to full style transfer template
            transfer_probability: Probability of applying style transfer (0-1)
            partial_template_path: Path to partial persona style transfer template (optional)
        """
        self.llm_client = llm_client
        self.template_path = template_path
        self.partial_template_path = partial_template_path
        self.transfer_probability = transfer_probability
        self.template = self._load_template(template_path)
        self.partial_template = self._load_template(partial_template_path) if partial_template_path else None
        
        # Get model info for token tracking
        self.model_name = getattr(llm_client, 'model', 'unknown')
        # Infer provider from client type
        client_class = type(llm_client).__name__
        if 'OpenAI' in client_class:
            self.provider_name = 'openai'
        elif 'Anthropic' in client_class:
            self.provider_name = 'anthropic'
        else:
            self.provider_name = 'unknown'

    def _load_template(self, path: str) -> str:
        """
        Load a style transfer template.

        Args:
            path: Path to the template file

        Returns:
            Template string
        """
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _format_persona(self, persona_features: Dict[str, str]) -> str:
        """
        Format persona features into readable text.

        Args:
            persona_features: Dictionary of persona features

        Returns:
            Formatted persona description
        """
        lines = []
        for dimension, value in persona_features.items():
            dimension_formatted = dimension.replace('_', ' ').title()
            lines.append(f"- {dimension_formatted}: {value}")
        return '\n'.join(lines)

    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse the JSON response from LLM.

        Args:
            response: Raw LLM response

        Returns:
            Parsed dictionary with adapted_query, inferred_domain, inferred_scenario
        """
        try:
            # Try to parse as JSON directly
            result = json.loads(response.strip())
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from response (in case of markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except json.JSONDecodeError:
                    pass

            # Fallback: return the response as adapted_query with default domain/scenario
            logging.warning(f"Failed to parse JSON response, using fallback")
            return {
                'adapted_query': response.strip(),
                'inferred_domain': 'general',
                'inferred_scenario': 'casual'
            }

    def _validate_domain(self, domain: str) -> str:
        """Validate and normalize domain value."""
        domain = domain.lower().strip()
        if domain in VALID_DOMAINS:
            return domain
        return 'others'

    def _validate_scenario(self, scenario: str) -> str:
        """Validate and normalize scenario value."""
        scenario = scenario.lower().strip()
        if scenario in VALID_SCENARIOS:
            return scenario
        return 'others'

    def transfer(self, query: str, persona_features: Dict[str, str],
                force: bool = False, target_length: Optional[str] = None,
                target_turns: int = 3) -> Tuple[str, bool, str, str, str, int, str]:
        """
        Apply style transfer to a query based on persona.

        Args:
            query: Original query text
            persona_features: User persona features
            force: Force style transfer (ignore probability)
            target_length: Target length category (if None, will sample from distribution)
            target_turns: Target number of conversation turns (3-4)

        Returns:
            Tuple of (transferred_query, was_transferred, inferred_domain, inferred_scenario,
                      target_length, target_turns, prompt_type)
            prompt_type: "full" or "partial" indicating which prompt was used
        """
        # Sample length if not provided
        if target_length is None:
            target_length = sample_length_category()

        # Randomly decide whether to apply style transfer
        if not force and random.random() > self.transfer_probability:
            return query, False, 'general', 'casual', target_length, target_turns, 'none'

        try:
            # Format persona
            persona_text = self._format_persona(persona_features)

            # Randomly select which template to use (50/50 if partial template exists)
            if self.partial_template and random.random() < 0.5:
                template = self.partial_template
                prompt_type = 'partial'
            else:
                template = self.template
                prompt_type = 'full'

            # Fill template
            prompt = template.replace('{persona}', persona_text)
            prompt = prompt.replace('{query}', query)
            prompt = prompt.replace('{target_length}', target_length)
            prompt = prompt.replace('{target_turns}', str(target_turns))

            # Generate style-transferred query with domain/scenario and track tokens
            response, input_tokens, output_tokens = self.llm_client.generate_with_tokens(prompt)
            
            # Record token usage
            record_tokens(
                module='query_generation',
                operation=f'style_transfer_{prompt_type}',
                model=self.model_name,
                provider=self.provider_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={'prompt_type': prompt_type, 'target_length': target_length}
            )

            # Parse the JSON response
            parsed = self._parse_response(response)

            adapted_query = parsed.get('adapted_query', query)
            inferred_domain = self._validate_domain(parsed.get('inferred_domain', 'general'))
            inferred_scenario = self._validate_scenario(parsed.get('inferred_scenario', 'casual'))

            return adapted_query, True, inferred_domain, inferred_scenario, target_length, target_turns, prompt_type

        except Exception as e:
            logging.error(f"Error in style transfer: {str(e)}")
            return query, False, 'general', 'casual', target_length, target_turns, 'none'

    def transfer_batch(self, queries: List[str], persona_features: Dict[str, str],
                      show_progress: bool = True, target_turns: int = 3) -> List[Tuple[str, bool, str, str, str, int, str]]:
        """
        Apply style transfer to multiple queries.

        Args:
            queries: List of query texts
            persona_features: User persona features
            show_progress: Whether to show progress
            target_turns: Target number of conversation turns (3-4)

        Returns:
            List of (transferred_query, was_transferred, inferred_domain, inferred_scenario,
                     target_length, target_turns, prompt_type) tuples
        """
        results = []

        for i, query in enumerate(queries):
            transferred, was_transferred, domain, scenario, length, turns, prompt_type = self.transfer(
                query, persona_features, target_turns=target_turns
            )
            results.append((transferred, was_transferred, domain, scenario, length, turns, prompt_type))

            if show_progress and (i + 1) % 10 == 0:
                logging.info(f"Style-transferred {i + 1}/{len(queries)} queries")

        return results


class UserQueryGenerator:
    """
    Main class for generating user queries based on personas.
    """

    def __init__(self, config: Dict[str, Any], llm_client: LLMClient):
        """
        Initialize UserQueryGenerator.

        Args:
            config: Configuration dictionary
            llm_client: LLM client instance
        """
        self.config = config
        self.llm_client = llm_client

        # Get query generation config
        query_gen_config = config['query_generation']

        # Load query dataset with max_queries limit
        dataset_paths = config['paths']['query_datasets']
        max_queries = query_gen_config.get('max_queries')
        # Treat 0 or null as "load all"
        if max_queries == 0:
            max_queries = None
        self.query_dataset = QueryDataset(dataset_paths, max_queries=max_queries)

        # Initialize style transfer
        if query_gen_config['style_transfer']['enabled']:
            partial_template = query_gen_config['style_transfer'].get('partial_template')
            self.style_transfer = QueryStyleTransfer(
                llm_client,
                query_gen_config['style_transfer']['template'],
                query_gen_config['style_transfer']['transfer_probability'],
                partial_template_path=partial_template
            )
        else:
            self.style_transfer = None

        self.queries_per_persona = query_gen_config['selection']['queries_per_persona']
        self.selection_strategy = query_gen_config['selection']['strategy']
    
    def generate_queries_for_persona(self, persona_features: Dict[str, str],
                                     num_queries: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate queries for a specific persona.

        Args:
            persona_features: User persona features
            num_queries: Optional override for number of queries to generate

        Returns:
            List of query dictionaries with metadata, including inferred_domain and inferred_scenario
        """
        # Use configured number of queries per persona
        num_queries = num_queries or self.queries_per_persona

        # Sample queries from dataset
        sampled_queries = self.query_dataset.sample_queries(
            num_queries,
            self.selection_strategy
        )

        # Apply style transfer and get inferred domain/scenario
        results = []
        for query_data in sampled_queries:
            original_query = query_data['text']

            # Sample target turns from normal distribution (mean=3)
            target_turns = sample_target_turns()

            # Use persona's query_length_pref if available, otherwise sample randomly
            target_length = persona_features.get('query_length_pref') or sample_length_category()

            if self.style_transfer:
                adapted_query, was_transferred, inferred_domain, inferred_scenario, target_length, target_turns, prompt_type = self.style_transfer.transfer(
                    original_query,
                    persona_features,
                    force=True,  # Always apply style transfer to get domain/scenario
                    target_length=target_length,
                    target_turns=target_turns
                )
            else:
                adapted_query = original_query
                was_transferred = False
                inferred_domain = 'general'
                inferred_scenario = 'casual'
                prompt_type = 'none'

            results.append({
                'query_id': query_data.get('query_id'),
                'original_query': original_query,
                'adapted_query': adapted_query,
                'was_style_transferred': was_transferred,
                'inferred_domain': inferred_domain,
                'inferred_scenario': inferred_scenario,
                'target_length': target_length,
                'target_turns': target_turns,
                'prompt_type': prompt_type,
                'source_dataset': query_data['source'],
                'metadata': query_data.get('metadata', {})
            })

        logging.info(
            f"Generated {len(results)} queries for persona "
            f"({sum(r['was_style_transferred'] for r in results)} style-transferred)"
        )

        return results
    
    def generate_queries_batch(self, personas: List[Dict[str, Any]],
                              show_progress: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate queries for multiple personas.
        
        Args:
            personas: List of persona specifications
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping persona IDs to their queries
        """
        all_queries = {}
        
        # Use tqdm for progress bar
        iterator = tqdm(personas, desc="Generating queries", unit="persona", 
                       ncols=100, colour='blue') if show_progress else personas
        
        for i, persona in enumerate(iterator):
            persona_id = persona.get('persona_id', f'persona_{i}')
            persona_features = persona.get('features', {})
            
            queries = self.generate_queries_for_persona(persona_features)
            all_queries[persona_id] = queries
            
            # Update progress bar description with stats
            if show_progress and isinstance(iterator, tqdm):
                total_queries = sum(len(q) for q in all_queries.values())
                iterator.set_postfix({'total_queries': total_queries})
        
        return all_queries

    def mark_used_query_ids(self, query_ids: Iterable[str]):
        """Mark query_ids as used to avoid resampling across runs."""
        self.query_dataset.mark_used_query_ids(query_ids)
