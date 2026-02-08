"""
Sampling Module

This module implements various sampling strategies for generating diverse personas
by selecting subsets of user features.
"""

import random
import itertools
import yaml
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

from .persona_bank import PersonaBank


class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies.
    """
    
    @abstractmethod
    def sample(self, persona_bank: PersonaBank) -> Dict[str, str]:
        """
        Sample features from the persona bank to create a persona.
        
        Args:
            persona_bank: PersonaBank instance
            
        Returns:
            Dictionary mapping dimension names to selected values
        """
        pass

    def on_accept(self, persona: Dict[str, str]) -> None:
        """
        Optional hook called when a sampled persona is accepted (e.g., passes diversity checks).
        """
        return None

    def on_reject(self, persona: Dict[str, str]) -> None:
        """
        Optional hook called when a sampled persona is rejected (e.g., fails diversity checks).
        """
        return None


def _enforce_feature_bounds(
    rng: random.Random,
    selected_dimensions: Set[str],
    all_dimensions: List[str],
    min_features: int,
    max_features: int,
    fixed_dimensions: Optional[Set[str]] = None,
) -> List[str]:
    fixed_dimensions = fixed_dimensions or set()

    if min_features < 0:
        raise ValueError("min_features must be >= 0")
    if max_features < 0:
        raise ValueError("max_features must be >= 0")
    if min_features > max_features:
        raise ValueError("min_features cannot be greater than max_features")

    selected_dimensions = set(selected_dimensions) | set(fixed_dimensions)

    max_allowed = min(max_features, len(all_dimensions))
    min_required = min(min_features, len(all_dimensions))

    if len(selected_dimensions) < min_required:
        remaining = [d for d in all_dimensions if d not in selected_dimensions]
        needed = min_required - len(selected_dimensions)
        if needed > 0:
            selected_dimensions.update(rng.sample(remaining, needed))

    if len(selected_dimensions) > max_allowed:
        must_keep = [d for d in all_dimensions if d in fixed_dimensions]
        optional = [
            d
            for d in all_dimensions
            if d in selected_dimensions and d not in fixed_dimensions
        ]
        keep_optional = max_allowed - len(must_keep)
        if keep_optional <= 0:
            return rng.sample(must_keep, max_allowed)
        return must_keep + rng.sample(optional, keep_optional)

    return [d for d in all_dimensions if d in selected_dimensions]


class RandomSampling(SamplingStrategy):
    """
    Random sampling strategy that randomly selects features based on availability rate.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize random sampling strategy.
        
        Args:
            config: Configuration dictionary with sampling parameters
        """
        self.feature_availability_rate = config.get('feature_availability_rate', 0.7)
        self.constraint_probability = config.get('constraint_probability', 0.0)
        self.min_features = config.get('min_features', 5)
        self.max_features = config.get('max_features', 15)
        self.seed = config.get('seed', None)
        self.rng = random.Random(self.seed)
    
    def sample(self, persona_bank: PersonaBank) -> Dict[str, str]:
        """
        Randomly sample features from the persona bank.
        
        Args:
            persona_bank: PersonaBank instance
            
        Returns:
            Dictionary mapping dimension names to selected values
        """
        all_dimensions = persona_bank.get_all_dimension_names()
        constraint_dimensions = {
            dim["name"] for dim in persona_bank.get_constraint_dimensions()
        }

        selected: Set[str] = set()
        for dim_name in all_dimensions:
            probability = (
                self.constraint_probability
                if dim_name in constraint_dimensions
                else self.feature_availability_rate
            )
            if self.rng.random() < probability:
                selected.add(dim_name)

        final_dimensions = _enforce_feature_bounds(
            self.rng,
            selected,
            all_dimensions,
            self.min_features,
            self.max_features,
        )

        persona: Dict[str, str] = {}
        for dim_name in final_dimensions:
            possible_values = persona_bank.get_values(dim_name)
            persona[dim_name] = self.rng.choice(possible_values)

        return persona


class StratifiedSampling(SamplingStrategy):
    """
    Stratified sampling strategy that ensures balanced representation across key dimensions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stratified sampling strategy.
        
        Args:
            config: Configuration dictionary with sampling parameters
        """
        self.stratify_by = config.get('stratify_by', [])
        self.samples_per_stratum = config.get('samples_per_stratum', 10)
        self.allow_overlap = config.get('allow_overlap', True)
        self.base_availability_rate = config.get('base_availability_rate', 0.6)
        self.constraint_probability = config.get('constraint_probability', 0.0)
        self.min_features = config.get('min_features', 5)
        self.max_features = config.get('max_features', 15)
        self.seed = config.get('seed', None)
        self.rng = random.Random(self.seed)
        
        self._current_stratum_counts: Dict[Tuple, int] = {}
        self._stratify_dimensions: Optional[List[str]] = None
        self._all_strata: Optional[List[Tuple[str, ...]]] = None
        self._pending_stratum: Optional[Tuple[str, ...]] = None
    
    def sample(self, persona_bank: PersonaBank) -> Dict[str, str]:
        """
        Sample features using stratified approach.
        
        Args:
            persona_bank: PersonaBank instance
            
        Returns:
            Dictionary mapping dimension names to selected values
        """
        all_dimensions = persona_bank.get_all_dimension_names()
        constraint_dimensions = {
            dim["name"] for dim in persona_bank.get_constraint_dimensions()
        }

        if self._stratify_dimensions is None or self._all_strata is None:
            self._stratify_dimensions = [
                dim for dim in self.stratify_by if dim in all_dimensions
            ]
            if not self._stratify_dimensions:
                fallback = RandomSampling(
                    {
                        "feature_availability_rate": self.base_availability_rate,
                        "constraint_probability": self.constraint_probability,
                        "min_features": self.min_features,
                        "max_features": self.max_features,
                        "seed": self.seed,
                    }
                )
                return fallback.sample(persona_bank)

            values_lists = [persona_bank.get_values(d) for d in self._stratify_dimensions]
            self._all_strata = list(itertools.product(*values_lists))

        candidate_strata = self._all_strata
        if not self.allow_overlap:
            candidate_strata = [
                s
                for s in self._all_strata
                if self._current_stratum_counts.get(s, 0) < self.samples_per_stratum
            ]
            if not candidate_strata:
                raise RuntimeError(
                    "No available strata remaining (allow_overlap is false)."
                )

        min_count = min(self._current_stratum_counts.get(s, 0) for s in candidate_strata)
        least_sampled = [
            s for s in candidate_strata if self._current_stratum_counts.get(s, 0) == min_count
        ]
        chosen_stratum = self.rng.choice(least_sampled)
        self._pending_stratum = chosen_stratum

        persona: Dict[str, str] = {}
        fixed_dimensions = set(self._stratify_dimensions)
        for dim_name, value in zip(self._stratify_dimensions, chosen_stratum):
            persona[dim_name] = value

        selected: Set[str] = set()
        for dim_name in all_dimensions:
            if dim_name in fixed_dimensions:
                continue
            probability = (
                self.constraint_probability
                if dim_name in constraint_dimensions
                else self.base_availability_rate
            )
            if self.rng.random() < probability:
                selected.add(dim_name)

        final_dimensions = _enforce_feature_bounds(
            self.rng,
            selected,
            all_dimensions,
            self.min_features,
            self.max_features,
            fixed_dimensions=fixed_dimensions,
        )

        for dim_name in final_dimensions:
            if dim_name in persona:
                continue
            possible_values = persona_bank.get_values(dim_name)
            persona[dim_name] = self.rng.choice(possible_values)

        return persona

    def on_accept(self, persona: Dict[str, str]) -> None:
        if self._pending_stratum is None:
            return None
        self._current_stratum_counts[self._pending_stratum] = (
            self._current_stratum_counts.get(self._pending_stratum, 0) + 1
        )
        self._pending_stratum = None
        return None

    def on_reject(self, persona: Dict[str, str]) -> None:
        self._pending_stratum = None
        return None


class CustomSampling(SamplingStrategy):
    """
    Custom sampling strategy that applies rules for more realistic persona generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize custom sampling strategy.
        
        Args:
            config: Configuration dictionary with sampling parameters and rules
        """
        self.rules = config.get('rules', [])
        self.probabilistic = config.get('probabilistic', True)
        self.base_availability_rate = config.get('base_availability_rate', 0.6)
        self.constraint_probability = config.get('constraint_probability', 0.0)
        self.min_features = config.get('min_features', 5)
        self.max_features = config.get('max_features', 15)
        self.seed = config.get('seed', None)
        self.rng = random.Random(self.seed)
    
    def sample(self, persona_bank: PersonaBank) -> Dict[str, str]:
        """
        Sample features using custom rules.
        
        Args:
            persona_bank: PersonaBank instance
            
        Returns:
            Dictionary mapping dimension names to selected values
        """
        all_dimensions = persona_bank.get_all_dimension_names()
        constraint_dimensions = {
            dim["name"] for dim in persona_bank.get_constraint_dimensions()
        }

        selected: Set[str] = set()
        for dim_name in all_dimensions:
            probability = (
                self.constraint_probability
                if dim_name in constraint_dimensions
                else self.base_availability_rate
            )
            if self.rng.random() < probability:
                selected.add(dim_name)

        final_dimensions = _enforce_feature_bounds(
            self.rng,
            selected,
            all_dimensions,
            self.min_features,
            self.max_features,
        )

        persona: Dict[str, str] = {}
        for dim_name in final_dimensions:
            possible_values = persona_bank.get_values(dim_name)
            persona[dim_name] = self.rng.choice(possible_values)

        protected_dimensions: Set[str] = set()
        for rule in self.rules:
            self._apply_rule(rule, persona, persona_bank, protected_dimensions)

        if len(persona) > self.max_features:
            self._trim_persona(persona, protected_dimensions)

        return persona
    
    def _apply_rule(
        self,
        rule: Dict[str, Any],
        persona: Dict[str, str],
        persona_bank: PersonaBank,
        protected_dimensions: Set[str],
    ):
        """
        Apply a custom sampling rule to a persona.
        
        Args:
            rule: Rule definition
            persona: Current persona being built
            persona_bank: PersonaBank instance
        """
        condition = rule.get('condition', {})
        action = rule.get('action', {})
        
        # Check if condition is met
        cond_dimension = condition.get('dimension')
        cond_values = condition.get('values', [])
        
        if cond_dimension in persona and persona[cond_dimension] in cond_values:
            # Apply action
            action_dimension = action.get('dimension')
            boost_prob = action.get('boost_probability', 0.8)
            preferred_values = action.get('preferred_values', [])
            
            # Decide whether to apply action
            if self.probabilistic:
                should_apply = self.rng.random() < boost_prob
            else:
                should_apply = True

            if should_apply:
                self._apply_action(
                    action_dimension,
                    preferred_values,
                    persona,
                    persona_bank,
                )
                if cond_dimension:
                    protected_dimensions.add(cond_dimension)
                if action_dimension:
                    protected_dimensions.add(action_dimension)
    
    def _apply_action(self, dimension: str, preferred_values: List[str],
                     persona: Dict[str, str], persona_bank: PersonaBank):
        """
        Apply an action by setting or modifying a dimension value.
        
        Args:
            dimension: Dimension to modify
            preferred_values: Preferred values for the dimension
            persona: Current persona
            persona_bank: PersonaBank instance
        """
        if dimension in persona_bank.get_all_dimension_names():
            if preferred_values:
                # Choose from preferred values
                available_preferred = [
                    v for v in preferred_values 
                    if v in persona_bank.get_values(dimension)
                ]
                if available_preferred:
                    persona[dimension] = self.rng.choice(available_preferred)
                else:
                    # Fallback to any value
                    persona[dimension] = self.rng.choice(
                        persona_bank.get_values(dimension)
                    )
            else:
                # Choose any value
                persona[dimension] = self.rng.choice(
                    persona_bank.get_values(dimension)
                )

    def _trim_persona(self, persona: Dict[str, str], protected_dimensions: Set[str]):
        if len(persona) <= self.max_features:
            return

        protected = [k for k in persona.keys() if k in protected_dimensions]
        optional = [k for k in persona.keys() if k not in protected_dimensions]

        if len(protected) >= self.max_features:
            keep = set(self.rng.sample(protected, self.max_features))
        else:
            remaining_slots = self.max_features - len(protected)
            keep = set(protected)
            keep.update(self.rng.sample(optional, remaining_slots))

        for key in list(persona.keys()):
            if key not in keep:
                del persona[key]


class PersonaSampler:
    """
    Main sampler class that coordinates persona generation with diversity enforcement.
    """

    def __init__(
        self,
        config_path: str,
        persona_bank: PersonaBank,
        overrides: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the PersonaSampler.

        Args:
            config_path: Path to sampling configuration file
            persona_bank: PersonaBank instance
            llm_client: Optional LLM client for generating "others" values
        """
        self.persona_bank = persona_bank
        self.llm_client = llm_client
        self.config = self._load_config(config_path)
        if overrides:
            self._apply_overrides(overrides)
        self.strategy = self._create_strategy()
        self.generated_personas: List[Dict[str, str]] = []

        seed_candidates = [
            self.config.get("seed"),
            self.config.get("random_sampling", {}).get("seed"),
            self.config.get("stratified_sampling", {}).get("seed"),
            self.config.get("custom_sampling", {}).get("seed"),
        ]
        seed = next((s for s in seed_candidates if s is not None), None)
        self.rng = random.Random(seed)
        
        # Diversity settings
        self.diversity_enabled = self.config.get('diversity', {}).get('enabled', True)
        self.min_hamming_distance = self.config.get('diversity', {}).get(
            'min_hamming_distance', 3
        )
        self.max_attempts = self.config.get('diversity', {}).get('max_attempts', 50)
        self.priority_dimensions = set(
            self.config.get("diversity", {}).get("priority_dimensions", [])
        )
        self.priority_weight = int(
            self.config.get("diversity", {}).get("priority_weight", 2)
        )
        
        # Constraint settings
        self.require_constraint = self.config.get('constraints', {}).get(
            'require_constraint', True
        )
        self.min_constraints = self.config.get('constraints', {}).get(
            'min_constraints', 1
        )
        self.max_constraints = self.config.get('constraints', {}).get(
            'max_constraints', 3
        )

        # Required dimensions - these MUST be included in every persona
        self.required_dimensions = set(
            self.config.get('required_dimensions', [])
        )

    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        strategy = overrides.get("strategy")
        if strategy:
            self.config["strategy"] = strategy

        feature_availability_rate = overrides.get("feature_availability_rate")
        if feature_availability_rate is not None:
            self.config.setdefault("random_sampling", {})[
                "feature_availability_rate"
            ] = feature_availability_rate
            self.config.setdefault("stratified_sampling", {})[
                "base_availability_rate"
            ] = feature_availability_rate
            self.config.setdefault("custom_sampling", {})[
                "base_availability_rate"
            ] = feature_availability_rate

        diversity_overrides = overrides.get("diversity")
        if isinstance(diversity_overrides, dict):
            self.config.setdefault("diversity", {}).update(diversity_overrides)

        seed = overrides.get("seed")
        if seed is not None:
            self.config.setdefault("random_sampling", {})["seed"] = seed
            self.config.setdefault("stratified_sampling", {})["seed"] = seed
            self.config.setdefault("custom_sampling", {})["seed"] = seed
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load sampling configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_strategy(self) -> SamplingStrategy:
        """
        Create sampling strategy based on configuration.
        
        Returns:
            SamplingStrategy instance
        """
        strategy_name = self.config.get('strategy', 'random')
        constraint_probability = self.config.get("constraints", {}).get(
            "constraint_probability", 0.0
        )

        if strategy_name == 'random':
            strategy_config = {
                **self.config.get('random_sampling', {}),
                "constraint_probability": constraint_probability,
            }
            return RandomSampling(strategy_config)
        elif strategy_name == 'stratified':
            strategy_config = {
                **self.config.get('stratified_sampling', {}),
                "constraint_probability": constraint_probability,
            }
            return StratifiedSampling(strategy_config)
        elif strategy_name == 'custom':
            strategy_config = {
                **self.config.get('custom_sampling', {}),
                "constraint_probability": constraint_probability,
            }
            return CustomSampling(strategy_config)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy_name}")
    
    def sample_persona(self) -> Dict[str, str]:
        """
        Generate a single persona with diversity enforcement.

        Returns:
            Dictionary representing a persona
        """
        for attempt in range(self.max_attempts):
            persona = self.strategy.sample(self.persona_bank)

            # Enforce required dimensions (must be included in every persona)
            persona = self._ensure_required_dimensions(persona)

            # Enforce constraint requirements
            if self.require_constraint:
                persona = self._ensure_constraints(persona)

            # Resolve "others" values
            persona = self._resolve_others(persona)

            # Check diversity
            if self.diversity_enabled:
                if self._is_diverse(persona):
                    self.generated_personas.append(persona)
                    self.strategy.on_accept(persona)
                    return persona
                self.strategy.on_reject(persona)
            else:
                self.generated_personas.append(persona)
                self.strategy.on_accept(persona)
                return persona

        # If max attempts reached, return the last generated persona
        print(f"Warning: Could not generate diverse persona after {self.max_attempts} attempts")
        self.generated_personas.append(persona)
        self.strategy.on_accept(persona)
        return persona

    def _ensure_required_dimensions(self, persona: Dict[str, str]) -> Dict[str, str]:
        """
        Ensure persona includes all required dimensions.

        Args:
            persona: Current persona

        Returns:
            Modified persona with required dimensions
        """
        for dim_name in self.required_dimensions:
            if dim_name not in persona:
                # Add the required dimension with a random value
                try:
                    possible_values = self.persona_bank.get_values(dim_name)
                    persona[dim_name] = self.rng.choice(possible_values)
                except Exception as e:
                    print(f"Warning: Could not add required dimension '{dim_name}': {e}")

        return persona
    
    def _ensure_constraints(self, persona: Dict[str, str]) -> Dict[str, str]:
        """
        Ensure persona has required number of constraint dimensions.
        
        Args:
            persona: Current persona
            
        Returns:
            Modified persona with constraints
        """
        constraint_dims = self.persona_bank.get_constraint_dimensions()
        current_constraints = [
            dim['name'] for dim in constraint_dims 
            if dim['name'] in persona
        ]
        
        num_current = len(current_constraints)
        
        if num_current < self.min_constraints:
            # Add more constraints
            available_constraints = [
                dim['name'] for dim in constraint_dims
                if dim['name'] not in persona
            ]
            num_to_add = min(
                self.min_constraints - num_current,
                len(available_constraints)
            )
            
            for dim_name in self.rng.sample(available_constraints, num_to_add):
                possible_values = self.persona_bank.get_values(dim_name)
                persona[dim_name] = self.rng.choice(possible_values)
        
        elif num_current > self.max_constraints:
            # Remove excess constraints
            num_to_remove = num_current - self.max_constraints
            dims_to_remove = self.rng.sample(current_constraints, num_to_remove)
            for dim_name in dims_to_remove:
                del persona[dim_name]
        
        return persona
    
    def _is_diverse(self, persona: Dict[str, str]) -> bool:
        """
        Check if persona is sufficiently diverse from existing personas.
        
        Args:
            persona: Persona to check
            
        Returns:
            True if persona is diverse enough, False otherwise
        """
        for existing in self.generated_personas:
            distance = self._hamming_distance(persona, existing)
            if distance < self.min_hamming_distance:
                return False
        return True
    
    def _hamming_distance(self, persona1: Dict[str, str],
                         persona2: Dict[str, str]) -> int:
        """
        Calculate Hamming distance between two personas.

        Args:
            persona1: First persona
            persona2: Second persona

        Returns:
            Number of differing features
        """
        all_keys = set(persona1.keys()) | set(persona2.keys())
        distance = 0

        for key in all_keys:
            val1 = persona1.get(key)
            val2 = persona2.get(key)
            if val1 != val2:
                distance += self.priority_weight if key in self.priority_dimensions else 1

        return distance

    def _resolve_others(self, persona: Dict[str, str]) -> Dict[str, str]:
        """
        Resolve "others" values by generating new unique values using LLM.

        Args:
            persona: Persona with potential "others" values

        Returns:
            Persona with "others" values replaced
        """
        if not self.llm_client:
            return persona

        for dim_name, value in list(persona.items()):
            if value == "others":
                dimension = self.persona_bank.get_dimension(dim_name)
                existing_values = self.persona_bank.get_values(dim_name)
                existing_values = [v for v in existing_values if v != "others"]

                prompt = f"""Generate a new value for the dimension "{dim_name}".

Dimension description: {dimension.get('description', 'No description available')}

Existing values (your generated value MUST be different from all of these):
{', '.join(existing_values)}

Requirements:
- Generate ONE new value that is distinct from all existing values
- The value should fit the dimension's purpose and description
- Use lowercase with underscores (e.g., "new_value")
- Be concise (1-3 words maximum)
- Output ONLY the new value, nothing else"""

                try:
                    new_value = self.llm_client.generate(prompt, temperature=0.8, max_completion_tokens=50).strip()
                    new_value = new_value.lower().replace(' ', '_').replace('-', '_')
                    persona[dim_name] = new_value
                except Exception as e:
                    print(f"Warning: Failed to generate value for {dim_name}: {e}")

        return persona
    
    def sample_batch(self, num_personas: int) -> List[Dict[str, str]]:
        """
        Generate a batch of personas.
        
        Args:
            num_personas: Number of personas to generate
            
        Returns:
            List of persona dictionaries
        """
        personas = []
        for i in range(num_personas):
            persona = self.sample_persona()
            personas.append(persona)
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_personas} personas")
        
        return personas
    
    def reset(self):
        """
        Reset the sampler state (clear generated personas).
        """
        self.generated_personas = []
