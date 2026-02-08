"""
Training Data Module

This module handles collection, labeling, and export of training data
for prompt optimization, including positive/negative pairs for RL training.
"""

import json
import random
from typing import Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging


@dataclass
class TrainingSample:
    """Represents a single training sample with conversation trajectory."""
    sample_id: str
    persona_id: str
    persona_features: Dict[str, str]
    original_query: str  # Original query from dataset (before style transfer)
    initial_query: str   # Query after style transfer
    prompt_trajectory: List[str]  # User prompts in order
    full_conversation: List[Dict[str, str]]  # Complete conversation with roles
    num_turns: int

    # Optional: noisy versions for robustness
    noisy_initial_queries: List[Dict[str, Any]] = None
    noisy_trajectories: List[Dict[str, Any]] = None

    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'sample_id': self.sample_id,
            'persona_id': self.persona_id,
            'persona_features': self.persona_features,
            'original_query': self.original_query,
            'initial_query': self.initial_query,
            'prompt_trajectory': self.prompt_trajectory,
            'full_conversation': self.full_conversation,
            'num_turns': self.num_turns
        }
        
        if self.noisy_initial_queries:
            result['noisy_initial_queries'] = self.noisy_initial_queries
        
        if self.noisy_trajectories:
            result['noisy_trajectories'] = self.noisy_trajectories
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        return result


# TrainingPair class removed - users will create their own positive/negative pairs


class TrainingDataCollector:
    """
    Collects and organizes training data from interactions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TrainingDataCollector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config.get('training_data', {})
        
        # Data format settings
        format_config = self.training_config.get('format', {})
        self.include_persona = format_config.get('include_persona', True)
        self.include_trajectory = format_config.get('include_trajectory', True)
        self.include_full_conversation = format_config.get('include_full_conversation', True)
        self.include_distractor = format_config.get('include_distractor_versions', True)
    
    def _extract_prompt_trajectory(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract user prompt trajectory from messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of user prompts in order
        """
        trajectory = []
        for msg in messages:
            if msg.get('role') == 'user':
                trajectory.append(msg.get('content', ''))
        return trajectory
    
    def interaction_to_training_sample(self, interaction: Dict[str, Any]) -> TrainingSample:
        """
        Convert an interaction to a training sample.
        
        Args:
            interaction: Interaction dictionary
            
        Returns:
            TrainingSample object
        """
        # Extract trajectories
        messages = interaction.get('messages', [])
        prompt_trajectory = self._extract_prompt_trajectory(messages)
        
        # Get full conversation if needed
        full_conversation = []
        if self.include_full_conversation:
            for msg in messages:
                msg_dict = {
                    'role': msg.get('role', ''),
                    'content': msg.get('content', '')
                }
                # Preserve metadata (includes model/provider info for assistant messages)
                if msg.get('metadata'):
                    msg_dict['metadata'] = msg.get('metadata')
                full_conversation.append(msg_dict)
        
        # Extract noisy versions if available
        noisy_versions = interaction.get('noisy_versions', {})
        noisy_initial = noisy_versions.get('initial_query', []) if self.include_distractor else None
        noisy_followups = noisy_versions.get('followups', []) if self.include_distractor else None
        
        # Create training sample (no labeling)
        sample = TrainingSample(
            sample_id=interaction.get('interaction_id', ''),
            persona_id=interaction.get('persona_id', ''),
            persona_features=interaction.get('persona_features', {}) if self.include_persona else {},
            original_query=interaction.get('original_query', interaction.get('initial_query', '')),
            initial_query=interaction.get('initial_query', ''),
            prompt_trajectory=prompt_trajectory if self.include_trajectory else [prompt_trajectory[0]] if prompt_trajectory else [],
            full_conversation=full_conversation,
            num_turns=interaction.get('num_turns', 0),
            noisy_initial_queries=noisy_initial,
            noisy_trajectories=noisy_followups,
            metadata=interaction.get('metadata', {})
        )
        
        return sample
    
    def collect_from_interactions(self, interactions: List[Dict[str, Any]]) -> List[TrainingSample]:
        """
        Collect training samples from interactions (no automatic labeling).
        
        Args:
            interactions: List of interaction dictionaries
            
        Returns:
            List of TrainingSample objects
        """
        samples = []
        
        for interaction in interactions:
            sample = self.interaction_to_training_sample(interaction)
            samples.append(sample)
        
        logging.info(f"Collected {len(samples)} samples from interactions")
        
        return samples


class TrainingDataExporter:
    """
    Exports training data to various formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TrainingDataExporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config.get('training_data', {})
        self.paths_config = config.get('paths', {}).get('training_data', {})
        
        self.output_dir = Path(self.paths_config.get('output_dir', 'data/training_data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split ratios
        split_config = self.training_config.get('split', {})
        self.train_ratio = split_config.get('train', 0.8)
        self.val_ratio = split_config.get('validation', 0.1)
        self.test_ratio = split_config.get('test', 0.1)
    
    def _split_data(self, data: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Split data into train/val/test sets.
        
        Args:
            data: List of data items
            
        Returns:
            Tuple of (train, val, test) lists
        """
        random.shuffle(data)
        
        n = len(data)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        train = data[:n_train]
        val = data[n_train:n_train + n_val]
        test = data[n_train + n_val:]
        
        return train, val, test
    
    def export_samples(self, samples: List[TrainingSample],
                      split: bool = True,
                      use_timestamp: bool = True) -> Dict[str, str]:
        """
        Export training samples to files (no automatic labeling).
        
        Args:
            samples: List of TrainingSample objects
            split: Whether to split into train/val/test
            use_timestamp: Whether to add timestamp to filename (default: True)
            
        Returns:
            Dictionary of exported file paths
        """
        from datetime import datetime
        
        exported_files = {}
        
        # Generate timestamp suffix if needed
        timestamp_suffix = ""
        if use_timestamp:
            timestamp_suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if split:
            # Split data
            train, val, test = self._split_data(samples)
            
            # Export splits
            for split_name, data in [('train', train), ('val', val), ('test', test)]:
                filepath = self.output_dir / f"{split_name}_samples{timestamp_suffix}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(
                        [sample.to_dict() for sample in data],
                        f,
                        indent=2,
                        ensure_ascii=False
                    )
                exported_files[split_name] = str(filepath)
                
                logging.info(f"Exported {len(data)} {split_name} samples to {filepath}")
        
        else:
            # Export all together to train_samples.json (not all_samples.json)
            filepath = self.output_dir / f"train_samples{timestamp_suffix}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(
                    [sample.to_dict() for sample in samples],
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            exported_files['train'] = str(filepath)
            
            logging.info(f"Exported {len(samples)} samples to {filepath}")
        
        return exported_files
    
    def export_statistics(self, samples: List[TrainingSample], use_timestamp: bool = True) -> str:
        """
        Export statistics about the training data.

        Args:
            samples: List of TrainingSample objects
            use_timestamp: Whether to add timestamp to filename (default: True)

        Returns:
            Path to statistics file
        """
        from datetime import datetime

        # Safely compute statistics, handling empty samples list
        num_samples = len(samples)
        if num_samples > 0:
            turns_list = [s.num_turns for s in samples]
            avg_turns = sum(turns_list) / num_samples
            min_turns = min(turns_list)
            max_turns = max(turns_list)
            unique_personas = len(set(s.persona_id for s in samples))
            total_user_prompts = sum(len(s.prompt_trajectory) for s in samples)
        else:
            avg_turns = 0
            min_turns = 0
            max_turns = 0
            unique_personas = 0
            total_user_prompts = 0

        stats = {
            'total_samples': num_samples,
            'avg_turns': avg_turns,
            'min_turns': min_turns,
            'max_turns': max_turns,
            'unique_personas': unique_personas,
            'total_user_prompts': total_user_prompts,
            'generated_at': datetime.now().isoformat()
        }
        
        # Generate timestamp suffix if needed
        timestamp_suffix = ""
        if use_timestamp:
            timestamp_suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filepath = self.output_dir / f"statistics{timestamp_suffix}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logging.info(f"Exported statistics to {filepath}")
        
        return str(filepath)
