"""
Persona Spec Module

This module handles storage, retrieval, and management of generated persona specifications.
"""

import json
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib


class PersonaSpec:
    """
    Represents a single persona specification with metadata.
    """
    
    def __init__(self, 
                 persona_id: str,
                 features: Dict[str, str],
                 system_prompt: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a PersonaSpec.
        
        Args:
            persona_id: Unique identifier for the persona
            features: Dictionary of persona features (dimension -> value)
            system_prompt: Generated system prompt (optional)
            metadata: Additional metadata (timestamps, version, etc.)
        """
        self.persona_id = persona_id
        self.features = features
        self.system_prompt = system_prompt
        self.metadata = metadata or {}
        
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PersonaSpec to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'persona_id': self.persona_id,
            'features': self.features,
            'system_prompt': self.system_prompt,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonaSpec':
        """
        Create PersonaSpec from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            PersonaSpec instance
        """
        return cls(
            persona_id=data['persona_id'],
            features=data['features'],
            system_prompt=data.get('system_prompt'),
            metadata=data.get('metadata', {})
        )
    
    def __repr__(self) -> str:
        return f"PersonaSpec(id={self.persona_id}, features={len(self.features)})"


class PersonaSpecStorage:
    """
    Manages storage and retrieval of persona specifications.
    """
    
    def __init__(self, output_dir: str, format: str = 'json'):
        """
        Initialize PersonaSpecStorage.
        
        Args:
            output_dir: Directory to store persona specs
            format: Storage format ('json' or 'yaml')
        """
        self.output_dir = Path(output_dir)
        self.format = format.lower()
        
        if self.format not in ['json', 'yaml']:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file to track all personas
        self.index_file = self.output_dir / 'index.json'
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """
        Load the index of all stored personas.
        
        Returns:
            Index dictionary
        """
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'personas': [], 'count': 0, 'last_updated': None}
    
    def _save_index(self):
        """
        Save the index to disk.
        """
        self.index['last_updated'] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
    
    def save(self, persona_spec: PersonaSpec) -> str:
        """
        Save a persona specification to disk.
        
        Args:
            persona_spec: PersonaSpec to save
            
        Returns:
            Path to saved file
        """
        # Generate filename
        filename = f"{persona_spec.persona_id}.{self.format}"
        filepath = self.output_dir / filename
        
        # Save to file
        data = persona_spec.to_dict()
        
        if self.format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:  # yaml
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        # Update index
        if persona_spec.persona_id not in [p['persona_id'] for p in self.index['personas']]:
            self.index['personas'].append({
                'persona_id': persona_spec.persona_id,
                'filepath': str(filepath),
                'created_at': persona_spec.metadata.get('created_at'),
                'num_features': len(persona_spec.features)
            })
            self.index['count'] += 1
            self._save_index()
        
        return str(filepath)
    
    def load(self, persona_id: str) -> PersonaSpec:
        """
        Load a persona specification from disk.
        
        Args:
            persona_id: ID of persona to load
            
        Returns:
            PersonaSpec instance
            
        Raises:
            FileNotFoundError: If persona not found
        """
        filename = f"{persona_id}.{self.format}"
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Persona {persona_id} not found")
        
        if self.format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:  # yaml
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        
        return PersonaSpec.from_dict(data)
    
    def save_batch(self, persona_specs: List[PersonaSpec]) -> List[str]:
        """
        Save multiple persona specifications.
        
        Args:
            persona_specs: List of PersonaSpec instances
            
        Returns:
            List of saved file paths
        """
        filepaths = []
        for spec in persona_specs:
            filepath = self.save(spec)
            filepaths.append(filepath)
        return filepaths
    
    def load_all(self) -> List[PersonaSpec]:
        """
        Load all stored persona specifications.
        
        Returns:
            List of PersonaSpec instances
        """
        personas = []
        for entry in self.index['personas']:
            persona_id = entry['persona_id']
            try:
                persona = self.load(persona_id)
                personas.append(persona)
            except FileNotFoundError:
                print(f"Warning: Persona {persona_id} in index but file not found")
        return personas
    
    def delete(self, persona_id: str):
        """
        Delete a persona specification.
        
        Args:
            persona_id: ID of persona to delete
        """
        filename = f"{persona_id}.{self.format}"
        filepath = self.output_dir / filename
        
        if filepath.exists():
            filepath.unlink()
        
        # Update index
        self.index['personas'] = [
            p for p in self.index['personas'] 
            if p['persona_id'] != persona_id
        ]
        self.index['count'] = len(self.index['personas'])
        self._save_index()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored personas.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_personas': self.index['count'],
            'last_updated': self.index['last_updated'],
            'storage_format': self.format,
            'output_directory': str(self.output_dir)
        }
    
    def export_to_dataset(self, output_file: str, include_features: bool = True,
                         include_prompts: bool = True):
        """
        Export all personas to a single dataset file.
        
        Args:
            output_file: Path to output file
            include_features: Whether to include persona features
            include_prompts: Whether to include system prompts
        """
        personas = self.load_all()
        
        dataset = []
        for persona in personas:
            entry = {'persona_id': persona.persona_id}
            
            if include_features:
                entry['features'] = persona.features
            
            if include_prompts and persona.system_prompt:
                entry['system_prompt'] = persona.system_prompt
            
            dataset.append(entry)
        
        output_path = Path(output_file)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        elif output_path.suffix in ['.yaml', '.yml']:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(dataset, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        print(f"Exported {len(dataset)} personas to {output_file}")

    def export_personas_to_dataset(
        self,
        personas: List[PersonaSpec],
        output_file: str,
        include_features: bool = True,
        include_prompts: bool = True,
    ):
        """
        Export a provided list of personas to a single dataset file.

        Args:
            personas: List of PersonaSpec instances to export
            output_file: Path to output file
            include_features: Whether to include persona features
            include_prompts: Whether to include system prompts
        """
        dataset: List[Dict[str, Any]] = []

        for persona in personas:
            entry: Dict[str, Any] = {"persona_id": persona.persona_id}

            if include_features:
                entry["features"] = persona.features

            if include_prompts and persona.system_prompt is not None:
                entry["system_prompt"] = persona.system_prompt

            dataset.append(entry)

        output_path = Path(output_file)

        if output_path.suffix == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        elif output_path.suffix in [".yaml", ".yml"]:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(dataset, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        print(f"Exported {len(dataset)} personas to {output_file}")


def generate_persona_id(features: Dict[str, str]) -> str:
    """
    Generate a unique ID for a persona based on its features.
    
    Args:
        features: Dictionary of persona features
        
    Returns:
        Unique persona ID
    """
    # Create a deterministic hash from features
    feature_str = json.dumps(features, sort_keys=True)
    hash_obj = hashlib.md5(feature_str.encode())
    hash_hex = hash_obj.hexdigest()[:12]
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return f"persona_{timestamp}_{hash_hex}"
