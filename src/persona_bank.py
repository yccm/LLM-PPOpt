"""
Persona Bank Module

This module handles loading and managing the persona bank,
including dimension definitions and possible values.
"""

import yaml
from typing import Dict, List, Any
from pathlib import Path


class PersonaBank:
    """
    Manages the persona bank including dimensions and their possible values.
    """
    
    def __init__(self, persona_path: str):
        """
        Initialize the PersonaBank.

        Args:
            persona_path: Path to the persona YAML file
        """
        self.persona_path = Path(persona_path)
        self.dimensions = self._load_dimensions()
        self.values = self._load_values()
        self._validate()
    
    def _load_dimensions(self) -> List[Dict[str, Any]]:
        """
        Load dimension definitions from YAML file.

        Returns:
            List of dimension definitions
        """
        with open(self.persona_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data.get('dimensions', [])
    
    def _load_values(self) -> Dict[str, List[str]]:
        """
        Extract possible values for each dimension from dimension definitions.

        Returns:
            Dictionary mapping dimension names to their possible values
        """
        values = {}
        for dim in self.dimensions:
            dim_name = dim.get('name')
            dim_values = dim.get('values', [])
            if dim_name:
                values[dim_name] = dim_values
        return values
    
    def _validate(self):
        """
        Validate that all dimensions have corresponding values defined.
        
        Raises:
            ValueError: If validation fails
        """
        dimension_names = {dim['name'] for dim in self.dimensions}
        value_names = set(self.values.keys())
        
        missing_values = dimension_names - value_names
        if missing_values:
            raise ValueError(
                f"Missing values for dimensions: {missing_values}"
            )
        
        extra_values = value_names - dimension_names
        if extra_values:
            print(f"Warning: Extra values defined for: {extra_values}")
    
    def get_dimension(self, name: str) -> Dict[str, Any]:
        """
        Get dimension definition by name.
        
        Args:
            name: Dimension name
            
        Returns:
            Dimension definition dictionary
            
        Raises:
            KeyError: If dimension not found
        """
        for dim in self.dimensions:
            if dim['name'] == name:
                return dim
        raise KeyError(f"Dimension '{name}' not found")
    
    def get_values(self, dimension_name: str) -> List[str]:
        """
        Get possible values for a dimension.
        
        Args:
            dimension_name: Name of the dimension
            
        Returns:
            List of possible values
            
        Raises:
            KeyError: If dimension not found
        """
        if dimension_name not in self.values:
            raise KeyError(f"Values for dimension '{dimension_name}' not found")
        return self.values[dimension_name]
    
    def get_all_dimension_names(self) -> List[str]:
        """
        Get all dimension names.
        
        Returns:
            List of dimension names
        """
        return [dim['name'] for dim in self.dimensions]
    
    def get_dimensions_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all dimensions belonging to a specific category.
        
        Args:
            category: Category name (e.g., 'basic_info', 'behavioral', 'constraint')
            
        Returns:
            List of dimension definitions
        """
        return [
            dim for dim in self.dimensions 
            if dim.get('category') == category
        ]
    
    def get_constraint_dimensions(self) -> List[Dict[str, Any]]:
        """
        Get all dimensions that are constraints.
        
        Returns:
            List of constraint dimension definitions
        """
        return [
            dim for dim in self.dimensions 
            if dim.get('is_constraint', False)
        ]
    
    def get_non_constraint_dimensions(self) -> List[Dict[str, Any]]:
        """
        Get all dimensions that are not constraints.
        
        Returns:
            List of non-constraint dimension definitions
        """
        return [
            dim for dim in self.dimensions 
            if not dim.get('is_constraint', False)
        ]
    
    def __repr__(self) -> str:
        return (
            f"PersonaBank(dimensions={len(self.dimensions)}, "
            f"total_values={sum(len(v) for v in self.values.values())})"
        )
