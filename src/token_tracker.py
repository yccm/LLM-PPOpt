"""
Token Tracker Module

This module tracks token usage across different modules and API calls.
Records input tokens, output tokens, and costs for each LLM interaction.
"""

import json
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict


@dataclass
class TokenUsage:
    """Represents token usage for a single API call."""
    module: str  # Module name (e.g., "persona_formulation", "query_generation")
    operation: str  # Specific operation (e.g., "formulate_prompt", "style_transfer")
    model: str  # Model used (e.g., "gpt-4o-mini")
    provider: str  # Provider (e.g., "openai", "anthropic")
    input_tokens: int  # Number of input tokens
    output_tokens: int  # Number of output tokens
    total_tokens: int  # Total tokens (input + output)
    timestamp: str  # ISO timestamp
    metadata: Dict[str, Any]  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TokenTracker:
    """
    Global token usage tracker (singleton).
    Thread-safe for concurrent API calls.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single global tracker."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the tracker."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.records: List[TokenUsage] = []
            self.records_lock = threading.Lock()
            self.logger = logging.getLogger(__name__)
            
            # Aggregated statistics by module
            self.stats_by_module: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'call_count': 0}
            )
            
            # Aggregated statistics by model
            self.stats_by_model: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'call_count': 0}
            )
    
    def record(
        self,
        module: str,
        operation: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record token usage for an API call.
        
        Args:
            module: Module name (e.g., "persona_formulation")
            operation: Operation name (e.g., "formulate_prompt")
            model: Model name (e.g., "gpt-4o-mini")
            provider: Provider name (e.g., "openai")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Additional metadata
        """
        total_tokens = input_tokens + output_tokens
        
        usage = TokenUsage(
            module=module,
            operation=operation,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        with self.records_lock:
            self.records.append(usage)
            
            # Update aggregated stats by module
            self.stats_by_module[module]['input_tokens'] += input_tokens
            self.stats_by_module[module]['output_tokens'] += output_tokens
            self.stats_by_module[module]['total_tokens'] += total_tokens
            self.stats_by_module[module]['call_count'] += 1
            
            # Update aggregated stats by model
            model_key = f"{provider}/{model}"
            self.stats_by_model[model_key]['input_tokens'] += input_tokens
            self.stats_by_model[model_key]['output_tokens'] += output_tokens
            self.stats_by_model[model_key]['total_tokens'] += total_tokens
            self.stats_by_model[model_key]['call_count'] += 1
        
        self.logger.debug(
            f"Token usage: {module}.{operation} - "
            f"{input_tokens} in + {output_tokens} out = {total_tokens} total ({provider}/{model})"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated token usage statistics.
        
        Returns:
            Dictionary with statistics by module and by model
        """
        with self.records_lock:
            # Calculate global totals
            total_input = sum(r.input_tokens for r in self.records)
            total_output = sum(r.output_tokens for r in self.records)
            total = total_input + total_output
            
            return {
                'summary': {
                    'total_calls': len(self.records),
                    'total_input_tokens': total_input,
                    'total_output_tokens': total_output,
                    'total_tokens': total,
                },
                'by_module': dict(self.stats_by_module),
                'by_model': dict(self.stats_by_model),
            }
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        """
        Get all individual token usage records.
        
        Returns:
            List of token usage records as dictionaries
        """
        with self.records_lock:
            return [r.to_dict() for r in self.records]
    
    def reset(self):
        """Reset all tracking data."""
        with self.records_lock:
            self.records.clear()
            self.stats_by_module.clear()
            self.stats_by_model.clear()
            self.logger.info("Token tracker reset")
    
    def export_to_file(self, output_path: str, include_records: bool = True):
        """
        Export token usage to JSON file.
        
        Args:
            output_path: Path to output JSON file
            include_records: Whether to include individual records (default: True)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'exported_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
        }
        
        if include_records:
            data['records'] = self.get_all_records()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Token usage exported to {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """Print a human-readable summary of token usage."""
        stats = self.get_statistics()
        summary = stats['summary']
        
        print("\n" + "=" * 80)
        print("TOKEN USAGE SUMMARY")
        print("=" * 80)
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
        print(f"Total Output Tokens: {summary['total_output_tokens']:,}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        
        print("\n" + "-" * 80)
        print("BY MODULE:")
        print("-" * 80)
        for module, data in sorted(stats['by_module'].items()):
            print(f"\n{module}:")
            print(f"  Calls: {data['call_count']}")
            print(f"  Input: {data['input_tokens']:,} tokens")
            print(f"  Output: {data['output_tokens']:,} tokens")
            print(f"  Total: {data['total_tokens']:,} tokens")
        
        print("\n" + "-" * 80)
        print("BY MODEL:")
        print("-" * 80)
        for model, data in sorted(stats['by_model'].items()):
            print(f"\n{model}:")
            print(f"  Calls: {data['call_count']}")
            print(f"  Input: {data['input_tokens']:,} tokens")
            print(f"  Output: {data['output_tokens']:,} tokens")
            print(f"  Total: {data['total_tokens']:,} tokens")
        
        print("\n" + "=" * 80)


# Global singleton instance
_global_tracker = TokenTracker()


def get_tracker() -> TokenTracker:
    """Get the global token tracker instance."""
    return _global_tracker


def record_tokens(
    module: str,
    operation: str,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Convenience function to record tokens to the global tracker.
    
    Args:
        module: Module name
        operation: Operation name
        model: Model name
        provider: Provider name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        metadata: Additional metadata
    """
    _global_tracker.record(
        module=module,
        operation=operation,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata=metadata
    )
