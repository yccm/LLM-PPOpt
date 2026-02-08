"""
LLM Client Module

This module provides a unified interface for interacting with various LLM APIs
(OpenAI, Anthropic, etc.) for persona-to-prompt formulation.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# Import token tracker
try:
    from .token_tracker import record_tokens
except ImportError:
    # Fallback if token_tracker not available
    def record_tokens(*args, **kwargs):
        pass


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def generate_with_tokens(self, prompt: str, **kwargs) -> Tuple[str, int, int]:
        """
        Generate text from the LLM and return token usage.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_text, input_tokens, output_tokens)
        """
        pass


class OpenAIClient(LLMClient):
    """
    Client for OpenAI API.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **default_params
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name
            **default_params: Default generation parameters
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.default_headers = default_headers
        self.default_params = default_params
        
        try:
            import openai
            self.openai = openai
            # OpenRouter (and other OpenAI-compatible gateways) can be used via base_url.
            # Example base_url: "https://openrouter.ai/api/v1"
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            if default_headers:
                client_kwargs["default_headers"] = default_headers
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        text, _, _ = self.generate_with_tokens(prompt, **kwargs)
        return text
    
    def generate_with_tokens(self, prompt: str, **kwargs) -> Tuple[str, int, int]:
        """
        Generate text using OpenAI API and return token usage.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Tuple of (generated_text, input_tokens, output_tokens)
        """
        params = {**self.default_params, **kwargs}
        
        try:
            # Support both max_tokens and max_completion_tokens
            # Prefer max_completion_tokens (new parameter name)
            max_tokens_param = params.get('max_completion_tokens') or params.get('max_tokens', 2048)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=params.get('temperature', 0.7),
                max_completion_tokens=max_tokens_param,
                top_p=params.get('top_p', 0.9),
                frequency_penalty=params.get('frequency_penalty', 0.0),
                presence_penalty=params.get('presence_penalty', 0.0)
            )
            
            text = response.choices[0].message.content.strip()
            
            # Extract token usage from response
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            return text, input_tokens, output_tokens
        
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")


class AnthropicClient(LLMClient):
    """
    Client for Anthropic API.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229", 
                 **default_params):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model: Model name
            **default_params: Default generation parameters
        """
        self.api_key = api_key
        self.model = model
        self.default_params = default_params
        
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Anthropic API.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        text, _, _ = self.generate_with_tokens(prompt, **kwargs)
        return text
    
    def generate_with_tokens(self, prompt: str, **kwargs) -> Tuple[str, int, int]:
        """
        Generate text using Anthropic API and return token usage.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Tuple of (generated_text, input_tokens, output_tokens)
        """
        params = {**self.default_params, **kwargs}
        
        try:
            # Anthropic uses max_tokens (not max_completion_tokens)
            max_tokens_param = params.get('max_completion_tokens') or params.get('max_tokens', 2048)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens_param,
                temperature=params.get('temperature', 0.7),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            text = response.content[0].text.strip()
            
            # Extract token usage from response
            usage = response.usage
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0
            
            return text, input_tokens, output_tokens
        
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without API calls.
    """
    
    def __init__(self):
        """
        Initialize mock client.
        """
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate mock response.
        
        Args:
            prompt: Input prompt
            **kwargs: Ignored
            
        Returns:
            Mock generated text
        """
        return (
            "You are an AI assistant optimized for this user's preferences. "
            "Provide clear, helpful responses tailored to their communication style "
            "and expertise level. Consider their constraints and adapt your responses "
            "accordingly."
        )
    
    def generate_with_tokens(self, prompt: str, **kwargs) -> Tuple[str, int, int]:
        """
        Generate mock response with mock token counts.
        
        Args:
            prompt: Input prompt
            **kwargs: Ignored
            
        Returns:
            Tuple of (generated_text, input_tokens, output_tokens)
        """
        text = self.generate(prompt, **kwargs)
        # Mock token counts
        input_tokens = len(prompt.split()) * 2  # Rough estimate
        output_tokens = len(text.split()) * 2
        return text, input_tokens, output_tokens


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """
    Factory function to create appropriate LLM client based on configuration.
    
    Args:
        config: Configuration dictionary with API settings
        
    Returns:
        LLMClient instance
    """
    provider = config.get('provider', 'openai').lower()
    api_key = config.get('api_key', '')
    
    # Check for API key in environment variable if not in config
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        env_var = f"{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var, '')
        
        if not api_key:
            print(f"Warning: No API key found. Using mock client.")
            return MockLLMClient()
    
    model = config.get('model', '')
    inference_params = config.get('inference', {})
    base_url = config.get('base_url')
    endpoint = config.get('endpoint')
    default_headers = config.get('default_headers') or config.get('headers')

    # If an OpenAI-compatible endpoint is provided (e.g. OpenRouter full URL),
    # normalize it to a base_url for the OpenAI SDK.
    if endpoint and not base_url:
        if endpoint.endswith("/chat/completions"):
            base_url = endpoint.rsplit("/chat/completions", 1)[0]
        else:
            base_url = endpoint
    
    if provider == 'openai':
        if not model:
            model = 'gpt-4o-mini'
        return OpenAIClient(api_key, model, base_url=base_url, default_headers=default_headers, **inference_params)

    # OpenRouter exposes an OpenAI-compatible Chat Completions endpoint.
    # We treat it as an OpenAI-compatible provider, configured via base_url/endpoint.
    elif provider == 'openrouter':
        if not model:
            # OpenRouter model names are typically like "anthropic/claude-3.5-sonnet"
            model = 'openai/gpt-4o-mini'
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
        return OpenAIClient(api_key, model, base_url=base_url, default_headers=default_headers, **inference_params)
    
    elif provider == 'anthropic':
        if not model:
            model = 'claude-3-opus-20240229'
        return AnthropicClient(api_key, model, **inference_params)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class LLMFormulator:
    """
    Handles formulation of persona specifications into system prompts using LLM.
    """
    
    def __init__(self, llm_client: LLMClient, template_path: str, 
                 max_retries: int = 3, retry_delay: int = 2,
                 validate_output: bool = True, min_prompt_length: int = 50):
        """
        Initialize LLMFormulator.
        
        Args:
            llm_client: LLM client instance
            template_path: Path to prompt template file
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            validate_output: Whether to validate generated prompts
            min_prompt_length: Minimum acceptable prompt length
        """
        self.llm_client = llm_client
        self.template_path = template_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.validate_output = validate_output
        self.min_prompt_length = min_prompt_length
        
        self.template = self._load_template()
        
        # Get model info for token tracking
        self.model_name = getattr(llm_client, 'model', 'unknown')
        self.provider_name = 'openai' if isinstance(llm_client, OpenAIClient) else \
                           'anthropic' if isinstance(llm_client, AnthropicClient) else 'mock'
    
    def _load_template(self) -> str:
        """
        Load the prompt template.
        
        Returns:
            Template string
        """
        with open(self.template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def formulate(self, persona_features: Dict[str, str]) -> str:
        """
        Generate a system prompt from persona features.
        
        Args:
            persona_features: Dictionary of persona features
            
        Returns:
            Generated system prompt
            
        Raises:
            RuntimeError: If generation fails after all retries
        """
        # Format persona features into readable text
        persona_text = self._format_persona(persona_features)
        
        # Fill template
        prompt = self.template.replace('{persona}', persona_text)
        
        # Try generation with retries
        for attempt in range(self.max_retries):
            try:
                # Generate with token tracking
                system_prompt, input_tokens, output_tokens = self.llm_client.generate_with_tokens(prompt)
                
                # Record token usage
                record_tokens(
                    module='persona_formulation',
                    operation='formulate_system_prompt',
                    model=self.model_name,
                    provider=self.provider_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    metadata={'attempt': attempt + 1, 'persona_features_count': len(persona_features)}
                )
                
                # Validate if enabled
                if self.validate_output:
                    if self._validate_prompt(system_prompt):
                        return system_prompt
                    else:
                        # Import here to avoid circular dependency
                        from .colored_logger import print_colored
                        print_colored(f"\n{'='*80}", 'VALIDATION')
                        print_colored(f"❌ Validation failed on attempt {attempt + 1}", 'VALIDATION')
                        print_colored(f"{'='*80}", 'VALIDATION')
                        
                        # Debug: Show why validation failed
                        if len(system_prompt) < self.min_prompt_length:
                            print_colored(f"  → Reason: Too short (len={len(system_prompt)}, min={self.min_prompt_length})", 'WARNING')
                        elif not system_prompt.strip():
                            print_colored(f"  → Reason: Empty or whitespace only", 'WARNING')
                        else:
                            # Check which error indicator was found
                            error_patterns = [
                                'i cannot generate', 'unable to generate', 'i apologize',
                                "i'm sorry, i cannot", 'error occurred', 'failed to generate',
                                'invalid request', 'cannot create'
                            ]
                            prompt_lower = system_prompt.lower()
                            prompt_start = prompt_lower[:100]
                            found_patterns = [pat for pat in error_patterns if pat in prompt_start]
                            
                            if found_patterns:
                                print_colored(f"  → Reason: Error pattern in first 100 chars: {found_patterns}", 'WARNING')
                            elif len(system_prompt) < 100:
                                simple_errors = ['sorry', 'error', 'cannot', 'unable', 'failed']
                                found_simple = [err for err in simple_errors if err in prompt_lower]
                                if found_simple:
                                    print_colored(f"  → Reason: Short response with error words: {found_simple}", 'WARNING')
                        
                        # Print full LLM response
                        print_colored(f"\n  📄 Full LLM Response:", 'INFO')
                        print_colored(f"  {'─'*76}", 'INFO')
                        # Print with line numbers for easier reading
                        lines = system_prompt.split('\n')
                        for i, line in enumerate(lines, 1):
                            print_colored(f"  {i:3d} | {line}", 'INFO')
                        print_colored(f"  {'─'*76}", 'INFO')
                        print_colored(f"  Length: {len(system_prompt)} characters, {len(lines)} lines\n", 'INFO')
                        
                        if attempt < self.max_retries - 1:
                            print_colored(f"  ⟳ Retrying... ({attempt + 2}/{self.max_retries})\n", 'VALIDATION')
                            time.sleep(self.retry_delay)
                            continue
                        else:
                            print_colored(f"  ✗ Max retries reached. Giving up.\n", 'ERROR')
                else:
                    return system_prompt
            
            except Exception as e:
                from .colored_logger import print_colored
                print_colored(f"Error on attempt {attempt + 1}: {str(e)}", 'ERROR')
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to generate prompt after {self.max_retries} attempts"
                    )
        
        raise RuntimeError("Failed to generate valid prompt")
    
    def _format_persona(self, features: Dict[str, str]) -> str:
        """
        Format persona features into human-readable text.
        
        Args:
            features: Dictionary of persona features
            
        Returns:
            Formatted persona description
        """
        lines = []
        for dimension, value in features.items():
            # Convert snake_case to Title Case
            dimension_formatted = dimension.replace('_', ' ').title()
            lines.append(f"- {dimension_formatted}: {value}")
        
        return '\n'.join(lines)
    
    def _validate_prompt(self, prompt: str) -> bool:
        """
        Validate generated system prompt.
        
        Args:
            prompt: Generated prompt
            
        Returns:
            True if valid, False otherwise
        """
        # Check minimum length
        if len(prompt) < self.min_prompt_length:
            return False
        
        # Check that it's not empty or just whitespace
        if not prompt.strip():
            return False
        
        # Check for error patterns that indicate generation failure
        # Use more specific patterns to avoid false positives
        error_patterns = [
            'i cannot generate',
            'unable to generate',
            'i apologize',
            "i'm sorry, i cannot",
            'error occurred',
            'failed to generate',
            'invalid request',
            'cannot create'
        ]
        prompt_lower = prompt.lower()
        
        # Only reject if these patterns appear near the start (first 100 chars)
        # This avoids false positives when these words appear in instructions
        prompt_start = prompt_lower[:100]
        for pattern in error_patterns:
            if pattern in prompt_start:
                return False
        
        # Also check if the entire response is just an error message (very short with error words)
        if len(prompt) < 100:
            simple_errors = ['sorry', 'error', 'cannot', 'unable', 'failed']
            if any(err in prompt_lower for err in simple_errors):
                return False
        
        return True
    
    def formulate_batch(self, personas: list[Dict[str, str]], 
                       show_progress: bool = True) -> list[str]:
        """
        Generate system prompts for multiple personas.
        
        Args:
            personas: List of persona feature dictionaries
            show_progress: Whether to show progress
            
        Returns:
            List of generated system prompts
        """
        prompts = []
        
        for i, persona in enumerate(personas):
            try:
                prompt = self.formulate(persona)
                prompts.append(prompt)
                
                if show_progress and (i + 1) % 10 == 0:
                    print(f"Generated prompts for {i + 1}/{len(personas)} personas")
            
            except Exception as e:
                print(f"Error generating prompt for persona {i}: {str(e)}")
                prompts.append("")  # Add empty string as placeholder
        
        return prompts
