"""
Pipeline Module

Main pipeline for generating synthetic persona data with system prompts.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .persona_bank import PersonaBank
from .sampling import PersonaSampler
from .persona_spec import PersonaSpec, PersonaSpecStorage, generate_persona_id
from .llm_client import create_llm_client, LLMFormulator
from .config_validation import validate_config


class PersonaGenerationPipeline:
    """
    Main pipeline for generating synthetic persona data.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        
        # Initialize components
        self.persona_bank = self._initialize_persona_bank()
        self.llm_client = self._initialize_llm_client()
        self.sampler = self._initialize_sampler()
        self.storage = self._initialize_storage()
        self.formulator = self._initialize_formulator()
        
        logging.info("Pipeline initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # Fail-fast validation (prevents mid-run crashes)
        cfg, _issues = validate_config(cfg, config_path=self.config_path)
        return cfg
    
    def _setup_logging(self):
        """
        Setup logging configuration.
        """
        log_dir = Path(self.config['paths']['logs']['dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_level = self.config['paths']['logs']['level']
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_persona_bank(self) -> PersonaBank:
        """
        Initialize PersonaBank.
        
        Returns:
            PersonaBank instance
        """
        persona_path = self.config['paths']['persona_bank']['persona']

        logging.info(f"Loading persona bank from {persona_path}")
        return PersonaBank(persona_path)
    
    def _initialize_sampler(self) -> PersonaSampler:
        """
        Initialize PersonaSampler.
        
        Returns:
            PersonaSampler instance
        """
        sampling_config_path = self.config['paths']['sampling']['config']

        persona_generation = self.config.get("persona_generation", {})
        overrides = {
            "strategy": persona_generation.get("sampling_strategy"),
            "feature_availability_rate": persona_generation.get("feature_availability_rate"),
            "diversity": persona_generation.get("diversity"),
            "seed": self.config.get("experiment", {}).get("seed"),
        }

        logging.info(f"Initializing sampler with config: {sampling_config_path}")
        return PersonaSampler(sampling_config_path, self.persona_bank, overrides=overrides, llm_client=self.llm_client)
    
    def _initialize_storage(self) -> PersonaSpecStorage:
        """
        Initialize PersonaSpecStorage.
        
        Returns:
            PersonaSpecStorage instance
        """
        output_dir = self.config['paths']['persona_specs']['output_dir']
        format = self.config['paths']['persona_specs']['format']
        
        logging.info(f"Initializing storage at {output_dir} with format {format}")
        return PersonaSpecStorage(output_dir, format)
    
    def _initialize_llm_client(self):
        """
        Initialize LLM client.
        
        Returns:
            LLMClient instance
        """
        api_config = self.config['api']
        
        logging.info(f"Initializing LLM client for provider: {api_config['provider']}")
        return create_llm_client(api_config)
    
    def _initialize_formulator(self) -> LLMFormulator:
        """
        Initialize LLMFormulator.
        
        Returns:
            LLMFormulator instance
        """
        template_path = self.config['formulation']['system_prompt_template']
        max_retries = self.config['formulation']['max_retries']
        retry_delay = self.config['formulation']['retry_delay']
        validate_output = self.config['formulation']['validate_output']
        min_prompt_length = self.config['formulation']['min_prompt_length']
        
        logging.info(f"Initializing formulator with template: {template_path}")
        return LLMFormulator(
            self.llm_client,
            template_path,
            max_retries,
            retry_delay,
            validate_output,
            min_prompt_length
        )
    
    def generate_persona(self, generate_prompt: bool = True) -> PersonaSpec:
        """
        Generate a single persona with optional system prompt.
        
        Args:
            generate_prompt: Whether to generate system prompt using LLM
            
        Returns:
            PersonaSpec instance
        """
        # Sample persona features
        features = self.sampler.sample_persona()
        
        # Generate persona ID
        persona_id = generate_persona_id(features)
        
        # Generate system prompt if requested
        system_prompt = None
        if generate_prompt:
            try:
                system_prompt = self.formulator.formulate(features)
                logging.info(f"Generated system prompt for persona {persona_id}")
            except Exception as e:
                logging.error(f"Failed to generate prompt for {persona_id}: {str(e)}")
        
        # Create PersonaSpec
        persona_spec = PersonaSpec(
            persona_id=persona_id,
            features=features,
            system_prompt=system_prompt,
            metadata={
                'num_features': len(features),
                'has_prompt': system_prompt is not None
            }
        )
        
        return persona_spec
    
    def generate_batch(self, num_personas: Optional[int] = None,
                      generate_prompts: bool = True,
                      save_to_disk: bool = True) -> List[PersonaSpec]:
        """
        Generate a batch of personas.
        
        Args:
            num_personas: Number of personas to generate (uses config if None)
            generate_prompts: Whether to generate system prompts
            save_to_disk: Whether to save to disk
            
        Returns:
            List of PersonaSpec instances
        """
        if num_personas is None:
            num_personas = self.config['persona_generation']['num_personas']
        
        logging.info(f"Starting batch generation of {num_personas} personas")
        
        personas = []
        
        for i in range(num_personas):
            try:
                persona = self.generate_persona(generate_prompt=generate_prompts)
                personas.append(persona)
                
                if save_to_disk:
                    self.storage.save(persona)
                
                if (i + 1) % 10 == 0:
                    logging.info(f"Generated {i + 1}/{num_personas} personas")
            
            except Exception as e:
                logging.error(f"Error generating persona {i + 1}: {str(e)}")
                continue
        
        logging.info(f"Batch generation complete. Generated {len(personas)} personas")
        
        return personas
    
    def run(self, num_personas: Optional[int] = None,
            generate_prompts: bool = True,
            export_dataset: bool = False,
            dataset_path: Optional[str] = None,
            export_all_stored: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            num_personas: Number of personas to generate
            generate_prompts: Whether to generate system prompts
            export_dataset: Whether to export to single dataset file
            dataset_path: Path for dataset export
            export_all_stored: Export all stored personas (historical) instead of only this run
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        logging.info("=" * 80)
        logging.info("Starting Persona Generation Pipeline")
        logging.info("=" * 80)
        
        # Generate personas
        personas = self.generate_batch(
            num_personas=num_personas,
            generate_prompts=generate_prompts,
            save_to_disk=True
        )
        
        # Export dataset if requested
        if export_dataset:
            if dataset_path is None:
                dataset_path = f"data/datasets/persona_dataset.json"
            
            Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
            if export_all_stored:
                self.storage.export_to_dataset(
                    dataset_path,
                    include_features=True,
                    include_prompts=generate_prompts,
                )
            else:
                self.storage.export_personas_to_dataset(
                    personas,
                    dataset_path,
                    include_features=True,
                    include_prompts=generate_prompts,
                )
            logging.info(f"Dataset exported to {dataset_path}")
        
        # Gather statistics
        stats = {
            'total_generated': len(personas),
            'with_prompts': sum(1 for p in personas if p.system_prompt is not None),
            'storage_stats': self.storage.get_statistics(),
            'config': {
                'num_personas': num_personas or self.config['persona_generation']['num_personas'],
                'sampling_strategy': self.sampler.config.get('strategy'),
                'llm_provider': self.config['api']['provider']
            }
        }
        
        logging.info("=" * 80)
        logging.info("Pipeline Complete")
        logging.info(f"Total personas generated: {stats['total_generated']}")
        logging.info(f"Personas with prompts: {stats['with_prompts']}")
        logging.info("=" * 80)
        
        return stats
    
    def reset(self):
        """
        Reset the pipeline state.
        """
        self.sampler.reset()
        logging.info("Pipeline state reset")
