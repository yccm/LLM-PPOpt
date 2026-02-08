"""
Enhanced Pipeline Module

Complete pipeline integrating persona generation, query generation,
interaction generation, distractor model, and training data collection.
"""

import io
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from .persona_bank import PersonaBank
from .sampling import PersonaSampler
from .persona_spec import PersonaSpec, PersonaSpecStorage, generate_persona_id
from .llm_client import create_llm_client, LLMFormulator
from .query_generator import UserQueryGenerator
from .query_storage import QueryStorage
from .interaction_generator import InteractionGenerator, InteractionStorage
from .distractor import DistractorModel, SemanticDistractorModel, create_distractor_model
from .training_data import TrainingDataCollector, TrainingDataExporter
from .config_validation import validate_config
from .colored_logger import ColoredLogger, print_colored
from tqdm import tqdm

# Import token tracker
try:
    from .token_tracker import get_tracker
except ImportError:
    def get_tracker():
        return None


class TeeOutput:
    """Capture stdout/stderr and write to both terminal and log file."""

    def __init__(self, original_stream, log_file, stream_name='stdout'):
        self.original_stream = original_stream
        self.log_file = log_file
        self.stream_name = stream_name

    def write(self, message):
        if message.strip():  # Only log non-empty messages
            try:
                self.original_stream.write(message)
                self.original_stream.flush()
            except UnicodeEncodeError:
                # Handle encoding issues on Windows
                self.original_stream.write(message.encode('utf-8', errors='replace').decode('ascii', errors='replace'))
                self.original_stream.flush()

            try:
                self.log_file.write(message)
                self.log_file.flush()
            except Exception:
                pass
        elif message == '\n':
            self.original_stream.write(message)
            self.log_file.write(message)

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return False


class EnhancedPersonaGenerationPipeline:
    """
    Complete pipeline for generating synthetic training data for prompt optimization.

    Stages:
    1. Persona Generation
    2. System Prompt Formulation
    3. Query Generation with Style Transfer
    4. Multi-turn Interaction Generation
    5. Distractor/Noise Application
    6. Training Data Collection and Export
    """

    def __init__(self, config_path: str):
        """
        Initialize the enhanced pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()

        # Track which stages to run
        self.stages = self.config.get('experiment', {}).get('stages', {})

        # Incremental mode
        self.incremental = self.config.get('experiment', {}).get('incremental', False)
        
        # Initialize token tracker
        self.token_tracker = get_tracker()
        if self.token_tracker:
            logging.info("Token tracking enabled")

        # Initialize LLM client first (needed by other components)
        self.llm_client = self._initialize_llm_client()

        # Initialize components
        if self.stages.get('persona_generation', True):
            self.persona_bank = self._initialize_persona_bank()
            self.sampler = self._initialize_sampler()
            self.storage = self._initialize_storage()
            self.formulator = self._initialize_formulator()

        if self.stages.get('query_generation', True):
            self.query_generator = self._initialize_query_generator()

        if self.stages.get('query_generation', True) or self.stages.get('interaction_generation', True):
            self.query_storage = self._initialize_query_storage()

        # Initialize distractor first (needed for interaction generation)
        if self.stages.get('interaction_generation', True) or self.stages.get('distractor_application', True):
            self.distractor = self._initialize_distractor()

        # Initialize training_collector first (needed by interaction_storage for format conversion)
        if self.stages.get('training_data_export', True) or self.stages.get('interaction_generation', True):
            self.training_collector = TrainingDataCollector(self.config)

        if self.stages.get('training_data_export', True):
            self.training_exporter = TrainingDataExporter(self.config)

        if self.stages.get('interaction_generation', True):
            self.interaction_generator = self._initialize_interaction_generator()
            # Initialize storage for incremental saving (pass collector for TrainingSample format)
            interactions_output_dir = self.config.get('paths', {}).get('interactions', {}).get(
                'output_dir', 'output/interactions'
            )
            collector = getattr(self, 'training_collector', None)
            self.interaction_storage = InteractionStorage(interactions_output_dir, collector=collector)

        logging.info("Enhanced pipeline initialized successfully")
        if self.incremental:
            logging.info("Incremental mode enabled: will skip stages with existing outputs")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # Fail-fast validation (prevents mid-run crashes)
        cfg, _issues = validate_config(cfg, config_path=self.config_path)
        return cfg
    
    def _setup_logging(self):
        """Setup logging configuration with full output capture."""
        log_dir = Path(self.config['paths']['logs']['dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        log_level = self.config['paths']['logs']['level']

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = log_dir / f'pipeline_{timestamp}.log'

        # Open log file with UTF-8 encoding
        self.log_file = open(log_file_path, 'w', encoding='utf-8')

        # Capture stdout and stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = TeeOutput(self._original_stdout, self.log_file, 'stdout')
        sys.stderr = TeeOutput(self._original_stderr, self.log_file, 'stderr')

        # Setup logging with custom handler
        class Utf8FileHandler(logging.FileHandler):
            def __init__(self, filename, mode='a', encoding='utf-8'):
                super().__init__(filename, mode, encoding)

        class SafeStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    stream = self.stream
                    try:
                        stream.write(msg + self.terminator)
                    except UnicodeEncodeError:
                        stream.write(msg.encode('utf-8', errors='replace').decode('ascii', errors='replace') + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)

        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers = []

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                Utf8FileHandler(log_file_path),
                SafeStreamHandler()
            ]
        )

        logging.info(f"Log file: {log_file_path}")
    
    def _initialize_persona_bank(self) -> PersonaBank:
        """Initialize PersonaBank."""
        persona_path = self.config['paths']['persona_bank']['persona']

        logging.info(f"Loading persona bank from {persona_path}")
        return PersonaBank(persona_path)
    
    def _initialize_sampler(self) -> PersonaSampler:
        """Initialize PersonaSampler."""
        sampling_config_path = self.config['paths']['sampling']['config']

        logging.info(f"Initializing sampler with config: {sampling_config_path}")
        return PersonaSampler(sampling_config_path, self.persona_bank, llm_client=self.llm_client)
    
    def _initialize_storage(self) -> PersonaSpecStorage:
        """Initialize PersonaSpecStorage."""
        output_dir = self.config['paths']['persona_specs']['output_dir']
        format = self.config['paths']['persona_specs']['format']
        
        logging.info(f"Initializing storage at {output_dir} with format {format}")
        return PersonaSpecStorage(output_dir, format)
    
    def _initialize_llm_client(self):
        """Initialize LLM client."""
        api_config = self.config['api']
        
        logging.info(f"Initializing LLM client for provider: {api_config['provider']}")
        return create_llm_client(api_config)
    
    def _initialize_formulator(self) -> LLMFormulator:
        """Initialize LLMFormulator."""
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
    
    def _initialize_query_generator(self) -> UserQueryGenerator:
        """Initialize UserQueryGenerator."""
        logging.info("Initializing query generator")
        return UserQueryGenerator(self.config, self.llm_client)

    def _initialize_query_storage(self) -> QueryStorage:
        """Initialize QueryStorage for Stage 2 persistence."""
        output_dir = self.config.get('paths', {}).get('queries', {}).get('output_dir', 'output/queries')
        logging.info(f"Initializing query storage at {output_dir}")
        return QueryStorage(output_dir)
    
    def _initialize_interaction_generator(self) -> InteractionGenerator:
        """Initialize InteractionGenerator with distractor integration."""
        logging.info("Initializing interaction generator")
        # Pass distractor to interaction generator for real-time noise injection
        distractor = self.distractor if hasattr(self, 'distractor') else None
        return InteractionGenerator(self.config, distractor=distractor)
    
    def _initialize_distractor(self):
        """Initialize DistractorModel or SemanticDistractorModel based on config."""
        logging.info("Initializing distractor model")
        use_semantic = self.config.get('distractor', {}).get('use_semantic', False)
        if use_semantic:
            logging.info("Using semantic distractor model")
        else:
            logging.info("Using legacy distractor model")
        return create_distractor_model(self.config, use_semantic=use_semantic, llm_client=self.llm_client)

    def _check_personas_exist(self, num_personas: int) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Check if enough personas already exist.

        Args:
            num_personas: Required number of personas

        Returns:
            Tuple of (has_enough, existing_personas)
        """
        if not hasattr(self, 'storage'):
            return False, []

        existing = self.storage.load_all()
        existing_dicts = [p.to_dict() for p in existing]

        if len(existing_dicts) >= num_personas:
            return True, existing_dicts[:num_personas]
        return False, existing_dicts

    def _check_training_data_exists(self) -> bool:
        """Check if training data files already exist."""
        output_dir = Path(self.config['paths']['training_data']['output_dir'])
        train_file = output_dir / "train_samples.json"
        val_file = output_dir / "val_samples.json"
        test_file = output_dir / "test_samples.json"

        return train_file.exists() and val_file.exists() and test_file.exists()

    def _load_existing_training_data(self) -> Dict[str, Any]:
        """Load existing training data statistics."""
        output_dir = Path(self.config['paths']['training_data']['output_dir'])
        stats_file = output_dir / "statistics.json"

        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            return {
                'total_samples': stats.get('total_samples', 0),
                'sample_files': {
                    'train': str(output_dir / "train_samples.json"),
                    'val': str(output_dir / "val_samples.json"),
                    'test': str(output_dir / "test_samples.json")
                },
                'stats_file': str(stats_file)
            }
        return None

    def _get_batch_size(self, num_personas: int) -> int:
        """Resolve batch size for streaming pipeline."""
        raw = self.config.get('experiment', {}).get('batch_size')
        try:
            batch_size = int(raw) if raw is not None else num_personas
        except Exception:
            batch_size = num_personas
        if batch_size <= 0:
            batch_size = num_personas
        return min(batch_size, num_personas)

    def _load_existing_queries(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load persisted Stage 2 queries from disk."""
        if not hasattr(self, 'query_storage'):
            return {}
        return self.query_storage.load_all()

    def _load_interaction_summary(self) -> tuple[Dict[str, int], Dict[str, set], set]:
        """
        Load interaction counts and used query_ids per persona (from storage index/files).

        Returns:
            success_count, used_query_ids_by_persona, personas_with_unknown_query_ids
        """
        success_count: Dict[str, int] = {}
        used_query_ids_by_persona: Dict[str, set] = {}
        unknown_query_id_personas: set = set()

        interactions_output_dir = self.config.get('paths', {}).get('interactions', {}).get(
            'output_dir', 'output/interactions'
        )
        storage = getattr(self, 'interaction_storage', None)
        if storage is None:
            storage = InteractionStorage(interactions_output_dir, collector=None)

        index = getattr(storage, 'index', None) or {'interactions': []}
        for entry in index.get('interactions', []):
            persona_id = entry.get('persona_id')
            if not persona_id:
                continue
            success_count[persona_id] = success_count.get(persona_id, 0) + 1
            query_id = entry.get('query_id')
            if not query_id and entry.get('interaction_id'):
                try:
                    data = storage.load(entry['interaction_id'])
                    query_id = (data.get('metadata') or {}).get('query_id')
                except Exception:
                    query_id = None
            if query_id:
                used_query_ids_by_persona.setdefault(persona_id, set()).add(query_id)
            else:
                unknown_query_id_personas.add(persona_id)

        return success_count, used_query_ids_by_persona, unknown_query_id_personas

    def _dedupe_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate queries based on query_id (fallback to text)."""
        seen = set()
        deduped = []
        for q in queries:
            if not isinstance(q, dict):
                continue
            qid = q.get('query_id')
            key = qid or (q.get('original_query'), q.get('adapted_query'))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(q)
        return deduped

    def run(self, num_personas: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            num_personas: Number of personas to generate

        Returns:
            Dictionary with pipeline results and statistics
        """
        print_colored("=" * 80, 'STAGE')
        print_colored("Starting Enhanced Persona Generation Pipeline", 'STAGE')
        if self.incremental:
            print_colored("Mode: INCREMENTAL (will skip existing outputs)", 'INFO')
        print_colored("=" * 80, 'STAGE')

        if num_personas is None:
            num_personas = self.config['persona_generation']['num_personas']

        results = {
            'personas': [],
            'queries': {},
            'interactions': []
        }

        batch_size = self._get_batch_size(num_personas)
        queries_per_persona = self.config['query_generation']['selection']['queries_per_persona']

        # Load existing personas (if any)
        existing_personas: List[Dict[str, Any]] = []
        if hasattr(self, 'storage'):
            existing_personas = [p.to_dict() for p in self.storage.load_all()]
        if existing_personas and len(existing_personas) > num_personas:
            existing_personas = existing_personas[:num_personas]

        # Load existing queries (Stage 2 persistence)
        if self.stages.get('query_generation', True) or self.stages.get('interaction_generation', True):
            results['queries'] = self._load_existing_queries()
        queries_map = results['queries']

        # Load interaction summary for resume/skip
        success_count: Dict[str, int] = {}
        used_query_ids_by_persona: Dict[str, set] = {}
        unknown_query_id_personas: set = set()
        if self.stages.get('interaction_generation', True):
            success_count, used_query_ids_by_persona, unknown_query_id_personas = self._load_interaction_summary()

        # Mark used query_ids in the dataset to avoid reuse across runs
        if hasattr(self, 'query_generator'):
            used_query_ids = set()
            for queries in queries_map.values():
                for q in queries:
                    qid = q.get('query_id')
                    if qid:
                        used_query_ids.add(qid)
            for qset in used_query_ids_by_persona.values():
                used_query_ids |= qset
            self.query_generator.mark_used_query_ids(used_query_ids)

        def iter_batches(items: List[Dict[str, Any]], size: int):
            for i in range(0, len(items), size):
                yield items[i:i + size], i // size + 1

        def process_batch(persona_batch: List[Dict[str, Any]], batch_label: str):
            if not persona_batch:
                return

            # Filter personas that already meet interaction quota
            pending = []
            for persona in persona_batch:
                pid = persona['persona_id']
                if self.stages.get('interaction_generation', True):
                    if success_count.get(pid, 0) >= queries_per_persona:
                        continue
                pending.append(persona)

            if not pending:
                return

            # Stage 2: Generate/restore queries for this batch
            batch_queries_map: Dict[str, List[Dict[str, Any]]] = {}
            if self.stages.get('query_generation', True) and pending:
                print_colored(f"\n[Stage 2] Generating Queries - {batch_label}...", 'STAGE')
            for persona in pending:
                pid = persona['persona_id']
                existing = queries_map.get(pid, [])

                if self.stages.get('interaction_generation', True):
                    target_needed = max(0, queries_per_persona - success_count.get(pid, 0))
                else:
                    target_needed = max(0, queries_per_persona - len(existing))

                if target_needed <= 0:
                    continue

                if pid in unknown_query_id_personas:
                    used_by_persona = {q.get('query_id') for q in existing if q.get('query_id')}
                else:
                    used_by_persona = used_query_ids_by_persona.get(pid, set())

                available = [q for q in existing if q.get('query_id') not in used_by_persona]
                selected = available[:target_needed]

                remaining = target_needed - len(selected)
                new_queries = []
                if remaining > 0 and self.stages.get('query_generation', True):
                    new_queries = self.query_generator.generate_queries_for_persona(
                        persona.get('features', {}),
                        num_queries=remaining
                    )
                    if new_queries:
                        if hasattr(self, 'query_storage'):
                            self.query_storage.append_persona(pid, new_queries)
                        queries_map[pid] = self._dedupe_queries(existing + new_queries)
                    else:
                        queries_map.setdefault(pid, existing)
                else:
                    queries_map.setdefault(pid, existing)

                if remaining > 0:
                    selected = available + new_queries

                if selected:
                    batch_queries_map[pid] = selected[:target_needed]

            # Stage 3: Interaction generation for this batch
            if self.stages.get('interaction_generation', True) and batch_queries_map:
                print_colored(f"\n[Stage 3] Generating Interactions - {batch_label}...", 'STAGE')
                interactions = self._run_interaction_generation(
                    pending,
                    batch_queries_map,
                    existing_success_count=success_count,
                    target_per_persona=queries_per_persona
                )
                results['interactions'].extend([inter.to_dict() for inter in interactions])
                print_colored(f"[OK] Generated {len(interactions)} interactions ({batch_label})", 'SUCCESS')

                for inter in interactions:
                    pid = inter.persona_id
                    success_count[pid] = success_count.get(pid, 0) + 1
                    qid = (inter.metadata or {}).get('query_id')
                    if qid:
                        used_query_ids_by_persona.setdefault(pid, set()).add(qid)

                if hasattr(self, 'distractor') and self.distractor and self.distractor.enabled:
                    print_colored("   (Distractor was applied in real-time during interaction)", 'INFO')

        # Stage 1: Persona Generation (streaming batches)
        remaining_to_generate = 0
        if self.stages.get('persona_generation', True):
            if self.incremental:
                if existing_personas:
                    if len(existing_personas) >= num_personas:
                        print_colored(
                            f"\n[Stage 1] Persona Generation - SKIPPED (found {len(existing_personas)} existing personas)",
                            'WARNING'
                        )
                    else:
                        print_colored(
                            f"\n[Stage 1] Found {len(existing_personas)} existing personas; will generate {num_personas - len(existing_personas)} more",
                            'INFO'
                        )
                remaining_to_generate = max(0, num_personas - len(existing_personas))
            else:
                existing_personas = []
                print_colored("\n[Stage 1] Generating Personas...", 'STAGE')
                remaining_to_generate = num_personas
        else:
            if existing_personas:
                print_colored(f"\n[Stage 1] Persona Generation - DISABLED (loaded {len(existing_personas)} personas)", 'WARNING')

        results['personas'] = existing_personas.copy()

        # Process existing personas first
        for batch, idx in iter_batches(existing_personas, batch_size):
            batch_label = f"batch {idx}"
            process_batch(batch, batch_label)

        # Generate and process remaining personas in batches
        generated_count = 0
        while remaining_to_generate > 0:
            this_batch = min(batch_size, remaining_to_generate)
            batch_label = f"new batch {generated_count // batch_size + 1}"
            print_colored(f"\n[Stage 1] Generating {this_batch} personas ({batch_label})...", 'STAGE')
            new_personas = self._run_persona_generation(this_batch)
            results['personas'].extend(new_personas)
            process_batch(new_personas, batch_label)
            remaining_to_generate -= this_batch
            generated_count += this_batch

        # Stage 4: Distractor Application (DEPRECATED - now integrated in Stage 3)
        if self.stages.get('distractor_application', True):
            print_colored("\n[Stage 4] Distractor Application - SKIPPED (already applied in Stage 3)", 'WARNING')
            results['enhanced_interactions'] = results['interactions']
        else:
            results['enhanced_interactions'] = results['interactions']

        # Stage 5: Training Data Export
        if self.stages.get('training_data_export', True):
            # Check if training data already exists (incremental mode)
            if self.incremental and self._check_training_data_exists():
                existing_data = self._load_existing_training_data()
                if existing_data:
                    print_colored(f"\n[Stage 5] Training Data Export - SKIPPED (found existing data: {existing_data['total_samples']} samples)", 'WARNING')
                    results['training_data'] = existing_data
                else:
                    print_colored("\n[Stage 5] Collecting and Exporting Training Data...", 'STAGE')
                    training_stats = self._run_training_data_export(results['enhanced_interactions'])
                    results['training_data'] = training_stats
                    print_colored(f"[OK] Exported training data", 'SUCCESS')
            else:
                print_colored("\n[Stage 5] Collecting and Exporting Training Data...", 'STAGE')
                training_stats = self._run_training_data_export(results['enhanced_interactions'])
                results['training_data'] = training_stats
                print_colored(f"[OK] Exported training data", 'SUCCESS')

        # Final statistics
        print_colored("\n" + "=" * 80, 'SUCCESS')
        print_colored("Pipeline Complete", 'SUCCESS')
        print_colored("=" * 80, 'SUCCESS')
        self._print_summary(results)
        
        # Export token usage statistics
        if self.token_tracker:
            self._export_token_statistics()

        # Cleanup logging
        self._cleanup_logging()

        return results

    def _cleanup_logging(self):
        """Restore stdout/stderr and close log file."""
        try:
            if hasattr(self, '_original_stdout'):
                sys.stdout = self._original_stdout
            if hasattr(self, '_original_stderr'):
                sys.stderr = self._original_stderr
            if hasattr(self, 'log_file') and self.log_file:
                self.log_file.close()
        except Exception:
            pass
    
    def _run_persona_generation(self, num_personas: Optional[int]) -> List[Dict[str, Any]]:
        """Run persona generation stage."""
        if num_personas is None:
            num_personas = self.config['persona_generation']['num_personas']
        
        personas = []
        
        # Use tqdm for progress bar
        for i in tqdm(range(num_personas), desc="Generating personas", 
                     unit="persona", ncols=100, colour='green'):
            # Sample features
            features = self.sampler.sample_persona()
            
            # Generate persona ID
            persona_id = generate_persona_id(features)
            
            # Generate system prompt
            try:
                system_prompt = self.formulator.formulate(features)
            except Exception as e:
                logging.error(f"Failed to generate prompt for {persona_id}: {str(e)}")
                system_prompt = None
            
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
            
            # Save
            self.storage.save(persona_spec)
            personas.append(persona_spec.to_dict())
        
        return personas
    
    def _run_query_generation(self, personas: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Run query generation stage."""
        queries_map = self.query_generator.generate_queries_batch(personas)
        return queries_map
    
    def _prepare_interaction_tasks(self, personas: List[Dict[str, Any]],
                                   queries_map: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Prepare persona-query task entries for interaction generation.

        For each query, the persona's domain/scenario are overwritten with the
        inferred values from the query to ensure consistency.
        """
        personas_with_queries = []

        for persona in personas:
            persona_id = persona['persona_id']
            if persona_id not in queries_map:
                continue
            queries = queries_map[persona_id]

            for query_data in queries:
                # Copy persona features and update domain/scenario from query
                updated_features = persona['features'].copy()

                inferred_domain = query_data.get('inferred_domain')
                inferred_scenario = query_data.get('inferred_scenario')

                if inferred_domain:
                    updated_features['domain'] = inferred_domain
                if inferred_scenario:
                    updated_features['scenario'] = inferred_scenario

                # Regenerate system prompt with updated features
                try:
                    updated_system_prompt = self.formulator.formulate(updated_features)
                except Exception as e:
                    logging.warning(f"Failed to regenerate prompt for {persona_id}, using original: {str(e)}")
                    updated_system_prompt = persona.get('system_prompt')

                personas_with_queries.append({
                    'persona_id': persona_id,
                    'persona_features': updated_features,
                    'system_prompt': updated_system_prompt,
                    'queries': [query_data]
                })

        return personas_with_queries

    def _run_interaction_generation(self, personas: List[Dict[str, Any]],
                                   queries_map: Dict[str, List[Dict[str, Any]]],
                                   existing_success_count: Optional[Dict[str, int]] = None,
                                   target_per_persona: Optional[int] = None) -> List[Any]:
        """
        Run interaction generation stage with supplement rounds.

        If some interactions fail, new queries are generated for deficient
        personas and additional rounds are attempted until every persona
        reaches queries_per_persona successful interactions (or the
        supplement budget is exhausted).
        """
        max_workers = self.config.get('interaction_generation', {}).get('max_workers', 5)
        storage = getattr(self, 'interaction_storage', None)
        queries_per_persona = target_per_persona or self.config['query_generation']['selection']['queries_per_persona']
        max_supplement_rounds = self.config.get('interaction_generation', {}).get('max_supplement_rounds', 3)

        all_interactions = []
        success_count: Dict[str, int] = {}
        for p in personas:
            pid = p['persona_id']
            base = 0
            if existing_success_count:
                base = existing_success_count.get(pid, 0)
            success_count[pid] = base
        current_queries_map = queries_map

        for round_num in range(max_supplement_rounds + 1):  # +1 for initial round
            # Prepare tasks
            personas_with_queries = self._prepare_interaction_tasks(personas, current_queries_map)

            if not personas_with_queries:
                break

            round_label = "" if round_num == 0 else f" (supplement round {round_num}/{max_supplement_rounds})"
            logging.info(f"Prepared {len(personas_with_queries)} persona-query pairs{round_label}")

            # Generate interactions
            interactions = self.interaction_generator.generate_interactions_batch(
                personas_with_queries,
                show_progress=True,
                max_workers=max_workers,
                storage=storage
            )
            all_interactions.extend(interactions)

            # Update success count
            for inter in interactions:
                success_count[inter.persona_id] = success_count.get(inter.persona_id, 0) + 1

            # Check which personas still need more
            deficient = []
            for persona in personas:
                pid = persona['persona_id']
                needed = queries_per_persona - success_count.get(pid, 0)
                if needed > 0:
                    deficient.append((persona, needed))

            if not deficient:
                break

            # Reached supplement budget
            if round_num >= max_supplement_rounds:
                for persona, needed in deficient:
                    print_colored(
                        f"Persona {persona['persona_id']} still needs {needed} more interactions "
                        f"after {max_supplement_rounds} supplement rounds",
                        'WARNING'
                    )
                break

            # Supplement: generate new queries for deficient personas
            if not hasattr(self, 'query_generator'):
                logging.warning("Cannot supplement queries: query_generator not available")
                break

            total_needed = sum(n for _, n in deficient)
            print_colored(
                f"Supplementing: {len(deficient)} personas need {total_needed} more interactions "
                f"(round {round_num + 1}/{max_supplement_rounds})...",
                'INFO'
            )

            current_queries_map = {}
            has_queries = False
            for persona, needed in deficient:
                pid = persona['persona_id']
                new_queries = self.query_generator.generate_queries_for_persona(
                    persona.get('features', {}),
                    num_queries=needed
                )
                if new_queries:
                    # Persist supplement queries
                    if hasattr(self, 'query_storage'):
                        try:
                            self.query_storage.append_persona(pid, new_queries)
                        except Exception as e:
                            logging.warning(f"Failed to persist supplement queries for {pid}: {str(e)}")
                    # Update global queries_map for downstream use
                    existing = queries_map.get(pid, [])
                    queries_map[pid] = self._dedupe_queries(existing + new_queries)

                    current_queries_map[pid] = new_queries[:needed]
                    has_queries = True
                else:
                    print_colored(f"No more queries available for persona {pid}", 'WARNING')

            if not has_queries:
                print_colored("No more queries available in dataset, stopping supplement", 'WARNING')
                break

        return all_interactions
    
    def _run_distractor(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run distractor application stage (DEPRECATED).
        
        Note: Distractor is now applied in real-time during interaction generation.
        This method is kept for backward compatibility.
        """
        logging.info("Distractor is now applied during interaction generation, skipping batch application.")
        return interactions
    
    def _run_training_data_export(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run training data export stage (no automatic labeling)."""
        # Collect training samples (just trajectories, no labeling)
        samples = self.training_collector.collect_from_interactions(interactions)
        
        # Export samples (all to train_samples.json with timestamp, no split)
        sample_files = self.training_exporter.export_samples(
            samples,
            split=False,
            use_timestamp=True
        )
        
        # Export statistics (with timestamp)
        stats_file = self.training_exporter.export_statistics(samples, use_timestamp=True)
        
        return {
            'total_samples': len(samples),
            'sample_files': sample_files,
            'stats_file': stats_file
        }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print pipeline summary."""
        print_colored("\nPIPELINE SUMMARY:", 'INFO')
        print_colored(f"  Personas Generated: {len(results.get('personas', []))}", 'INFO')
        print_colored(f"  Queries Generated: {sum(len(q) for q in results.get('queries', {}).values())}", 'INFO')
        print_colored(f"  Interactions Generated: {len(results.get('interactions', []))}", 'INFO')
        
        if 'training_data' in results:
            td = results['training_data']
            print_colored(f"  Training Samples: {td['total_samples']}", 'INFO')
        
        # Print token usage summary if tracker is available
        if self.token_tracker:
            print_colored("\nTOKEN USAGE SUMMARY:", 'INFO')
            stats = self.token_tracker.get_statistics()
            summary = stats['summary']
            print_colored(f"  Total API Calls: {summary['total_calls']}", 'INFO')
            print_colored(f"  Total Input Tokens: {summary['total_input_tokens']:,}", 'INFO')
            print_colored(f"  Total Output Tokens: {summary['total_output_tokens']:,}", 'INFO')
            print_colored(f"  Total Tokens: {summary['total_tokens']:,}", 'INFO')
    
    def _export_token_statistics(self):
        """Export token usage statistics to file."""
        if not self.token_tracker:
            return
        
        try:
            # Create output directory
            output_dir = Path(self.config['paths']['training_data']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            token_stats_file = output_dir / f'token_usage_{timestamp}.json'
            
            # Export statistics
            self.token_tracker.export_to_file(str(token_stats_file), include_records=True)
            
            print_colored(f"\n✓ Token usage statistics exported to: {token_stats_file}", 'SUCCESS')
            
            # Also print detailed summary to console
            print_colored("\n" + "=" * 80, 'INFO')
            self.token_tracker.print_summary()
            
        except Exception as e:
            logging.error(f"Failed to export token statistics: {str(e)}")
