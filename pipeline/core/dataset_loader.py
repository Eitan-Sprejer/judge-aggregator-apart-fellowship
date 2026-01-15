"""
Dataset Loading Pipeline

Loads and processes datasets for multi-judge interpretability experiments.
All datasets are preprocessed into a standardized format:

Core Fields:
    - question: The input/prompt/instruction (for dialogs: full conversation history)
    - response: The model's answer/completion
    - dataset: Source dataset name

Human Annotations:
    - target_human_aggregated: Dict of aggregated human scores by dimension (None if not available)
                               e.g., {"fluency": 2.0, "population": 1.5} for multi-dimensional
                               or {"overall": 7.5} for single dimension
    - target_human_individual: List[Dict] of individual annotator scores (None if not available)
                               e.g., [{"fluency": 2, "population": 2}, {"fluency": 1, "population": 2}]
    - score_range_human: Dict mapping dimension to (min, max) tuple (None if not available)
                         e.g., {"fluency": (0, 2), "overall": (1, 5)}

Synthetic Annotations:
    - target_synthetic: Dict of synthetic/persona scores by dimension (None if not available)
                        e.g., {"overall": 7.5} for single dimension
    - score_range_synthetic: Dict mapping dimension to (min, max) tuple (None if not available)
                            e.g., {"overall": (0, 10)}

Metadata Fields:
    - dimensions: List[str] of scoring dimensions, e.g., ["fluency", "coherence"] or ["overall"]
    - task_type: Optional[str] task category (e.g., "summarization", "dialog", "qa")
    - reference_output: Optional[str] gold reference for comparison tasks (e.g., gold translation)
    - context: Optional[List[Dict]] multi-turn dialog context (e.g., [{"speaker": "A", "utterance": "Hi"}])
    - response_metadata: Optional[Dict] metadata about response (model, system_id, etc.)
    - annotator_metadata: Optional[Dict] annotator information (agreement metrics, demographics)
    - original_index: Any - original index from source dataset for traceability

Supports: UltraFeedback, JUDGE-BENCH (separate dataset per task), MAJ-Eval, StorySparkQA, MSLR, HelpSteer2, and custom datasets.
"""

import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading and processing of evaluation datasets into standardized format."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize dataset loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir

    def load(
        self,
        dataset_name: str,
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load dataset and preprocess to standardized format.

        All datasets are preprocessed to have columns:
            Core: question, response, dataset
            Human: target_human_aggregated, target_human_individual, score_range_human
            Synthetic: target_synthetic, score_range_synthetic
            Metadata: dimensions, task_type, reference_output, context, response_metadata, annotator_metadata
            Traceability: original_index

        Args:
            dataset_name: Name of dataset ('ultrafeedback', 'judge_bench_*', 'maj_eval', 'story_spark_qa', 'mslr', 'helpsteer2')
                         Note: JUDGE-BENCH tasks are separate datasets (e.g., 'judge_bench_switchboard')
            use_cache: If True, load from datasets/processed/ cache if exists (default: True)
            **kwargs: Dataset-specific arguments

        Returns:
            DataFrame in standardized format
        """
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(dataset_name, **kwargs)
            if cache_path.exists():
                logger.info(f"Loading cached dataset from {cache_path}")
                df_full = pd.read_pickle(cache_path)

                # Apply sampling if n_samples specified
                n_samples = kwargs.get('n_samples')
                if n_samples is not None and n_samples < len(df_full):
                    random_seed = kwargs.get('random_seed', 42)
                    df = df_full.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
                    logger.info(f"Sampled {n_samples} from {len(df_full)} cached samples (seed={random_seed})")
                else:
                    df = df_full

                return df

        # Load from source (always loads FULL dataset)
        # Remove n_samples from kwargs so preprocessing methods load full data
        source_kwargs = {k: v for k, v in kwargs.items() if k != 'n_samples'}
        df_full = self._load_from_source(dataset_name, **source_kwargs)

        # Save FULL dataset to cache
        cache_path = self._get_cache_path(dataset_name, **kwargs)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_full.to_pickle(cache_path)
        logger.info(f"Cached full dataset ({len(df_full)} samples) to {cache_path}")

        # Apply sampling if n_samples specified
        n_samples = kwargs.get('n_samples')
        if n_samples is not None and n_samples < len(df_full):
            random_seed = kwargs.get('random_seed', 42)
            df = df_full.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
            logger.info(f"Sampled {n_samples} from {len(df_full)} samples (seed={random_seed})")
        else:
            df = df_full

        return df

    def _get_cache_path(self, dataset_name: str, **kwargs) -> Path:
        """Generate cache path for FULL dataset only.

        We cache only the full preprocessed dataset, never subsamples.
        Sampling happens at load time (cheap and deterministic with seed).

        Cache includes:
        - Dataset name
        - Split name (train/val/test have different data)
        """
        # Determine base name
        if dataset_name == 'judge_bench':
            name = kwargs.get('task_name', 'unknown')
        elif dataset_name == 'mslr':
            name = 'mslr_annotated'  # Always filtered version
        else:
            name = dataset_name

        # Include split if specified (different splits = different data)
        if 'split' in kwargs:
            cache_name = f"{name}_{kwargs['split']}_full.pkl"
        else:
            cache_name = f"{name}_full.pkl"

        return Path('datasets/processed') / cache_name

    def _load_from_source(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """Load dataset from original source (HuggingFace, files, etc.)."""
        logger.info(f"Loading {dataset_name} from source...")

        if dataset_name == 'ultrafeedback':
            return self._preprocess_ultrafeedback(**kwargs)
        elif dataset_name == 'judge_bench':
            return self._preprocess_judge_bench(**kwargs)
        elif dataset_name == 'maj_eval':
            return self._preprocess_maj_eval(**kwargs)
        elif dataset_name == 'story_spark_qa':
            return self._preprocess_story_spark_qa(**kwargs)
        elif dataset_name == 'mslr':
            return self._preprocess_mslr(**kwargs)
        elif dataset_name == 'helpsteer2':
            return self._preprocess_helpsteer2(**kwargs)
        elif dataset_name == 'custom':
            # For custom datasets, expect user to provide preprocessed data
            if 'data' not in kwargs:
                raise ValueError("Custom dataset requires 'data' argument with preprocessed DataFrame")
            return self._validate_standardized_format(kwargs['data'])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Supported: ultrafeedback, judge_bench, maj_eval, story_spark_qa, mslr, helpsteer2, custom")
    
    def _select_random_completion(self, completions: List[Dict]) -> Optional[Dict]:
        """
        Select a random completion to avoid bias.
        
        Args:
            completions: List of completion dictionaries
            
        Returns:
            Random completion or None if no completions
        """
        if not completions:
            return None
        
        # Randomly select a completion
        import random
        return random.choice(completions)
    
    def _preprocess_ultrafeedback(
        self,
        split: str = "train",
        n_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load UltraFeedback and preprocess to standardized format.

        UltraFeedback contains instruction-response pairs from various models.
        No human annotations available. Synthetic target scores will be filled
        by persona simulation later (0-10 scale).

        Args:
            split: Dataset split ("train" or "test")
            n_samples: Number of samples (None = all)
            random_seed: Random seed for sampling

        Returns:
            DataFrame in standardized format
        """
        logger.info(f"Loading UltraFeedback dataset (split: {split})")

        # Load dataset
        try:
            dataset = load_dataset("openbmb/UltraFeedback", split=split, cache_dir=self.cache_dir)
            logger.info(f"Loaded {len(dataset)} samples from UltraFeedback")
        except Exception as e:
            logger.error(f"Failed to load UltraFeedback: {e}")
            raise

        # Sample if requested
        if n_samples is not None and n_samples < len(dataset):
            logger.info(f"Sampling {n_samples} examples from {len(dataset)} total")
            dataset = dataset.shuffle(seed=random_seed).select(range(n_samples))

        # Process into standardized format
        processed_data = []
        for i, item in enumerate(dataset):
            try:
                # UltraFeedback format has instruction and completions
                question = item.get('instruction', '')

                # Get a random completion/response to avoid bias
                completions = item.get('completions', [])
                if not completions:
                    logger.warning(f"Sample {i} has no completions, skipping")
                    continue

                # Select random completion to avoid bias
                random_completion = self._select_random_completion(completions)
                response = random_completion.get('response', '') if random_completion else ''

                if not question or not response:
                    logger.warning(f"Sample {i} missing question or response, skipping")
                    continue

                # Standardized format
                processed_data.append({
                    # Core fields
                    'question': question,
                    'response': response,
                    'dataset': 'ultrafeedback',
                    # Human annotations
                    'target_human_aggregated': None,  # No human annotations in UltraFeedback
                    'target_human_individual': None,
                    'score_range_human': None,
                    # Synthetic annotations
                    'target_synthetic': None,  # Will be filled by persona simulation
                    'score_range_synthetic': {'overall': (0.0, 10.0)},  # Persona scores are 0-10
                    # Metadata
                    'dimensions': ['overall'],  # Persona simulation uses overall score
                    'task_type': 'instruction_following',
                    'reference_output': None,
                    'context': None,
                    'response_metadata': None,
                    'annotator_metadata': None,
                    # Traceability
                    'original_index': i
                })

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_data)} samples")

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df
    
    def _preprocess_judge_bench(
        self,
        task_name: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load JUDGE-BENCH task and preprocess to standardized format.

        JUDGE-BENCH contains 20 diverse NLP evaluation tasks with human annotations.
        Each task is treated as a SEPARATE dataset (e.g., 'judge_bench_summeval').

        Implemented tasks for fellowship experiments:
        - Summarization: SummEval (G-Eval), NewsRoom
        - Generation: Recipe Generation

        Planned tasks:
        - Dialog: Topical Chat, Persona Chat
        - Translation: WMT (all language pairs)

        Args:
            task_name: Name of JUDGE-BENCH task to load (e.g., 'summeval', 'newsroom', 'recipe')
            **kwargs: Additional task-specific arguments (n_samples, random_seed)

        Returns:
            DataFrame in standardized format with task-specific fields:
            - Dialog tasks: context field populated with conversation history
            - Translation tasks: reference_output field populated with gold translation
            - All tasks: dimensions field reflects task-specific evaluation criteria
        """
        # Dispatcher to task-specific loaders
        if task_name == 'summeval':
            return self._preprocess_judge_bench_summeval(**kwargs)
        elif task_name == 'newsroom':
            return self._preprocess_judge_bench_newsroom(**kwargs)
        elif task_name == 'recipe':
            return self._preprocess_judge_bench_recipe(**kwargs)
        elif task_name == 'wmt_en_de':
            return self._preprocess_judge_bench_wmt(language_pair='en_de', **kwargs)
        elif task_name == 'wmt_zh_en':
            return self._preprocess_judge_bench_wmt(language_pair='zh_en', **kwargs)
        elif task_name == 'topical_chat':
            return self._preprocess_judge_bench_topical_chat(**kwargs)
        else:
            raise NotImplementedError(
                f"JUDGE-BENCH task '{task_name}' not yet implemented. "
                f"Available tasks: summeval, newsroom, recipe, wmt_en_de, wmt_zh_en. "
                f"Planned tasks: topical_chat, persona_chat."
            )

    def _preprocess_judge_bench_summeval(
        self,
        n_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load SummEval (G-Eval) dataset and preprocess to standardized format.

        Uses the REAL individual annotator scores from Yale-LILY/SummEval:
        - 3 expert annotators + 5 crowdworker annotators = 8 total per instance
        - Merges with Judge-Bench version to get source articles and references

        SummEval evaluates news article summarization with 4 graded dimensions:
        - coherence (1-5): Collective quality and organization
        - consistency (1-5): Factual alignment with source
        - fluency (1-5): Grammar and readability
        - relevance (1-5): Selection of important content

        Args:
            n_samples: Number of samples (None = all)
            random_seed: Random seed for sampling

        Returns:
            DataFrame in standardized format with 8 individual annotator scores per instance
        """
        import json

        logger.info("Loading SummEval with real individual annotator scores (8 per instance)")

        # Load REAL individual annotations from Yale-LILY (3 experts + 5 turkers)
        real_annotations_file = "datasets/judge-bench/summeval/original_data/model_annotations.aligned.jsonl"
        with open(real_annotations_file, 'r') as f:
            real_annotations = [json.loads(line) for line in f]

        # Create lookup by (doc_id, model_id)
        annotations_lookup = {}
        for item in real_annotations:
            key = (item['id'], item['model_id'])
            annotations_lookup[key] = item

        logger.info(f"Loaded {len(annotations_lookup)} instances with real individual annotations")

        # Load Judge-Bench version for source articles and references
        jb_file = "datasets/judge-bench/summeval/original_data/summeval.json"
        with open(jb_file, 'r') as f:
            jb_data = json.load(f)

        logger.info(f"Loaded {len(jb_data)} instances from Judge-Bench (for source/reference)")

        # Sample if requested
        if n_samples is not None and n_samples < len(jb_data):
            logger.info(f"Sampling {n_samples} examples from {len(jb_data)} total")
            random.seed(random_seed)
            jb_data = random.sample(jb_data, n_samples)

        # Process into standardized format
        DIMENSIONS = ['coherence', 'consistency', 'fluency', 'relevance']
        SCORE_RANGES = {
            'coherence': (1, 5),
            'consistency': (1, 5),
            'fluency': (1, 5),
            'relevance': (1, 5)
        }

        processed_data = []
        for idx, jb_instance in enumerate(jb_data):
            try:
                # Extract source/reference from Judge-Bench version
                source = jb_instance.get('source', '')
                summary = jb_instance.get('system_output', '')
                reference = jb_instance.get('reference', '')
                doc_id = jb_instance.get('doc_id')
                system_id = jb_instance.get('system_id')

                if not source or not summary:
                    logger.warning(f"Instance {idx} missing source or summary, skipping")
                    continue

                # Look up real individual annotations
                key = (doc_id, system_id)
                if key not in annotations_lookup:
                    logger.warning(f"Instance {idx} ({key}) not found in real annotations, skipping")
                    continue

                real_annot = annotations_lookup[key]

                # Combine expert and turker annotations (3 + 5 = 8 annotators)
                expert_annots = real_annot.get('expert_annotations', [])
                turker_annots = real_annot.get('turker_annotations', [])
                all_annotators = expert_annots + turker_annots

                if len(all_annotators) != 8:
                    logger.warning(f"Instance {idx} has {len(all_annotators)} annotations instead of 8")

                # Build individual annotator scores (list of dicts, one per annotator)
                individual = []
                for annot in all_annotators:
                    annotator_dict = {
                        dim: annot.get(dim)
                        for dim in DIMENSIONS
                        if annot.get(dim) is not None
                    }
                    if annotator_dict:
                        individual.append(annotator_dict)

                # Compute aggregated mean across all 8 annotators
                aggregated = {}
                for dim in DIMENSIONS:
                    scores = [annot[dim] for annot in individual if dim in annot]
                    if scores:
                        aggregated[dim] = sum(scores) / len(scores)

                processed_data.append({
                    # Core fields
                    'question': source,  # Source article
                    'response': summary,  # Generated summary
                    'dataset': 'summeval',
                    # Human annotations (8 real individual annotators!)
                    'target_human_aggregated': aggregated if aggregated else None,
                    'target_human_individual': individual if individual else None,
                    'score_range_human': SCORE_RANGES,
                    # Synthetic annotations
                    'target_synthetic': None,
                    'score_range_synthetic': None,
                    # Metadata
                    'dimensions': DIMENSIONS,
                    'task_type': 'summarization',
                    'reference_output': reference,
                    'context': None,
                    'response_metadata': {
                        'g_eval': True,
                        'system_id': system_id,
                        'expert_annotators': len(expert_annots),
                        'crowdworker_annotators': len(turker_annots)
                    },
                    'annotator_metadata': {
                        'num_annotators': len(individual),
                        'num_experts': len(expert_annots),
                        'num_crowdworkers': len(turker_annots)
                    },
                    # Traceability
                    'original_index': f"{doc_id}_{system_id}"
                })

            except Exception as e:
                logger.warning(f"Error processing instance {idx}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_data)} samples with 8 individual annotators each")

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df

    def _preprocess_judge_bench_newsroom(
        self,
        n_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load NewsRoom dataset and preprocess to standardized format.

        NewsRoom evaluates news article summarization with 4 graded dimensions:
        - Informativeness (1-5): Captures key points
        - Relevance (1-5): Consistent with article details
        - Fluency (1-5): Grammar and sentence quality
        - Coherence (1-5): Sentences fit together

        Args:
            n_samples: Number of samples (None = all)
            random_seed: Random seed for sampling

        Returns:
            DataFrame in standardized format
        """
        import json
        import re

        logger.info("Loading NewsRoom dataset from Judge-Bench")

        # Load JSON data
        data_file = "datasets/judge-bench/newsroom/newsroom.json"
        with open(data_file, 'r') as f:
            data = json.load(f)

        instances = data['instances']
        logger.info(f"Loaded {len(instances)} instances from NewsRoom")

        # Sample if requested
        if n_samples is not None and n_samples < len(instances):
            logger.info(f"Sampling {n_samples} examples from {len(instances)} total")
            random.seed(random_seed)
            instances = random.sample(instances, n_samples)

        # Process into standardized format
        DIMENSIONS = ['Informativeness', 'Relevance', 'Fluency', 'Coherence']
        SCORE_RANGES = {
            'Informativeness': (1, 5),
            'Relevance': (1, 5),
            'Fluency': (1, 5),
            'Coherence': (1, 5)
        }

        processed_data = []
        for idx, instance in enumerate(instances):
            try:
                # Parse the instance field which contains both summary and source
                instance_text = instance.get('instance', '')
                annotations = instance.get('annotations', {})

                # Extract summary and source from formatted text
                # Format: "### Generated Summary\n\n{summary}\n\n### Source Article\n\n{source}"
                summary_match = re.search(r'### Generated Summary\n\n(.*?)\n\n### Source Article', instance_text, re.DOTALL)
                source_match = re.search(r'### Source Article\n\n(.*)', instance_text, re.DOTALL)

                if summary_match and source_match:
                    summary = summary_match.group(1).strip()
                    source = source_match.group(1).strip()
                else:
                    logger.warning(f"Instance {idx} couldn't parse summary/source, skipping")
                    continue

                # Build aggregated scores
                aggregated = {}
                individual_scores = {}

                for dim in DIMENSIONS:
                    if dim in annotations:
                        aggregated[dim] = annotations[dim].get('mean_human')
                        individual_scores[dim] = annotations[dim].get('individual_human_scores', [])

                # Convert individual scores from per-dimension lists to per-annotator dicts
                individual = None
                if individual_scores:
                    # Get the maximum number of annotators across all dimensions
                    max_annotators = max(len(scores) for scores in individual_scores.values())

                    # Build per-annotator dictionaries
                    individual = []
                    for annotator_idx in range(max_annotators):
                        annotator_dict = {}
                        for dim in DIMENSIONS:
                            if dim in individual_scores and annotator_idx < len(individual_scores[dim]):
                                annotator_dict[dim] = individual_scores[dim][annotator_idx]
                        if annotator_dict:  # Only add if has at least one score
                            individual.append(annotator_dict)

                processed_data.append({
                    # Core fields
                    'question': source,  # Source article
                    'response': summary,  # Generated summary
                    'dataset': 'newsroom',
                    # Human annotations
                    'target_human_aggregated': aggregated if aggregated else None,
                    'target_human_individual': individual if individual else None,
                    'score_range_human': SCORE_RANGES,
                    # Synthetic annotations
                    'target_synthetic': None,
                    'score_range_synthetic': None,
                    # Metadata
                    'dimensions': DIMENSIONS,
                    'task_type': 'summarization',
                    'reference_output': None,  # NewsRoom doesn't have reference summaries
                    'context': None,
                    'response_metadata': {'crowdsourced': True},
                    'annotator_metadata': {'num_annotators': len(individual)} if individual else None,
                    # Traceability
                    'original_index': instance.get('id', idx)
                })

            except Exception as e:
                logger.warning(f"Error processing instance {idx}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_data)} samples from NewsRoom")

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df

    def _preprocess_judge_bench_recipe(
        self,
        n_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load Recipe Generation dataset and preprocess to standardized format.

        Recipe evaluates cooking recipe generation with 6 graded dimensions:
        - grammar (1-6): Grammatical correctness
        - fluency (1-6): Reading smoothness
        - verbosity (1-6): Conciseness
        - structure (1-6): Helpful ordering of steps
        - success (1-6): Recipe would enable successful preparation
        - overall (1-6): Overall recipe quality

        Args:
            n_samples: Number of samples (None = all)
            random_seed: Random seed for sampling

        Returns:
            DataFrame in standardized format
        """
        import json

        logger.info("Loading Recipe Generation dataset from Judge-Bench")

        # Load JSON data
        data_file = "datasets/judge-bench/recipe_crowd_sourcing_data/meta_evaluation_recipes.json"
        with open(data_file, 'r') as f:
            data = json.load(f)

        instances = data['instances']
        logger.info(f"Loaded {len(instances)} instances from Recipe Generation")

        # Sample if requested
        if n_samples is not None and n_samples < len(instances):
            logger.info(f"Sampling {n_samples} examples from {len(instances)} total")
            random.seed(random_seed)
            instances = random.sample(instances, n_samples)

        # Process into standardized format
        DIMENSIONS = ['grammar', 'fluency', 'verbosity', 'structure', 'success', 'overall']
        SCORE_RANGES = {dim: (1, 6) for dim in DIMENSIONS}

        processed_data = []
        for idx, instance in enumerate(instances):
            try:
                # Extract fields
                recipe_text = instance.get('instance', '')
                annotations = instance.get('annotations', {})

                if not recipe_text:
                    logger.warning(f"Instance {idx} missing recipe text, skipping")
                    continue

                # Build aggregated scores
                aggregated = {}
                individual_scores = {}

                for dim in DIMENSIONS:
                    if dim in annotations:
                        aggregated[dim] = annotations[dim].get('mean_human')
                        individual_scores[dim] = annotations[dim].get('individual_human_scores', [])

                # Convert individual scores from per-dimension lists to per-annotator dicts
                individual = None
                if individual_scores:
                    # Get the maximum number of annotators across all dimensions
                    max_annotators = max(len(scores) for scores in individual_scores.values())

                    # Build per-annotator dictionaries
                    individual = []
                    for annotator_idx in range(max_annotators):
                        annotator_dict = {}
                        for dim in DIMENSIONS:
                            if dim in individual_scores and annotator_idx < len(individual_scores[dim]):
                                annotator_dict[dim] = individual_scores[dim][annotator_idx]
                        if annotator_dict:  # Only add if has at least one score
                            individual.append(annotator_dict)

                processed_data.append({
                    # Core fields
                    'question': "Evaluate this recipe:",  # Standard prompt since recipes are standalone
                    'response': recipe_text,  # The recipe text
                    'dataset': 'recipe',
                    # Human annotations
                    'target_human_aggregated': aggregated if aggregated else None,
                    'target_human_individual': individual if individual else None,
                    'score_range_human': SCORE_RANGES,
                    # Synthetic annotations
                    'target_synthetic': None,
                    'score_range_synthetic': None,
                    # Metadata
                    'dimensions': DIMENSIONS,
                    'task_type': 'recipe_generation',
                    'reference_output': None,
                    'context': None,
                    'response_metadata': {'crowdsourced': True},
                    'annotator_metadata': {'num_annotators': len(individual)} if individual else None,
                    # Traceability
                    'original_index': instance.get('id', idx)
                })

            except Exception as e:
                logger.warning(f"Error processing instance {idx}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_data)} samples from Recipe Generation")

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df

    def _preprocess_judge_bench_topical_chat(
        self,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load Topical Chat dataset and preprocess to standardized format.

        Topical Chat requires conversion from original data first.
        Run: cd datasets/judge-bench/topical_chat && python convert_topical_chat.py --prompt short

        Note:
            Not yet implemented - requires data conversion.
        """
        raise NotImplementedError(
            "Topical Chat loader not yet implemented. "
            "First run data conversion: cd datasets/judge-bench/topical_chat && python convert_topical_chat.py --prompt short"
        )

    def _preprocess_judge_bench_wmt(
        self,
        language_pair: str,
        n_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load WMT-20 machine translation quality dataset and preprocess to standardized format.

        WMT-20 evaluates machine translation quality with expert annotations:
        - quality (0-6): Overall translation quality
          - 0: Nonsense/No meaning preserved
          - 2: Some meaning preserved
          - 4: Most meaning preserved, few grammar mistakes
          - 6: Perfect meaning and grammar

        Typically 3 expert annotators per instance (99%+ of data).

        Args:
            language_pair: 'en_de' (English→German) or 'zh_en' (Chinese→English)
            n_samples: Number of samples (None = all)
            random_seed: Random seed for sampling

        Returns:
            DataFrame in standardized format

        Dataset sizes:
            - en_de: 9,871 instances
            - zh_en: 15,981 instances
        """
        import json

        logger.info(f"Loading WMT-20 translation quality dataset ({language_pair})")

        # Load JSON data
        data_file = f"datasets/judge-bench/wmt-human/wmt-human_{language_pair}.json"
        with open(data_file, 'r') as f:
            data = json.load(f)

        instances = data['instances']
        logger.info(f"Loaded {len(instances)} instances from WMT-20 {language_pair}")

        # Sample if requested
        if n_samples is not None and n_samples < len(instances):
            logger.info(f"Sampling {n_samples} examples from {len(instances)} total")
            random.seed(random_seed)
            instances = random.sample(instances, n_samples)

        # Process into standardized format
        DIMENSIONS = ['quality']
        SCORE_RANGES = {'quality': (0, 6)}

        # Map language pair to source/target languages
        lang_mapping = {
            'en_de': {'source_lang': 'English', 'target_lang': 'German'},
            'zh_en': {'source_lang': 'Chinese', 'target_lang': 'English'}
        }
        langs = lang_mapping.get(language_pair, {'source_lang': 'unknown', 'target_lang': 'unknown'})

        processed_data = []
        for idx, instance in enumerate(instances):
            try:
                # Extract fields
                instance_data = instance.get('instance', {})
                source = instance_data.get('source', '')
                translation = instance_data.get('translation', '')
                reference = instance_data.get('reference', '')
                annotations = instance.get('annotations', {})

                if not source or not translation:
                    logger.warning(f"Instance {idx} missing source or translation, skipping")
                    continue

                # Extract quality scores
                quality_annot = annotations.get('quality', {})
                mean_score = quality_annot.get('mean_human')
                individual_scores = quality_annot.get('individual_human_scores', [])

                if not individual_scores:
                    logger.warning(f"Instance {idx} has no individual scores, skipping")
                    continue

                # Build aggregated and individual scores
                aggregated = {'quality': mean_score} if mean_score is not None else None

                # Individual annotator scores
                individual = [{'quality': score} for score in individual_scores if score is not None]

                processed_data.append({
                    # Core fields
                    'question': source,  # Source text
                    'response': translation,  # Machine translation
                    'dataset': f'wmt_{language_pair}',
                    # Human annotations
                    'target_human_aggregated': aggregated,
                    'target_human_individual': individual if individual else None,
                    'score_range_human': SCORE_RANGES,
                    # Synthetic annotations
                    'target_synthetic': None,
                    'score_range_synthetic': None,
                    # Metadata
                    'dimensions': DIMENSIONS,
                    'task_type': 'translation',
                    'reference_output': reference,  # Human reference translation
                    'context': {
                        'source_language': langs['source_lang'],
                        'target_language': langs['target_lang'],
                        'language_pair': language_pair
                    },
                    'response_metadata': {
                        'wmt_20': True,
                        'expert_annotators': len(individual)
                    },
                    'annotator_metadata': {
                        'num_annotators': len(individual),
                        'expert_annotators': True
                    },
                    # Traceability
                    'original_index': instance.get('id', idx)
                })

            except Exception as e:
                logger.warning(f"Error processing instance {idx}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_data)} samples from WMT-20 {language_pair}")

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df

    def _preprocess_maj_eval(
        self,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load MAJ-Eval data and preprocess to standardized format.

        MAJ-Eval contains multi-agent debate evaluation data.

        Args:
            **kwargs: Dataset-specific arguments

        Returns:
            DataFrame in standardized format

        Note:
            TODO: Implement MAJ-Eval loader when ready to run Track 1.2 experiments.
            We have their code, need to adapt their data format to our standardized format.
        """
        raise NotImplementedError(
            "MAJ-Eval loader not yet implemented. "
            "This will be added when starting Track 1.2 experiments."
        )

    def _preprocess_story_spark_qa(
        self,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load StorySparkQA and preprocess to standardized format.

        StorySparkQA is a long-form QA dataset that lacks human annotations.
        Not implemented for fellowship experiments.

        Note:
            Not implemented - lacks human annotations required for experiments.
        """
        raise NotImplementedError(
            "StorySparkQA loader not implemented. "
            "Dataset lacks human annotations required for fellowship experiments."
        )

    def _load_mslr_source_docs(self) -> dict:
        """Load source documents from local CSV files for MSLR reviews.

        Reads from datasets/mslr-source/mslr_data/cochrane/*.csv files.
        These contain the study titles and abstracts for each Cochrane review.

        Returns:
            Dict mapping review_id -> {'titles': [...], 'abstracts': [...]}
        """
        from pathlib import Path

        # Calculate path to source data
        project_root = Path(__file__).parent.parent.parent
        source_dir = project_root / "datasets" / "mslr-source" / "mslr_data" / "cochrane"

        if not source_dir.exists():
            logger.warning(
                f"MSLR source documents not found at {source_dir}. "
                "Download from: https://ai2-s2-mslr.s3.us-west-2.amazonaws.com/mslr_data.tar.gz"
            )
            return {}

        logger.info(f"Loading source documents from {source_dir}...")

        try:
            # Load all CSV files and combine
            all_data = []
            for split in ['train', 'dev', 'test']:
                csv_path = source_dir / f"{split}-inputs.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    all_data.append(df)
                    logger.info(f"  Loaded {len(df)} rows from {split}-inputs.csv")

            if not all_data:
                logger.warning("No CSV files found in source directory")
                return {}

            # Combine all splits
            combined = pd.concat(all_data, ignore_index=True)

            # Group by ReviewID and collect titles/abstracts
            source_docs = {}
            for review_id, group in combined.groupby('ReviewID'):
                source_docs[review_id] = {
                    'titles': group['Title'].dropna().tolist(),
                    'abstracts': group['Abstract'].dropna().tolist()
                }

            logger.info(f"Loaded source documents for {len(source_docs)} reviews")
            return source_docs

        except Exception as e:
            logger.warning(f"Failed to load source documents: {e}")
            return {}

    def _build_mslr_question(self, review_id: str, target: str, source_docs: dict) -> str:
        """Build question with source documents and reference summary."""
        parts = [f"Cochrane Review {review_id}"]

        if source_docs and source_docs.get('abstracts'):
            parts.append("\n\nSource Documents (to be summarized):")
            for i, (title, abstract) in enumerate(zip(source_docs['titles'], source_docs['abstracts']), 1):
                parts.append(f"\n[Document {i}]\nTitle: {title}\nAbstract: {abstract}")

        parts.append(f"\n\nReference Summary (gold standard):\n{target}")
        return "".join(parts)

    def _process_mslr_annotations(self, annotations: list) -> tuple:
        """Process raw MSLR annotations into MAJ-Eval format. Returns (aggregated, individual, raw)."""
        CORE_DIMS = ['fluency', 'population', 'intervention', 'outcome']

        # Extract valid raw annotations
        raw_annotations = []
        for annot in annotations:
            if all(annot.get(d) is not None for d in CORE_DIMS):
                raw_annotations.append({k: v for k, v in annot.items() if v is not None})

        if not raw_annotations:
            return None, None, None

        # Compute MAJ-Eval dimensions
        aggregated = {}
        individual = []

        for raw in raw_annotations:
            # Compute MAJ-Eval dimensions for this annotator
            maj_eval = {
                'fluency': raw['fluency'],
                'pio_consistency': (raw['population'] + raw['intervention'] + raw['outcome']) / 3
            }
            if 'ed_agree' in raw:
                maj_eval['effect_direction'] = float(raw['ed_agree'])
            if 'strength_agree' in raw:
                maj_eval['evidence_strength'] = float(raw['strength_agree'])

            individual.append(maj_eval)

        # Average across annotators
        for dim in individual[0].keys():
            aggregated[dim] = sum(annot[dim] for annot in individual) / len(individual)

        return aggregated, individual, raw_annotations

    def _preprocess_mslr(
        self,
        n_samples: Optional[int] = None,
        random_seed: int = 42,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load MSLR annotated dataset and preprocess to standardized format.

        MSLR (Medical Summarization for Literature Reviews) contains human-annotated
        quality judgments for medical summary outputs. Only ~600 samples have annotations
        (100 per model × 6 models) - this loader filters for annotated samples only.

        Dimensions (MAJ-Eval format):
        - fluency: 0-2 (no errors → major errors)
        - pio_consistency: 0-2 (average of Population, Intervention, Outcome alignment)
        - effect_direction: 0-1 (whether intervention effect direction matches)
        - evidence_strength: 0-1 (whether claim strength matches evidence)

        Args:
            n_samples: Number of samples to return (None = all annotated samples, ~600)
            random_seed: Random seed for sampling

        Returns:
            DataFrame in standardized 15-column format
        """
        import json
        import urllib.request
        import os
        from pathlib import Path

        logger.info("Loading MSLR annotated dataset")

        # Define paths - calculate project root from this file's location
        # This file is at: project/pipeline/core/dataset_loader.py
        project_root = Path(__file__).parent.parent.parent
        local_path = project_root / "datasets" / "mslr-annotated" / "data" / "data_with_overlap_scores.json"
        github_url = "https://raw.githubusercontent.com/allenai/mslr-annotated-dataset/main/data/data_with_overlap_scores.json"

        # Auto-download if not present
        if not local_path.exists():
            logger.info(f"Downloading MSLR annotated data from GitHub...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(github_url, str(local_path))
                logger.info(f"Downloaded to {local_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download MSLR data from {github_url}: {e}")

        # Load annotated JSON
        logger.info(f"Loading annotated data from {local_path}")
        with open(local_path, 'r', encoding='utf-8') as f:
            # File is JSON Lines format (one JSON object per line)
            data = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(data)} reviews from MSLR")

        # Load source documents from HuggingFace (for building questions)
        source_docs = self._load_mslr_source_docs()

        # Define dimensions and score ranges
        DIMENSIONS = ['fluency', 'pio_consistency', 'effect_direction', 'evidence_strength']
        SCORE_RANGE = {
            'fluency': (0, 2),
            'pio_consistency': (0, 2),
            'effect_direction': (0, 1),
            'evidence_strength': (0, 1)
        }

        # Process each review and its predictions
        processed_data = []
        total_predictions = 0
        annotated_predictions = 0

        for review in data:
            review_id = review.get('review_id', '')
            target = review.get('target', '')  # Reference summary

            # Get source docs for this review
            review_source_docs = source_docs.get(review_id, {})

            # Process each model prediction
            for pred in review.get('predictions', []):
                total_predictions += 1
                model_id = pred.get('exp_short', 'unknown')
                prediction_text = pred.get('prediction', '')
                annotations = pred.get('annotations', [])

                # Skip if no annotations
                if not annotations:
                    continue

                # Process annotations using existing helper
                aggregated, individual, raw = self._process_mslr_annotations(annotations)

                if aggregated is None:
                    continue

                annotated_predictions += 1

                # Build question with source docs and reference
                question = self._build_mslr_question(review_id, target, review_source_docs)

                # Create standardized record
                processed_data.append({
                    # Core fields
                    'question': question,
                    'response': prediction_text,
                    'dataset': 'mslr',
                    # Human annotations
                    'target_human_aggregated': aggregated,
                    'target_human_individual': individual,
                    'score_range_human': SCORE_RANGE,
                    # Synthetic annotations
                    'target_synthetic': None,
                    'score_range_synthetic': None,
                    # Metadata
                    'dimensions': DIMENSIONS,
                    'task_type': 'summarization',
                    'reference_output': target,
                    'context': None,
                    'response_metadata': {'model_id': model_id, 'review_id': review_id},
                    'annotator_metadata': {'num_annotators': len(annotations), 'raw_annotations': raw},
                    # Traceability
                    'original_index': f"{review_id}_{model_id}"
                })

        logger.info(f"Total predictions: {total_predictions}, Annotated: {annotated_predictions}")
        logger.info(f"Successfully processed {len(processed_data)} annotated samples")

        # Sample if requested
        if n_samples is not None and n_samples < len(processed_data):
            random.seed(random_seed)
            processed_data = random.sample(processed_data, n_samples)
            logger.info(f"Sampled {n_samples} samples")

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df

    def _preprocess_helpsteer2(
        self,
        split: str = "train",
        n_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load HelpSteer2 and preprocess to standardized format.

        HelpSteer2 is NVIDIA's open-source helpfulness dataset with 5-dimensional
        human annotations (0-4 scale): helpfulness, correctness, coherence,
        complexity, verbosity.

        Args:
            split: Dataset split ("train" or "validation")
            n_samples: Number of samples (None = all)
            random_seed: Random seed for sampling

        Returns:
            DataFrame in standardized format
        """
        logger.info(f"Loading HelpSteer2 dataset (split: {split})")

        # Load dataset
        try:
            dataset = load_dataset("nvidia/HelpSteer2", split=split, cache_dir=self.cache_dir)
            logger.info(f"Loaded {len(dataset)} samples from HelpSteer2")
        except Exception as e:
            logger.error(f"Failed to load HelpSteer2: {e}")
            raise

        # Sample if requested
        if n_samples is not None and n_samples < len(dataset):
            logger.info(f"Sampling {n_samples} examples from {len(dataset)} total")
            dataset = dataset.shuffle(seed=random_seed).select(range(n_samples))

        # Define dimensions and score ranges
        DIMENSIONS = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
        SCORE_RANGE = {dim: (0, 4) for dim in DIMENSIONS}

        # Process into standardized format
        processed_data = []
        for i, item in enumerate(dataset):
            try:
                # Extract question and response
                question = item.get('prompt', '')
                response = item.get('response', '')

                if not question or not response:
                    logger.warning(f"Sample {i} missing prompt or response, skipping")
                    continue

                # Extract human annotations for all dimensions
                annotations = {}
                all_present = True
                for dim in DIMENSIONS:
                    value = item.get(dim)
                    if value is None:
                        logger.warning(f"Sample {i} missing annotation for {dim}, skipping")
                        all_present = False
                        break
                    annotations[dim] = float(value)

                if not all_present:
                    continue

                # Standardized format
                processed_data.append({
                    # Core fields
                    'question': question,
                    'response': response,
                    'dataset': 'helpsteer2',
                    # Human annotations (aggregated scores already provided)
                    'target_human_aggregated': annotations,
                    'target_human_individual': None,  # Individual annotator scores not provided
                    'score_range_human': SCORE_RANGE,
                    # Synthetic annotations
                    'target_synthetic': None,  # No synthetic scores
                    'score_range_synthetic': None,
                    # Metadata
                    'dimensions': DIMENSIONS,
                    'task_type': 'helpfulness_evaluation',
                    'reference_output': None,
                    'context': None,
                    'response_metadata': None,
                    'annotator_metadata': None,
                    # Traceability
                    'original_index': i
                })

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_data)} samples")

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df

    def _validate_standardized_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that DataFrame has required standardized columns.

        Args:
            data: DataFrame to validate

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        # Core required columns
        required_cols = [
            # Core fields
            'question', 'response', 'dataset',
            # Human annotations
            'target_human_aggregated', 'target_human_individual', 'score_range_human',
            # Synthetic annotations
            'target_synthetic', 'score_range_synthetic',
            # Metadata
            'dimensions', 'task_type', 'reference_output', 'context',
            'response_metadata', 'annotator_metadata',
            # Traceability
            'original_index'
        ]
        missing = [col for col in required_cols if col not in data.columns]

        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}. "
                f"Standardized format requires: {required_cols}"
            )

        logger.info(f"Validated standardized format: {len(data)} samples")
        return data

    def create_experiment_subset(
        self,
        data: pd.DataFrame,
        n_samples: int,
        random_seed: int = 42,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a subset for experiment and optionally save it.
        
        Args:
            data: Full dataset
            n_samples: Number of samples for subset
            random_seed: Random seed for sampling
            output_path: Path to save subset (optional)
            
        Returns:
            Subset DataFrame
        """
        logger.info(f"Creating experiment subset: {n_samples} samples from {len(data)}")
        
        if n_samples >= len(data):
            logger.info("Requested samples >= available data, using all data")
            subset = data.copy()
        else:
            subset = data.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
        
        if output_path:
            logger.info(f"Saving subset to {output_path}")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(subset, f)
        
        logger.info(f"Created subset with {len(subset)} samples")
        return subset


def main():
    """Example usage and testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Load and process datasets")
    parser.add_argument('--dataset',
                        choices=['ultrafeedback', 'judge_bench', 'maj_eval', 'story_spark_qa', 'mslr', 'helpsteer2'],
                        default='ultrafeedback',
                        help='Dataset to load (Note: judge_bench requires --task-name)')
    parser.add_argument('--task-name',
                        help='For JUDGE-BENCH: specific task name (e.g., switchboard, dailydialog)')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples to load (default: 100)')
    parser.add_argument('--output', help='Output path for processed data')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    loader = DatasetLoader()

    # Build kwargs for dataset loading
    load_kwargs = {
        'n_samples': args.n_samples,
        'random_seed': args.random_seed
    }

    # Add task_name for JUDGE-BENCH
    if args.dataset == 'judge_bench':
        if not args.task_name:
            print("\nError: --task-name required for judge_bench dataset")
            print("Example: --dataset judge_bench --task-name switchboard")
            return
        load_kwargs['task_name'] = args.task_name

    # Use new standardized load() method
    try:
        data = loader.load(
            dataset_name=args.dataset,
            **load_kwargs
        )

        print(f"\nLoaded {args.dataset} dataset in standardized format:")
        if args.dataset == 'judge_bench':
            print(f"  Task: {args.task_name}")
        print(f"  Samples: {len(data)}")
        print(f"  Columns: {list(data.columns)}")
        print(f"\nSample data:")
        print(data.head(3))

        # Show target info
        print(f"\nTarget info:")
        human_range = data['score_range_human'].iloc[0]
        synthetic_range = data['score_range_synthetic'].iloc[0]
        has_human_agg = data['target_human_aggregated'].notna().sum()
        has_human_ind = data['target_human_individual'].notna().sum()
        has_synthetic = data['target_synthetic'].notna().sum()
        dimensions = data['dimensions'].iloc[0]
        task_type = data['task_type'].iloc[0]

        print(f"  Task type: {task_type}")
        print(f"  Dimensions: {dimensions}")
        print(f"  Human annotations (aggregated): {has_human_agg}/{len(data)} samples")
        print(f"  Human annotations (individual): {has_human_ind}/{len(data)} samples")
        if human_range:
            print(f"    Score range: {human_range}")
        print(f"  Synthetic annotations: {has_synthetic}/{len(data)} samples")
        if synthetic_range:
            print(f"    Score range: {synthetic_range}")

        if args.output:
            with open(args.output, 'wb') as f:
                pickle.dump(data, f)
            print(f"\nSaved to: {args.output}")

    except NotImplementedError as e:
        print(f"\n⚠️  {e}")
        print("This dataset loader will be implemented when needed for fellowship experiments.")


if __name__ == "__main__":
    main()