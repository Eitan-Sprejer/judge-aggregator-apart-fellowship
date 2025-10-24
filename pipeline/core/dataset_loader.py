"""
Dataset Loading Pipeline

Loads and processes datasets for multi-judge interpretability experiments.
All datasets are preprocessed into a standardized format:
    - question: The input/prompt/instruction
    - response: The model's answer/completion
    - dataset: Source dataset name
    - target_human: Dict of human annotation scores by dimension (None if not available)
                    e.g., {"fluency": 2.0, "population": 1.5} for multi-dimensional
                    or {"overall": 7.5} for single dimension
    - target_synthetic: Dict of synthetic/persona scores by dimension (None if not available)
                        e.g., {"overall": 7.5} for single dimension
    - score_range_human: Tuple (min, max) for human scores (None if not available)
    - score_range_synthetic: Tuple (min, max) for synthetic scores (None if not available)

Supports: UltraFeedback, JUDGE-BENCH, MAJ-Eval, StorySparkQA, MSLR, and custom datasets.
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
        **kwargs
    ) -> pd.DataFrame:
        """
        Load dataset and preprocess to standardized format.

        All datasets are preprocessed to have columns:
            - question: The input/prompt
            - response: The model output
            - dataset: Source dataset name
            - target_human: Human annotation score (None if not available)
            - target_synthetic: Synthetic/persona score (None if not available)
            - score_range_human: (min, max) tuple for human scores (None if not available)
            - score_range_synthetic: (min, max) tuple for synthetic scores (None if not available)

        Args:
            dataset_name: Name of dataset ('ultrafeedback', 'judge_bench', 'maj_eval', 'story_spark_qa', 'mslr')
            **kwargs: Dataset-specific arguments

        Returns:
            DataFrame in standardized format
        """
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
        elif dataset_name == 'custom':
            # For custom datasets, expect user to provide preprocessed data
            if 'data' not in kwargs:
                raise ValueError("Custom dataset requires 'data' argument with preprocessed DataFrame")
            return self._validate_standardized_format(kwargs['data'])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Supported: ultrafeedback, judge_bench, maj_eval, story_spark_qa, mslr, custom")
    
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
                    'question': question,
                    'response': response,
                    'dataset': 'ultrafeedback',
                    'target_human': None,  # No human annotations in UltraFeedback
                    'target_synthetic': None,  # Will be filled by persona simulation
                    'score_range_human': None,
                    'score_range_synthetic': (0.0, 10.0),  # Persona scores are 0-10
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
        Each task has specific score ranges depending on the evaluation metric.

        Args:
            task_name: Name of JUDGE-BENCH task to load
            **kwargs: Additional task-specific arguments

        Returns:
            DataFrame in standardized format

        Note:
            TODO: Implement JUDGE-BENCH loader when ready to run Track 1.3 and 2.2 experiments.
            Will need to define task subsets and implement loader for each task type.
        """
        raise NotImplementedError(
            "JUDGE-BENCH loader not yet implemented. "
            "This will be added when starting Track 1.3 and Track 2.2 experiments."
        )

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
        split: str = "train",
        n_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load StorySparkQA and preprocess to standardized format.

        StorySparkQA is a long-form QA dataset from HuggingFace.
        This will auto-download from HuggingFace when first called.

        Args:
            split: Dataset split (default: "train")
            n_samples: Number of samples (None = all)
            random_seed: Random seed for sampling

        Returns:
            DataFrame in standardized format

        Note:
            Used for Track 1.2 (MAJ-Eval comparison experiments).
        """
        logger.info(f"Loading StorySparkQA dataset (split: {split})")

        # Auto-download from HuggingFace
        try:
            dataset = load_dataset("NEU-HAI/StorySparkQA", split=split, cache_dir=self.cache_dir)
            logger.info(f"Loaded {len(dataset)} samples from StorySparkQA")
        except Exception as e:
            logger.error(f"Failed to load StorySparkQA: {e}")
            raise

        # Sample if requested
        if n_samples is not None and n_samples < len(dataset):
            logger.info(f"Sampling {n_samples} examples from {len(dataset)} total")
            dataset = dataset.shuffle(seed=random_seed).select(range(n_samples))

        # Process into standardized format
        processed_data = []
        for i, item in enumerate(dataset):
            try:
                # Extract question and response
                # Note: Actual field names may differ - adjust as needed when implementing
                question = item.get('question', item.get('input', ''))
                response = item.get('response', item.get('output', ''))

                if not question or not response:
                    logger.warning(f"Sample {i} missing question or response, skipping")
                    continue

                # Standardized format
                processed_data.append({
                    'question': question,
                    'response': response,
                    'dataset': 'story_spark_qa',
                    'target_human': None,  # To be determined by experiment
                    'target_synthetic': None,  # To be determined by experiment
                    'score_range_human': None,  # To be determined by experiment
                    'score_range_synthetic': None,  # To be determined by experiment
                    'original_index': i
                })

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_data)} samples")

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df

    def _preprocess_mslr(
        self,
        data_file: str = "datasets/mslr-annotated/data/data_with_overlap_scores.json",
        n_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load MSLR annotated dataset and preprocess to standardized format.

        MSLR contains medical literature summarization with human facet annotations.
        The dataset is manually downloaded from allenai/mslr-annotated-dataset.

        Args:
            data_file: Path to MSLR data file (default: datasets/mslr-annotated/data/)
            n_samples: Number of samples (None = all)
            random_seed: Random seed for sampling

        Returns:
            DataFrame in standardized format

        Note:
            Used for Track 1.2 (MAJ-Eval comparison experiments).
            The data is already included in the repository (see datasets/mslr-annotated/VERSION).
        """
        import json

        logger.info(f"Loading MSLR dataset from {data_file}")

        # Load JSONL data (JSON Lines format - one JSON object per line)
        try:
            data = []
            with open(data_file, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
            logger.info(f"Loaded MSLR data with {len(data)} review entries")
        except Exception as e:
            logger.error(f"Failed to load MSLR data: {e}")
            logger.info(f"Make sure the MSLR dataset exists at {data_file}")
            logger.info("See datasets/mslr-annotated/VERSION for version information")
            raise

        # Process into standardized format
        processed_data = []
        for review_idx, review in enumerate(data):
            # Each review has multiple predictions (system outputs)
            review_id = review.get('review_id', f'review_{review_idx}')
            target = review.get('target', '')

            for pred_idx, prediction in enumerate(review.get('predictions', [])):
                try:
                    # Use review background/question as "question"
                    # The target summary acts as reference
                    question = f"Review ID: {review_id}"  # Simplified, may want more context
                    response = prediction.get('prediction', '')

                    # Extract human annotations if available
                    annotations = prediction.get('annotations', [])
                    human_score = None
                    if annotations:
                        # Store multi-dimensional facet scores as dict
                        annot = annotations[0]  # Use first annotation
                        facets = {
                            'fluency': annot.get('fluency'),
                            'population': annot.get('population'),
                            'intervention': annot.get('intervention'),
                            'outcome': annot.get('outcome')
                        }
                        # Only include if all facets are present
                        if all(v is not None for v in facets.values()):
                            human_score = facets
                        else:
                            human_score = None

                    if not response:
                        logger.warning(f"Review {review_idx}, prediction {pred_idx} missing response, skipping")
                        continue

                    # Standardized format
                    processed_data.append({
                        'question': question,
                        'response': response,
                        'dataset': 'mslr',
                        'target_human': human_score,  # Dict of facet scores (0-2 scale each) or None
                        'target_synthetic': None,  # No synthetic scores in MSLR
                        'score_range_human': (0.0, 2.0),  # MSLR facets are 0-2 scale
                        'score_range_synthetic': None,
                        'original_index': f"{review_idx}_{pred_idx}",
                        'review_id': review_id,
                        'system_id': prediction.get('exp_short', 'unknown')
                    })

                except Exception as e:
                    logger.warning(f"Error processing review {review_idx}, prediction {pred_idx}: {e}")
                    continue

        logger.info(f"Successfully processed {len(processed_data)} prediction samples from MSLR")

        # Sample if requested
        df = pd.DataFrame(processed_data)
        if n_samples is not None and n_samples < len(df):
            logger.info(f"Sampling {n_samples} examples from {len(df)} total")
            df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)

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
        required_cols = [
            'question', 'response', 'dataset',
            'target_human', 'target_synthetic',
            'score_range_human', 'score_range_synthetic'
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
    parser.add_argument('--dataset', choices=['ultrafeedback', 'judge_bench', 'maj_eval', 'story_spark_qa', 'mslr'],
                        default='ultrafeedback',
                        help='Dataset to load')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples to load (default: 100)')
    parser.add_argument('--output', help='Output path for processed data')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    loader = DatasetLoader()

    # Use new standardized load() method
    try:
        data = loader.load(
            dataset_name=args.dataset,
            n_samples=args.n_samples,
            random_seed=args.random_seed
        )

        print(f"\nLoaded {args.dataset} dataset in standardized format:")
        print(f"  Samples: {len(data)}")
        print(f"  Columns: {list(data.columns)}")
        print(f"\nSample data:")
        print(data.head(3))

        # Show target info
        print(f"\nTarget info:")
        human_range = data['score_range_human'].iloc[0]
        synthetic_range = data['score_range_synthetic'].iloc[0]
        has_human = data['target_human'].notna().sum()
        has_synthetic = data['target_synthetic'].notna().sum()

        print(f"  Human annotations: {has_human}/{len(data)} samples")
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