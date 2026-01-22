"""
Judge Evaluation Pipeline

Evaluates question-response pairs using configurable sets of specialized judges.
Handles parallel evaluation, rate limiting, and checkpoint saving.

Now supports dynamic judge selection for fellowship experiments.
"""

import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

from pipeline.utils import judge_rubrics
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS
from pipeline.utils.martian_client import MartianClient

# Set up logging - INFO to console, DEBUG to file only
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Don't propagate to root logger (prevents duplicate output)

# Console handler - INFO and above only
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(_console_handler)

# File handler - DEBUG level (captures LLM outputs)
_file_handler = logging.FileHandler('judge_evaluation_debug.log', mode='w')
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(_file_handler)

import sys
from pathlib import Path

# Add the root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parents[2]  # Adjust the number based on your file's location
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Default configuration
DEFAULT_MAX_WORKERS = 50  # Maximum parallel workers (50 workers × 5 judges = 250 concurrent)
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY = 1.0

# Default judge set (all judges from JUDGE_RUBRICS for backward compatibility)
DEFAULT_JUDGE_IDS = list(JUDGE_RUBRICS.keys())


class JudgeEvaluator:
    """Handles evaluation of Q&A pairs using judges loaded from YAML files."""

    def __init__(
        self,
        judge_file_paths: List[str],
        use_local: bool = False,
        config_path: Optional[str] = None,
        model: Optional[str] = None, 
    ):
        """
        Initialize the evaluator with Martian API client.

        Args:
            judge_file_paths: List of paths to YAML files with judge definitions
                             Can be relative to repo root or absolute
            config_path: Optional path to configuration file (kept for compatibility)
            model: Model to use for judge evaluation (defaults to gpt-5-mini)
        """
        self.judge_file_paths = judge_file_paths
        logger.info(f"Initializing evaluator with judges from {len(judge_file_paths)} file(s)")

        # Initialize Martian client with specified model
        self.client = MartianClient(default_model=model, use_local=use_local) if model else MartianClient(use_local=use_local)

        # Load judge rubrics from files
        self.judges = self._load_judges()
        self.judge_ids = list(self.judges.keys())

        logger.info(f"✅ Loaded {len(self.judge_ids)} judges total")

    def _load_judges(self) -> Dict[str, str]:
        """Load judge rubrics from YAML files."""
        # Load rubric functions from files
        judge_rubric_functions = judge_rubrics.load_judges_from_files(self.judge_file_paths)

        # Convert rubric functions to rubric text
        judges = {}
        for judge_id, rubric_func in judge_rubric_functions.items():
            rubric = rubric_func()
            judges[judge_id] = rubric
            logger.info(f"  ✅ Loaded judge {judge_id}")

        return judges
    
    def evaluate_single(self, question: str, answer: str, judge_id: str) -> float:
        """
        Evaluate a single Q&A pair with a specific judge.

        Args:
            question: The user's question/instruction
            answer: The model's response
            judge_id: ID of the judge to use

        Returns:
            Score from the judge (0.0-4.0)
        """
        if judge_id not in self.judges:
            raise ValueError(f"Judge {judge_id} not loaded")

        # Get judge rubric
        rubric = self.judges[judge_id]

        # Evaluate using Martian client
        result = self.client.evaluate_with_rubric(
            rubric=rubric,
            question=question,
            answer=answer
        )
        logger.debug(result)
        return result["score"]
    
    def _evaluate_with_retry(
        self,
        args: Tuple[str, str, str],
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY
    ) -> Tuple[int, float]:
        """
        Evaluate with exponential backoff on failure.

        Args:
            args: Tuple of (question, answer, judge_id)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds

        Returns:
            Tuple of (judge_index, score)
        """
        question, answer, judge_id = args
        judge_index = self.judge_ids.index(judge_id)

        delay = initial_delay
        for attempt in range(max_retries):
            try:
                score = self.evaluate_single(question, answer, judge_id)
                return (judge_index, score)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to evaluate with {judge_id} after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed for {judge_id}, retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2

        # Should not reach here
        raise RuntimeError(f"Failed to evaluate with {judge_id}")
    
    def evaluate_parallel(
        self,
        question: str,
        answer: str,
        max_workers: Optional[int] = None
    ) -> List[float]:
        """
        Evaluate a Q&A pair with all configured judges in parallel.

        Args:
            question: The user's question/instruction
            answer: The model's response
            max_workers: Number of parallel workers

        Returns:
            List of scores in judge order
        """
        import time
        from concurrent.futures import as_completed

        start_time = time.time()

        scores = [0.0] * len(self.judge_ids)
        eval_args = [(question, answer, judge_id) for judge_id in self.judge_ids]

        # Use submit() and as_completed() for true parallel execution
        with ThreadPoolExecutor(max_workers=max_workers or min(len(self.judge_ids), DEFAULT_MAX_WORKERS)) as executor:
            # Submit all tasks simultaneously
            future_to_args = {executor.submit(self._evaluate_with_retry, args): args for args in eval_args}

            # Collect results as they complete (not in order)
            for future in as_completed(future_to_args):
                try:
                    judge_index, score = future.result()
                    scores[judge_index] = score
                except Exception as e:
                    args = future_to_args[future]
                    logger.error(f"Failed to evaluate with {args[2]}: {e}")
                    # Keep default score of 0.0 for failed judges

        elapsed = time.time() - start_time
        logger.info(f"Parallel evaluation took {elapsed:.2f}s for {len(self.judge_ids)} judges")

        return scores
    
    def evaluate_dataset(
        self,
        data: pd.DataFrame,
        question_col: str = 'question',
        answer_col: str = 'response',
        output_path: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        max_workers: Optional[int] = None,
        resume_from: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate an entire dataset with all configured judges.

        REQUIRES standardized format with 'question' and 'response' columns.
        Use dataset standardization utilities to convert old formats first.

        Args:
            data: DataFrame with questions and answers in standardized format
            question_col: Name of question column (default: 'question')
            answer_col: Name of answer column (default: 'response')
            output_path: Path to save final results
            checkpoint_dir: Directory for checkpoint files
            checkpoint_interval: Save checkpoint every N rows
            max_workers: Number of parallel workers
            resume_from: Resume from specific row index

        Returns:
            DataFrame with judge scores added
        """
        # Validate standardized format
        if question_col is None:
            question_col = 'question'

        if answer_col is None:
            answer_col = 'response'

        # Check required columns exist
        missing = [col for col in [question_col, answer_col] if col not in data.columns]
        if missing:
            raise ValueError(
                f"Data missing required columns: {missing}. "
                f"Standardized format requires 'question' and 'response' columns. "
                f"Available columns: {data.columns.tolist()}. "
                f"Use dataset standardization utilities to convert old formats first."
            )

        logger.info(f"Using columns: question={question_col}, answer={answer_col}")
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = [None] * len(data)
        start_idx = resume_from or 0
        
        # Create tasks
        tasks = [
            (i, row[question_col], row[answer_col])
            for i, (_, row) in enumerate(data.iterrows())
            if i >= start_idx
        ]
        
        logger.info(f"Evaluating {len(tasks)} rows starting from index {start_idx}")
        
        completed = start_idx
        checkpoint_batch = start_idx // checkpoint_interval
        
        # Process with parallelization across rows
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def eval_row(args):
                idx, question, answer = args
                # Use full judge parallelization (all configured judges), regardless of row concurrency
                scores = self.evaluate_parallel(question, answer, max_workers=None)
                return idx, scores
            
            for idx, scores in executor.map(eval_row, tasks):
                results[idx] = scores
                completed += 1
                
                # Log progress
                if completed % 10 == 0:
                    logger.info(f"Completed {completed}/{len(data)} rows")
                
                # Save checkpoint
                if checkpoint_dir and completed % checkpoint_interval == 0:
                    checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_batch:03d}.pkl"
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info(f"Saved checkpoint to {checkpoint_file}")
                    checkpoint_batch += 1
        
        # Add scores to dataframe
        data['judge_scores'] = results
        
        # Save final results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved final results to {output_path}")
        
        return data


def load_checkpoint(checkpoint_dir: Path, batch_num: int) -> List:
    """
    Load a specific checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        batch_num: Checkpoint batch number to load
        
    Returns:
        List of results from checkpoint
    """
    checkpoint_file = checkpoint_dir / f"checkpoint_{batch_num:03d}.pkl"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_file} not found")
    
    with open(checkpoint_file, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Loaded checkpoint from {checkpoint_file}")
    return results


def main():
    """Main entry point for judge evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Q&A pairs with judges")
    parser.add_argument('--input', required=True, help='Path to input pickle file')
    parser.add_argument('--output', required=True, help='Path to output pickle file')
    parser.add_argument('--checkpoint-dir', help='Directory for checkpoint files')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N rows')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Number of parallel workers')
    parser.add_argument('--resume-from', type=int,
                        help='Resume from specific row index')
    # parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--judge_file', help='Path to judge file')
    parser.add_argument('--model', help='Model to use for judge evaluation')
    parser.add_argument('--use-local', action='store_true',
                        help='Use local model')
    args = parser.parse_args()
    
    # Load input data
    logger.info(f"Loading data from {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to DataFrame if needed
    if isinstance(data, list):
        data = pd.DataFrame(data)
    
    logger.info(f"Loaded {len(data)} rows")
    
    # Initialize evaluator
    evaluator = JudgeEvaluator(judge_file_paths=[args.judge_file] if args.judge_file else [], model=args.model, use_local=args.use_local)
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(
        data,
        output_path=args.output,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        checkpoint_interval=args.checkpoint_interval,
        max_workers=args.max_workers,
        resume_from=args.resume_from
    )
    
    logger.info(f"Evaluation complete! Results saved to {args.output}")
    
    # Print summary statistics
    if 'judge_scores' in results.columns:
        scores_df = pd.DataFrame(results['judge_scores'].tolist(), columns=evaluator.judge_ids)
        print("\nScore Statistics:")
        print(scores_df.describe())
        print(f"\nUsed {len(evaluator.judge_ids)} judges: {', '.join(evaluator.judge_ids[:3])}{'...' if len(evaluator.judge_ids) > 3 else ''}")


if __name__ == "__main__":
    main()