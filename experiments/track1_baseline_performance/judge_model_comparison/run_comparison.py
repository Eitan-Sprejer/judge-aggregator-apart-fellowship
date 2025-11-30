#!/usr/bin/env python3
"""
Run judge model comparison experiment: GPT-5-Mini vs GPT-5-Nano

This script:
1. Loads the same dataset subset for both models
2. Evaluates all judges with gpt-5-mini
3. Evaluates all judges with gpt-5-nano (same samples)
4. Saves raw judge scores for both
5. Runs analysis to compare

Usage:
    python experiments/track1_baseline_performance/judge_model_comparison/run_comparison.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pickle
import json
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from pipeline.core.dataset_loader import DatasetLoader
from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.judge_creation_orchestrator import JudgeCreationOrchestrator


def load_dataset(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Load HelpSteer2 dataset subset."""
    print(f"Loading HelpSteer2 dataset (n={n_samples}, seed={seed})...")

    loader = DatasetLoader()
    df = loader.load(
        dataset_name="helpsteer2",
        use_cache=True,
        n_samples=n_samples,
        random_seed=seed,
        split="train"
    )

    print(f"  Loaded {len(df)} samples")
    return df


def ensure_judges_exist(
    dataset_name: str = "helpsteer2",
    dimensions: list = ["helpfulness", "correctness"],
    depth: int = 1,
    df: pd.DataFrame = None
) -> list:
    """Ensure judge definitions exist (creates if needed).

    Follows same logic as run_experiment.py:create_judges_if_needed()
    """
    print(f"Checking judges for {dataset_name} dimensions: {dimensions}...")

    # Check if judge files already exist
    judges_dir = project_root / "judges" / dataset_name
    depth_files = list(judges_dir.glob("depth_*.yaml")) if judges_dir.exists() else []

    if depth_files:
        print(f"  Found existing judge files: {[f.name for f in depth_files]}")
        return [str(f.relative_to(project_root)) for f in depth_files]

    # Create judges if needed
    print("  Creating judges...")
    import yaml

    rubrics_path = project_root / "pipeline" / "utils" / "dataset_rubrics.yaml"
    with open(rubrics_path) as f:
        rubrics = yaml.safe_load(f)

    rubric_config = rubrics[dataset_name]
    rubric_dimensions = {d['name']: d['description'] for d in rubric_config.get('dimensions', [])}
    full_rubric = rubric_config.get('rubric')

    orchestrator = JudgeCreationOrchestrator(
        dataset_name=dataset_name,
        model="openai/gpt-5",
        temperature=0.4,
        max_tokens=10000
    )

    dims = [{'name': d, 'description': rubric_dimensions[d]} for d in dimensions]

    # Extract sample Q&A pairs for context (matches run_experiment.py)
    sample_qa_pairs = None
    if df is not None and len(df) > 0:
        question_col = 'question' if 'question' in df.columns else 'instruction'
        response_col = 'response' if 'response' in df.columns else 'answer'
        sample_qa_pairs = [
            {'question': row[question_col], 'response': row[response_col]}
            for _, row in df.head(3).iterrows()
        ]

    orchestrator.create_judges_for_dataset(
        dimensions=dims,
        cache_strategy="auto",
        decomposition_depth=depth,
        sample_qa_pairs=sample_qa_pairs,
        full_rubric=full_rubric
    )

    # Get created files
    depth_files = sorted(judges_dir.glob("depth_*.yaml"))
    return [str(f.relative_to(project_root)) for f in depth_files]


def evaluate_with_model(
    df: pd.DataFrame,
    judge_files: list,
    model: str,
    concurrency: int = 10
) -> pd.DataFrame:
    """Evaluate dataset with judges using specified model."""
    print(f"\nEvaluating with model: {model}")
    print(f"  Judge files: {judge_files}")

    evaluator = JudgeEvaluator(
        judge_file_paths=judge_files,
        model=model
    )

    print(f"  Running evaluation on {len(df)} samples with {len(evaluator.judge_ids)} judges...")

    # Run evaluation (will use cache if available for this model)
    df_with_scores = evaluator.evaluate_dataset(
        df,
        max_workers=concurrency
    )

    print(f"  Evaluation complete")
    return df_with_scores


def extract_judge_scores(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract judge scores into a dict of arrays."""
    if 'judge_scores' not in df.columns:
        raise ValueError("DataFrame missing judge_scores column")

    # Get judge IDs from first row
    first_scores = df.iloc[0]['judge_scores']
    if isinstance(first_scores, dict):
        judge_ids = list(first_scores.keys())
    else:
        judge_ids = [f"judge_{i}" for i in range(len(first_scores))]

    # Extract scores for each judge
    scores = {}
    for judge_id in judge_ids:
        if isinstance(first_scores, dict):
            scores[judge_id] = np.array([
                row['judge_scores'].get(judge_id, np.nan)
                for _, row in df.iterrows()
            ])
        else:
            idx = judge_ids.index(judge_id)
            scores[judge_id] = np.array([
                row['judge_scores'][idx] if len(row['judge_scores']) > idx else np.nan
                for _, row in df.iterrows()
            ])

    return scores


def extract_human_labels(df: pd.DataFrame, dimensions: list) -> Dict[str, np.ndarray]:
    """Extract human labels for specified dimensions."""
    labels = {}
    for dim in dimensions:
        labels[dim] = np.array([
            row['target_human_aggregated'].get(dim, np.nan)
            if row['target_human_aggregated'] is not None else np.nan
            for _, row in df.iterrows()
        ])
    return labels


def main():
    # Configuration
    N_SAMPLES = 100
    SEED = 42
    DIMENSIONS = ["helpfulness", "correctness"]
    MODELS = {
        "mini": "openai/gpt-5-mini",
        "nano": "openai/gpt-5-nano"
    }
    CONCURRENCY = 10

    # Results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"comparison_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Judge Model Comparison: GPT-5-Mini vs GPT-5-Nano")
    print("=" * 60)
    print(f"Samples: {N_SAMPLES}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    # Step 1: Load dataset
    df = load_dataset(n_samples=N_SAMPLES, seed=SEED)

    # Step 2: Ensure judges exist (pass df for sample Q&A context if creating)
    judge_files = ensure_judges_exist(
        dataset_name="helpsteer2",
        dimensions=DIMENSIONS,
        depth=1,
        df=df
    )

    # Step 3: Evaluate with each model
    results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*40}")
        print(f"Running {model_name.upper()} evaluation")
        print(f"{'='*40}")

        df_scored = evaluate_with_model(
            df=df.copy(),
            judge_files=judge_files,
            model=model_path,
            concurrency=CONCURRENCY
        )

        # Extract scores
        judge_scores = extract_judge_scores(df_scored)

        results[model_name] = {
            "model": model_path,
            "judge_scores": judge_scores,
            "n_samples": len(df_scored),
            "n_judges": len(judge_scores)
        }

        # Save individual result
        model_result_path = run_dir / f"{model_name}_scores.pkl"
        with open(model_result_path, 'wb') as f:
            pickle.dump({
                "judge_scores": judge_scores,
                "model": model_path,
                "n_samples": N_SAMPLES,
                "seed": SEED
            }, f)
        print(f"  Saved to: {model_result_path}")

    # Step 4: Extract human labels
    human_labels = extract_human_labels(df, DIMENSIONS)

    # Step 5: Save combined results
    combined_results = {
        "mini": results["mini"]["judge_scores"],
        "nano": results["nano"]["judge_scores"],
        "human_labels": human_labels,
        "config": {
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "dimensions": DIMENSIONS,
            "models": MODELS,
            "judge_files": judge_files,
            "timestamp": timestamp
        }
    }

    combined_path = run_dir / "combined_results.pkl"
    with open(combined_path, 'wb') as f:
        pickle.dump(combined_results, f)
    print(f"\nSaved combined results to: {combined_path}")

    # Save config as JSON for easy reading
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(combined_results["config"], f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {run_dir}")
    print("\nNext step: Run analysis")
    print(f"  python experiments/track1_baseline_performance/judge_model_comparison/analyze_comparison.py {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
