#!/usr/bin/env python3
"""
Cross-Model Persona Comparison Experiment

Runs the persona simulation once per model on the same dataset,
saving each run separately for later cross-model comparison.

Usage:
    # Run a specific model as run 2
    python rebutal_augusto_pt3/run_reproducibility.py --run-id 2 --model deepseek/deepseek-v3.2

    # Small test (50 samples)
    python rebutal_augusto_pt3/run_reproducibility.py --run-id 2 --model deepseek/deepseek-v3.2 --n-samples 50
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.core.persona_simulation import PersonaSimulator, PERSONAS


def load_original_dataset(n_samples=None, random_seed=42):
    """Load the original UltraFeedback dataset used in experiments.

    Uses the pre-existing data_with_judge_scores.pkl which has the
    instruction/answer pairs that were originally scored.

    Args:
        n_samples: Number of samples to use (None = all 2000)
        random_seed: Random seed for reproducible subsampling

    Returns:
        DataFrame with 'question' and 'response' columns (standardized names)
    """
    data_path = Path("datasets/data_with_judge_scores.pkl")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Original dataset not found at {data_path}. "
            "Run the full experiment pipeline first."
        )

    df = pd.read_pickle(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    # Subsample if requested
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
        print(f"Subsampled to {n_samples} samples (seed={random_seed})")

    # Rename columns to match what PersonaSimulator expects
    df = df.rename(columns={"instruction": "question", "answer": "response"})

    return df


def extract_scores_from_feedback(df):
    """Extract per-persona scores from human_feedback column.

    Args:
        df: DataFrame with 'human_feedback' column from PersonaSimulator

    Returns:
        Dictionary with:
        - per_persona: {persona_name: [score_sample_0, score_sample_1, ...]}
        - average_scores: [avg_score_sample_0, avg_score_sample_1, ...]
    """
    per_persona = {name: [] for name in PERSONAS.keys()}
    average_scores = []

    for _, row in df.iterrows():
        feedback = row.get("human_feedback")
        if feedback is None or "personas" not in feedback:
            # Missing feedback — fill with None
            for name in PERSONAS:
                per_persona[name].append(None)
            average_scores.append(None)
            continue

        personas = feedback["personas"]
        for name in PERSONAS:
            if name in personas and "score" in personas[name]:
                per_persona[name].append(personas[name]["score"])
            else:
                per_persona[name].append(None)

        average_scores.append(feedback.get("average_score") or feedback.get("score"))

    return {"per_persona": per_persona, "average_scores": average_scores}


async def run_single_simulation(df, run_id, api_key, api_base, model,
                                 temperature, concurrency, output_dir):
    """Run one persona simulation pass and save results.

    Args:
        df: DataFrame with 'question' and 'response' columns
        run_id: Run number (1-indexed)
        api_key: API key for LLM provider
        api_base: API base URL
        model: Model name
        temperature: Sampling temperature
        concurrency: Number of concurrent API requests
        output_dir: Directory to save results

    Returns:
        Dictionary with extracted scores
    """
    print(f"\n{'='*60}")
    print(f"  RUN {run_id} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {model} | Temp: {temperature} | Samples: {len(df)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Create a fresh simulator for each run
    simulator = PersonaSimulator(
        api_key=api_key,
        api_base=api_base,
        model=model,
    )
    # Override temperature by monkey-patching _get_single_feedback
    # (PersonaSimulator hardcodes temperature=0.7)
    original_get_feedback = simulator._get_single_feedback

    async def patched_get_feedback(persona_name, query, answer, **kwargs):
        """Patched version that uses configurable temperature."""
        persona_bio = PERSONAS[persona_name]
        system_prompt, user_prompt = simulator._get_prompts(
            persona_name, query, answer, persona_bio
        )

        delay = kwargs.get("initial_delay", 1.0)
        max_retries = kwargs.get("max_retries", 3)

        for attempt in range(max_retries):
            try:
                response = await simulator.client.chat.completions.create(
                    model=simulator.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=100,
                    temperature=temperature,
                    timeout=45.0,  # Prevent indefinite hangs
                )
                content = response.choices[0].message.content
                result = json.loads(content)
                result["persona"] = persona_name
                return result
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    return {"persona": persona_name, "score": 5,
                            "analysis": "Error: Invalid JSON", "error": str(e)}
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    return {"persona": persona_name, "score": 5,
                            "analysis": "Error occurred", "error": str(e)}

    simulator._get_single_feedback = patched_get_feedback

    # Determine resume_from by checking for checkpoints
    checkpoint_dir = output_dir / f"checkpoints_run_{run_id}"
    resume_from = None
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            checkpoint_num = int(last_checkpoint.stem.split("_")[1])
            resume_from = checkpoint_num * 50  # Since checkpoint_interval=50 below

    if resume_from:
        print(f"  Resuming from sample {resume_from} (checkpoint: {last_checkpoint.name})")

    # Run simulation
    df_result = await simulator.simulate_dataset(
        df.copy(),
        question_col="question",
        answer_col="response",
        concurrency=concurrency,
        checkpoint_interval=50,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
    )

    elapsed = time.time() - start_time

    # Extract scores
    scores = extract_scores_from_feedback(df_result)

    # Save full result
    result_path = output_dir / f"run_{run_id}.pkl"
    result_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "api_base": api_base,
        "temperature": temperature,
        "n_samples": len(df),
        "elapsed_seconds": elapsed,
        "scores": scores,
        "df": df_result,  # Full dataframe with human_feedback column
    }
    with open(result_path, "wb") as f:
        pickle.dump(result_data, f)

    # Print summary
    valid_scores = [s for s in scores["average_scores"] if s is not None]
    errors = sum(1 for s in scores["average_scores"] if s is None)
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    print(f"\n✅ Run {run_id} complete in {elapsed:.1f}s")
    print(f"   Mean score: {mean_score:.3f}")
    print(f"   Valid: {len(valid_scores)} | Errors: {errors}")
    print(f"   Saved to: {result_path}")

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Run persona simulation multiple times for reproducibility analysis"
    )
    parser.add_argument(
        "--n-runs", type=int, default=5,
        help="Number of simulation runs (default: 5, ignored if --run-id is set)"
    )
    parser.add_argument(
        "--run-id", type=int, default=None,
        help="Run a single specific run ID (for parallel execution). "
             "If set, only this run is executed."
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of samples to use (default: all 2000)"
    )
    parser.add_argument(
        "--resume-from-run", type=int, default=1,
        help="Start from this run number (default: 1, only used in sequential mode)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7, same as original)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Number of concurrent API requests (default: 10)"
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="API base URL (default: https://api.lambda.ai/v1)"
    )
    parser.add_argument(
        "--api-key-env", type=str, default="OPEN_AI_API_KEY",
        help="Environment variable name for API key (default: OPEN_AI_API_KEY)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g. deepseek/deepseek-v3.2)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="rebutal_augusto_pt3/results",
        help="Output directory (default: rebutal_augusto_pt3/results)"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"ERROR: {args.api_key_env} not set in environment.")
        sys.exit(1)

    api_base = args.api_base or "https://api.lambda.ai/v1"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = load_original_dataset(n_samples=args.n_samples)

    # Determine which runs to execute
    if args.run_id is not None:
        run_ids = [args.run_id]
    else:
        run_ids = list(range(args.resume_from_run, args.n_runs + 1))

    # Save experiment metadata
    metadata = {
        "start_time": datetime.now().isoformat(),
        "n_runs": args.n_runs,
        "run_ids": run_ids,
        "n_samples": len(df),
        "model": args.model,
        "api_base": api_base,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "personas": list(PERSONAS.keys()),
    }
    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n🔬 Persona Reproducibility Experiment")
    print(f"   Run(s): {run_ids}")
    print(f"   Samples: {len(df)}")
    print(f"   Model: {args.model}")
    print(f"   API: {api_base}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Output: {output_dir}")
    print(flush=True)

    # Run simulations
    async def run_all():
        for run_id in run_ids:
            await run_single_simulation(
                df=df,
                run_id=run_id,
                api_key=api_key,
                api_base=api_base,
                model=args.model,
                temperature=args.temperature,
                concurrency=args.concurrency,
                output_dir=output_dir,
            )

    asyncio.run(run_all())

    print(f"\n🎉 Run(s) {run_ids} complete!")
    print(f"   Results saved to: {output_dir}/")
    print(f"   Next: python rebutal_augusto_pt3/analyze_reproducibility.py")


if __name__ == "__main__":
    main()
