#!/usr/bin/env python3
"""
Persona Score Phrasing Sensitivity Experiment

Runs the persona simulation on a dataset using different wording variations
for the persona bios.

Usage:
    python rebutal_augusto_pt2/run_phrasing.py --variant base
    python rebutal_augusto_pt2/run_phrasing.py --variant v1
"""

import argparse
import asyncio
import importlib
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add local directory to path to use modified versions of simulator
sys.path.insert(0, str(Path(__file__).parent))

from persona_simulation import PersonaSimulator


def load_original_dataset(n_samples=None, random_seed=42):
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


def extract_scores_from_feedback(df, personas_dict):
    per_persona = {name: [] for name in personas_dict.keys()}
    average_scores = []

    for _, row in df.iterrows():
        feedback = row.get("human_feedback")
        if feedback is None or "personas" not in feedback:
            for name in personas_dict:
                per_persona[name].append(None)
            average_scores.append(None)
            continue

        personas = feedback["personas"]
        for name in personas_dict:
            if name in personas and isinstance(personas[name], dict) and "score" in personas[name]:
                per_persona[name].append(personas[name]["score"])
            else:
                per_persona[name].append(None)

        average_scores.append(feedback.get("average_score") or feedback.get("score"))

    return {"per_persona": per_persona, "average_scores": average_scores}


async def run_variant_simulation(df, variant, personas_dict, api_key, api_base, model,
                                 temperature, concurrency, output_dir):
    print(f"\n{'='*60}")
    print(f"  VARIANT {variant} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {model} | Temp: {temperature} | Samples: {len(df)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    simulator = PersonaSimulator(
        api_key=api_key,
        api_base=api_base,
        model=model,
        personas_dict=personas_dict
    )

    original_get_feedback = simulator._get_single_feedback

    async def patched_get_feedback(persona_name, query, answer, **kwargs):
        persona_bio = personas_dict[persona_name]
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
                    timeout=45.0, 
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

    checkpoint_dir = output_dir / f"checkpoints_{variant}"
    resume_from = None
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            checkpoint_num = int(last_checkpoint.stem.split("_")[1])
            resume_from = checkpoint_num * 50

    if resume_from:
        print(f"  Resuming from sample {resume_from} (checkpoint: {last_checkpoint.name})")

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

    scores = extract_scores_from_feedback(df_result, personas_dict)

    result_path = output_dir / f"{variant}.pkl"
    result_data = {
        "variant": variant,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "api_base": api_base,
        "temperature": temperature,
        "n_samples": len(df),
        "elapsed_seconds": elapsed,
        "scores": scores,
        "df": df_result,
        "personas_dict": personas_dict
    }
    with open(result_path, "wb") as f:
        pickle.dump(result_data, f)

    valid_scores = [s for s in scores["average_scores"] if s is not None]
    errors = sum(1 for s in scores["average_scores"] if s is None)
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    print(f"\n✅ Variant {variant} complete in {elapsed:.1f}s")
    print(f"   Mean score: {mean_score:.3f}")
    print(f"   Valid: {len(valid_scores)} | Errors: {errors}")
    print(f"   Saved to: {result_path}")

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Run persona simulation for a specific wording variant"
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        help="Variant name mapping to a module (e.g., 'base' -> personas.py, 'v1' -> persona1.py)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Number of samples to use (default: 500)"
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
        "--model", type=str, default="llama3.1-405b-instruct-fp8",
        help="Model name (default: llama3.1-405b-instruct-fp8)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="rebutal_augusto_pt2/results",
        help="Output directory (default: rebutal_augusto_pt2/results)"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"ERROR: {args.api_key_env} not set in environment.")
        sys.exit(1)

    api_base = args.api_base or "https://api.lambda.ai/v1"

    # Map variant to python module
    variant_map = {
        "base": "personas",
        "v1": "persona1",
        "v2": "persona2",
        "v3": "persona3",
        "v4": "persona4",
    }
    
    if args.variant not in variant_map:
        print(f"ERROR: Variant {args.variant} is not recognized. Pick one of: {list(variant_map.keys())}")
        sys.exit(1)
        
    module_name = variant_map[args.variant]
    try:
        mod = importlib.import_module(module_name)
        personas_dict = mod.PERSONAS
        print(f"Loaded {len(personas_dict)} personas from {module_name}.py")
    except Exception as e:
        print(f"ERROR loading '{module_name}': {e}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_original_dataset(n_samples=args.n_samples)

    print(f"\n🔬 Persona Phrasing Sensitivity Experiment")
    print(f"   Variant: {args.variant} ({module_name}.py)")
    print(f"   Samples: {len(df)}")
    print(f"   Model: {args.model}")
    print(f"   Output: {output_dir}")
    print(flush=True)

    asyncio.run(
        run_variant_simulation(
            df=df,
            variant=args.variant,
            personas_dict=personas_dict,
            api_key=api_key,
            api_base=api_base,
            model=args.model,
            temperature=args.temperature,
            concurrency=args.concurrency,
            output_dir=output_dir,
        )
    )

if __name__ == "__main__":
    main()
