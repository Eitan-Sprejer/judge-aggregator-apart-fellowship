#!/usr/bin/env python3
"""
Re-evaluate missing judge scores (defaults to 0.0) and save a repaired dataset.
"""

import argparse
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Set

import pandas as pd

from pipeline.utils import judge_rubrics
from pipeline.utils.martian_client import MartianClient


DEFAULT_INPUT = "datasets/helpsteer2_full_30_judges.pkl"
DEFAULT_JUDGE_FILES = [
    "judges/helpsteer2/depth_0_parents.yaml",
    "judges/helpsteer2/depth_1_children.yaml",
]


def _load_df(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return pd.DataFrame(data)
    return data


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(df, f)


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _is_missing(value: object, treat_zero: bool, treat_nan: bool) -> bool:
    if value is None:
        return treat_nan
    if treat_nan and _is_nan(value):
        return True
    if treat_zero and (value == 0 or value == 0.0):
        return True
    return False


def _validate_judge_ids(judge_ids: List[str], rubrics: Dict[str, str]) -> None:
    missing = [jid for jid in judge_ids if jid not in rubrics]
    if missing:
        raise ValueError(f"Missing rubrics for judge IDs: {missing}")


def _load_judge_ids(
    df: pd.DataFrame,
    judge_files: List[str],
    judge_ids_col: str,
) -> List[str]:
    if judge_ids_col in df.columns and df[judge_ids_col].notna().any():
        judge_ids = df[judge_ids_col].iloc[0]
        if not isinstance(judge_ids, list):
            raise ValueError(f"{judge_ids_col} column is not a list")
        if not all(df[judge_ids_col].apply(lambda x: x == judge_ids)):
            raise ValueError(f"{judge_ids_col} column varies across rows; expected a single judge order")
        return judge_ids

    judge_ids = judge_rubrics.get_judge_ids_from_files(judge_files)
    df[judge_ids_col] = [judge_ids] * len(df)
    return judge_ids


def _normalize_scores(scores: object, expected_len: int) -> List[Optional[float]]:
    if scores is None or _is_nan(scores):
        return [float("nan")] * expected_len
    if not isinstance(scores, list):
        raise ValueError("judge_scores must be a list per row")
    if len(scores) != expected_len:
        raise ValueError(f"judge_scores length {len(scores)} != expected {expected_len}")
    return scores


def _build_missing_tasks(
    scores_list: List[List[Optional[float]]],
    judge_ids: List[str],
    treat_zero: bool,
    treat_nan: bool,
    only_judges: Optional[set],
    max_missing: Optional[int],
    recompute_all: bool,
    completed_keys: Optional[Set[int]],
) -> List[Tuple[int, int]]:
    tasks: List[Tuple[int, int]] = []
    judge_count = len(judge_ids)
    for row_idx, row_scores in enumerate(scores_list):
        for j_idx, (jid, score) in enumerate(zip(judge_ids, row_scores)):
            if only_judges and jid not in only_judges:
                continue
            if recompute_all or _is_missing(score, treat_zero, treat_nan):
                if completed_keys is not None:
                    key = row_idx * judge_count + j_idx
                    if key in completed_keys:
                        continue
                tasks.append((row_idx, j_idx))
                if max_missing and len(tasks) >= max_missing:
                    return tasks
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate missing HelpSteer2 judge scores and save a repaired dataset."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input dataset pickle path")
    parser.add_argument("--output", help="Output dataset pickle path")
    parser.add_argument(
        "--judge-files",
        nargs="+",
        default=DEFAULT_JUDGE_FILES,
        help="One or more YAML files defining the judges",
    )
    parser.add_argument("--question-col", default="question", help="Question column name")
    parser.add_argument("--answer-col", default="response", help="Answer column name")
    parser.add_argument("--scores-col", default="judge_scores", help="Scores column name")
    parser.add_argument("--judge-ids-col", default="judge_ids", help="Judge IDs column name")
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per judge")
    parser.add_argument("--initial-delay", type=float, default=1.0, help="Initial retry delay (s)")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save every N updates")
    parser.add_argument("--resume", action="store_true", help="Resume from output if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Only report missing counts")
    parser.add_argument(
        "--only-judges",
        nargs="*",
        help="Only re-evaluate these judge IDs",
    )
    parser.add_argument("--max-missing", type=int, help="Limit number of missing scores to re-run")
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Re-evaluate all scores, ignoring missing detection",
    )
    parser.add_argument(
        "--progress-path",
        help="Path to progress file for resume (defaults to <output>.progress.pkl when recomputing)",
    )
    parser.add_argument(
        "--keep-zero",
        action="store_true",
        help="Do not treat 0.0 as missing",
    )
    parser.add_argument(
        "--skip-nan",
        action="store_true",
        help="Do not treat NaN/None as missing",
    )
    parser.add_argument("--use-local", action="store_true", help="Use local vLLM endpoints")
    parser.add_argument("--model", help="Model name for judge evaluation")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_repaired.pkl"
    )

    if args.use_local and not args.model:
        parser.error("--use-local requires --model (vLLM model name)")

    if args.resume and output_path.exists():
        print(f"Resuming from {output_path}")
        df = _load_df(output_path)
    else:
        print(f"Loading {input_path}")
        df = _load_df(input_path)

    if args.question_col not in df.columns or args.answer_col not in df.columns:
        missing_cols = [c for c in [args.question_col, args.answer_col] if c not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")

    judge_ids = _load_judge_ids(df, args.judge_files, args.judge_ids_col)
    rubric_funcs = judge_rubrics.load_judges_from_files(args.judge_files)
    rubrics = {jid: func() for jid, func in rubric_funcs.items()}
    _validate_judge_ids(judge_ids, rubrics)

    if args.scores_col not in df.columns:
        raise ValueError(f"Missing required column: {args.scores_col}")

    scores_list: List[List[Optional[float]]] = [
        _normalize_scores(scores, len(judge_ids)) for scores in df[args.scores_col].tolist()
    ]

    treat_zero = not args.keep_zero
    treat_nan = not args.skip_nan
    only_judges = set(args.only_judges) if args.only_judges else None

    progress_path = None
    completed_keys: Optional[Set[int]] = None
    if args.recompute_all or args.progress_path:
        progress_path = Path(args.progress_path) if args.progress_path else output_path.with_suffix(".progress.pkl")
        if args.resume and progress_path.exists():
            with progress_path.open("rb") as f:
                completed_keys = pickle.load(f)
            print(f"Loaded progress from {progress_path} ({len(completed_keys)} completed)")
        elif args.resume:
            print(f"Resume requested but no progress file found at {progress_path}; starting fresh")
        else:
            completed_keys = set()

    tasks = _build_missing_tasks(
        scores_list,
        judge_ids,
        treat_zero=treat_zero,
        treat_nan=treat_nan,
        only_judges=only_judges,
        max_missing=args.max_missing,
        recompute_all=args.recompute_all,
        completed_keys=completed_keys,
    )

    mode = "all scores" if args.recompute_all else "missing scores"
    print(
        f"Found {len(tasks)} evaluations for {mode} across {len(df)} rows "
        f"and {len(judge_ids)} judges"
    )

    if args.dry_run or not tasks:
        return 0

    client = MartianClient(
        default_model=args.model if args.model else "openai/gpt-5-mini",
        use_local=args.use_local,
    )

    questions = ["" if q is None else str(q) for q in df[args.question_col].tolist()]
    answers = ["" if a is None else str(a) for a in df[args.answer_col].tolist()]

    failures: List[Tuple[int, int, str]] = []
    completed = 0
    start_time = time.time()
    judge_count = len(judge_ids)

    def evaluate_task(task: Tuple[int, int]) -> Tuple[int, int, Optional[float], Optional[str]]:
        row_idx, j_idx = task
        judge_id = judge_ids[j_idx]
        rubric = rubrics[judge_id]
        question = questions[row_idx]
        answer = answers[row_idx]
        delay = args.initial_delay
        for attempt in range(args.max_retries):
            try:
                result = client.evaluate_with_rubric(
                    rubric=rubric,
                    question=question,
                    answer=answer,
                )
                score = float(result["score"])
                return row_idx, j_idx, score, None
            except Exception as exc:
                if attempt == args.max_retries - 1:
                    return row_idx, j_idx, None, str(exc)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        return row_idx, j_idx, None, "unknown error"

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            row_idx, j_idx, score, err = future.result()
            if score is not None:
                scores_list[row_idx][j_idx] = score
                if completed_keys is not None:
                    completed_keys.add(row_idx * judge_count + j_idx)
            else:
                failures.append((row_idx, j_idx, err or "unknown error"))

            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed else 0.0
                print(f"Completed {completed}/{len(tasks)} ({rate:.2f} updates/sec)")

            if completed % args.checkpoint_every == 0:
                df[args.scores_col] = scores_list
                _save_df(df, output_path)
                print(f"Checkpoint saved to {output_path}")
                if completed_keys is not None and progress_path is not None:
                    with progress_path.open("wb") as f:
                        pickle.dump(completed_keys, f)
                    print(f"Progress saved to {progress_path}")

    df[args.scores_col] = scores_list
    _save_df(df, output_path)
    print(f"Saved repaired dataset to {output_path}")
    if completed_keys is not None and progress_path is not None:
        with progress_path.open("wb") as f:
            pickle.dump(completed_keys, f)
        print(f"Progress saved to {progress_path}")

    if failures:
        print(f"{len(failures)} evaluations failed after retries")
        sample = failures[:5]
        for row_idx, j_idx, err in sample:
            print(f"  row {row_idx} judge {judge_ids[j_idx]}: {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
