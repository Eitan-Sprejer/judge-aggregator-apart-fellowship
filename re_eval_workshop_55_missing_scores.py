#!/usr/bin/env python3
"""
Re-evaluate missing Workshop UltraFeedback 55-judge scores (0.0) using local vLLM.
"""

import argparse
import json
import logging
import math
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import yaml
from openai import OpenAI


DEFAULT_INPUT = "datasets/ultrafeedback_workshop_55_judges.pkl"
DEFAULT_OUTPUT = "datasets/ultrafeedback_workshop_55_judges_repaired.pkl"
DEFAULT_JUDGE_FILE = (
    "experiments/track3_automated_selection/generated_judges/"
    "all-judges-decomposed-20251111T005856Z.yaml"
)

VLLM_URLS = ["http://localhost:8000/v1", "http://localhost:8001/v1"]
DEFAULT_MODEL = "nvidia/Llama-3.3-70B-Instruct-FP8"
DEFAULT_TIMEOUT = 30

logger = logging.getLogger("workshop_missing_scores")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)


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


def _load_df(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data if not isinstance(data, list) else __import__("pandas").DataFrame(data)


def _save_df(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(df, f)


def load_judges(judge_file: Path):
    with judge_file.open("r") as f:
        data = yaml.safe_load(f)
    judges = data["judges"]
    return judges, {j["id"]: j for j in judges}


def build_prompt(judge, question, response) -> str:
    criteria_text = ""
    for c in judge.get("criteria", []):
        rng = c.get("range", [0, 0])
        label = c.get("label", "")
        indicators = c.get("indicators", [])
        criteria_text += f"\n{rng[0]}-{rng[1]}: {label}\n"
        for ind in indicators:
            criteria_text += f"  - {ind}\n"

    prompt = f"""You are evaluating an AI assistant's response.

Judge: {judge['name']}
Description: {judge['description']}
Score Range: {judge['score_range'][0]} to {judge['score_range'][1]}

Criteria:
{criteria_text}

User Question:
{question}

Assistant Response:
{response}

Provide your score following the system output instructions."""

    return prompt


def evaluate_single(client, judge, question, response, model: str, timeout: int) -> float:
    prompt = build_prompt(judge, question, response)
    system_message = (
        "You MUST respond with ONLY a valid JSON object in this exact format: "
        "{\"score\": <number>} . No extra keys or text."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=32,
        timeout=timeout,
        extra_body={
            "guided_json": {
                "type": "object",
                "properties": {"score": {"type": "number"}},
                "required": ["score"],
            }
        },
    )
    content = completion.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty response")
    content = content.strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse JSON from: {content[:120]}")
        payload = json.loads(match.group(0))

    if "score" not in payload:
        raise ValueError(f"Missing score in response: {content[:120]}")
    return float(payload["score"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate missing Workshop UltraFeedback 55-judge scores."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input dataset pickle path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output dataset pickle path")
    parser.add_argument("--judge-file", default=DEFAULT_JUDGE_FILE, help="Judges YAML file")
    parser.add_argument("--scores-col", default="judge_scores_55", help="Scores column name")
    parser.add_argument("--judge-ids-col", default="judge_ids_55", help="Judge IDs column name")
    parser.add_argument("--question-col", default="instruction", help="Question column name")
    parser.add_argument("--answer-col", default="answer", help="Answer column name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name for vLLM")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout")
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per judge")
    parser.add_argument("--initial-delay", type=float, default=1.0, help="Initial retry delay (s)")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save every N updates")
    parser.add_argument("--resume", action="store_true", help="Resume from output if it exists")
    parser.add_argument("--only-judges", nargs="*", help="Only re-evaluate these judge IDs")
    parser.add_argument(
        "--max-missing",
        type=int,
        help="Limit number of evaluations to run (applies to any mode)",
    )
    parser.add_argument("--keep-zero", action="store_true", help="Do not treat 0.0 as missing")
    parser.add_argument("--skip-nan", action="store_true", help="Do not treat NaN/None as missing")
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Re-evaluate all scores, ignoring missing detection",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.resume and output_path.exists():
        logger.info(f"Resuming from {output_path}")
        df = _load_df(output_path)
    else:
        logger.info(f"Loading {input_path}")
        df = _load_df(input_path)

    for col in [args.scores_col, args.question_col, args.answer_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    judges_list, judges_by_id = load_judges(Path(args.judge_file))

    if args.judge_ids_col in df.columns and df[args.judge_ids_col].notna().any():
        judge_ids = df[args.judge_ids_col].iloc[0]
        if not isinstance(judge_ids, list):
            raise ValueError(f"{args.judge_ids_col} is not a list")
        if not all(df[args.judge_ids_col].apply(lambda x: x == judge_ids)):
            raise ValueError(f"{args.judge_ids_col} varies across rows")
    else:
        judge_ids = [j["id"] for j in judges_list]
        df[args.judge_ids_col] = [judge_ids] * len(df)

    missing_judges = [jid for jid in judge_ids if jid not in judges_by_id]
    if missing_judges:
        raise ValueError(f"Judge IDs missing from YAML: {missing_judges}")

    ordered_judges = [judges_by_id[jid] for jid in judge_ids]

    scores_list: List[List[Optional[float]]] = df[args.scores_col].tolist()
    if any(not isinstance(row, list) for row in scores_list):
        raise ValueError(f"{args.scores_col} must contain lists of scores")

    treat_zero = not args.keep_zero
    treat_nan = not args.skip_nan
    only_judges = set(args.only_judges) if args.only_judges else None

    tasks: List[Tuple[int, int]] = []
    for row_idx, row_scores in enumerate(scores_list):
        if len(row_scores) != len(judge_ids):
            raise ValueError(f"Row {row_idx} score length {len(row_scores)} != {len(judge_ids)}")
        for j_idx, (jid, score) in enumerate(zip(judge_ids, row_scores)):
            if only_judges and jid not in only_judges:
                continue
            if args.recompute_all or _is_missing(score, treat_zero, treat_nan):
                tasks.append((row_idx, j_idx))
                if args.max_missing and len(tasks) >= args.max_missing:
                    break
        if args.max_missing and len(tasks) >= args.max_missing:
            break

    mode = "all scores" if args.recompute_all else "missing scores"
    logger.info(f"Found {len(tasks)} evaluations for {mode} across {len(df)} rows and {len(judge_ids)} judges")
    if not tasks:
        return 0

    questions = ["" if q is None else str(q) for q in df[args.question_col].tolist()]
    answers = ["" if a is None else str(a) for a in df[args.answer_col].tolist()]

    failures: List[Tuple[int, int, str]] = []
    completed = 0
    start_time = time.time()

    clients = [OpenAI(base_url=url, api_key="dummy") for url in VLLM_URLS]

    def evaluate_task(task: Tuple[int, int]):
        row_idx, j_idx = task
        judge = ordered_judges[j_idx]
        question = questions[row_idx]
        answer = answers[row_idx]
        client = clients[(row_idx + j_idx) % len(clients)]

        delay = args.initial_delay
        for attempt in range(args.max_retries):
            try:
                score = evaluate_single(client, judge, question, answer, args.model, args.timeout)
                return row_idx, j_idx, score, None
            except Exception as exc:
                logger.warning(
                    "Attempt %d failed for %s (row %d): %s",
                    attempt + 1,
                    judge["id"],
                    row_idx,
                    exc,
                )
                if attempt == args.max_retries - 1:
                    return row_idx, j_idx, math.nan, str(exc)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        return row_idx, j_idx, math.nan, "unknown error"

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            row_idx, j_idx, score, err = future.result()
            if err is None:
                scores_list[row_idx][j_idx] = score
            else:
                scores_list[row_idx][j_idx] = math.nan
                failures.append((row_idx, j_idx, err or "unknown error"))
                logger.error(
                    "Failed after retries for %s (row %d): %s",
                    judge_ids[j_idx],
                    row_idx,
                    err or "unknown error",
                )

            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed else 0.0
                logger.info(f"Completed {completed}/{len(tasks)} ({rate:.2f} updates/sec)")

            if completed % args.checkpoint_every == 0:
                df[args.scores_col] = scores_list
                _save_df(df, output_path)
                logger.info(f"Checkpoint saved to {output_path}")

    df[args.scores_col] = scores_list
    _save_df(df, output_path)
    logger.info(f"Saved repaired dataset to {output_path}")

    if failures:
        logger.error(f"{len(failures)} evaluations failed after retries")
        for row_idx, j_idx, err in failures[:5]:
            logger.error(f"  row {row_idx} judge {judge_ids[j_idx]}: {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
