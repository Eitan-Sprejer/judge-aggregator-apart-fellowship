# Re-evaluation Summary

## Overview
- Workshop UltraFeedback 55-judge scores were fully recomputed with local vLLM using guided JSON output enforcement.
- HelpSteer2 parent + children (30 judges total) were fully recomputed with resume/progress tracking.

## Scripts
- `re_eval_workshop_55_missing_scores.py`
  - Uses guided JSON (`{"score": <number>}`) to prevent parse failures.
  - Reuses OpenAI clients to avoid "too many open files" errors.
  - Supports full recompute with `--recompute-all`.
- `re_eval_helpsteer2_missing_scores.py`
  - Supports full recompute with `--recompute-all`.
  - Supports resume using a progress file (`--progress-path`).

## Data Outputs
- Workshop UltraFeedback:
  - `datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl`
    - Full recompute of 55 judges for 2000 rows.
    - Columns include `judge_scores_55` and `judge_ids_55`.
  - `re_eval_workshop_55_missing_scores_full.log`
    - Full run log for the workshop recompute.
- HelpSteer2:
  - `datasets/helpsteer2_full_30_judges_recomputed.pkl`
    - Full recompute of 30 judges (5 parents + 25 children) for 20324 rows.
    - Columns include `judge_scores` and `judge_ids`.
  - `datasets/helpsteer2_full_30_judges_recomputed.progress.pkl`
    - Resume tracking for the full recompute.
  - `re_eval_helpsteer2_full_recompute.log`
    - Full run log for the HelpSteer recompute and resume attempts.

## Notes
- Workshop recompute completed and saved results; see the log for any retries.
- HelpSteer recompute completed and saved results; the last resume attempt reported 2 remaining failures after retries (logged in `re_eval_helpsteer2_full_recompute.log`).
- Both outputs preserve the original dataset schema and use the same judge ordering stored in the `judge_ids` columns.
