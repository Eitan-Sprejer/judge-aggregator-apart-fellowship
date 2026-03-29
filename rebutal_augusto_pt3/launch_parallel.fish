#!/usr/bin/env fish
# Launch cross-model persona experiment (4 models in parallel)
# Run 1 (llama-3.3-70b-instruct) is pre-copied from rebutal_augusto

set -l OUTPUT_DIR rebutal_augusto_pt3/results
set -l COMMON_ARGS \
    --n-samples 500 \
    --api-base https://api.withmartian.com/v1 \
    --api-key-env MARTIAN_API_KEY \
    --output-dir $OUTPUT_DIR

mkdir -p $OUTPUT_DIR

echo "🚀 Launching 4 cross-model runs (run 1 = llama already copied)..."

# Run 2: DeepSeek v3.2
set -l logfile "$OUTPUT_DIR/run_{2}.log"
echo "   Starting run 2 (deepseek/deepseek-v3.2) → $logfile"
.venv/bin/python rebutal_augusto_pt3/run_reproducibility.py \
    --run-id 2 --model deepseek/deepseek-v3.2 $COMMON_ARGS \
    > $logfile 2>&1 &

# Run 3: Martian Lobster
set -l logfile "$OUTPUT_DIR/run_{3}.log"
echo "   Starting run 3 (martian/lobster) → $logfile"
.venv/bin/python rebutal_augusto_pt3/run_reproducibility.py \
    --run-id 3 --model martian/lobster $COMMON_ARGS \
    > $logfile 2>&1 &

# Run 4: Mixtral 8x7B
set -l logfile "$OUTPUT_DIR/run_{4}.log"
echo "   Starting run 4 (mistralai/mixtral-8x7b-instruct) → $logfile"
.venv/bin/python rebutal_augusto_pt3/run_reproducibility.py \
    --run-id 4 --model mistralai/mixtral-8x7b-instruct $COMMON_ARGS \
    > $logfile 2>&1 &

# Run 5: Nemotron
set -l logfile "$OUTPUT_DIR/run_{5}.log"
echo "   Starting run 5 (nvidia/nemotron-3-super-120b-a12b) → $logfile"
.venv/bin/python rebutal_augusto_pt3/run_reproducibility.py \
    --run-id 5 --model nvidia/nemotron-3-super-120b-a12b $COMMON_ARGS \
    > $logfile 2>&1 &

echo ""
echo "All 4 runs launched in background."
echo "Monitor with: tail -f rebutal_augusto_pt3/results/run_*.log"
echo "Check progress: ls rebutal_augusto_pt3/results/run_*.pkl"
echo ""

# Wait for all background jobs
echo "Waiting for all runs to finish..."
wait
echo "🎉 All runs complete!"
