#!/usr/bin/env fish
# Launch all 5 reproducibility runs in parallel
# Each run is a separate background process

set -l COMMON_ARGS \
    --n-samples 500 \
    --api-base https://api.withmartian.com/v1 \
    --api-key-env MARTIAN_API_KEY \
    --model meta-llama/llama-3.3-70b-instruct \
    --output-dir rebutal_augusto/results

echo "🚀 Launching 5 parallel runs..."

for run_id in 1 2 3 4 5
    set -l logfile "rebutal_augusto/results/run_{$run_id}.log"
    echo "   Starting run $run_id → $logfile"
    .venv/bin/python rebutal_augusto/run_reproducibility.py \
        --run-id $run_id $COMMON_ARGS \
        > $logfile 2>&1 &
end

echo ""
echo "All 5 runs launched in background."
echo "Monitor with: tail -f rebutal_augusto/results/run_*.log"
echo "Check progress: ls rebutal_augusto/results/run_*.pkl"
echo ""

# Wait for all background jobs
echo "Waiting for all runs to finish..."
wait
echo "🎉 All runs complete!"
