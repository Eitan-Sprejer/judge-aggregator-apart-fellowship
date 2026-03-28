#!/usr/bin/env fish
# Launch all 5 wording variants in parallel
# Each run is a separate background process

set -l COMMON_ARGS \
    --n-samples 500 \
    --api-base https://api.withmartian.com/v1 \
    --api-key-env MARTIAN_API_KEY \
    --model meta-llama/llama-3.3-70b-instruct \
    --output-dir rebutal_augusto_pt2/results

echo "🚀 Launching 5 parallel phrasing variants..."

for variant in base v1 v2 v3 v4
    set -l logfile "rebutal_augusto_pt2/results/variant_{$variant}.log"
    echo "   Starting variant $variant → $logfile"
    .venv/bin/python rebutal_augusto_pt2/run_phrasing.py \
        --variant $variant $COMMON_ARGS \
        > $logfile 2>&1 &
end

echo ""
echo "All 5 variants launched in background."
echo "Monitor with: tail -f rebutal_augusto_pt2/results/variant_*.log"
echo ""

# Wait for all background jobs
echo "Waiting for all variants to finish..."
wait
echo "🎉 All wording variants complete! Run analyze_phrasing.py to see results."
