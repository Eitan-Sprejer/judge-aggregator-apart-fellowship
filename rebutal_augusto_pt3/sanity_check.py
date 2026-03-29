import pickle
from pathlib import Path
import random

results_dir = Path("rebutal_augusto/results")

# Load all runs into memory
runs = []
for run_id in range(1, 6):
    with open(results_dir / f"run_{run_id}.pkl", "rb") as f:
        runs.append(pickle.load(f)["df"])

# Pick 2 random sample indices (keeping it consistent by seeding or just picking static ones for clarity)
sample_indices = [10, 420] 

print("==================================================")
print("             SANITY CHECK: RAW SCORES             ")
print("==================================================\n")

for idx in sample_indices:
    row = runs[0].iloc[idx]
    question = row.get("question", "")[:80].replace("\n", " ") + "..."
    answer = row.get("response", "")[:80].replace("\n", " ") + "..."
    
    print(f"🔹 SAMPLE {idx}")
    print(f"  Q: {question}")
    print(f"  A: {answer}")
    print("-" * 50)
    
    # Get persona names from Run 1
    personas = runs[0].iloc[idx]["human_feedback"]["personas"].keys()
    
    # We will pick 4 personas to keep the output readable
    personas_to_show = list(personas)[:4]
    
    # Print header
    header = f"{'Persona':<20} | " + " | ".join([f"Run {i}" for i in range(1, 6)]) + " | Variance"
    print(header)
    print("-" * len(header))
    
    for p in personas_to_show:
        scores = []
        for run_df in runs:
            # Safely extract the score for this persona in this run
            score = run_df.iloc[idx]["human_feedback"]["personas"].get(p, {}).get("score", "X")
            scores.append(score)
            
        str_scores = [f"{s:>5}" for s in scores]
        
        # Calculate variance/range if possible
        valid_scores = [s for s in scores if isinstance(s, (int, float))]
        if len(valid_scores) > 0:
            s_min, s_max = min(valid_scores), max(valid_scores)
            s_range = s_max - s_min
            var_str = f"Range: {s_range}" if s_range > 0 else "STABLE"
        else:
            var_str = "N/A"
            
        print(f"{p:<20} | " + " | ".join(str_scores) + f" | ➔ {var_str}")
        
    print("\n")
