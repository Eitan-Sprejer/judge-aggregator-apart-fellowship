import pickle
from pathlib import Path

total_missing = 0
for run_id in range(1, 6):
    file_path = Path(f"rebutal_augusto/results/run_{run_id}.pkl")
    if not file_path.exists(): continue
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        
    df = data["df"]
    
    missing_indices = []
    for idx, row in df.iterrows():
        fb = row.get("human_feedback", {})
        personas = fb.get("personas", {})
        
        if len(personas) < 14:
            missing_indices.append(idx)
            total_missing += (14 - len(personas))
            
    print(f"Run {run_id} completely missing sample indices: {missing_indices}")

print(f"Total missing individual scores: {total_missing}")
