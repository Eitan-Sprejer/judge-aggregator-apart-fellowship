import pickle
from pathlib import Path

results_dir = Path("rebutal_augusto_pt3/results")
total_samples = 0
total_personas = 0
total_errors = 0
error_details = []

for run_id in range(1, 6):
    pkl_file = results_dir / f"run_{run_id}.pkl"
    if not pkl_file.exists():
        continue
    
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    
    model = data.get("model", f"unknown (run {run_id})")
    df = data["df"]
    run_errors = 0
    
    for idx, row in df.iterrows():
        total_samples += 1
        feedback = row.get("human_feedback", {})
        personas = feedback.get("personas", {})
        
        for p_name, p_data in personas.items():
            total_personas += 1
            analysis = p_data.get("analysis", "")
            if isinstance(analysis, str) and analysis.startswith("Error"):
                total_errors += 1
                run_errors += 1
                error_details.append(f"Run {run_id} ({model}), Sample {idx}, Persona {p_name}: {analysis} | {p_data.get('error', '')}")

    print(f"Run {run_id} ({model}): {run_errors} errors")

print(f"\nScanned {total_samples} samples, {total_personas} individual persona scores")
print(f"Total fallback errors found: {total_errors}")

if total_errors > 0:
    print("\nFirst 10 errors:")
    for e in error_details[:10]:
        print(f"  - {e}")
