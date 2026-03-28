import os
import json
import pickle
import asyncio
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.core.persona_simulation import PersonaSimulator, PERSONAS

async def repair():
    load_dotenv()
    simulator = PersonaSimulator(
        api_key=os.getenv("MARTIAN_API_KEY"),
        api_base="https://api.withmartian.com/v1",
        model="meta-llama/llama-3.3-70b-instruct",
    )
    
    results_dir = Path("rebutal_augusto/results")
    
    for run_id in range(1, 6):
        pkl_file = results_dir / f"run_{run_id}.pkl"
        if not pkl_file.exists(): 
            continue
        
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            
        df = data["df"]
        changed = False
        
        for idx, row in df.iterrows():
            feedback = row.get("human_feedback", {})
            personas = feedback.get("personas", {})
            
            for p_name, p_data in personas.items():
                if isinstance(p_data.get("analysis", ""), str) and str(p_data.get("analysis", "")).startswith("Error"):
                    print(f"Repairing Run {run_id}, Sample {idx}, Persona {p_name}...")
                    
                    query = row["question"]
                    answer = row["response"]
                    persona_bio = PERSONAS[p_name]
                    sys_prompt, user_prompt = simulator._get_prompts(p_name, query, answer, persona_bio)
                    
                    for attempt in range(3):
                        try:
                            resp = await simulator.client.chat.completions.create(
                                model=simulator.model,
                                messages=[
                                    {"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                max_tokens=100, 
                                temperature=0.7, 
                                timeout=45.0
                            )
                            content = resp.choices[0].message.content
                            new_data = json.loads(content)
                            new_data["persona"] = p_name
                            
                            # Replace the old error data with the fixed response
                            p_data.clear()
                            p_data.update(new_data)
                            
                            # Recalculate the entire sample's average score
                            valid_scores = [p["score"] for p in personas.values() if isinstance(p.get("score"), (int, float))]
                            if valid_scores:
                                new_avg = sum(valid_scores) / len(valid_scores)
                                feedback["average_score"] = new_avg
                                feedback["score"] = new_avg
                                
                            changed = True
                            print(f"  -> Fixed! New score: {new_data['score']}, Reason: {new_data.get('analysis')[:50]}...")
                            break
                        except Exception as e:
                            print(f"  Attempt {attempt+1} failed: {e}")
                            
        if changed:
            # Rebuild the nested score extractions for analysis script
            per_persona = {name: [] for name in PERSONAS.keys()}
            average_scores = []
            
            for _, r in df.iterrows():
                fb = r.get("human_feedback", {})
                ps = fb.get("personas", {})
                for n in PERSONAS:
                    per_persona[n].append(ps.get(n, {}).get("score"))
                average_scores.append(fb.get("average_score"))
            
            data["scores"]["per_persona"] = per_persona
            data["scores"]["average_scores"] = average_scores
            
            with open(pkl_file, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved repairs to {pkl_file.name}")

if __name__ == "__main__":
    asyncio.run(repair())
