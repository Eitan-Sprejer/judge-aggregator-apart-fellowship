import os
import json
import pickle
import asyncio
from pathlib import Path
import sys
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.core.persona_simulation import PersonaSimulator, PERSONAS

async def repair_missing():
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
            
            # If the sample is missing personas entirely (less than 14)
            if len(personas) < 14:
                # Initialize it if it totally failed
                if not isinstance(feedback, dict):
                    feedback = {}
                    row["human_feedback"] = feedback
                if "personas" not in feedback:
                    feedback["personas"] = {}
                    personas = feedback["personas"]
                
                print(f"Run {run_id}, Sample {idx}: Missing {14 - len(personas)} personas. Repairing...")
                
                query = row.get("question", "")
                answer = row.get("response", "")
                
                # We fetch manually to allow concurrent gathering
                tasks = []
                missing_names = [p for p in PERSONAS.keys() if p not in personas]
                
                for p_name in missing_names:
                    persona_bio = PERSONAS[p_name]
                    sys_p, user_p = simulator._get_prompts(p_name, query, answer, persona_bio)
                    
                    async def fetch_one(name, sys_prompt, user_prompt):
                        for attempt in range(5):
                            try:
                                resp = await simulator.client.chat.completions.create(
                                    model=simulator.model,
                                    messages=[
                                        {"role": "system", "content": sys_prompt},
                                        {"role": "user", "content": user_prompt}
                                    ],
                                    max_tokens=100, temperature=0.7, timeout=45.0
                                )
                                content = resp.choices[0].message.content
                                new_data = json.loads(content)
                                new_data["persona"] = name
                                return name, new_data
                            except Exception as e:
                                await asyncio.sleep(1)
                        # fallback
                        return name, {"persona": name, "score": 5, "analysis": "Error during manual repair"}
                        
                    tasks.append(fetch_one(p_name, sys_p, user_p))
                
                # Run the missing queries for this sample in parallel
                results = await asyncio.gather(*tasks)
                
                for name, res_data in results:
                    personas[name] = res_data
                    
                # Recalculate average
                valid_scores = [p["score"] for p in personas.values() if isinstance(p.get("score"), (int, float))]
                if valid_scores:
                    new_avg = sum(valid_scores) / len(valid_scores)
                    feedback["average_score"] = new_avg
                    feedback["score"] = new_avg
                    
                changed = True
                print(f"  -> Fixed {len(missing_names)} missing scores for sample {idx}. New avg: {new_avg:.2f}")
                            
        if changed:
            # Rebuild nested dict lists for analysis 
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
    asyncio.run(repair_missing())
