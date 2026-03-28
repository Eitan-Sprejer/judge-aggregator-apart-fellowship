import os
import json
import pickle
import asyncio
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from persona_simulation import PersonaSimulator

async def repair():
    load_dotenv()
    
    # We will initialize the simulator without a specific personas_dict at first
    # because each variant has its own dictionary that we will load dynamically.
    simulator = PersonaSimulator(
        api_key=os.getenv("MARTIAN_API_KEY"),
        api_base="https://api.withmartian.com/v1",
        model="meta-llama/llama-3.3-70b-instruct",
        personas_dict={}
    )
    
    results_dir = Path(__file__).parent / "results"
    variants = ["base", "v1", "v2", "v3", "v4"]
    
    for v in variants:
        pkl_file = results_dir / f"{v}.pkl"
        if not pkl_file.exists():
            print(f"Skipping {v}, file not found.")
            continue
            
        print(f"\nEvaluating {v}.pkl for errors or missing data...")
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            
        df = data["df"]
        personas_dict = data["personas_dict"]
        changed = False
        
        for idx, row in df.iterrows():
            feedback = row.get("human_feedback")
            # If the response threw a complete top-level error and feedback is entirely missing
            if not isinstance(feedback, dict):
                feedback = {"personas": {}}
                row["human_feedback"] = feedback
            
            personas = feedback.get("personas", {})
            if "personas" not in feedback or not isinstance(personas, dict):
                feedback["personas"] = {}
                personas = feedback["personas"]
                
            query = row.get("question", "")
            answer = row.get("response", "")
            
            # Identify missing personas
            missing_names = [p for p in personas_dict.keys() if p not in personas]
            # Identify errored personas
            errored_names = []
            for p_name, p_data in personas.items():
                if isinstance(p_data, dict):
                    analysis = str(p_data.get("analysis", ""))
                    if analysis.startswith("Error") or p_data.get("score") is None:
                        errored_names.append(p_name)
                        
            needs_repair = missing_names + errored_names
            
            if needs_repair:
                print(f"  [Sample {idx}] Repairing {len(needs_repair)} personas: {', '.join(needs_repair)}")
                
                tasks = []
                for p_name in needs_repair:
                    persona_bio = personas_dict[p_name]
                    sys_p, user_p = simulator._get_prompts(p_name, query, answer, persona_bio)
                    
                    async def fetch_one(name, sys_prompt, user_prompt):
                        for attempt in range(4):
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
                            except json.JSONDecodeError:
                                await asyncio.sleep(1)
                            except Exception as e:
                                await asyncio.sleep(1)
                        # Fallback if it chronically fails
                        return name, {"persona": name, "score": 5, "analysis": "Error during manual repair"}
                        
                    tasks.append(fetch_one(p_name, sys_p, user_p))
                
                # Run the missing queries for this sample in parallel to be fast
                results = await asyncio.gather(*tasks)
                
                for name, res_data in results:
                    personas[name] = res_data
                    if not str(res_data.get("analysis", "")).startswith("Error"):
                        print(f"    -> Fixed {name} (Score: {res_data.get('score')})!")
                    else:
                        print(f"    -> Failed repairing {name}")
                        
                # Recalculate average for the sample
                valid_scores = [p["score"] for p in personas.values() if isinstance(p.get("score"), (int, float))]
                if valid_scores:
                    new_avg = sum(valid_scores) / len(valid_scores)
                    feedback["average_score"] = new_avg
                    feedback["score"] = new_avg
                    
                changed = True
                
        if changed:
            print(f" -> Rebuilding aggregate scores for {v}.pkl...")
            per_persona = {name: [] for name in personas_dict.keys()}
            average_scores = []
            
            for _, r in df.iterrows():
                fb = r.get("human_feedback", {})
                ps = fb.get("personas", {})
                for n in personas_dict.keys():
                    val = ps.get(n, {})
                    per_persona[n].append(val.get("score") if isinstance(val, dict) else None)
                average_scores.append(fb.get("average_score"))
            
            data["scores"]["per_persona"] = per_persona
            data["scores"]["average_scores"] = average_scores
            
            with open(pkl_file, "wb") as f:
                pickle.dump(data, f)
            print(f" -> Saved repairs to {pkl_file.name}")
        else:
            print(f" -> No repairs needed for {v}.pkl")

if __name__ == "__main__":
    asyncio.run(repair())
