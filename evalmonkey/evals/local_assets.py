import yaml
import json
import csv
from pydantic import BaseModel
from typing import List, Optional

class EvalScenario(BaseModel):
    id: str
    description: str
    input_payload: dict
    expected_behavior_rubric: str
    target_endpoint: Optional[str] = None

def load_local_evals(filepath: str) -> List[EvalScenario]:
    """
    Loads Bring-Your-Own evaluation assets from YAML, JSON, or CSV.
    """
    try:
        data = []
        if filepath.endswith(".csv"):
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse core fields, shove the rest into input_payload
                    item_id = row.get("id", str(len(data)))
                    desc = row.get("description", "")
                    rubric = row.get("expected_behavior_rubric", "")
                    endpoint = row.get("target_endpoint", None)
                    
                    payload = {}
                    for k, v in row.items():
                        if k and k not in ["id", "description", "expected_behavior_rubric", "target_endpoint"]:
                            payload[k] = v
                    
                    data.append({
                        "id": item_id,
                        "description": desc,
                        "expected_behavior_rubric": rubric,
                        "target_endpoint": endpoint,
                        "input_payload": payload
                    })
        elif filepath.endswith(".json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
        if not data or not isinstance(data, list):
            return []
            
        scenarios = []
        for item in data:
            scenarios.append(EvalScenario(**item))
        return scenarios
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading custom evaluations from {filepath}: {e}")
        return []
