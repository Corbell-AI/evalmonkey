import yaml
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
    Loads Bring-Your-Own evaluation assets from a YAML file.
    """
    try:
        with open(filepath, 'r') as f:
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
