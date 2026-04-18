import pytest
import os
import tempfile
import json
import csv
import yaml
from evalmonkey.evals.local_assets import load_local_evals, EvalScenario

def test_load_local_evals_json():
    data = [
        {
            "id": "json_test",
            "description": "JSON Test",
            "expected_behavior_rubric": "Does JSON work?",
            "input_payload": {"question": "Hello?"}
        }
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w', encoding='utf-8') as f:
        json.dump(data, f)
        temp_path = f.name
        
    try:
        scenarios = load_local_evals(temp_path)
        assert len(scenarios) == 1
        assert scenarios[0].id == "json_test"
        assert scenarios[0].input_payload == {"question": "Hello?"}
    finally:
        os.remove(temp_path)

def test_load_local_evals_csv():
    data = [
        {"id": "csv_test", "description": "CSV Desc", "expected_behavior_rubric": "CSV Rubric", "topic": "science", "question": "What is 2+2?"}
    ]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "description", "expected_behavior_rubric", "topic", "question"])
        writer.writeheader()
        writer.writerows(data)
        temp_path = f.name
        
    try:
        scenarios = load_local_evals(temp_path)
        assert len(scenarios) == 1
        assert scenarios[0].id == "csv_test"
        # Everything not reserved gets pushed into input_payload
        assert scenarios[0].input_payload == {"topic": "science", "question": "What is 2+2?"}
        assert scenarios[0].expected_behavior_rubric == "CSV Rubric"
    finally:
        os.remove(temp_path)

def test_load_local_evals_yaml():
    data = [
        {
            "id": "yaml_test",
            "description": "YAML Test",
            "expected_behavior_rubric": "Does YAML work?",
            "input_payload": {"message": "Hi"}
        }
    ]
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode='w', encoding='utf-8') as f:
        yaml.dump(data, f)
        temp_path = f.name
        
    try:
        scenarios = load_local_evals(temp_path)
        assert len(scenarios) == 1
        assert scenarios[0].id == "yaml_test"
    finally:
        os.remove(temp_path)

def test_load_local_evals_not_found():
    scenarios = load_local_evals("does/not/exist.csv")
    assert scenarios == []
