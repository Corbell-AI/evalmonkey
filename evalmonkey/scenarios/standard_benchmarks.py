from evalmonkey.evals.local_assets import EvalScenario
from typing import List

SUPPORTED_BENCHMARKS = {
    "gsm8k": "Grade School Math word problems focusing on multi-step reasoning capabilities.",
    "xlam": "XLAM Function Calling 60k: Tests agent tool execution logic and parameter structuring.",
    "swe-bench": "SWE-Bench: Resolving real-world GitHub issues for coding agents.",
    "gaia-benchmark": "GAIA: General AI Assistants testing on real-world web/tool multi-step tasks.",
    "webarena": "WebArena: Highly interactive computer usage and browser manipulation.",
    "human-eval": "HumanEval: Fundamental Python code generation from docstrings.",
    "mmlu": "Massive Multitask Language Understanding: Broad generalized knowledge across 57 subjects.",
    "arc": "AI2 Reasoning Challenge: Complex grade-school science questions.",
    "truthfulqa": "TruthfulQA: Tests whether an agent mimics human falsehoods or hallucination.",
    "hella-swag": "HellaSwag: Commonsense natural language inferences."
}

def get_supported_benchmarks() -> dict:
    return SUPPORTED_BENCHMARKS

def load_standard_benchmark(benchmark_name: str, limit: int = 5) -> List[EvalScenario]:
    """
    Adapter for well-known standard agent benchmarks from HuggingFace Datasets.
    Automatically downloads datasets and converts them to standard HTTP scenarios!
    """
    scenarios = []
    
    if benchmark_name.lower() == "gsm8k":
        try:
            from datasets import load_dataset
            print(f"Loading {benchmark_name} from HuggingFace Datasets...")
            # We load the main split for GSM8k to evaluate the agent properly
            dataset = load_dataset("gsm8k", "main", split="test")
            
            for idx, item in enumerate(dataset):
                if idx >= limit:
                    break
                    
                # Parsing the ground truth answer out of the GSM8k target text
                target_str = item["answer"].split("####")[1].strip() if "####" in item["answer"] else item["answer"]
                
                scenarios.append(EvalScenario(
                    id=f"gsm8k_{idx}",
                    description="GSM8K Math Agent Benchmark",
                    input_payload={"question": item["question"]},
                    expected_behavior_rubric=f"The agent MUST use its reasoning or tools to mathematically deduce and return EXACTLY this answer logic: {target_str}."
                ))
        except ImportError:
            print("datasets library not installed, run pip install datasets")
        except Exception as e:
            print(f"Failed to fetch {benchmark_name} from HF datasets: {e}")
            
    elif benchmark_name.lower() == "xlam":
        # A standard function calling benchmark 
        try:
            from datasets import load_dataset
            dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
            for idx, item in enumerate(dataset):
                if idx >= limit:
                    break
                scenarios.append(EvalScenario(
                    id=f"xlam_{idx}",
                    description="Function Calling Agent Benchmark",
                    input_payload={"prompt": item["query"], "tools": item["tools"]},
                    expected_behavior_rubric=f"Agent MUST structure a precise tool call matching: {item['answers']}"
                ))
        except Exception as e:
            print(f"Failed to fetch XLAM from HF datasets: {e}")
            
    elif benchmark_name.lower() in SUPPORTED_BENCHMARKS:
        try:
            from datasets import load_dataset
            hf_map = {
                "mmlu": ("cais/mmlu", "all", "test", "question", "answer"),
                "arc": ("ai2_arc", "ARC-Challenge", "test", "question", "answerKey"),
                "truthfulqa": ("truthful_qa", "generation", "validation", "question", "best_answer"),
                "hella-swag": ("hellaswag", None, "validation", "ctx", "label"),
                "human-eval": ("openai_humaneval", None, "test", "prompt", "canonical_solution"),
                "swe-bench": ("princeton-nlp/SWE-bench", None, "test", "problem_statement", "patch"),
                "gaia-benchmark": ("gaia-benchmark/GAIA", "2023_all", "validation", "Question", "Final answer")
            }
            if benchmark_name.lower() in hf_map:
                path, name, split, q_col, a_col = hf_map[benchmark_name.lower()]
                print(f"Loading {benchmark_name} from HuggingFace Datasets ({path})...")
                dataset = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
                for idx, item in enumerate(dataset):
                    if idx >= limit:
                        break
                    scenarios.append(EvalScenario(
                        id=f"{benchmark_name}_{idx}",
                        description=SUPPORTED_BENCHMARKS[benchmark_name.lower()],
                        input_payload={"question": item.get(q_col, "No question")},
                        expected_behavior_rubric=f"Agent MUST deduce or output this answer: {item.get(a_col, 'Unknown')}"
                    ))
            else:
                # Fallback for webarena etc
                print(f"Dataset mappings for {benchmark_name} are currently stubbed.")
        except Exception as e:
            print(f"Failed to fetch {benchmark_name} from HF datasets: {e}")

    return scenarios
