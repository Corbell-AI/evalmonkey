from evalmonkey.evals.local_assets import EvalScenario
from typing import List, Dict

# Each entry: description + agent_category
# Categories: Q&A, Reasoning, Coding, Research, Tool Use, Safety, Instruction Following
SUPPORTED_BENCHMARKS: Dict[str, Dict[str, str]] = {
    # ── Original 10 ──────────────────────────────────────────────────────────
    "gsm8k": {
        "description": "Grade School Math word problems focusing on multi-step reasoning capabilities.",
        "agent_category": "Reasoning",
    },
    "xlam": {
        "description": "XLAM Function Calling 60k: Tests agent tool execution logic and parameter structuring.",
        "agent_category": "Tool Use",
    },
    "swe-bench": {
        "description": "SWE-Bench: Resolving real-world GitHub issues for coding agents.",
        "agent_category": "Coding",
    },
    "gaia-benchmark": {
        "description": "GAIA: General AI Assistants testing on real-world web/tool multi-step tasks.",
        "agent_category": "Research",
    },
    "webarena": {
        "description": "WebArena: Highly interactive computer usage and browser manipulation.",
        "agent_category": "Research",
    },
    "human-eval": {
        "description": "HumanEval: Fundamental Python code generation from docstrings.",
        "agent_category": "Coding",
    },
    "mmlu": {
        "description": "Massive Multitask Language Understanding: Broad generalized knowledge across 57 subjects.",
        "agent_category": "Q&A",
    },
    "arc": {
        "description": "AI2 Reasoning Challenge: Complex grade-school science questions.",
        "agent_category": "Reasoning",
    },
    "truthfulqa": {
        "description": "TruthfulQA: Tests whether an agent mimics human falsehoods or hallucination.",
        "agent_category": "Safety",
    },
    "hella-swag": {
        "description": "HellaSwag: Commonsense natural language inferences.",
        "agent_category": "Reasoning",
    },
    # ── New 10 ───────────────────────────────────────────────────────────────
    "bbh": {
        "description": "BIG-Bench Hard: 23 hard reasoning tasks from BIG-Bench where LLMs fall below human baselines.",
        "agent_category": "Reasoning",
    },
    "winogrande": {
        "description": "WinoGrande: Commonsense pronoun-resolution problems designed to defeat statistical shortcuts.",
        "agent_category": "Q&A",
    },
    "drop": {
        "description": "DROP: Discrete Reasoning Over Paragraphs – reading comprehension with numerical & date math.",
        "agent_category": "Research",
    },
    "natural-questions": {
        "description": "Natural Questions: Real Google search queries with Wikipedia passage answers.",
        "agent_category": "Q&A",
    },
    "hotpotqa": {
        "description": "HotpotQA: Multi-hop reasoning requiring evidence from two Wikipedia paragraphs.",
        "agent_category": "Research",
    },
    "mbpp": {
        "description": "MBPP: Mostly Basic Programming Problems – entry-level Python function synthesis.",
        "agent_category": "Coding",
    },
    "apps": {
        "description": "APPS: Automated Programming Progress Standard – competitive & interview-style code challenges.",
        "agent_category": "Coding",
    },
    "mt-bench": {
        "description": "MT-Bench: Multi-turn conversation quality benchmark across writing, roleplay, reasoning, and STEM.",
        "agent_category": "Instruction Following",
    },
    "alpacaeval": {
        "description": "AlpacaEval: Instruction-following quality judged via GPT-4 head-to-head comparisons.",
        "agent_category": "Instruction Following",
    },
    "toxigen": {
        "description": "ToxiGen: Detects whether agents generate or amplify hateful/toxic content across 13 groups.",
        "agent_category": "Safety",
    },
}


def get_supported_benchmarks() -> dict:
    """Return the full benchmark catalogue."""
    return {k: v["description"] for k, v in SUPPORTED_BENCHMARKS.items()}


def get_benchmark_categories() -> dict:
    """Return a mapping of benchmark → agent_category."""
    return {k: v["agent_category"] for k, v in SUPPORTED_BENCHMARKS.items()}


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
                # Original benchmarks
                "mmlu":             ("cais/mmlu",                        "all",        "test",       "question",          "answer"),
                "arc":              ("ai2_arc",                          "ARC-Challenge", "test",    "question",          "answerKey"),
                "truthfulqa":       ("truthful_qa",                      "generation", "validation", "question",          "best_answer"),
                "hella-swag":       ("hellaswag",                        None,         "validation", "ctx",               "label"),
                "human-eval":       ("openai_humaneval",                 None,         "test",       "prompt",            "canonical_solution"),
                "swe-bench":        ("princeton-nlp/SWE-bench",          None,         "test",       "problem_statement", "patch"),
                "gaia-benchmark":   ("gaia-benchmark/GAIA",              "2023_all",   "validation", "Question",          "Final answer"),
                # New benchmarks
                "bbh":              ("lukaemon/bbh",                     "boolean_expressions", "test", "input",         "target"),
                "winogrande":       ("winogrande",                       "winogrande_xl", "validation", "sentence",      "answer"),
                "drop":             ("ucinlp/drop",                      None,         "validation", "passage",           "answer"),
                "natural-questions":("google-research-datasets/natural_questions", "default", "validation", "question",  "answers"),
                "hotpotqa":         ("hotpot_qa",                        "distractor", "validation", "question",          "answer"),
                "mbpp":             ("mbpp",                             "sanitized",  "test",       "text",              "code"),
                "apps":             ("codeparrot/apps",                  "all",        "test",       "question",          "solutions"),
                "mt-bench":         ("HuggingFaceH4/mt_bench_prompts",   None,         "train",      "prompt",            "reference"),
                "alpacaeval":       ("tatsu-lab/alpaca_eval",            "alpaca_eval","eval",       "instruction",       "output"),
                "toxigen":          ("skg/toxigen-data",                 "train",      "train",      "text",              "toxicity_ai"),
            }
            if benchmark_name.lower() in hf_map:
                path, name, split, q_col, a_col = hf_map[benchmark_name.lower()]
                desc = SUPPORTED_BENCHMARKS[benchmark_name.lower()]["description"]
                print(f"Loading {benchmark_name} from HuggingFace Datasets ({path})...")
                dataset = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
                for idx, item in enumerate(dataset):
                    if idx >= limit:
                        break
                    scenarios.append(EvalScenario(
                        id=f"{benchmark_name}_{idx}",
                        description=desc,
                        input_payload={"question": str(item.get(q_col, "No question"))},
                        expected_behavior_rubric=f"Agent MUST deduce or output this answer: {item.get(a_col, 'Unknown')}"
                    ))
            else:
                # Fallback for webarena etc.
                print(f"Dataset mappings for {benchmark_name} are currently stubbed.")
        except Exception as e:
            print(f"Failed to fetch {benchmark_name} from HF datasets: {e}")

    return scenarios
