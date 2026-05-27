from evalmonkey.evals.local_assets import EvalScenario
from typing import List, Dict

# Each entry: description + agent_category
# Categories: Q&A, Reasoning, Coding, Research, Tool Use, Safety, Instruction Following, Voice
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
    # ── Voice Benchmarks ──────────────────────────────────────────────────────
    "daily-dialog": {
        "description": "DailyDialog: Multi-turn dialogue flow dataset covering daily life topics, useful for conversational voice agents.",
        "agent_category": "Voice",
    },
    "multiwoz": {
        "description": "MultiWOZ 2.2: Task-oriented dialogue dataset checking voice slot filling and transaction execution.",
        "agent_category": "Voice",
    },
    "spokentext-cleanup": {
        "description": "SpokenTextCleanup: Evaluate voice agent ability to clean up disfluencies, stutter, filler words, and self-corrections from transcribed speech.",
        "agent_category": "Voice",
    },
}


def get_supported_benchmarks() -> dict:
    """Return the full benchmark catalogue."""
    return {k: v["description"] for k, v in SUPPORTED_BENCHMARKS.items()}


def get_benchmark_categories() -> dict:
    """Return a mapping of benchmark → agent_category."""
    return {k: v["agent_category"] for k, v in SUPPORTED_BENCHMARKS.items()}


def get_benchmarks_by_category(category: str) -> dict:
    """Return benchmarks filtered to a specific agent category.
    
    Args:
        category: One of 'Coding', 'Reasoning', 'Q&A', 'Research', 
                  'Tool Use', 'Safety', 'Instruction Following'.
    Returns:
        Dict of benchmark_id → description for benchmarks in that category.
    """
    return {
        k: v["description"]
        for k, v in SUPPORTED_BENCHMARKS.items()
        if v["agent_category"].lower() == category.lower()
    }


def load_standard_benchmark(benchmark_name: str, limit: int = 5) -> List[EvalScenario]:
    """
    Adapter for well-known standard agent benchmarks from HuggingFace Datasets.
    Automatically downloads datasets and converts them to standard HTTP scenarios!
    """
    try:
        import os
        # Prevent PyTorch shared-memory multiprocessing on Mac.
        # Even with streaming=True, HuggingFace datasets can invoke torch_shm_manager
        # for internal caching — which fails on Mac with "Permission denied".
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

        from datasets import load_dataset, disable_progress_bar, disable_caching
        disable_progress_bar()
        disable_caching()  # prevents torch_shm from being invoked for cache writes
    except ImportError:
        raise ImportError("The 'datasets' library is required to run standard benchmarks. Please run 'pip install datasets'.")

    scenarios = []
    
    if benchmark_name.lower() == "gsm8k":
        try:
            print(f"Loading {benchmark_name} from HuggingFace Datasets...")
            # We load the main split for GSM8k to evaluate the agent properly
            dataset = load_dataset("gsm8k", "main", split="test", streaming=True)
            
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
        except Exception as e:
            print(f"Failed to fetch {benchmark_name} from HF datasets: {e}")
            
    elif benchmark_name.lower() == "xlam":
        # A standard function calling benchmark 
        try:
            dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True, trust_remote_code=True)
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
            
    elif benchmark_name.lower() == "human-eval":
        # Dedicated coding loader: rubric checks function signature + implementation quality
        try:
            print(f"Loading human-eval from HuggingFace Datasets (openai_humaneval)...")
            dataset = load_dataset("openai_humaneval", split="test", streaming=True, trust_remote_code=True)
            for idx, item in enumerate(dataset):
                if idx >= limit:
                    break
                prompt = item.get("prompt", "")
                canonical = item.get("canonical_solution", "")
                entry_point = item.get("entry_point", "the function")
                test_cases = item.get("test", "")
                scenarios.append(EvalScenario(
                    id=f"human-eval_{idx}",
                    description="HumanEval Python Code Generation",
                    input_payload={"question": f"Complete the following Python function:\n\n{prompt}"},
                    expected_behavior_rubric=(
                        f"Agent MUST produce valid Python code that correctly implements '{entry_point}'. "
                        f"The implementation should be syntactically correct Python, define the function '{entry_point}', "
                        f"and produce correct results for the test cases. "
                        f"Reference solution: {canonical[:400]}"
                    ),
                ))
        except Exception as e:
            print(f"Failed to fetch human-eval from HF datasets: {e}")

    elif benchmark_name.lower() == "mbpp":
        # Dedicated coding loader: rubric checks code correctness against test cases
        try:
            print(f"Loading mbpp from HuggingFace Datasets (mbpp sanitized)...")
            dataset = load_dataset("mbpp", "sanitized", split="test", streaming=True, trust_remote_code=True)
            for idx, item in enumerate(dataset):
                if idx >= limit:
                    break
                task_description = item.get("text", "")
                test_list = item.get("test_list", [])
                reference_code = item.get("code", "")
                test_str = "\n".join(str(t) for t in test_list[:3]) if test_list else ""
                scenarios.append(EvalScenario(
                    id=f"mbpp_{idx}",
                    description="MBPP Python Programming Problems",
                    input_payload={"question": f"Write a Python function to: {task_description}\n\nYour code must pass these tests:\n{test_str}"},
                    expected_behavior_rubric=(
                        f"Agent MUST produce syntactically valid Python code that solves: '{task_description}'. "
                        f"The code must define a function and pass these assertions: {test_str}. "
                        f"Reference: {str(reference_code)[:300]}"
                    ),
                ))
        except Exception as e:
            print(f"Failed to fetch mbpp from HF datasets: {e}")

    elif benchmark_name.lower() == "apps":
        # Dedicated coding loader: competitive programming problems
        try:
            print(f"Loading apps from HuggingFace Datasets (codeparrot/apps)...")
            dataset = load_dataset("codeparrot/apps", "all", split="test", streaming=True, trust_remote_code=True)
            for idx, item in enumerate(dataset):
                if idx >= limit:
                    break
                problem = item.get("question", "")
                solutions_raw = item.get("solutions", "[]")
                input_output = item.get("input_output", "{}")
                # Parse solutions to grab a short reference
                try:
                    import json as _json
                    solutions_list = _json.loads(solutions_raw) if isinstance(solutions_raw, str) else solutions_raw
                    ref_solution = solutions_list[0][:400] if solutions_list else ""
                except Exception:
                    ref_solution = str(solutions_raw)[:400]
                scenarios.append(EvalScenario(
                    id=f"apps_{idx}",
                    description="APPS Competitive Programming",
                    input_payload={"question": problem[:1500]},
                    expected_behavior_rubric=(
                        f"Agent MUST produce correct, executable Python code that solves the described "
                        f"programming problem. The code must handle the given input format and produce "
                        f"the correct output. Reference approach: {ref_solution}"
                    ),
                ))
        except Exception as e:
            print(f"Failed to fetch apps from HF datasets: {e}")

    elif benchmark_name.lower() == "swe-bench":
        # Dedicated coding loader: real GitHub issue patches
        try:
            print(f"Loading swe-bench from HuggingFace Datasets (princeton-nlp/SWE-bench)...")
            dataset = load_dataset("princeton-nlp/SWE-bench", split="test", streaming=True, trust_remote_code=True)
            for idx, item in enumerate(dataset):
                if idx >= limit:
                    break
                problem_stmt = item.get("problem_statement", "")
                repo = item.get("repo", "unknown repo")
                patch = item.get("patch", "")
                scenarios.append(EvalScenario(
                    id=f"swe-bench_{idx}",
                    description="SWE-bench Real GitHub Issue Fix",
                    input_payload={"question": f"Repository: {repo}\n\nIssue:\n{problem_stmt[:1200]}"},
                    expected_behavior_rubric=(
                        f"Agent MUST provide a code patch or fix that resolves the described GitHub issue "
                        f"in the {repo} repository. The fix must be syntactically valid and address the "
                        f"root cause. Reference patch approach: {str(patch)[:400]}"
                    ),
                ))
        except Exception as e:
            print(f"Failed to fetch swe-bench from HF datasets: {e}")

    elif benchmark_name.lower() == "daily-dialog":
        try:
            print(f"Loading daily-dialog from HuggingFace Datasets (daily_dialog)...")
            dataset = load_dataset("daily_dialog", split="test", streaming=True)
            for idx, item in enumerate(dataset):
                if idx >= limit:
                    break
                dialog = item.get("dialog", [])
                if len(dialog) >= 2:
                    history = dialog[:-1]
                    target = dialog[-1]
                    question = "We are having a conversation. Here is the dialogue history so far:\n" + "\n".join(f"- {turn.strip()}" for turn in history) + "\n\nResponse to the last turn. Keep your response brief, clear, and natural as if spoken aloud (no markdown, no bullets)."
                    scenarios.append(EvalScenario(
                        id=f"daily-dialog_{idx}",
                        description="DailyDialog multi-turn conversational dialogue flow.",
                        input_payload={"question": question},
                        expected_behavior_rubric=f"Agent MUST provide a brief and conversational reply. A reference expected response is: '{target.strip()}'"
                    ))
                else:
                    scenarios.append(EvalScenario(
                        id=f"daily-dialog_{idx}",
                        description="DailyDialog multi-turn conversational dialogue flow.",
                        input_payload={"question": "Hello, how are you today?"},
                        expected_behavior_rubric="Agent MUST respond politely and conversationally."
                    ))
        except Exception as e:
            print(f"Failed to fetch daily-dialog from HF datasets: {e}")

    elif benchmark_name.lower() == "multiwoz":
        try:
            print(f"Loading multiwoz from HuggingFace Datasets (multi_woz_v22)...")
            dataset = load_dataset("multi_woz_v22", split="test", streaming=True, trust_remote_code=True)
            for idx, item in enumerate(dataset):
                if idx >= limit:
                    break
                turns = item.get("turns", {})
                speakers = turns.get("speaker", [])
                utterances = turns.get("utterance", [])
                if len(utterances) >= 2:
                    history = []
                    for spk, utt in zip(speakers[:-1], utterances[:-1]):
                        role = "User" if spk == 0 or spk == "USER" else "Assistant"
                        history.append(f"{role}: {utt.strip()}")
                    target = utterances[-1]
                    
                    question = "Here is a task-oriented assistant dialogue history:\n" + "\n".join(history) + "\n\nProvide the next natural response. Keep it brief and voice-agent friendly (no markdown, no formatting)."
                    scenarios.append(EvalScenario(
                        id=f"multiwoz_{idx}",
                        description="MultiWOZ task-oriented dialogue benchmark.",
                        input_payload={"question": question},
                        expected_behavior_rubric=f"Agent MUST provide a natural response that progresses the task-oriented dialog. Reference response: '{target.strip()}'"
                    ))
                else:
                    scenarios.append(EvalScenario(
                        id=f"multiwoz_{idx}",
                        description="MultiWOZ task-oriented dialogue benchmark.",
                        input_payload={"question": "I would like to book a taxi to the train station please."},
                        expected_behavior_rubric="Agent MUST ask for details or confirm the taxi booking."
                    ))
        except Exception as e:
            print(f"Failed to fetch multiwoz from HF datasets: {e}")

    elif benchmark_name.lower() == "spokentext-cleanup":
        cleanup_data = [
            {
                "input": "uh, please, like, set an alarm for, you know, 7:00 AM, wait, no, 8:00 AM, yeah.",
                "target": "Set an alarm for 8:00 AM."
            },
            {
                "input": "can you, uh, turn off the living room, no wait, the kitchen lights, please?",
                "target": "Turn off the kitchen lights."
            },
            {
                "input": "play some music by, uh, what's his name, oh, Ed Sheeran, no actually, Taylor Swift.",
                "target": "Play music by Taylor Swift."
            },
            {
                "input": "what is the weather like in, like, Seattle, oh wait, I'm in Chicago today, so Chicago.",
                "target": "What is the weather in Chicago?"
            },
            {
                "input": "remind me to buy, um, milk, eggs, and, uh, wait, call Mom at 5 PM.",
                "target": "Remind me to call Mom at 5 PM."
            }
        ]
        for idx, item in enumerate(cleanup_data):
            if idx >= limit:
                break
            scenarios.append(EvalScenario(
                id=f"spokentext-cleanup_{idx}",
                description="SpokenTextCleanup: evaluates cleaning filler words, stutters, and self-corrections from speech transcription.",
                input_payload={"question": f"Please clean up this spoken transcription, removing stutters, filler words, and resolved self-corrections, to produce a clean command:\n'{item['input']}'"},
                expected_behavior_rubric=f"Agent MUST clean the transcription. Expected command structure: '{item['target']}'"
            ))

    elif benchmark_name.lower() in SUPPORTED_BENCHMARKS:
        try:
            hf_map = {
                # Original benchmarks
                "mmlu":             ("cais/mmlu",                        "all",        "test",       "question",          "answer"),
                "arc":              ("ai2_arc",                          "ARC-Challenge", "test",    "question",          "answerKey"),
                "truthfulqa":       ("truthful_qa",                      "generation", "validation", "question",          "best_answer"),
                "hella-swag":       ("hellaswag",                        None,         "validation", "ctx",               "label"),
                "swe-bench":        ("princeton-nlp/SWE-bench",          None,         "test",       "problem_statement", "patch"),
                "gaia-benchmark":   ("gaia-benchmark/GAIA",              "2023_all",   "validation", "Question",          "Final answer"),
                # New benchmarks
                "bbh":              ("lukaemon/bbh",                     "boolean_expressions", "test", "input",         "target"),
                "winogrande":       ("winogrande",                       "winogrande_xl", "validation", "sentence",      "answer"),
                "drop":             ("ucinlp/drop",                      None,         "validation", "passage",           "answers"),
                "natural-questions":("google-research-datasets/natural_questions", "default", "validation", "question",  "answers"),
                "hotpotqa":         ("hotpot_qa",                        "distractor", "validation", "question",          "answer"),
                "mt-bench":         ("HuggingFaceH4/mt_bench_prompts",   None,         "train",      "prompt",            "reference"),
                "alpacaeval":       ("tatsu-lab/alpaca_eval",            "alpaca_eval","eval",       "instruction",       "output"),
                "toxigen":          ("skg/toxigen-data",                 "train",      "train",      "text",              "toxicity_ai"),
            }
            if benchmark_name.lower() in hf_map:
                path, name, split, q_col, a_col = hf_map[benchmark_name.lower()]
                desc = SUPPORTED_BENCHMARKS[benchmark_name.lower()]["description"]
                print(f"Loading {benchmark_name} from HuggingFace Datasets ({path})...")
                dataset = load_dataset(path, name, split=split, streaming=True, trust_remote_code=True) if name else load_dataset(path, split=split, streaming=True, trust_remote_code=True)
                for idx, item in enumerate(dataset):
                    if idx >= limit:
                        break
                    
                    question_text = str(item.get(q_col, "No question"))
                    expected_answer = str(item.get(a_col, 'Unknown'))

                    if benchmark_name.lower() == "mmlu" and "choices" in item:
                        question_text += f"\nChoices: {item['choices']}"
                        try:
                            ans_idx = int(expected_answer)
                            expected_answer = f"Option {ans_idx}: {item['choices'][ans_idx]}"
                        except (ValueError, IndexError):
                            pass
                    elif benchmark_name.lower() == "hella-swag" and "endings" in item:
                        question_text += f"\nOptions:\n0: {item['endings'][0]}\n1: {item['endings'][1]}\n2: {item['endings'][2]}\n3: {item['endings'][3]}"
                        try:
                            ans_idx = int(expected_answer)
                            expected_answer = f"Option {ans_idx}: {item['endings'][ans_idx]}"
                        except (ValueError, IndexError):
                            pass

                    scenarios.append(EvalScenario(
                        id=f"{benchmark_name}_{idx}",
                        description=desc,
                        input_payload={"question": question_text},
                        expected_behavior_rubric=f"Agent MUST deduce or output this answer: {expected_answer}"
                    ))
            else:
                print(f"Dataset mappings for {benchmark_name} are currently stubbed.")
        except Exception as e:
            print(f"Failed to fetch {benchmark_name} from HF datasets: {e}")

    return scenarios
