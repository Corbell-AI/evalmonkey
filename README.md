<p align="center">
  <img src="assets/evalmonkey-logo.png" alt="EvalMonkey Logo" width="400"/>
</p>

# EvalMonkey

<p align="center">
  <b>Agent Benchmarking & Chaos Engineering Framework</b><br>
  <i>"Don't just trust your agent. Prove it works. Then break it."</i>
</p>

https://github.com/Corbell-AI/evalmonkey/raw/main/assets/eval_monkey_github_video.mp4


## Overview
Agents are fundamentally non-deterministic. They rely on external APIs, tool loops, and massive context windows.
**EvalMonkey** is the ultimate, strictly local, open-source execution harness that enables developers to:
1. 🎯 **Benchmark Capabilities**: Run standard Agent benchmark datasets against your agent endpoints natively!
2. 🔥 **Inject Chaos**: Mutate headers, spike latency, and corrupt schemas dynamically to prove true resilience.
3. 📈 **Track Production Reliability**: Locally store all scores to visualize a single Production Reliability metric that aggregates capability plus chaos-resilience over time!

EvalMonkey natively supports evaluating ANY LLM: **AWS Bedrock**, **Azure**, **GCP**, **OpenAI**, and **Ollama**.

> **Note on API Keys:** If you have special setups that generate long-lived, static API keys for Bedrock, Azure, or GCP, simply supply them in the `.env`! EvalMonkey seamlessly supports both standard IAM / Service Account credential flows *and* long-term stateless authentication strings.

## ⚡️ Quick Start
```bash
git clone https://github.com/Corbell-AI/evalmonkey
cd evalmonkey
pip install -e .

cp .env.example .env
# Edit .env and supply your desired BYO LLM provider keys (e.g. OPENAI_API_KEY)
```

## 🌍 Supported Standard Benchmarks
EvalMonkey natively supports **10** off-the-shelf benchmark datasets pulled directly from HuggingFace. List them anytime via the CLI:
```bash
evalmonkey list-benchmarks
```
| Scenario ID | Description |
|---|---|
| `gsm8k` | Grade School Math word problems focusing on multi-step reasoning capabilities. |
| `xlam` | XLAM Function Calling 60k: Tests agent tool execution logic and parameter structuring. |
| `swe-bench` | SWE-Bench: Resolving real-world GitHub issues for coding agents. |
| `gaia-benchmark` | GAIA: General AI Assistants testing on real-world web/tool multi-step tasks. |
| `webarena` | WebArena: Highly interactive computer usage and browser manipulation. |
| `human-eval` | HumanEval: Fundamental Python code generation from docstrings. |
| `mmlu` | Massive Multitask Language Understanding: Broad generalized knowledge across 57 subjects. |
| `arc` | AI2 Reasoning Challenge: Complex grade-school science questions. |
| `truthfulqa` | TruthfulQA: Tests whether an agent mimics human falsehoods or hallucination. |
| `hella-swag` | HellaSwag: Commonsense natural language inferences. |

---

## 🛠️ Experiences 

### Experience 1: Local Sample Agents (Single Command Start)
**Easiest Experience**: Test our built-in sample agents with a single command! EvalMonkey will spawn the sample agent in the background automatically and run the benchmark.
```bash
# Run against just the first 5 records
evalmonkey run-benchmark --scenario gsm8k --sample-agent rag_app

# Run a statistically robust test against 50 different records!
evalmonkey run-benchmark --scenario gsm8k --sample-agent rag_app --limit 50
```

**Metrics Output:**
```
╭──────────────────────────────────────────────────────────╮
│ Benchmark Results                                        │
│ ──────────────────────────────────────────────────────── │
│ Scenario  gsm8k                                          │
│ Score     90/100 (Diff: +5)                              │
│ Previous  85/100                                         │
│ Reasoning Agent correctly utilized calculator for ...    │
╰──────────────────────────────────────────────────────────╯
```

### Experience 2: Benchmarking Your Custom Local Agents
Provide your own API target!
```bash
evalmonkey run-benchmark --scenario mmlu --target-url http://localhost:8000/my-custom-agent
```

### 💡 Why Chaos Benchmark Your Agents?
Resiliency and Reliability are arguably the most crucial components of any highly distributed system. Multi-agent workflows—with their isolated contexts, recursive tool calls, and cascading API dependencies—behave fundamentally identically to microservice architectures! As your agents push logic out to the real world, you **must** securely benchmark against brutal realities, dropped schemas, and malicious payload injections.

---

### Experience 3: Injecting AI-Specific Chaos Engineering (Next-Gen)
EvalMonkey goes far beyond standard network testing by deeply assessing your agent's **Production Resilience**! We support two distinct classes of Chaos injections depending on how deeply you wish to test:

#### Class A: Client-Side Injections (Zero Code Changes Required)
You don't need to change a single line of your target agent's code for these tests! EvalMonkey intercepts the benchmark dataset payload **before** transmission and maliciously damages the HTTP body so you can measure your agent's LLM fallbacks against bad actors!
| Profile | Description |
| --- | --- |
| `client_prompt_injection` | Appends adversarial "IGNORE PREVIOUS INSTRUCTIONS" jailbreaks to test system-message robustness. |
| `client_typo_injection` | Heavily obfuscates spelled words to test your LLM's semantic inference flexibility. |
| `client_schema_mutation` | Alters incoming JSON schema keys (e.g. `question` -> `query`) to verify robust API strictness handling without crashing. |
| `client_language_shift` | Radically changes request instructions to attempt safety bypasses. |

```bash
# Testing a prompt injection against your agent without modifying your code!
evalmonkey run-chaos --scenario arc --target-url http://localhost:8000/api --chaos-profile client_prompt_injection
```

#### Class B: Agent-Side Injections (Middleware Catch Required)
To deeply verify context truncation, multi-step LLM hallucination recovery, and tool back-offs, EvalMonkey attaches the `X-Chaos-Profile` header over HTTP. You write 3 lines of logic in your FastAPI/Flask proxy to trigger the exact system breakage! (See our Sample Apps for reference!)
| Profile | Description |
| --- | --- |
| `schema_error` | Simulates internal tools crashing and returning completely malformed strings mid-generation. |
| `latency_spike` | Simulates violent HTTP lag, letting you verify recursive timeouts. |
| `rate_limit_429` | Simulates your core LLM provider suddenly hitting API Request Limits mid-workflow. |
| `context_overflow` | Safely floods context sizes natively to test intelligent prompt truncation. |
| `hallucinated_tool` | Maliciously injects fake data into tool memory to test your agent's logic verification steps. |
| `empty_response` | Completely drops state parameters abruptly. |

```bash
# Testing a server-side framework context overflow!
evalmonkey run-chaos --scenario mmlu --sample-agent research_agent --chaos-profile context_overflow
```
**Metrics Output:**
```
╭──────────────────────────────────────────────────────────╮
│ 🔥 Chaos Engineering Report 🔥                             │
│ ──────────────────────────────────────────────────────── │
│ Scenario:                  xlam                          │
│ Chaos Profile:             schema_error                  │
│ Baseline Capability Score: 90                            │
│ Post-Chaos Resilience:     30                            │
│ Status:                    DEGRADED CAPABILITY           │
╰──────────────────────────────────────────────────────────╯
```

### Experience 4: Historical Production Reliability 
Check your agent's reliability trends over time!
```bash
evalmonkey history --scenario gsm8k
```
**Metrics Output:**
```
📈 Historical Trend for: gsm8k 📈
╭──────────────────┬──────────┬───────╮
│ Date             │ Run Type │ Score │
├──────────────────┼──────────┼───────┤
│ 2026-04-16 18:32 │ BASELINE │    85 │
│ 2026-04-16 18:33 │ BASELINE │    90 │
│ 2026-04-16 18:35 │ CHAOS    │    30 │
╰──────────────────┴──────────┴───────╯

🚀 Production Reliability Metric: 66.0 / 100.0
(Calculated as 60% of most recent baseline capability + 40% most recent chaos resilience)
```

## 📄 License
This project is licensed under **Apache 2.0**. See the `LICENSE` file for details.
