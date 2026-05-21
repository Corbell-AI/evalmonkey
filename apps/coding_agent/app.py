"""
Lightweight Coding Agent — EvalMonkey sample app.

Mirrors the structure of apps/rag_app/app.py but targets code generation tasks.
Handles HumanEval / MBPP / APPS style requests: given a task description (with
optional starter code / test cases in the prompt), produces a Python implementation.

Run with:
    python apps/coding_agent/app.py
"""
import os
import asyncio
from evalmonkey.utils.llm import call_llm
from fastapi import FastAPI, Request

app = FastAPI(title="Coding Agent API")

SYSTEM_PROMPT = """\
You are an expert Python programmer. When given a coding task or a function \
stub to complete, you MUST:
1. Return ONLY valid, runnable Python code — no prose, no markdown fences.
2. Define the exact function name requested (or a sensible one if none is given).
3. Handle edge cases (empty input, None, type errors) gracefully.
4. Keep the implementation concise but correct.

If the request is a multi-step algorithm, break it into clear helper functions \
inside the same code block. Do NOT add any explanation outside of inline comments.\
"""


@app.post("/solve")
async def solve(request: Request):
    payload = await request.json()

    chaos_profile = request.headers.get("X-Chaos-Profile")

    # ── Server-side chaos profiles ───────────────────────────────────────────
    if chaos_profile == "latency_spike":
        await asyncio.sleep(5)
    elif chaos_profile == "timeout_no_response":
        await asyncio.sleep(120)
    elif chaos_profile == "model_downgrade":
        import os as _os
        _os.environ["EVAL_MODEL"] = "gpt-3.5-turbo"
    elif chaos_profile == "rate_limit_429":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={"error": "Rate Limit Exceeded", "retry_after": 60}
        )

    model_name = os.getenv("EVAL_MODEL", "gpt-4o")
    question = payload.get("question", "")

    # ── Server-side coding-specific chaos profiles ───────────────────────────
    if chaos_profile == "corrupt_output":
        # Returns syntactically broken Python — mimics a truncated stream response.
        try:
            response = call_llm(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            good_code = response.choices[0].message.content
            # Slice mid-line to simulate a dropped connection
            return {"status": "success", "data": good_code[:len(good_code) // 2]}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    elif chaos_profile == "wrong_language_response":
        # Pretend the agent ignores the Python requirement and returns JavaScript.
        return {
            "status": "success",
            "data": (
                "// JavaScript (Node.js) response — ignoring Python requirement\n"
                "function solve(arr) {\n  return arr.reduce((a, b) => a + b, 0);\n}"
            ),
        }

    elif chaos_profile == "empty_response":
        return {"status": "success", "data": ""}

    elif chaos_profile == "hallucinated_api":
        # Returns code that calls a completely made-up stdlib function.
        return {
            "status": "success",
            "data": (
                "import python_magic_solver\n\n"
                "def solve(nums):\n"
                "    return python_magic_solver.auto_solve(nums)"
            ),
        }

    # ── Normal code generation path ──────────────────────────────────────────
    try:
        response = call_llm(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        code_output = response.choices[0].message.content

        # Strip any accidental markdown fences the model might add
        if "```" in code_output:
            lines = code_output.splitlines()
            code_lines = []
            inside_fence = False
            for line in lines:
                if line.strip().startswith("```"):
                    inside_fence = not inside_fence
                    continue
                if inside_fence or not any(
                    line.strip().startswith("```") for _ in [None]
                ):
                    code_lines.append(line)
            code_output = "\n".join(code_lines).strip()

        return {"status": "success", "data": code_output}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
