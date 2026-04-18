"""
EvalMonkey Adapter: Microsoft AutoGen
========================================
Wraps an AutoGen ConversableAgent pipeline inside a FastAPI
endpoint so EvalMonkey can benchmark and chaos-test it.

Install deps:
    pip install pyautogen fastapi uvicorn

Usage:
    OPENAI_API_KEY=sk-... python autogen_adapter.py
    evalmonkey run-benchmark --scenario truthfulqa --target-url http://localhost:8014/solve
"""
import os
import uvicorn
from fastapi import FastAPI, Request
import autogen

app = FastAPI(title="EvalMonkey AutoGen Adapter")

LLM_CONFIG = {
    "config_list": [{"model": os.getenv("EVAL_MODEL", "gpt-4o"), "api_key": os.getenv("OPENAI_API_KEY")}],
    "temperature": 0,
}

@app.post("/solve")
async def solve(request: Request):
    payload = await request.json()
    question = payload.get("question", payload.get("prompt", ""))

    try:
        # Build a single-shot AssistantAgent
        assistant = autogen.AssistantAgent(name="assistant", llm_config=LLM_CONFIG)
        user_proxy = autogen.UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )

        # Capture output from conversation
        user_proxy.initiate_chat(assistant, message=question, silent=True)
        last_message = assistant.last_message(user_proxy)
        answer = last_message.get("content", "") if last_message else ""
        return {"status": "success", "data": answer}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8014)
