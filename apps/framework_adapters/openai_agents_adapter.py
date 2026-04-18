"""
EvalMonkey Adapter: OpenAI Agents SDK
========================================
Wraps an OpenAI Responses API agent (or Assistants API) in a FastAPI
endpoint compatible with EvalMonkey's universal HTTP contract.

Install deps:
    pip install openai fastapi uvicorn

Usage:
    OPENAI_API_KEY=sk-... python openai_agents_adapter.py
    evalmonkey run-benchmark --scenario arc --target-url http://localhost:8012/solve
"""
import os
import uvicorn
from fastapi import FastAPI, Request
from openai import OpenAI

app = FastAPI(title="EvalMonkey OpenAI Agents Adapter")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/solve")
async def solve(request: Request):
    payload = await request.json()
    question = payload.get("question", payload.get("prompt", ""))

    try:
        response = client.chat.completions.create(
            model=os.getenv("EVAL_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are a knowledgeable AI assistant."},
                {"role": "user", "content": question},
            ]
        )
        return {"status": "success", "data": response.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8012)
