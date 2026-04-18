"""
EvalMonkey Adapter: LangChain Agent
====================================
Wraps any LangChain agent/chain in a FastAPI endpoint so EvalMonkey
can fire benchmark payloads and chaos injections against it.

Install deps:
    pip install langchain langchain-openai fastapi uvicorn

Usage:
    EVAL_MODEL=gpt-4o python langchain_adapter.py
    evalmonkey run-benchmark --scenario mmlu --target-url http://localhost:8010/solve
"""
import os
import uvicorn
from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

app = FastAPI(title="EvalMonkey LangChain Adapter")

# ---------------------------------------------------
# Build your LangChain agent/chain here — swap this
# for any Chain, AgentExecutor, LCEL pipe, etc.
# ---------------------------------------------------
llm = ChatOpenAI(model=os.getenv("EVAL_MODEL", "gpt-4o"), temperature=0)

@app.post("/solve")
async def solve(request: Request):
    payload = await request.json()
    question = payload.get("question", payload.get("prompt", ""))

    try:
        response = llm.invoke([
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=question)
        ])
        return {"status": "success", "data": response.content}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
