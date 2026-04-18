"""
EvalMonkey Adapter: CrewAI
============================
Wraps a CrewAI Crew in a FastAPI endpoint so EvalMonkey
can fire benchmark payloads and chaos injections against it.

Install deps:
    pip install crewai crewai-tools fastapi uvicorn

Usage:
    EVAL_MODEL=gpt-4o python crewai_adapter.py
    evalmonkey run-benchmark --scenario gsm8k --target-url http://localhost:8011/solve
"""
import os
import uvicorn
from fastapi import FastAPI, Request
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

app = FastAPI(title="EvalMonkey CrewAI Adapter")

# ---------------------------------------------------
# Customize your Crew below — add specialists, tools,
# memory, etc. as needed.
# ---------------------------------------------------
def build_crew(question: str) -> Crew:
    llm = ChatOpenAI(model=os.getenv("EVAL_MODEL", "gpt-4o"), temperature=0)

    analyst = Agent(
        role="Research Analyst",
        goal="Answer questions accurately and concisely.",
        backstory="You are a world-class research expert.",
        llm=llm,
        verbose=False,
    )
    task = Task(
        description=question,
        agent=analyst,
        expected_output="A concise, accurate answer to the given question."
    )
    return Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)

@app.post("/solve")
async def solve(request: Request):
    payload = await request.json()
    question = payload.get("question", payload.get("prompt", ""))

    try:
        crew = build_crew(question)
        result = crew.kickoff()
        return {"status": "success", "data": str(result)}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8011)
