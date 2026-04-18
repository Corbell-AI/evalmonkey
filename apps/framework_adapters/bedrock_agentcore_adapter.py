"""
EvalMonkey Adapter: AWS Bedrock Agent Core / Converse API
===========================================================
Wraps AWS Bedrock's Converse API (or a Bedrock AgentCore runtime)
inside a FastAPI endpoint so EvalMonkey can benchmark it.

Supports both:
  - Standard IAM credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
  - Long-lived static Bearer tokens (BEDROCK_API_KEY)

Install deps:
    pip install boto3 fastapi uvicorn

Usage:
    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION_NAME=us-east-1 python bedrock_agentcore_adapter.py
    evalmonkey run-benchmark --scenario mmlu --target-url http://localhost:8013/solve
"""
import os
import uvicorn
import boto3
from fastapi import FastAPI, Request

app = FastAPI(title="EvalMonkey Bedrock Agent Core Adapter")

# ---------------------------------------------------
# Change model_id to any Bedrock Foundation Model you
# have access to (Claude, Titan, Llama, Mistral…)
# ---------------------------------------------------
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

def get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION_NAME", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),  # optional
    )

@app.post("/solve")
async def solve(request: Request):
    payload = await request.json()
    question = payload.get("question", payload.get("prompt", ""))

    try:
        bedrock = get_bedrock_client()
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": question}]}],
            system=[{"text": "You are a knowledgeable AI assistant."}],
        )
        answer = response["output"]["message"]["content"][0]["text"]
        return {"status": "success", "data": answer}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8013)
