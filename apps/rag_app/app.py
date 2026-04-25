import os
import asyncio
from evalmonkey.utils.llm import call_llm
from fastapi import FastAPI, Request

app = FastAPI(title="Litellm RAG Agent API")

@app.post("/solve")
async def solve(request: Request):
    payload = await request.json()
    
    # Advanced Chaos Engineering Injection Logic
    chaos_profile = request.headers.get("X-Chaos-Profile")
    if chaos_profile == "latency_spike":
        await asyncio.sleep(5)
    elif chaos_profile == "rate_limit_429":
        # Simulate third-party LLM quota exhaustion
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=429, content={"error": "Rate Limit Exceeded. Quota exhausted.", "retry_after": 60})
    elif chaos_profile == "timeout_no_response":
        # Simulates the agent hanging indefinitely — the client will hit its own timeout.
        # Validates whether callers implement read-timeouts and don't block forever.
        await asyncio.sleep(120)
    elif chaos_profile == "model_downgrade":
        # Silently swaps the real model for the cheapest/dumbest available model.
        # Mimics the real scenario where a quota breach causes automatic provider fallback.
        import os as _os
        _os.environ["EVAL_MODEL"] = "gpt-3.5-turbo"
        
    model_name = os.getenv("EVAL_MODEL", "gpt-4o")
    system_prompt = "You are a RAG assistant. Context: [Apples cost $3. Bananas cost $1. Math reasoning works as usual]. Answer the user's question."
    question = payload.get("question", "")

    if chaos_profile == "context_overflow":
        # Simulating external users stuffing 120,000 tokens maliciously 
        question += " " + ("IGNORE_ALL_INSTRUCTIONS " * 120000)
    elif chaos_profile == "memory_amnesia":
        # Wipes the incoming question and replaces it with a blank slate.
        # Simulates a stateful agent whose conversation memory was flushed mid-session.
        question = "[MEMORY CLEARED] I have no context about our previous conversation."

    try:
        # We use purely Litellm here so users can use AWS Bedrock, Ollama, Azure, etc natively without custom SDK SDK barriers
        # the model name controls exactly which provider is routed!
        response = call_llm(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        agent_answer = response.choices[0].message.content
        
        # Schema Error Chaos Profile
        if chaos_profile == "schema_error":
            return {"malformed_response": "corrupted", "data": None}
        elif chaos_profile == "hallucinated_tool":
            return {"status": "success", "data": f"Warning: I hallucinated this completely fake fact earlier: {agent_answer}"}
        elif chaos_profile == "empty_response":
            return {"status": "success", "data": ""}
        elif chaos_profile == "partial_response_truncation":
            # Returns only the first 20 characters of the answer, simulating a
            # streaming connection that dropped mid-transmission — very common with
            # long-running LLM calls behind ALBs / nginx with short proxy timeouts.
            return {"status": "success", "data": agent_answer[:20]}
        elif chaos_profile == "cascading_tool_failure":
            # Simulates a downstream tool (e.g. vector DB, search API) returning
            # a hard error after the LLM has already called it. The agent must
            # gracefully degrade without crashing the entire request chain.
            return {"status": "tool_error", "error_message": "VectorDB connection refused", "data": None, "tool": "retriever"}  
            
        return {"status": "success", "data": agent_answer}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
