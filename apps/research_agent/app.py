import os
import asyncio
from evalmonkey.utils.llm import call_llm
from fastapi import FastAPI, Request

app = FastAPI(title="Litellm Research Agent API")

@app.post("/solve")
async def solve(request: Request):
    payload = await request.json()
    
    chaos_profile = request.headers.get("X-Chaos-Profile")
    if chaos_profile == "latency_spike":
        await asyncio.sleep(5)
    elif chaos_profile == "rate_limit_429":
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=429, content={"error_message": "LLM Provider HTTP 429 Too Many Requests"})
    elif chaos_profile == "timeout_no_response":
        # Simulates a completely hung agent — tests client read-timeout hygiene.
        await asyncio.sleep(120)
    elif chaos_profile == "model_downgrade":
        # Silently replaces the configured model with the cheapest fallback to
        # simulate a quota breach causing automatic provider downgrade.
        import os as _os
        _os.environ["EVAL_MODEL"] = "gpt-3.5-turbo"
        
    model_name = os.getenv("EVAL_MODEL", "gpt-4o")
    system_prompt = "You are a multi-step research synthesis agent analyzing inputs deeply."
    topic = payload.get("question", payload.get("prompt", "No topic provided"))
    
    if chaos_profile == "context_overflow":
        topic += " " + ("IGNORE_ALL_INSTRUCTIONS " * 120000)
    elif chaos_profile == "memory_amnesia":
        # Clears the incoming message to simulate flushed session state —
        # a real failure mode when Redis/session stores are unreachable.
        topic = "[MEMORY CLEARED] I have no context about our previous conversation."
    
    try:
        response = call_llm(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": topic}
            ]
        )
        agent_answer = response.choices[0].message.content
        
        if chaos_profile == "schema_error":
            return {"broken_response_format": 0}
        elif chaos_profile == "hallucinated_tool":
            return {"status": "success", "data": f"Here is a completely hallucinated response bypassing logic: {agent_answer}"}
        elif chaos_profile == "empty_response":
            return {"status": "success", "data": ""}
        elif chaos_profile == "partial_response_truncation":
            # Drops the response mid-stream — simulates proxy timeout cutting off
            # long LLM outputs before they reach the client.
            return {"status": "success", "data": agent_answer[:20]}
        elif chaos_profile == "cascading_tool_failure":
            # Simulates a downstream tool crashing after LLM already invoked it —
            # tests whether tool-calling agents gracefully handle mid-chain errors.
            return {"status": "tool_error", "error_message": "SearchAPI connection timeout", "data": None, "tool": "web_search"}
            
        return {"status": "success", "data": agent_answer}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
