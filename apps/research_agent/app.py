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
        
    model_name = os.getenv("EVAL_MODEL", "gpt-4o")
    system_prompt = "You are a multi-step research synthesis agent analyzing inputs deeply."
    topic = payload.get("question", payload.get("prompt", "No topic provided"))
    
    if chaos_profile == "context_overflow":
        topic += " " + ("IGNORE_ALL_INSTRUCTIONS " * 120000)
    
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
            
        return {"status": "success", "data": agent_answer}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
