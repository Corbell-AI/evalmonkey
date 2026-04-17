import os
import requests
import litellm

class MockLLMResponse:
    def __init__(self, content):
        class Message:
            def __init__(self, text):
                self.content = text
        class Choice:
            def __init__(self, text):
                self.message = Message(text)
        self.choices = [Choice(content)]

def call_llm(model: str, messages: list, **kwargs):
    """
    Universal wrapper around litellm that intelligently natively hooks 
    corbel's Long-Tern Bearer AWS Proxies before deferring to native routing.
    """
    bedrock_key = os.getenv("BEDROCK_API_KEY")
    
    # Natively trap long-term bedrock token scenarios explicitly bypassing boto3
    if model.startswith("bedrock/") and bedrock_key and not os.getenv("AWS_ACCESS_KEY_ID"):
        region = os.getenv("AWS_REGION_NAME", "us-east-1")
        model_id = model.replace("bedrock/", "")
        endpoint_url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke"
        
        system_prompt = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_prompt += m["content"] + "\n"
            else:
                user_msgs.append(m)
        
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "system": system_prompt.strip(),
            "messages": user_msgs,
            "temperature": kwargs.get("temperature", 0.0)
        }
        
        res = requests.post(
            endpoint_url,
            headers={"Authorization": f"Bearer {bedrock_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        if res.status_code == 200:
            data = res.json()
            text = data.get("content", [{}])[0].get("text", "")
            return MockLLMResponse(text)
        else:
            raise Exception(f"Bedrock Proxy Authentication Error: {res.status_code} {res.text}")
    
    # Let litellm handle standard routing (Ollama, OpenCV, Azure, GCP, or standard IAM AWS Bedrock)
    return litellm.completion(model=model, messages=messages, **kwargs)
