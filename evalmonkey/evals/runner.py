import os
import json
from evalmonkey.utils.llm import call_llm

class LLMJudgeProvider:
    """
    LLMJudgeProvider uses litellm to abstract all common backend API LLM providers.
    Out-of-the-box it supports OpenAI, Anthropic, Bedrock, GCP Vertex, Azure, etc.
    It reads environment variables for authentication (e.g. AWS_ACCESS_KEY_ID).
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = os.getenv("EVAL_MODEL", model_name)
    
    def score_run(self, rubric: str, agent_output: str) -> dict:
        """
        Uses litellm to ask the judge to grade the agent_output against the rubric.
        Returns a dict containing 'score' (0-100) and 'reasoning'.
        """
        prompt = (
            f"You are an expert evaluator.\n\n"
            f"**Rubric/Expected Behavior**:\n{rubric}\n\n"
            f"**Agent Actual Output**:\n{agent_output}\n\n"
            f"Grade the agent's output. Provide a JSON response EXACTLY matching this format:\n"
            f"{{\"score\": <integer 0-100 indicating percentage success>, \"reasoning\": \"<detailed analysis>\"}}"
        )

        try:
            response = call_llm(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            # Fallback if there's a JSON parse error or API issue
            return {"score": 0, "reasoning": f"Evaluation failed due to error: {str(e)}"}
