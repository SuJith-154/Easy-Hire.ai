import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

def generate_candidate_evaluation(resume_text: str, jd_text: str) -> dict:
    """Evaluate a candidate using NVIDIA NIM LLM."""
    prompt = f"""
You are an expert technical recruiter and AI system. 
Please compare the following candidate's resume to the Job Description.

Job Description:
{jd_text}

Resume:
{resume_text}

Analyze the candidate and provide exactly this JSON structure (and no other text):
{{
  "score": 85,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "decision": "Good fit",
  "explanation": "Brief 2-3 sentence explanation of the reasoning."
}}
"""
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )
        
        output = response.choices[0].message.content
        
        # Ensure we can parse JSON even if the model returns markdown block
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()
            
        return json.loads(output)
    except Exception as e:
        print(f"Error evaluating candidate: {e}")
        return {
            "score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "decision": "Error",
            "explanation": f"Failed to evaluate candidate due to model error: {str(e)}"
        }
