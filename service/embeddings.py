import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

def get_embedding(text: str) -> list[float]:
    """Get embedding from NVIDIA NIM APIs"""
    try:
        # Limit input slightly but let NVIDIA server handle exact token truncation safely
        text_truncated = text[:15000]
        
        response = client.embeddings.create(
            input=[text_truncated],
            model="nvidia/nv-embedqa-e5-v5",
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"}
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding from NVIDIA NIM: {e}")
        return []
