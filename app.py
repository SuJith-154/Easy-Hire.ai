from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from service.resume_parser import extract_text
from service.embeddings import get_embedding
from service.ranker import rank_resumes
from service.llm import generate_candidate_evaluation

app = FastAPI(title="AI Resume Ranker API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads/resumes/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- REQUEST MODELS ---
class TextRequest(BaseModel):
    text: str

class EvalRequest(BaseModel):
    resume_text: str
    jd_text: str


# --- INDIVIDUAL ISOLATED APIs ---

@app.post("/api/extract-text")
async def api_extract_text(file: UploadFile = File(...)):
    """1. Extract and clean text from a single PDF/DOCX file."""
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    text = extract_text(file_path, file.filename)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text.")
    return {"filename": file.filename, "extracted_text": text}

@app.post("/api/generate-embedding")
async def api_generate_embedding(req: TextRequest):
    """2. Generate a vector embedding for the provided text using NVIDIA API."""
    embedding = get_embedding(req.text)
    if not embedding:
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")
    return {"embedding_length": len(embedding), "embedding": embedding[:10]} # Only returning top 10 elements to keep response clean, but length is accurate.

@app.post("/api/evaluate-candidate")
async def api_evaluate_candidate(req: EvalRequest):
    """3. Use NVIDIA NIM LLM to evaluate a candidate text against a JD."""
    evaluation = generate_candidate_evaluation(req.resume_text, req.jd_text)
    return {"evaluation": evaluation}


# --- MAIN PIPELINE ORCHESTRATION API ---

@app.post("/api/rank-resumes")
async def rank_resumes_endpoint(files: list[UploadFile] = File(...), jd: UploadFile = File(...)):
    """Full pipeline: Takes multiple resumes and a JD, and returns the top ranked candidates."""
    resume_texts = []
    resume_files = []

    # Read Job Description
    try:
        jd_path = os.path.join(UPLOAD_FOLDER, jd.filename)
        with open(jd_path, "wb") as f:
            f.write(await jd.read())
        jd_text = extract_text(jd_path, jd.filename)
        if not jd_text.strip():
            raise ValueError("Parsed JD text is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JD file. Could not extract text from the uploaded JD.")

    # Save and process resumes
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

        text = extract_text(file_path, file.filename)
        if text.strip():
            resume_texts.append(text)
            resume_files.append(file.filename)
        else:
            print(f"Failed to extract text from {file.filename}")

    if not resume_texts:
        raise HTTPException(status_code=400, detail="No readable text found in the uploaded resumes.")

    # Get Embeddings
    print("Generating embeddings...")
    resume_embeddings = [get_embedding(text) for text in resume_texts]
    jd_embedding = get_embedding(jd_text)
    
    if not jd_embedding or not any(resume_embeddings):
        raise HTTPException(status_code=500, detail="Failed to general vectors from NVIDIA APIs.")

    # Ranking
    print("Ranking candidates...")
    top_candidates = rank_resumes(resume_embeddings, jd_embedding, resume_files)
    
    # Limit to Top 5
    top_5 = top_candidates[:5]

    results = []
    
    print("Evaluating top candidates with LLM...")
    for filename, sim_score in top_5:
        idx = resume_files.index(filename)
        resume_text = resume_texts[idx]
        
        # Get reasoning from LLM
        evaluation = generate_candidate_evaluation(resume_text, jd_text)
        
        # Combine the data
        evaluation["name"] = filename
        evaluation["similarity_score"] = float(sim_score)
        
        results.append(evaluation)

    return {"top_candidates": results}