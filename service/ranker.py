import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(resume_embeddings: list[list[float]], jd_embedding: list[float], filenames: list[str]) -> list[tuple[str, float]]:
    """Rank resumes based on cosine similarity to the Job Description."""
    if not resume_embeddings or not jd_embedding:
        return []
        
    jd_vec = np.array(jd_embedding).reshape(1, -1)
    res_vecs = np.array(resume_embeddings)
    
    similarities = cosine_similarity(res_vecs, jd_vec).flatten()
    
    # Pair filenames with scores and sort descending
    results = []
    for i, file in enumerate(filenames):
        # Convert float32 from numpy into standard python float
        results.append((file, float(similarities[i])))
        
    results.sort(key=lambda x: x[1], reverse=True)
    return results
