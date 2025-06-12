import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS = [
    "Gemini is a large language model developed by Google.",
    "GroqCloud provides fast inference for open-source models.",
    "FAISS is a library for efficient similarity search.",
    "Flask is a lightweight Python web framework.",
    "Loguru is a simple logging library for structured logs.",
    "You can integrate Gemini and Groq for multi-LLM apps."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model.encode(CORPUS).astype('float32')

dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

def semantic_search(query: str, k: int = 1) -> str:
    query_vector = model.encode([query]).astype('float32')
    D, I = index.search(query_vector, k)
    return CORPUS[I[0][0]]
