import os
from huggingface_hub import InferenceClient # type: ignore
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# --- IMPORTANT ---
# Use a FREE model that supports Inference API
# all-MiniLM-L6-v2 does NOT work on HF Inference API (403 Forbidden)
# Use this embedding model instead:
MODEL = "sentence-transformers/all-MiniLM-L6-v2" # sentence-transformers/all-MiniLM-L6-v2 # mixedbread-ai/mxbai-embed-large-v1

client = InferenceClient(
    model=MODEL,
    token=HUGGINGFACE_API_TOKEN
)

text = "Hello, world!"

# Must send as a list: ["text"]
embedding_vector = client.feature_extraction([text])

# HF returns a list of vectors → take first one
embedding = embedding_vector[0]

print("Embedding length:", len(embedding))
print("First 5 values:", embedding)
