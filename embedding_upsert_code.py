import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import uuid

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "genai-training-v1"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Init Pinecone + HF
pc = Pinecone(api_key=PINECONE_API_KEY)
client = InferenceClient(model=MODEL, token=HUGGINGFACE_API_TOKEN)

# ---- STEP 1: Get embedding and dimension ----
text = "Space is Cold!"
embedding_vector = client.feature_extraction([text])
embedding = embedding_vector[0]
embedding_dim = len(embedding)

print("Embedding length:", embedding_dim)

# ---- STEP 2: Check existing index dimension ----
existing_indexes = {i["name"]: i for i in pc.list_indexes()}

if INDEX_NAME in existing_indexes:
    desc = pc.describe_index(INDEX_NAME)
    
    if desc.dimension != embedding_dim:
        print(
            f"Dimension mismatch: index={desc.dimension}, embedding={embedding_dim}"
        )
        INDEX_NAME = f"{INDEX_NAME}-{embedding_dim}"
        
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created new index '{INDEX_NAME}'")
else:
    # Create a new index for this model dimension
    pc.create_index(
        name=INDEX_NAME,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created index '{INDEX_NAME}'")

# ---- STEP 3: Connect and upsert ----
index = pc.Index(INDEX_NAME)
print("Connected to index:", INDEX_NAME)


vector_id = str(uuid.uuid4())

index.upsert(vectors=[
    {
        "id": vector_id,
        "values": embedding,
        "metadata": {"text": text}
    }
])

print("Upsert complete.")
