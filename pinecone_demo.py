import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Your Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "genai-training")

INDEX_NAME = "genai-training-v1"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Allow specifying index dimension via environment variable; defaults to 1536
PINECONE_INDEX_DIMENSION = int(os.getenv("PINECONE_INDEX_DIMENSION", "1536"))

# Check if index exists, create if not. If it exists, print its current dimension.
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=PINECONE_INDEX_DIMENSION,       # Choose the dimension for your embeddings
        metric="cosine",      # Similarity metric: cosine, euclidean, or dotproduct
        spec=ServerlessSpec(
            cloud="aws",       # Pinecone-managed cloud, no AWS account needed
            region="us-east-1" # Region for the index
        )
    )
    print(f"Index '{INDEX_NAME}' created with dimension {PINECONE_INDEX_DIMENSION}!")
else:
    desc = pc.describe_index(name=INDEX_NAME)
    existing_dim = getattr(desc, "dimension", None)
    print(
        f"Index '{INDEX_NAME}' already exists with dimension {existing_dim}.")

# Connect to the index
index = pc.Index(INDEX_NAME)
print("Connected to index:", INDEX_NAME)
