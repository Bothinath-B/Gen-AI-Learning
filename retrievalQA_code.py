from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os, numpy as np

load_dotenv()

# Embedding model
embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

docs = [
    Document(page_content="Quasars are powered by black holes."),
    Document(page_content="A quasar shines when gas heats up and emits light."),
    Document(page_content="In binary code, 1 represents 'on' and 0 represents 'off'."),
    Document(page_content="The speed of light is approximately 299,792 kilometers per second."),

]

class SimpleRetriever:
    def __init__(self, docs, embed_model):
        self.docs = docs
        self.embed_model = embed_model
        self.texts = [d.page_content for d in docs]
        self.vectors = np.array(embed_model.embed_documents(self.texts))

    def get_relevant_documents(self, query, k=3):
        qv = np.array(self.embed_model.embed_query(query))
        sims = (self.vectors @ qv) / (
            np.linalg.norm(self.vectors, axis=1) * (np.linalg.norm(qv) + 1e-12)
        )
        print("Similarities:", sims)
        idxs = sims.argsort()[::-1][:k]
        return [self.docs[i] for i in idxs]

retriever = SimpleRetriever(docs, embed_model)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate(
    template="""Use the context to answer or do your own reasoning.
Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"],
)

query = "What is 1/infinity?"
docs_for_context = retriever.get_relevant_documents(query)
context_text = "\n\n".join(d.page_content for d in docs_for_context)

formatted_prompt = prompt.format(
    context=context_text,
    question=query
)

response = llm.invoke([
    HumanMessage(content=formatted_prompt)
])
print("\nLLM Response:\n", response.content)
