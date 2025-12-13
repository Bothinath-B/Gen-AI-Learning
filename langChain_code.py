from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()
#client = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=os.getenv("GROQ_API_KEY"))
resp = llm.invoke([HumanMessage(content="Explain quasars simply.")])
print(resp.content)
