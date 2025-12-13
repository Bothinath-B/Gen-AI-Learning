from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List
from dotenv import load_dotenv
import os

load_dotenv()


# ----------------- 1. Graph State -----------------
class GraphState(TypedDict):
    messages: List


# ----------------- 2. Node: Preprocess -----------------
def preprocess_node(state: GraphState):
    """
    Adds a system instruction and cleans the user input.
    """
    user_msg = state["messages"][0].content.strip()

    system_msg = SystemMessage(
        content="You are a helpful scientific assistant. Keep explanations simple."
    )

    cleaned_user_msg = HumanMessage(content=user_msg)

    return {"messages": [system_msg, cleaned_user_msg]}


# ----------------- 3. Node: LLM -----------------
def llm_node(state: GraphState):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    msgs = state["messages"]
    ai_response = llm.invoke(msgs)

    return {"messages": msgs + [ai_response]}


# ----------------- 4. Node: Postprocess -----------------
def postprocess_node(state: GraphState):
    """
    Cleans final output. 
    Adds a summary at the end.
    """
    msgs = state["messages"]
    final_ai_response = msgs[-1].content

    final_msg = f"{final_ai_response}\n\n---\nSummary: {final_ai_response[:120]}..."

    return {"messages": msgs + [HumanMessage(content=final_msg)]}


# ----------------- 5. Build Graph -----------------
graph = StateGraph(GraphState)

graph.add_node("preprocess", preprocess_node)
graph.add_node("call_llm", llm_node)
graph.add_node("postprocess", postprocess_node)

graph.set_entry_point("preprocess")

# Node chain:
graph.add_edge("preprocess", "call_llm")
graph.add_edge("call_llm", "postprocess")
graph.add_edge("postprocess", END)

app = graph.compile()


# ----------------- 6. Run Graph -----------------
if __name__ == "__main__":
    user_input = [HumanMessage(content="Explain quasars in simple words.")]

    result = app.invoke({"messages": user_input})

    final_output = result["messages"][-1].content
    print("\nFINAL OUTPUT:\n", final_output)
