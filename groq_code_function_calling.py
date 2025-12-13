import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------
# 1. Python function (Tool)
# ---------------------------------------------
def get_patient_symptoms(disease: str):
    data = {
        "diabetes": ["Frequent urination", "Excessive thirst", "Fatigue", "Slow wound healing"],
        "flu": ["Fever", "Cough", "Body aches"],
        "covid": ["Fever", "Cough", "Loss of smell", "Breathing difficulty"],
    }
    return data.get(disease.lower(), ["No data found"])


# ---------------------------------------------
# 2. LLM Request
# ---------------------------------------------
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",

    messages=[
        {"role": "user", "content": "What are the symptoms of denuge?"}
    ],

    tools=[{
        "type": "function",
        "function": {
            "name": "get_patient_symptoms",
            "description": "Returns symptoms for a disease name",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease": {"type": "string"}
                },
                "required": ["disease"]
            }
        }
    }],

    tool_choice="auto"
)

message = completion.choices[0].message


# ---------------------------------------------
# 3. Check if LLM called a tool (Correct Groq syntax)
# ---------------------------------------------
if not message.tool_calls:
    print("LLM response:", message.content)
    exit()

# There could be multiple tools
tool_call = message.tool_calls[0]

tool_name = tool_call.function.name
tool_args = json.loads(tool_call.function.arguments)

print("🔧 Tool requested:", tool_name)
print("📥 Args:", tool_args)


# ---------------------------------------------
# 4. Run the local Python function
# ---------------------------------------------
if tool_name == "get_patient_symptoms":
    function_result = get_patient_symptoms(**tool_args)
else:
    function_result = {"error": "unknown tool"}

print("🧪 Function Output:", function_result)


# ---------------------------------------------
# 5. Send tool result back to LLM
# ---------------------------------------------
followup = client.chat.completions.create(
    model="openai/gpt-oss-120b",

    messages=[
        {"role": "user", "content": "What are the symptoms of diabetes?"},
        message,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(function_result)
        }
    ]
)

print("\n💬 Final Answer:\n")
print(followup.choices[0].message.content)
