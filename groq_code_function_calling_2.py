from groq import Groq
from dotenv import load_dotenv
import os
import json

# Load environment
load_dotenv(dotenv_path=".env", override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ✅ WEATHER FUNCTION
def get_weather(city: str, unit: str = "celsius"):
    return {
        "city": city,
        "temperature": 32,
        "unit": unit,
        "condition": "Sunny"
    }


# ✅ TEMPERATURE FUNCTION ONLY
def get_temperature(city: str, unit: str = "celsius"):
    return {
        "city": city,
        "temperature": 32,
        "unit": unit
    }


# ✅ FUNCTIONS DEFINITION FOR LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the full weather report of a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get only the temperature of a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]


# ✅ ASK MODEL
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are an assistant that uses tools when required."},
        {"role": "user", "content": "What is the temperature in Chennai now?"}
    ],
    tools=tools,
    tool_choice="auto"
)


# ✅ HANDLE FUNCTION CALL
tool_calls = response.choices[0].message.tool_calls

if tool_calls:
    for call in tool_calls:
        func_name = call.function.name
        args = json.loads(call.function.arguments)

        print("Function Called:", func_name)
        print("Arguments:", args)

        # ✅ MAP FUNCTIONS
        if func_name == "get_weather":
            result = get_weather(**args)
        elif func_name == "get_temperature":
            result = get_temperature(**args)

        print("\n✅ Function Result:", result)

else:
    print("✅ Model Text Response:")
    print(response.choices[0].message.content)