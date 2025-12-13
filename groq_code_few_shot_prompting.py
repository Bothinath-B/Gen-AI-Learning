from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Create completion request
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        
        {"role": "user", "content": " This is awesome" },
        {"role": "user", "content": " I hate this weather" },
        {"role": "user", "content": " The movie was fantastic" },
        {"role": "user", "content": " I am so sad today" },
        {"role": "user", "content": " The food was terrible" },
        {"role": "user", "content": " The service at the restaurant was awful" },
        {"role": "user", "content": " Had a great time at the concert" },
        {"role": "user", "content": " Feeling very disappointed with the results" },
        {"role": "system", "content": "You are a sentiment analysis assistant. Provide final one sentiment answer based on all the above sentences."},
    ],
    temperature=1,
    max_completion_tokens=8192,
    top_p=1,
    reasoning_effort="medium",
    stream=True,
    stop=None
)

# Stream the response
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
