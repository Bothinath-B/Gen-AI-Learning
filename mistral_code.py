## Importing necessary libraries
import os
from transformers import pipeline ## For sequential text generation
from transformers import AutoModelForCausalLM, AutoTokenizer # For leading the model and tokenizer from huggingface repository
from dotenv import load_dotenv # type: ignore
import warnings
warnings.filterwarnings("ignore") ## To remove warning messages from output

load_dotenv() ## Loading environment variables from .env file

api_key = os.getenv("HUGGINGFACE_API_TOKEN")

## Providing the huggingface model repository name for mistral 7B
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

## Downloading the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token = api_key)
tokenizer = AutoTokenizer.from_pretrained(model_name, token = api_key)

## Creating a text generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.3,top_p=0.85,
top_k = 10,
max_new_tokens = 200
)


## Setting up the system prompt and asking the first question (user prompt)
messages = [
    {"role": "system", "content": "You are a helpful medical assistant chatbot. You provide accurate and informative responses"},
    {"role": "user", "content": "what are the symptoms of the flu?"}]


## Generating a response from the model
response = chatbot(messages)


# print("User:", messages[-1]["content"])
# print("Formatted prompt:\n", formatted_prompt)
print("Response:", response[0]['generated_text'][-1]["content"])