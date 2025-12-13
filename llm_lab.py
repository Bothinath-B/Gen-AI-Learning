from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer & model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input text
prompt = "Artificial intelligence will"

# Convert to tokens
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7
)

# Decode and print
print(tokenizer.decode(output[0], skip_special_tokens=True))
