import torch
from gpt2 import GPT
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# load custom gpt2 model
gpt = GPT.from_pretrained("gpt2")
# load gpt2 model from HF
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# test
text = "Hi, I am a language model,"
encoded_input = tokenizer(text, return_tensors="pt")

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

hf_out = model(encoded_input["input_ids"]).logits
custom_out = gpt(encoded_input["input_ids"])
print("HF GPT2 output:\n", hf_out)
print("Custom GPT2 output:\n", custom_out)
print("Sanity check:", "Passed" if torch.allclose(hf_out, custom_out) else "Failed")
