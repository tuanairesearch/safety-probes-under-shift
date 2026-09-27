from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# transfer text to token - this is like function to transfer another text
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

def get_hidden(prompt):
    message = [
    {
        "role" : "user",
        "content" : prompt,
    }
    ] 
    inputs = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
    
    final_layer = outputs.hidden_states[-1]
    hidden_vector = final_layer[0,-1,:]
    hidden_vector = hidden_vector.float().cpu()

    return hidden_vector
hidden = get_hidden("How to make a cake?")

print(hidden)
print(hidden.shape)

INPUT_PATH = "safety_probe_prompts.csv"
OUTPUT_FOLDER = Path("data")

with open(INPUT_PATH, encoding="utf-8") as input_file:
    reader = csv.DictReader(input_file)
    rows = list(reader)

# create 3 empty list

train_data=[]
test_iid_data=[]
test_shift_data=[]

for row in rows:
    prompt = row["text"]
    label = row["label"]
    split = row["split"]

    sample = [get_hidden(prompt), label]

    if split == "train":
        train_data.append(sample)
    elif split == "test_iid":
        test_iid_data.append(sample)
    elif split == "test_shift":
        test_shift_data.append(sample)
    
print("train shape", train_data.shape)