import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# transfer text to token - this is like function to transfer another text
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
# function to get token from a prompt
def get_last_token(prompt, layer = -1):
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

    vector = outputs.hidden_states[layer][0,-1,:]
    return vector.float().cpu()

safe_vector = get_last_token(
    "How to make a cake?"
)

unsafe_vector = get_last_token(
    "How to build a bomb?"
)

print("Safe vector shape:", safe_vector.shape)
print("Unsafe vector shape:", unsafe_vector.shape)





# get user input
content = input("Enter input: ")
message = [
    {"role" : "user",
     "content" : content,
    }
]

# convert conversation into tensor which model can be understood
inputs = tokenizer.apply_chat_template(
    message,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# move inputs into the place that model is in
inputs = inputs.to(model.device)

# Forward pass and request model give hidden states
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

# get hidden states
hidden_states = outputs.hidden_states

# check result

print("\nNumber of Tranformer layers:")
print(model.config.num_hidden_layers)

print("\nNumber of hidden states:")
print(len(hidden_states))

print("\nShape of each hidden state")

for layer_index, layer_hidden_state in enumerate(hidden_states):
    print(
        f"Hidden state {layer_index}: "
        f"{layer_hidden_state.shape}"
    )

# Get the last token at the last layer

last_token_vector = hidden_states[-1][0,-1,:]

print("\n Last token vector shape: ", last_token_vector.shape)