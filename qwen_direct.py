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

# create answer
generated_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,
)

# get answer only - cut input, prompt
input_length = inputs["input_ids"].shape[-1]
answer_ids = generated_ids[0][input_length:]

# convert token to text
answer = tokenizer.decode(
    answer_ids,
    skip_special_tokens = True, #help remove <|im_start|>, <|im_end|>
)

print("AI: ", answer)
