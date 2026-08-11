from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=0, #put at GPU0 - RTX 3060 - 12GB - Which is in my computer
)

messages = [
    {
        "role": "user",
        "content": "What is linear algebra? Answer in one sentence.",
    }
]

result = chatbot(
    messages,
    max_new_tokens=50,
    do_sample=False,
)

content = input("Enter input: ")

answer = "AI: " + result[0]["generated_text"][-1]["content"]
print("\n\n")
print(answer)