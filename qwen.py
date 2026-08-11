from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=0, #put at GPU0 - RTX 3060 - 12GB - Which is in my computer
)
content = input("Enter input: ")

messages = [
    {
        "role": "user",
        "content": content,
    }
]

result = chatbot(
    messages,
    max_new_tokens=50,
    do_sample=False,
)


answer = "AI: " + result[0]["generated_text"][-1]["content"]
content = "User: " + content
print("\n\n")
print(content)
print(answer)