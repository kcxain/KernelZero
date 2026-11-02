from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_API_URL", "http://127.0.0.1:8000/v1"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "who are you?",
        }
    ],
    model="oss",
    max_tokens=10240,
)

print(chat_completion.choices[0].message.content)
# 获取 token 使用信息
print(f"Prompt tokens: {chat_completion.usage.prompt_tokens}")
print(f"Completion tokens: {chat_completion.usage.completion_tokens}")
print(f"Total tokens: {chat_completion.usage.total_tokens}")
