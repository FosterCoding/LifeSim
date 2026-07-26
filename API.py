import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env into the environment

client = OpenAI(
    api_key=os.environ["AI_API_KEY"],
    base_url="https://api.deepseek.com"  # the one line that changes per provider
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Say hello in five words or fewer."}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)