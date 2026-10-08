from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()
# Create the client
client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
# Your prompt
prompt = "Hello"
# Call Gemini through the OpenAI-compatible API
response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant. "
                    "Answer questions accurately using the provided context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )
# Print the response
print(response.choices[0].message.content)