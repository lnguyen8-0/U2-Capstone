# smoke_test.py
import os
from google import genai

from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Say hello in one sentence.",
    config=types.GenerateContentConfig(
        max_output_tokens=50,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    ),
)

usage = response.usage_metadata
print((response.text or "").strip())
print(f"Finish reason: {response.candidates[0].finish_reason}")
print(f"Input tokens: {usage.prompt_token_count}")
print(f"Output tokens: {usage.candidates_token_count}")
print(f"Thinking tokens: {usage.thoughts_token_count or 0}")