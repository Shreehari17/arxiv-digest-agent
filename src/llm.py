import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_MODEL = "openai/gpt-oss-120b"
_TIMEOUT = 60
_MAX_RETRIES = 1


def call_llm(prompt: str, system: str | None = None) -> str:
    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system,
        })

    messages.append({
        "role": "user",
        "content": prompt,
    })

    last_error = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                temperature=0,
                timeout=_TIMEOUT,
            )

            return response.choices[0].message.content.strip()

        except Exception as exc:
            last_error = exc

            if attempt < _MAX_RETRIES:
                time.sleep(1)

    raise RuntimeError(f"LLM request failed: {last_error}")