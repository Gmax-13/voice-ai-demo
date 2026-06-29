import os
import time
from groq import AsyncGroq
from dotenv import load_dotenv
from typing import AsyncIterator

load_dotenv()

_client = None

SYSTEM_PROMPT = """You are a concise, helpful voice assistant. 
Keep responses short and conversational — 1 to 3 sentences max.
Avoid markdown, bullet points, or lists. Speak naturally."""

MODEL = "llama-3.3-70b-versatile"


def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


async def stream_response(
    transcript: str,
    conversation_history: list[dict] | None = None,
) -> AsyncIterator[dict]:
    """
    Stream LLM response tokens.
    Yields dicts:
      {"type": "token", "text": str, "ttft_ms": float}   — on first token
      {"type": "token", "text": str}                       — subsequent tokens
      {"type": "done", "full_text": str, "total_ms": float}
    """
    client = get_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": transcript})

    t0 = time.perf_counter()
    first_token = True
    full_text = ""

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        max_tokens=200,
        temperature=0.7,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue

        full_text += delta
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        if first_token:
            first_token = False
            yield {"type": "token", "text": delta, "ttft_ms": elapsed_ms}
        else:
            yield {"type": "token", "text": delta}

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    yield {"type": "done", "full_text": full_text, "total_ms": total_ms}
