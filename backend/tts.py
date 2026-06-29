import os
import time
import httpx
from dotenv import load_dotenv
from typing import AsyncIterator

load_dotenv()

VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
MODEL_ID = "eleven_turbo_v2"  # lowest latency model
CHUNK_SIZE = 4096  # bytes per audio chunk streamed to client

# Sentence boundary characters — flush TTS on these for lower perceived latency
SENTENCE_ENDS = {".", "!", "?", "\n"}


async def stream_tts(text: str) -> AsyncIterator[dict]:
    """
    Stream TTS audio chunks for the given text.
    Yields dicts:
      {"type": "audio_chunk", "data": bytes, "chunk_index": int}
      {"type": "done", "total_ms": float, "chunks": int}
    """
    api_key = os.environ["ELEVENLABS_API_KEY"]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
        "output_format": "mp3_44100_128",
    }

    t0 = time.perf_counter()
    chunk_index = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()

            async for chunk in resp.aiter_bytes(chunk_size=CHUNK_SIZE):
                if chunk:
                    chunk_index += 1
                    yield {
                        "type": "audio_chunk",
                        "data": chunk,
                        "chunk_index": chunk_index,
                    }

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    yield {"type": "done", "total_ms": total_ms, "chunks": chunk_index}


def split_into_sentences(text: str) -> list[str]:
    """
    Split text on sentence boundaries for chunked TTS playback.
    First sentence is sent immediately for low latency.
    """
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in SENTENCE_ENDS and current.strip():
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences
