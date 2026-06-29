import os
import time
import asyncio
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client() -> DeepgramClient:
    global _client
    if _client is None:
        _client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
    return _client


async def transcribe_pcm(pcm_bytes: bytes, sample_rate: int = 16000) -> dict:
    """
    Transcribe raw 16-bit mono PCM bytes via Deepgram prerecorded API.
    Returns dict with keys: transcript (str), latency_ms (float), words (list)
    """
    t0 = time.perf_counter()

    client = get_client()

    options = PrerecordedOptions(
        model="nova-2-general",
        language="en",
        smart_format=True,
        punctuate=True,
        utterances=False,
    )

    # Deepgram expects audio bytes + mimetype
    # For raw PCM we wrap it as wav-like container
    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate)

    payload = {"buffer": wav_bytes, "mimetype": "audio/wav"}

    response = await asyncio.to_thread(
        client.listen.prerecorded.v("1").transcribe_file,
        payload,
        options,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    transcript = (
        response.results.channels[0].alternatives[0].transcript
        if response.results and response.results.channels
        else ""
    )

    words = []
    if response.results and response.results.channels:
        alt = response.results.channels[0].alternatives[0]
        words = [
            {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
            for w in (alt.words or [])
        ]

    return {
        "transcript": transcript,
        "latency_ms": round(latency_ms, 1),
        "words": words,
    }


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM in a minimal WAV header."""
    import struct

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_bytes
