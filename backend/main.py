import os
import json
import time
import base64
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from vad import VADProcessor
from stt import transcribe_pcm
from llm import stream_response
from tts import stream_tts, split_into_sentences

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voice-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Voice AI Demo starting up")
    yield
    log.info("Voice AI Demo shutting down")


app = FastAPI(title="Voice AI Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected")

    vad = VADProcessor()
    conversation_history: list[dict] = []

    async def send_json(data: dict):
        await ws.send_text(json.dumps(data))

    async def send_bytes(data: bytes):
        await ws.send_bytes(data)

    try:
        while True:
            message = await ws.receive()

            # Binary = raw PCM audio from browser
            if "bytes" in message and message["bytes"]:
                pcm_chunk = message["bytes"]

                utterance = vad.process_chunk(pcm_chunk)
                if utterance is None:
                    continue

                # --- VAD detected end of speech ---
                t_vad_done = time.perf_counter()
                await send_json({"type": "vad_end", "utterance_bytes": len(utterance)})

                # --- STT ---
                t_stt_start = time.perf_counter()
                stt_result = await transcribe_pcm(utterance)
                t_stt_done = time.perf_counter()

                transcript = stt_result["transcript"].strip()
                stt_latency = round((t_stt_done - t_stt_start) * 1000, 1)

                log.info(f"STT [{stt_latency}ms]: {transcript!r}")

                if not transcript:
                    await send_json({"type": "stt_empty"})
                    continue

                await send_json({
                    "type": "transcript",
                    "text": transcript,
                    "latency_ms": stt_latency,
                })

                # --- LLM streaming ---
                t_llm_start = time.perf_counter()
                full_response = ""
                ttft_ms = None
                sentence_buffer = ""
                tts_tasks = []
                first_sentence_sent = False

                async def run_tts_for_sentence(sentence: str, sent_index: int):
                    """Stream TTS for one sentence and forward audio to client."""
                    t_tts = time.perf_counter()
                    chunk_count = 0
                    async for event in stream_tts(sentence):
                        if event["type"] == "audio_chunk":
                            chunk_count += 1
                            # Send audio as base64 in JSON so frontend can queue it
                            await send_json({
                                "type": "audio_chunk",
                                "data": base64.b64encode(event["data"]).decode(),
                                "sentence_index": sent_index,
                                "chunk_index": event["chunk_index"],
                            })
                        elif event["type"] == "done":
                            tts_latency = round((time.perf_counter() - t_tts) * 1000, 1)
                            log.info(
                                f"TTS sentence {sent_index} [{tts_latency}ms, "
                                f"{chunk_count} chunks]: {sentence!r}"
                            )
                    await send_json({"type": "sentence_done", "sentence_index": sent_index})

                sentence_index = 0

                async for token_event in stream_response(transcript, conversation_history):
                    if token_event["type"] == "token":
                        token = token_event["text"]
                        full_response += token
                        sentence_buffer += token

                        if "ttft_ms" in token_event:
                            ttft_ms = token_event["ttft_ms"]
                            await send_json({
                                "type": "llm_start",
                                "ttft_ms": ttft_ms,
                            })

                        await send_json({"type": "token", "text": token})

                        # Fire TTS on sentence boundaries for low latency
                        has_sentence_end = any(c in sentence_buffer for c in ".!?\n")
                        if has_sentence_end:
                            sentences = split_into_sentences(sentence_buffer)
                            # Keep last partial sentence in buffer
                            if len(sentences) > 1 or (sentences and sentence_buffer.rstrip()[-1] in ".!?\n"):
                                complete = sentences[:-1] if sentence_buffer.rstrip()[-1] not in ".!?\n" else sentences
                                remaining = sentences[-1] if sentence_buffer.rstrip()[-1] not in ".!?\n" else ""
                                for s in complete:
                                    if s.strip():
                                        task = asyncio.create_task(
                                            run_tts_for_sentence(s, sentence_index)
                                        )
                                        tts_tasks.append(task)
                                        sentence_index += 1
                                sentence_buffer = remaining

                    elif token_event["type"] == "done":
                        llm_total_ms = token_event["total_ms"]

                        # Flush any remaining text
                        if sentence_buffer.strip():
                            task = asyncio.create_task(
                                run_tts_for_sentence(sentence_buffer.strip(), sentence_index)
                            )
                            tts_tasks.append(task)
                            sentence_index += 1

                        await send_json({
                            "type": "llm_done",
                            "full_text": full_response,
                            "total_ms": llm_total_ms,
                            "ttft_ms": ttft_ms,
                        })

                # Wait for all TTS tasks
                if tts_tasks:
                    await asyncio.gather(*tts_tasks)

                t_total = time.perf_counter()
                total_ms = round((t_total - t_vad_done) * 1000, 1)

                log.info(
                    f"[LATENCY] STT: {stt_latency}ms | "
                    f"LLM TTFT: {ttft_ms}ms | "
                    f"Total: {total_ms}ms"
                )

                await send_json({
                    "type": "turn_done",
                    "latency": {
                        "stt_ms": stt_latency,
                        "llm_ttft_ms": ttft_ms,
                        "total_ms": total_ms,
                    },
                })

                # Update conversation history
                conversation_history.append({"role": "user", "content": transcript})
                conversation_history.append({"role": "assistant", "content": full_response})

                # Keep last 10 turns to avoid context overflow
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]

            # Text = control messages from frontend
            elif "text" in message and message["text"]:
                try:
                    ctrl = json.loads(message["text"])
                    if ctrl.get("type") == "reset":
                        conversation_history.clear()
                        vad.reset()
                        await send_json({"type": "reset_ack"})
                    elif ctrl.get("type") == "flush":
                        utterance = vad.flush()
                        if utterance:
                            log.info("VAD flushed on client request")
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
