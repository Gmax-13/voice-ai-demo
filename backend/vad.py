import webrtcvad
import numpy as np

# WebRTC VAD works on 16kHz, 16-bit mono PCM
# Frame durations: 10, 20, or 30ms
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples
FRAME_BYTES = FRAME_SIZE * 2  # 16-bit = 2 bytes per sample

# Aggressiveness 0-3: higher = more aggressive speech filtering
VAD_AGGRESSIVENESS = 2

# Silence threshold: how many consecutive silent frames = end of utterance
SILENCE_FRAMES_THRESHOLD = 20  # 20 * 30ms = 600ms of silence


class VADProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.buffer = b""
        self.speech_buffer = b""
        self.silent_frames = 0
        self.is_speaking = False

    def process_chunk(self, audio_bytes: bytes) -> bytes | None:
        """
        Feed raw 16kHz 16-bit mono PCM bytes.
        Returns a complete utterance (bytes) when silence is detected after speech,
        or None if still accumulating.
        """
        self.buffer += audio_bytes

        while len(self.buffer) >= FRAME_BYTES:
            frame = self.buffer[:FRAME_BYTES]
            self.buffer = self.buffer[FRAME_BYTES:]

            try:
                is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            except Exception:
                is_speech = False

            if is_speech:
                self.is_speaking = True
                self.silent_frames = 0
                self.speech_buffer += frame
            elif self.is_speaking:
                self.silent_frames += 1
                self.speech_buffer += frame  # include trailing silence for natural cut

                if self.silent_frames >= SILENCE_FRAMES_THRESHOLD:
                    utterance = self.speech_buffer
                    self.reset()
                    return utterance

        return None

    def reset(self):
        self.speech_buffer = b""
        self.silent_frames = 0
        self.is_speaking = False

    def flush(self) -> bytes | None:
        """Call on disconnect to get any remaining buffered speech."""
        if self.speech_buffer:
            utterance = self.speech_buffer
            self.reset()
            return utterance
        return None


def pcm_to_float(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
