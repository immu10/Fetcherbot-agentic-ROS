"""
Speech-to-text with faster-whisper.

This module is responsible for the *microphone* input path. Other audio
sources (sim audio, ROS topics, file playback, network streams) will live
in their own modules and feed audio chunks into `transcribe_audio` /
`run_on_source`.

Model: Systran/faster-whisper-base — CTranslate2 reimplementation of
OpenAI Whisper. ~4x faster than vanilla whisper, half the RAM, same
accuracy. Swap to tiny/small/medium/large-v3 as needed.

Whisper itself is not a true streaming model, so the live mic path uses
a simple energy-based VAD: record while you speak, transcribe the chunk
once you go silent for ~0.8s.

Install:  pip install faster-whisper sounddevice numpy
"""

import queue

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

MODEL_SIZE = "base"           # tiny / base / small / medium / large-v3
DEVICE = "cpu"                # "cuda", "cpu", or "auto"
COMPUTE_TYPE = "int8"         # "float16" on GPU, "int8" for fast CPU, "default" lets it pick

SAMPLE_RATE = 16000           # whisper expects 16 kHz mono
BLOCK_SECONDS = 0.03          # mic callback granularity (~30 ms)
SILENCE_RMS = 0.01            # below this is "silence" — tune to your mic/room
SILENCE_HANG = 0.8            # seconds of silence that ends an utterance
MIN_UTTERANCE = 0.4           # ignore blips shorter than this
DEBUG_LEVELS = True          # print live RMS to help tune SILENCE_RMS


def load_model(size: str = MODEL_SIZE) -> WhisperModel:
    return WhisperModel(size, device=DEVICE, compute_type=COMPUTE_TYPE)


def transcribe_audio(model: WhisperModel, audio: np.ndarray, language: str | None = None) -> str:
    """Run STT on a float32 mono 16 kHz numpy array. Returns the transcript."""
    segments, _ = model.transcribe(audio, language=language, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


def run_on_source(model: WhisperModel, source: int | None = None, language: str | None = None):
    """Generic loop over a sounddevice-compatible mic input.
    `source` is an input device index (None = system default). File / sim
    audio modules can call `transcribe_audio` directly with their own
    audio producers instead of going through this helper.
    """
    q: queue.Queue[np.ndarray] = queue.Queue()
    blocksize = int(SAMPLE_RATE * BLOCK_SECONDS)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", flush=True)
        q.put(indata[:, 0].copy())

    print("Listening… (Ctrl+C to quit)", flush=True)
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        device=source,
        callback=callback,
    ):
        buf: list[np.ndarray] = []
        speaking = False
        silence_run = 0.0
        try:
            while True:
                block = q.get()
                rms = float(np.sqrt(np.mean(block**2)))

                if DEBUG_LEVELS:
                    bar = "#" * min(40, int(rms * 400))
                    print(f"\rrms={rms:.4f} {bar:<40} speaking={speaking}", end="", flush=True)

                if rms >= SILENCE_RMS:
                    if not speaking:
                        speaking = True
                    buf.append(block)
                    silence_run = 0.0
                elif speaking:
                    buf.append(block)
                    silence_run += BLOCK_SECONDS
                    if silence_run >= SILENCE_HANG:
                        audio = np.concatenate(buf)
                        buf.clear()
                        speaking = False
                        silence_run = 0.0
                        if len(audio) / SAMPLE_RATE >= MIN_UTTERANCE:
                            text = transcribe_audio(model, audio, language=language)
                            if text:
                                print(f"> {text}", flush=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    # Mic entry point. Other input sources will have their own entry files.
    model = load_model()
    run_on_source(model, source=None)
