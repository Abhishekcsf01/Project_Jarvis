
import io
import os
import time
import wave
import threading
from collections import deque

import numpy as np
from flask import Flask, jsonify, request
from google import genai
from google.genai import types

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "").strip()

AUDIO_MODEL = os.environ.get("AUDIO_MODEL", "gemini-3.8-flash").strip()
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-3.8-flash").strip()

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.5
SILENCE_TO_END_SECONDS = 1.25
MIN_SPEECH_SECONDS = 0.45
MAX_UTTERANCE_SECONDS = 18
RMS_THRESHOLD = 450

# Timeout is in milliseconds.
client = (
    genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=45000),
    )
    if GEMINI_API_KEY else None
)

states = {}
states_lock = threading.Lock()


def new_state():
    return {
        "pre_roll": deque(maxlen=4),
        "utterance": [],
        "speaking": False,
        "silence_chunks": 0,
        "last_transcript": "",
        "last_reply": "",
        "status": "listening",
        "updated_at": time.time(),
        "job_id": 0,
    }


def get_state(device_id):
    with states_lock:
        if device_id not in states:
            states[device_id] = new_state()
        return states[device_id]


def authorized():
    token = request.headers.get("X-Device-Token", "")
    return bool(DEVICE_TOKEN) and token == DEVICE_TOKEN


def pcm_to_wav_bytes(pcm_bytes):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm_bytes)
    return output.getvalue()


def gemini_generate(model, contents, label, max_attempts=3):
    """Retry temporary Gemini failures with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            app.logger.info(
                "%s: request attempt %d/%d model=%s",
                label, attempt + 1, max_attempts, model
            )
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0 if label == "transcription" else 0.4,
                    max_output_tokens=300 if label == "transcription" else 160,
                ),
            )
        except Exception as exc:
            code = getattr(exc, "code", None)
            status = getattr(exc, "status", None)
            message = str(exc).lower()

            temporary = (
                code in (429, 500, 502, 503, 504)
                or str(status).upper() in (
                    "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                    "INTERNAL", "DEADLINE_EXCEEDED"
                )
                or any(term in message for term in (
                    "503 unavailable",
                    "429 resource_exhausted",
                    "high demand",
                    "temporarily unavailable",
                    "deadline exceeded",
                    "timed out",
                ))
            )

            if not temporary or attempt == max_attempts - 1:
                app.logger.exception(
                    "%s failed permanently; code=%s status=%s",
                    label, code, status
                )
                raise

            delay = 2 ** attempt
            app.logger.warning(
                "%s temporary failure; retrying in %ds: %s",
                label, delay, type(exc).__name__
            )
            time.sleep(delay)


def set_error(device_id, job_id, message):
    with states_lock:
        state = states.get(device_id)
        if state and state["job_id"] == job_id:
            state["status"] = "cloud_error"
            state["last_reply"] = message
            state["updated_at"] = time.time()


def process_utterance(device_id, pcm_bytes, job_id):
    try:
        if client is None:
            raise RuntimeError("GEMINI_API_KEY is missing.")

        wav_data = pcm_to_wav_bytes(pcm_bytes)
        audio_part = types.Part.from_bytes(
            data=wav_data,
            mime_type="audio/wav"
        )

        app.logger.info(
            "Transcription started: device=%s bytes=%d",
            device_id, len(wav_data)
        )

        transcription = gemini_generate(
            AUDIO_MODEL,
            [
                "Transcribe the spoken words in this audio. "
                "The speaker may use Hindi, English or Hinglish. "
                "Return only the transcript.",
                audio_part,
            ],
            "transcription",
        )

        transcript = (getattr(transcription, "text", "") or "").strip()

        if len(transcript) < 2:
            app.logger.warning("No speech transcript produced.")
            set_listening(device_id, job_id)
            return

        with states_lock:
            state = states.get(device_id)
            if state and state["job_id"] == job_id:
                state["last_transcript"] = transcript[:500]
                state["status"] = "thinking"
                state["updated_at"] = time.time()

        app.logger.info("Transcript received: %s", transcript[:160])

        completion = gemini_generate(
            CHAT_MODEL,
            (
                "You are JARVIS, a helpful voice assistant. "
                "Reply in the same language as the user, including "
                "Hindi or Hinglish where appropriate. Keep the answer "
                "concise, about 1-3 short sentences. No markdown tables.\n\n"
                "User said: " + transcript
            ),
            "chat",
        )

        reply = (getattr(completion, "text", "") or "").strip()
        if not reply:
            reply = "Maaf karna, mujhe jawab nahi mil paaya. Dobara bolo."

        with states_lock:
            state = states.get(device_id)
            if state and state["job_id"] == job_id:
                state["last_transcript"] = transcript[:500]
                state["last_reply"] = reply[:1200]
                state["status"] = "response_ready"
                state["updated_at"] = time.time()

        app.logger.info("JARVIS response ready for %s", device_id)

    except Exception as exc:
        app.logger.exception(
            "Cloud processing failed: device=%s error=%s",
            device_id, type(exc).__name__
        )
        set_error(
            device_id,
            job_id,
            "Gemini abhi jawab nahi de pa raha. Thodi der baad dobara bolo."
        )


def set_listening(device_id, job_id):
    with states_lock:
        state = states.get(device_id)
        if state and state["job_id"] == job_id:
            state["status"] = "listening"
            state["updated_at"] = time.time()


@app.get("/")
def index():
    return jsonify({
        "service": "JARVIS Cloud AI",
        "status": "online",
        "health": "/health",
    })


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "device_token_configured": bool(DEVICE_TOKEN),
        "audio_model": AUDIO_MODEL,
        "chat_model": CHAT_MODEL,
        "rms_threshold": RMS_THRESHOLD,
    })


@app.post("/audio")
def audio():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401
    if client is None:
        return jsonify({"error": "Gemini API key missing"}), 503

    device_id = request.headers.get("X-Device-Id", "jarvis-1")[:64]
    raw = request.get_data(cache=False)

    expected_bytes = int(SAMPLE_RATE * CHUNK_SECONDS * 2)
    if len(raw) != expected_bytes:
        return jsonify({
            "error": "invalid PCM size",
            "expected": expected_bytes,
            "received": len(raw),
        }), 400

    samples = np.frombuffer(raw, dtype="<i2")
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    state = get_state(device_id)
    utterance = None
    job_id = None

    with states_lock:
        state["updated_at"] = time.time()
        state["pre_roll"].append(raw)

        if rms >= RMS_THRESHOLD:
            if not state["speaking"]:
                state["speaking"] = True
                state["utterance"] = list(state["pre_roll"])
                state["silence_chunks"] = 0
                state["status"] = "hearing"
            else:
                state["utterance"].append(raw)
                state["silence_chunks"] = 0
        elif state["speaking"]:
            state["utterance"].append(raw)
            state["silence_chunks"] += 1

        chunks = len(state["utterance"])
        silence_limit = int(SILENCE_TO_END_SECONDS / CHUNK_SECONDS)
        max_chunks = int(MAX_UTTERANCE_SECONDS / CHUNK_SECONDS)
        min_chunks = int(MIN_SPEECH_SECONDS / CHUNK_SECONDS)

        if state["speaking"] and (
            state["silence_chunks"] >= silence_limit
            or chunks >= max_chunks
        ):
            candidate = b"".join(state["utterance"])
            state["speaking"] = False
            state["utterance"] = []
            state["silence_chunks"] = 0
            state["pre_roll"].clear()

            if chunks >= min_chunks:
                state["job_id"] += 1
                job_id = state["job_id"]
                state["status"] = "thinking"
                utterance = candidate
                app.logger.info(
                    "Utterance complete: chunks=%d rms=%.1f",
                    chunks, rms
                )
            else:
                state["status"] = "listening"

        current_status = state["status"]

    if utterance is not None:
        threading.Thread(
            target=process_utterance,
            args=(device_id, utterance, job_id),
            daemon=True,
        ).start()

    return jsonify({
        "ok": True,
        "rms": round(rms, 1),
        "status": current_status,
    })


@app.get("/result")
def result():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401

    device_id = request.headers.get("X-Device-Id", "jarvis-1")[:64]
    state = get_state(device_id)

    with states_lock:
        return jsonify({
            "status": state["status"],
            "transcript": state["last_transcript"],
            "reply": state["last_reply"],
            "updated_at": state["updated_at"],
        })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        debug=False,
    )
