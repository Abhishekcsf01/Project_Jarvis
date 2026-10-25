import io
import os
import time
import wave
import threading
from collections import deque

import numpy as np
from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "").strip()
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# This in-memory state is intended for a single-device prototype.
# Render restarts/instance changes clear this state; do not treat it as durable storage.
states = {}
states_lock = threading.Lock()

SAMPLE_RATE = 16000
# Each firmware request contains 0.5 seconds of mono signed 16-bit PCM.
CHUNK_SECONDS = 0.5
SILENCE_TO_END_SECONDS = 1.25
MIN_SPEECH_SECONDS = 0.45
MAX_UTTERANCE_SECONDS = 18
# RMS is calculated on the server. Tune if room noise triggers speech or quiet speech is missed.
RMS_THRESHOLD = 450


def authorized():
    if not DEVICE_TOKEN:
        return False
    return request.headers.get("X-Device-Token", "") == DEVICE_TOKEN


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


def pcm_to_wav_bytes(pcm_bytes):
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm_bytes)
    return bio.getvalue()


def process_utterance(device_id, pcm_bytes, job_id):
    state = get_state(device_id)
    try:
        if client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured on the server.")

        wav_data = pcm_to_wav_bytes(pcm_bytes)
        audio_file = io.BytesIO(wav_data)
        audio_file.name = "speech.wav"

        transcription = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio_file,
            # Let the transcription model detect Hindi, English, or Hinglish automatically.
        )
        transcript = (getattr(transcription, "text", "") or "").strip()

        if not transcript or len(transcript) < 2:
            with states_lock:
                current = states.get(device_id)
                if current and current["job_id"] == job_id:
                    current["status"] = "listening"
                    current["updated_at"] = time.time()
            return

        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are JARVIS, a helpful concise voice assistant. "
                        "Reply in the same language as the user. The user may speak Hindi, "
                        "English, or Hinglish. Keep replies short enough for a small OLED "
                        "(ideally 1-3 short sentences). No markdown tables."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
            max_tokens=120,
        )
        reply = (completion.choices[0].message.content or "").strip()

        with states_lock:
            current = states.get(device_id)
            if current and current["job_id"] == job_id:
                current["last_transcript"] = transcript[:500]
                current["last_reply"] = reply[:1200]
                current["status"] = "response_ready"
                current["updated_at"] = time.time()

    except Exception as exc:
        # Keep detailed errors in Render logs, but return a generic status to the device.
        app.logger.exception("Cloud processing failed for device %s", device_id)
        with states_lock:
            current = states.get(device_id)
            if current and current["job_id"] == job_id:
                current["status"] = "cloud_error"
                current["last_reply"] = "Cloud error. Check Render logs and API settings."
                current["updated_at"] = time.time()


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
        "openai_key_configured": bool(OPENAI_API_KEY),
        "device_token_configured": bool(DEVICE_TOKEN),
    })


@app.post("/audio")
def audio():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401
    if client is None:
        return jsonify({"error": "server API key is not configured"}), 503

    device_id = request.headers.get("X-Device-Id", "jarvis-1")[:64]
    raw = request.get_data(cache=False)
    expected_bytes = int(SAMPLE_RATE * CHUNK_SECONDS * 2)
    if len(raw) != expected_bytes:
        return jsonify({"error": f"expected {expected_bytes} PCM bytes, got {len(raw)}"}), 400

    # Audio is signed little-endian 16-bit PCM. Energy/VAD runs on the cloud, not the ESP32.
    samples = np.frombuffer(raw, dtype="<i2")
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    state = get_state(device_id)
    utterance_to_process = None
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

        elapsed_chunks = len(state["utterance"])
        silence_limit = int(SILENCE_TO_END_SECONDS / CHUNK_SECONDS)
        max_chunks = int(MAX_UTTERANCE_SECONDS / CHUNK_SECONDS)
        min_chunks = int(MIN_SPEECH_SECONDS / CHUNK_SECONDS)

        if state["speaking"] and (
            state["silence_chunks"] >= silence_limit or elapsed_chunks >= max_chunks
        ):
            candidate = b"".join(state["utterance"])
            state["speaking"] = False
            state["utterance"] = []
            state["silence_chunks"] = 0
            state["pre_roll"].clear()

            if elapsed_chunks >= min_chunks:
                state["job_id"] += 1
                job_id = state["job_id"]
                state["status"] = "thinking"
                utterance_to_process = candidate

    if utterance_to_process is not None:
        thread = threading.Thread(
            target=process_utterance,
            args=(device_id, utterance_to_process, job_id),
            daemon=True,
        )
        thread.start()

    return jsonify({"ok": True, "rms": round(rms, 1), "status": state["status"]})


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
    # Local test only. Render uses gunicorn.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")), debug=False)
