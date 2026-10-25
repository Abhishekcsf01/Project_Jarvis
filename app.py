
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

# ================= CONFIGURATION =================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "").strip()

# Render Environment variables override these defaults.
AUDIO_MODEL = os.environ.get(
    "AUDIO_MODEL", "gemini-3.8-flash"
).strip()

CHAT_MODEL = os.environ.get(
    "CHAT_MODEL", "gemini-3.8-flash"
).strip()

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.5
SILENCE_TO_END_SECONDS = 1.25
MIN_SPEECH_SECONDS = 0.45
MAX_UTTERANCE_SECONDS = 18

# Increase this if background noise triggers speech.
# Decrease it if quiet speech is not detected.
RMS_THRESHOLD = 450

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Intended for a single Render instance and one Gunicorn worker.
states = {}
states_lock = threading.Lock()


# ================= AUTHENTICATION =================

def authorized():
    if not DEVICE_TOKEN:
        return False

    supplied_token = request.headers.get("X-Device-Token", "")
    return supplied_token == DEVICE_TOKEN


# ================= DEVICE STATE =================

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


# ================= AUDIO CONVERSION =================

def pcm_to_wav_bytes(pcm_bytes):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm_bytes)

    return buffer.getvalue()


# ================= GEMINI PROCESSING =================

def process_utterance(device_id, pcm_bytes, job_id):
    try:
        if client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is missing in Render Environment."
            )

        wav_data = pcm_to_wav_bytes(pcm_bytes)

        app.logger.info(
            "Transcription started: device=%s model=%s audio_bytes=%d",
            device_id, AUDIO_MODEL, len(wav_data)
        )

        audio_part = types.Part.from_bytes(
            data=wav_data,
            mime_type="audio/wav"
        )

        transcription = client.models.generate_content(
            model=AUDIO_MODEL,
            contents=[
                (
                    "Transcribe only the spoken words in this audio. "
                    "The speaker may use Hindi, English, or Hinglish. "
                    "Return only the transcript, without commentary."
                ),
                audio_part,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=300,
            ),
        )

        transcript = (transcription.text or "").strip()

        app.logger.info(
            "Transcription finished: device=%s characters=%d",
            device_id, len(transcript)
        )

        if len(transcript) < 2:
            with states_lock:
                state = states.get(device_id)

                if state and state["job_id"] == job_id:
                    state["status"] = "listening"
                    state["updated_at"] = time.time()

            app.logger.warning("Empty transcript; returning to listening.")
            return

        # Generate the assistant response.
        app.logger.info(
            "Chat started: device=%s model=%s",
            device_id, CHAT_MODEL
        )

        completion = client.models.generate_content(
            model=CHAT_MODEL,
            contents=(
                "You are JARVIS, a helpful voice assistant. "
                "Reply in the same language as the user. "
                "The user may speak Hindi, English, or Hinglish. "
                "Keep the reply concise, ideally 1-3 short sentences. "
                "Avoid markdown tables.\n\n"
                "User said: " + transcript
            ),
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=160,
            ),
        )

        reply = (completion.text or "").strip()

        if not reply:
            reply = "Sorry, I could not generate a response."

        app.logger.info(
            "Chat finished: device=%s reply_characters=%d",
            device_id, len(reply)
        )

        with states_lock:
            state = states.get(device_id)

            if state and state["job_id"] == job_id:
                state["last_transcript"] = transcript[:500]
                state["last_reply"] = reply[:1200]
                state["status"] = "response_ready"
                state["updated_at"] = time.time()

    except Exception as exc:
        # Full details go to Render logs; secrets are not sent to ESP32.
        app.logger.exception(
            "Cloud processing failed: device=%s error=%s",
            device_id, type(exc).__name__
        )

        with states_lock:
            state = states.get(device_id)

            if state and state["job_id"] == job_id:
                state["status"] = "cloud_error"
                state["last_reply"] = (
                    "Cloud error. Check Render logs and Gemini API settings."
                )
                state["updated_at"] = time.time()


# ================= HEALTH CHECK =================

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
        "sample_rate": SAMPLE_RATE,
        "rms_threshold": RMS_THRESHOLD,
    })


# ================= RECEIVE ESP32 AUDIO =================

@app.post("/audio")
def audio():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401

    if client is None:
        return jsonify({
            "error": "GEMINI_API_KEY is not configured on the server"
        }), 503

    device_id = request.headers.get(
        "X-Device-Id", "jarvis-1"
    )[:64]

    raw = request.get_data(cache=False)

    expected_bytes = int(SAMPLE_RATE * CHUNK_SECONDS * 2)

    if len(raw) != expected_bytes:
        app.logger.warning(
            "Invalid audio size: expected=%d received=%d",
            expected_bytes, len(raw)
        )

        return jsonify({
            "error": "invalid PCM chunk size",
            "expected_bytes": expected_bytes,
            "received_bytes": len(raw),
        }), 400

    # Mono signed 16-bit little-endian PCM.
    samples = np.frombuffer(raw, dtype="<i2")
    rms = float(
        np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    )

    state = get_state(device_id)
    utterance_to_process = None
    job_id = None

    with states_lock:
        state["updated_at"] = time.time()
        state["pre_roll"].append(raw)

        app.logger.info(
            "audio_chunk device=%s rms=%.1f threshold=%d speaking=%s status=%s",
            device_id,
            rms,
            RMS_THRESHOLD,
            state["speaking"],
            state["status"],
        )

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
        silence_limit = int(
            SILENCE_TO_END_SECONDS / CHUNK_SECONDS
        )
        max_chunks = int(
            MAX_UTTERANCE_SECONDS / CHUNK_SECONDS
        )
        min_chunks = int(
            MIN_SPEECH_SECONDS / CHUNK_SECONDS
        )

        if state["speaking"] and (
            state["silence_chunks"] >= silence_limit
            or elapsed_chunks >= max_chunks
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

                app.logger.info(
                    "utterance_complete device=%s chunks=%d bytes=%d job_id=%d",
                    device_id,
                    elapsed_chunks,
                    len(candidate),
                    job_id,
                )
            else:
                state["status"] = "listening"

                app.logger.info(
                    "utterance_discarded chunks=%d minimum=%d",
                    elapsed_chunks, min_chunks
                )

        response_status = state["status"]

    # Gemini processing happens outside the request lock.
    if utterance_to_process is not None:
        thread = threading.Thread(
            target=process_utterance,
            args=(device_id, utterance_to_process, job_id),
            daemon=True,
        )
        thread.start()

    return jsonify({
        "ok": True,
        "rms": round(rms, 1),
        "status": response_status,
    })


# ================= ESP32 POLLS FOR REPLY =================

@app.get("/result")
def result():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401

    device_id = request.headers.get(
        "X-Device-Id", "jarvis-1"
    )[:64]

    state = get_state(device_id)

    with states_lock:
        return jsonify({
            "status": state["status"],
            "transcript": state["last_transcript"],
            "reply": state["last_reply"],
            "updated_at": state["updated_at"],
        })


# ================= LOCAL TEST =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
