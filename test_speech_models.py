#!/usr/bin/env python3
import os
import sys
import time
import struct
import socket
import threading
import queue
import urllib.request
import numpy as np

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Fix for scikit-learn OpenMP TLS allocation error (copied from demo_gestures)
os.environ['LD_PRELOAD'] = '/home/unitree/miniconda3/envs/demo/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0'

# ── Config ────────────────────────────────────────────────────────────────────
NETWORK_INTERFACE = sys.argv[1] if len(sys.argv) > 1 else "enP8p1s0"
MULTICAST_GROUP   = "239.168.123.161"
MULTICAST_PORT    = 5555
SAMPLE_RATE       = 16000
CHUNK_SIZE        = 1280
VAD_THRESHOLD     = 0.025
SPEECH_TIMEOUT_S  = 1.0

# Piper TTS Config
PIPER_MODEL_NAME = "en_US-ryan-high"
PIPER_ONNX_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/{PIPER_MODEL_NAME}.onnx"
PIPER_JSON_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/{PIPER_MODEL_NAME}.onnx.json"
PIPER_ONNX_PATH = f"models/{PIPER_MODEL_NAME}.onnx"
PIPER_JSON_PATH = f"models/{PIPER_MODEL_NAME}.onnx.json"

# STT Config
DEFAULT_STT_MODEL = "openai/whisper-tiny.en"

def get_local_ip(interface: str) -> str:
    import subprocess
    try:
        out = subprocess.check_output(["ip", "-4", "addr", "show", interface], text=True)
        for line in out.splitlines():
            if "inet " in line:
                return line.strip().split()[1].split("/")[0]
    except Exception:
        pass
    return "127.0.0.1"

LOCAL_IP = get_local_ip(NETWORK_INTERFACE)
print(f"[TEST] Interface={NETWORK_INTERFACE}, local_ip={LOCAL_IP}")

# ── Hardware Init ─────────────────────────────────────────────────────────────
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.rpc.client import Client

print(f"[TEST] Initializing DDS on {NETWORK_INTERFACE} ...")
ChannelFactoryInitialize(0, NETWORK_INTERFACE)
time.sleep(0.5)

audio_client = AudioClient()
audio_client.Init()
audio_client.SetVolume(100)

voice_client = Client("voice", False)
voice_client.SetTimeout(5.0)
voice_client._SetApiVerson("1.0.0.0")
voice_client._RegistApi(1008, 0)
voice_client._Call(1008, '{"mode": 1}') # Mic active
print("[TEST] Hardware initialized.")

# ── TTS Setup ─────────────────────────────────────────────────────────────────
def download_piper_model():
    if not os.path.exists(PIPER_ONNX_PATH):
        print(f"[TEST] Downloading TTS model {PIPER_MODEL_NAME} (ONNX)...")
        urllib.request.urlretrieve(PIPER_ONNX_URL, PIPER_ONNX_PATH)
    if not os.path.exists(PIPER_JSON_PATH):
        print(f"[TEST] Downloading TTS model {PIPER_MODEL_NAME} (JSON)...")
        urllib.request.urlretrieve(PIPER_JSON_URL, PIPER_JSON_PATH)

download_piper_model()
from piper.voice import PiperVoice
piper_voice = PiperVoice.load(PIPER_ONNX_PATH, PIPER_JSON_PATH)
target_fs = 16000
piper_fs = piper_voice.config.sample_rate

def speak(text: str):
    print(f"[TTS] Synthesizing: {text}")
    stream_id = f"test_{int(time.time() * 1000)}"
    boost_factor = 5.0
    
    for audio_chunk in piper_voice.synthesize(text):
        pcm = audio_chunk.audio_int16_bytes
        samples = np.frombuffer(pcm, dtype=np.int16)
        
        # Resample if needed
        if piper_fs != target_fs:
            n_out = int(len(samples) * target_fs / piper_fs)
            samples = np.interp(
                np.linspace(0, len(samples), n_out, endpoint=False),
                np.arange(len(samples)),
                samples
            ).astype(np.int16)
        
        boosted = np.clip(samples.astype(np.float32) * boost_factor, -32768, 32767).astype(np.int16)
        
        try:
            audio_client.PlayStream("jarvis_brain", stream_id, boosted.tobytes())
        except TypeError:
            audio_client.PlayStream("jarvis_brain", stream_id, list(boosted.tobytes()))
            
        time.sleep((len(samples) / target_fs) * 0.95)
        
    try:
        audio_client.PlayStop("jarvis_brain")
    except Exception:
        pass

# ── STT Setup ─────────────────────────────────────────────────────────────────
from transformers import pipeline

class STTTester:
    def __init__(self, model_id=DEFAULT_STT_MODEL):
        self.model_id = model_id
        self.load_model(model_id)
        
    def load_model(self, model_id):
        print(f"[STT] Loading Hugging Face model '{model_id}' on CPU...")
        self.model_id = model_id
        # We specify cache_dir="./models" so it downloads locally to our models folder
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device="cpu",
            model_kwargs={"cache_dir": "./models"}
        )
        print(f"[STT] Model '{model_id}' ready.")
        
    def transcribe(self, audio_np: np.ndarray) -> str:
        # Pipeline expects audio as dict or dict array for timestamps, but raw array works for simple audio
        res = self.pipe(audio_np)
        if isinstance(res, dict):
            return res.get("text", "").strip()
        elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
            return res[0].get("text", "").strip()
        return str(res)

stt_tester = STTTester()

# ── Mic Multicast Receiver ────────────────────────────────────────────────────
audio_q = queue.Queue(maxsize=200)

def mic_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    sock.bind(('', MULTICAST_PORT))
    
    try:
        mreq = struct.pack("4s4s", socket.inet_aton(MULTICAST_GROUP), socket.inet_aton(LOCAL_IP))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    except Exception:
        print(f"[TEST] Multicast join failed on {LOCAL_IP}, trying INADDR_ANY")
        mreq = struct.pack("4sl", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
    sock.settimeout(1.0)
    print(f"[TEST] Listening to mic on {MULTICAST_GROUP}:{MULTICAST_PORT}")
    
    buf = np.array([], dtype=np.int16)
    while True:
        try:
            data, _ = sock.recvfrom(8192)
            packet = np.frombuffer(data, dtype=np.int16)
            buf = np.concatenate([buf, packet])
            while len(buf) >= CHUNK_SIZE:
                chunk, buf = buf[:CHUNK_SIZE].copy(), buf[CHUNK_SIZE:]
                if audio_q.full():
                    try: audio_q.get_nowait()
                    except queue.Empty: pass
                audio_q.put_nowait(chunk)
        except socket.timeout:
            continue
        except Exception as exc:
            print(f"[TEST MIC ERROR] {exc}")
            break

threading.Thread(target=mic_thread, daemon=True).start()

def vad_prob(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) / 32768.0

# ── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    speak("Testing framework ready. I am listening.")
    
    while True:
        # Drain queue before starting to avoid immediate false triggers
        while not audio_q.empty():
            try: audio_q.get_nowait()
            except queue.Empty: break
            
        print("\n[TEST] Waiting for speech...")
        
        # 1. Wait for speech to start
        speech_consec = 0
        while True:
            chunk = audio_q.get()
            if vad_prob(chunk) >= VAD_THRESHOLD:
                speech_consec += 1
            else:
                speech_consec = 0
            if speech_consec >= 3:
                break
                
        print("[TEST] Speech detected, recording...")
        frames = [chunk]
        
        # 2. Record until silence
        silence_limit = max(1, round(SPEECH_TIMEOUT_S * SAMPLE_RATE / CHUNK_SIZE))
        silence_chunks = 0
        
        while True:
            chunk = audio_q.get()
            frames.append(chunk)
            if vad_prob(chunk) < VAD_THRESHOLD:
                silence_chunks += 1
                if silence_chunks >= silence_limit:
                    break
            else:
                silence_chunks = 0
                
        print(f"[TEST] Recording complete. Chunks: {len(frames)}")
        
        # Normalize to float32 for Transformers
        audio_np = np.concatenate(frames).astype(np.float32) / 32768.0
        
        print("[TEST] Transcribing...")
        text = stt_tester.transcribe(audio_np)
        print(f"[STT Result] '{text}'")
        
        if text:
            speak(f"You said: {text}")
        else:
            speak("I didn't catch that.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import json
        print("\n[TEST] Shutting down.")
        try:
            voice_client._Call(1008, json.dumps({"mode": 2})) # Mic idle
        except Exception:
            pass
