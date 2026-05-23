#!/usr/bin/env python3
"""
test_speech_models.py — STT model benchmark for Unitree G1 robot.

Tests ANY HuggingFace speech-to-text model for accuracy using the robot's
onboard mic and Piper TTS (ryan-high voice) for audio feedback.

── Usage ────────────────────────────────────────────────────────────────────────
  python test_speech_models.py [interface] [--model MODEL_ID] [--keep-models]

  python test_speech_models.py enP8p1s0
  python test_speech_models.py enP8p1s0 --model openai/whisper-base.en
  python test_speech_models.py enP8p1s0 --model ibm-granite/granite-speech-4.1-2b-plus
  python test_speech_models.py enP8p1s0 --model UsefulSensors/moonshine-tiny
  python test_speech_models.py enP8p1s0 --model facebook/wav2vec2-base-960h

  --keep-models   Do NOT delete model files from disk when switching.
                  Default: old model files ARE deleted when you switch.

── Voice Commands (while running) ───────────────────────────────────────────────
  "switch model whisper base"           → unload current, download+load whisper-base.en
  "switch model moonshine tiny"         → unload current, download+load moonshine-tiny
  "switch model granite"                → unload current, download+load granite-speech
  "switch model wav2vec"                → unload current, download+load wav2vec2-base
  "switch model vosk"                   → download+load vosk-model-small-en-us
  "switch model <full/hf-id>"           → load any HuggingFace model by full ID
  "delete model"                        → delete current model files from disk, reload
  "list models"                         → speak which models are cached on disk
  Ctrl-C                                → graceful shutdown

── Supported model families (auto-detected) ─────────────────────────────────────
  Whisper          openai/whisper-{tiny,base,small,medium,large}[.en]
  Moonshine        UsefulSensors/moonshine-{tiny,base}
  Granite Speech   ibm-granite/granite-speech-*
  Wav2Vec2 / CTC   facebook/wav2vec2-*, facebook/hubert-*
  SpeechT5         microsoft/speecht5_asr  (uses pipeline API)
  Vosk             vosk-model-small-en-us-0.15 (uses vosk pip package)
  Generic fallback anything else with generate()

── Download & cache ─────────────────────────────────────────────────────────────
  All models saved to ./models/  (HuggingFace standard cache layout)
  Cache folder per model: models/models--<org>--<model-name>/
  Deletion removes the entire per-model cache folder.
"""

import os
import sys
import gc
import json
import re
import time
import shutil
import struct
import signal
import socket
import threading
import queue
import urllib.request
import zipfile
import traceback
import numpy as np

os.makedirs("models", exist_ok=True)

# ── CLI args ──────────────────────────────────────────────────────────────────
NETWORK_INTERFACE = "enP8p1s0"
DEFAULT_STT_MODEL = "openai/whisper-tiny.en"
KEEP_MODELS       = False   # if True, never delete downloaded model files

_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    a = _args[_i]
    if a == "--model" and _i + 1 < len(_args):
        DEFAULT_STT_MODEL = _args[_i + 1]; _i += 2
    elif a == "--keep-models":
        KEEP_MODELS = True; _i += 1
    elif not a.startswith("--"):
        NETWORK_INTERFACE = a; _i += 1
    else:
        _i += 1

# ── Config ────────────────────────────────────────────────────────────────────
MULTICAST_GROUP    = "239.168.123.161"
MULTICAST_PORT     = 5555
SAMPLE_RATE        = 16000
CHUNK_SIZE         = 1280          # 80 ms @ 16 kHz

VAD_THRESHOLD      = 0.025         # normalized RMS energy
VAD_CONFIRM_CHUNKS = 3             # ~240 ms sustained to confirm real speech
SPEECH_TIMEOUT_S   = 1.2           # silence after speech → stop recording
MAX_RECORD_S       = 8.0           # hard cap
TTS_ECHO_DRAIN_S   = 0.20          # extra silence after TTS clears before VAD starts

# [CRITICAL FIX]
# Jetson Orin / ARM CPUs do NOT have native bfloat16 mathematical hardware.
# PyTorch silently emulates it in software, causing models to be 100x slower.
# We MUST use float32 for PyTorch to leverage actual hardware execution.
USE_BFLOAT16       = False         

MODELS_DIR       = "./models"
PIPER_MODEL_NAME = "en_US-ryan-high"
PIPER_ONNX_URL   = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    f"en/en_US/ryan/high/{PIPER_MODEL_NAME}.onnx"
)
PIPER_JSON_URL   = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    f"en/en_US/ryan/high/{PIPER_MODEL_NAME}.onnx.json"
)
PIPER_ONNX_PATH  = f"{MODELS_DIR}/{PIPER_MODEL_NAME}.onnx"
PIPER_JSON_PATH  = f"{MODELS_DIR}/{PIPER_MODEL_NAME}.onnx.json"

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_local_ip(interface: str) -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            ["ip", "-4", "addr", "show", interface], text=True
        )
        for line in out.splitlines():
            if "inet " in line:
                return line.strip().split()[1].split("/")[0]
    except Exception:
        pass
    return "127.0.0.1"

LOCAL_IP = get_local_ip(NETWORK_INTERFACE)
print(f"[TEST] Interface={NETWORK_INTERFACE}, local_ip={LOCAL_IP}")

def model_cache_folder(model_id: str) -> str:
    safe = model_id.replace("/", "--")
    return os.path.join(MODELS_DIR, f"models--{safe}")

def model_disk_size_mb(model_id: str) -> float:
    folder = model_cache_folder(model_id)
    if not os.path.isdir(folder):
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / (1024 * 1024)

def delete_model_files(model_id: str) -> bool:
    folder = model_cache_folder(model_id)
    if os.path.isdir(folder):
        size_mb = model_disk_size_mb(model_id)
        print(f"[CACHE] Deleting {folder}  ({size_mb:.0f} MB) ...")
        shutil.rmtree(folder, ignore_errors=True)
        print(f"[CACHE] Deleted.")
        return True
    print(f"[CACHE] No cache folder found for '{model_id}' at {folder}")
    return False

def list_cached_models() -> list:
    cached = []
    if not os.path.isdir(MODELS_DIR):
        return cached
    for name in os.listdir(MODELS_DIR):
        if name.startswith("models--"):
            parts = name[len("models--"):].split("--", 1)
            if len(parts) == 2:
                cached.append(f"{parts[0]}/{parts[1]}")
            else:
                cached.append(parts[0])
    return cached

def _download_vosk_model(model_id: str):
    """Downloads and extracts the Vosk model from alphacephei."""
    target_dir = model_cache_folder(model_id)
    if os.path.isdir(target_dir):
        return
    print(f"[VOSK] Downloading {model_id} ...")
    zip_path = os.path.join(MODELS_DIR, f"{model_id}.zip")
    urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path)
    print(f"[VOSK] Extracting {model_id} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    os.remove(zip_path)

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
voice_client._Call(1008, '{"mode": 1}')   # mic active
print("[TEST] Hardware initialized.")

def _cleanup(sig=None, frame=None):
    print("\n[TEST] Shutting down ...")
    try:
        voice_client._Call(1008, json.dumps({"mode": 2}))
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT,  _cleanup)
signal.signal(signal.SIGTERM, _cleanup)

# ── TTS — Piper ryan-high ─────────────────────────────────────────────────────
def _download_piper():
    if not os.path.exists(PIPER_ONNX_PATH):
        print(f"[TTS] Downloading {PIPER_MODEL_NAME}.onnx ...")
        urllib.request.urlretrieve(PIPER_ONNX_URL, PIPER_ONNX_PATH)
    if not os.path.exists(PIPER_JSON_PATH):
        print(f"[TTS] Downloading {PIPER_MODEL_NAME}.onnx.json ...")
        urllib.request.urlretrieve(PIPER_JSON_URL, PIPER_JSON_PATH)

_download_piper()

from piper.voice import PiperVoice
_piper_voice = PiperVoice.load(PIPER_ONNX_PATH, PIPER_JSON_PATH)
_piper_fs    = _piper_voice.config.sample_rate
_TARGET_FS   = 16000
_TTS_BOOST   = 5.0

_speaking_lock = threading.Lock()
_is_speaking   = threading.Event()

def speak(text: str):
    with _speaking_lock:
        _is_speaking.set()
        print(f"[TTS] {text}")
        stream_id = f"stt_test_{int(time.time() * 1000)}"
        try:
            for chunk in _piper_voice.synthesize(text):
                samples = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                if _piper_fs != _TARGET_FS:
                    n_out   = int(len(samples) * _TARGET_FS / _piper_fs)
                    samples = np.interp(
                        np.linspace(0, len(samples), n_out, endpoint=False),
                        np.arange(len(samples)),
                        samples,
                    ).astype(np.int16)
                boosted   = np.clip(
                    samples.astype(np.float32) * _TTS_BOOST, -32768, 32767
                ).astype(np.int16)
                pcm_bytes = boosted.tobytes()
                try:
                    audio_client.PlayStream("stt_test", stream_id, pcm_bytes)
                except TypeError:
                    audio_client.PlayStream("stt_test", stream_id, list(pcm_bytes))
                time.sleep((len(samples) / _TARGET_FS) * 0.95)
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            try:
                audio_client.PlayStop("stt_test")
            except Exception:
                pass
            time.sleep(0.4)
            _is_speaking.clear()

# ── STT — model family detection & per-family pipelines ──────────────────────
import torch
import multiprocessing
try:
    torch.set_num_threads(multiprocessing.cpu_count())
except Exception:
    pass

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoModelForCausalLM,
)

def _detect_family(model_id: str, arch: str) -> str:
    mid = model_id.lower()
    arc = arch.lower()
    if "vosk" in mid:           return "vosk"
    if "granite" in mid:        return "granite"
    if "moonshine" in mid:      return "moonshine"
    if "speecht5" in mid:       return "speecht5"
    if "whisper" in mid:        return "whisper"
    if "wav2vec2" in mid:       return "ctc"
    if "hubert" in mid:         return "ctc"
    if "wav2vec" in mid:        return "ctc"
    if "ctc" in arc:            return "ctc"
    if "wav2vec" in arc:        return "ctc"
    if "speechseq2seq" in arc:  return "whisper"
    if "whisper" in arc:        return "whisper"
    if "moonshine" in arc:      return "moonshine"
    return "seq2seq"

def _torch_dtype():
    if USE_BFLOAT16:
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16) + torch.zeros(1, dtype=torch.bfloat16)
            return torch.bfloat16
        except Exception:
            print("[STT] bfloat16 not supported on this CPU, falling back to float32.")
    return torch.float32

class STTModel:
    def __init__(self, model_id: str):
        self.model_id  = model_id
        self.model     = None
        self.processor = None
        self.pipeline  = None
        self.family    = None
        self._dtype    = _torch_dtype()
        self._load(model_id)

    def _load(self, model_id: str):
        cache = MODELS_DIR
        print(f"\n[STT] ── Loading '{model_id}' ──")
        print(f"[STT] Cache folder : {model_cache_folder(model_id)}")
        
        self.family = _detect_family(model_id, "")
        
        if self.family == "vosk":
            _download_vosk_model(model_id)
            try:
                import vosk
            except ImportError:
                print("\n[STT] Vosk is not installed. Please run `pip install vosk`.\n[STT] Falling back to whisper tiny...")
                self.model_id = "openai/whisper-tiny.en"
                self._load(self.model_id)
                return

            model_path = os.path.join(model_cache_folder(model_id), model_id)
            vosk.SetLogLevel(-1)
            self.model = vosk.Model(model_path)
            self.processor = None
            self.pipeline = None
            size_mb = model_disk_size_mb(model_id)
            print(f"[STT] '{model_id}' ready. Disk: {size_mb:.0f} MB")
            return

        # ── Hugging Face fallback loading ──
        print(f"[STT] torch dtype  : {self._dtype}")
        print("[STT] Loading processor/config ...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, cache_dir=cache, trust_remote_code=True,
            )
        except Exception as e:
            print(f"[STT] AutoProcessor failed ({e}), trying pipeline fallback ...")
            self._load_as_pipeline(model_id, cache)
            return

        config = AutoConfig.from_pretrained(
            model_id, cache_dir=cache, trust_remote_code=True
        )
        archs       = getattr(config, "architectures", [""]) or [""]
        arch        = archs[0] if archs else ""
        self.family = _detect_family(model_id, arch)
        print(f"[STT] Architecture : {arch}")
        print(f"[STT] Family       : {self.family}")

        print("[STT] Loading model weights ...")

        if self.family == "speecht5":
            self._load_as_pipeline(model_id, cache)
            return

        if self.family == "ctc":
            self.model = AutoModelForCTC.from_pretrained(
                model_id, cache_dir=cache, trust_remote_code=True,
                torch_dtype=self._dtype,
            )
        elif self.family == "granite":
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, cache_dir=cache, trust_remote_code=True,
                torch_dtype=self._dtype,
            )
        elif self.family == "moonshine":
            try:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, cache_dir=cache, trust_remote_code=True,
                    torch_dtype=self._dtype,
                )
            except Exception:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    model_id, cache_dir=cache, trust_remote_code=True,
                    torch_dtype=self._dtype,
                )
        elif self.family == "whisper":
            try:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, cache_dir=cache, trust_remote_code=True,
                    torch_dtype=self._dtype,
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, cache_dir=cache, trust_remote_code=True,
                    torch_dtype=self._dtype,
                )
        else:
            try:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, cache_dir=cache, trust_remote_code=True,
                    torch_dtype=self._dtype,
                )
            except Exception as e1:
                print(f"[STT] Seq2Seq failed ({e1}), trying CausalLM ...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id, cache_dir=cache, trust_remote_code=True,
                        torch_dtype=self._dtype,
                    )
                except Exception as e2:
                    print(f"[STT] CausalLM failed ({e2}), using pipeline ...")
                    self._load_as_pipeline(model_id, cache)
                    return

        self.model.eval()
        self.model.to("cpu")
        size_mb = model_disk_size_mb(model_id)
        print(f"[STT] '{model_id}' ready. Disk: {size_mb:.0f} MB")

    def _load_as_pipeline(self, model_id: str, cache: str):
        from transformers import pipeline
        print("[STT] Loading via transformers.pipeline ...")
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            model_kwargs={"cache_dir": cache},
            device="cpu",
            trust_remote_code=True,
        )
        self.family = "pipeline"
        print(f"[STT] '{model_id}' loaded via pipeline.")

    def unload(self, delete_files: bool = False):
        mid = self.model_id
        if self.model     is not None: del self.model;     self.model     = None
        if self.processor is not None: del self.processor; self.processor = None
        if self.pipeline  is not None: del self.pipeline;  self.pipeline  = None
        gc.collect()
        print(f"[STT] '{mid}' unloaded from RAM.")
        if delete_files:
            delete_model_files(mid)

    def transcribe(self, audio_np: np.ndarray) -> str:
        try:
            if self.family == "vosk":
                return self._infer_vosk(audio_np)
            elif self.family == "granite":
                return self._infer_granite(audio_np)
            elif self.family == "moonshine":
                return self._infer_moonshine(audio_np)
            elif self.family == "ctc":
                return self._infer_ctc(audio_np)
            elif self.family == "pipeline":
                return self._infer_pipeline(audio_np)
            else:
                return self._infer_seq2seq(audio_np)
        except Exception as e:
            print(f"[STT ERROR] transcribe() raised: {e}")
            traceback.print_exc()
            return ""
            
    # ── Vosk ──────────────────────────────────────────────────────────────────
    def _infer_vosk(self, audio_np: np.ndarray) -> str:
        import vosk
        # Convert float32 [-1.0, 1.0] back to int16 PCM expected by Kaldi
        pcm_data = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        rec = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.AcceptWaveform(pcm_data)
        res = json.loads(rec.FinalResult())
        return res.get("text", "").strip()

    # ── Granite Speech (granite-4.0-1b-speech AND granite-speech-4.1-*) ──────
    def _infer_granite(self, audio_np: np.ndarray) -> str:
        tokenizer = self.processor.tokenizer

        chat = [{"role": "user", "content": "<|audio|>Transcribe the speech into written text."}]
        prompt_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        try:
            model_inputs = self.processor(
                text=prompt_text,
                audio=audio_np,       
                return_tensors="pt",
                sampling_rate=SAMPLE_RATE,
            )
        except Exception:
            try:
                model_inputs = self.processor(
                    text=prompt_text,
                    audios=audio_np,  
                    return_tensors="pt",
                    sampling_rate=SAMPLE_RATE,
                )
            except Exception:
                wav_tensor = torch.from_numpy(audio_np)
                model_inputs = self.processor(
                    prompt_text,
                    wav_tensor,
                    return_tensors="pt",
                    sampling_rate=SAMPLE_RATE,
                )

        model_inputs = {k: v.to("cpu") for k, v in model_inputs.items()
                        if isinstance(v, torch.Tensor)}

        duration       = len(audio_np) / SAMPLE_RATE
        max_new_tokens = max(int(duration * 8), 32)

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, num_input_tokens:]
        text = tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )
        return text.strip()

    # ── Moonshine ─────────────────────────────────────────────────────────────
    def _infer_moonshine(self, audio_np: np.ndarray) -> str:
        inputs = self.processor(audio_np, return_tensors="pt", sampling_rate=SAMPLE_RATE)
        inputs = {k: v.to("cpu") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        duration       = len(audio_np) / SAMPLE_RATE
        max_new_tokens = max(int(duration * 5), 16)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    # ── Whisper / generic Seq2Seq ─────────────────────────────────────────────
    def _infer_seq2seq(self, audio_np: np.ndarray) -> str:
        inputs = self.processor(audio_np, return_tensors="pt", sampling_rate=SAMPLE_RATE)
        inputs = {k: v.to("cpu") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        duration       = len(audio_np) / SAMPLE_RATE
        max_new_tokens = max(int(duration * 6), 30)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # ── CTC / Wav2Vec2 ────────────────────────────────────────────────────────
    def _infer_ctc(self, audio_np: np.ndarray) -> str:
        inputs = self.processor(audio_np, return_tensors="pt", sampling_rate=SAMPLE_RATE)
        inputs = {k: v.to("cpu") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            logits   = self.model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0].strip()

    # ── pipeline() fallback ───────────────────────────────────────────────────
    def _infer_pipeline(self, audio_np: np.ndarray) -> str:
        result = self.pipeline({"array": audio_np, "sampling_rate": SAMPLE_RATE})
        return result.get("text", "").strip()


# ── Mic multicast receiver ────────────────────────────────────────────────────
_audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=300)

def _mic_thread():
    while True:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
            sock.bind(('', MULTICAST_PORT))
            try:
                mreq = struct.pack("4s4s",
                    socket.inet_aton(MULTICAST_GROUP), socket.inet_aton(LOCAL_IP))
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            except Exception:
                mreq = struct.pack("4sl",
                    socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.settimeout(1.0)
            print(f"[MIC] Listening on {MULTICAST_GROUP}:{MULTICAST_PORT}")
            buf = np.array([], dtype=np.int16)
            while True:
                try:
                    data, _ = sock.recvfrom(8192)
                except socket.timeout:
                    continue
                packet = np.frombuffer(data, dtype=np.int16)
                buf    = np.concatenate([buf, packet])
                while len(buf) >= CHUNK_SIZE:
                    chunk, buf = buf[:CHUNK_SIZE].copy(), buf[CHUNK_SIZE:]
                    if _audio_q.full():
                        try: _audio_q.get_nowait()
                        except queue.Empty: pass
                    _audio_q.put_nowait(chunk)
        except Exception as exc:
            print(f"[MIC ERROR] {exc} — reconnecting in 2 s ...")
        finally:
            if sock:
                try: sock.close()
                except Exception: pass
        time.sleep(2.0)

threading.Thread(target=_mic_thread, daemon=True).start()

def _drain_queue():
    while not _audio_q.empty():
        try: _audio_q.get_nowait()
        except queue.Empty: break

def _vad_prob(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) / 32768.0

# ── VAD-based recording ───────────────────────────────────────────────────────
def record_utterance() -> np.ndarray:
    _drain_queue()
    silence_limit = max(1, round(SPEECH_TIMEOUT_S * SAMPLE_RATE / CHUNK_SIZE))

    speech_consec = 0
    while True:
        chunk = _audio_q.get()
        if _is_speaking.is_set():
            continue                 
        if _vad_prob(chunk) >= VAD_THRESHOLD:
            speech_consec += 1
        else:
            speech_consec = 0
        if speech_consec >= VAD_CONFIRM_CHUNKS:
            print("[VAD] Speech onset confirmed.")
            break

    frames         = [chunk]
    silence_chunks = 0
    deadline       = time.time() + MAX_RECORD_S

    while time.time() < deadline:
        try:
            chunk = _audio_q.get(timeout=0.15)
        except queue.Empty:
            continue
        frames.append(chunk)
        if _vad_prob(chunk) < VAD_THRESHOLD:
            silence_chunks += 1
            if silence_chunks >= silence_limit:
                print(f"[VAD] End of speech — {len(frames)} chunks ({len(frames)*CHUNK_SIZE/SAMPLE_RATE:.1f}s)")
                break
        else:
            silence_chunks = 0
    else:
        print(f"[VAD] Hard cap {MAX_RECORD_S}s reached.")

    return np.concatenate(frames).astype(np.float32) / 32768.0

# ── Voice command: model switching & management ───────────────────────────────
MODEL_ALIASES = {
    "whisper tiny":           "openai/whisper-tiny.en",
    "whisper tiny en":        "openai/whisper-tiny.en",
    "whisper tiny english":   "openai/whisper-tiny.en",
    "whisper base":           "openai/whisper-base.en",
    "whisper base en":        "openai/whisper-base.en",
    "whisper small":          "openai/whisper-small.en",
    "whisper small en":       "openai/whisper-small.en",
    "whisper medium":         "openai/whisper-medium.en",
    "whisper large":          "openai/whisper-large-v3",
    "moonshine tiny":         "UsefulSensors/moonshine-tiny",
    "moonshine base":         "UsefulSensors/moonshine-base",
    "granite":                "ibm-granite/granite-speech-4.1-2b-plus",
    "granite speech":         "ibm-granite/granite-speech-4.1-2b-plus",
    "wav2vec":                "facebook/wav2vec2-base-960h",
    "wav2vec2":               "facebook/wav2vec2-base-960h",
    "wav2vec large":          "facebook/wav2vec2-large-960h-lv60-self",
    "speecht5":               "microsoft/speecht5_asr",
    "speech t5":              "microsoft/speecht5_asr",
    "hubert":                 "facebook/hubert-large-ls960-ft",
    "vosk":                   "vosk-model-small-en-us-0.15",
    "vosk small":             "vosk-model-small-en-us-0.15",
}

def _resolve_model_id(text: str) -> str | None:
    t = text.lower().strip()
    for alias in sorted(MODEL_ALIASES, key=len, reverse=True):
        if alias in t:
            return MODEL_ALIASES[alias]
    m = re.search(r'[\w\-\.]+/[\w\-\.]+', t)
    if m:
        return m.group(0)
    return None

def handle_voice_commands(transcript: str, stt: STTModel) -> tuple:
    t = transcript.lower().strip()

    if "list model" in t or "what model" in t or "which model" in t:
        cached = list_cached_models()
        current = stt.model_id.split("/")[-1]
        if cached:
            names = ", ".join(m.split("/")[-1] for m in cached)
            speak(f"Current model: {current}. Cached on disk: {names}.")
        else:
            speak(f"Current model: {current}. No other models cached.")
        return stt, True

    if "delete model" in t or "remove model" in t:
        mid = stt.model_id
        speak(f"Deleting {mid.split('/')[-1]} from disk. Reloading.")
        stt.unload(delete_files=True)
        try:
            stt = STTModel(mid)
            speak("Model re-downloaded. Ready.")
        except Exception as e:
            print(f"[STT] Reload failed: {e}")
            speak("Re-download failed. Loading whisper tiny.")
            stt = STTModel("openai/whisper-tiny.en")
        return stt, True

    if "switch model" in t or "load model" in t or "try model" in t or "test model" in t:
        new_id = _resolve_model_id(t)
        if new_id is None:
            speak("I didn't catch the model name. Say: switch model whisper base, or switch model moonshine tiny.")
            return stt, True
        if new_id == stt.model_id:
            speak(f"{new_id.split('/')[-1]} is already loaded.")
            return stt, True

        short_name = new_id.split("/")[-1]
        speak(f"Switching to {short_name}. Downloading if needed, please wait.")

        delete_old = not KEEP_MODELS
        stt.unload(delete_files=delete_old)
        if delete_old:
            print(f"[CACHE] Old model files deleted (use --keep-models to retain).")

        try:
            stt = STTModel(new_id)
            speak(f"{short_name} loaded. Ready.")
        except Exception as e:
            print(f"[STT] Failed to load {new_id}: {e}")
            traceback.print_exc()
            speak("Model load failed. Falling back to whisper tiny.")
            stt = STTModel("openai/whisper-tiny.en")

        return stt, True

    return stt, False

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print(f"\n[TEST] ══ STT Benchmark ══")
    print(f"[TEST] Model : {DEFAULT_STT_MODEL}")
    print(f"[TEST] Keep models on disk: {KEEP_MODELS}")

    cached = list_cached_models()
    if cached:
        print(f"[TEST] Already cached: {', '.join(cached)}")

    stt = STTModel(DEFAULT_STT_MODEL)
    speak(
        f"Speech testing ready. "
        f"Model is {DEFAULT_STT_MODEL.split('/')[-1]}. "
        f"I am listening."
    )

    while True:
        try:
            print(f"\n[TEST] [{stt.model_id.split('/')[-1]}] Waiting for speech ...")
            audio_np = record_utterance()

            duration = len(audio_np) / SAMPLE_RATE
            print(f"[TEST] Recorded {duration:.2f}s — transcribing ...")

            t0   = time.time()
            text = stt.transcribe(audio_np)
            elapsed = time.time() - t0
            print(f"[STT Result] '{text}'  ({elapsed:.2f}s)")

            if not text:
                speak("I didn't catch that.")
                continue

            stt, handled = handle_voice_commands(text, stt)
            if handled:
                continue

            speak(f"You said: {text}")

        except KeyboardInterrupt:
            _cleanup()
        except Exception as e:
            print(f"[MAIN ERROR] {e}")
            traceback.print_exc()
            time.sleep(1.0)

if __name__ == "__main__":
    main()