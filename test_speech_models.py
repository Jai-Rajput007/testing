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
MAX_RECORD_S       = 15.0          # hard cap
TTS_ECHO_DRAIN_S   = 0.15          # extra drain window after _is_speaking clears
TRANSCRIBE_TIMEOUT = 40.0          # inference wall-clock limit

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
    """
    Return the path to the HuggingFace cache folder for a model.

    HF convention: 'org/model-name'  →  models/models--org--model-name
    Models without an org prefix     →  models/models--model-name
    """
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
    """Delete cached model files from disk. Returns True if anything was deleted."""
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
    """Return list of model_ids found in MODELS_DIR cache."""
    cached = []
    if not os.path.isdir(MODELS_DIR):
        return cached
    for name in os.listdir(MODELS_DIR):
        if name.startswith("models--"):
            # Reverse the folder name → model_id
            # "models--openai--whisper-tiny.en" → "openai/whisper-tiny.en"
            parts = name[len("models--"):].split("--", 1)
            if len(parts) == 2:
                cached.append(f"{parts[0]}/{parts[1]}")
            else:
                cached.append(parts[0])
    return cached

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

# ── Cleanup on exit ───────────────────────────────────────────────────────────
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
_is_speaking   = threading.Event()   # set while TTS is playing

def speak(text: str):
    """Synthesize text with Piper and stream to robot speaker."""
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
            time.sleep(0.4)   # let speaker echo decay before mic opens
            _is_speaking.clear()

# ── STT — model family detection & per-family pipelines ──────────────────────
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoModelForCausalLM,
)

def _detect_family(model_id: str, arch: str) -> str:
    """
    Map a model to one of our inference pipelines.

    Priority: explicit model-ID substring check first (most reliable),
    then fall back to architecture class name from config.
    """
    mid = model_id.lower()
    arc = arch.lower()

    if "granite" in mid:              return "granite"
    if "moonshine" in mid:            return "moonshine"
    if "speecht5" in mid:             return "speecht5"
    if "whisper" in mid:              return "whisper"
    if "wav2vec2" in mid:             return "ctc"
    if "hubert" in mid:               return "ctc"
    if "wav2vec" in mid:              return "ctc"

    # Architecture fallbacks
    if "ctc" in arc:                  return "ctc"
    if "wav2vec" in arc:              return "ctc"
    if "speechseq2seq" in arc:        return "whisper"
    if "whisper" in arc:              return "whisper"
    if "moonshine" in arc:            return "moonshine"

    # Anything else with generate() → generic seq2seq
    return "seq2seq"


class STTModel:
    """
    Unified wrapper for any HuggingFace speech-to-text model.

    ── Download behaviour ────────────────────────────────────────────────────
    All models are saved to ./models/ using HuggingFace's standard
    cache layout:  models/models--<org>--<model-name>/snapshots/<hash>/

    ── Family-specific notes ─────────────────────────────────────────────────
    granite   GraniteSpeechPlus is a multimodal LLM. The processor MUST
              receive audio + a text prompt via apply_chat_template.
              Passing audio alone → 'Invalid text provided' TypeError.

    moonshine Uses AutoModelForCausalLM + processor.decode() (singular).
              Does NOT use batch_decode.

    whisper   AutoModelForSpeechSeq2Seq + processor.batch_decode.

    ctc       AutoModelForCTC → argmax(logits) → processor.batch_decode.

    speecht5  Uses transformers pipeline("automatic-speech-recognition")
              which handles its unique speaker-embedding requirement.

    seq2seq   Generic fallback for any other model with generate().
    """

    def __init__(self, model_id: str):
        self.model_id  = model_id
        self.model     = None
        self.processor = None
        self.pipeline  = None   # used for speecht5 and pipeline-only models
        self.family    = None
        self._load(model_id)

    # ── Download progress callback ────────────────────────────────────────────
    @staticmethod
    def _print_download_progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(block_num * block_size * 100 / total_size, 100)
            mb  = block_num * block_size / (1024 * 1024)
            print(f"\r[DL] {pct:.1f}%  {mb:.1f} MB", end="", flush=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    def _load(self, model_id: str):
        cache = MODELS_DIR
        print(f"\n[STT] ── Loading '{model_id}' ──")
        print(f"[STT] Cache folder: {model_cache_folder(model_id)}")

        # --- Processor + Config (needed for family detection) -----------------
        print(f"[STT] Downloading processor/config ...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=cache,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[STT] AutoProcessor failed ({e}), trying pipeline fallback ...")
            self._load_as_pipeline(model_id, cache)
            return

        config = AutoConfig.from_pretrained(
            model_id, cache_dir=cache, trust_remote_code=True
        )
        archs      = getattr(config, "architectures", [""]) or [""]
        arch       = archs[0] if archs else ""
        self.family = _detect_family(model_id, arch)
        print(f"[STT] Architecture: {arch}")
        print(f"[STT] Pipeline family: {self.family}")

        # --- Model weights ----------------------------------------------------
        print(f"[STT] Downloading model weights (this may take a while) ...")

        if self.family == "speecht5":
            self._load_as_pipeline(model_id, cache)
            return

        if self.family == "ctc":
            self.model = AutoModelForCTC.from_pretrained(
                model_id, cache_dir=cache, trust_remote_code=True
            )
        elif self.family == "granite":
            # GraniteSpeechPlus is its own model class — not registered under
            # AutoModelForCausalLM. Must use AutoModel with trust_remote_code
            # so transformers resolves it to GraniteSpeechPlusForConditionalGeneration.
            from transformers import AutoModel
            try:
                self.model = AutoModel.from_pretrained(
                    model_id, cache_dir=cache, trust_remote_code=True
                )
            except Exception as e1:
                print(f"[STT] AutoModel failed ({e1}), trying AutoModelForSpeechSeq2Seq ...")
                try:
                    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_id, cache_dir=cache, trust_remote_code=True
                    )
                except Exception as e2:
                    print(f"[STT] Seq2Seq failed ({e2}), using pipeline fallback ...")
                    self._load_as_pipeline(model_id, cache)
                    return
        elif self.family == "moonshine":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=cache, trust_remote_code=True
            )
        elif self.family == "whisper":
            try:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, cache_dir=cache, trust_remote_code=True
                )
            except Exception:
                # Some Whisper variants only register as CausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, cache_dir=cache, trust_remote_code=True
                )
        else:
            # Generic seq2seq — try Seq2Seq first, fall back to CausalLM
            try:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, cache_dir=cache, trust_remote_code=True
                )
            except Exception as e1:
                print(f"[STT] Seq2Seq load failed ({e1}), trying CausalLM ...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id, cache_dir=cache, trust_remote_code=True
                    )
                except Exception as e2:
                    print(f"[STT] CausalLM load failed ({e2}), using pipeline ...")
                    self._load_as_pipeline(model_id, cache)
                    return

        self.model.eval()
        self.model.to("cpu")
        size_mb = model_disk_size_mb(model_id)
        print(f"[STT] '{model_id}' ready. Disk: {size_mb:.0f} MB")

    def _load_as_pipeline(self, model_id: str, cache: str):
        """
        Last-resort loader using transformers.pipeline().
        Handles models like SpeechT5 or anything with a non-standard processor.
        """
        from transformers import pipeline
        print(f"[STT] Loading via transformers.pipeline ...")
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            model_kwargs={"cache_dir": cache},
            device="cpu",
            trust_remote_code=True,
        )
        self.family = "pipeline"
        print(f"[STT] '{model_id}' loaded via pipeline.")

    # ── Unload (free RAM) ─────────────────────────────────────────────────────
    def unload(self, delete_files: bool = False):
        mid = self.model_id
        if self.model is not None:
            del self.model;     self.model     = None
        if self.processor is not None:
            del self.processor; self.processor = None
        if self.pipeline is not None:
            del self.pipeline;  self.pipeline  = None
        gc.collect()
        print(f"[STT] '{mid}' unloaded from RAM.")
        if delete_files:
            delete_model_files(mid)

    # ── Transcribe (public) ───────────────────────────────────────────────────
    def transcribe(self, audio_np: np.ndarray) -> str:
        """
        audio_np : float32 waveform normalised to [-1, 1], 16 kHz mono.
        Returns  : transcribed string (empty string on any failure).
        """
        try:
            if self.family == "granite":
                return self._infer_granite(audio_np)
            elif self.family == "moonshine":
                return self._infer_moonshine(audio_np)
            elif self.family == "ctc":
                return self._infer_ctc(audio_np)
            elif self.family == "pipeline":
                return self._infer_pipeline(audio_np)
            else:
                # whisper + seq2seq share the same path
                return self._infer_seq2seq(audio_np)
        except Exception as e:
            print(f"[STT ERROR] transcribe() raised: {e}")
            traceback.print_exc()
            return ""

    # ── Granite Speech ────────────────────────────────────────────────────────
    def _infer_granite(self, audio_np: np.ndarray) -> str:
        """
        GraniteSpeechPlusForConditionalGeneration is a multimodal LLM.
        The processor requires BOTH audio AND a text prompt via apply_chat_template.
        Passing audio alone raises: 'Invalid text provided' TypeError.

        Output decoding:
          - Decoder-only style: generated_ids includes prompt tokens → strip them
          - Encoder-decoder style: generated_ids is already just the new tokens
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_np},
                    {"type": "text",  "text":  "Transcribe the speech in this audio clip."},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        duration       = len(audio_np) / SAMPLE_RATE
        max_new_tokens = max(int(duration * 8), 32)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        # Strip prompt tokens if decoder-only style output
        prompt_len = inputs["input_ids"].shape[-1]
        if generated_ids.shape[-1] > prompt_len:
            decode_ids = generated_ids[:, prompt_len:]
        else:
            decode_ids = generated_ids

        # Try processor.batch_decode; fall back to tokenizer directly
        try:
            text = self.processor.batch_decode(decode_ids, skip_special_tokens=True)[0]
        except Exception:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            text = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0]

        return text.strip()

    # ── Moonshine ─────────────────────────────────────────────────────────────
    def _infer_moonshine(self, audio_np: np.ndarray) -> str:
        inputs = self.processor(
            audio_np, return_tensors="pt", sampling_rate=SAMPLE_RATE
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
        duration       = len(audio_np) / SAMPLE_RATE
        max_new_tokens = max(int(duration * 5), 16)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
        # processor.decode (not batch_decode) is the documented Moonshine path
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    # ── Whisper / generic Seq2Seq ─────────────────────────────────────────────
    def _infer_seq2seq(self, audio_np: np.ndarray) -> str:
        inputs = self.processor(
            audio_np, return_tensors="pt", sampling_rate=SAMPLE_RATE
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
        duration       = len(audio_np) / SAMPLE_RATE
        max_new_tokens = max(int(duration * 6), 30)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # ── CTC / Wav2Vec2 ────────────────────────────────────────────────────────
    def _infer_ctc(self, audio_np: np.ndarray) -> str:
        inputs = self.processor(
            audio_np, return_tensors="pt", sampling_rate=SAMPLE_RATE
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            logits   = self.model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0].strip()

    # ── pipeline() fallback ───────────────────────────────────────────────────
    def _infer_pipeline(self, audio_np: np.ndarray) -> str:
        result = self.pipeline(
            {"array": audio_np, "sampling_rate": SAMPLE_RATE}
        )
        return result.get("text", "").strip()


# ── Transcription with wall-clock timeout ─────────────────────────────────────
def transcribe_with_timeout(stt: STTModel, audio_np: np.ndarray) -> str:
    result = [""]
    exc    = [None]

    def _run():
        try:
            result[0] = stt.transcribe(audio_np)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=TRANSCRIBE_TIMEOUT)

    if t.is_alive():
        print(f"[STT] Inference timed out after {TRANSCRIBE_TIMEOUT}s — skipping.")
        return ""
    if exc[0]:
        print(f"[STT ERROR] {exc[0]}")
        return ""
    return result[0]


# ── Mic multicast receiver ────────────────────────────────────────────────────
_audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=300)

def _mic_thread():
    """Receives UDP multicast from robot mic. Auto-reconnects on any error."""
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
    """
    Block until speech is detected, record until silence or MAX_RECORD_S.
    Returns float32 waveform normalised to [-1, 1].

    TTS-echo guard (mirrors demo_gestures.py drain_queue() pattern):
      - Phase 1 busy-waits while _is_speaking is set — no chunks counted
      - Once TTS clears, drains the queue + sleeps TTS_ECHO_DRAIN_S to let
        any remaining speaker echo flush through before counting begins
      - Requires VAD_CONFIRM_CHUNKS consecutive above-threshold chunks to
        confirm real speech (vs a transient pop or residual echo)
      - Phase 2 records until SPEECH_TIMEOUT_S silence or MAX_RECORD_S cap
    """
    silence_limit = max(1, round(SPEECH_TIMEOUT_S * SAMPLE_RATE / CHUNK_SIZE))

    # Phase 1 — wait for TTS to finish, then for confirmed speech onset
    speech_consec = 0
    was_speaking  = False
    while True:
        if _is_speaking.is_set():
            # Robot is talking — discard everything and reset counter
            try:
                _audio_q.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
            speech_consec = 0
            was_speaking  = True
            continue

        if was_speaking:
            # TTS just finished — drain residual echo then reset
            _drain_queue()
            time.sleep(TTS_ECHO_DRAIN_S)
            _drain_queue()
            was_speaking  = False
            speech_consec = 0
            print("[VAD] TTS echo cleared, now listening ...")
            continue

        chunk = _audio_q.get()
        if _vad_prob(chunk) >= VAD_THRESHOLD:
            speech_consec += 1
        else:
            speech_consec = 0
        if speech_consec >= VAD_CONFIRM_CHUNKS:
            print("[VAD] Speech onset confirmed.")
            break

    # Phase 2 — record until silence / hard cap
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
    # spoken name            → HuggingFace model ID
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
}

def _resolve_model_id(text: str) -> str | None:
    """Try to resolve spoken text to a known model ID."""
    t = text.lower().strip()
    # Check aliases longest-first so "whisper base en" beats "whisper base"
    for alias in sorted(MODEL_ALIASES, key=len, reverse=True):
        if alias in t:
            return MODEL_ALIASES[alias]
    # Check if there's a raw HF-style ID in the text (contains /)
    m = re.search(r'[\w\-\.]+/[\w\-\.]+', t)
    if m:
        return m.group(0)
    return None


def handle_voice_commands(transcript: str, stt: STTModel) -> tuple:
    """
    Parse transcript for management commands.
    Returns (new_stt, handled: bool).
    """
    t = transcript.lower().strip()

    # ── list models ──────────────────────────────────────────────────────────
    if "list model" in t or "what model" in t or "which model" in t:
        cached = list_cached_models()
        current = stt.model_id.split("/")[-1]
        if cached:
            names = ", ".join(m.split("/")[-1] for m in cached)
            speak(f"Current model: {current}. Cached on disk: {names}.")
        else:
            speak(f"Current model: {current}. No other models cached.")
        return stt, True

    # ── delete model (current) ────────────────────────────────────────────────
    if "delete model" in t or "remove model" in t:
        mid = stt.model_id
        speak(f"Deleting {mid.split('/')[-1]} from disk. Reloading.")
        stt.unload(delete_files=True)
        try:
            stt = STTModel(mid)   # re-download from scratch
            speak("Model re-downloaded. Ready.")
        except Exception as e:
            print(f"[STT] Reload failed: {e}")
            speak("Re-download failed. Loading whisper tiny.")
            stt = STTModel("openai/whisper-tiny.en")
        return stt, True

    # ── switch model ──────────────────────────────────────────────────────────
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
            text = transcribe_with_timeout(stt, audio_np)
            elapsed = time.time() - t0
            print(f"[STT Result] '{text}'  ({elapsed:.2f}s)")

            if not text:
                speak("I didn't catch that.")
                continue

            # Check for management commands first
            stt, handled = handle_voice_commands(text, stt)
            if handled:
                continue

            # Normal echo response
            speak(f"You said: {text}")

        except KeyboardInterrupt:
            _cleanup()
        except Exception as e:
            print(f"[MAIN ERROR] {e}")
            traceback.print_exc()
            time.sleep(1.0)


if __name__ == "__main__":
    main()