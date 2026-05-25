"""
Quick test for hey_jarvis.onnx
Reads from microphone and prints score when wake word is detected.
"""

import numpy as np
import pyaudio
from livekit.wakeword import WakeWordModel

MODEL_PATH  = '/home/surya/Downloads/daksh.onnx'
THRESHOLD   = 0.3
SAMPLE_RATE = 16000
CHUNK_SIZE  = 2048

# ── List available input devices ───────────────────────────────────────────
pa = pyaudio.PyAudio()
print("Available input devices:")
input_devices = []
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    if d['maxInputChannels'] > 0:
        print(f"  [{i}] {d['name']}")
        input_devices.append(i)

# Pick device — change this number if 0.000 score persists while speaking
DEVICE_INDEX = None  # None = system default; set to a number from the list above
print(f"\nUsing device: {'system default' if DEVICE_INDEX is None else DEVICE_INDEX}")
print("If score stays 0.000 while speaking, set DEVICE_INDEX to your mic number above.\n")
# ───────────────────────────────────────────────────────────────────────────

model = WakeWordModel(models=[MODEL_PATH])
print(f"Model loaded. Threshold: {THRESHOLD}")
print("Listening... say 'Hey Jarvis' (Ctrl+C to stop)\n")

buffer = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

stream = pa.open(
    rate=SAMPLE_RATE,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK_SIZE,
)

try:
    while True:
        raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        buffer = np.roll(buffer, -len(chunk))
        buffer[-len(chunk):] = chunk

        # RMS = volume level — if this is 0.000 while speaking, mic is wrong
        rms = float(np.sqrt(np.mean(buffer ** 2)))
        scores = model.predict(buffer)
        score = max(scores.values()) if scores else 0.0

        bar = '#' * int(score * 40)
        print(f"\r  vol: {rms:.4f}  score: {score:.3f}  [{bar:<40}]", end='', flush=True)

        if score >= THRESHOLD:
            print(f"\n  *** Hey Jarvis detected! (score={score:.3f}) ***\n")

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
