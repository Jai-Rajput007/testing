import asyncio
import os
import socket
import struct
from typing import AsyncGenerator
from piper.voice import PiperVoice
from core.interfaces import ITTSProvider

class PiperTTS(ITTSProvider):
    """
    On-device TTS using Piper.
    Extremely fast C++ engine with Python bindings.
    """
    def __init__(self, model_path: str = "models/en_US-lessac-medium.onnx"):
        # Piper expects the .json config file to be at model_path + ".json"
        self.model_path = model_path
        config_path = model_path + ".json"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model not found at {model_path}. Please download it first.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Piper config not found at {config_path}. Please place the matching .json file next to the model."
            )
            
        self.voice = PiperVoice.load(model_path, config_path)

    async def speak(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesizes speech and yields raw PCM bytes in chunks.
        Default sample rate is usually 22050Hz or 16000Hz depending on the model.
        """
        for audio_chunk in self.voice.synthesize(text):
            yield audio_chunk.audio_int16_bytes
            
    def get_sample_rate(self) -> int:
        return self.voice.config.sample_rate


class G1BuiltinTTS(ITTSProvider):
    """
    Standard G1 Robot TTS using the onboard TtsMaker service.
    Bypasses PulseAudio/PCM streaming for maximum stability.
    """
    def __init__(self, interface: str = "eth0", speaker_id: int = 1):
        self.interface = interface
        self.speaker_id = speaker_id
        self.is_builtin = True
        self._client = None
        
        # Initialize DDS immediately on startup
        try:
            # CIRCULAR IMPORT FIX: Import root before submodules
            import unitree_sdk2py
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            
            print(f"[TTS:G1] Initializing built-in TTS on {self.interface}...")
            ChannelFactoryInitialize(0, self.interface)
            self._client = AudioClient()
            self._client.Init()
            # Note: TtsMaker doesn't always need a long sleep after Init(), 
            # but we assume the factory handles the handshake.
            self._client.SetVolume(100)
            print("[TTS:G1] DDS initialised successfully.")
        except Exception as e:
            print(f"[TTS:G1 ERROR] Initialization failed: {e}")

    async def speak(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Triggers the robot's built-in TTS via TtsMaker.
        """
        if self._client:
            print(f"[TTS:G1] Speaking (TtsMaker): {text}")
            ret = self._client.TtsMaker(text, self.speaker_id)
            if ret != 0:
                 print(f"[TTS:G1 ERROR] TtsMaker returned code {ret}. Interface: {self.interface}")
        else:
            print(f"[TTS:G1 ERROR] AudioClient not available for: {text}")
        
        # Generator must yield at least once to be valid in main loops
        if False: yield b""

    def get_sample_rate(self) -> int:
        return 16000 # Default for G1 built-in

class G1DirectTTS(ITTSProvider):
    """
    High-performance TTS for G1.
    Uses Piper for voice generation (Jarvis), then sends PCM
    directly to Robot DDS PlayStream with 5.0x boost.
    Bypasses PulseAudio entirely.

    is_builtin=True so _play_tts does NOT also spawn aplay — DDS is the only audio path.
    """
    is_builtin = True  # Tell _play_tts to iterate speak() without spawning aplay

    def __init__(self, model_path: str = "models/en_US-lessac-medium.onnx", interface: str = "eth0"):
        self.interface = interface
        self.model_path = model_path
        self.piper = PiperTTS(model_path)
        self._client = None
        self.stream_id = f"jarvis_{int(os.getpid())}"

        try:
            import unitree_sdk2py
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

            print(f"[TTS:G1-DIRECT] Initializing DDS PlayStream on {self.interface}...")
            ChannelFactoryInitialize(0)
            self._client = AudioClient()
            self._client.Init()
            self._client.SetVolume(100)
            print("[TTS:G1-DIRECT] DDS initialised successfully.")
        except Exception as e:
            print(f"[TTS:G1-DIRECT ERROR] Initialization failed: {e}")

    def _play_stream(self, stream_id: str, data: bytes):
        """Send PCM bytes to the robot speaker, with list fallback for older SDKs."""
        try:
            self._client.PlayStream("jarvis_brain", stream_id, data)
        except TypeError:
            # Some SDK versions require a list instead of bytes
            self._client.PlayStream("jarvis_brain", stream_id, list(data))

    async def speak(self, text: str) -> AsyncGenerator[bytes, None]:
        import asyncio
        import numpy as np
        import time as _time

        if not self._client:
            print(f"[TTS:G1-DIRECT ERROR] AudioClient not available for: {text}")
            async for chunk in self.piper.speak(text):
                yield chunk
            return

        # Use a unique stream_id per speak() call so a fresh stream is opened
        # each time (avoids robot rejecting a reused ID after PlayStop)
        stream_id = f"jarvis_{int(_time.time() * 1000)}"
        print(f"[TTS:G1-DIRECT] Streaming to robot: '{text}' (stream={stream_id})")

        target_fs = 16000
        source_fs = self.piper.get_sample_rate()
        boost_factor = 5.0

        try:
            async for pcm_chunk in self.piper.speak(text):
                samples = np.frombuffer(pcm_chunk, dtype=np.int16)

                # Resample from Piper rate to 16000Hz
                if source_fs != target_fs:
                    num_samples_out = int(len(samples) * target_fs / source_fs)
                    samples = np.interp(
                        np.linspace(0, len(samples), num_samples_out, endpoint=False),
                        np.arange(len(samples)),
                        samples
                    ).astype(np.int16)

                # Apply volume boost
                boosted = np.clip(
                    samples.astype(np.float32) * boost_factor, -32768, 32767
                ).astype(np.int16)

                # Send chunk to robot speaker (with bytes→list fallback)
                try:
                    self._play_stream(stream_id, boosted.tobytes())
                except Exception as e:
                    print(f"[TTS:G1-DIRECT ERROR] PlayStream failed: {e}")

                # Pace delivery so the robot buffer doesn't get flooded
                chunk_duration = len(samples) / target_fs
                await asyncio.sleep(chunk_duration * 0.95)

                yield boosted.tobytes()
        finally:
            # Signal end-of-stream so the robot stops playback cleanly
            try:
                self._client.PlayStop("jarvis_brain")
            except Exception as e:
                print(f"[TTS:G1-DIRECT ERROR] PlayStop failed: {e}")

    def get_sample_rate(self) -> int:
        return self.piper.get_sample_rate()


class G1BridgeTTS(ITTSProvider):
    """
    TTS for G1 via robot_agent TCP bridge (port 7789).
    Piper generates PCM on AGX; robot_agent forwards to AudioClient.PlayStream over DDS.
    Protocol: [4-byte LE length][PCM bytes] ... [4-byte LE 0] (end-of-speech)
    is_builtin=True → _play_tts iterates speak() without spawning aplay.
    """
    is_builtin = True

    def __init__(self, model_path: str = "models/en_US-lessac-medium.onnx",
                 agent_host: str = "127.0.0.1", agent_port: int = 7789):
        self.piper = PiperTTS(model_path)
        self.agent_host = agent_host
        self.agent_port = agent_port
        self._target_fs = 16000
        self._boost = 5.0
        print(f"[TTS:BRIDGE] Ready → robot_agent {agent_host}:{agent_port}")

    async def speak(self, text: str) -> AsyncGenerator[bytes, None]:
        import numpy as np

        source_fs = self.piper.get_sample_rate()
        target_fs = self._target_fs

        def _connect_and_stream():
            with socket.create_connection((self.agent_host, self.agent_port), timeout=5) as s:
                return s

        try:
            sock = await asyncio.to_thread(
                socket.create_connection, (self.agent_host, self.agent_port), 5
            )
        except Exception as e:
            print(f"[TTS:BRIDGE ERROR] Cannot connect to robot_agent: {e}")
            async for chunk in self.piper.speak(text):
                yield chunk
            return

        print(f"[TTS:BRIDGE] Streaming: '{text}'")
        try:
            last_chunk_duration = 0.0
            async for pcm_chunk in self.piper.speak(text):
                samples = np.frombuffer(pcm_chunk, dtype=np.int16)

                if source_fs != target_fs:
                    n_out = int(len(samples) * target_fs / source_fs)
                    samples = np.interp(
                        np.linspace(0, len(samples), n_out, endpoint=False),
                        np.arange(len(samples)),
                        samples
                    ).astype(np.int16)

                boosted = np.clip(
                    samples.astype(np.float32) * self._boost, -32768, 32767
                ).astype(np.int16)

                data = boosted.tobytes()
                header = struct.pack('<I', len(data))
                await asyncio.to_thread(sock.sendall, header + data)

                chunk_duration = len(samples) / target_fs
                last_chunk_duration = chunk_duration
                await asyncio.sleep(chunk_duration * 0.95)
                yield data

            # Wait for the robot's audio buffer to drain before signalling end-of-speech.
            # Without this the sentinel arrives while the last word is still playing,
            # causing it to be cut off mid-syllable.
            await asyncio.sleep(last_chunk_duration * 0.05 + 0.30)

            # end-of-speech sentinel
            await asyncio.to_thread(sock.sendall, struct.pack('<I', 0))
        except Exception as e:
            print(f"[TTS:BRIDGE ERROR] {e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def get_sample_rate(self) -> int:
        return self._target_fs
