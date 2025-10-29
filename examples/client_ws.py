#!/usr/bin/env python3
"""
Python WebSocket client to test the server without a browser.
Sends microphone chunks if sounddevice is available, else sends a sine wave.
"""
import asyncio
import argparse
import numpy as np
import websockets

try:
    import sounddevice as sd
except Exception:
    sd = None

SR = 24000
CHUNK = 1920  # ~80ms

async def run(url: str, secs: int = 5):
    async with websockets.connect(url, ping_interval=None, max_size=2**24) as ws:
        print("Connected:", url)
        # Playback task
        async def recv_task():
            while True:
                data = await ws.recv()
                if isinstance(data, (bytes, bytearray)):
                    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)/32768.0
                    if sd:
                        sd.play(arr, SR, blocking=False)
        # Capture task
        async def send_task():
            if sd:
                with sd.InputStream(channels=1, samplerate=SR, dtype='float32', blocksize=CHUNK) as stream:
                    for _ in range(int(secs*SR/CHUNK)):
                        frames, _ = stream.read(CHUNK)
                        f = frames[:,0]
                        buf = (np.clip(f, -1, 1) * 32767.0).astype(np.int16).tobytes()
                        await ws.send(buf)
                        await asyncio.sleep(CHUNK/SR)
            else:
                t = np.arange(CHUNK)
                wave = 0.1*np.sin(2*np.pi*440*t/SR).astype(np.float32)
                for _ in range(int(secs*SR/CHUNK)):
                    buf = (wave*32767.0).astype(np.int16).tobytes()
                    await ws.send(buf)
                    await asyncio.sleep(CHUNK/SR)
        await asyncio.gather(recv_task(), send_task())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, required=True, help="wss://<host>/ws/stream")
    ap.add_argument("--secs", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(run(args.url, args.secs))
