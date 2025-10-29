# RunPod Deployment Guide for Testing-S2S

This guide shows exactly how to launch and test the realtime Speech-to-Speech server on RunPod using only the Web Terminal.

## Pod Configuration
- GPU: RTX 4090 or RTX 4500 (24GB VRAM)
- Container Image: `runpod/pytorch:2.3.0-py3.11-cuda12.1.1-devel-ubuntu22.04`
- Container Disk: 50 GB
- Volume Disk: 100–200 GB
- Volume Mount Path: `/workspace`
- Start Command: `bash`
- Expose Port: 8000 TCP

### Environment Variables (set in Pod UI)
- `MODEL_CACHE_DIR=/workspace/cache/models`
- `TORCH_HOME=/workspace/cache/torch`
- `PYTHONPATH=/workspace/Testing-S2S`
- `CUDA_VISIBLE_DEVICES=0`
- `REPLY_MODE=stream` (or `turn` for turn-based mode)

> **No HF_TOKEN Required**: The public HiFiGAN vocoder downloads directly from GitHub releases without authentication.

---

## One-Time Setup (copy-paste into RunPod Web Terminal)

```bash
# 1) System setup
apt update && apt upgrade -y
apt install -y git curl wget build-essential pkg-config \
    libssl-dev libsndfile1 libsndfile1-dev libsox-dev sox ffmpeg \
    portaudio19-dev libasound2-dev \
    python3-dev python3-pip python3-venv

# Create cache directories
mkdir -p /workspace/cache/models /workspace/cache/torch /workspace/data
cd /workspace

# 2) Clone repository
if [ ! -d "/workspace/Testing-S2S" ]; then
  git clone https://github.com/devasphn/Testing-S2S.git
fi
cd /workspace/Testing-S2S

# 3) Python environment
python3 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 4) Install PyTorch for CUDA 12.1 first
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.0 torchaudio==2.3.0

# 5) Install project dependencies
pip install -r requirements.txt
pip install -e .

# 6) Sanity check
python - << 'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY
```

---

## Start the Realtime Server

### Stream Mode (Continuous Response)
```bash
# Activate venv if not active
. /workspace/Testing-S2S/venv/bin/activate

# Start server in streaming mode
python src/server.py
```

### Turn-Based Mode (VAD Turn Detection)
```bash
# Activate venv if not active
. /workspace/Testing-S2S/venv/bin/activate

# Start server in turn-based mode
REPLY_MODE=turn python src/server.py
```

## API Endpoints

- Health check: `https://<your-pod-id>.direct.runpod.net:8000/health`
- Latency stats: `https://<your-pod-id>.direct.runpod.net:8000/api/stats`
- WebSocket stream: `wss://<your-pod-id>.direct.runpod.net:8000/ws/stream`
- Web UI: `https://<your-pod-id>.direct.runpod.net:8000/`

> **HTTPS Required**: Microphone access in browsers requires HTTPS. RunPod exposes HTTPS automatically on direct.runpod.net.

---

## HiFiGAN Vocoder Setup

**Automatic Download**: The public HiFiGAN vocoder (~55MB) downloads automatically on first run:
- No authentication required
- Downloads from GitHub releases
- MIT License (commercial use allowed)
- Cached in `/workspace/cache/models/hifigan_public/`

**First Run Output**:
```
[INFO] Downloading public HiFiGAN generator from GitHub releases...
[INFO] Downloaded 100.0%
[INFO] generator download complete: /workspace/cache/models/hifigan_public/generator.pth
[INFO] HiFiGAN generator loaded successfully
```

---

## Mode Comparison

### Stream Mode (Default)
- **REPLY_MODE=stream**
- Continuous response generation
- Lower perceived latency (~400ms)
- Real-time interaction
- Best for: Interactive applications, voice assistants

### Turn-Based Mode 
- **REPLY_MODE=turn**
- Waits for user to finish speaking
- Generates longer, more coherent responses
- VAD-based turn detection (30 frames of silence)
- Best for: Conversations, interviews, presentations

**Turn Mode Debug Output**:
```
[DEBUG] Turn ended, generating response from 15 chunks
[DEBUG] Response generated, latency: 450.2ms
[INFO] StreamingProcessor reset (mode: turn)
```

---

## Performance Monitoring

```bash
# Check latency stats
curl https://<your-pod-id>.direct.runpod.net:8000/api/stats

# Expected response:
{
  "latency_ms": {
    "mean": 412.5,
    "min": 285.1,
    "max": 650.3,
    "p95": 580.2,
    "mode": "turn",
    "turn_buffer_size": 0
  }
}
```

---

## Testing Commands

```bash
# Pull latest changes
cd /workspace/Testing-S2S
git pull

# Test turn-based mode
REPLY_MODE=turn python src/server.py

# Test stream mode (default)
python src/server.py

# Check model cache
ls -la /workspace/cache/models/hifigan_public/

# Clear cache if needed
rm -rf /workspace/cache/models/hifigan_public/
```

---

## Troubleshooting

### Vocoder Issues
- **Download failed**: Check internet connection, GitHub access
- **Load failed**: Check disk space in `/workspace/cache/`
- **Quality degraded**: Re-download by clearing cache directory

### Turn Mode Issues
- **No response**: Check VAD threshold, ensure sufficient silence
- **Response too fast**: Increase `silence_frames` in StreamingProcessor
- **Response too slow**: Decrease VAD threshold or silence detection

### General Issues
- **Microphone blocked**: Ensure HTTPS access via RunPod direct URL
- **CUDA OOM**: Reduce batch sizes or use smaller models
- **No audio output**: Verify 24kHz mono PCM format
- **Port closed**: Verify 8000/TCP is exposed in Pod settings

---

## Next Steps
- WebRTC integration for echo cancellation
- Custom voice adaptation training
- Multi-language support
- Production-grade deployment

See [`VOCODER_NOTICE.md`](VOCODER_NOTICE.md) for complete HiFiGAN license details.
