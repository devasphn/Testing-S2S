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
- `HF_HOME=/workspace/cache/huggingface`
- `TORCH_HOME=/workspace/cache/torch`
- `PYTHONPATH=/workspace/Testing-S2S`
- `CUDA_VISIBLE_DEVICES=0`

Optional:
- `HF_TOKEN=<your_hf_token>` if you need private HF models. Not required for this repo to run the MVP.

> Note: If you intend to pull private models from Hugging Face, set `HF_TOKEN` here or run `huggingface-cli login` inside the pod. For public models and our current code, HF_TOKEN is not mandatory.

---

## One-Time Setup (copy-paste into RunPod Web Terminal)

```bash
# 1) System setup
apt update && apt upgrade -y
apt install -y git curl wget build-essential pkg-config \
    libssl-dev libsndfile1 libsndfile1-dev libsox-dev sox ffmpeg \
    portaudio19-dev libasound2-dev \
    python3-dev python3-pip python3-venv

# (Optional) Rust for some audio libs
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. ~/.cargo/env

mkdir -p /workspace/cache/huggingface /workspace/cache/torch /workspace/models /workspace/data
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

```bash
# Activate venv if not active
. /workspace/Testing-S2S/venv/bin/activate

# Start FastAPI/WS server on port 8000
python src/server.py
```

- Health check: `http://<your-pod-id>.direct.runpod.net:8000/health`
- Latency stats: `http://<your-pod-id>.direct.runpod.net:8000/api/stats`
- WebSocket stream: `wss://<your-pod-id>.direct.runpod.net:8000/ws/stream`

> Microphone access in browsers requires HTTPS. RunPod exposes HTTPS automatically on direct.runpod.net.

---

## Minimal Web UI (Test Page)

You can host a simple static HTML page from the pod using `python -m http.server`, but the simplest is to open this local file using your browser with the correct WS URL.

Create a file `webui/index.html` with the content from this repo (examples/minimal_websocket_client.html) and open it locally, or use a static server in the pod:

```bash
# From repo root
python -m http.server 8080 --bind 0.0.0.0
# Then open: http://<your-pod-id>.direct.runpod.net:8080/webui/index.html
```

Make sure the WS URL in the page points to:
`wss://<your-pod-id>.direct.runpod.net:8000/ws/stream`

---

## Do I need HF_TOKEN?

- For this MVP: Not required.
- If you pull private Hugging Face models or gated repos: set `HF_TOKEN` in the Pod env or run `huggingface-cli login` in the terminal.
- Cache paths are already set using `HF_HOME`.

```bash
# Optional: authenticate to Hugging Face
pip install --upgrade huggingface_hub
huggingface-cli login --token $HF_TOKEN
```

---

## Troubleshooting

- Microphone blocked: Ensure you access via HTTPS (RunPod direct URL is HTTPS). Some browsers require user interaction before mic can be used.
- CUDA OOM: Reduce batch sizes or model sizes; restart server.
- No audio output: Check that the browser sample rate is 24kHz mono, and that your UI sends 80ms PCM chunks.
- Port closed: Verify 8000/TCP is exposed in Pod settings.

---

## Next Steps
- Add a small HTML/JS Web UI (provided in `examples/minimal_websocket_client.html`).
- Integrate WebRTC for echo cancellation + automatic device selection.
- Add dataset loaders and training scripts for Indian/SEA language adaptation.
