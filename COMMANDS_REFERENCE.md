# Commands Reference - Testing-S2S

Complete command reference for setup, deployment, and operation of the Testing-S2S real-time speech-to-speech system.

---

## 🚀 **Production Deployment (RunPod)**

### **Initial Setup**
```bash
# System dependencies
apt update && apt upgrade -y
apt install -y git curl wget build-essential pkg-config \
    libssl-dev libsndfile1 libsndfile1-dev libsox-dev sox ffmpeg \
    portaudio19-dev libasound2-dev \
    python3-dev python3-pip python3-venv

# Create cache directories
mkdir -p /workspace/cache/models /workspace/cache/torch /workspace/data
cd /workspace

# Clone repository
git clone https://github.com/devasphn/Testing-S2S.git
cd /workspace/Testing-S2S

# Python environment
python3 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.0 torchaudio==2.3.0

# Install project dependencies
pip install -r requirements.txt
pip install -e .
```

### **Verification**
```bash
# Hardware check
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

## 🎧 **Server Operation**

### **Stream Mode (Continuous Response)**
```bash
# Activate environment
. /workspace/Testing-S2S/venv/bin/activate

# Start server (default mode)
python src/server.py

# OR explicitly set stream mode
REPLY_MODE=stream python src/server.py
```

### **Turn Mode (VAD-based Turns)**
```bash
# Activate environment
. /workspace/Testing-S2S/venv/bin/activate

# Start server in turn-based mode
REPLY_MODE=turn python src/server.py
```

### **Development Commands**
```bash
# Pull latest changes
cd /workspace/Testing-S2S
git pull

# Run tests
python -m pytest tests/

# Smoke test
python scripts/smoke_test.py

# Check model cache
ls -la /workspace/cache/models/hifigan_public/

# Clear cache if needed
rm -rf /workspace/cache/models/hifigan_public/
```

---

## 🌍 **API Testing**

### **Health Check**
```bash
# Replace <pod-id> with your RunPod ID
curl https://<pod-id>.direct.runpod.net:8000/health

# Expected response:
# {"status":"ok","device":"cuda"}
```

### **Performance Stats**
```bash
# Get latency statistics
curl https://<pod-id>.direct.runpod.net:8000/api/stats

# Expected response:
# {
#   "latency_ms": {
#     "mean": 250.5,
#     "min": 145.2,
#     "max": 450.8,
#     "p95": 380.1,
#     "mode": "turn",
#     "turn_buffer_size": 0
#   }
# }
```

### **Web Interface Access**
```bash
# Built-in web UI
https://<pod-id>.direct.runpod.net:8000/web

# WebSocket endpoint
wss://<pod-id>.direct.runpod.net:8000/ws/stream
```

---

## 🔧 **Development Setup (Local)**

### **Local Development**
```bash
# Clone repository
git clone https://github.com/devasphn/Testing-S2S.git
cd Testing-S2S

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .

# Start development server
REPLY_MODE=turn python src/server.py
```

### **Local Testing**
```bash
# Health check
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/api/stats

# Web UI
open http://localhost:8000/web
```

---

## 🐛 **Troubleshooting Commands**

### **Common Issues**

#### **CUDA Out of Memory**
```bash
# Check GPU memory
nvidia-smi

# Restart server with smaller batch
CUDA_VISIBLE_DEVICES=0 python src/server.py
```

#### **Audio Issues**
```bash
# Test audio devices
python - << 'PY'
import sounddevice as sd
print("Audio devices:", sd.query_devices())
PY

# Check sample rates
python - << 'PY'
import librosa
print("librosa version:", librosa.__version__)
PY
```

#### **Model Download Issues**
```bash
# Clear PyTorch Hub cache
rm -rf ~/.cache/torch/hub/

# Clear model cache
rm -rf /workspace/cache/models/

# Force re-download
REPLY_MODE=turn python src/server.py
```

#### **Dependency Issues**
```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt

# Check versions
pip list | grep torch
pip list | grep fastapi
pip list | grep librosa
```

---

## 📊 **Monitoring Commands**

### **Real-time Monitoring**
```bash
# Watch server logs
tail -f server.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor network
ss -tulnp | grep :8000
```

### **Performance Testing**
```bash
# Load test (if you have wrk)
wrk -t4 -c10 -d30s http://localhost:8000/health

# WebSocket test
python examples/client_ws.py
```

---

## 🔄 **Environment Variables**

### **Core Configuration**
```bash
# Mode selection
export REPLY_MODE=turn          # or 'stream'
export MODEL_CACHE_DIR=/workspace/cache/models
export TORCH_HOME=/workspace/cache/torch

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/Testing-S2S

# Optional optimizations
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### **Development Settings**
```bash
# Debug mode
export FASTAPI_ENV=development
export LOG_LEVEL=DEBUG

# Performance profiling
export PYTORCH_PROFILER_ENABLED=1
```

---

## 📅 **Command History (Session)**

These are the exact commands used during our development session:

```bash
# Repository updates (performed via GitHub API)
git pull
REPLY_MODE=turn python src/server.py

# Performance monitoring
curl -s https://c50prx7arirkfn-8000.proxy.runpod.net/api/stats

# Model testing
REPLY_MODE=turn python src/server.py
```

### **Latest Session – Nov 16, 2025 (RunPod GPU Pod)**

```bash
# Sync repo with upstream changes
git pull origin main

# Launch inference server (streaming mode, 24 kHz transport)
uvicorn src.server:app --host 0.0.0.0 --port 8000

# Optional gain tweak when speaker embedding sounds quiet
export SPEAKER_GAIN=1.4
uvicorn src.server:app --host 0.0.0.0 --port 8000

# Collect stats or warmup (run from inside venv)
curl -s http://localhost:8000/api/stats
python scripts/smoke_test.py --input wav_files/sample_user.wav --output outputs/sample_ai.wav
```

---

## ⚙️ **Configuration Files**

### **Environment Template** (`.env.example`)
```bash
# Copy and customize
cp .env.example .env
nano .env
```

### **Requirements** (`requirements.txt`)
```bash
# Install all dependencies
pip install -r requirements.txt
```

---

**Last Updated**: October 30, 2025  
**Version**: 0.1.4  
**Status**: Production Ready with Minor Tuning Needed
