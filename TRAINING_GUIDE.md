# Training Guide: Build Your Own Luna AI

**Goal**: Train a production-ready speech-to-speech AI model from scratch, independent of any external APIs.

**Budget**: $300-1,700 depending on dataset size and training iterations

**Timeline**: 2-8 weeks depending on GPU availability and dataset preparation

---

## Table of Contents

1. [Quick Start (Get Training in 30 minutes)](#quick-start)
2. [System Requirements](#system-requirements)
3. [Data Preparation](#data-preparation)
4. [Phase 1: Train Speech Tokenizer](#phase-1-train-speech-tokenizer)
5. [Phase 2: Train Hybrid S2S Model](#phase-2-train-hybrid-s2s-model)
6. [Phase 3: Add Emotional Control](#phase-3-add-emotional-control)
7. [Deployment to Production](#deployment-to-production)
8. [Cost Breakdown](#cost-breakdown)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### On RunPod (Recommended)

```bash
# 1. Launch RunPod A100 80GB pod (Secure Cloud or Community)
# 2. Connect via SSH or Jupyter

# 3. Clone repository
cd /workspace
git clone https://github.com/devasphn/Testing-S2S.git
cd Testing-S2S

# 4. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA 12.1
pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
pip install -r requirements-training.txt

# 5. Download LibriSpeech dataset (100 hours)
mkdir -p /workspace/data
cd /workspace/data
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Optional: Download larger datasets for better quality
# wget http://www.openslr.org/resources/12/train-clean-360.tar.gz
# wget http://www.openslr.org/resources/12/train-other-500.tar.gz

# 6. Update config with your data path
cd /workspace/Testing-S2S
nano training/configs/tokenizer_config.yaml
# Change data_dir to: /workspace/data

# 7. Start training!
python training/train_tokenizer.py --config training/configs/tokenizer_config.yaml

# 8. Monitor training
# - Watch console output
# - Check WandB dashboard (if enabled)
# - Monitor GPU: watch -n 1 nvidia-smi
```

**Expected Output**:
```
[DEVICE] Using cuda
[DATASET] Loaded 28539 files from train-clean-100
[DATA] Train: 27112, Val: 1427
[MODEL] Parameters: 42,853,376

Epoch 1/100
============================================================
Epoch 1: 100%|███████| 1695/1695 [05:12<00:00, 5.42it/s, loss=0.3245, mel_l1=0.2891, commit=0.0354]
Validation: 100%|███████| 90/90 [00:18<00:00, 4.89it/s, val_loss=0.2987]

Epoch 1 Summary:
  Train Loss: 0.3245
  Val Loss:   0.2987
  LR:         1.00e-04
  ✓ Saved best model (val_loss: 0.2987)
```

---

## System Requirements

### Minimum (Training Tokenizer Only)
- **GPU**: NVIDIA A40 (48GB VRAM) or better
- **RAM**: 32GB system RAM
- **Storage**: 200GB SSD
- **Network**: High-speed for dataset downloads (100GB+ datasets)

### Recommended (Full Training Pipeline)
- **GPU**: NVIDIA A100 80GB
- **RAM**: 64GB system RAM
- **Storage**: 500GB NVMe SSD
- **Network**: 1 Gbps+

### Software
- **OS**: Ubuntu 22.04 LTS (or RunPod Docker image)
- **Python**: 3.10+
- **CUDA**: 12.1+
- **PyTorch**: 2.3.0+

---

## Data Preparation

### Option 1: LibriSpeech (Free, English)

**Datasets**:
- `train-clean-100`: 100 hours, clean speech ($10 training cost)
- `train-clean-360`: 360 hours, clean speech ($40 training cost)
- `train-other-500`: 500 hours, diverse speakers ($60 training cost)

**Download**:
```bash
cd /workspace/data

# Small (100h) - Good for testing
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Medium (360h) - Good quality
wget http://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xzf train-clean-360.tar.gz

# Large (500h) - Best quality
wget http://www.openslr.org/resources/12/train-other-500.tar.gz
tar -xzf train-other-500.tar.gz
```

### Option 2: Common Voice (Free, Multilingual)

**Languages**: 100+ including Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam

**Download**:
1. Go to https://commonvoice.mozilla.org/datasets
2. Select languages (e.g., Hindi, English)
3. Download `.tar.gz` files
4. Extract to `/workspace/data/CommonVoice/`

**Note**: Common Voice requires preprocessing. LibriSpeech is ready to use.

### Option 3: Synthetic Data (Paid, Custom)

For conversational data (needed for S2S training):

```bash
# Generate synthetic conversations using GPT-4 + TTS
python scripts/generate_synthetic_conversations.py \
    --num_samples 10000 \
    --output_dir /workspace/data/synthetic \
    --openai_api_key YOUR_KEY

# Cost: ~$100-300 for 10K conversation pairs
```

---

## Phase 1: Train Speech Tokenizer

### What It Does

The speech tokenizer is like a "speech compression" model that converts audio into discrete tokens (similar to how JPEG compresses images).

**Architecture**:
```
Audio (24kHz) → Mel Spectrogram → CNN Encoder → RVQ (8 quantizers) → CNN Decoder → Mel → HiFiGAN Vocoder → Audio
```

### Training Steps

**1. Prepare Config**:

```yaml
# training/configs/tokenizer_config.yaml
model:
  sample_rate: 24000  # Match Luna AI
  codebook_size: 1024
  hidden_dim: 512
  num_quantizers: 8

data:
  data_dir: "/workspace/data"
  train_split: "train-clean-100"  # or train-clean-360
  
training:
  epochs: 100  # Increase to 200 for better quality
  batch_size: 16  # Adjust for your GPU (8 for A40, 16-32 for A100)
  learning_rate: 1e-4
```

**2. Start Training**:

```bash
cd /workspace/Testing-S2S
source venv/bin/activate

# Basic training
python training/train_tokenizer.py

# With custom config
python training/train_tokenizer.py --config my_config.yaml

# Resume from checkpoint
python training/train_tokenizer.py --resume checkpoints/tokenizer/tokenizer_epoch_50.pt
```

**3. Monitor Progress**:

```bash
# Terminal 1: Training logs
python training/train_tokenizer.py

# Terminal 2: GPU monitoring
watch -n 1 nvidia-smi

# Terminal 3: Tensorboard (if not using wandb)
tensorboard --logdir checkpoints/tokenizer/logs --port 6006
```

**4. Evaluate Quality**:

```bash
# Test tokenizer on sample audio
python scripts/test_tokenizer.py \
    --checkpoint checkpoints/tokenizer/tokenizer_best.pt \
    --input test_audio.wav \
    --output reconstructed.wav

# Listen to reconstructed audio
# Good quality: Clear speech, minimal artifacts
# Bad quality: Muffled, robotic, missing details
```

### Expected Results

| Dataset | Training Time | Cost | Quality |
|---------|--------------|------|----------|
| train-clean-100 (100h) | 8-12 hours | $10-15 | Good for testing |
| train-clean-360 (360h) | 30-40 hours | $36-48 | Production-ready |
| train-clean-100+360+500 (960h) | 80-120 hours | $95-143 | Excellent |

**Metrics to Watch**:
- **Mel L1 Loss**: Should drop below 0.15 (lower = better)
- **Commitment Loss**: Should stabilize around 0.01-0.05
- **Validation Loss**: Should decrease steadily without overfitting

---

## Phase 2: Train Hybrid S2S Model

*Coming in next commit: Complete S2S training pipeline*

**Preview**:
- Uses trained tokenizer from Phase 1
- Trained on conversational data
- Architecture similar to Moshi/GLM-4-Voice
- Estimated cost: $300-500

---

## Phase 3: Add Emotional Control

*Coming in next commit: Emotion fine-tuning pipeline*

**Preview**:
- Fine-tune on IEMOCAP emotional speech dataset
- Add emotion embedding layer
- Support for laughter, sighs, whispers
- Estimated cost: $100-200

---

## Deployment to Production

### Update Your Server

Once training is complete, integrate trained models into your existing FastAPI server:

```python
# src/server.py
from src.models.speech_tokenizer_trainable import TrainableSpeechTokenizer
from src.models.hybrid_s2s import HybridS2SModel

# Load trained models
tokenizer = TrainableSpeechTokenizer(
    checkpoint_path="/workspace/checkpoints/tokenizer/tokenizer_best.pt"
).to(device)

s2s_model = HybridS2SModel(
    checkpoint_path="/workspace/checkpoints/s2s/s2s_best.pt"
).to(device)

# Use in inference
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    # ... your existing streaming logic
    
    # Tokenize user audio
    user_tokens = tokenizer.tokenize(user_audio)
    
    # Generate AI response tokens
    ai_tokens = s2s_model.generate_streaming(user_tokens)
    
    # Detokenize to audio
    ai_audio = tokenizer.detokenize(ai_tokens)
    
    # Stream back to user
    await websocket.send_bytes(ai_audio)
```

---

## Cost Breakdown

### Budget Option (~$300)
- **Tokenizer**: LibriSpeech 100h, 10 hours training = $12
- **S2S Model**: Synthetic 5K pairs, 100 hours training = $119
- **Emotion Fine-tuning**: IEMOCAP, 50 hours = $60
- **Experimentation**: 100 hours = $119
- **Total**: ~$310

### Production Option (~$1,500)
- **Tokenizer**: LibriSpeech 960h, 120 hours training = $143
- **S2S Model**: Synthetic 20K pairs, 400 hours training = $476
- **Emotion Fine-tuning**: Multiple datasets, 150 hours = $179
- **Experimentation**: 500 hours = $595
- **Total**: ~$1,393

### RunPod Pricing (as of Nov 2025)
- **A100 80GB (Secure Cloud)**: $1.19/hr
- **A100 80GB (Community)**: $0.89/hr (40% cheaper!)
- **A40 48GB (Community)**: $0.49/hr (good for inference)

**Cost Optimization Tips**:
1. Use **Community Cloud** (40% cheaper, slightly less reliable)
2. Use **Spot Instances** for training (70% cheaper, can be interrupted)
3. Train tokenizer on smaller dataset first, validate, then scale up
4. Use mixed precision training (fp16) to fit larger batches

---

## Troubleshooting

### "Checkpoint not found"

**Problem**: Model starts with random weights

**Solution**:
```bash
# Check if checkpoint exists
ls -la checkpoints/tokenizer/

# If missing, train first
python training/train_tokenizer.py
```

### "CUDA out of memory"

**Problem**: GPU memory exhausted

**Solutions**:
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 8  # Instead of 16
   ```

2. Use gradient accumulation:
   ```yaml
   training:
     batch_size: 8
     gradient_accumulation_steps: 2  # Effective batch size = 16
   ```

3. Use mixed precision:
   ```python
   # In training script
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       outputs = model(audio)
   ```

### "Training loss not decreasing"

**Problem**: Model not learning

**Solutions**:
1. Check learning rate (try 1e-5 to 1e-3 range)
2. Verify data preprocessing is correct
3. Reduce model size temporarily for debugging
4. Check for NaN gradients: `torch.isnan(loss).any()`

### "Audio quality is poor"

**Problem**: Reconstructed audio is muffled or robotic

**Solutions**:
1. Train longer (more epochs)
2. Increase model capacity (hidden_dim: 768 instead of 512)
3. Use larger dataset (960h instead of 100h)
4. Adjust commitment_weight in config (try 0.1 to 0.5)

---

## Next Steps

**Immediate (This Week)**:
1. ✓ Set up RunPod environment
2. ✓ Download LibriSpeech dataset
3. ✓ Start tokenizer training
4. Monitor first 10 epochs, validate quality

**Short-term (Next 2 Weeks)**:
1. Complete tokenizer training
2. Test reconstruction quality
3. Prepare conversational dataset for S2S
4. Start S2S model training

**Medium-term (Next Month)**:
1. Train full S2S pipeline
2. Add emotion control
3. Deploy to production
4. Collect real user feedback

**Long-term (Next 3 Months)**:
1. Fine-tune on Indian languages
2. Add breathing sounds and advanced prosody
3. Scale to multiple voices/personalities
4. Optimize for <500ms latency

---

## Support & Resources

**GitHub Repository**: https://github.com/devasphn/Testing-S2S

**Useful Links**:
- LibriSpeech Dataset: http://www.openslr.org/12/
- RunPod Documentation: https://docs.runpod.io/
- WandB Monitoring: https://wandb.ai/
- PyTorch Tutorials: https://pytorch.org/tutorials/

**Community**:
- Open issues on GitHub for bugs
- Join WandB workspace for training logs
- Share checkpoints with team via RunPod Network Storage

---

**Last Updated**: November 17, 2025

**Version**: 1.0 (Tokenizer Training Phase)

**Status**: ✓ Ready to start training!
