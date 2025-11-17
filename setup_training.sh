#!/bin/bash
# Automated Training Environment Setup for Testing-S2S
# Usage: bash setup_training.sh

set -e  # Exit on error

echo "=========================================="
echo "Testing-S2S Training Environment Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    error "This script is designed for Linux (RunPod/Ubuntu). Detected: $OSTYPE"
fi

# 1. Check Python version
info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.10"

if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
    error "Python $REQUIRED_VERSION+ required. Found: $PYTHON_VERSION"
fi

info "Python version OK: $PYTHON_VERSION"

# 2. Check CUDA availability
info "Checking CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    warn "nvidia-smi not found. Training will be VERY slow on CPU!"
    warn "Consider using RunPod with GPU instances."
else
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    info "GPU detected: $GPU_INFO"
fi

# 3. Set up directories
info "Creating directory structure..."
mkdir -p /workspace/data/LibriSpeech
mkdir -p /workspace/cache/models
mkdir -p /workspace/cache/torch
mkdir -p checkpoints/tokenizer
mkdir -p checkpoints/s2s
mkdir -p logs

info "Directories created:"
info "  - /workspace/data (datasets)"
info "  - /workspace/cache (model cache)"
info "  - checkpoints/ (training checkpoints)"

# 4. Create Python virtual environment
if [ ! -d "venv" ]; then
    info "Creating Python virtual environment..."
    python3 -m venv venv
    info "Virtual environment created"
else
    info "Virtual environment already exists"
fi

# 5. Activate virtual environment
info "Activating virtual environment..."
source venv/bin/activate

# 6. Upgrade pip
info "Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

# 7. Install PyTorch with CUDA
info "Installing PyTorch 2.3.0 with CUDA 12.1..."
pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 --quiet

info "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 8. Install project dependencies
info "Installing project dependencies..."
pip install -r requirements.txt --quiet

info "Installing training dependencies..."
pip install -r requirements-training.txt --quiet

# 9. Install project in editable mode
info "Installing Testing-S2S in editable mode..."
pip install -e . --quiet

# 10. Download LibriSpeech (optional, user can skip)
echo ""
echo "=========================================="
echo "Dataset Download (Optional)"
echo "=========================================="
echo ""
echo "LibriSpeech datasets available:"
echo "  1. train-clean-100 (100h, 6GB)   - Good for testing"
echo "  2. train-clean-360 (360h, 23GB)  - Production quality"
echo "  3. train-other-500 (500h, 30GB)  - Best quality"
echo "  4. Skip dataset download"
echo ""

read -p "Select option [1-4]: " DATASET_OPTION

case $DATASET_OPTION in
    1)
        info "Downloading train-clean-100 (6GB)..."
        cd /workspace/data
        if [ ! -f "train-clean-100.tar.gz" ]; then
            wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
        fi
        info "Extracting..."
        tar -xzf train-clean-100.tar.gz
        info "Dataset ready: /workspace/data/LibriSpeech/train-clean-100"
        cd -
        ;;
    2)
        info "Downloading train-clean-360 (23GB)..."
        cd /workspace/data
        if [ ! -f "train-clean-360.tar.gz" ]; then
            wget http://www.openslr.org/resources/12/train-clean-360.tar.gz
        fi
        info "Extracting..."
        tar -xzf train-clean-360.tar.gz
        info "Dataset ready: /workspace/data/LibriSpeech/train-clean-360"
        cd -
        ;;
    3)
        info "Downloading train-other-500 (30GB)..."
        cd /workspace/data
        if [ ! -f "train-other-500.tar.gz" ]; then
            wget http://www.openslr.org/resources/12/train-other-500.tar.gz
        fi
        info "Extracting..."
        tar -xzf train-other-500.tar.gz
        info "Dataset ready: /workspace/data/LibriSpeech/train-other-500"
        cd -
        ;;
    4)
        info "Skipping dataset download"
        warn "You'll need to download datasets manually before training"
        ;;
    *)
        warn "Invalid option. Skipping dataset download"
        ;;
esac

# 11. Create default training config if needed
if [ ! -f "training/configs/tokenizer_config.yaml" ]; then
    warn "Config file missing. This should not happen!"
    error "Please ensure training/configs/tokenizer_config.yaml exists"
fi

# 12. Set up WandB (optional)
echo ""
echo "=========================================="
echo "WandB Integration (Optional)"
echo "=========================================="
echo ""
echo "WandB provides real-time training visualization."
read -p "Do you want to set up WandB? [y/N]: " SETUP_WANDB

if [[ $SETUP_WANDB =~ ^[Yy]$ ]]; then
    read -p "Enter your WandB API key: " WANDB_KEY
    echo "export WANDB_API_KEY=$WANDB_KEY" >> ~/.bashrc
    export WANDB_API_KEY=$WANDB_KEY
    info "WandB configured. Login with: wandb login"
else
    info "Skipping WandB setup"
    info "To disable WandB logging, edit training/configs/tokenizer_config.yaml:"
    info "  logging.use_wandb: false"
fi

# 13. Summary
echo ""
echo "=========================================="
echo "Setup Complete! \u2713"
echo "=========================================="
echo ""
info "Environment is ready for training."
echo ""
echo "Quick Start:"
echo "  1. Activate environment:"
echo "     $ source venv/bin/activate"
echo ""
echo "  2. Edit config (optional):"
echo "     $ nano training/configs/tokenizer_config.yaml"
echo ""
echo "  3. Start training:"
echo "     $ python training/train_tokenizer.py"
echo ""
echo "  4. Monitor progress:"
echo "     - Console logs show training metrics"
echo "     - WandB dashboard (if enabled): https://wandb.ai"
echo "     - GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "Documentation:"
echo "  - Training Guide: cat TRAINING_GUIDE.md"
echo "  - README: cat README.md"
echo ""
echo "Estimated Costs (RunPod A100 @ \$1.19/hr):"
echo "  - train-clean-100 (100h):   8-12 GPU hours = \$10-15"
echo "  - train-clean-360 (360h):   30-40 GPU hours = \$36-48"
echo "  - train-other-500 (500h):   40-50 GPU hours = \$48-60"
echo "  - Full pipeline (all):       80-120 GPU hours = \$95-143"
echo ""
info "Happy training! 🚀"
