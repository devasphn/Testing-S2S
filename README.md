# Testing-S2S

Real-time Speech-to-Speech AI Model with HiFiGAN Vocoder

## Features

✅ **Duplex Streaming**: ~400ms mean latency  
✅ **Public HiFiGAN**: No auth required, commercial-safe  
✅ **Turn-based Mode**: VAD-based conversation management  
✅ **Self-contained**: Automatic model downloading and caching  

## Architecture

- **Speech Processing**: GLM-4-Voice + Moshi streaming techniques
- **Vocoder**: Public HiFiGAN (MIT License, 22kHz)
- **VAD**: Adaptive threshold with turn detection
- **Backend**: FastAPI + WebSocket streaming

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Basic Usage
```bash
# Stream mode (continuous response)
python src/server.py

# Turn-based mode (wait for user turn completion)
REPLY_MODE=turn python src/server.py
```

### Environment Variables
- `REPLY_MODE`: `stream` (default) or `turn`
- `MODEL_CACHE_DIR`: Cache directory for downloaded models
- `CUDA_VISIBLE_DEVICES`: GPU selection

## API Endpoints

- `GET /health` - Health check
- `GET /api/stats` - Latency statistics
- `WebSocket /ws/stream` - Real-time audio streaming
- `GET /` - Built-in web interface

## Vocoder Details

**HiFiGAN Universal**:
- ✅ MIT License (commercial use allowed)
- ✅ No authentication required
- ✅ Auto-downloads from GitHub releases
- ✅ 22kHz high-quality synthesis
- ✅ ~55MB checkpoint size

See [`VOCODER_NOTICE.md`](VOCODER_NOTICE.md) for full license details.

## Performance

**Streaming Mode**:
- Mean latency: ~400ms
- Continuous response generation
- Real-time VAD processing

**Turn Mode** (REPLY_MODE=turn):
- Wait for user speech completion
- Generate longer, more coherent responses
- Better for conversational applications

## Deployment

See [`RUNPOD_DEPLOYMENT.md`](RUNPOD_DEPLOYMENT.md) for cloud deployment instructions.

## Development

```bash
# Clone and setup
git clone https://github.com/devasphn/Testing-S2S.git
cd Testing-S2S
pip install -e .

# Run tests
python -m pytest tests/

# Start development server
REPLY_MODE=turn python src/server.py
```

## License

MIT License - see LICENSE file for details.
HiFiGAN vocoder: MIT License - see VOCODER_NOTICE.md
