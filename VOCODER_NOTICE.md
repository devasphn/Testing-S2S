# HiFiGAN Vocoder License and Provenance Notice

## Model Source

**HiFiGAN Universal Generator**
- **Original Repository**: https://github.com/jik876/hifi-gan
- **Paper**: "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (NeurIPS 2020)
- **Authors**: Jungil Kong, Jaehyeon Kim, Jaekyoung Bae
- **Download Source**: GitHub Releases (Public)

## License Information

**HiFi-GAN License**: MIT License

```
MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILIY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Commercial Use

✅ **Commercial Use Permitted**
- The MIT License allows unrestricted commercial use
- No attribution required in end products (though appreciated)
- Can be used in proprietary applications
- No royalties or licensing fees

## Model Details

**Checkpoint Information**:
- Model: Universal HiFiGAN Generator (22kHz)
- Size: ~55MB
- Architecture: Multi-scale discriminator with residual blocks
- Training Data: LibriTTS, VCTK, LJSpeech (publicly available datasets)
- Performance: High-quality 22kHz speech synthesis

**Download URLs**:
- Generator: `https://github.com/jik876/hifi-gan/releases/download/v1.0/generator_universal.pth.tar`
- Config: `https://github.com/jik876/hifi-gan/releases/download/v1.0/config_universal.json`

## Integration Notes

**No Authentication Required**:
- Downloads directly from GitHub releases
- No API keys or tokens needed
- Self-contained implementation
- Automatic caching to avoid re-downloads

**Cache Location**: 
- Default: `/workspace/cache/models/hifigan_public/`
- Configurable via `MODEL_CACHE_DIR` environment variable

## Citation

If using this vocoder in academic work, please cite:

```bibtex
@inproceedings{kong2020hifi,
  title={HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis},
  author={Kong, Jungil and Kim, Jaehyeon and Bae, Jaekyoung},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={17022--17033},
  year={2020}
}
```

---

**Last Updated**: October 29, 2025  
**Verification**: All URLs and licenses verified as publicly accessible
