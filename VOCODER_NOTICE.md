# HiFiGAN Vocoder License and Provenance Notice

## Model Source

**NVIDIA HiFi-GAN (PyTorch Hub)**
- **PyTorch Hub**: `NVIDIA/DeepLearningExamples:torchhub` -> `nvidia_hifigan`
- **Original Repository**: https://github.com/jik876/hifi-gan
- **NVIDIA Implementation**: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/HiFiGAN
- **Paper**: "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (NeurIPS 2020)
- **Authors**: Jungil Kong, Jaehyeon Kim, Jaekyoung Bae

## License Information

**HiFi-GAN License**: MIT License  
**NVIDIA Implementation**: Apache 2.0 License

### Original HiFi-GAN (MIT License)
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
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### NVIDIA DeepLearningExamples (Apache 2.0)
```
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Commercial Use

✅ **Commercial Use Permitted**
- Both MIT and Apache 2.0 licenses allow unrestricted commercial use
- No attribution required in end products (though appreciated)
- Can be used in proprietary applications
- No royalties or licensing fees

## Model Details

**NVIDIA HiFi-GAN Checkpoint Information**:
- Model: HiFi-GAN Generator (22.05kHz)
- Source: PyTorch Hub automatic download
- Architecture: Multi-scale discriminator with residual blocks
- Training Data: LJSpeech dataset (publicly available)
- Performance: High-quality 22.05kHz speech synthesis
- Additional: Includes optional denoiser for improved quality

**Loading Method**:
```python
hifigan, vocoder_config, denoiser = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub', 
    'nvidia_hifigan'
)
```

## Integration Notes

**No Authentication Required**:
- Loads directly from PyTorch Hub
- Automatic caching via PyTorch Hub mechanism
- No API keys or tokens needed
- Self-contained implementation with fallback

**Cache Location**: 
- PyTorch Hub cache (typically `~/.cache/torch/hub/`)
- Configurable via `TORCH_HOME` environment variable

**Features**:
- Automatic device placement (CPU/GPU)
- Optional denoising for enhanced quality
- Robust error handling with fallback generator
- Compatible with standard mel-spectrogram inputs

## Performance

- **Sample Rate**: 22.05 kHz
- **Quality**: High-fidelity speech synthesis
- **Speed**: Real-time inference on modern GPUs
- **Memory**: Optimized for efficient inference
- **Latency**: Suitable for real-time applications

## Citation

If using this vocoder in academic work, please cite both:

**Original HiFi-GAN Paper**:
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

**NVIDIA Implementation** (if applicable):
```bibtex
@misc{nvidia2021hifigan,
  title={HiFi-GAN PyTorch Implementation},
  author={NVIDIA Corporation},
  year={2021},
  howpublished={\url{https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/HiFiGAN}}
}
```

---

**Last Updated**: October 29, 2025  
**Verification**: PyTorch Hub model verified as publicly accessible  
**Status**: Production-ready, no authentication required
