class PCMPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sampleRateTarget = 24000;
    this.buffer = new Float32Array(24000 * 4); // 4s ring buffer
    this.writePos = 0;
    this.readPos = 0;
    this.available = 0; // samples available

    this.port.onmessage = (e) => {
      const data = e.data;
      if (data && data.type === 'push' && data.samples) {
        const f32 = data.samples;
        const n = f32.length;
        // If overflow, drop oldest
        if (n > this.buffer.length) return;
        while (this.available + n > this.buffer.length) {
          this.readPos = (this.readPos + 128) % this.buffer.length;
          this.available = Math.max(0, this.available - 128);
        }
        // Write into ring
        for (let i = 0; i < n; i++) {
          this.buffer[(this.writePos + i) % this.buffer.length] = f32[i];
        }
        this.writePos = (this.writePos + n) % this.buffer.length;
        this.available += n;
        this.port.postMessage({ type: 'level', available: this.available });
      }
    };
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0];
    const outCh = output[0]; // mono
    const N = outCh.length; // usually 128 frames

    if (this.available >= N) {
      for (let i = 0; i < N; i++) {
        outCh[i] = this.buffer[(this.readPos + i) % this.buffer.length];
      }
      this.readPos = (this.readPos + N) % this.buffer.length;
      this.available -= N;
      if (this.available < 0) this.available = 0;
    } else {
      // Underrun: output silence and notify UI
      for (let i = 0; i < N; i++) outCh[i] = 0;
      this.port.postMessage({ type: 'underrun', need: N, available: this.available });
    }

    return true;
  }
}

registerProcessor('pcm-player', PCMPlayerProcessor);
