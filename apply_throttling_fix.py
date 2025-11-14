#!/usr/bin/env python3
"""
Automated fix script for STREAM mode throttling
Run this from your Testing-S2S directory:
  python apply_throttling_fix.py
"""

import re

def apply_fix():
    file_path = "src/models/streaming_processor.py"
    
    print("[INFO] Reading streaming_processor.py...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Add throttling variables to __init__
    init_pattern = r'(self\.generating_response = False\s+self\.response_generated = False)'
    init_replacement = r'''\1
        
        # Response throttling for STREAM mode (prevents buffer accumulation)
        self.last_response_time = 0
        self.min_response_interval = 2.0  # 2 seconds between responses
        self.stream_chunks_buffer = []
        self.min_chunks_for_response = 10  # ~800ms of speech minimum'''
    
    content = re.sub(init_pattern, init_replacement, content)
    print("[OK] Added throttling variables to __init__")
    
    # Fix 2: Replace _process_stream_mode function
    old_function = r'async def _process_stream_mode\(self, chunk: torch\.Tensor, vad_result: Dict\[str, bool\], t0: float\) -> Optional\[torch\.Tensor\]:.*?(?=\n    async def|\n    def|\nclass|\Z)'
    
    new_function = '''async def _process_stream_mode(self, chunk: torch.Tensor, vad_result: Dict[str, bool], t0: float) -> Optional[torch.Tensor]:
        """
        STREAM MODE with response throttling to prevent buffer accumulation
        Only generates response every 2 seconds during continuous speech
        """
        if not vad_result["is_voice"]:
            # Clear buffer on silence
            if self.stream_chunks_buffer:
                print(f"[STREAM] Silence detected, clearing {len(self.stream_chunks_buffer)} buffered chunks")
                self.stream_chunks_buffer.clear()
            return None
        
        # Accumulate voice chunks
        self.stream_chunks_buffer.append(chunk)
        
        # Check throttling conditions
        current_time = time.time()
        time_since_last = current_time - self.last_response_time
        enough_chunks = len(self.stream_chunks_buffer) >= self.min_chunks_for_response
        enough_time = time_since_last >= self.min_response_interval
        
        # Only generate if we have enough chunks AND enough time has passed
        if not (enough_chunks and enough_time):
            if len(self.stream_chunks_buffer) % 25 == 0:  # Log every ~2 seconds
                print(f"[STREAM] Buffering: {len(self.stream_chunks_buffer)} chunks | "
                      f"Time since last: {time_since_last:.1f}s | "
                      f"Speech prob: {vad_result.get('speech_prob', 0):.2f}")
            return None
        
        # Generate response from accumulated chunks
        speech_prob = vad_result.get("speech_prob", 0.0)
        print(f"[STREAM] 🎯 THROTTLED RESPONSE | Chunks: {len(self.stream_chunks_buffer)} | "
              f"Interval: {time_since_last:.1f}s | Speech: {speech_prob:.2f}")
        
        # TEMPORARY: Test tone generation (models not trained yet)
        sample_rate = 24000
        duration = 0.5
        num_samples = int(sample_rate * duration)
        
        t = torch.linspace(0, duration, num_samples, device=self.device)
        frequency = 440.0 + (len(self.stream_chunks_buffer) * 5.0)  # Vary by buffer size
        amplitude = 0.3
        
        out_audio = amplitude * torch.sin(2 * torch.pi * frequency * t)
        
        # Envelope
        envelope_len = int(sample_rate * 0.05)
        envelope = torch.ones_like(out_audio)
        envelope[:envelope_len] = torch.linspace(0, 1, envelope_len, device=self.device)
        envelope[-envelope_len:] = torch.linspace(1, 0, envelope_len, device=self.device)
        out_audio = out_audio * envelope
        
        print(f"[STREAM] Generated: {num_samples} samples | {frequency:.0f}Hz | "
              f"Latency: {(time.time() - t0)*1000:.0f}ms")
        
        # Update throttling state
        self.last_response_time = current_time
        self.stream_chunks_buffer.clear()
        
        self.lat_hist.append((time.time() - t0) * 1000.0)
        return out_audio
'''
    
    content = re.sub(old_function, new_function, content, flags=re.DOTALL)
    print("[OK] Replaced _process_stream_mode with throttled version")
    
    # Fix 3: Update reset() method
    reset_pattern = r'(self\.vad\.reset\(\)\s+print\(".*?StreamingProcessor state reset.*?"\))'
    reset_replacement = r'''self.last_response_time = 0
        self.stream_chunks_buffer.clear()
        self.vad.reset()
        print("[INFO] StreamingProcessor state reset (including throttling)")'''
    
    content = re.sub(reset_pattern, reset_replacement, content)
    print("[OK] Updated reset() method")
    
    # Backup original
    import shutil
    backup_path = file_path + ".backup"
    shutil.copy2(file_path, backup_path)
    print(f"[INFO] Backup saved to {backup_path}")
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("\n✅ Fix applied successfully!")
    print("\n📋 Next steps:")
    print("  1. Review changes: git diff src/models/streaming_processor.py")
    print("  2. Test the server: REPLY_MODE=stream python src/server.py")
    print("  3. Confirm latency is now under 2 seconds")
    print("  4. Commit: git add src/models/streaming_processor.py")
    print("  5. Commit: git commit -m 'Add response throttling to STREAM mode'")
    print("  6. Push: git push")

if __name__ == "__main__":
    apply_fix()
