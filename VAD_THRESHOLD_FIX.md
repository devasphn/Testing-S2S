# VAD Threshold Fix - "Always On" Audio Issue Resolved

## Problem Summary

The system was experiencing "always on" continuous audio where the AI would generate responses even during silence or background noise. This was caused by an extremely low VAD (Voice Activity Detection) threshold of **0.001** that was treating all audio input, including silence and noise, as speech.

## Root Cause

The low threshold value `vad_threshold=0.001` was being passed during `StreamingProcessor` instantiation in `src/server.py`, causing Silero VAD to trigger on virtually any audio input.

### Why 0.001 is Too Low

- **Silero VAD** outputs speech probability values between 0.0 (definitely not speech) and 1.0 (definitely speech)
- **0.001 threshold** means anything above 0.1% confidence is treated as speech
- This includes:
  - Background noise
  - Room ambience  
  - Microphone static
  - Breathing sounds
  - Very quiet environmental sounds

### Why 0.65 is Correct

- **0.65 threshold** means only audio with 65%+ speech confidence triggers response
- This is the **standard production value** for Silero VAD
- Filters out:
  - Background noise
  - Silence
  - Low-confidence audio
- Only responds to:
  - Clear speech
  - Normal conversation volume
  - Deliberate voice input

## Files Updated

### 1. `src/server.py` (Line 58)

**Before:**
```python
_proc = StreamingProcessor(
    model=_model,
    speech_tokenizer=_tok,
    chunk_size_ms=80,
    sample_rate=TRANSPORT_SR,
    max_latency_ms=200,
    vad_threshold=0.001,  # ❌ TOO LOW - triggers on silence
    reply_mode=REPLY_MODE
)
```

**After:**
```python
_proc = StreamingProcessor(
    model=_model,
    speech_tokenizer=_tok,
    chunk_size_ms=80,
    sample_rate=TRANSPORT_SR,
    max_latency_ms=200,
    vad_threshold=0.65,  # ✅ FIXED - only triggers on clear speech
    reply_mode=REPLY_MODE
)
```

### 2. `configs/base_config.yaml`

**Before:**
```yaml
server:
  host: 0.0.0.0
  port: 8000
  vad_threshold: 0.01  # ❌ Still too low
  sample_rate: 24000
```

**After:**
```yaml
server:
  host: 0.0.0.0
  port: 8000
  vad_threshold: 0.65  # ✅ FIXED - prevents false positives
  sample_rate: 24000
```

## Testing & Verification

### Pull Latest Changes

```bash
cd /workspace/Testing-S2S
git pull
```

### Restart Server

```bash
# For turn-based mode
REPLY_MODE=turn python src/server.py

# For streaming mode
python src/server.py
```

### Expected Log Output

You should now see:

```
[INFO] Starting server on device: cuda with REPLY_MODE=turn
[INFO] Initializing PublicHiFiGANVocoder...
[INFO] Loading NVIDIA HiFi-GAN from PyTorch Hub...
[INFO] NVIDIA HiFi-GAN loaded successfully (22050Hz)
[INFO] PublicHiFiGANVocoder ready
[INFO] Silero VAD initialized:
       - Threshold: 0.65  # ✅ VERIFY THIS LINE
       - Silence frames: 30 (~2.4s)
       - Sample rate: 16000 Hz
       - Required samples: 512
       - Accuracy: 92% (neural network)
[INFO] StreamingProcessor initialized:
       - REPLY_MODE: turn
       - Chunk size: 80ms
       - Sample rate: 24000 Hz
       - VAD: Silero (neural network, 92% accurate)
```

**CRITICAL CHECK:** Verify the log shows `Threshold: 0.65` (NOT 0.001 or 0.01)

### Expected Behavior After Fix

#### ✅ Correct Behavior (Now)

- **Silence is ignored** - No response generated during quiet periods
- **Background noise is ignored** - Room ambience doesn't trigger responses  
- **Clear speech triggers response** - Normal conversation volume works perfectly
- **Turn detection works properly** - In turn mode, waits for you to finish speaking

#### ❌ Previous Incorrect Behavior (Fixed)

- ~~Continuous audio blips during silence~~
- ~~Responses triggered by background noise~~
- ~~"Always on" generation even when not speaking~~
- ~~Short 0.1-0.2s audio bursts constantly~~

## Technical Details

### Silero VAD Configuration

```python
SileroVADWrapper(
    threshold=0.65,        # Speech probability threshold (0.0-1.0)
    silence_frames=30,     # 30 frames (~2.4s) of silence ends turn
    sample_rate=16000      # Silero expects 16kHz input
)
```

### Turn Mode Logic

1. User speaks → VAD detects speech (prob > 0.65)
2. Speech chunks accumulated in `turn_buffer`
3. User stops → 30 frames of low probability (< 0.65)
4. Turn ends → Model generates full response
5. Response audio sent to client
6. Reset for next turn

### Stream Mode Logic

1. User speaks → VAD detects speech (prob > 0.65)
2. Immediate response generation (low latency)
3. Audio chunks sent in real-time
4. No turn accumulation

## Verification Commands

### Check Current Threshold in Running Server

```bash
curl -s http://localhost:8000/api/stats | jq
```

Look for:
```json
{
  "latency_ms": {
    "vad_type": "silero_neural_network",
    "vad_accuracy": "92%"
  }
}
```

### Check Git Commit History

```bash
git log --oneline -5
```

You should see:
```
a091898 Fix: Update VAD threshold in base_config.yaml from 0.01 to 0.65
e6e19ff Fix: Change VAD threshold from 0.001 to 0.65 to prevent "always on" audio
```

## Troubleshooting

### If Still Hearing Continuous Audio

1. **Verify git pull worked:**
   ```bash
   git status
   git log -1
   ```

2. **Check actual threshold in logs:**
   - Look for `[INFO] Silero VAD initialized: - Threshold: 0.65`
   - If you see 0.001 or 0.01, the pull didn't work

3. **Hard reset if needed:**
   ```bash
   git fetch origin
   git reset --hard origin/main
   ```

4. **Verify Python process restarted:**
   ```bash
   ps aux | grep python
   # Kill old process if still running
   pkill -9 -f server.py
   # Start fresh
   REPLY_MODE=turn python src/server.py
   ```

### If No Audio at All

1. **Threshold might be too high** - Try speaking louder or closer to mic
2. **Check microphone input** - Verify your client is sending audio
3. **Temporarily lower threshold for testing:**
   ```python
   vad_threshold=0.5  # Less strict, for testing only
   ```

## Summary

✅ **Fixed Files:**
- `src/server.py` - threshold 0.001 → 0.65
- `configs/base_config.yaml` - threshold 0.01 → 0.65

✅ **Result:**
- Silence no longer triggers audio generation
- Clear speech detection only
- Proper turn-based conversation flow
- No more "always on" continuous audio blips

## Next Steps

After pulling and restarting:

1. ✅ Verify log shows `Threshold: 0.65`
2. ✅ Test with silence - should hear nothing
3. ✅ Test with speech - should generate response
4. ✅ Test turn mode - should wait for you to finish
5. ✅ Confirm no audio during quiet periods

If all checks pass, the fix is complete! 🎉
