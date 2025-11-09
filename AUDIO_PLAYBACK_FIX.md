# Audio Playback Fix - Complete Guide

## Problem Summary

Your server was working perfectly (generating and sending audio), but the browser couldn't play it. Three critical issues were identified:

### Issue 1: ❌ ScriptProcessor Connection Bug (CRITICAL)
**Location**: `src/web/index.html` line 297
**Problem**: The microphone input processor was connected to speakers
```javascript
// OLD CODE (WRONG):
source.connect(proc);
proc.connect(ctx.destination); // ❌ This creates feedback!
```

**Fix**: Connect to a silent gain node instead
```javascript
// NEW CODE (CORRECT):
silentGain = ctx.createGain();
silentGain.gain.value = 0; // Completely silent
silentGain.connect(ctx.destination);

source.connect(proc);
proc.connect(silentGain); // ✅ Keeps processor alive without feedback
```

**Why this matters**: ScriptProcessorNode must be connected to something to stay active and process audio. Connecting to a muted gain node (gain=0) keeps it alive without creating audio feedback or hearing your own voice.

---

### Issue 2: ❌ Insufficient AudioContext Unlocking
**Problem**: Browsers require explicit user interaction to enable audio playback

**Fix**: Enhanced the unlock mechanism with:
- Louder test tone (0.01 → 0.1 amplitude)
- Longer duration (0.1s → 0.15s)
- Better state verification
- Fallback timeout handling

---

### Issue 3: ❌ AudioContext State Management
**Problem**: AudioContext can suspend during operation

**Fix**: Added auto-resume logic in multiple places:
- Before playing each audio frame
- When receiving WebSocket messages
- During microphone recording
- On tab visibility changes

---

## Changes Made

### File: `src/web/index.html`

1. **Added audio unlock tracking**:
   ```javascript
   let audioUnlocked = false; // Track if audio context is unlocked
   ```

2. **Enhanced unlock function** (lines 125-171):
   - More robust state checking
   - Better error handling
   - Verification of successful unlock

3. **Improved playback function** (lines 173-223):
   - Auto-resume if suspended
   - Better logging with context state
   - Improved scheduling timing

4. **Fixed WebSocket message handler** (lines 292-308):
   - Auto-resume before playback
   - Proper async handling

5. **Removed critical bug** (line 352):
   - Removed `proc.connect(ctx.destination)`

---

## Testing Instructions

### On RunPod

1. **Start your server** (if not already running):
   ```bash
   cd /workspace/Testing-S2S
   . venv/bin/activate
   python src/server.py
   ```

2. **Access the web UI**:
   - Get your RunPod public URL from the dashboard
   - Open: `https://[your-pod-id]-8000.proxy.runpod.net/web`

3. **Test audio playback**:
   - Click "Start Audio" button
   - Allow microphone access
   - **IMPORTANT**: You should hear a brief test tone (beep) - this confirms audio is unlocked
   - Speak into microphone
   - Wait for pause detection
   - **You should now hear AI response**

4. **Check browser console logs**:
   - Press F12 to open DevTools
   - Look for these success messages:
     ```
     🔓 ✅ AudioContext unlocked successfully - test tone played
     🔊 Playing: 480 samples (20.0ms) scheduled at +50ms | ctx=running
     ```

---

## Expected Behavior

### What You Should See in Browser Logs:
```
[timestamp] 🔍 Auto-detected WebSocket URL: wss://...
[timestamp] 🔌 Connecting to WebSocket...
[timestamp] 🎵 AudioContext: running, 24000 Hz
[timestamp] 🎚️ Gain node connected to destination
[timestamp] ⏰ Timeline initialized at 0.000s
[timestamp] 🔓 ✅ AudioContext unlocked successfully - test tone played
[timestamp] ✅ WebSocket connected successfully
[timestamp] 🎤 Requesting microphone access...
[timestamp] ✅ Microphone access granted
[timestamp] 🎵 Audio pipeline ready - speak now!
[timestamp] 🔊 Playing: 480 samples (20.0ms) scheduled at +50ms | ctx=running
[timestamp] 🔊 Playing: 480 samples (20.0ms) scheduled at +70ms | ctx=running
```

### What You Should See in Server Logs:
```
[STREAM] 🤖 Generated response: 1115 samples (0.05s) for [client-ip]
[STREAM] 📦 Queued 3 frames (0.05s) for [client-ip]
[STREAM] 🔊 Sent 50 frames to [client-ip]
```

---

## Troubleshooting

### Problem: Still no audio after fix

**Solution 1: Clear browser cache**
```bash
# Press Ctrl+Shift+R (hard refresh)
# Or clear cache: Ctrl+Shift+Delete
```

**Solution 2: Check volume settings**
- Browser is not muted
- System volume is up
- Volume slider in UI is at 90%
- "Mute" button shows 🔊 (not 🔇)

**Solution 3: Verify AudioContext state**
Open browser console and check:
```javascript
// Should see: "running"
console.log(ctx.state);
```

**Solution 4: Check for browser audio permission**
- Click the 🔒 icon in browser address bar
- Ensure "Sound" is set to "Allow"

**Solution 5: Try different browser**
- Chrome/Edge: Best support
- Firefox: Good support
- Safari: May require additional interaction

---

## System Requirements Verification

Your RunPod setup looks correct:
```bash
✅ PyTorch 2.3.0 with CUDA 12.1
✅ L4 GPU (sufficient for HiFiGAN)
✅ All audio libraries installed:
   - libsndfile1
   - libsox-dev
   - sox
   - ffmpeg
   - portaudio19-dev
   - libasound2-dev
```

**Note**: The system audio libraries (sox, ffmpeg, etc.) are for server-side processing. Browser audio playback uses Web Audio API and doesn't depend on server-side installations.

---

## APT Installations - What They're For

Your apt packages are correct for the **server side**:

| Package | Purpose | Required for Browser Audio? |
|---------|---------|------------------------------|
| libsndfile1 | Audio file I/O | No (server-side only) |
| sox/libsox-dev | Audio conversion | No (server-side only) |
| ffmpeg | Media processing | No (server-side only) |
| portaudio19-dev | Audio device access | No (server-side only) |
| libasound2-dev | ALSA sound system | No (server-side only) |

**Browser audio playback does NOT require any server-side audio libraries**. It uses the browser's Web Audio API, which is completely independent.

---

## Key Differences: Server vs Browser Audio

### Server Side (Python):
- Generates audio waveforms
- Uses HiFiGAN vocoder
- Converts to PCM16 format
- Sends via WebSocket

### Browser Side (JavaScript):
- Receives PCM16 data
- Converts to Float32
- Schedules playback via Web Audio API
- Plays through system audio

**The fix addresses the browser side only** - your server code was already working perfectly!

---

## Verification Checklist

After deploying the fix:

- [ ] Test tone plays when clicking "Start Audio" (you should hear a brief beep)
- [ ] Browser console shows `AudioContext unlocked successfully`
- [ ] No errors in browser console
- [ ] Server logs show frames being sent
- [ ] You can hear AI responses after speaking
- [ ] Volume control works
- [ ] Mute button works

---

## Additional Notes

### ScriptProcessorNode Deprecation Warning

You may still see this warning in console:
```
[Deprecation] The ScriptProcessorNode is deprecated. Use AudioWorkletNode instead.
```

**This is just a warning** and doesn't affect functionality. The fix addresses the actual bug (connecting input to output), not the deprecation warning.

To fully remove the warning, you would need to migrate to AudioWorklet, which is a larger refactoring. The current fix makes audio work correctly.

---

## Support

If audio still doesn't work after applying this fix:

1. Check browser console for errors
2. Verify server logs show audio being sent
3. Try a different browser
4. Ensure no browser extensions are blocking audio
5. Test with headphones (to rule out speaker issues)

---

## Summary

The fix resolves **three critical browser-side bugs**:
1. ✅ Removed incorrect microphone→speaker connection
2. ✅ Enhanced AudioContext unlocking
3. ✅ Added automatic context resumption

**No server-side changes needed** - your Python code and apt installations are correct.

The issue was purely in the browser JavaScript code managing audio playback.
