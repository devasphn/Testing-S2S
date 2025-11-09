# Quick Test Guide - Audio Playback Fix

## 🚀 Quick Start (After Fix)

### 1. Deploy Updated Code to RunPod
```bash
cd /workspace/Testing-S2S
git pull origin main  # Pull the latest changes with audio fix
# OR manually update src/web/index.html with the fixed version
```

### 2. Restart Server (if running)
```bash
# Stop current server (Ctrl+C)
python src/server.py
```

### 3. Test in Browser
1. Open: `https://[your-pod-id]-8000.proxy.runpod.net/web`
2. Click **"Start Audio"**
3. **Listen for the test beep** 🔊 (confirms audio is working!)
4. Allow microphone access
5. Speak and pause
6. **You should hear AI response!** 🎉

---

## ✅ Success Indicators

### Browser Console Should Show:
```
✅ AudioContext unlocked successfully - test tone played
🔊 Playing: 480 samples (20.0ms) scheduled at +50ms | ctx=running
```

### You Should Hear:
1. **Brief beep** when starting (test tone)
2. **AI voice response** after you speak

---

## ❌ If Still No Audio

### Check 1: Browser Console Errors?
- Press F12 → Console tab
- Look for red errors

### Check 2: AudioContext State
```javascript
// In browser console, type:
ctx.state  // Should show: "running"
```

### Check 3: Volume/Mute
- Check system volume
- Check browser volume
- UI Volume slider at 90%
- Mute button shows 🔊 (not 🔇)

### Check 4: Browser Permissions
- Click 🔒 in address bar
- Ensure "Sound" = Allow

### Check 5: Try Hard Refresh
```
Ctrl + Shift + R  (Windows/Linux)
Cmd + Shift + R   (Mac)
```

---

## 🔧 Key Fix Applied

**The Bug**: Line 297 in `index.html` had:
```javascript
proc.connect(ctx.destination); // ❌ WRONG - creates feedback
```

**The Fix**: Connect to silent gain node instead:
```javascript
silentGain = ctx.createGain();
silentGain.gain.value = 0;
proc.connect(silentGain); // ✅ CORRECT - keeps processor alive, no feedback
```

This keeps the microphone processor active while preventing audio feedback.

---

## 📊 Expected Logs

### Browser (Good):
```
[time] 🔓 ✅ AudioContext unlocked - test tone played
[time] 🔊 Playing: 480 samples | ctx=running
[time] 🔊 Playing: 480 samples | ctx=running
```

### Server (Good):
```
[STREAM] 🤖 Generated response: 1115 samples
[STREAM] 🔊 Sent 50 frames to [client]
```

### Browser (Bad - needs fix):
```
❌ Cannot play: AudioContext or gain not initialized
⚠️ AudioContext state: suspended
```

---

## 🎯 Testing Commands

### Check if server is running:
```bash
curl http://localhost:8000/health
```

### Check WebSocket endpoint:
```bash
wscat -c ws://localhost:8000/ws/stream
# (if wscat is installed)
```

### View server logs in real-time:
```bash
# Server should show:
[STREAM] 🤖 Generated response: ...
[STREAM] 🔊 Sent X frames to ...
```

---

## 🔍 Debugging Commands

### In Browser Console:
```javascript
// Check AudioContext state
console.log('Context:', ctx?.state);

// Check gain node
console.log('Gain:', gain);

// Check WebSocket
console.log('WS:', ws?.readyState); // 1 = OPEN
```

---

## 💡 Pro Tips

1. **Use Chrome/Edge** for best Web Audio support
2. **Use headphones** to avoid speaker feedback
3. **Speak clearly** and pause 2-3 seconds for response
4. **Check volume first** - test with YouTube/music
5. **Hard refresh (Ctrl+Shift+R)** after code changes

---

## 📞 Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| No beep on start | Audio unlock failed | Hard refresh, try different browser |
| Beep plays, no AI voice | Server not generating | Check server logs |
| Crackling/distortion | Volume too high | Lower volume slider |
| Delayed audio | Network latency | Normal for cloud GPUs |
| ScriptProcessor warning | Browser deprecation | Just a warning, ignore it |

---

## ✨ What Changed?

### Before (Broken):
- Microphone connected to speakers → feedback
- Weak audio unlock → suspended context
- No auto-resume → audio stops playing

### After (Fixed):
- Microphone isolated from speakers ✅
- Strong audio unlock with test tone ✅
- Auto-resume on multiple triggers ✅

---

## 🎉 Expected Result

You should now hear:
1. **Test beep** (440Hz, 0.15s) when starting
2. **AI voice responses** after speaking
3. **Clear, continuous audio** playback

All with the same server setup - only browser code changed!
