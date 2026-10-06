import numpy as np
import sounddevice as sd
import soundfile as sf
import Quartz
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import threading
import time
import sys
import json
sys.path.insert(0, "/Users/noor/Nova")
from core.agent_loop import run as agent_run
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

STATE_FILE = "/Users/noor/Nova/nova_state.json"

def set_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"state": state, "ts": time.time()}, f)
    except Exception:
        pass

SAMPLE_RATE = 16000
CLICK_RATE = 24000
SILENCE_THRESHOLD = 0.01
SILENCE_LIMIT = 10.0
HOLD_THRESHOLD = 0.8

whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
kokoro = Kokoro("/Users/noor/.kokoro/kokoro-v1.0.onnx", "/Users/noor/.kokoro/voices-v1.0.bin")

awake = False
audio_buffer = []
last_loud_time = 0
enter_press_time = None

def click(freq=1000, duration=0.1, volume=0.3):
    t = np.linspace(0, duration, int(CLICK_RATE * duration), False)
    tone = np.sin(freq * 2 * np.pi * t)
    envelope = np.exp(-t * 30)
    attack = np.minimum(1.0, t / 0.002)  # 2ms fade-in to avoid onset pop
    return (tone * envelope * attack * volume).astype(np.float32)

wake_click = click(freq=1200)
sleep_click = click(freq=800)
thinking_chime = click(freq=700, duration=0.12, volume=0.18)  # subtle "working" cue, no spoken ack

def audio_callback(indata, frames, time_info, status):
    global last_loud_time
    if awake:
        audio_buffer.append(indata.copy())
        if np.abs(indata).max() > SILENCE_THRESHOLD:
            last_loud_time = time.time()

stream = None

def start_stream():
    global stream
    if stream is not None:
        return
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
    try:
        stream.start()
    except Exception:
        try:
            stream.close()
        except Exception:
            pass
        stream = None
        raise

def stop_stream():
    global stream
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None

def play_output(samples, sr):
    """Smooth TTS playback: dedicated OutputStream with jitter tolerance + fades."""
    samples = samples.astype(np.float32, copy=True)
    fade = int(sr * 0.01)
    if fade > 1 and len(samples) > fade * 2:
        samples[:fade] *= np.linspace(0.0, 1.0, fade)
        samples[-fade:] *= np.linspace(1.0, 0.0, fade)
    with sd.OutputStream(samplerate=sr, channels=1, dtype="float32", latency="high") as out:
        out.write(samples)

MAX_TTS_CHARS = 200  # keep phonemes safely under Kokoro's 510-phoneme cap

def speak(text):
    """Speak a TTS reply. Never fatal: TTS failures are logged and a short
    tone plays instead, so a crash can't nuke an otherwise-good answer."""
    if not text or not text.strip():
        return
    set_state("speaking")
    text = text.strip()
    print(f"Nova: {text}")  # always log the FULL answer so the transcript keeps it
    tts_text = text
    if len(tts_text) > MAX_TTS_CHARS:
        tts_text = tts_text[:MAX_TTS_CHARS].rstrip() + "…"
    try:
        samples, sr = kokoro.create(tts_text, voice="af_heart", speed=1.0, lang="en-us")
        play_output(samples, sr)
    except Exception as e:
        print(f"[error] TTS failed — full answer is in the transcript above: {e}")
        sd.play(click(freq=320, duration=0.15, volume=0.2), CLICK_RATE)
    finally:
        set_state("idle")

def go_to_sleep(reason):
    global awake
    set_state("idle")
    awake = False
    stop_stream()
    time.sleep(0.08)  # let the input device close before the click
    sd.play(sleep_click, CLICK_RATE)
    print(f"Sleeping ({reason}).")
    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0)
        sf.write("/Users/noor/Nova/last_recording.wav", audio_data, SAMPLE_RATE)
        segments, info = whisper_model.transcribe("/Users/noor/Nova/last_recording.wav")
        text = " ".join(seg.text for seg in segments).strip()
        print(f"Heard: {text}")
        if text:
            print("Thinking...")
            set_state("thinking")
            sd.play(thinking_chime, CLICK_RATE)  # subtle non-verbal ack — no robotic "one sec"
            sd.wait()
            try:
                result = agent_run(text, clearance="GREEN")
                answer = result[0] if isinstance(result, tuple) else result
                speak(str(answer))
            except Exception as e:
                print(f"[error] agent_run failed: {e}")
                speak("Sorry, that request took too long or hit a snag. "
                      "If a permission prompt showed up on screen, click Allow, "
                      "then ask me again.")

def wake_up():
    global awake, audio_buffer, last_loud_time
    set_state("listening")
    awake = True
    audio_buffer = []
    last_loud_time = time.time()
    start_stream()
    time.sleep(0.08)  # let the mic device settle before the click
    sd.play(wake_click, CLICK_RATE)
    print("Awake. Listening... (tap Enter to sleep, or 10s silence)")

def silence_watcher():
    while True:
        if awake and (time.time() - last_loud_time > SILENCE_LIMIT):
            try:
                go_to_sleep("10s silence")
            except Exception as e:
                print(f"[error] go_to_sleep failed: {e}")
        time.sleep(0.5)

# ── Enter detection: CGEventSourceKeyState (NO accessibility permission needed) ──
# pynput/CGEventTap required macOS Accessibility trust (blocked the launchd agent).
# CGEventSourceKeyState polls key state from the system event source and works
# untrusted — so Enter wake works under launchd's minimal environment.
ENTER_KEYCODES = (36, 76)  # Return, Keypad Enter

def enter_watcher():
    global enter_press_time, awake, stream
    src = Quartz.kCGEventSourceStateCombinedSessionState
    while True:
        down = any(Quartz.CGEventSourceKeyState(src, code) for code in ENTER_KEYCODES)
        if down and enter_press_time is None:
            enter_press_time = time.time()
        elif not down and enter_press_time is not None:
            held_duration = time.time() - enter_press_time
            enter_press_time = None
            if held_duration >= HOLD_THRESHOLD:
                if not awake:
                    try:
                        wake_up()
                    except Exception as e:
                        print(f"[error] wake_up failed: {e}")
                        awake = False
                        stream = None
                        set_state("idle")
                        try:
                            speak("The microphone is busy or unavailable. "
                                  "Check your input device, then hold Enter to wake me again.")
                        except Exception as se:
                            print(f"[error] busy-announcement failed: {se}")
            else:
                if awake:
                    try:
                        go_to_sleep("Enter tapped")
                    except Exception as e:
                        print(f"[error] go_to_sleep failed: {e}")
        time.sleep(0.05)

threading.Thread(target=silence_watcher, daemon=True).start()
threading.Thread(target=enter_watcher, daemon=True).start()
print("Ready. Hold Enter ~0.8s to wake Nova, tap Enter to sleep. Ctrl+C to quit.")
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("Quit.")
