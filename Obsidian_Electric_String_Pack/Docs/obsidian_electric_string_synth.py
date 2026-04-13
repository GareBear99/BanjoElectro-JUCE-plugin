#!/usr/bin/env python3
"""
Obsidian Electric String Synth
Renders dark picked electric-string / haunted electric-guitar-adjacent one-shots and loops.
"""
import argparse, math, wave
from pathlib import Path
import numpy as np

SR = 48000
NOTE_TO_SEMITONE = {"C":0,"C#":1,"DB":1,"D":2,"D#":3,"EB":3,"E":4,"F":5,"F#":6,"GB":6,"G":7,"G#":8,"AB":8,"A":9,"A#":10,"BB":10,"B":11}

def note_to_midi(note):
    note = note.strip().upper()
    if len(note) < 2: raise ValueError(note)
    if len(note) >= 3 and note[1] in ("#", "B"):
        name = note[:2]; octave = int(note[2:])
    else:
        name = note[:1]; octave = int(note[1:])
    return 12*(octave+1) + NOTE_TO_SEMITONE[name]

def midi_to_freq(midi): return 440.0 * (2 ** ((midi - 69) / 12.0))

def write_wav_float32(path, audio, sr=SR):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1: audio = audio[:,None]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(audio.shape[1]); wf.setsampwidth(4); wf.setframerate(sr); wf.writeframes(audio.astype(np.float32).tobytes())

def norm(x, peak=0.92):
    x = np.asarray(x, dtype=np.float32)
    return (x * (peak / (np.max(np.abs(x)) + 1e-12))).astype(np.float32)

def onepole_lowpass(x, cutoff_hz):
    x = np.asarray(x, dtype=np.float32)
    if cutoff_hz <= 0: return np.zeros_like(x)
    rc = 1.0 / (2 * np.pi * cutoff_hz); dt = 1.0 / SR; alpha = dt / (rc + dt)
    y = np.zeros_like(x)
    for i in range(1, len(x)): y[i] = y[i-1] + alpha * (x[i] - y[i-1])
    return y

def highpass(x, cutoff_hz=40): return x - onepole_lowpass(x, cutoff_hz)
def softclip(x, drive=1.2): return np.tanh(x * drive) / np.tanh(max(drive, 1e-6))

def mono_to_stereo(sig, pan=0.0):
    l = math.sqrt((1-pan)/2); r = math.sqrt((1+pan)/2)
    return np.stack([sig*l, sig*r], axis=1).astype(np.float32)

def simple_delay_stereo(st, ms_l=90, ms_r=140, fb=0.18, mix=0.12):
    st = np.asarray(st, dtype=np.float32); out = st.copy(); delays = [int(SR*ms_l/1000), int(SR*ms_r/1000)]
    for c, d in enumerate(delays):
        y = out[:, c].copy()
        for i in range(d, len(y)): y[i] += y[i-d] * fb
        out[:, c] = st[:, c]*(1-mix) + y*mix
    return out.astype(np.float32)

def simple_reverb_stereo(st, mix=0.10):
    taps = [(47, 0.14), (89, 0.10), (131, 0.08), (197, 0.06), (281, 0.04)]
    out = st.copy()
    for c in range(2):
        y = st[:, c].copy()
        for ms, g in taps:
            d = int(SR*ms/1000)
            if d < len(y): y[d:] += st[:-d, c] * g
        y = onepole_lowpass(y, 5200)
        out[:, c] = st[:, c]*(1-mix) + y*mix
    return out.astype(np.float32)

def make_electric_string_pluck(freq, dur=0.8, brightness=0.72, darkness=0.26, pick=0.07, amp_edge=0.42, body_mix=0.08):
    n = int(dur * SR); delay_len = max(2, int(SR / max(freq, 30.0)))
    buf = (np.random.rand(delay_len).astype(np.float32) * 2 - 1); buf -= np.mean(buf)
    if np.max(np.abs(buf)) > 0: buf /= np.max(np.abs(buf))
    out = np.zeros(n, dtype=np.float32); idx = 0; damping = 0.992 - (0.018 * darkness); tone = 0.56 + 0.30 * brightness; last = 0.0
    for i in range(n):
        cur = buf[idx]; nxt = buf[(idx + 1) % delay_len]
        new = damping * (tone * 0.5 * (cur + nxt) + (1 - tone) * last)
        out[i] = cur; buf[idx] = new; last = new; idx = (idx + 1) % delay_len
    t = np.arange(n, dtype=np.float32) / SR; phase = 2*np.pi*freq*t
    string_harm = 0.10*np.sin(phase*2.0) + 0.07*np.sin(phase*3.0 + 0.1) + 0.04*np.sin(phase*4.02 + 0.3)
    body = body_mix * (0.50*np.sin(2*np.pi*180*t) * np.exp(-t*8.5) + 0.25*np.sin(2*np.pi*360*t) * np.exp(-t*10.5))
    click = (np.random.randn(n).astype(np.float32) * np.exp(-t*110) * pick)
    pickup = 0.18*np.sin(phase*2.97 + 0.25) + 0.10*np.sin(phase*5.01) + 0.05*np.sin(phase*7.0 + 0.4)
    pickup = highpass(pickup, 500); pickup = onepole_lowpass(pickup, 4200); pickup = softclip(pickup, 1.9) * amp_edge
    sig = out + string_harm + body + click + pickup
    sig = highpass(sig, 65); sig = onepole_lowpass(sig, 4800 if brightness < 0.8 else 6200)
    dark_copy = onepole_lowpass(sig, 1700); sig = sig * (1 - darkness * 0.18) + dark_copy * (darkness * 0.12)
    sig *= np.exp(-t * (2.8 + darkness*1.0)).astype(np.float32)
    return softclip(sig * 1.12, 1.35).astype(np.float32)

def add_event(buf, start_s, audio, pan=0.0):
    start = int(start_s * SR); st = mono_to_stereo(audio, pan=pan) if audio.ndim == 1 else audio; end = min(len(buf), start + len(st))
    if start < len(buf): buf[start:end] += st[:end-start]

def render_oneshot(note, out_path, dur=1.2, brightness=0.72, darkness=0.24, amp_edge=0.42):
    mono = make_electric_string_pluck(midi_to_freq(note_to_midi(note)), dur=dur, brightness=brightness, darkness=darkness, amp_edge=amp_edge, body_mix=0.05)
    st = np.stack([mono, mono], axis=1); st = simple_reverb_stereo(st, mix=0.08); st = norm(st, 0.88); write_wav_float32(out_path, st)

def render_loop(events, bpm, out_path, bars=4, tonal_darkness=0.22, reverb_mix=0.10):
    beat_s = 60.0 / bpm; total_s = bars * 4 * beat_s; n = int(total_s * SR); st = np.zeros((n, 2), dtype=np.float32)
    for beat, midi, dur_beats, vel, pan, bright, dark, edge in events:
        pl = make_electric_string_pluck(midi_to_freq(midi), dur=dur_beats*beat_s, brightness=bright, darkness=dark, pick=0.05 + 0.025*vel, amp_edge=edge, body_mix=0.05) * vel
        add_event(st, beat * beat_s, pl, pan=pan)
    st[:,0] = highpass(st[:,0], 70); st[:,1] = highpass(st[:,1], 70)
    st[:,0] = onepole_lowpass(st[:,0], 6000 - int(tonal_darkness*1400)); st[:,1] = onepole_lowpass(st[:,1], 6000 - int(tonal_darkness*1400))
    st = simple_delay_stereo(st, ms_l=85, ms_r=130, fb=0.15, mix=0.10); st = simple_reverb_stereo(st, mix=reverb_mix)
    fade = 256; a = np.linspace(0, 1, fade, dtype=np.float32)[:, None]; st[:fade] = st[:fade] * a + st[-fade:] * (1 - a); st[-fade:] = st[:fade]
    st = norm(st, 0.90); write_wav_float32(out_path, st)

PRESETS = {
    "hollows_electric_arp": {"bpm":110, "events":[(i*0.5, m, 0.48, 0.84, -0.17 if i%2==0 else 0.17, 0.74, 0.24, 0.46) for i,m in enumerate([45,52,57,60,57,52,45,52,43,50,55,58,55,50,43,50,40,47,52,55,52,47,40,47,43,50,55,59,55,50,43,50])]},
    "ritual_electric_pulse": {"bpm":120, "events":[(b, m, 0.86, 0.90 if b%4==0 else 0.76, -0.1 if i%2==0 else 0.1, 0.60, 0.34, 0.34) for i,(b,m) in enumerate([(0,45),(1,45),(2,48),(3,50),(4,45),(5,43),(6,48),(7,50),(8,40),(9,40),(10,43),(11,47),(12,45),(13,43),(14,48),(15,50)])]},
    "veilglass_electric_harmonics": {"bpm":140, "events":[(i*0.5, m, 0.40, 0.66, -0.20 if i%2 else 0.20, 0.80, 0.18, 0.50) for i,m in enumerate([69,72,76,72,67,71,74,71,64,67,71,67,62,66,69,66,69,72,76,79,71,74,78,74,69,72,76,72,67,71,74,71])]}
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oneshot","loop"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="C4")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="hollows_electric_arp")
    ap.add_argument("--brightness", type=float, default=0.72)
    ap.add_argument("--darkness", type=float, default=0.24)
    ap.add_argument("--amp_edge", type=float, default=0.42)
    ap.add_argument("--duration", type=float, default=1.2)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "oneshot":
        render_oneshot(args.note, out, dur=args.duration, brightness=args.brightness, darkness=args.darkness, amp_edge=args.amp_edge)
    else:
        p = PRESETS[args.preset]
        render_loop(p["events"], p["bpm"], out)
    print(f"Rendered: {out}")
