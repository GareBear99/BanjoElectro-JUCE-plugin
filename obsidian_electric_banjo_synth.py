#!/usr/bin/env python3
"""
Obsidian Electric Banjo Synth
-----------------------------

Renders dark / electric banjo-inspired one-shots and loops to 48 kHz 32-bit float WAV.

No third-party packages required beyond numpy.

Examples:
    python obsidian_electric_banjo_synth.py --mode oneshot --note C4 --out banjo_c4.wav
    python obsidian_electric_banjo_synth.py --mode loop --preset hollows_arp --out hollows.wav
"""

import argparse
import math
import wave
from pathlib import Path

import numpy as np

SR = 48000


NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4,
    "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11,
}


def note_to_midi(note: str) -> int:
    note = note.strip().upper()
    if len(note) < 2:
        raise ValueError(f"Invalid note: {note}")
    if note[1] in ("#", "B"):
        name = note[:2]
        octave = int(note[2:])
    else:
        name = note[:1]
        octave = int(note[1:])
    if name not in NOTE_TO_SEMITONE:
        raise ValueError(f"Invalid note name: {name}")
    return 12 * (octave + 1) + NOTE_TO_SEMITONE[name]


def midi_to_freq(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def write_wav_float32(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(audio.shape[1])
        wf.setsampwidth(4)
        wf.setframerate(sr)
        wf.writeframes(audio.astype(np.float32).tobytes())


def norm(x: np.ndarray, peak: float = 0.92) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.max(np.abs(x)) + 1e-12
    return (x * (peak / m)).astype(np.float32)


def onepole_lowpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if cutoff_hz <= 0:
        return np.zeros_like(x)
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    dt = 1.0 / SR
    alpha = dt / (rc + dt)
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def highpass(x: np.ndarray, cutoff_hz: float = 40.0) -> np.ndarray:
    return x - onepole_lowpass(x, cutoff_hz)


def softclip(x: np.ndarray, drive: float = 1.2) -> np.ndarray:
    return np.tanh(x * drive) / np.tanh(drive)


def simple_delay_stereo(st: np.ndarray, ms_l: float = 110, ms_r: float = 150,
                        fb: float = 0.2, mix: float = 0.16) -> np.ndarray:
    st = np.asarray(st, dtype=np.float32)
    out = st.copy()
    dl = int(SR * ms_l / 1000.0)
    dr = int(SR * ms_r / 1000.0)
    for c, d in enumerate([dl, dr]):
        y = out[:, c].copy()
        for i in range(d, len(y)):
            y[i] += y[i - d] * fb
        out[:, c] = st[:, c] * (1 - mix) + y * mix
    return out.astype(np.float32)


def simple_reverb_stereo(st: np.ndarray, mix: float = 0.12) -> np.ndarray:
    taps = [(59, 0.16), (101, 0.12), (147, 0.09), (223, 0.07), (307, 0.05)]
    out = st.copy()
    for c in range(2):
        y = st[:, c].copy()
        for ms, gain in taps:
            d = int(SR * ms / 1000.0)
            if d < len(y):
                y[d:] += st[:-d, c] * gain
        y = onepole_lowpass(y, 5200)
        out[:, c] = st[:, c] * (1 - mix) + y * mix
    return out.astype(np.float32)


def mono_to_stereo(sig: np.ndarray, pan: float = 0.0) -> np.ndarray:
    l = math.sqrt((1 - pan) / 2)
    r = math.sqrt((1 + pan) / 2)
    return np.stack([sig * l, sig * r], axis=1).astype(np.float32)


def make_banjo_pluck(freq: float,
                     dur: float = 0.75,
                     brightness: float = 0.72,
                     darkness: float = 0.28,
                     pick: float = 0.08,
                     electric: float = 0.35,
                     body_mix: float = 0.18) -> np.ndarray:
    """
    Dark electric-banjo-ish pluck:
    - Karplus-Strong plucked string core
    - pick transient
    - body resonances
    - subtle electric edge from clipped upper harmonic layer
    """
    n = int(dur * SR)
    delay_len = max(2, int(SR / max(freq, 30.0)))

    buf = (np.random.rand(delay_len).astype(np.float32) * 2 - 1)
    buf -= np.mean(buf)
    buf *= np.hanning(delay_len).astype(np.float32)
    if np.max(np.abs(buf)) > 0:
        buf /= np.max(np.abs(buf))

    out = np.zeros(n, dtype=np.float32)
    idx = 0
    damping = 0.986 - (0.03 * darkness)
    tone = 0.48 + 0.42 * brightness
    last = 0.0

    for i in range(n):
        cur = buf[idx]
        nxt = buf[(idx + 1) % delay_len]
        new = damping * (tone * 0.5 * (cur + nxt) + (1 - tone) * last)
        out[i] = cur
        buf[idx] = new
        last = new
        idx = (idx + 1) % delay_len

    t = np.arange(n, dtype=np.float32) / SR
    ring = (
        0.08 * np.sin(2 * np.pi * freq * 2.0 * t) +
        0.05 * np.sin(2 * np.pi * freq * 3.01 * t)
    )

    body = body_mix * (
        0.55 * np.sin(2 * np.pi * 220.0 * t) * np.exp(-t * 7.5) +
        0.35 * np.sin(2 * np.pi * 440.0 * t) * np.exp(-t * 9.0)
    )

    click = (np.random.randn(n).astype(np.float32) * np.exp(-t * 95) * pick)

    electric_layer = (
        0.14 * np.sin(2 * np.pi * freq * 4.0 * t + 0.2) +
        0.08 * np.sin(2 * np.pi * freq * 6.02 * t)
    )
    electric_layer = highpass(electric_layer, 700)
    electric_layer = softclip(electric_layer, 1.8) * electric

    sig = out + ring + body + click + electric_layer
    sig = highpass(sig, 45)
    sig = onepole_lowpass(sig, 4200 if brightness < 0.8 else 5600)

    dark_copy = onepole_lowpass(sig, 1800)
    sig = sig * (1 - darkness * 0.24) + dark_copy * (darkness * 0.18)

    env = np.exp(-t * (4.5 + darkness * 1.8)).astype(np.float32)
    sig *= env

    sig = softclip(sig, 1.15)
    return sig.astype(np.float32)


def add_event(buf: np.ndarray, start_s: float, audio: np.ndarray, pan: float = 0.0) -> None:
    start = int(start_s * SR)
    st = mono_to_stereo(audio, pan=pan) if audio.ndim == 1 else audio
    end = min(len(buf), start + len(st))
    if start < len(buf):
        buf[start:end] += st[:end - start]


def render_oneshot(note: str,
                   out_path: Path,
                   dur: float = 1.2,
                   brightness: float = 0.72,
                   darkness: float = 0.28,
                   electric: float = 0.35) -> None:
    midi = note_to_midi(note)
    freq = midi_to_freq(midi)
    mono = make_banjo_pluck(freq, dur=dur, brightness=brightness,
                            darkness=darkness, electric=electric)
    st = np.stack([mono, mono], axis=1)
    st = simple_reverb_stereo(st, mix=0.09)
    st = norm(st, 0.88)
    write_wav_float32(out_path, st)


def render_loop(events, bpm: float, out_path: Path, bars: int = 4,
                tonal_darkness: float = 0.26, reverb_mix: float = 0.14) -> None:
    beat_s = 60.0 / bpm
    total_s = bars * 4 * beat_s
    n = int(total_s * SR)
    st = np.zeros((n, 2), dtype=np.float32)

    for beat, midi, dur_beats, vel, pan, bright, dark, electric in events:
        dur_s = dur_beats * beat_s
        pluck = make_banjo_pluck(
            midi_to_freq(midi),
            dur=dur_s,
            brightness=bright,
            darkness=dark,
            electric=electric,
            pick=0.07 + 0.03 * vel,
        ) * vel
        add_event(st, beat * beat_s, pluck, pan=pan)

    st[:, 0] = highpass(st[:, 0], 45)
    st[:, 1] = highpass(st[:, 1], 45)
    st[:, 0] = onepole_lowpass(st[:, 0], 5300 - int(tonal_darkness * 1500))
    st[:, 1] = onepole_lowpass(st[:, 1], 5300 - int(tonal_darkness * 1500))

    st = simple_delay_stereo(st, ms_l=95, ms_r=140, fb=0.16, mix=0.12)
    st = simple_reverb_stereo(st, mix=reverb_mix)

    fade = 256
    a = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
    st[:fade] = st[:fade] * a + st[-fade:] * (1 - a)
    st[-fade:] = st[:fade]

    st = norm(st, 0.90)
    write_wav_float32(out_path, st)


PRESETS = {
    "hollows_arp": {
        "bpm": 110,
        "events": [
            (i * 0.5, midi, 0.45, 0.86, -0.18 if i % 2 == 0 else 0.18, 0.72, 0.28, 0.34)
            for i, midi in enumerate([
                45, 52, 57, 60, 57, 52, 45, 52,
                43, 50, 55, 58, 55, 50, 43, 50,
                40, 47, 52, 55, 52, 47, 40, 47,
                43, 50, 55, 59, 55, 50, 43, 50,
            ])
        ],
    },
    "ritual_pulse": {
        "bpm": 120,
        "events": [
            (beat, midi, 0.78, 0.92 if beat % 4 == 0 else 0.76,
             -0.12 if i % 2 == 0 else 0.12, 0.58, 0.42, 0.26)
            for i, (beat, midi) in enumerate([
                (0, 45), (1, 45), (2, 48), (3, 50),
                (4, 45), (5, 43), (6, 48), (7, 50),
                (8, 40), (9, 40), (10, 43), (11, 47),
                (12, 45), (13, 43), (14, 48), (15, 50),
            ])
        ],
    },
    "veilglass_harmonics": {
        "bpm": 140,
        "events": [
            (i * 0.5, midi, 0.35, 0.68, -0.22 if i % 2 else 0.22, 0.78, 0.22, 0.42)
            for i, midi in enumerate([
                69, 72, 76, 72, 67, 71, 74, 71,
                64, 67, 71, 67, 62, 66, 69, 66,
                69, 72, 76, 79, 71, 74, 78, 74,
                69, 72, 76, 72, 67, 71, 74, 71,
            ])
        ],
    },
}


def parse_args():
    ap = argparse.ArgumentParser(description="Render dark electric banjo one-shots or loops.")
    ap.add_argument("--mode", choices=["oneshot", "loop"], required=True)
    ap.add_argument("--out", required=True, help="Output WAV path")
    ap.add_argument("--note", default="C4", help="For oneshot mode")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="hollows_arp", help="For loop mode")
    ap.add_argument("--brightness", type=float, default=0.72)
    ap.add_argument("--darkness", type=float, default=0.28)
    ap.add_argument("--electric", type=float, default=0.35)
    ap.add_argument("--duration", type=float, default=1.2, help="oneshot duration in seconds")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "oneshot":
        render_oneshot(
            note=args.note,
            out_path=out_path,
            dur=args.duration,
            brightness=args.brightness,
            darkness=args.darkness,
            electric=args.electric,
        )
    else:
        preset = PRESETS[args.preset]
        render_loop(
            events=preset["events"],
            bpm=preset["bpm"],
            out_path=out_path,
        )

    print(f"Rendered: {out_path}")


if __name__ == "__main__":
    main()
