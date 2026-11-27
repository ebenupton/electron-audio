#!/usr/bin/env python3
"""Render a MOD (as modfile.py reads it) at a high rate, for offline modulators.

  render(path, fs=62500) -> float array, roughly -1..1 (4 channels summed / 4)
  modrender.py MOD [--fs 62500] [--out x.wav]
"""
import argparse, os, sys, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modfile import parse, linearise, PAULA


def render(path, fs=62500, order=None):
    samples, order0, pats = parse(path)
    order = order0 if order is None else order
    rows, speed = linearise(samples, order, pats)
    nrow = int(round(speed * 0.02 * fs))
    out = np.zeros(len(rows) * nrow)
    st = [None] * 4          # (sample, pos, inc, vol)
    for ri, r in enumerate(rows):
        for c, s, p, v, vo in r:
            if vo:
                if st[c] is not None:
                    st[c] = (st[c][0], st[c][1], st[c][2], v)
            else:
                st[c] = (s, 0.0, PAULA / p / fs, v)
        for c in range(4):
            if st[c] is None:
                continue
            s, pos, inc, vol = st[c]
            smp = samples[s]
            data = smp["data"].astype(float) / 128.0
            loop = smp["loop"]
            ps = pos + inc * np.arange(nrow)
            if loop:
                ls, ll = loop
                ps = np.where(ps >= ls + ll, ls + np.mod(ps - ls, ll), ps)
                end = ls + ll
            else:
                end = len(data)
            ok = ps < end - 1
            i = np.floor(ps).astype(int)
            i = np.clip(i, 0, len(data) - 2)
            fr = ps - np.floor(ps)
            val = (data[i] * (1 - fr) + data[i + 1] * fr) * ok
            out[ri * nrow:(ri + 1) * nrow] += val * vol / 64.0
            npos = pos + inc * nrow
            if loop and npos >= loop[0] + loop[1]:
                npos = loop[0] + np.mod(npos - loop[0], loop[1])
            st[c] = (s, npos, inc, vol) if (loop or npos < len(data)) else None
    return out / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mod")
    ap.add_argument("--fs", type=int, default=62500)
    ap.add_argument("--out")
    a = ap.parse_args()
    y = render(a.mod, a.fs)
    print("%.1f s, peak %.3f, rms %.4f" % (len(y) / a.fs, np.abs(y).max(), np.sqrt(np.mean(y ** 2))))
    if a.out:
        z = np.clip(y / np.abs(y).max() * 0.9, -1, 1)
        with wave.open(a.out, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(a.fs)
            wf.writeframes((z * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    main()
