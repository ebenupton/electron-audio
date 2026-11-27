#!/usr/bin/env python3
"""Check an SDMLP emulator log (clkh --soundlog): every &FE06 write 32 cycles
after the last, and the ULA (grid model) playing back the intended stream.
  sdmcheck.py [LOG]"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/sdm/emu.log")
ev = [l.split() for l in open(log)]
t = np.array([int(a) for a, r, v in ev if r == "6"])
v = np.array([int(x, 16) for a, r, x in ev if r == "6"])
s0 = np.flatnonzero(np.diff(t) > 1000)[-1] + 1
t, v = t[s0:], v[s0:]
d = np.diff(t)
lay = json.load(open(os.path.join(ROOT, "build/sdm/layout.json")))
blk_reg = lay["per"] // 1
bad = np.flatnonzero(d != 32)
print("playback: %d writes (%.2f s); intervals not 32: %d" % (len(t), len(t) * 16e-6, len(bad)))
for b in bad[:20]:
    i = b + 1
    blk = i // 8
    print("  %d cycles before write %d: block %d (byte %d of the stream), tick %d" % (d[b], i, blk, blk + 2, i % 8 + 1))
# ULA: counter ticks every 32 cycles at phase ph; at expiry, reload with the
# latest S written before that edge, and toggle.  Compare the levels with the stream.
bits = np.load(os.path.join(ROOT, "build/sdm/bits.npy")).astype(int)
nb = len(bits)
for ph in (0, 8, 16, 24):
    S = 0
    cnt = 1
    lvl = 0
    out = np.zeros(len(t), int)
    j = 0
    edges = t[0] - 16 + ph + 32 * np.arange(len(t))
    k = 0
    for i, e in enumerate(edges):
        while k < len(t) and t[k] < e:
            S = v[k]
            k += 1
        out[i] = lvl
        cnt -= 1
        if cnt == 0:
            lvl ^= 1
            cnt = S + 1
    # align with the stream: out[i] ~ bits[i + off] (either polarity)
    ref = np.tile(bits, 4)
    best = None
    for off in range(-40, 40):
        o = out[300:20300]
        r = ref[300 + off:20300 + off]
        for inv in (0, 1):
            mm = np.mean((o ^ inv) == r)
            if best is None or mm > best[0]:
                best = (mm, off, inv)
    m, off, inv = best
    n = min(len(out) - 300, len(ref) - 300 - off)
    full = np.mean((out[300:300 + n] ^ inv) == ref[300 + off:300 + off + n])
    print("phase %2d: match %.5f over the first 20000 ticks (offset %d, inverted %d); whole run %.6f" % (ph, m, off, inv, full))
