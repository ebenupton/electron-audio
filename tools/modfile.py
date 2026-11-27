#!/usr/bin/env python3
"""Read a 4-channel Amiga MOD: its samples, and its rows in play order.

Only what Lost Patrol needs is understood: notes, sample numbers, set volume
(C), set speed (F), position jump (B) and pattern break (D).
"""
import numpy as np

PAULA = 3546895.0            # the PAL Amiga's clock: a note of period p plays its sample at PAULA / p Hz


def parse(path):
    d = open(path, "rb").read()
    m31 = d[1080:1084] in (b"M.K.", b"M!K!", b"FLT4", b"4CHN")
    ns = 31 if m31 else 15
    oh = 20 + 30 * ns
    L = d[oh]
    order = list(d[oh + 2:oh + 2 + L])
    npat = max(d[oh + 2:oh + 130]) + 1
    pat0 = oh + 130 + (4 if m31 else 0)
    samples = {}
    off = pat0 + npat * 1024
    for i in range(ns):
        h = d[20 + 30 * i:50 + 30 * i]
        ln = (h[22] << 8 | h[23]) * 2
        vol = h[25]
        rs = (h[26] << 8 | h[27])
        rl = (h[28] << 8 | h[29]) * 2
        rs = rs if not m31 else rs * 2          # 15-sample: repeat start in bytes
        data = np.frombuffer(d[off:off + ln], np.int8).astype(np.int16)
        off += ln
        if ln:
            loop = (rs, rl) if rl > 2 else None
            samples[i + 1] = {"data": data, "vol": vol, "loop": loop}
    pats = []
    for p in range(npat):
        rows = []
        for r in range(64):
            cells = []
            for c in range(4):
                b = d[pat0 + p * 1024 + r * 16 + c * 4:pat0 + p * 1024 + r * 16 + c * 4 + 4]
                cells.append(((b[0] & 0xF0) | (b[2] >> 4), ((b[0] & 15) << 8) | b[1], b[2] & 15, b[3]))
            rows.append(cells)
        pats.append(rows)
    return samples, order, pats


def linearise(samples, order, pats):
    """Rows in play order: [(c, sample, period, volume or None, volonly)]."""
    speed = 6
    out = []
    cur = [0, 0, 0, 0]
    pos = 0
    seen = set()
    while pos < len(order) and pos not in seen:
        seen.add(pos)
        jump = None
        for r in range(64):
            ev = []
            for c, (smp, per, cmd, par) in enumerate(pats[order[pos]][r]):
                if cmd == 15 and 0 < par < 32:
                    speed = par
                if smp:
                    cur[c] = smp
                vol = min(par, 64) if cmd == 12 else None
                if per and cur[c] in samples:
                    v = vol if vol is not None else samples[cur[c]]["vol"]
                    ev.append((c, cur[c], per, v, False))
                elif vol is not None:
                    ev.append((c, 0, 0, vol, True))
                if cmd == 11:
                    jump = par
                if cmd == 13:
                    jump = pos + 1
            out.append(ev)
            if jump is not None:
                break
        pos = pos + 1 if jump is None or jump <= pos else jump
    return out, speed
