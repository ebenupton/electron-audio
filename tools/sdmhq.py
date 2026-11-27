#!/usr/bin/env python3
"""Offline high-order 1-bit sigma-delta at the ULA's tick rate (62.5 kHz).

NTF = B(z)/A(z): zeros on the unit circle in band (optimal spread), poles of a
Butterworth highpass, scaled so h[0] = 1; error-feedback form.
  ntf(order, band_hz, hinf) -> (b, a)
  modulate(x, b, a) -> bits (0/1), x in 0..1
"""
import numpy as np
from numba import njit
from scipy import signal

FS = 62500.0
ZEROS = {1: [0.0], 2: [0.577], 3: [0.0, 0.775], 4: [0.34, 0.861], 5: [0.0, 0.539, 0.906],
         6: [0.239, 0.664, 0.935]}


def ntf(order, band, hinf=1.5):
    zs = []
    for z in ZEROS[order]:
        w = np.pi * z * band / (FS / 2)
        zs += [np.exp(1j * w)] if z == 0 else [np.exp(1j * w), np.exp(-1j * w)]
    b = np.real(np.poly(zs))
    lo, hi = 0.001, 0.99
    for _ in range(60):
        fc = (lo + hi) / 2
        _, p, _ = signal.butter(order, fc, "high", output="zpk")
        a = np.real(np.poly(p))
        w, h = signal.freqz(b, a, 2048)
        if np.max(np.abs(h)) > hinf:
            hi = fc
        else:
            lo = fc
    _, p, _ = signal.butter(order, lo, "high", output="zpk")
    return b, np.real(np.poly(p))


@njit(cache=True)
def _mod(x, c, a):
    # f[n] = sum_k c[k] e[n-k] - sum_k a[k] f[n-k], k >= 1;  w = x + f;  y = Q(w); e = y - w
    n = len(x)
    m = len(c)
    eh = np.zeros(m)
    fh = np.zeros(m)
    y = np.zeros(n)
    bad = 0
    for i in range(n):
        f = 0.0
        for k in range(1, m):
            f += c[k] * eh[k - 1] - a[k] * fh[k - 1]
        w = x[i] + f
        v = 1.0 if w >= 0.5 else 0.0
        e = v - w
        if abs(e) > 20:
            bad += 1
            e = 0.0
            f = 0.0
            for k in range(m):
                eh[k] = 0.0
                fh[k] = 0.0
        for k in range(m - 1, 0, -1):
            eh[k] = eh[k - 1]
            fh[k] = fh[k - 1]
        eh[0] = e
        fh[0] = f
        y[i] = v
    return y, bad


def modulate(x, b, a):
    m = max(len(b), len(a))
    b = np.concatenate([b, np.zeros(m - len(b))])
    a = np.concatenate([a, np.zeros(m - len(a))])
    c = b - a            # (B - A): c[0] = 0
    y, bad = _mod(np.ascontiguousarray(x, float), np.ascontiguousarray(c, float), np.ascontiguousarray(a, float))
    return y, bad


def snr(y, x, lo=100, hi=4000):
    Y = np.fft.rfft(y - y.mean())
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(y), 1 / FS)
    m = (f > lo) & (f < hi)
    return 10 * np.log10(np.sum(np.abs(X[m]) ** 2) / np.sum(np.abs(Y[m] - X[m]) ** 2))


# dB re 500 Hz.  Up to 6 kHz from SDMTEST's lines; above, from the hiss of the
# first SDMLP recording (the modulator's known noise as the probe), anchored
# at 5-6 kHz; 15-18 kHz assumed (the room has a 15.6 kHz line there).
RESP2 = [(450, 0.0), (800, -2.5), (1250, -4.6), (1750, -3.9), (2250, -2.0), (2750, -0.3), (3250, 4.5), (3750, 7.8),
         (4500, -1.6), (5500, -7.1), (6500, -14.4), (7500, -11.9), (9000, -20.4), (11000, -22.4), (13500, -23.2),
         (16500, -30.0), (20000, -51.0), (26000, -60.0), (31250, -65.0)]


def speaker_db2(f, lo_db=-20.0):
    ff, vv = zip(*RESP2)
    d = np.interp(f, ff, vv)
    # below 300 Hz: a small speaker's roll-off (2nd order at ~300 Hz)
    return np.where(f < 450, d + 40 * np.log10(np.maximum(f, 20) / 450) * (f < 450), d)


def weight2(f):
    return 10 ** ((speaker_db2(f) + aweight_db(f)) / 10)


def ntf_weighted2(L, lam, n=8192):
    from scipy.linalg import solve_toeplitz
    f = np.linspace(0, FS / 2, n + 1)
    W = weight2(f)
    W = W / W.max() + lam
    r = np.fft.irfft(W)[:L + 1]
    a = solve_toeplitz(r[:L], -r[1:L + 1])
    return np.concatenate([[1.0], a]), np.array([1.0])


def wsnr2(y, x, lo=50):
    Y = np.fft.rfft(y - y.mean())
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(y), 1 / FS)
    W = weight2(f) * (f > lo)
    return 10 * np.log10(np.sum(W * np.abs(X) ** 2) / np.sum(W * np.abs(Y - X) ** 2))


def speaker_db(f, floor_db=-30.0):
    """Measured Electron speaker + PodMic response (data/speaker_resp.npy, from
    the SDMTEST recording), dB re 500 Hz, smoothed; HF attenuation capped."""
    import os
    r = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "speaker_resp.npy"))
    ref = np.median(r[(r[:, 0] > 400) & (r[:, 0] < 600), 1])
    edges = np.array([300, 600, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 17000])
    cf, cv = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.sum():
            cf.append((lo + hi) / 2)
            cv.append(np.median(r[m, 1]) - ref)
    d = np.interp(f, cf, cv, left=cv[0], right=cv[-1] - 6)
    return np.maximum(d, floor_db)


def aweight_db(f):
    f = np.maximum(f, 1.0)
    ra = (12194.0 ** 2 * f ** 4) / ((f ** 2 + 20.6 ** 2) * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2)) * (f ** 2 + 12194.0 ** 2))
    return 20 * np.log10(ra) + 2.0


def weight(f, floor_db=-30.0):
    """Perceived-noise weight (power): speaker x A-weighting."""
    return 10 ** ((speaker_db(f, floor_db) + aweight_db(f)) / 10)


def ntf_weighted(L, lam, floor_db=-30.0, n=8192):
    """FIR NTF (h[0] = 1) minimising the weighted noise: the prediction-error
    filter of the spectrum W + lam (lam limits the out-of-band gain)."""
    from scipy.linalg import solve_toeplitz
    f = np.linspace(0, FS / 2, n + 1)
    W = weight(f, floor_db)
    W = W / W.max() + lam
    r = np.fft.irfft(W)[:L + 1]
    a = solve_toeplitz(r[:L], -r[1:L + 1])
    return np.concatenate([[1.0], a]), np.array([1.0])


def wsnr(y, x, floor_db=-30.0, lo=50):
    """Weighted SNR: perceived (speaker x A) power of the signal over that of the error."""
    Y = np.fft.rfft(y - y.mean())
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(y), 1 / FS)
    W = weight(f, floor_db) * (f > lo)
    return 10 * np.log10(np.sum(W * np.abs(X) ** 2) / np.sum(W * np.abs(Y - X) ** 2))


@njit(cache=True)
def _modr(x, c, a, rmax):
    # as _mod, but no run longer than rmax ticks (forced toggle)
    n = len(x)
    m = len(c)
    eh = np.zeros(m)
    fh = np.zeros(m)
    y = np.zeros(n)
    bad = 0
    run = 0
    last = -1.0
    for i in range(n):
        f = 0.0
        for k in range(1, m):
            f += c[k] * eh[k - 1] - a[k] * fh[k - 1]
        w = x[i] + f
        v = 1.0 if w >= 0.5 else 0.0
        if v == last and run >= rmax:
            v = 1.0 - v
        if v == last:
            run += 1
        else:
            run = 1
        last = v
        e = v - w
        if abs(e) > 20:
            bad += 1
            e = 0.0
            f = 0.0
            for k in range(m):
                eh[k] = 0.0
                fh[k] = 0.0
        for k in range(m - 1, 0, -1):
            eh[k] = eh[k - 1]
            fh[k] = fh[k - 1]
        eh[0] = e
        fh[0] = f
        y[i] = v
    return y, bad


def modulate_r(x, b, a, rmax):
    m = max(len(b), len(a))
    b = np.concatenate([b, np.zeros(m - len(b))])
    a = np.concatenate([a, np.zeros(m - len(a))])
    return _modr(np.ascontiguousarray(x, float), np.ascontiguousarray(b - a, float), np.ascontiguousarray(a, float), rmax)


def limit(y, fs=FS, thr_db=-6.0, att_ms=2.0, rel_ms=150.0, look_ms=3.0):
    """Look-ahead peak limiter: returns y scaled so peaks sit at 1, with
    everything above thr (re the original peak) squashed."""
    pk = np.abs(y).max()
    thr = pk * 10 ** (thr_db / 20)
    look = int(look_ms * fs / 1000)
    from scipy.ndimage import maximum_filter1d
    env = maximum_filter1d(np.abs(y), 2 * look + 1)
    g = np.minimum(1.0, thr / np.maximum(env, 1e-9))
    # smooth: fast attack, slow release (one-pole on the gain, min-tracking)
    ra = np.exp(-1 / (att_ms * fs / 1000))
    rr = np.exp(-1 / (rel_ms * fs / 1000))
    gs = _smooth(g, ra, rr)
    return y * gs / thr


@njit(cache=True)
def _smooth(g, ra, rr):
    out = np.empty_like(g)
    s = 1.0
    for i in range(len(g)):
        t = g[i]
        if t < s:
            s = ra * s + (1 - ra) * t
        else:
            s = rr * s + (1 - rr) * t
        out[i] = min(s, t) if False else s
    return out


@njit(cache=True)
def _modm(x, a, M, rmax):
    """M-algorithm lookahead: keep the M lowest-cost paths; a path's cost is
    the running power of the error filtered by 1/A(z) (the perceptual
    weighting the NTF A(z) was designed from).  M = 1 is the plain
    error-feedback modulator.  Runs capped at rmax."""
    n = len(x)
    L = len(a) - 1
    st = np.zeros((M, L))          # past filtered errors, newest first
    cost = np.full(M, 1e30)
    cost[0] = 0.0
    last = np.zeros(M, np.int64)
    run = np.zeros(M, np.int64)
    bp = np.zeros((n, M), np.int16)     # backpointer * 2 + bit
    nst = np.zeros((2 * M, L))
    ncost = np.zeros(2 * M)
    nlast = np.zeros(2 * M, np.int64)
    nrun = np.zeros(2 * M, np.int64)
    for i in range(n):
        for p in range(M):
            fb = 0.0
            for k in range(L):
                fb += a[k + 1] * st[p, k]
            for b in range(2):
                c = 2 * p + b
                if cost[p] >= 1e29 or (b == last[p] and run[p] >= rmax):
                    ncost[c] = 1e30
                    continue
                e = (b - x[i]) - fb
                ncost[c] = cost[p] + e * e
                nst[c, 0] = e
                for k in range(1, L):
                    nst[c, k] = st[p, k - 1]
                nlast[c] = b
                nrun[c] = run[p] + 1 if b == last[p] else 1
        order = np.argsort(ncost)
        base = ncost[order[0]]
        for j in range(M):
            c = order[j]
            cost[j] = ncost[c] - base if ncost[c] < 1e29 else 1e30
            st[j, :] = nst[c, :]
            last[j] = nlast[c]
            run[j] = nrun[c]
            bp[i, j] = (c // 2) * 2 + (c % 2)
    y = np.zeros(n)
    j = 0
    for i in range(n - 1, -1, -1):
        v = bp[i, j]
        y[i] = v % 2
        j = v // 2
    return y


def modulate_m(x, a, M=32, rmax=8):
    return _modm(np.ascontiguousarray(x, float), np.ascontiguousarray(a, float), M, rmax)


@njit(cache=True)
def _modm_tv(x, A, hop, M, rmax):
    """As _modm, the NTF switching every hop samples: A[k] for samples
    k*hop .. (k+1)*hop-1 (offset by half a frame: frame k is centred there)."""
    n = len(x)
    L = A.shape[1] - 1
    st = np.zeros((M, L))
    cost = np.full(M, 1e30)
    cost[0] = 0.0
    last = np.zeros(M, np.int64)
    run = np.zeros(M, np.int64)
    bp = np.zeros((n, M), np.int16)
    nst = np.zeros((2 * M, L))
    ncost = np.zeros(2 * M)
    nlast = np.zeros(2 * M, np.int64)
    nrun = np.zeros(2 * M, np.int64)
    nf = A.shape[0]
    for i in range(n):
        k = min(max((i - 768) // hop, 0), nf - 1)
        a = A[k]
        for p in range(M):
            fb = 0.0
            for j in range(L):
                fb += a[j + 1] * st[p, j]
            for b in range(2):
                c = 2 * p + b
                if cost[p] >= 1e29 or (b == last[p] and run[p] >= rmax):
                    ncost[c] = 1e30
                    continue
                e = (b - x[i]) - fb
                ncost[c] = cost[p] + e * e
                nst[c, 0] = e
                for j in range(1, L):
                    nst[c, j] = st[p, j - 1]
                nlast[c] = b
                nrun[c] = run[p] + 1 if b == last[p] else 1
        order = np.argsort(ncost)
        base = ncost[order[0]]
        for j in range(M):
            c = order[j]
            cost[j] = ncost[c] - base if ncost[c] < 1e29 else 1e30
            st[j, :] = nst[c, :]
            last[j] = nlast[c]
            run[j] = nrun[c]
            bp[i, j] = c
    y = np.zeros(n)
    j = 0
    for i in range(n - 1, -1, -1):
        v = bp[i, j]
        y[i] = v % 2
        j = v // 2
    return y


def modulate_m_tv(x, A, hop=512, M=32, rmax=8):
    return _modm_tv(np.ascontiguousarray(x, float), np.ascontiguousarray(A, float), hop, M, rmax)


def speaker_db3(f):
    """From the SPKCHAR white-noise recording (data/speaker_resp2.npy):
    |H| in dB re 1-2 kHz, smoothed over 1/6 octave where the coherence is
    above 0.5; below 450 Hz a 12 dB/octave roll-off from the measured -14 dB
    at 375 Hz; above 16 kHz (hidden by the mic's own cut-off) -30 dB."""
    import os
    global _SP3
    try:
        _SP3
    except NameError:
        r = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "speaker_resp2.npy"))
        fw, dbw, cw = r[0], r[1], r[2]
        ref = np.mean(dbw[(fw > 1000) & (fw < 2000)])
        cf = np.geomspace(450, 16000, 60)
        cv = []
        for c in cf:
            m = (fw > c / 2 ** (1 / 12)) & (fw < c * 2 ** (1 / 12)) & (cw > 0.5)
            cv.append(10 * np.log10(np.mean(10 ** ((dbw[m] - ref) / 10))) if m.any() else np.nan)
        cv = np.array(cv)
        ok = ~np.isnan(cv)
        _SP3 = (cf[ok], cv[ok])
    cf, cv = _SP3
    f = np.asarray(f, float)
    d = np.interp(f, cf, cv)
    d = np.where(f < 450, cv[0] - 12 * np.log2(450 / np.maximum(f, 20)) - 0, d)
    d = np.where(f > 16000, -30.0, d)
    return d


def weight3(f):
    return 10 ** ((speaker_db3(f) + aweight_db(f)) / 10)


@njit(cache=True)
def _modvq(x, a, cb, M, rmax):
    """Block-constrained modulator: every 8-tick block is one of the codebook
    patterns cb (bytes, MSB first in time); M-path search over blocks on the
    error filtered by 1/A(z), runs capped at rmax across blocks.  Returns the
    chosen codeword index per block."""
    nb = len(x) // 8
    L = len(a) - 1
    K = len(cb)
    pat = np.zeros((K, 8))
    for k in range(K):
        for t in range(8):
            pat[k, t] = (cb[k] >> (7 - t)) & 1
    st = np.zeros((M, L))
    cost = np.full(M, 1e30)
    cost[0] = 0.0
    last = np.zeros(M, np.int64)
    run = np.zeros(M, np.int64)
    bp = np.zeros((nb, M), np.int32)
    C = M * K
    nst = np.zeros((C, L))
    ncost = np.zeros(C)
    nlast = np.zeros(C, np.int64)
    nrun = np.zeros(C, np.int64)
    tmp = np.zeros(L)
    for i in range(nb):
        for p in range(M):
            for k in range(K):
                c = p * K + k
                if cost[p] >= 1e29:
                    ncost[c] = 1e30
                    continue
                for j in range(L):
                    tmp[j] = st[p, j]
                cc = cost[p]
                l = last[p]
                r = run[p]
                ok = True
                for t in range(8):
                    b = pat[k, t]
                    bi = int(b)
                    if bi == l:
                        r += 1
                        if r > rmax:
                            ok = False
                            break
                    else:
                        r = 1
                        l = bi
                    fb = 0.0
                    for j in range(L):
                        fb += a[j + 1] * tmp[j]
                    e = (b - x[i * 8 + t]) - fb
                    cc += e * e
                    for j in range(L - 1, 0, -1):
                        tmp[j] = tmp[j - 1]
                    tmp[0] = e
                if not ok:
                    ncost[c] = 1e30
                    continue
                ncost[c] = cc
                for j in range(L):
                    nst[c, j] = tmp[j]
                nlast[c] = l
                nrun[c] = r
        order = np.argsort(ncost)
        base = ncost[order[0]]
        for j in range(M):
            c = order[j]
            cost[j] = ncost[c] - base if ncost[c] < 1e29 else 1e30
            st[j, :] = nst[c, :]
            last[j] = nlast[c]
            run[j] = nrun[c]
            bp[i, j] = c
    out = np.zeros(nb, np.int64)
    j = 0
    for i in range(nb - 1, -1, -1):
        c = bp[i, j]
        out[i] = c % K
        j = c // K
    return out


def modulate_vq(x, a, cb, M=8, rmax=8):
    idx = _modvq(np.ascontiguousarray(x, float), np.ascontiguousarray(a, float), np.asarray(cb, np.int64), M, rmax)
    bits = np.unpackbits(np.asarray(cb, np.uint8)[idx])
    return bits.astype(float), idx
