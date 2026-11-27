#!/usr/bin/env python3
"""Psychoacoustic noise shaping for the 1-bit SDMLP stream.

A masking model (Johnston/MPEG-1 style) runs on the music as the Electron's
speaker radiates it, frame by frame (N = 2048 at 62.5 kHz, hop 512 = 8.2
ms): critical-band (Bark) energies, Schroeder spreading, a tonality-dependent
offset from the spectral flatness, the absolute threshold of hearing at an
assumed playback level, and temporal (post/pre) masking.  The result is a
masking threshold T(f) for each frame.

Each frame's noise weight is speaker(f) / T(f): noise as radiated, against
what would be masked.  A per-frame NTF A(z) is the prediction-error filter of
that weight (as sdmhq.ntf_weighted2), its out-of-band gain held to HMAX for
the modulator's stability; sdmhq.modulate_m_tv runs the lookahead modulator
switching NTF each frame.

  thresholds(x) -> (T: frames x bins, freqs)
  ntfs(T, f, ...) -> (frames x L+1)
  nmr(bits, x, T, f) -> noise-to-mask statistics
"""
import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz

FS = 62500.0
N = 2048
HOP = 512
FULL_SPL = 75.0        # assumed SPL of the full-scale 1-bit square wave at the listener


def bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def ath_db(f):
    """Threshold of hearing, dB SPL (Terhardt)."""
    k = np.maximum(f, 20.0) / 1000.0
    return 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4


def frames(x):
    n = 1 + (len(x) - N) // HOP
    idx = np.arange(N)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(N)[None, :]


def thresholds(x, speaker_db, full_spl=FULL_SPL, tonal=None):
    """x: the modulator input (0..1, 62.5 kHz).  Returns T (frames x bins,
    power, in the same units as |FFT|^2 of the signal's deviation) and the
    bin frequencies."""
    d = x - 0.5
    F = np.fft.rfft(frames(d), axis=1)
    f = np.fft.rfftfreq(N, 1 / FS)
    spk = 10 ** (speaker_db(f) / 10)
    P = np.abs(F) ** 2 * spk[None, :]                  # radiated music power per bin
    z = bark(f)
    zb = np.arange(0, 25.01, 0.25)                    # quarter-Bark bands
    bi = np.clip(np.searchsorted(zb, z) - 1, 0, len(zb) - 1)
    nb = len(zb)
    E = np.zeros((P.shape[0], nb))
    for b in range(nb):
        m = bi == b
        if m.any():
            E[:, b] = P[:, m].sum(1)
    # spreading (Schroeder), in quarter-Bark steps
    dz = (np.arange(nb)[:, None] - np.arange(nb)[None, :]) * 0.25    # maskee - masker
    sf = 15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1 + (dz + 0.474) ** 2)
    S = 10 ** (sf / 10)                                # (maskee, masker)
    C = E @ S.T
    # tonality from the spectral flatness of the radiated spectrum, per frame
    # over 300 Hz - 8 kHz (where the music lives)
    band = (f > 300) & (f < 8000)
    Pb = P[:, band] + 1e-20
    sfm = 10 * np.log10(np.exp(np.mean(np.log(Pb), 1)) / np.mean(Pb, 1))
    alpha = np.clip(sfm / -60.0, 0, 1)[:, None]
    if tonal is not None:                 # force the tonality (1: every masker tonal: the least masking)
        alpha = np.full_like(alpha, tonal)
    off = alpha * (14.5 + zb[None, :]) + (1 - alpha) * 5.5
    Tb = C * 10 ** (-off / 10)
    # renormalise for the spreading's energy gain (so a flat noise masks ~itself - offset)
    Tb = Tb / S.sum(1)[None, :] * 1.0
    # back to bins, per bin (divide the band threshold among its bins)
    cnt = np.bincount(bi, minlength=nb).astype(float)
    T = Tb[:, bi] / np.maximum(cnt[bi], 1)[None, :]
    # absolute threshold: full-scale square wave (deviation +-0.5, power 0.25)
    # = full_spl; a bin's power p (of N-point Hann frames) -> SPL
    ref = 0.25 * N * np.sum(np.hanning(N) ** 2)        # total |F|^2 of the full-scale square
    ath = ref * 10 ** ((ath_db(f) - full_spl) / 10) / (N / 2)   # per bin
    T = T + ath[None, :] * spk[None, :] * 0 + ath[None, :]
    # temporal masking: post (decay 1.5 dB a hop ~ 0.18 dB/ms), pre (one hop, -10 dB)
    Tt = T.copy()
    dec = 10 ** (-1.5 / 10)
    for i in range(1, len(Tt)):
        Tt[i] = np.maximum(Tt[i], Tt[i - 1] * dec)
    Tt[:-1] = np.maximum(Tt[:-1], T[1:] * 0.1)
    return Tt, f


def ntfs(T, f, speaker_db, L=16, lam=1e-3, hmax=2.0):
    """Per-frame NTF from the weight speaker/T.  lam (relative floor) is
    raised per frame until max|NTF| <= hmax."""
    spk = 10 ** (speaker_db(f) / 10)
    out = np.zeros((T.shape[0], L + 1))
    w = np.linspace(0, np.pi, 512)
    E = np.exp(-1j * np.outer(w, np.arange(L + 1)))
    for i in range(T.shape[0]):
        W = spk / T[i]
        W = W / W.max()
        lm = lam
        for _ in range(30):
            Wl = W + lm
            r = np.fft.irfft(Wl)[:L + 1]
            a = solve_toeplitz(r[:L], -r[1:L + 1])
            a = np.concatenate([[1.0], a])
            if np.abs(E @ a).max() <= hmax:
                break
            lm *= 1.6
        out[i] = a
    return out


def ntfs_loud(T, f, speaker_db, L=16, lam=1e-4, hmax=2.0, alpha=0.3, iters=4, sig2=None, wstatic=None, blend=1.0):
    """Per-frame NTF minimising the partial loudness of the unmasked noise,
    sum_f (N(f) spk(f) / T(f))^alpha (alpha ~0.3: loudness is compressive,
    so a little noise above the mask in many places is worse than more in
    a few), by iteratively reweighted least squares from ntfs()'s design."""
    spk = 10 ** (speaker_db(f) / 10)
    out = np.zeros((T.shape[0], L + 1))
    w = np.linspace(0, np.pi, 512)
    E = np.exp(-1j * np.outer(w, np.arange(L + 1)))
    wf = np.linspace(0, np.pi, len(f))
    Ef = np.exp(-1j * np.outer(wf, np.arange(L + 1)))
    if sig2 is None:
        sig2 = 0.08 * N * np.sum(np.hanning(N) ** 2) / (N / 2) / N   # 1-bit error power per bin (approx.)
    for i in range(T.shape[0]):
        W0 = spk / T[i]
        if wstatic is not None and blend < 1.0:     # geometric blend with a static weighting
            W0 = (wstatic / wstatic.max()) ** (1 - blend) * (W0 / W0.max()) ** blend
        Wk = W0.copy()
        a = None
        for it in range(iters + 1):
            Wn = Wk / Wk.max()
            lm = lam
            for _ in range(30):
                r = np.fft.irfft(Wn + lm)[:L + 1]
                aa = solve_toeplitz(r[:L], -r[1:L + 1])
                aa = np.concatenate([[1.0], aa])
                if np.abs(E @ aa).max() <= hmax:
                    break
                lm *= 1.6
            a = aa
            if it == iters:
                break
            Nf = np.abs(Ef @ a) ** 2 * sig2
            rr = np.maximum(Nf * W0, 1e-6)
            Wk = W0 * rr ** (alpha - 1)
        out[i] = a
    return out


def nmr(bits, x, T, f, speaker_db, alpha=0.3):
    """Noise-to-mask: per frame and quarter-Bark band, the radiated noise
    power over the threshold.  Returns (mean NMR dB over bands and frames,
    fraction of (frame, band) cells with noise above the mask, the worst 5%
    mean)."""
    e = bits - x
    Fe = np.fft.rfft(frames(e), axis=1)
    spk = 10 ** (speaker_db(f) / 10)
    Pn = np.abs(Fe) ** 2 * spk[None, :]
    n = min(len(Pn), len(T))
    z = bark(f)
    zb = np.arange(0, 25.01, 0.5)
    bi = np.clip(np.searchsorted(zb, z) - 1, 0, len(zb) - 1)
    r = np.zeros((n, len(zb)))
    for b in range(len(zb)):
        m = bi == b
        if m.any():
            r[:, b] = Pn[:n, m].sum(1) / T[:n, m].sum(1)
    db = 10 * np.log10(r + 1e-12)
    # partial loudness of the unmasked noise (relative units): sum of r^alpha over cells above the mask
    pl = np.mean(np.where(r > 1, r ** alpha - 1, 0).sum(1))
    return db.mean(), np.mean(db > 0), np.mean(np.sort(db.ravel())[-len(db.ravel()) // 20:]), pl


def true_peak_limit(y, fs=FS, thr_db=-6.0, look_ms=2.0, rel_ms=120.0):
    """Never exceeds thr (re the peak): instant attack on the lookahead
    minimum, slow release, then a centred moving average no longer than the
    lookahead (which keeps it under the required gain).  Returns y scaled so
    the limit is 1."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    pk = np.abs(y).max()
    thr = pk * 10 ** (thr_db / 20)
    K = int(look_ms * fs / 1000)
    greq = np.minimum(1.0, thr / np.maximum(np.abs(y), 1e-12))
    g1 = minimum_filter1d(greq, 2 * K + 1)
    r = np.exp(-1 / (rel_ms * fs / 1000))
    g2 = _release(g1, r)
    g3 = uniform_filter1d(g2, 2 * K + 1)
    g3 = np.minimum(g3, greq)          # (guards the ends)
    return y * g3 / thr


def _release(g, r):
    from numba import njit

    @njit(cache=True)
    def run(g, r):
        out = np.empty_like(g)
        s = 1.0
        for i in range(len(g)):
            s = min(g[i], r * s + (1 - r) * 1.0)
            out[i] = s
        return out
    return run(g, r)


def compress(y, fs=FS, thr_db=-18.0, ratio=3.0, att_ms=10.0, rel_ms=200.0, makeup=True):
    """RMS compressor (feed-forward, smoothed level in dB): above thr (re the
    signal's peak), the level rises at 1/ratio.  Returns y with the gain
    applied (unnormalised; the true-peak limiter follows)."""
    from numba import njit

    @njit(cache=True)
    def env(p, a, r):
        out = np.empty_like(p)
        s = p[0]
        for i in range(len(p)):
            c = a if p[i] > s else r
            s = c * s + (1 - c) * p[i]
            out[i] = s
        return out
    pk = np.abs(y).max()
    a = np.exp(-1 / (att_ms * fs / 1000))
    r = np.exp(-1 / (rel_ms * fs / 1000))
    lev = 10 * np.log10(env(y * y, a, r) + 1e-12) - 20 * np.log10(pk)
    over = np.maximum(lev - thr_db, 0)
    gdb = -over * (1 - 1 / ratio)
    return y * 10 ** (gdb / 20)


def dynamics(y, comp_thr=None, ratio=3.0, drive_db=0.0, look_ms=2.0, rel_ms=120.0, att_ms=10.0, crel_ms=200.0):
    """Optional RMS compressor, then drive (dB re: the 99.9th percentile
    sitting at full scale) into a true-peak limiter with its ceiling at 1."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    s = y if comp_thr is None else compress(y, thr_db=comp_thr, ratio=ratio, att_ms=att_ms, rel_ms=crel_ms)
    s = s / np.percentile(np.abs(s), 99.9) * 10 ** (drive_db / 20)
    K = int(look_ms * FS / 1000)
    greq = np.minimum(1.0, 1.0 / np.maximum(np.abs(s), 1e-12))
    g1 = minimum_filter1d(greq, 2 * K + 1)
    g2 = _release(g1, np.exp(-1 / (rel_ms * FS / 1000)))
    g3 = np.minimum(uniform_filter1d(g2, 2 * K + 1), greq)
    return s * g3
