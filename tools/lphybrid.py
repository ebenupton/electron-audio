#!/usr/bin/env python3
"""Lost Patrol, hybrid: the intro as a stored bit stream, the rest decoded as it plays.

  lphybrid.py [--row 63] [--nbank 6] [--check] [--secs N]

The first --row rows (thin, quiet: where the real-time player is weakest)
are computed offline as SDMLP 3c was (tools/sdmbuild.py --mode static
--dyn=-20,3,9: compressed, driven into a limiter, a 62.5 kHz 1-bit stream
with static noise shaping for the speaker and the ear) and played from --nbank
sideways RAM banks by SDMLP's player.  The rest is LPRT (tools/lprt.py,
lprtbuild.py) in one more bank, its samples in main RAM.  Both write &FE06
on the same 16 us grid, and each hands over to the other between two writes:

  bit stream -> tracker: after its last write the stream's player loads the
    tracker's first pattern into X and pages the tracker's bank; the code
    there (at the same address) is a frame that sets every channel, the
    song's place and the modulator from literals, then goes on as usual.
    The stream's last bits are that first pattern's, so its player's writes
    are already right.
  tracker -> bit stream (the song goes round): the EXIT record's frame makes
    the last pattern's writes and pages the stream's first bank, whose code
    makes the next write a tick later.

The low samples (&0400-&1DFF) are parked at the top of the last stream bank
while loading.
"""
import argparse, json, os, re, shutil, subprocess, sys
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import lprt, lprtbuild, sdmbuild
from lprt import Cfg, FS
from lprtbuild import Gen, Path, CODEPAGES, LOSPLIT, BEEBASM
from modfile import PAULA

OUT = os.path.join(ROOT, "build", "lph")
NBP = 57                     # pages of stream a bank
XSTUB = 0x8048               # the stream player's STY ULA_PG on the way out; the tracker's entry is 3 on
HOT = 0x8005                 # the stream player's way in (the tracker's LDY # : STY ULA_PG is at &8000)
PARKPAGES = (LOSPLIT - Cfg.main0) // 256
PARK = 0xC000 - 256 * PARKPAGES


# ------------------------------------------------------------------ the intro, exactly
def timeline(sg, nrows):
    """Per channel: the notes sounding through the first nrows rows as
    (first tick, last tick + 1, note, tick it began, volume); and what is
    sounding at the end."""
    rt = sg["rowticks"]
    segs = [[] for _ in range(4)]
    cur = [None] * 4

    def close(c, t):
        if cur[c]:
            n, t_on, v, t0 = cur[c]
            if t > t0:
                segs[c].append((t0, t, n, t_on, v))

    for r in range(nrows):
        t = r * rt
        for c, n, v in sg["rows"][r]:
            if n is None:
                if cur[c]:
                    close(c, t)
                    cur[c] = (cur[c][0], cur[c][1], v, t)
            elif n < 0:
                close(c, t)
                cur[c] = None
            else:
                close(c, t)
                cur[c] = (n, t, sg["notes"][n][2], t)
    for c in range(4):
        close(c, nrows * rt)
    return segs, cur


def position(sg, n, dt):
    """Where note n is, dt ticks after it began, in its original samples (None: it has run out)."""
    s, p, _ = sg["notes"][n]
    sm = sg["smp"][s]
    pos = dt * PAULA / p / FS
    if sm["loop"]:
        ls, ll = sm["oloop"]
        pos = np.where(pos >= ls + ll, ls + np.mod(pos - ls, ll), pos)
    return pos


def render(sg, nrows):
    """The first nrows rows mixed ideally at 62.5 kHz, in the units of the tracker's sum less its centre."""
    segs, _ = timeline(sg, nrows)
    out = np.zeros(nrows * sg["rowticks"])
    for c in range(4):
        for t0, t1, n, t_on, v in segs[c]:
            s, p, _ = sg["notes"][n]
            d = sg["smp"][s]["orig"]
            pos = position(sg, n, np.arange(t0, t1) - t_on)
            i0 = np.floor(pos).astype(int)
            fr = pos - i0
            ok = (i0 >= 0) & (i0 < len(d) - 1)
            i0 = np.clip(i0, 0, len(d) - 2)
            out[t0:t1] += (d[i0] * (1 - fr) + d[i0 + 1] * fr) * ok * v / 64.0 * 31.5 * (128 / 127.0)
    return out


def carry(sg, pl, row, tick):
    """The channels as they stand at tick (in row's first moments): (pointer, step, volume index, note)."""
    _, cur = timeline(sg, row)
    chan = []
    for c in range(4):
        st = (Cfg.main0 << 8, 0, 0, -1)
        if cur[c]:
            n, t_on, v, _ = cur[c]
            s, p, _ = sg["notes"][n]
            sm = sg["smp"][s]
            pos = float(position(sg, n, np.array([tick - t_on]))[0]) * sm["ratio"]
            if sm["loop"] or pos < sm["n"] - 2:
                st = ((sm["addr"] << 8) + int(pos * 256), lprt.step16(sg, Cfg, s, p), pl.vix[v], n)
        chan.append(st)
    return chan


def target(sg):
    """The whole song as SDMLP 3c had it (sdmbuild.py --mode static --dyn=-20,3,9):
    low-passed at 7 kHz, compressed, driven 9 dB into the true-peak limiter.
    (The 450 Hz high-pass is already in the samples: lprt.Cfg.hp.)"""
    import psy
    u = render(sg, len(sg["rows"]))
    seg = signal.sosfiltfilt(signal.butter(8, 7000, fs=FS, output="sos"), u)
    seg = psy.dynamics(seg, comp_thr=-20, ratio=3, drive_db=9)
    fi = int(0.02 * FS)
    seg[:fi] *= np.linspace(0, 1, fi)
    return 0.5 + 0.5 * seg


def encode(x):
    """x (0..1 at 62.5 kHz) as bits, as SDMLP 3c: the static noise shaping
    from the measured speaker and ear weighting, runs of at most 8."""
    from sdmhq import modulate_m, weight3
    from scipy.linalg import solve_toeplitz
    ff = np.linspace(0, FS / 2, 8193)
    W = weight3(ff)
    W = W / W.max() + 3e-4
    r = np.fft.irfft(W)[:17]
    a = np.concatenate([[1.0], solve_toeplitz(r[:16], -r[1:17])])
    return modulate_m(x, a, 32, 8).astype(np.uint8)


def lentab():
    out = []
    for w in range(256):
        b = [(w >> (7 - i)) & 1 for i in range(8)]
        n = 1
        while n < 8 and b[n] == b[0]:
            n += 1
        out.append(n - 1)
    return np.array(out)


# ------------------------------------------------------------------ the stream's player
def sdm_src(corr, s6, x0, gap, start):
    """SDMLP's player for banks only, with a way in from the tracker (hot:
    the first write gap cycles after the paging) and out to it (path_exit,
    in the last bank)."""
    g = sdmbuild.Gen(corr, s6)
    L = g.L
    PG = 4 + s6
    W = lambda k: "X" if k % 2 else "Y"
    lp, op, cnt, rd = "bl", "bop", "bcnt", 4
    L("IF (P% AND &FF) > &100 - 130 : ALIGN &100 : ENDIF")
    L(".bl_span")
    L(".bl_rg")
    g.adj = []
    g.free = "Y"
    g.fill("blR2", spent=6 + 3)
    g.core_nc(W(3))
    L("    JMP (rvec)")                             # next bank, or out (in the last)
    L(".bl_pg")
    g.adj = []
    g.free = "X"
    g.fill("blP1", spent=6 + 3)
    g.core_nc(W(2))
    L("    DEC bcnt : BEQ bl_rg")
    L("    ASSERT HI(P%) = HI(bl_rg)")
    g.fill("blP2", spent=6 + 2)
    g.core_nc(W(3))
    g.fill("blP3", [("INC bop + 2", 6)], [("JMP bl_4", 3)])
    L(".bl")
    g.core_nc(W(1))
    L("    INC bop + 1 : BEQ bl_pg")
    L("    ASSERT HI(P%) = HI(bl_pg)")
    L("    ASSERT HI(P%) = HI(bl_span)")
    g.fill("bl1", spent=6 + 2)
    for k in range(2, 8):
        L(".bl_%d" % k)
        g.tick("bl%d" % k, W(k))
    L(".bl_8")
    g.core(W(8))
    L(".bop LDA &FFFF")
    g.fill("bl8", [], [("JMP bl", 3)], spent=rd)
    L(".path_next")                                 # (gap 3: JMP (rvec) 5 spent)
    g.adj = []
    g.pad(g.gap + corr.get("nx3", 0) - 5, [])
    g.core(W(4))
    L("    LDY #0")
    L("pnb = P% - 1")                               # the next bank (the loader)
    g.fill("nx4", [("STY ULA_PG", PG)], spent=2)
    g.tick("nx5", W(5), [("LD{R} #0", 2), ("ST{R} bop + 1", 4)])
    g.tick("nx6", W(6), [("LD{R} mycnt", 4), ("ST{R} bcnt", 4)])
    g.tick("nx7", W(7), [("LD{R} #HI(DATA)", 2), ("ST{R} bop + 2", 4)], [("JMP bl_8", 3)])
    L(".path_exit")
    g.adj = []
    g.pad(g.gap + corr.get("nx3", 0) - 5, [])
    for k in range(4, 8):
        g.tick("ex%d" % k, W(k))
    L("    LDX lentab, Y : STX ULA_CT")             # the stream's last write
    L("    LDX #%d" % x0)                           # the tracker's first pattern
    L("    LDY #0")
    L("pxh = P% - 1")                               # the tracker's bank (the loader)
    L("    JMP xstub")
    L(".bcnt EQUB %d" % NBP)
    body = "\n".join(g.lines)
    pad = gap - (12 + 8 + 8 + 3 + 7)
    assert pad >= 0 and pad != 1, pad
    pads = (["JMP P% + 3"] if pad & 1 else []) + ["NOP"] * ((pad - (3 if pad & 1 else 0)) // 2)
    return r"""\ LPH stream player (generated by tools/lphybrid.py, after tools/sdmbuild.py)
ULA_PG = &FE05 : ULA_CT = &FE06 : ULA_MC = &FE07
DATA = &C000 - 256 * %d
START = &%04X
ORG &8000
    JMP init : NOP : NOP
.hot                              \ (from the tracker's bank: its STY ULA_PG ends where this begins)
    LDA #LO(START + 1) : STA bop + 1 : LDA #HI(START + 1) : STA bop + 2
    LDA mycnt : STA bcnt
    LDX START : LDA START + 1
    %s
    JMP bl
.init
    SEI
    LDA &282 : AND #&F9 : ORA #2 : STA ULA_MC
    JMP hot
ASSERT hot = &%04X AND P%% <= &%04X
ORG &%04X
.xstub
    STY ULA_PG                    \ (and on in the tracker's bank)
    EQUB 0, 0, 0, 0
.rvec EQUW path_next              \ (path_exit in the last bank: the packer)
.mycnt EQUB %d                    \ pages of stream in this bank (the packer)
""" % (NBP, start, " : ".join(pads), HOT, XSTUB, XSTUB, NBP) + body + r"""
.code_end
ALIGN &100
.lentab
""" + sdmbuild.lentab_src() + r"""
.sh0
FOR w, 0, 255 : EQUB (w * 2) AND 255 : NEXT
.sh1
FOR w, 0, 255 : EQUB (w * 2 + 1) AND 255 : NEXT
ASSERT P% <= DATA
ASSERT LO(rvec) <> &FF
PRINT "LPHSYM", "init", ~init, "rvec", ~rvec, "mycnt", ~mycnt, "pnb", ~pnb, "pxh", ~pxh, "path_exit", ~path_exit, "code_end", ~code_end
SAVE "build/lph/sdm.bin", &8000, DATA
"""


# ------------------------------------------------------------------ the tracker
def tracker_src(g, body, entry, lay, m, D):
    dhi, n1 = lprt.holds(D)
    dlo = 255 - n1
    order = sorted(g.table.values())
    ticks = "\n".join("tick_s%d = tick + 256 * %d" % (k, order.index(j)) for k, j in sorted(g.table.items()))
    return r"""\ LPH tracker (generated by tools/lphybrid.py, after tools/lprtbuild.py)
ULA_PG = &FE05 : ULA_CT = &FE06 : ULA_MC = &FE07
tick = &%04X
%s
CLIP = &%04X
DUAL = &%04X
VOL0 = DUAL
NT = &%04X
STRM0 = &%04X
S0 = %d
DLO = %d : DHI = %d
ORG &8000
.swapout
    LDY #0 : STY ULA_PG           \ (the loader puts the stream's first bank here; its code goes on from &8005)
ORG &%04X
.enter                            \ (from the stream's last bank)
%s
ALIGN &100
""" % (lay["TICK"], ticks, lay["CLIP"], lay["DUAL"], lay["NT"], lay["STRM0"], m["sinit"], dlo, dhi, XSTUB + 3,
       "\n".join(entry)) + body + r"""
.code_end
IF (P% AND 255) > 255 - """ + str(2 * len(g.vec) + 1) + r""" : ALIGN &100 : ENDIF
.vecs
""" + "\n".join("    EQUW %s" % v for v in g.vec) + r"""
.cntlo EQUB 0
.cnthi EQUB 0
.offcell EQUB 0
.argcell EQUB 0
IF (P% AND 255) > 255 - 16 : ALIGN &100 : ENDIF
.vhtab
    FOR i, 0, 15 : EQUB HI(DUAL) + i : NEXT
IF (P% AND 255) > 255 - 80 : ALIGN &100 : ENDIF
.spg
    INCBIN "build/lph/spg.bin"
.end
ASSERT HI(entryS) = HI(entryN) AND HI(entryW) = HI(entryN)
ASSERT end <= &8000 + 256 * """ + str(CODEPAGES) + "\n" + 'PRINT "LPHSYM", "end", ~end, "vecs", ~vecs\nSAVE "build/lph/code.bin", &8000, end\n'


def assemble(name, key="LPHSYM"):
    r = subprocess.run([BEEBASM, "-i", name, "-v"], cwd=ROOT, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    if r.returncode:
        print("\n".join(txt.splitlines()[-25:]))
        sys.exit(1)
    line = [l for l in txt.splitlines() if l.startswith(key)]
    return dict((k, int(v, 16)) for k, v in re.findall(r"([a-z_0-9]+)&([0-9A-F]+)", line[0].replace(key, ""))) if line else {}


def build(row=63, nbank=6, check=False, secs=None):
    os.makedirs(OUT, exist_ok=True)
    T = Cfg.T
    sg = lprt.song(Cfg)
    rt = sg["rowticks"]
    nmain = Cfg.main1 - Cfg.main0
    nvol = len(sg["vols"])
    lay = lprtbuild.layout(nvol)
    caps = [c for _, c in lay["pages"]]
    # ---- where the stream ends: N ticks of it, in nb bytes, the tracker's first frames just before the row
    N = row * rt - 2 * T
    N -= (N - 7) % 8                                # (the stream's player leaves a block at its 8th tick)
    nb = (N + 9) // 8
    o0 = (-nb) % 256
    pagesall = (nb + o0) // 256
    last = pagesall - NBP * (nbank - 1)
    assert 1 <= last and NBP - last >= PARKPAGES, "the stream is %d pages: %d in the last of %d banks (room for %d)" % (
        pagesall, last, nbank, NBP - PARKPAGES)
    # ---- the tracker, from there
    pl = lprt.Player(sg, Cfg, caps)
    fv = int(round(N / T))
    chan = carry(sg, pl, row, N)
    sums, plog, meta = pl.run(start=dict(row=row, frame=fv, chan=chan))
    assert len(pl.pages) <= len(caps), "song stream: %d pages, room for %d" % (len(pl.pages), len(caps))
    M = len(sums)
    xi = lprt.cliptab(Cfg)[sums]
    m = lprt.fsm(xi, Cfg)
    tbits, codes = lprt._run(xi.astype(np.int64), m["PAT"], m["NXT"], m["pats"], m["sinit"], T, 2 * Cfg.K)
    s2 = int(m["NXT"][m["NXT"][m["sinit"], xi[0]], xi[1]])
    tabs = lprt.ticktabs(m, Cfg.skip)
    NS = 4 * Cfg.K
    open(os.path.join(OUT, "spg.bin"), "wb").write(bytes([a >> 8 for a, _ in lay["pages"]][:len(pl.pages)]))
    import itertools
    best = None
    for free in itertools.combinations(range(T), len(Cfg.skip)):
        for w in range(8, 28):
            for svcv in (0, 1, 2):
                try:
                    g = Gen(T, w, free, Cfg.leak, svcv, hybrid=True)
                    _, info = g.build()
                except AssertionError:
                    continue
                if info["skip"] != tuple(Cfg.skip):
                    continue
                key = (min(info["spare"], min(info["hspare"].values())), info["spare"])
                if best is None or key > best[0]:
                    best = (key, w, free, svcv)
    assert best, "no frame fits"
    _, w, free, svcv = best
    g = Gen(T, w, free, Cfg.leak, svcv, hybrid=True)
    body, info = g.build()
    stores = []
    for c, (p0, st0, v0, n0) in enumerate(chan):
        stores += [(p0 & 255, "c%d_fr + 1" % c), ((p0 >> 8) & 255, "c%d_f + 1" % c), (p0 >> 16, "c%d_f + 2" % c),
                   (st0 & 255, "c%d_fi + 1" % c), (st0 >> 8, "c%d_ii + 1" % c), ((lay["DUAL"] >> 8) + v0, "c%d_v + 2" % c)]
    stores += [(lay["STRM0"] >> 8, "st0 + 2"), (lay["STRM0"] >> 8, "st1 + 2"), (lay["STRM0"] >> 8, "st2 + 2"),
               (0, "offcell"), (s2, "simm + 1")]
    stores = [("%d" % v, a) for v, a in stores] + [("LO(entryS)", "endj + 1")]
    entry, espare = lprtbuild.entry_path(g, ["FF", "FF", "FFF", "FFFS"], stores, int(codes[1]))
    open(os.path.join(OUT, "tracker.asm"), "w").write(tracker_src(g, body, entry, lay, m, pl.D))
    sym = assemble("build/lph/tracker.asm")
    vecs = sym["vecs"]
    code = open(os.path.join(OUT, "code.bin"), "rb").read()
    print("tracker: %d bytes of code; frame %d cycles, w0 %d; spare cycles: normal %d, others %d, entry frame %d; exit gap %d" % (
        len(code), 32 * T, w, info["spare"], min(info["hspare"].values()), espare, g.exit_gap))
    vt = {v: (vecs & 255) + 2 * i for i, v in enumerate(g.vec)}
    assert (vecs & 255) + 2 * len(g.vec) <= 255
    img = bytearray(0x4000)
    img[:len(code)] = code
    o = lambda a: a - 0x8000
    have = [j for j in range(T) if j - 2 not in Cfg.skip]
    for i, j in enumerate(have):
        img[o(lay["TICK"]) + 256 * i:o(lay["TICK"]) + 256 * (i + 1)] = tabs[j].tobytes()
    img[o(lay["CLIP"]):o(lay["CLIP"]) + 256] = bytes((lay["DUAL"] >> 8) + int(v) for v in lprt.cliptab(Cfg))
    for x in range(32):
        base = o(lay["DUAL"]) + 256 * x
        for s in range(NS):
            img[base + 128 + s] = int(m["PAT"][s, x])
            img[base + 192 + s] = int(m["NXT"][s, x])
    for i, v in enumerate(sg["vols"]):
        img[o(lay["DUAL"]) + 256 * i:o(lay["DUAL"]) + 256 * i + 128] = lprt.voltab(v)[:128].tobytes()
    nt = o(lay["NT"])
    sil = len(sg["notes"])
    for i, (s, p, v) in enumerate(sg["notes"] + [(0, 0, 0)]):
        addr, st, vi = (Cfg.main0, 0, 0) if i == sil else (sg["smp"][s]["addr"], lprt.step16(sg, Cfg, s, p), pl.vix[v])
        img[nt + i], img[nt + 256 + i], img[nt + 512 + i], img[nt + 768 + i] = addr & 255, addr >> 8, st & 255, st >> 8
        img[nt + 1024 + i] = (lay["DUAL"] >> 8) + vi
    nbytes = 0
    for pi, recs in enumerate(pl.pages):
        a, cap = lay["pages"][pi]
        b = bytearray()
        for r in recs:
            k = r[0]
            if k == "WAIT":
                hi, n1 = lprt.holds(r[1])
                b += bytes([vt["p_WAIT_N"], hi, 255 - n1])
            elif k == "GOTO":
                b += bytes([vt["p_GOTO_C"], r[1]])
            elif k == "EXIT":
                b += bytes([vt["p_EXIT_X"], 0])
            elif k == "NOP":
                b += bytes([vt["p_NOP_" + ("L" if r[-1] else "C")], 0])
            else:
                c, arg, lst = r[1], r[2], r[3]
                b += bytes([vt["p_%s%d_%s" % (k, c, "L" if lst else "C")], sil if (k == "NOTE" and arg < 0) else arg])
        assert len(b) <= cap, (pi, len(b), cap)
        img[o(a):o(a) + len(b)] = b
        nbytes += len(b)
    print("song from row %d: %d records in %d pages (%d bytes; room for %d); %d frames, %.1f s" % (
        row, len(pl.stream), len(pl.pages), nbytes, sum(caps), M, M * T / FS))
    # ---- the stream
    x = target(sg)
    B = encode(x[:N + 4000])[:N]
    pat0 = int(m["pats"][codes[0]])
    x0abs = [((pat0 >> (T - 1 - k)) & 1) ^ int(B[-1]) for k in range(T)]
    S = np.concatenate([[1 - B[0]], B, x0abs]).astype(np.uint8)
    assert len(S) == 8 * nb and x0abs[0] != B[-1]
    by = np.packbits(S)
    ssym = None
    for _ in range(2):
        open(os.path.join(OUT, "sdm.asm"), "w").write(sdm_src({}, 1, int(codes[0]), g.exit_gap, 0xC000 - 256 * NBP + o0))
        ssym = assemble("build/lph/sdm.asm")
    scode = open(os.path.join(OUT, "sdm.bin"), "rb").read()
    DATA = 0xC000 - 256 * NBP
    assert len(scode) == DATA - 0x8000
    per = 256 * NBP
    lay0 = bytes(o0) + by.tobytes()
    main = bytearray(sg["mem"]) + bytes([lprt.SIL] * (nmain - len(sg["mem"])))
    imgs = []
    for k in range(nbank):
        im = bytearray(scode) + lay0[k * per:(k + 1) * per]
        im += bytes(0x4000 - len(im))
        if k == nbank - 1:
            im[ssym["rvec"] - 0x8000:ssym["rvec"] - 0x8000 + 2] = bytes([ssym["path_exit"] & 255, ssym["path_exit"] >> 8])
            im[ssym["mycnt"] - 0x8000] = last
            assert DATA + 256 * last <= PARK
            im[PARK - 0x8000:] = main[:LOSPLIT - Cfg.main0]
        imgs.append(bytes(im))
    imgs.append(bytes(img))
    print("stream: %d ticks (%.2f s) in %d bytes: %d banks, %d pages in the last (room for %d beside the parked samples)" % (
        N, N / FS, nb, nbank, last, NBP - PARKPAGES))
    # ---- the disc
    open(os.path.join(OUT, "lphload_cfg.asm"), "w").write(
        "NSTREAM = %d\nPNB = &%04X\nPXH = &%04X\nINIT = &%04X\nPARK = &%04X\nPARKPAGES = %d\nLO0 = &%04X\n" % (
            nbank, ssym["pnb"], ssym["pxh"], ssym["init"], PARK, PARKPAGES, Cfg.main0))
    assemble("src/lprt/lphload.asm")
    import ssd
    files = [("!BOOT", 0, 0, b"*RUN LPHLOAD\r"),
             ("LPHLOAD", 0x0900, 0x0900, open(os.path.join(OUT, "lphload.bin"), "rb").read())]
    for k, im in enumerate(imgs):
        files.append(("LPB%d" % k, 0x2000, 0x2000, im))
    files.append(("LPM", LOSPLIT, LOSPLIT, bytes(main[LOSPLIT - Cfg.main0:])))
    disc = os.path.join(OUT, "lph.ssd")
    name = "electron-lost-patrol-hybrid.ssd" if nbank == 6 else "build/lph/lph-%dbank.ssd" % nbank
    print("disc: %d sectors -> %s" % (ssd.make(disc, "LPH", 3, files), name))
    shutil.copyfile(disc, os.path.join(ROOT, name))    # (the emulator has 6 free banks, the ElkSD 7: check with --row 52 --nbank 5)
    # ---- what the emulator should show, a pass: the stream's writes, then the tracker's (a tick each; -1: none)
    lt = lentab()
    win = np.convolve(np.concatenate([S, np.zeros(8, np.uint8)]).astype(np.int64), 1 << np.arange(8), "full")[7:7 + len(S)]
    nf = 8 * nb - 9
    eb = lt[win[:nf + 1]]
    et = tabs[:, codes].T.astype(np.int16)
    et[:, [k + 2 for k in Cfg.skip]] = -1
    et = et.reshape(-1)
    if et[0] != eb[-1]:
        print("(the tracker's first pattern would have written %d where the stream writes %d: no run starts there)" % (et[0], eb[-1]))
    exp = np.concatenate([eb, et[1:]]).astype(np.int16)
    assert len(exp) == nf + 8 * M
    np.save(os.path.join(OUT, "expect.npy"), exp)
    np.save(os.path.join(OUT, "bits.npy"), np.concatenate([B, tbits]))
    json.dump({"row": row, "N": int(N), "frames": int(M), "nbank": nbank}, open(os.path.join(OUT, "build.json"), "w"))
    if check:
        emu(secs or 40, exp, nf)
    return exp


def emu(secs, exp, nf):
    log = os.path.join(OUT, "emu.log")
    env = dict(os.environ, ELKSDLAYOUT="1", SWFILL="rand:7")
    subprocess.run([lprtbuild.CLKH, "--rompath", lprtbuild.ROMS, "--media", os.path.join(OUT, "lph.ssd"), "--shiftbreak", "1000000",
                    "--soundlog", log, "boot", str(int(secs))], env=env, capture_output=True)
    t, v = [], []
    for l in open(log):
        a, r, x = l.split()
        if r == "6":
            t.append(int(a))
            v.append(int(x, 16))
    t, v = np.array(t), np.array(v)
    if len(t) < 100:
        print("emulator: only %d writes to &FE06" % len(t))
        return False
    d = np.diff(t)
    s0 = np.flatnonzero(d > 1000)[-1] + 1 if (d > 1000).any() else 0
    t, v = t[s0:], v[s0:]
    n = t - t[0]
    bad = np.flatnonzero(n % 32)
    print("emulator: %d writes (%.2f s); not a whole number of ticks after the first: %d" % (len(t), n[-1] / 2e6, len(bad)))
    for b in bad[:8]:
        i = (t[b - 1] - t[0]) // 32
        print("   write %d: %d cycles after the one before (tick %d: %s)" % (
            b, t[b] - t[b - 1], i, "stream" if i % len(exp) <= nf else "tracker frame %d" % ((i % len(exp) - nf) // 8)))
    if len(bad):
        return False
    got = np.full(n[-1] // 32 + 1, -1, np.int16)
    got[n // 32] = v
    ref = np.tile(exp, len(got) // len(exp) + 2)
    for off in range(0, 16):
        m = len(got) - off
        if (got[off:off + 2000] == ref[:2000]).all():
            mm = np.flatnonzero(got[off:] != ref[:m])
            print("values: %d of %d ticks differ over %.2f passes%s" % (len(mm), m, m / len(exp),
                  "" if not len(mm) else " (first at tick %d: %s)" % (mm[0], "stream" if mm[0] % len(exp) <= nf else "tracker frame %d" % ((mm[0] % len(exp) - nf) // 8))))
            return not len(mm)
    print("values: no alignment found; first ticks", got[:24].tolist(), "expected", exp[:24].tolist())
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=63)
    ap.add_argument("--nbank", type=int, default=6)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--secs", type=float)
    a = ap.parse_args()
    build(a.row, a.nbank, a.check, a.secs)
