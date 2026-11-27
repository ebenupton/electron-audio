#!/usr/bin/env python3
"""Lost Patrol as a precomputed 62.5 kHz 1-bit sigma-delta stream (SDMLP).

  sdmbuild.py [--start 71.68] [--slot N] [--corr build/sdm/corr.json]

The ULA's sound flip-flop is the output; the stream is computed offline (a
noise-shaped modulator weighted by the measured speaker response and the
ear: tools/sdmhq.py) and played by a cycle-exact player that writes &FE06
every 16 us tick.  At each tick the player writes S = (length of the run
that starts after this tick) - 1, looked up from an 8-bit window of the
coming bits; the ULA latches it at the next reload, so any bit stream with
runs of at most 8 ticks plays exactly.

Layout: the player and its tables at &8000 in each of 7 sideways RAM banks
(identical code: paging switches the copy that runs), stream data from DATA
to &BFFF in each, then main RAM &1E00-&7FFF.  State: X = the window (the
next 8 bits), A = the rest of the current byte, ptr (zero page) = the last
byte loaded; so nothing has to be copied when the bank changes.

Every tick is a 19-cycle core plus a gap of exactly GAP cycles in which the
block's work (pointer, page, bank) is scheduled; corr.json holds per-gap
cycle corrections measured in the emulator (the 1 MHz stretches).
"""
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
BEEBASM = os.environ.get("BEEBASM", os.path.expanduser("~/elite-electron-tools/beebasm"))
OUT = os.path.join(ROOT, "build", "sdm")
MOD = os.path.join(ROOT, "mods/lost patrol (theme).mod")
NBANK = 7
MAIN0, MAIN1 = 0x1E00, 0x8000
FS = 62500


# ------------------------------------------------------------------ player
class Gen:
    """Emit tick cores with scheduled gaps, all in beebasm source.

    The core (19 cycles + the &FE06 stretch) has a variant for when its BCC
    would cross a page (taken: +1): both paths padded to 21.  Which variant,
    and so how long the following gap's padding is, is decided at assembly
    time from P% (cr_N).  Short branches in gaps likewise (br_N)."""

    def __init__(self, corr, s6):
        self.lines = []
        self.corr = corr
        self.gap = 13 - s6          # 32 - 19 - stretch of the &FE06 write
        self.n = 0
        self.nb = 0
        self.free = "X"
        self.crcost = {}
        self.adj = []               # assembly-time adjustments to the current gap

    def L(self, s):
        self.lines.append(s)

    def core(self, win, v3=False):
        """A tick: S = lentab[window] to &FE06, then window = window << 1 |
        the next bit (from A).  win: the register holding the window ("X" or
        "Y"); afterwards the window is in the other one and win is free."""
        n = self.n
        self.n += 1
        o = "Y" if win == "X" else "X"
        self.free = win
        self.L("cr_%d = (HI(P%% + 9) <> HI(P%% + 15))" % n)
        self.L("    LD%s lentab, %s : ST%s ULA_CT : ASL A" % (o, win, o))
        self.L("IF cr_%d" % n)
        self.L("    BCC t0_%d" % n)
        if v3:      # both paths 22: the next gap's padding is 3 short, not 2
            self.L("    LD%s sh1, %s : JMP P%% + 3 : JMP tn_%d" % (o, win, n))
            self.L(".t0_%d LD%s sh0, %s : NOP : NOP" % (n, o, win))
        else:
            self.L("    LD%s sh1, %s : NOP : JMP tn_%d" % (o, win, n))
            self.L(".t0_%d LD%s sh0, %s : JMP P%% + 3" % (n, o, win))
        self.L("ELSE")
        self.L("    BCC t0x_%d" % n)
        self.L("    LD%s sh1, %s : JMP tn_%d" % (o, win, n))
        self.L(".t0x_%d LD%s sh0, %s : NOP" % (n, o, win))
        self.L("ENDIF")
        self.L(".tn_%d" % n)
        self.adj = ["cr_%d" % n]    # the variant costs 2 (v3: 3) more: taken from the gap
        self.crcost["cr_%d" % n] = 3 if v3 else 2

    def core_nc(self, win):
        """A core that must not cross a page (asserted): no variant."""
        self.core(win)
        self.L("    ASSERT NOT(cr_%d)" % (self.n - 1))
        self.adj = []

    def skip(self, target):
        """BNE over a JMP to target (the rare path): common path 3 cycles
        (4 if the branch crosses a page), rare path 2 + 3."""
        n = self.nb
        self.nb += 1
        self.L("br_%d = (HI(P%% + 2) <> HI(P%% + 5))" % n)
        self.L("    BNE bn_%d : JMP %s" % (n, target))
        self.L(".bn_%d" % n)
        return "br_%d" % n

    def pad(self, c, adj):
        """c cycles of padding, less 2 for each true cr_ flag and 1 for each
        true br_ flag in adj (assembly time)."""
        def seq(c):
            assert c >= 0 and c != 1, c
            o = []
            if c % 2:
                o.append("JMP P% + 3")
                c -= 3
            return o + ["NOP"] * (c // 2)
        crs = [x for x in adj if x.startswith("cr")]
        brs = [x for x in adj if x.startswith("br")]
        combos = []
        import itertools
        for vc in itertools.product((0, 1), repeat=len(crs)):
            for vb in itertools.product((0, 1), repeat=len(brs)):
                cond = " AND ".join([("%s" % x if v else "NOT(%s)" % x) for v, x in zip(vc, crs)] +
                                    [("%s" % x if v else "NOT(%s)" % x) for v, x in zip(vb, brs)])
                combos.append((cond, c - sum(self.crcost[x] for v, x in zip(vc, crs) if v) - sum(vb)))
        if len(combos) == 1:
            o = seq(combos[0][1])
            if o:
                self.L("    " + " : ".join(o))
            return
        for i, (cond, cc) in enumerate(combos):
            self.L(("IF " if i == 0 else "ELIF ") + cond)
            o = seq(cc)
            if o:
                self.L("    " + " : ".join(o))
        self.L("ENDIF")

    def fill(self, label, items=(), tail=(), spent=0, extra_adj=()):
        used = spent + sum(c for _, c in items) + sum(c for _, c in tail)
        for a, _ in items:
            self.L("    " + a.replace("{R}", self.free))
        adj = list(extra_adj) if any(x.startswith("cr") for x in extra_adj) else self.adj + list(extra_adj)
        self.pad(self.gap - used + self.corr.get(label, 0), adj)
        for a, _ in tail:
            self.L("    " + a)

    def tick(self, label, win, items=(), tail=()):
        base = self.gap - sum(c for _, c in items) - sum(c for _, c in tail) + self.corr.get(label, 0)
        self.core(win, v3=(base == 3))
        self.fill(label, items, tail)


def player(corr, s6, sm, nbp, nmp):
    """The player source.  s6: stretch of an &FE0x write; sm: extra cycles
    of a main RAM read (guesses; corr.json fixes the rest per gap).
    nbp/nmp: pages of data in a bank / in main RAM.

    Tick k of a block (1..8) has its window in X if k is odd, Y if even; the
    gap after it may use that register ({R}).  Per loop (bank, main), one
    span kept inside a page (no branch or core there crosses one; asserted):
    region chunk, page path, main line tick 1.  Then the main line's ticks
    2..8, then the rare paths' remaining ticks."""
    g = Gen(corr, s6)
    L = g.L
    PG = 4 + s6
    W = lambda k: "X" if k % 2 else "Y"
    for lp, op, cnt, rd in (("bl", "bop", "bcnt", 4), ("ml", "mop", "mcnt", 4 + sm)):
        L("IF (P% AND &FF) > &100 - 130 : ALIGN &100 : ENDIF")
        L(".%s_span" % lp)
        # ---- region chunk: arrives in gap 2 (page path) after DEC (6) + BEQ taken (3)
        L(".%s_rg" % lp)
        g.adj = []
        g.free = "Y"
        g.fill(lp + "R2", spent=6 + 3)
        g.core_nc(W(3))
        if lp == "bl":
            L("    JMP (rvec)")                         # next bank, or main RAM (in the last bank)
        else:
            g.fill("mb3", [], [("JMP mb_4", 3)])
        # ---- page path: arrives in gap 1 after INC (6) + BEQ taken (3)
        L(".%s_pg" % lp)
        g.adj = []
        g.free = "X"
        g.fill(lp + "P1", spent=6 + 3)
        g.core_nc(W(2))
        L("    DEC %s : BEQ %s_rg" % (cnt, lp))
        L("    ASSERT HI(P%%) = HI(%s_rg)" % lp)
        g.fill(lp + "P2", spent=6 + 2)
        g.core_nc(W(3))
        g.fill(lp + "P3", [("INC %s + 2" % op, 6)], [("JMP %s_4" % lp, 3)])
        # ---- main line
        L(".%s" % lp)
        g.core_nc(W(1))
        L("    INC %s + 1 : BEQ %s_pg" % (op, lp))
        L("    ASSERT HI(P%%) = HI(%s_pg)" % lp)
        L("    ASSERT HI(P%%) = HI(%s_span)" % lp)
        g.fill(lp + "1", spent=6 + 2)
        for k in range(2, 8):
            L(".%s_%d" % (lp, k))
            g.tick("%s%d" % (lp, k), W(k))
        L(".%s_8" % lp)
        g.core(W(8))
        L(".%s LDA &FFFF" % op)                         # the reload (operand: the pointer)
        g.fill(lp + "8", [], [("JMP %s" % lp, 3)], spent=rd)
        # ---- rare paths' remaining ticks
        if lp == "bl":
            L(".path_next")                             # (gap 3: JMP (rvec) 5 spent)
            g.adj = []
            g.pad(g.gap + corr.get("nx3", 0) - 5, [])
            g.core(W(4))
            L("    LDY #0")
            L("pnb = P% - 1")                           # the next bank (patched by the loader)
            g.fill("nx4", [("STY ULA_PG", PG)], spent=2)
            g.tick("nx5", W(5), [("LD{R} #0", 2), ("ST{R} bop + 1", 4)])
            g.tick("nx6", W(6), [("LD{R} #HI(DATA)", 2), ("ST{R} bop + 2", 4)])
            g.tick("nx7", W(7), [("LD{R} #%d" % nbp, 2), ("ST{R} bcnt", 4)], [("JMP bl_8", 3)])
            L(".path_main")
            g.adj = []
            g.pad(g.gap + corr.get("pm3", 0) - 5, [])
            g.tick("pm4", W(4), [("LD{R} #0", 2), ("ST{R} mop + 1", 4)])
            g.tick("pm5", W(5), [("LD{R} #HI(&%04X)" % MAIN0, 2), ("ST{R} mop + 2", 4)])
            g.tick("pm6", W(6), [("LD{R} #%d" % nmp, 2), ("ST{R} mcnt", 4)])
            g.tick("pm7", W(7), [], [("JMP ml_8", 3)])
        else:
            L(".mb_4")
            g.core(W(4))
            L("    LDY #0")
            L("pb0 = P% - 1")                           # bank 0 (patched)
            g.fill("mb4", [("STY ULA_PG", PG)], spent=2)
            g.tick("mb5", W(5), [("LD{R} #0", 2), ("ST{R} bop + 1", 4)])
            g.tick("mb6", W(6), [("LD{R} #HI(DATA)", 2), ("ST{R} bop + 2", 4)])
            g.tick("mb7", W(7), [("LD{R} #%d" % nbp, 2), ("ST{R} bcnt", 4)], [("JMP bl_8", 3)])
    L(".bcnt EQUB %d" % nbp)
    L(".mcnt EQUB %d" % nmp)
    return "\n".join(g.lines)


def lentab_src():
    out = []
    for w in range(256):
        b = [(w >> (7 - i)) & 1 for i in range(8)]
        n = 1
        while n < 8 and b[n] == b[0]:
            n += 1
        out.append(n - 1)
    return "\n".join("EQUB " + ", ".join(str(v) for v in out[i:i + 16]) for i in range(0, 256, 16))


# --------------------------------------------------------------- the stream
def stream(start, nbytes, rmax=8, hp=450.0, thr=-6.0, amp=0.5, lam=3e-4, M=32, mode="psy", spl=70.0, dyn=None, src=None):
    """The bit stream.  v3: bass below hp high-passed away (the speaker
    rolls off below 450 Hz: SPKCHAR), a true-peak limiter (no overshoot), the
    speaker response measured with SPKCHAR (sdmhq.speaker_db3), and either a
    psychoacoustic per-frame NTF (psy.py: noise placed where the music masks
    it, assuming full scale = spl dB SPL) or a static one from the measured
    weighting; lookahead (M-path) modulator either way."""
    from modrender import render
    from sdmhq import modulate_m, modulate_m_tv, speaker_db3, weight3, wsnr2
    import psy
    from scipy import signal
    from scipy.linalg import solve_toeplitz
    cache = os.path.join(OUT, "render62k.npy")
    if src:
        y = np.load(src).astype(float)
    elif os.path.exists(cache):
        y = np.load(cache)
    else:
        y = render(MOD, FS)
        np.save(cache, y)
    n = nbytes * 8 + 1
    seg = y[int(start * FS):int(start * FS) + n]
    seg = signal.sosfiltfilt(signal.butter(8, 7000, fs=FS, output="sos"), seg)
    if hp:
        seg = signal.sosfiltfilt(signal.butter(2, hp, "high", fs=FS, output="sos"), seg)
    if dyn:      # (comp_thr, ratio, drive_db): compressor + drive into the true-peak limiter
        seg = psy.dynamics(seg, comp_thr=dyn[0], ratio=dyn[1], drive_db=dyn[2])
    else:
        seg = psy.true_peak_limit(seg, thr_db=thr)
    fi, fo = int(0.02 * FS), int(0.4 * FS)
    env = np.ones(len(seg))
    env[:fi] = np.linspace(0, 1, fi)
    env[-fo:] = np.linspace(1, 0, fo)
    x = 0.5 + amp * seg * env
    T, f = psy.thresholds(x, speaker_db3, full_spl=spl)
    if mode == "psy":
        A = psy.ntfs_loud(T, f, speaker_db3, L=16, lam=1e-4, hmax=2.0, alpha=0.3, iters=5)
        bits = modulate_m_tv(x, A, psy.HOP, M, rmax)
    else:
        ff = np.linspace(0, FS / 2, 8193)
        W = weight3(ff)
        W = W / W.max() + lam
        r = np.fft.irfft(W)[:17]
        a = np.concatenate([[1.0], solve_toeplitz(r[:16], -r[1:17])])
        bits = modulate_m(x, a, M, rmax)
    runs = np.diff(np.flatnonzero(np.diff(bits) != 0))
    r = psy.nmr(bits, x, T, f, speaker_db3)
    print("stream (%s): %.2f s, peak |x-0.5| %.3f, noise loudness %.1f, audible cells %.0f%%, longest run %d" % (
        mode, len(bits) / FS, np.abs(x - 0.5).max(), r[3], 100 * r[1], runs.max()))
    np.save(os.path.join(OUT, "bits.npy"), bits.astype(np.uint8))
    np.save(os.path.join(OUT, "x.npy"), x.astype(np.float32))
    by = np.packbits(bits[1:].astype(np.uint8))[:nbytes]     # window at tick t = bits t+1..t+8
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=71.68)
    ap.add_argument("--corr", default=os.path.join(OUT, "corr.json"))
    ap.add_argument("--s6", type=int, default=1)
    ap.add_argument("--sm", type=int, default=1)
    ap.add_argument("--nbp", type=int, default=57, help="data pages a bank (DATA = &C000 - 256 * nbp)")
    ap.add_argument("--nbank", type=int, default=7, help="sideways RAM banks (the emulator has 6 free: its DFS is in 0)")
    ap.add_argument("--out", default="electron-lost-patrol-sdm.ssd")
    ap.add_argument("--mode", default="psy", choices=["psy", "static"])
    ap.add_argument("--dyn", help="compressor threshold,ratio,drive dB (e.g. -20,3,9)")
    ap.add_argument("--src", help="source audio: .npy of mono samples at 62.5 kHz (instead of the MOD)")
    ap.add_argument("--bits", help="play this bit stream (.npy of 0/1) instead of the music")
    ap.add_argument("--title", default="SDMLP")
    ap.add_argument("--nostream", action="store_true", help="reuse build/sdm/stream.bin")
    a = ap.parse_args()
    global NBANK
    NBANK = a.nbank
    os.makedirs(OUT, exist_ok=True)
    corr = json.load(open(a.corr)) if os.path.exists(a.corr) else {}
    nbp, nmp = a.nbp, (MAIN1 - MAIN0) // 256
    src = r"""\ SDMLP player (generated by tools/sdmbuild.py)
ULA_PG = &FE05 : ULA_CT = &FE06 : ULA_MC = &FE07
DATA = &C000 - 256 * %d
ORG &8000
.init
    SEI
    LDA &282 : AND #&F9 : ORA #2 : STA ULA_MC
    LDA #LO(DATA + 1) : STA bop + 1 : LDA #HI(DATA + 1) : STA bop + 2
    LDA #%d : STA bcnt
    LDX DATA : LDA DATA + 1
    JMP bl
.rvec EQUW path_next          \ (path_main in the last bank: set by the packer)
""" % (nbp, nbp) + player(corr, a.s6, a.sm, nbp, nmp) + r"""
.code_end
ALIGN &100
.lentab
""" + lentab_src() + r"""
.sh0
FOR w, 0, 255 : EQUB (w * 2) AND 255 : NEXT
.sh1
FOR w, 0, 255 : EQUB (w * 2 + 1) AND 255 : NEXT
ASSERT P% <= DATA
ASSERT LO(rvec) <> &FF
PRINT "SDMCODE", ~init, ~code_end, "DATA", ~DATA, "rvec", ~rvec, "pnb", ~pnb, "pb0", ~pb0, "path_main", ~path_main
SAVE "build/sdm/code.bin", init, DATA
"""
    open(os.path.join(OUT, "player.asm"), "w").write(src)
    r = subprocess.run([BEEBASM, "-i", "build/sdm/player.asm", "-v"], cwd=ROOT, capture_output=True, text=True)
    info = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("SDMCODE") or "rror" in l]
    print("\n".join(info))
    if r.returncode or not info or "rror" in info[0]:
        print("\n".join((r.stdout + r.stderr).splitlines()[-20:]))
        sys.exit(1)
    import re
    sym = dict(re.findall(r"([A-Za-z_0-9]+)&([0-9A-F]+)", info[0].replace("SDMCODE&8000 &", "SDMCODE&8000 end&")))
    DATA, rvec, pnb, pb0, pmain = (int(sym[k], 16) for k in ("DATA", "rvec", "pnb", "pb0", "path_main"))
    code = open(os.path.join(OUT, "code.bin"), "rb").read()
    per = 0xC000 - DATA
    nmain = MAIN1 - MAIN0
    nbytes = NBANK * per + nmain
    print("data: %d bytes a bank from &%04X, %d in main RAM: %d bytes, %.2f s" % (per, DATA, nmain, nbytes, nbytes * 8 / FS))
    if a.bits:
        bits = np.load(a.bits).astype(np.uint8)
        assert len(bits) >= nbytes * 8 + 1, (len(bits), nbytes * 8 + 1)
        by = np.packbits(bits[1:nbytes * 8 + 1])
    elif a.nostream:
        by = np.fromfile(os.path.join(OUT, "stream.bin"), np.uint8)
    else:
        by = stream(a.start, nbytes, mode=a.mode, src=a.src, dyn=[float(v) for v in a.dyn.split(",")] if a.dyn else None)
        by.tofile(os.path.join(OUT, "stream.bin"))
    imgs = []
    for k in range(NBANK):
        img = bytearray(code) + bytes(by[k * per:(k + 1) * per])
        assert len(img) == 0x4000
        if k == NBANK - 1:
            img[rvec - 0x8000:rvec - 0x8000 + 2] = bytes([pmain & 255, pmain >> 8])
        imgs.append(bytes(img))
    mainimg = bytes(by[NBANK * per:])
    # the loader
    lsrc = open(os.path.join(ROOT, "src/sdm/sdload.asm")).read()
    open(os.path.join(OUT, "sdload_cfg.asm"), "w").write(
        "NBANKS = %d\nPNB = &%04X\nPB0 = &%04X\nMAIN0 = &%04X\n" % (NBANK, pnb, pb0, MAIN0))
    r = subprocess.run([BEEBASM, "-i", "src/sdm/sdload.asm", "-v"], cwd=ROOT, capture_output=True, text=True)
    li = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("SDLOAD") or "rror" in l]
    print("\n".join(li))
    if r.returncode:
        print("\n".join((r.stdout + r.stderr).splitlines()[-20:]))
        sys.exit(1)
    import ssd
    files = [("!BOOT", 0, 0, b"*RUN SDLOAD\r"),
             ("SDLOAD", 0x0900, 0x0900, open(os.path.join(OUT, "sdload.bin"), "rb").read())]
    for k, img in enumerate(imgs):
        files.append(("SDB%d" % k, 0x2000, 0x2000, img))
    files.append(("SDM", MAIN0, MAIN0, mainimg))
    disc = os.path.join(OUT, "sdmlp.ssd")
    print("disc: %d sectors" % ssd.make(disc, a.title, 3, files))
    import shutil
    shutil.copyfile(disc, os.path.join(ROOT, a.out))
    json.dump({"DATA": DATA, "per": per, "nmain": nmain, "start": a.start}, open(os.path.join(OUT, "layout.json"), "w"))


if __name__ == "__main__":
    main()
