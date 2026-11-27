#!/usr/bin/env python3
"""Build LPRT: the whole Lost Patrol MOD, decoded on the Electron as it plays.

  lprtbuild.py [--check] [--secs N] [--slot N]

The design and the model are in lprt.py.  This writes the player (6502,
generated: every path through a frame takes exactly T * 32 cycles, with the
&FE06 writes at the same multiples of 32 in each), the bank image, the
samples and the disc:

  !BOOT    *RUN LPLOAD
  LPLOAD   the loader (src/lprt/lpload.asm)
  LPH      16K for a sideways RAM bank: player, tables, song
  LPLO     samples for &0400-&1DFF (parked in a second bank while loading)
  LPM      samples for &1E00-&7FFF

Bank layout (pages): code; the tick tables (run lengths by pattern code); the
clip table (sum -> the modulator page of its level); 32 "dual" pages, the
top half of each the modulator's tables for one input level (pattern code
and next state, by state), the bottom half a volume table, a note table
column or song stream; then whole pages of song stream.

--check runs the disc in clkh and compares every &FE06 write (time and
value) with the model.
"""
import argparse, json, os, re, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import lprt
from lprt import Cfg, FS

BEEBASM = os.environ.get("BEEBASM", os.path.expanduser("~/elite-electron-tools/beebasm"))
CLKH = os.environ.get("CLKH", os.path.expanduser("~/elite-electron-elksd/6-harness/clkh/build/clkh"))
ROMS = os.environ.get("ELK_ROMS", os.path.expanduser("~/elite-electron-roms"))
OUT = os.path.join(ROOT, "build", "lprt")
CODEPAGES = 11
LOSPLIT = 0x1E00

HANDLERS = ["NOTE0", "NOTE1", "NOTE2", "NOTE3", "VOL0", "VOL1", "VOL2", "VOL3", "HI0", "HI1", "HI2", "HI3", "NOP", "GOTO"]


# ------------------------------------------------------------------ the generator
class Path:
    """Straight-line code with the tick writes put in.  Time is in 2 MHz
    cycles from the frame's start; slot k's write (if it has one: not the
    slots in gen.free) has its slow cycle start at w0 + 32 k.  A slow cycle
    (main RAM, &FExx) waits for an even time (relative to w0) and takes two."""

    def __init__(self, gen, t, k):
        self.g, self.t, self.k = gen, t, k
        self.lines = []
        self.yv = False              # Y holds the record's argument
        self._free()

    def _free(self):
        while self.k < self.g.T and self.k in self.g.free:
            self.k += 1

    def W(self):
        return self.g.w0 + 32 * self.k

    def cost(self, cyc, t):
        for ch in cyc:
            if ch == "S":
                if (t - self.g.w0) & 1:
                    t += 1
                t += 2
            else:
                t += 1
        return t

    def padding(self, n):
        assert n >= 0 and n != 1, n
        o = []
        m = n
        if m & 1:
            o.append("JMP P% + 3")
            m -= 3
        o += ["NOP"] * (m // 2)
        if o:
            self.lines.append("    " + " : ".join(o))
        self.t += n

    def write(self):
        """The tick write (it must start at W - 8 or W - 7)."""
        gap = self.W() - 8 - self.t
        assert gap >= -1, "write %d missed: t %d, W %d" % (self.k, self.t, self.W())
        if gap == 1:
            self.padding(2)
        elif gap > 1:
            self.padding(gap)
        assert self.t in (self.W() - 8, self.W() - 7)
        self.lines.append("    LDY tick_s%d, X : STY ULA_CT" % self.k)
        self.t = self.W() + 2
        self.k += 1
        self.yv = False
        self._free()

    def fits(self, *items):
        t = self.t
        for _, cyc in items:
            t = self.cost(cyc, t)
        return self.k >= self.g.T or t <= self.W() - 7

    def grp(self, *items):
        """An indivisible group of (asm, cycles)."""
        while not self.fits(*items):
            self.write()
        for a, cyc in items:
            self.lines.append("    " + a if not a.startswith(".") else a)
            self.t = self.cost(cyc, self.t)

    def ygrp(self, item):
        """An instruction that wants the record's argument in Y."""
        while True:
            items = ([] if self.yv else [("LDY argcell", "FFFF")]) + [item]
            if self.fits(*items):
                break
            self.write()
        self.grp(*items)
        self.yv = True

    def to(self, t1):
        """Pad (and write) until exactly t1."""
        while self.k < self.g.T and self.W() - 7 < t1:
            self.write()
        n = t1 - self.t
        assert n >= 0 and n != 1, "cannot pad %d cycles (t %d, target %d)" % (n, self.t, t1)
        self.padding(n)
        return n

    def rejoin(self):
        """Back to the normal path where the fetches start."""
        self.spare = self.to(self.g.tj - 3)
        assert self.k == self.g.kj
        self.lines.append("    JMP csum")


class Gen:
    """free: the slots with no write.  leak: see Cfg.leak.  svcv: which way
    the record is read."""

    def __init__(self, T, w0, free=(), leak=True, svcv=0, hybrid=False):
        self.T, self.w0, self.free, self.leak, self.svcv = T, w0, set(free), leak, svcv
        self.hybrid = hybrid

    def build(self):
        T = self.T
        FRAME = 32 * T
        slots = [k for k in range(T) if k not in self.free]
        # ---- the normal frame: the counter, the four pointers, the fetches, the modulator
        n = Path(self, 0, 0)
        n.lines.append(".entryN")
        n.grp(("INC cntlo", "FFFFFF"), ("BEQ exj", "FF"))
        tx, kx = n.t + 1 + 3, n.k
        for c in range(4):
            for it in [(".c%d_fr LDA #0" % c, "FF"), (".c%d_fi ADC #0" % c, "FF"), ("STA c%d_fr + 1" % c, "FFFF"),
                       ("LDA c%d_f + 1" % c, "FFFF"), (".c%d_ii ADC #0" % c, "FF"), ("STA c%d_f + 1" % c, "FFFF")]:
                n.grp(it)
            if self.leak:
                n.grp(("BCC c%d_nc" % c, "FF"), ("INC c%d_f + 2" % c, "FFFFFF"))
                n.lines.append(".c%d_bk" % c)
            else:
                for it in [("LDA c%d_f + 2" % c, "FFFF"), ("ADC #0", "FF"), ("STA c%d_f + 2" % c, "FFFF")]:
                    n.grp(it)
        n.lines.append(".csum")              # (the other paths come back here)
        self.tj, self.kj = n.t, n.k
        for c in range(4):
            n.grp((".c%d_f LDY &%04X" % (c, Cfg.main0), "FFFS"),
                  (".c%d_v %s VOL0, Y" % (c, "LDA" if c == 0 else "ADC"), "FFFF"))
        n.grp(("TAY", "FF"), ("LDA CLIP, Y", "FFFF"))
        n.grp(("STA lkp + 2", "FFFF"))
        n.grp(("STA lkn + 2", "FFFF"))
        n.grp((".simm LDY #S0", "FF"), (".lkp LDX DUAL + 128, Y", "FFFF"), (".lkn LDA DUAL + 192, Y", "FFFF"))
        self.q = n.k if n.k < T else slots[0]        # the first write with the new pattern: its table 0
        n.grp(("STA simm + 1", "FFFF"))
        self.spare = FRAME - 3 - n.t - 10 * len([k for k in slots if k >= n.k])
        n.to(FRAME - 3)
        assert n.k == T, "only %d writes in the frame" % n.k
        n.lines.append(".endj JMP entryS")
        self.table = {k: (k - self.q) % T for k in slots}
        self.skip = tuple(sorted((k - self.q) % T - 2 for k in self.free))
        assert all(1 <= k <= T - 3 for k in self.skip), "a slot with no write too soon after the pattern is chosen"
        # ---- a service frame: the record's type and argument, then its handler
        s = Path(self, 0, 0)
        s.lines.append(".entryS")
        if self.svcv == 0:
            s.grp(("LDA offcell", "FFFF"))
            s.grp(("TAY", "FF"), ("ADC #2", "FF"), ("STA offcell", "FFFF"),
                  (".st0 LDA STRM0, Y", "FFFF"), ("STA jv + 1", "FFFF"), (".st1 LDA STRM0 + 1, Y", "FFFF"))
            s.grp(("STA argcell", "FFFF"))
        elif self.svcv == 1:
            s.grp(("LDY offcell", "FFFF"), (".st0 LDA STRM0, Y", "FFFF"), ("STA jv + 1", "FFFF"), (".st1 LDA STRM0 + 1, Y", "FFFF"))
            s.grp(("STA argcell", "FFFF"))
            s.grp(("LDA offcell", "FFFF"))
            s.grp(("ADC #2", "FF"))
            s.grp(("STA offcell", "FFFF"))
        else:
            s.grp(("LDY offcell", "FFFF"), (".st0 LDA STRM0, Y", "FFFF"))
            s.grp(("STA jv + 1", "FFFF"))
            s.grp(("LDY offcell", "FFFF"), (".st1 LDA STRM0 + 1, Y", "FFFF"))
            s.grp(("STA argcell", "FFFF"))
            s.grp(("LDA offcell", "FFFF"))
            s.grp(("ADC #2", "FF"))
            s.grp(("STA offcell", "FFFF"))
        s.grp((".jv JMP (vecs)", "FFFFF"))
        tb, kb = s.t, s.k
        # ---- the counter's low byte ran out: nothing moves; if the wait is over, the next frame is a service frame
        x = Path(self, tx, kx)
        x.lines.append(".exh NOP : JMP exm")
        x.lines.append(".ex2")
        x.grp(("DEC cnthi", "FFFFFF"), ("BPL exh", "FF"), ("LDA #LO(entryS)", "FF"), ("STA endj + 1", "FFFF"))
        x.lines.append(".exm")
        x.rejoin()
        # ---- the default wait
        w = Path(self, 3, 0)
        w.lines.append(".ew2")
        for it in (("LDA #DLO", "FF"), ("STA cntlo", "FFFF"), ("LDA #DHI", "FF"), ("STA cnthi", "FFFF"),
                   ("LDA #LO(entryN)", "FF"), ("STA endj + 1", "FFFF")):
            w.grp(it)
        w.rejoin()
        # ---- the handlers: one copy for each way the frame can end
        H = []
        vec = []
        self.hspare = {"expire": x.spare, "wait": w.spare}

        def handler(name, items, variants=("C", "L")):
            for v in variants:
                p = Path(self, tb, kb)
                p.lines.append(".p_%s_%s" % (name, v))
                if v != "C":
                    p.grp(("LDA #LO(%s)" % {"L": "entryW", "N": "entryN"}[v], "FF"))
                    p.grp(("STA endj + 1", "FFFF"))
                for it in items:
                    if it[-1] == "y":
                        p.ygrp(it[:2])
                    elif isinstance(it, list):
                        p.grp(*it)
                    else:
                        p.grp(it)
                p.rejoin()
                self.hspare[name + v] = p.spare
                H.extend(p.lines)
                vec.append("p_%s_%s" % (name, v))

        for c in range(4):
            handler("NOTE%d" % c, [
                ("LDA NT + 0 * 256, Y", "FFFF", "y"), ("STA c%d_f + 1" % c, "FFFF"),
                ("LDA NT + 1 * 256, Y", "FFFF", "y"), ("STA c%d_f + 2" % c, "FFFF"),
                ("LDA NT + 2 * 256, Y", "FFFF", "y"), ("STA c%d_fi + 1" % c, "FFFF"),
                ("LDA NT + 3 * 256, Y", "FFFF", "y"), ("STA c%d_ii + 1" % c, "FFFF"),
                ("LDA NT + 4 * 256, Y", "FFFF", "y"), ("STA c%d_v + 2" % c, "FFFF")])
        for c in range(4):
            handler("VOL%d" % c, [("LDA vhtab, Y", "FFFF", "y"), ("STA c%d_v + 2" % c, "FFFF")])
        for c in range(4):
            handler("HI%d" % c, [("LDA c%d_f + 2" % c, "FFFF"), ("SEC", "FF"), ("SBC argcell", "FFFF"),
                                 ("STA c%d_f + 2" % c, "FFFF"), ("CLC", "FF")])
        handler("NOP", [])
        handler("GOTO", [("LDA spg, Y", "FFFF", "y"), ("STA st0 + 2", "FFFF"), ("STA st1 + 2", "FFFF"),
                         ("STA st2 + 2", "FFFF"), ("LDA #0", "FF"), ("STA offcell", "FFFF")])
        handler("WAIT", [("LDA argcell", "FFFF"), ("STA cnthi", "FFFF"),
                         [("LDY offcell", "FFFF"), (".st2 LDA STRM0, Y", "FFFF")], ("STA cntlo", "FFFF"),
                         ("INC offcell", "FFFFFF")], variants=("N",))
        if self.hybrid:
            # EXIT: the pattern's last writes, then the bitstream player's bank (it makes the next write)
            assert kb < self.q < T, "EXIT: the new pattern's first write is not late in the frame"
            p = Path(self, tb, kb)
            p.lines.append(".p_EXIT_X")
            while p.k != self.q:
                p.write()
            t = p.cost("FFF", p.t)                  # JMP swapout
            t = p.cost("FF", t)                     # LDY #
            t = p.cost("FFFS", t)                   # STY ULA_PG
            self.exit_gap = p.W() - t               # from there to the next write's slow cycle
            p.lines.append("    JMP swapout")
            H.extend(p.lines)
            vec.append("p_EXIT_X")
        self.vec = vec
        nc = lambda c: [".c%d_nc NOP : JMP c%d_bk" % (c, c)] if self.leak else []
        out = ([".entryW JMP ew2"] + s.lines + [".exj JMP ex2"] + nc(0) + nc(1) + nc(2) + n.lines + nc(3)
               + x.lines + w.lines + H)
        return "\n".join(out), dict(tj=self.tj, kj=self.kj, tb=tb, kb=kb, spare=self.spare, hspare=self.hspare,
                                    skip=self.skip, q=self.q)


def entry_path(g, pre, stores, x1):
    """The hybrid's way in from the bitstream player, whose last write was a
    tick before this frame's first: pre is what it ran after that write (cycle
    strings, ending with the paging); then here every store of stores
    ((value, address)), the frame's writes with the pattern already in X, the
    next pattern x1, and on to the first service frame."""
    t = g.w0 - 32 + 2
    p = Path(g, 0, 0)
    for cyc in pre:
        t = p.cost(cyc, t)
    p.t = t
    last = None
    for v, a in sorted(stores):
        if v != last:
            p.grp(("LDA #%s" % v, "FF"))
            last = v
        p.grp(("STA %s" % a, "FFFF"))
    assert p.k <= g.q, "the entry frame's stores run past the new pattern's first write"
    while p.k != g.q:
        p.write()
    p.grp(("LDX #%d" % x1, "FF"), ("CLC", "FF"))
    spare = p.to(32 * g.T - 3)
    assert p.k == g.T
    p.lines.append("    JMP entryS")
    return p.lines, spare


def fit(T, verbose=False):
    """Every frame that fits: (skip, w0, free, leak, svcv, spare cycles)."""
    import itertools
    found = []
    for nfree in range(0, 4):
        for free in itertools.combinations(range(T), nfree):
            for leak in (False, True):
                for w0 in range(8, 28):
                    for svcv in (0, 1, 2):
                        g = Gen(T, w0, free, leak, svcv)
                        try:
                            _, info = g.build()
                        except AssertionError:
                            continue
                        found.append((info["skip"], w0, free, leak, svcv, info["spare"], min(info["hspare"].values())))
                        break
    return found


def player_src(T, w0, free, leak, svcv, lay, m, D):
    g = Gen(T, w0, free, leak, svcv)
    body, info = g.build()
    dhi, n1 = lprt.holds(D)
    dlo = 255 - n1
    order = sorted(g.table.values())
    ticks = "\n".join("tick_s%d = tick + 256 * %d" % (k, order.index(j)) for k, j in sorted(g.table.items()))
    src = r"""\ LPRT player (generated by tools/lprtbuild.py)
ULA_CT = &FE06 : ULA_MC = &FE07
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
.init
    SEI
    LDA &282 : AND #&F9 : ORA #2 : STA ULA_MC
    LDA #0 : STA ULA_CT
    LDX #X0
    CLC
    JMP entryS
ALIGN &100
""" % (lay["TICK"], ticks, lay["CLIP"], lay["DUAL"], lay["NT"], lay["STRM0"], m["s0"], dlo, dhi)
    src += body + r"""
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
    INCBIN "build/lprt/spg.bin"
.end
ASSERT HI(entryS) = HI(entryN) AND HI(entryW) = HI(entryN)
ASSERT end <= &8000 + 256 * """ + str(CODEPAGES) + "\n"
    src += 'PRINT "LPSYM", "end", ~end, "vecs", ~vecs, "code_end", ~code_end\n'
    src += 'SAVE "build/lprt/code.bin", init, end\n'
    return src, g, info


# ------------------------------------------------------------------ the build
def assemble(name):
    r = subprocess.run([BEEBASM, "-i", name, "-v"], cwd=ROOT, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    if r.returncode:
        print("\n".join(txt.splitlines()[-25:]))
        sys.exit(1)
    return txt


def layout(nvol):
    T = Cfg.T
    lay = {"TICK": 0x8000 + 256 * CODEPAGES}
    lay["CLIP"] = lay["TICK"] + 256 * (T - len(Cfg.skip))
    lay["DUAL"] = lay["CLIP"] + 256
    lay["NT"] = lay["DUAL"] + 256 * nvol
    full0 = lay["DUAL"] + 32 * 256
    pages = [(a, 256) for a in range(full0, 0xC000, 256)]
    pages += [(lay["DUAL"] + 256 * h, 128) for h in range(nvol + 5, 32)]
    lay["pages"] = pages
    lay["STRM0"] = pages[0][0]
    return lay


def build(check=False, secs=None, w0=None, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    T = Cfg.T
    sg = lprt.song(Cfg)
    nmain = Cfg.main1 - Cfg.main0
    assert len(sg["mem"]) <= nmain, "samples: %d bytes, room for %d" % (len(sg["mem"]), nmain)
    nvol = len(sg["vols"])
    lay = layout(nvol)
    caps = [c for _, c in lay["pages"]]
    pl = lprt.Player(sg, Cfg, caps)
    sums, plog, meta = pl.run()
    assert len(pl.pages) <= len(caps), "song stream: %d pages, room for %d" % (len(pl.pages), len(caps))
    xi = lprt.cliptab(Cfg)[sums]
    m = lprt.fsm(xi, Cfg)
    bits, codes = lprt.modulate(m, xi, Cfg)
    tabs = lprt.ticktabs(m, Cfg.skip)
    NS = 4 * Cfg.K
    assert NS <= 64 and Cfg.nx == 32 and len(sg["notes"]) < 128
    # ---- the player
    open(os.path.join(OUT, "spg.bin"), "wb").write(bytes([a >> 8 for a, _ in lay["pages"]][:len(pl.pages)]))
    import itertools
    best = None
    for free in itertools.combinations(range(T), len(Cfg.skip)):
        for w in ([w0] if w0 else range(8, 28)):
            for svcv in (0, 1, 2):
                try:
                    g = Gen(T, w, free, Cfg.leak, svcv)
                    _, info = g.build()
                except AssertionError:
                    continue
                if info["skip"] != tuple(Cfg.skip):
                    continue
                key = (min(info["spare"], min(info["hspare"].values())), info["spare"])
                if best is None or key > best[0]:
                    best = (key, w, free, svcv)
    if best is None:
        sys.exit("no frame fits with skip %s" % (Cfg.skip,))
    _, w, free, svcv = best
    src, g, info = player_src(T, w, free, Cfg.leak, svcv, lay, m, pl.D)
    src = src.replace("LDX #X0", "LDX #%d" % int(codes[0]))
    open(os.path.join(OUT, "player.asm"), "w").write(src)
    txt = assemble("build/lprt/player.asm")
    symline = [l for l in txt.splitlines() if l.startswith("LPSYM")][0]
    sym = dict(re.findall(r"([a-z_0-9]+)&([0-9A-F]+)", symline.replace("LPSYM", "")))
    vecs = int(sym["vecs"], 16)
    code = open(os.path.join(OUT, "code.bin"), "rb").read()
    if verbose:
        print("player: %d bytes of code (room for %d); frame %d cycles, w0 %d, no write in slot %s (tick %s of the pattern); spare cycles: normal %d, others %d (%s)" % (
            len(code), 256 * CODEPAGES, 32 * T, w, list(free), list(info["skip"]), info["spare"], min(info["hspare"].values()), min(info["hspare"], key=info["hspare"].get)))
    vt = {v: (vecs & 255) + 2 * i for i, v in enumerate(g.vec)}
    assert (vecs & 255) + 2 * len(g.vec) <= 255
    # ---- the bank image
    img = bytearray(0x4000)
    img[:len(code)] = code
    o = lambda a: a - 0x8000
    have = [j for j in range(T) if j - 2 not in Cfg.skip]
    for i, j in enumerate(have):
        img[o(lay["TICK"]) + 256 * i:o(lay["TICK"]) + 256 * (i + 1)] = tabs[j].tobytes()
    clip = lprt.cliptab(Cfg)
    img[o(lay["CLIP"]):o(lay["CLIP"]) + 256] = bytes((lay["DUAL"] >> 8) + int(v) for v in clip)
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
        if i == sil:
            addr, st, vi = Cfg.main0, 0, 0
        else:
            addr, st, vi = sg["smp"][s]["addr"], lprt.step16(sg, Cfg, s, p), pl.vix[v]
        assert st < 65536
        img[nt + i] = addr & 255
        img[nt + 256 + i] = addr >> 8
        img[nt + 512 + i] = st & 255
        img[nt + 768 + i] = st >> 8
        img[nt + 1024 + i] = (lay["DUAL"] >> 8) + vi
    nbytes = 0
    for pi, recs in enumerate(pl.pages):
        a, cap = lay["pages"][pi]
        b = bytearray()
        for r in recs:
            k = r[0]
            if k == "WAIT":
                hi, n1 = lprt.holds(r[1])
                lo = 255 - n1
                assert 0 <= hi < 128
                b += bytes([vt["p_WAIT_N"], hi, lo])
            elif k == "GOTO":
                b += bytes([vt["p_GOTO_C"], r[1]])
            elif k == "NOP":
                b += bytes([vt["p_NOP_" + ("L" if r[-1] else "C")], 0])
            else:
                c, arg, last = r[1], r[2], r[3]
                if k == "NOTE" and arg < 0:
                    arg = sil
                b += bytes([vt["p_%s%d_%s" % (k, c, "L" if last else "C")], arg])
        assert len(b) <= cap, (pi, len(b), cap)
        img[o(a):o(a) + len(b)] = b
        nbytes += len(b)
    if verbose:
        print("song: %d records in %d pages (%d bytes; room for %d); samples %d bytes (room for %d); %d notes, %d volumes" % (
            len(pl.stream), len(pl.pages), nbytes, sum(caps), len(sg["mem"]), nmain, len(sg["notes"]), nvol))
        print("      %d frames: %.1f s" % (len(sums), len(sums) * T / FS))
    # ---- the files
    main = bytearray(sg["mem"]) + bytes([lprt.SIL] * (nmain - len(sg["mem"])))
    lo = bytes(main[:LOSPLIT - Cfg.main0])
    hi = bytes(main[LOSPLIT - Cfg.main0:])
    open(os.path.join(OUT, "lpload_cfg.asm"), "w").write("LOLEN = %d\nLO0 = &%04X\n" % (len(lo), Cfg.main0))
    assemble("src/lprt/lpload.asm")
    import ssd
    files = [("!BOOT", 0, 0, b"*RUN LPLOAD\r"),
             ("LPLOAD", 0x0900, 0x0900, open(os.path.join(OUT, "lpload.bin"), "rb").read()),
             ("LPH", 0x2000, 0x2000, bytes(img)),
             ("LPLO", 0x2000, 0x2000, lo),
             ("LPM", LOSPLIT, LOSPLIT, hi)]
    disc = os.path.join(OUT, "lprt.ssd")
    nsec = ssd.make(disc, "LPRT", 3, files)
    shutil.copyfile(disc, os.path.join(ROOT, "electron-lost-patrol-rt.ssd"))
    if verbose:
        print("disc: %d sectors -> electron-lost-patrol-rt.ssd" % nsec)
    # ---- what the emulator should show
    np.save(os.path.join(OUT, "bits.npy"), bits)
    np.save(os.path.join(OUT, "sums.npy"), sums.astype(np.uint8))
    # twice round, as the Electron goes on to play it (the modulator's state carries over)
    s2 = lprt.replay(sg, Cfg, pl, 2 * len(sums))
    assert (s2[:len(sums)] == sums).all(), "the replayed stream differs from the scheduler's run"
    b2, c2 = lprt.modulate(m, lprt.cliptab(Cfg)[s2], Cfg)
    exp = tabs[:, c2].T.astype(np.int16)               # frame f's writes follow its LDX, a tick apart
    exp[:, [k + 2 for k in Cfg.skip]] = -1             # (but for the ticks with none)
    exp = exp.reshape(-1)
    np.save(os.path.join(OUT, "expect.npy"), exp)
    json.dump({"T": T, "frames": int(len(sums)), "w0": w, "free": list(free), "skip": list(Cfg.skip)}, open(os.path.join(OUT, "build.json"), "w"))
    if check:
        emu(secs or (len(sums) * T / FS + 12), exp, T)
    return sg, pl, sums, xi, m, bits, codes


def emu(secs, exp, T, layout_env=True):
    log = os.path.join(OUT, "emu.log")
    env = dict(os.environ)
    if layout_env:
        env["ELKSDLAYOUT"] = "1"
        env["SWFILL"] = "rand:7"
    subprocess.run([CLKH, "--rompath", ROMS, "--media", os.path.join(OUT, "lprt.ssd"), "--shiftbreak", "1000000",
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
    for b in bad[:12]:
        print("   write %d: %d cycles after the one before" % (b, t[b] - t[b - 1]))
    if len(bad):
        return False
    got = np.full(n[-1] // 32 + 1, -1, np.int16)
    got[n // 32] = v
    # align: the first writes use the initial X until the first LDX
    for off in range(0, 3 * T):
        m = min(len(got) - off, len(exp))
        if m > 4000 and (got[off:off + 2000] == exp[:2000]).all():
            mm = np.flatnonzero(got[off:off + m] != exp[:m])
            print("values: offset %d ticks; %d of %d ticks differ (%d writes, %d ticks with none)%s" % (
                off, len(mm), m, (exp[:m] >= 0).sum(), (exp[:m] < 0).sum(),
                "" if not len(mm) else " (first at tick %d, frame %d)" % (mm[0], mm[0] // T)))
            return not len(mm)
    print("values: no alignment found; first ticks", got[:40].tolist(), "expected", exp[:27].tolist())
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--secs", type=float)
    ap.add_argument("--w0", type=int)
    a = ap.parse_args()
    build(a.check, a.secs, a.w0)
