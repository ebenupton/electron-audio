# electron-audio

Sampled music on an unexpanded-sound Acorn Electron: the theme from *Lost
Patrol* (a 4-channel Amiga MOD) played through the ULA's single square-wave
channel.

The ULA's sound output is a flip-flop that toggles each time a counter runs
out, and the counter is reloaded from `&FE06` at that moment. The counter
ticks every 16 µs, so writing `&FE06` once a tick with "the length of the
next run, less one" plays any 62.5 kHz 1-bit stream exactly. Everything here
is built on that.

There are three discs, all for an Electron with sideways RAM (developed on an
ElkSD128, which has seven free 16K banks) and all cycle-exact: code runs from
sideways RAM at 2 MHz with interrupts off, and every path through it is
padded so the writes land 32 cycles apart.

| Disc | What it plays | How |
|---|---|---|
| `electron-lost-patrol-sdm3c.ssd` | 16.3 s of the tune, looping | A stored bit stream, computed offline |
| `electron-lost-patrol-rt.ssd` | The whole tune (175 s), looping | Decoded as it plays: a 4-channel tracker |
| `electron-lost-patrol-hybrid.ssd` | The whole tune, looping | The first 10 s stored, the rest decoded |

Boot any of them with SHIFT+BREAK. The screen shows garbage while they play:
screen memory holds sound data.

## The stored bit stream

`tools/sdmbuild.py` renders the MOD at 62.5 kHz, compresses and limits it,
and turns it into bits with a noise-shaping modulator weighted by the measured
response of the Electron's speaker and by the ear (`tools/sdmhq.py`,
`tools/psy.py`). The player keeps the next eight bits in a register and looks
the run length up from them each tick. A second holds 7.8 KB, so seven banks
and main RAM give 16.3 s.

    python3 tools/sdmbuild.py --mode static --dyn=-20,3,9 --title SDMLP3C --out electron-lost-patrol-sdm3c.ssd

## The tracker

`tools/lprt.py` is the design and a bit-exact model; `tools/lprtbuild.py`
generates the 6502.

- A frame is 8 ticks (256 cycles, 7812 Hz). In it four channels' sample
  pointers are stepped, a sample is fetched for each from main RAM, scaled
  through a volume table and summed.
- The sum goes through a soft-clip table into a table-driven modulator: a
  64-state machine that picks one of 255 eight-tick bit patterns. The
  pattern's run lengths come from tables indexed by its code.
- Seven of the eight ticks have a write. A tick without one only requires that
  a run starting there is as long as the one before, which costs little and
  buys the cycles that make an 8-tick frame fit.
- The song is compiled to an event stream held in the bank. Each event takes
  one frame in which the pointers stand still.
- Samples are high-passed at 450 Hz (the speaker gives nothing below) and fill
  main RAM from `&0400`; the player, its tables and the song fill one bank.

    python3 tools/lprtbuild.py --check

`--check` runs the disc in an emulator and compares every `&FE06` write, in
time and value, with the model.

## The hybrid

The tracker is weakest where the music is thin and quiet, which is the first
63 rows. `tools/lphybrid.py` stores those as a bit stream in six banks and
starts the tracker, in a seventh, at row 63. Both write on the same 16 µs
grid and each hands over to the other between two writes, so the join is
seamless and the tune loops.

    python3 tools/lphybrid.py
    python3 tools/lphybrid.py --row 52 --nbank 5 --check --secs 250

The second form is for an emulator with six free banks: a shorter stored
intro, checked over more than one pass so that both handovers are covered.

## Building

Needs Python 3 with numpy, scipy and numba, and
[beebasm](https://github.com/stardot/beebasm). Paths are taken from the
environment, with these defaults:

| Variable | Default | What |
|---|---|---|
| `BEEBASM` | `~/elite-electron-tools/beebasm` | the assembler |
| `CLKH` | `~/elite-electron-elksd/6-harness/clkh/build/clkh` | headless Electron emulator, for `--check` only |
| `ELK_ROMS` | `~/elite-electron-roms` | MOS and BASIC ROMs for the emulator |

`tools/sdupdate.py --image X.ssd --slot N` writes a disc into a slot of an
ElkSD card's `BEEB.MMB`, backing the file up first.

## Layout

    tools/modfile.py    reads the MOD
    tools/modrender.py  renders it at 62.5 kHz
    tools/sdmhq.py      modulators and the speaker/ear weighting
    tools/psy.py        dynamics, and a psychoacoustic variant of the shaping
    tools/sdmbuild.py   the stored-stream disc
    tools/sdmcheck.py   checks an emulator log of it
    tools/lprt.py       the tracker: design and model
    tools/lprtbuild.py  the tracker disc
    tools/lphybrid.py   the hybrid disc
    tools/ssd.py        writes DFS disc images
    src/                the three loaders
    data/               the measured speaker response
    mods/               the tune

## What to expect

Measured in the model against the stored stream, over the same 30 seconds:
the tracker is about 13 dB worse on a speaker-and-ear-weighted measure and
about 10 dB worse from 300 to 3500 Hz. Three quarters of that is bandwidth: a
7.8 kHz mix cannot carry the 3.9 to 7 kHz that the stream can, so the tracker
sounds duller rather than noisier.

## Credits

The tune is the theme from *Lost Patrol* (Ocean, 1990), music by Chris
Glaister. It is included so that the discs can be rebuilt; the rights to it
are not mine. The idea of high-rate sample playback on Acorn sound hardware
owes a great deal to Chris Evans's (scarybeasts) work on the BBC Micro.
