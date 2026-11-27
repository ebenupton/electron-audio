#!/usr/bin/env python3
"""Wait for the ElkSD card (/Volumes/ELKSD), back up BEEB.MMB, write
a disc image (default electron-lost-patrol-hybrid.ssd) into an MMFS slot (default
240), titled from the image, verify, eject.

  sdupdate.py [--image X.ssd] [--slot 240] [--wait 3600] [--no-eject]
"""
import argparse, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SZ = 204800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=240)
    ap.add_argument("--wait", type=float, default=3600)
    ap.add_argument("--image", default=os.path.join(ROOT, "electron-lost-patrol-hybrid.ssd"))
    ap.add_argument("--no-eject", action="store_true", help="leave the card mounted (for another write)")
    a = ap.parse_args()
    mmb = "/Volumes/ELKSD/BEEB.MMB"
    t0 = time.time()
    while not os.path.exists(mmb):
        if time.time() - t0 > a.wait:
            sys.exit("no card")
        time.sleep(2)
    time.sleep(2)
    print(time.strftime("%H:%M:%S"), "card found", flush=True)
    bak = os.path.join(ROOT, "sdcard-backup", "BEEB.MMB." + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.dirname(bak), exist_ok=True)
    shutil.copyfile(mmb, bak)
    img = open(a.image, "rb").read()
    assert len(img) <= SZ
    title = (img[0:8] + img[256:260]).rstrip(b"\0 ")[:12]
    with open(mmb, "r+b") as f:
        f.seek(16 + 16 * a.slot)
        e = f.read(16)
        name = e[:12].rstrip(b"\0")
        if e[15] != 0xF0 and name != title:
            sys.exit("slot %d holds %r, not %r: not overwriting" % (a.slot, name, title))
        f.seek(8192 + a.slot * SZ)
        f.write(img + b"\0" * (SZ - len(img)))
        f.seek(16 + 16 * a.slot)
        f.write(title.ljust(12, b"\0") + b"\0\0\0\x0f")
    os.sync()
    with open(mmb, "rb") as f:
        f.seek(8192 + a.slot * SZ)
        ok = f.read(len(img)) == img
    print(time.strftime("%H:%M:%S"), "slot %d %s; backup %s" % (a.slot, "written and verified" if ok else "MISMATCH", bak), flush=True)
    if not a.no_eject:
        subprocess.run(["diskutil", "eject", "/Volumes/ELKSD"])


if __name__ == "__main__":
    main()
