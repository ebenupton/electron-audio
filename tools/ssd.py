#!/usr/bin/env python3
"""Write a DFS .ssd: ssd.py OUT.ssd TITLE BOOTOPT  NAME:LOAD:EXEC:HOSTFILE ..."""
import sys


def make(out, title, boot, files):
    img = bytearray(200 * 1024)
    sector = 2
    cat0, cat1 = bytearray(256), bytearray(256)
    t = title.encode().ljust(12)[:12]
    cat0[0:8] = t[:8]
    cat1[0:4] = t[8:12]
    for i, (name, load, exe, data) in enumerate(files):
        n = name.encode().ljust(7)
        cat0[8 + 8 * i:15 + 8 * i] = n[:7]
        cat0[15 + 8 * i] = ord("$")
        ln = len(data)
        img[sector * 256:sector * 256 + ln] = data
        e = 8 + 8 * i
        cat1[e:e + 8] = bytes([load & 255, (load >> 8) & 255, exe & 255, (exe >> 8) & 255, ln & 255, (ln >> 8) & 255,
                               (((exe >> 16) & 3) << 6) | (((ln >> 16) & 3) << 4) | (((load >> 16) & 3) << 2) | ((sector >> 8) & 3),
                               sector & 255])
        sector += (ln + 255) // 256
    cat1[5] = len(files) * 8
    cat1[6] = (boot & 3) << 4 | ((800 >> 8) & 3)
    cat1[7] = 800 & 255
    img[0:256], img[256:512] = cat0, cat1
    open(out, "wb").write(bytes(img[:sector * 256]))
    return sector


if __name__ == "__main__":
    out, title, boot = sys.argv[1], sys.argv[2], int(sys.argv[3])
    files = []
    for spec in sys.argv[4:]:
        name, load, exe, host = spec.split(":", 3)
        files.append((name, int(load, 16), int(exe, 16), open(host, "rb").read()))
    print("%s: %d sectors" % (out, make(out, title, boot, files)))
