#!/usr/bin/env python3
"""Block-level control for the comma experiment.

Splits every payload into 22-bit blocks and counts the fraction of blocks that
are decoded correctly and in place. For the unsegmented codecs (Naive2Bit,
GoldmanRotate) a block survives only if it lies before the first effective
indel; for MarkerResync the blocks are the comma-delimited segments. This puts
all three codecs on the same metric, so the comma's gain is measured against
an unsegmented rotating code rather than against whole-oligo recovery.
"""
from __future__ import annotations
import random, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dna_storage_bench as B
from extend_bench import MarkerResync

SEED = B.SEED + 2
N = 600
BLOCK = 22
L = 150
CHANNELS = {"substitution": B.inject_substitutions,
            "insertion": B.inject_insertions,
            "deletion": B.inject_deletions}
RATES = (0.0, 0.001, 0.005, 0.01, 0.02)

def main():
    rng = random.Random(SEED)
    rows = []
    for codec in (B.Naive2Bit(), B.GoldmanRotate(), MarkerResync()):
        n_bits = codec.payload_bits_for_oligo(L)
        clean = []
        for _ in range(N):
            bits = B.random_bits(n_bits, rng)
            clean.append((bits, codec.encode(bits)))
        nblk = n_bits // BLOCK
        for ch, fn in CHANNELS.items():
            for p in RATES:
                exact, blk = 0, []
                for bits, seq in clean:
                    noisy = fn(seq, p, rng)
                    exp = [bits[i*BLOCK:(i+1)*BLOCK] for i in range(nblk)]
                    if isinstance(codec, MarkerResync):
                        segs = codec.decode_segments(noisy)
                        hit = sum(int(g == e) for e, g in zip(exp, segs))
                        rec = "".join(s or "" for s in segs)
                        ok = rec[:n_bits] == bits and len(segs) == nblk and all(s is not None for s in segs)
                    else:
                        rec = codec.decode(noisy)
                        hit = sum(int(rec[i*BLOCK:(i+1)*BLOCK] == exp[i]) for i in range(nblk))
                        ok = rec[:n_bits] == bits
                    exact += int(ok)
                    blk.append(hit / nblk)
                rows.append({"codec": codec.name, "channel": ch, "p": p,
                             "exact_recovery": round(exact / N, 4),
                             "block_recovery": round(sum(blk) / N, 4),
                             "bits_per_nt": round(n_bits / (sum(len(s) for _, s in clean) / N), 4),
                             "n_blocks": nblk})
    df = pd.DataFrame(rows)
    df.to_csv(Path(B.OUT) / "comma_control.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
