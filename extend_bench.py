#!/usr/bin/env python3
"""Extra workshop experiments: constraint sensitivity + marker resync.

Does not use an LLM. Same seed family as dna_storage_bench.py.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dna_storage_bench as B

OUT = Path(B.OUT)
N_SENS = 2000
N_ERR = 600
LENGTHS = (100, 150, 200)
WINDOWS = (
    ("GC40-60_HP3", 0.40, 0.60, 3),
    ("GC45-55_HP3", 0.45, 0.55, 3),
    ("GC40-60_HP2", 0.40, 0.60, 2),
    ("GC40-60_HP4", 0.40, 0.60, 4),
)


class MarkerResync(B.Codec):
    """Goldman segments delimited by repeating the last base once.

    Rotation never emits a repeat, so a length-2 run is a comma that cannot
    occur in the payload. The next segment is encoded with that repeated base
    as the forbidden predecessor, so the comma does not merge with the next
    symbol. One indel breaks one comma and at most two segments.
    Density: 22 bits / 15 nt ≈ 1.467 bits/nt. Ten segments fill L=150 exactly.
    """

    name = "MarkerResync"
    nominal_bits_per_nt = 22 / 15
    PAYLOAD_NT = 14
    inner = None

    def __init__(self):
        self.inner = B.GoldmanRotate()

    def _bits_per_seg(self) -> int:
        return self.inner.payload_bits_for_oligo(self.PAYLOAD_NT)

    def payload_bits_for_oligo(self, n_nt: int) -> int:
        return (n_nt // (self.PAYLOAD_NT + 1)) * self._bits_per_seg()

    def _encode_seg(self, bits: str, prev: str | None) -> tuple[str, str]:
        trits = self.inner._bits_to_trits(bits)
        seq = []
        order = ["A", "C", "G", "T"]
        for t in trits:
            avail = [b for b in order if b != prev]
            seq.append(avail[t % 3])
            prev = seq[-1]
        return "".join(seq), prev

    def _decode_seg(self, seq: str, prev: str | None) -> str:
        trits = []
        order = ["A", "C", "G", "T"]
        for b in seq:
            avail = [x for x in order if x != prev]
            try:
                trits.append(avail.index(b))
            except ValueError:
                trits.append(0)
            prev = b
        t = self.inner.TRITS_PER_BLOCK
        trits = trits[: len(trits) - (len(trits) % t)]
        return self.inner._trits_to_bits(trits)

    def encode(self, bits: str) -> str:
        step = self._bits_per_seg()
        out = []
        prev = None
        for i in range(0, len(bits) - step + 1, step):
            chunk, last = self._encode_seg(bits[i : i + step], prev)
            if len(chunk) != self.PAYLOAD_NT:
                raise RuntimeError(f"segment length {len(chunk)}")
            out.append(chunk)
            out.append(last)  # deliberate repeat: the comma
            prev = last
        return "".join(out)

    def decode_segments(self, seq: str) -> list[str | None]:
        step = self._bits_per_seg()
        got: list[str | None] = []
        buf: list[str] = []
        prev = None
        for b in seq:
            if buf and b == buf[-1]:
                payload = "".join(buf)
                if len(payload) == self.PAYLOAD_NT:
                    got.append(self._decode_seg(payload, prev)[:step])
                else:
                    got.append(None)
                prev = b
                buf = []
            else:
                buf.append(b)
        return got

    def decode(self, seq: str) -> str:
        step = self._bits_per_seg()
        return "".join(s if s is not None else "?" * step for s in self.decode_segments(seq))


def sensitivity(rng: random.Random) -> pd.DataFrame:
    codecs = [B.Naive2Bit(), B.Church1Bit(), B.GoldmanRotate(), B.Block6to4()]
    rows = []
    for n_nt in LENGTHS:
        generated = {}
        for codec in codecs:
            n_bits = codec.payload_bits_for_oligo(n_nt)
            seqs = []
            for _ in range(N_SENS):
                seqs.append(codec.encode(B.random_bits(n_bits, rng)))
            generated[codec.name] = (n_bits, seqs)
        for label, lo, hi, hp in WINDOWS:
            for codec in codecs:
                n_bits, seqs = generated[codec.name]
                n = len(seqs)
                valid = gc_ok = hp_ok = 0
                for seq in seqs:
                    g = B.gc_frac(seq)
                    h = B.max_homopolymer(seq)
                    gok = lo <= g <= hi
                    hok = h <= hp
                    gc_ok += int(gok)
                    hp_ok += int(hok)
                    valid += int(gok and hok)
                mean_nt = sum(len(s) for s in seqs) / n
                rows.append(
                    {
                        "codec": codec.name,
                        "L": n_nt,
                        "window": label,
                        "valid_rate": round(valid / n, 4),
                        "gc_ok": round(gc_ok / n, 4),
                        "hp_ok": round(hp_ok / n, 4),
                        "bits_per_nt": round(n_bits / mean_nt, 4),
                    }
                )
    return pd.DataFrame(rows)


def marker_vs_naive(rng: random.Random) -> pd.DataFrame:
    codecs = [B.Naive2Bit(), MarkerResync()]
    n_nt = 150
    rows = []
    channels = {
        "substitution": B.inject_substitutions,
        "insertion": B.inject_insertions,
        "deletion": B.inject_deletions,
    }
    for codec in codecs:
        n_bits = codec.payload_bits_for_oligo(n_nt)
        clean = []
        for _ in range(N_ERR):
            bits = B.random_bits(n_bits, rng)
            seq = codec.encode(bits)
            clean.append((bits, seq))
        mean_nt = sum(len(s) for _, s in clean) / len(clean)
        for ch, fn in channels.items():
            for p in (0.0, 0.001, 0.01, 0.02):
                exact = 0
                seg_rec = []
                for bits, seq in clean:
                    noisy = fn(seq, p, rng)
                    if isinstance(codec, MarkerResync):
                        segs = codec.decode_segments(noisy)
                        step = codec._bits_per_seg()
                        nseg = len(bits) // step
                        exp = [bits[i : i + step] for i in range(0, nseg * step, step)]
                        # compare in order, only while both exist
                        hit = 0
                        for e, got in zip(exp, segs):
                            if got == e:
                                hit += 1
                        # leftover expected segments count as misses
                        seg_rec.append(hit / nseg)
                        rec_bits = "".join(s or "" for s in segs)
                        exact += int(rec_bits[:n_bits] == bits and len(segs) == nseg and all(x is not None for x in segs))
                    else:
                        rec = codec.decode(noisy)
                        exact += int(rec[:n_bits] == bits)
                        seg_rec.append(1.0 if rec[:n_bits] == bits else 0.0)
                rows.append(
                    {
                        "codec": codec.name,
                        "channel": ch,
                        "p": p,
                        "exact_recovery": round(exact / N_ERR, 4),
                        "segment_recovery": round(sum(seg_rec) / len(seg_rec), 4),
                        "bits_per_nt": round(n_bits / mean_nt, 4),
                        "mean_nt": round(mean_nt, 2),
                    }
                )
    return pd.DataFrame(rows)


def main():
    rng = random.Random(B.SEED + 1)
    sens = sensitivity(rng)
    sens.to_csv(OUT / "sensitivity_summary.csv", index=False)
    print("=== sensitivity (valid rate) ===")
    pivot = sens.pivot_table(index=["codec", "L"], columns="window", values="valid_rate")
    print(pivot.to_string())
    mark = marker_vs_naive(rng)
    mark.to_csv(OUT / "marker_resync.csv", index=False)
    print("\n=== marker vs naive ===")
    print(mark.to_string(index=False))


if __name__ == "__main__":
    main()
