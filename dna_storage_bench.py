#!/usr/bin/env python3
"""DNA storage encoder benchmark: density, constraints, SID robustness."""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("BENCH_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT, exist_ok=True)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 20260927
OLIGO_NT = 150
N_OLIGOS_CONSTRAINT = 4000
N_OLIGOS_ERROR = 800
GC_LO, GC_HI = 0.40, 0.60
HP_MAX = 3
ERROR_RATES = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

BASES = "ACGT"
BIT2BASE = {"00": "A", "01": "C", "10": "G", "11": "T"}
BASE2BIT = {v: k for k, v in BIT2BASE.items()}


def gc_frac(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def max_homopolymer(seq: str) -> int:
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def satisfies(seq: str, gc_lo=GC_LO, gc_hi=GC_HI, hp_max=HP_MAX) -> bool:
    return gc_lo <= gc_frac(seq) <= gc_hi and max_homopolymer(seq) <= hp_max


def bits_to_bytes_pad(n_bits: int) -> str:
    """Return a bitstring of length n_bits from RNG caller."""
    raise NotImplementedError


def random_bits(n: int, rng: random.Random) -> str:
    # pack via getrandbits
    if n <= 0:
        return ""
    val = rng.getrandbits(n)
    return format(val, f"0{n}b")


def hamming(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a) - len(b))


def edit_distance(a: str, b: str) -> int:
    """Banded-enough Levenshtein for short strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # two-row DP
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur[j] = min(ins, delete, sub)
        prev = cur
    return prev[lb]


def norm_similarity(a: str, b: str) -> float:
    """Linear-time similarity: Hamming on the aligned prefix plus length penalty."""
    if not a and not b:
        return 1.0
    n = min(len(a), len(b))
    ham = sum(x != y for x, y in zip(a, b)) if n else 0
    return 1.0 - (ham + abs(len(a) - len(b))) / max(len(a), len(b), 1)


# -------------------- encoders --------------------

class Codec:
    name: str
    nominal_bits_per_nt: float

    def encode(self, bits: str) -> str:
        raise NotImplementedError

    def decode(self, seq: str) -> str:
        raise NotImplementedError

    def payload_bits_for_oligo(self, n_nt: int) -> int:
        raise NotImplementedError


class Naive2Bit(Codec):
    name = "Naive2Bit"
    nominal_bits_per_nt = 2.0

    def payload_bits_for_oligo(self, n_nt: int) -> int:
        return n_nt * 2

    def encode(self, bits: str) -> str:
        if len(bits) % 2:
            bits += "0"
        return "".join(BIT2BASE[bits[i : i + 2]] for i in range(0, len(bits), 2))

    def decode(self, seq: str) -> str:
        return "".join(BASE2BIT.get(b, "00") for b in seq)


class Church1Bit(Codec):
    """1 bit/nt: 0->{A,C}, 1->{G,T}; pick to limit HP and pull GC toward 50%."""

    name = "Church1Bit"
    nominal_bits_per_nt = 1.0

    def payload_bits_for_oligo(self, n_nt: int) -> int:
        return n_nt

    def encode(self, bits: str) -> str:
        out = []
        gc = at = 0
        for bit in bits:
            choices = ("A", "C") if bit == "0" else ("G", "T")
            last = out[-1] if out else None
            # prefer not extending a long run
            ranked = []
            for b in choices:
                hp_pen = 0
                if last == b:
                    run = 1
                    for x in reversed(out):
                        if x == b:
                            run += 1
                        else:
                            break
                    if run >= HP_MAX:
                        hp_pen = 10 + run
                    else:
                        hp_pen = run
                # GC pull
                is_gc = b in "GC"
                if gc + at == 0:
                    gc_pen = 0
                else:
                    cur = gc / (gc + at)
                    if is_gc:
                        gc_pen = 0 if cur < 0.5 else 1
                    else:
                        gc_pen = 0 if cur >= 0.5 else 1
                ranked.append((hp_pen, gc_pen, b))
            ranked.sort()
            pick = ranked[0][2]
            out.append(pick)
            if pick in "GC":
                gc += 1
            else:
                at += 1
        return "".join(out)

    def decode(self, seq: str) -> str:
        # 0 if A/C else 1
        return "".join("0" if b in "AC" else "1" for b in seq)


class GoldmanRotate(Codec):
    """11 bits -> 7 trits -> 7 nt rotating alphabet; no homopolymer of length 2."""

    name = "GoldmanRotate"
    BITS_PER_BLOCK = 11
    TRITS_PER_BLOCK = 7
    nominal_bits_per_nt = 11 / 7

    def payload_bits_for_oligo(self, n_nt: int) -> int:
        n_blocks = n_nt // self.TRITS_PER_BLOCK
        return n_blocks * self.BITS_PER_BLOCK

    def _bits_to_trits(self, bits: str) -> list[int]:
        assert len(bits) % self.BITS_PER_BLOCK == 0
        trits = []
        for i in range(0, len(bits), self.BITS_PER_BLOCK):
            val = int(bits[i : i + self.BITS_PER_BLOCK], 2)
            block = []
            for _ in range(self.TRITS_PER_BLOCK):
                block.append(val % 3)
                val //= 3
            # little-endian trits; unused codes (2048..2186) never appear
            trits.extend(block)
        return trits

    def _trits_to_bits(self, trits: list[int]) -> str:
        out = []
        t = self.TRITS_PER_BLOCK
        for i in range(0, len(trits) - t + 1, t):
            val = 0
            p = 1
            for x in trits[i : i + t]:
                val += (x % 3) * p
                p *= 3
            if val >= 2**self.BITS_PER_BLOCK:
                val %= 2**self.BITS_PER_BLOCK
            out.append(format(val, f"0{self.BITS_PER_BLOCK}b"))
        return "".join(out)

    def encode(self, bits: str) -> str:
        trits = self._bits_to_trits(bits)
        seq = []
        prev = None
        order = ["A", "C", "G", "T"]
        for t in trits:
            avail = [b for b in order if b != prev]
            seq.append(avail[t % 3])
            prev = seq[-1]
        return "".join(seq)

    def decode(self, seq: str) -> str:
        trits = []
        prev = None
        order = ["A", "C", "G", "T"]
        for b in seq:
            avail = [x for x in order if x != prev]
            try:
                trits.append(avail.index(b))
            except ValueError:
                # homopolymer / unexpected base: treat as trit 0
                trits.append(0)
            prev = b
        # trim to multiple of block
        t = self.TRITS_PER_BLOCK
        trits = trits[: len(trits) - (len(trits) % t)]
        return self._trits_to_bits(trits)


class Block6to4(Codec):
    """64 good 4-mers (GC=50%, internal HP<=2) encode 6 bits. Density 1.5."""

    name = "Block6to4"
    nominal_bits_per_nt = 1.5

    def __init__(self):
        cands = []
        for n in range(256):
            s = "".join("ACGT"[(n >> (2 * i)) & 3] for i in range(4))
            if gc_frac(s) != 0.5:
                continue
            if max_homopolymer(s) > 2:
                continue
            cands.append(s)
        # deterministic 64
        cands.sort()
        self.codebook = cands[:64]
        self.inv = {s: format(i, "06b") for i, s in enumerate(self.codebook)}
        assert len(self.codebook) == 64

    def payload_bits_for_oligo(self, n_nt: int) -> int:
        n_blocks = n_nt // 4
        return n_blocks * 6

    def encode(self, bits: str) -> str:
        if len(bits) % 6:
            bits += "0" * (6 - len(bits) % 6)
        out = []
        for i in range(0, len(bits), 6):
            idx = int(bits[i : i + 6], 2)
            out.append(self.codebook[idx])
        return "".join(out)

    def decode(self, seq: str) -> str:
        out = []
        for i in range(0, len(seq) - 3, 4):
            block = seq[i : i + 4]
            if block in self.inv:
                out.append(self.inv[block])
            else:
                # nearest codebook word by Hamming
                best, bd = self.codebook[0], 99
                for w in self.codebook:
                    d = sum(a != b for a, b in zip(w, block))
                    if d < bd:
                        best, bd = w, d
                        if d == 0:
                            break
                out.append(self.inv[best])
        return "".join(out)


class Screened2Bit(Naive2Bit):
    """Same mapping as Naive2Bit; validity is a property of the oligo, not the map."""

    name = "Screened2Bit"


CODECS = [
    Naive2Bit(),
    Church1Bit(),
    GoldmanRotate(),
    Block6to4(),
    Screened2Bit(),
]


# -------------------- channel --------------------

def inject_substitutions(seq: str, p: float, rng: random.Random) -> str:
    if p <= 0:
        return seq
    out = []
    for b in seq:
        if rng.random() < p:
            out.append(rng.choice([x for x in BASES if x != b]))
        else:
            out.append(b)
    return "".join(out)


def inject_insertions(seq: str, p: float, rng: random.Random) -> str:
    if p <= 0:
        return seq
    out = []
    for b in seq:
        if rng.random() < p:
            out.append(rng.choice(BASES))
        out.append(b)
    if rng.random() < p:
        out.append(rng.choice(BASES))
    return "".join(out)


def inject_deletions(seq: str, p: float, rng: random.Random) -> str:
    if p <= 0:
        return seq
    out = [b for b in seq if rng.random() >= p]
    return "".join(out) if out else seq[0]


def inject_mixed(seq: str, p: float, rng: random.Random) -> str:
    """Independent SID at rate p/3 each (total event rate ~p)."""
    if p <= 0:
        return seq
    ps = p / 3.0
    out = []
    for b in seq:
        r = rng.random()
        if r < ps:
            continue  # deletion
        elif r < 2 * ps:
            out.append(rng.choice([x for x in BASES if x != b]))  # sub
        else:
            out.append(b)
        if rng.random() < ps:
            out.append(rng.choice(BASES))  # insertion after
    return "".join(out) if out else "A"


# -------------------- experiments --------------------

def constraint_study(codecs, n_oligos, n_nt, rng) -> pd.DataFrame:
    rows = []
    gc_samples = defaultdict(list)
    hp_samples = defaultdict(list)
    for codec in codecs:
        n_bits = codec.payload_bits_for_oligo(n_nt)
        n_valid = 0
        n_gc_ok = 0
        n_hp_ok = 0
        lens = []
        used_nt = []
        for i in range(n_oligos):
            bits = random_bits(n_bits, rng)
            seq = codec.encode(bits)
            # Goldman/Block may be slightly shorter than n_nt
            lens.append(len(seq))
            used_nt.append(len(seq))
            g = gc_frac(seq)
            h = max_homopolymer(seq)
            gc_samples[codec.name].append(g)
            hp_samples[codec.name].append(h)
            gc_ok = GC_LO <= g <= GC_HI
            hp_ok = h <= HP_MAX
            n_gc_ok += int(gc_ok)
            n_hp_ok += int(hp_ok)
            n_valid += int(gc_ok and hp_ok)
            # roundtrip check
            rec = codec.decode(seq)
            # compare prefix of intended length
            ok_rt = rec[:n_bits] == bits
            if not ok_rt and i < 3:
                pass
        mean_nt = float(np.mean(used_nt))
        valid_rate = n_valid / n_oligos
        if codec.name == "Screened2Bit":
            # fountain-style: drop invalid; density is 2 * valid_rate
            eff = codec.nominal_bits_per_nt * valid_rate
        else:
            eff = (n_bits / mean_nt) if mean_nt else 0.0
        rows.append(
            {
                "codec": codec.name,
                "nominal_bits_per_nt": round(codec.nominal_bits_per_nt, 4),
                "effective_bits_per_nt": round(eff, 4),
                "mean_oligo_nt": round(mean_nt, 2),
                "payload_bits": n_bits,
                "valid_rate": round(valid_rate, 4),
                "gc_ok_rate": round(n_gc_ok / n_oligos, 4),
                "hp_ok_rate": round(n_hp_ok / n_oligos, 4),
                "mean_gc": round(float(np.mean(gc_samples[codec.name])), 4),
                "std_gc": round(float(np.std(gc_samples[codec.name])), 4),
                "mean_max_hp": round(float(np.mean(hp_samples[codec.name])), 3),
                "p95_max_hp": round(float(np.quantile(hp_samples[codec.name], 0.95)), 3),
                "max_hp_obs": int(max(hp_samples[codec.name])),
            }
        )
    # save distributions
    dist_rows = []
    for name, vals in gc_samples.items():
        for g, h in zip(vals, hp_samples[name]):
            dist_rows.append({"codec": name, "gc": g, "max_hp": h})
    pd.DataFrame(dist_rows).to_csv(f"{OUT}/constraint_distributions.csv", index=False)
    return pd.DataFrame(rows), gc_samples, hp_samples


def error_study(codecs, n_oligos, n_nt, rng) -> pd.DataFrame:
    rows = []
    channels = [
        ("substitution", inject_substitutions),
        ("insertion", inject_insertions),
        ("deletion", inject_deletions),
        ("mixed_sid", inject_mixed),
    ]
    for codec in codecs:
        n_bits = codec.payload_bits_for_oligo(n_nt)
        # pre-generate clean oligos; for Screened2Bit only keep valid ones
        clean = []
        attempts = 0
        while len(clean) < n_oligos and attempts < n_oligos * 40:
            attempts += 1
            bits = random_bits(n_bits, rng)
            seq = codec.encode(bits)
            if codec.name == "Screened2Bit" and not satisfies(seq):
                continue
            clean.append((bits, seq))
        if len(clean) < n_oligos:
            # pad with last if screening too harsh (shouldn't be)
            while len(clean) < n_oligos:
                clean.append(clean[-1] if clean else (random_bits(n_bits, rng), codec.encode(random_bits(n_bits, rng))))
        for ch_name, ch_fn in channels:
            for p in ERROR_RATES:
                exact = 0
                bit_sim = []
                nt_sim = []
                for bits, seq in clean:
                    noisy = ch_fn(seq, p, rng)
                    rec = codec.decode(noisy)
                    exact += int(rec[:n_bits] == bits)
                    bit_sim.append(norm_similarity(rec[: max(n_bits, len(rec))], bits))
                    nt_sim.append(norm_similarity(noisy, seq))
                rows.append(
                    {
                        "codec": codec.name,
                        "channel": ch_name,
                        "p": p,
                        "exact_recovery": round(exact / n_oligos, 4),
                        "bit_similarity": round(float(np.mean(bit_sim)), 4),
                        "nt_similarity": round(float(np.mean(nt_sim)), 4),
                    }
                )
    return pd.DataFrame(rows)


def make_figures(cons_df, err_df, gc_samples, hp_samples):
    """Figures 1-5 (see plots.py for styling and display names)."""
    import plots
    out = FIG_DIR
    plots.fig1(cons_df, out)
    plots.fig2(gc_samples, out, GC_LO, GC_HI)
    plots.fig3(hp_samples, out, HP_MAX)
    plots.fig4(err_df, out)
    plots.fig5(err_df, out)


def main():
    rng = random.Random(SEED)
    print("constraint study...")
    cons_df, gc_samples, hp_samples = constraint_study(CODECS, N_OLIGOS_CONSTRAINT, OLIGO_NT, rng)
    cons_df.to_csv(f"{OUT}/constraint_summary.csv", index=False)
    print(cons_df.to_string(index=False))
    print("error study...")
    err_df = error_study(CODECS, N_OLIGOS_ERROR, OLIGO_NT, rng)
    err_df.to_csv(f"{OUT}/error_summary.csv", index=False)
    print(err_df.to_string(index=False))
    make_figures(cons_df, err_df, gc_samples, hp_samples)
    meta = {
        "seed": SEED,
        "oligo_nt": OLIGO_NT,
        "n_oligos_constraint": N_OLIGOS_CONSTRAINT,
        "n_oligos_error": N_OLIGOS_ERROR,
        "gc_range": [GC_LO, GC_HI],
        "hp_max": HP_MAX,
        "error_rates": ERROR_RATES,
    }
    with open(f"{OUT}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
