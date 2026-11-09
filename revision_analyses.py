#!/usr/bin/env python3
"""Camera-ready analyses (reviewer responses).

1. Re-runs Tables I and III with the original seed and adds an alignment-based (edit-distance) bit similarity.
   The random stream is consumed exactly as in dna_storage_bench.py, so the original numbers are reproduced.
2. CommaRotate in the constraint table, the length/window sweep, and the SID table (separate seed 20260930).
3. Re-runs the comma experiment (seed 20260929) with a strict comma decoder that uses the known segment length,
   and cluster-bootstrap 95% intervals for block recovery (resampling oligos).
4. Wilson 95% intervals for every proportion.
"""
from __future__ import annotations
import math, random, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dna_storage_bench as B
from extend_bench import MarkerResync, WINDOWS, LENGTHS
from editdist import edit_similarity

OUT = Path(B.OUT)
CHANNELS = [("substitution", B.inject_substitutions), ("insertion", B.inject_insertions),
            ("deletion", B.inject_deletions), ("mixed_sid", B.inject_mixed)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def error_rows(codec, clean, n_bits, rng, rates, decode=None):
    decode = decode or codec.decode
    rows = []
    for ch_name, ch_fn in CHANNELS:
        for p in rates:
            exact, pos, edt = 0, [], []
            for bits, seq in clean:
                noisy = ch_fn(seq, p, rng)
                rec = decode(noisy)
                exact += int(rec[:n_bits] == bits)
                pos.append(B.norm_similarity(rec[: max(n_bits, len(rec))], bits))
                edt.append(edit_similarity(rec, bits))
            n = len(clean)
            lo, hi = wilson(exact, n)
            rows.append({"codec": codec.name, "channel": ch_name, "p": p, "n": n,
                         "exact_recovery": round(exact / n, 4), "exact_ci_lo": round(lo, 4), "exact_ci_hi": round(hi, 4),
                         "bit_similarity": round(float(np.mean(pos)), 4),
                         "edit_similarity": round(float(np.mean(edt)), 4)})
    return rows


# ---------- 1. original Tables I and III, plus edit similarity ----------
def part1():
    rng = random.Random(B.SEED)
    cons, _, _ = B.constraint_study(B.CODECS, B.N_OLIGOS_CONSTRAINT, B.OLIGO_NT, rng)
    rows = []
    for codec in B.CODECS:  # identical generation loop to B.error_study
        n_bits = codec.payload_bits_for_oligo(B.OLIGO_NT)
        clean, attempts = [], 0
        while len(clean) < B.N_OLIGOS_ERROR and attempts < B.N_OLIGOS_ERROR * 40:
            attempts += 1
            bits = B.random_bits(n_bits, rng)
            seq = codec.encode(bits)
            if codec.name == "Screened2Bit" and not B.satisfies(seq):
                continue
            clean.append((bits, seq))
        rows += error_rows(codec, clean, n_bits, rng, B.ERROR_RATES)
    err = pd.DataFrame(rows)
    return cons, err


# ---------- 2. CommaRotate in Tables I-III ----------
def part2():
    rng = random.Random(20260930)
    codec = MarkerResync()
    cons_rows, sweep_rows = [], []
    for L in LENGTHS:
        n_bits = codec.payload_bits_for_oligo(L)
        n = B.N_OLIGOS_CONSTRAINT if L == 150 else 2000
        seqs = [codec.encode(B.random_bits(n_bits, rng)) for _ in range(n)]
        gcs = [B.gc_frac(s) for s in seqs]; hps = [B.max_homopolymer(s) for s in seqs]
        for label, lo, hi, hp in WINDOWS:
            v = sum(lo <= g <= hi and h <= hp for g, h in zip(gcs, hps))
            sweep_rows.append({"codec": codec.name, "L": L, "window": label, "n": n, "valid_rate": round(v / n, 4),
                               "bits_per_nt": round(n_bits / np.mean([len(s) for s in seqs]), 4), "mean_nt": float(np.mean([len(s) for s in seqs])),
                               "payload_bits": n_bits})
        if L == 150:
            v = sum(B.GC_LO <= g <= B.GC_HI and h <= B.HP_MAX for g, h in zip(gcs, hps))
            cons_rows.append({"codec": codec.name, "nominal_bits_per_nt": round(22 / 15, 4), "payload_bits": n_bits,
                              "mean_oligo_nt": float(np.mean([len(s) for s in seqs])), "n": n,
                              "valid_rate": round(v / n, 4), "gc_ok_rate": round(np.mean([B.GC_LO <= g <= B.GC_HI for g in gcs]), 4),
                              "hp_ok_rate": round(np.mean([h <= B.HP_MAX for h in hps]), 4),
                              "mean_gc": round(float(np.mean(gcs)), 4), "std_gc": round(float(np.std(gcs)), 4),
                              "mean_max_hp": round(float(np.mean(hps)), 3), "max_hp_obs": int(max(hps))})
    n_bits = codec.payload_bits_for_oligo(150)
    clean = []
    for _ in range(B.N_OLIGOS_ERROR):
        bits = B.random_bits(n_bits, rng)
        clean.append((bits, codec.encode(bits)))
    err = pd.DataFrame(error_rows(codec, clean, n_bits, rng, B.ERROR_RATES))
    return pd.DataFrame(cons_rows), pd.DataFrame(sweep_rows), err


# ---------- 3. comma experiment with strict decoder ----------
def strict_segments(codec: MarkerResync, seq: str, n_seg: int, tol: int = 2):
    """Parse segments sequentially, accepting a doubled base as a comma only within +-tol of the expected
    position (14 nt after the segment start). Doubled bases elsewhere are ignored. If no comma is found,
    assume the comma was erased and step 15 nt."""
    step = codec._bits_per_seg()
    P = codec.PAYLOAD_NT
    out, s = [], 0
    while len(out) < n_seg and s < len(seq):
        cands = [j for j in range(max(s + 1, s + P - tol), min(len(seq), s + P + tol + 1)) if seq[j] == seq[j - 1]]
        prev = seq[s - 1] if s > 0 else None
        if cands:
            j = min(cands, key=lambda x: (abs(x - (s + P)), x))
            payload = seq[s:j]
            out.append(codec._decode_seg(payload, prev)[:step] if len(payload) == P else None)
            s = j + 1
        else:
            payload = seq[s:s + P]
            out.append(codec._decode_seg(payload, prev)[:step] if len(payload) == P else None)
            s += P + 1
    out += [None] * (n_seg - len(out))
    return out


def cluster_ci(per_oligo, n_boot=2000, seed=0):
    r = np.random.default_rng(seed)
    a = np.asarray(per_oligo)
    m = a[r.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def part3():
    import comma_control as C
    rng = random.Random(C.SEED)
    rows = []
    for codec in (B.Naive2Bit(), B.GoldmanRotate(), MarkerResync()):
        n_bits = codec.payload_bits_for_oligo(C.L)
        clean = []
        for _ in range(C.N):
            bits = B.random_bits(n_bits, rng)
            clean.append((bits, codec.encode(bits)))
        nblk = n_bits // C.BLOCK
        for ch, fn in C.CHANNELS.items():
            for p in C.RATES:
                exact, blk, blk_strict, exact_strict = 0, [], [], 0
                for bits, seq in clean:
                    noisy = fn(seq, p, rng)
                    exp = [bits[i * C.BLOCK:(i + 1) * C.BLOCK] for i in range(nblk)]
                    if isinstance(codec, MarkerResync):
                        segs = codec.decode_segments(noisy)
                        hit = sum(int(g == e) for e, g in zip(exp, segs))
                        rec = "".join(s or "" for s in segs)
                        ok = rec[:n_bits] == bits and len(segs) == nblk and all(s is not None for s in segs)
                        ss = strict_segments(codec, noisy, nblk)
                        hs = sum(int(g == e) for e, g in zip(exp, ss))
                        blk_strict.append(hs / nblk)
                        exact_strict += int(hs == nblk)
                    else:
                        rec = codec.decode(noisy)
                        hit = sum(int(rec[i * C.BLOCK:(i + 1) * C.BLOCK] == exp[i]) for i in range(nblk))
                        ok = rec[:n_bits] == bits
                    exact += int(ok)
                    blk.append(hit / nblk)
                lo, hi = cluster_ci(blk)
                row = {"codec": codec.name, "channel": ch, "p": p, "n": C.N, "n_blocks": nblk,
                       "exact_recovery": round(exact / C.N, 4), "block_recovery": round(float(np.mean(blk)), 4),
                       "block_ci_lo": round(lo, 4), "block_ci_hi": round(hi, 4)}
                if blk_strict:
                    lo2, hi2 = cluster_ci(blk_strict)
                    row.update({"block_recovery_strict": round(float(np.mean(blk_strict)), 4),
                                "block_strict_ci_lo": round(lo2, 4), "block_strict_ci_hi": round(hi2, 4),
                                "exact_recovery_strict": round(exact_strict / C.N, 4)})
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    cons, err = part1()
    cons.to_csv(OUT / "rev_constraint_summary.csv", index=False)
    err.to_csv(OUT / "rev_error_summary.csv", index=False)
    cc, cs, ce = part2()
    cc.to_csv(OUT / "rev_comma_constraint.csv", index=False)
    cs.to_csv(OUT / "rev_comma_sweep.csv", index=False)
    ce.to_csv(OUT / "rev_comma_error.csv", index=False)
    cm = part3()
    cm.to_csv(OUT / "rev_comma_control.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
