"""Capacity of the HP<=k constraint (transfer matrix) and exact finite-length density with a GC window."""
import math
import numpy as np


def asymptotic_capacity(k):
    # states: run length r=1..k of the current base; next base same -> r+1 (if <=k), different (3 choices) -> 1
    M = np.zeros((k, k))
    for r in range(k):
        M[r, 0] += 3
        if r + 1 < k:
            M[r, r + 1] += 1
    return math.log2(max(abs(np.linalg.eigvals(M))))


def finite_count(L, k, gc_lo, gc_hi):
    # DP over (gc count, run length, current base is GC?) ; exact integer counts
    # state: dict[(g, r, isgc)] -> count; bases: 2 GC bases, 2 AT bases
    from collections import defaultdict
    st = defaultdict(int)
    st[(1, 1, True)] = 2   # first base G or C
    st[(0, 1, False)] = 2  # first base A or T
    for _ in range(L - 1):
        nx = defaultdict(int)
        for (g, r, isgc), c in st.items():
            # same base
            if r < k:
                nx[(g + isgc, r + 1, isgc)] += c
            # different base of same class (1 choice) and other class (2 choices)
            nx[(g + isgc, 1, isgc)] += c
            nx[(g + (not isgc), 1, not isgc)] += 2 * c
        st = nx
    lo, hi = math.ceil(gc_lo * L - 1e-9), math.floor(gc_hi * L + 1e-9)
    tot = sum(c for (g, r, isgc), c in st.items() if lo <= g <= hi)
    return tot


if __name__ == "__main__":
    for k in (2, 3, 4):
        print(f"HP<={k}: asymptotic {asymptotic_capacity(k):.4f} bits/nt")
    for L in (100, 150, 200):
        for (lo, hi, k) in [(0.4, 0.6, 3), (0.45, 0.55, 3), (0.4, 0.6, 2), (0.4, 0.6, 4), (0.0, 1.0, 3)]:
            n = finite_count(L, k, lo, hi)
            print(f"L={L} GC[{lo},{hi}] HP<={k}: {math.log2(n)/L:.4f} bits/nt  (fraction of all 4^L: {n/4**L:.4f})")
