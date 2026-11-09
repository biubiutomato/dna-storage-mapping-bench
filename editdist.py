"""Myers/Hyyro bit-parallel Levenshtein distance (global), using Python big ints."""


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    m = len(a)
    if m == 0:
        return len(b)
    if len(b) == 0:
        return m
    peq = {}
    for i, c in enumerate(a):
        peq[c] = peq.get(c, 0) | (1 << i)
    mask = (1 << m) - 1
    pv, mv, score = mask, 0, m
    hb = 1 << (m - 1)
    for c in b:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & hb:
            score += 1
        elif mh & hb:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def edit_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))
