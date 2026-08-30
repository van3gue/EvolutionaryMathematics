# EVOLVE-BLOCK-START
"""Baseline split construction for Erdos' minimal overlap problem (n=100)."""


def construct_split(n):
    """
    Construct a set A of size n from {0, ..., 2n-1}. The complement
    B = {0,...,2n-1} \\ A is implied.

    Goal: choose A so that, over every nonzero shift k, the overlap
    between A and (B shifted by k) is as small as possible in the
    worst case.

    Returns:
        a: list of n distinct integers in [0, 2n)
    """
    # Trivial baseline: contiguous first half. Very bad on purpose -
    # shifting B down by n aligns it exactly onto A, giving worst-case
    # overlap of n. Evolution should replace this with something smarter
    # (e.g. interleaving, randomized local search, known constructions).
    a = list(range(n))
    return a


# EVOLVE-BLOCK-END


def run_minimal_overlap():
    """Fixed entry point: builds the split and reports its worst-case overlap."""
    n = 100
    N = 2 * n
    a = construct_split(n)
    a_set = set(a)
    b_set = set(range(N)) - a_set

    worst = 0
    for k in range(1, N):
        overlap_up = sum(1 for b in b_set if (b + k) in a_set)
        if overlap_up > worst:
            worst = overlap_up
        overlap_down = sum(1 for b in b_set if (b - k) in a_set)
        if overlap_down > worst:
            worst = overlap_down

    return a, worst, n
