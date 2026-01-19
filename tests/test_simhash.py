# tests/test_simhash.py

from minigpt.common.hashing import SimHash


def test_simhash_distance_ranks_similarity():
    a = "The quick brown fox jumps over the lazy dog."
    b = "The quick brown fox jumped over the lazy dog!"
    c = "Quantum chromodynamics describes the strong nuclear force in particle physics."

    da = SimHash.from_text(a)
    db = SimHash.from_text(b)
    dc = SimHash.from_text(c)

    dab = da.hamming_distance(db)
    dac = da.hamming_distance(dc)

    assert dab < dac
