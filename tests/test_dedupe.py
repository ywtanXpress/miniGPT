from minigpt.common.hashing import SimHash
from minigpt.data.dedupe import DedupeState


def test_exact_dedupe():
    st = DedupeState()
    assert st.is_exact_dup("hello") is False
    assert st.is_exact_dup("hello") is True


def test_simhash_distance_ranking():
    a = "This is a document about machine learning and transformers."
    b = "This is a doc about machine learning & transformers!"
    c = "Bananas are yellow and I like to ride bicycles in the rain."

    da = SimHash.from_text(a)
    db = SimHash.from_text(b)
    dc = SimHash.from_text(c)

    dab = da.hamming_distance(db)
    dac = da.hamming_distance(dc)

    assert dab < dac


def test_near_dedupe_works_on_high_overlap_text():
    st = DedupeState()

    a = (
        "Machine learning uses models trained on data. Transformers are a neural network "
        "architecture widely used for language tasks."
    )
    b = (
        "Machine learning uses models trained on data. Transformers are a neural network "
        "architecture widely used for language tasks!"
    )

    assert st.is_near_dup(a, threshold=4) is False
    assert st.is_near_dup(b, threshold=4) is True
