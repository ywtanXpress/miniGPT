# tests/test_cleaning.py

from minigpt.common.text_cleaning import (
    FilterConfig,
    clean_text_final,
    clean_text_structural,
    passes_filters,
    repeated_line_ratio,
)


def test_structural_cleaning_preserves_newlines():
    s = "Line A\r\nLine B\r\nLine C"
    out = clean_text_structural(s)
    assert "\n" in out
    assert out.count("\n") == 2


def test_final_cleaning_removes_newlines():
    s = "Line A\r\nLine B\r\nLine C"
    out = clean_text_final(clean_text_structural(s))
    assert "\n" not in out
    assert "Line A" in out and "Line B" in out and "Line C" in out


def test_repeated_line_ratio_works_with_newlines():
    s = "\n".join(["nav"] * 10) + "\ncontent line\nmore content\n"
    out = clean_text_structural(s)
    r = repeated_line_ratio(out)
    assert r > 0.5


def test_passes_filters_uses_structural_text():
    cfg = FilterConfig(min_chars=5, max_chars=1000, max_repeated_line_ratio=0.30)

    good = "Line one\nLine two\nLine three\nLine four\nLine five"
    bad = "\n".join(["repeat"] * 10)

    good_s = clean_text_structural(good)
    bad_s = clean_text_structural(bad)

    assert passes_filters(good_s, cfg) is True
    assert passes_filters(bad_s, cfg) is False
