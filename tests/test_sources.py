from minigpt.data.sources import SourceSpec, stream_hf_text


def test_stream_hf_text_requires_hf_kind():
    spec = SourceSpec(
        name="bad",
        kind="local",
        dataset="unused",
        subset=None,
        split="train",
        text_field="text",
        weight=1.0,
    )

    try:
        next(stream_hf_text(spec))
    except ValueError as exc:
        assert "only 'hf' is implemented" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported source kind")


def test_stream_hf_text_skips_non_string_records(monkeypatch):
    rows = iter(
        [
            {"text": None},
            {"text": ["not", "text"]},
            {"text": "   "},
            {"text": " hello world "},
        ]
    )

    def fake_load_dataset(dataset, subset, split, streaming):
        assert dataset == "demo"
        assert subset is None
        assert split == "train"
        assert streaming is True
        return rows

    monkeypatch.setattr("minigpt.data.sources.load_dataset", fake_load_dataset)

    spec = SourceSpec(
        name="demo",
        kind="hf",
        dataset="demo",
        subset=None,
        split="train",
        text_field="text",
        weight=1.0,
    )

    out = list(stream_hf_text(spec))

    assert out == [{"text": "hello world", "source": "demo", "meta": {}}]
