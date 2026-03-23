import pytest

from minigpt.data.mix import MixedStream
from minigpt.data.sources import SourceSpec


def test_mixed_stream_requires_positive_weight():
    stream = MixedStream(
        specs=[
            SourceSpec(
                name="a",
                kind="hf",
                dataset="unused",
                subset=None,
                split="train",
                text_field="text",
                weight=0.0,
            )
        ]
    )

    with pytest.raises(ValueError, match="positive weight"):
        iter(stream).__next__()
