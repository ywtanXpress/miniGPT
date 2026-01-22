# tests/test_sample.py

from pathlib import Path

from minigpt.train.sample import find_latest_checkpoint


def test_find_latest_checkpoint(tmp_path: Path):
    (tmp_path / "ckpt_000010.pt").write_bytes(b"x")
    (tmp_path / "ckpt_000200.pt").write_bytes(b"x")
    (tmp_path / "ckpt_000050.pt").write_bytes(b"x")

    p = find_latest_checkpoint(str(tmp_path))
    assert p.endswith("ckpt_000200.pt")
