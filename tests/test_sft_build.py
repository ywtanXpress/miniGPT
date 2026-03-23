from minigpt.sft.build import _build_sequence


def test_build_sequence_appends_eos_when_response_fits():
    prompt_ids = [11, 12]
    resp_ids = [21, 22, 23]

    seq, real_len, added_eos = _build_sequence(
        prompt_ids=prompt_ids,
        resp_ids=resp_ids,
        seq_len=8,
        min_resp_tokens=2,
        eos_id=99,
        pad_id=0,
    )

    assert added_eos is True
    assert real_len == 6
    assert seq[:6] == [11, 12, 21, 22, 23, 99]
    assert seq[6:] == [0, 0]


def test_build_sequence_does_not_append_eos_when_response_is_truncated():
    prompt_ids = [11, 12, 13]
    resp_ids = [21, 22, 23, 24]

    seq, real_len, added_eos = _build_sequence(
        prompt_ids=prompt_ids,
        resp_ids=resp_ids,
        seq_len=6,
        min_resp_tokens=2,
        eos_id=99,
        pad_id=0,
    )

    assert added_eos is False
    assert real_len == 6
    assert seq == [11, 12, 13, 21, 22, 23]
