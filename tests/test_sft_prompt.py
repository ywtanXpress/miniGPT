# tests/test_sft_prompt.py

from minigpt.sft.sample import _format_prompt


def test_format_prompt_contains_response_header():
    template = {
        "system": "",
        "user_prefix": "### Instruction:\n",
        "assistant_prefix": "\n\n### Response:\n",
        "end": "\n",
    }
    p = _format_prompt(template, "Say hi", None)
    assert p.startswith("### Instruction:\n")
    assert "\n\n### Response:\n" in p
    assert p.endswith("\n\n### Response:\n")


def test_format_prompt_merges_input():
    template = {
        "system": "",
        "user_prefix": "### Instruction:\n",
        "assistant_prefix": "\n\n### Response:\n",
        "end": "\n",
    }
    p = _format_prompt(template, "Summarize", "This text.")
    assert "Summarize" in p
    assert "This text." in p
