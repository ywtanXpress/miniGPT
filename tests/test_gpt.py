# tests/test_gpt.py

import torch

from minigpt.model.gpt import GPT, GPTConfig


def test_gpt_forward_smoke():
    cfg = GPTConfig(vocab_size=1000, block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    m = GPT(cfg)

    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), dtype=torch.long)
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), dtype=torch.long)

    logits, loss = m(x, y)
    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss).item() is True


def test_gpt_generate_smoke():
    cfg = GPTConfig(vocab_size=1000, block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    m = GPT(cfg)

    prompt = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    out = m.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)

    assert out.shape[0] == 1
    assert out.shape[1] == 18
