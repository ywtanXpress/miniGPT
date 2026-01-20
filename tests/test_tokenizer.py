# tests/test_tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


def test_tokenizer_train_and_encode_smoke():
    texts = [
        "hello world",
        "hello there",
        "general kenobi",
        "hello world!",
        "Tabs\tand\nnewlines should work.",
    ]

    tok = Tokenizer(BPE(unk_token="<|unk|>"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(vocab_size=200, min_frequency=1, special_tokens=["<|unk|>"])
    tok.train_from_iterator(iter(texts), trainer=trainer)

    enc = tok.encode("hello world")
    assert len(enc.ids) > 0
    assert tok.get_vocab_size() > 0

    decoded = tok.decode(enc.ids)
    assert "hello" in decoded
