# scripts/mgpt.py

import typer

from minigpt.data.pipeline import build_corpus
from minigpt.tokenizer.train import train_tokenizer
from minigpt.tokenizer.analyze import analyze_tokenizer
from minigpt.tokenizer.encode import encode_corpus
from minigpt.train.pretrain import pretrain
from minigpt.train.sample import sample_text
from minigpt.sft.build import build_sft_tokens
from minigpt.sft.train import sft_train
from minigpt.sft.sample import sft_sample

app = typer.Typer(help="miniGPT CLI")

data_app = typer.Typer(help="Data pipeline commands")
tokenizer_app = typer.Typer(help="Tokenizer commands")
train_app = typer.Typer(help="Training commands")
sft_app = typer.Typer(help="Supervised fine-tuning (SFT) commands")

app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")
app.add_typer(sft_app, name="sft")


@data_app.command("build")
def data_build(config: str = typer.Option(..., help="Path to YAML config")):
    build_corpus(config_path=config)


@tokenizer_app.command("train")
def tok_train(config: str = typer.Option(..., help="Path to YAML config")):
    train_tokenizer(config_path=config)


@tokenizer_app.command("analyze")
def tok_analyze(config: str = typer.Option(..., help="Path to YAML config")):
    analyze_tokenizer(config_path=config)


@tokenizer_app.command("encode")
def tok_encode(config: str = typer.Option(..., help="Path to YAML config")):
    encode_corpus(config_path=config)


@train_app.command("pretrain")
def train_pretrain(
    config: str = typer.Option(..., help="Path to YAML config"),
    resume: bool = typer.Option(False, help="Resume from latest checkpoint in out_dir"),
    ckpt: str = typer.Option(None, help="Checkpoint path to resume from (overrides --resume)"),
):
    pretrain(config_path=config, resume=bool(resume), ckpt_path=ckpt)


@train_app.command("sample")
def train_sample(
    config: str = typer.Option(..., help="Path to YAML config"),
    prompt: str = typer.Option(..., help="Prompt text"),
    ckpt: str = typer.Option(None, help="Checkpoint path (defaults to latest in out_dir)"),
    max_new_tokens: int = typer.Option(200, help="Number of tokens to generate"),
    temperature: float = typer.Option(1.0, help="Sampling temperature"),
    top_k: int = typer.Option(50, help="Top-k sampling (set 0 to disable)"),
):
    tk = None if top_k == 0 else int(top_k)
    text = sample_text(
        config_path=config,
        prompt=prompt,
        ckpt_path=ckpt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=tk,
    )
    print(text)


@sft_app.command("build")
def sft_build(config: str = typer.Option(..., help="Path to YAML config")):
    build_sft_tokens(config_path=config)


@sft_app.command("train")
def sft_train_cmd(
    config: str = typer.Option(..., help="Path to YAML config"),
    resume: bool = typer.Option(False, help="Resume from latest checkpoint in out_dir"),
    ckpt: str = typer.Option(None, help="Checkpoint path to resume from"),
):
    sft_train(config_path=config, resume=bool(resume), ckpt_path=ckpt)


@sft_app.command("sample")
def sft_sample_cmd(
    config: str = typer.Option(..., help="Path to YAML config"),
    instruction: str = typer.Option(..., help="Instruction text"),
    inp: str = typer.Option(None, help="Optional input/context"),
    ckpt: str = typer.Option(None, help="Checkpoint path (default: latest in out_dir)"),
    max_new_tokens: int = typer.Option(200),
    temperature: float = typer.Option(0.8),
    top_k: int = typer.Option(50),
):
    text = sft_sample(
        config_path=config,
        instruction=instruction,
        inp=inp,
        ckpt_path=ckpt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k) if top_k is not None else None,
    )
    print(text)


if __name__ == "__main__":
    app()
