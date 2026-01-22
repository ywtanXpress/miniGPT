# scripts/mgpt.py

import typer

from minigpt.data.pipeline import build_corpus
from minigpt.tokenizer.train import train_tokenizer
from minigpt.tokenizer.analyze import analyze_tokenizer
from minigpt.tokenizer.encode import encode_corpus
from minigpt.train.pretrain import pretrain
from minigpt.train.sample import sample_text

app = typer.Typer(help="miniGPT CLI")

data_app = typer.Typer(help="Data pipeline commands")
tokenizer_app = typer.Typer(help="Tokenizer commands")
train_app = typer.Typer(help="Training commands")

app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")


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


if __name__ == "__main__":
    app()
