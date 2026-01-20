# scripts/mgpt.py

import typer

from minigpt.data.pipeline import build_corpus
from minigpt.tokenizer.train import train_tokenizer
from minigpt.tokenizer.analyze import analyze_tokenizer
from minigpt.tokenizer.encode import encode_corpus

app = typer.Typer(help="miniGPT CLI")

data_app = typer.Typer(help="Data pipeline commands")
tokenizer_app = typer.Typer(help="Tokenizer commands")

app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")


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


if __name__ == "__main__":
    app()
