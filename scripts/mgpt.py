# scripts/mgpt.py

import typer
from minigpt.data.pipeline import build_corpus

app = typer.Typer(help="miniGPT CLI")

data_app = typer.Typer(help="Data pipeline commands")
app.add_typer(data_app, name="data")


@data_app.command("build")
def data_build(config: str = typer.Option(..., help="Path to YAML config")):
    """
    Step 1: Build a cleaned, filtered, deduplicated, balanced text corpus and shard to JSONL.
    """
    build_corpus(config_path=config)


if __name__ == "__main__":
    app()
