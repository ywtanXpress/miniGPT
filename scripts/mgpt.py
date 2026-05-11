# scripts/mgpt.py

import typer
import json

from minigpt.data.pipeline import build_corpus
from minigpt.tokenizer.train import train_tokenizer
from minigpt.tokenizer.analyze import analyze_tokenizer
from minigpt.tokenizer.encode import encode_corpus
from minigpt.train.pretrain import pretrain
from minigpt.train.sample import sample_text
from minigpt.sft.build import build_sft_tokens
from minigpt.sft.train import sft_train
from minigpt.sft.sample import sft_sample
from minigpt.eval.run import run_dpo_eval, run_pretrain_eval, run_rag_eval, run_sft_eval
from minigpt.dpo.build import build_dpo_pairs
from minigpt.dpo.train import dpo_train
from minigpt.dpo.sample import dpo_sample
from minigpt.rag.build import build_rag_index
from minigpt.rag.query import rag_query
from minigpt.rag.retrieve import retrieve_chunks

app = typer.Typer(help="miniGPT CLI")

data_app = typer.Typer(help="Data pipeline commands")
tokenizer_app = typer.Typer(help="Tokenizer commands")
train_app = typer.Typer(help="Training commands")
sft_app = typer.Typer(help="Supervised fine-tuning (SFT) commands")
eval_app = typer.Typer(help="Evaluation commands")
dpo_app = typer.Typer(help="Preference optimization (DPO) commands")
rag_app = typer.Typer(help="Retrieval-augmented generation (RAG) commands")

app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")
app.add_typer(sft_app, name="sft")
app.add_typer(eval_app, name="eval")
app.add_typer(dpo_app, name="dpo")
app.add_typer(rag_app, name="rag")


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
    temperature: float = typer.Option(0.8, help="Sampling temperature"),
    top_k: int = typer.Option(40, help="Top-k sampling (set 0 to disable)"),
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
    top_k: int = typer.Option(40),
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


@eval_app.command("pretrain")
def eval_pretrain_cmd(config: str = typer.Option(..., help="Path to eval YAML config")):
    out = run_pretrain_eval(config_path=config)
    print(out)


@eval_app.command("sft")
def eval_sft_cmd(config: str = typer.Option(..., help="Path to eval YAML config")):
    out = run_sft_eval(config_path=config)
    print(out)


@eval_app.command("dpo")
def eval_dpo_cmd(config: str = typer.Option(..., help="Path to eval YAML config")):
    out = run_dpo_eval(config_path=config)
    print(out)


@eval_app.command("rag")
def eval_rag_cmd(config: str = typer.Option(..., help="Path to eval YAML config")):
    out = run_rag_eval(config_path=config)
    print(out)


@dpo_app.command("build")
def dpo_build_cmd(config: str = typer.Option(..., help="Path to DPO YAML config")):
    build_dpo_pairs(config_path=config)


@dpo_app.command("train")
def dpo_train_cmd(
    config: str = typer.Option(..., help="Path to DPO YAML config"),
    resume: bool = typer.Option(False, help="Resume from latest checkpoint in out_dir"),
    ckpt: str = typer.Option(None, help="Checkpoint path to resume from"),
):
    dpo_train(config_path=config, resume=bool(resume), ckpt_path=ckpt)


@dpo_app.command("sample")
def dpo_sample_cmd(
    config: str = typer.Option(..., help="Path to DPO YAML config"),
    instruction: str = typer.Option(..., help="Instruction text"),
    inp: str = typer.Option(None, help="Optional input/context"),
    ckpt: str = typer.Option(None, help="Checkpoint path (default: latest in out_dir)"),
    max_new_tokens: int = typer.Option(200),
    temperature: float = typer.Option(0.8),
    top_k: int = typer.Option(40),
):
    text = dpo_sample(
        config_path=config,
        instruction=instruction,
        inp=inp,
        ckpt_path=ckpt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k) if top_k is not None else None,
    )
    print(text)


@rag_app.command("build")
def rag_build_cmd(config: str = typer.Option(..., help="Path to RAG YAML config")):
    build_rag_index(config_path=config)


@rag_app.command("retrieve")
def rag_retrieve_cmd(
    config: str = typer.Option(..., help="Path to RAG YAML config"),
    query: str = typer.Option(..., help="Search query"),
    top_k: int = typer.Option(None, help="Override retrieval top-k"),
):
    results = retrieve_chunks(config_path=config, query=query, top_k=top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


@rag_app.command("query")
def rag_query_cmd(
    config: str = typer.Option(..., help="Path to RAG YAML config"),
    question: str = typer.Option(..., help="User question"),
    retrieval_top_k: int = typer.Option(None, help="Override retrieval top-k"),
    max_new_tokens: int = typer.Option(None, help="Override generation length"),
    temperature: float = typer.Option(None, help="Override generation temperature"),
    top_k: int = typer.Option(None, help="Override generation top-k"),
):
    report = rag_query(
        config_path=config,
        question=question,
        retrieval_top_k=retrieval_top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    print(report["text"])
    print()
    print("Retrieved chunks:")
    print(json.dumps(report["retrieved"], ensure_ascii=False, indent=2))
    print()
    print(f"Saved report: {report['report_path']}")


if __name__ == "__main__":
    app()
