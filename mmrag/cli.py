"""Command line interface: python -m mmrag <command> --help"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .index import DEFAULT_EMBEDDER, MODES, ChunkIndex, load_encoder
from .ingest import IMAGE_EXTS, chunk_pages, load_documents


def _expand(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS | {".pdf"})
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    if not files:
        raise SystemExit("no PDF or image files found in the given inputs")
    return files


def _env() -> dict:
    import torch

    return {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def cmd_synth(a):
    from .synth import generate

    qa = generate(Path(a.out), a.n_docs, a.seed)
    print(f"wrote {sum(1 for _ in qa.open())} questions to {qa}")


def cmd_index(a):
    files = _expand(a.inputs)
    out = Path(a.out)
    pages = load_documents(files, out / "page_images")
    chunks = chunk_pages(pages, a.max_words, a.overlap, a.header_words)
    index = ChunkIndex.build(chunks, load_encoder(a.embedder))
    index.save(out)
    print(f"indexed {len(files)} files, {len(pages)} pages, {len(chunks)} chunks -> {out}")


def cmd_ask(a):
    index = ChunkIndex.load(Path(a.index), load_encoder(a.embedder))
    if a.no_vlm:
        for d, p, s in index.search_pages(a.question, k=a.k, mode=a.mode):
            print(f"{s:.4f}  {d} page {p}")
        return
    from .pipeline import RAGPipeline
    from .vlm import QwenVLReader

    pipe = RAGPipeline(index, QwenVLReader(a.vlm, load_in_4bit=not a.no_4bit), k=a.k, mode=a.mode)
    ans = pipe.ask(a.question)
    print(f"Q: {ans.question}\nA: {ans.answer}\nSource: {ans.doc_id} page {ans.page} ({ans.image_path})")


def cmd_eval_retrieval(a):
    from .evaluate import load_qa, retrieval_metrics

    qa = load_qa(Path(a.qa))
    encoder = load_encoder(a.embedder)
    report = {"embedder": a.embedder, "qa_file": a.qa, "env": _env(), "results": {}}
    for idx in a.index:
        index = ChunkIndex.load(Path(idx), encoder)
        report["results"][idx] = {m: retrieval_metrics(index, qa, m) for m in MODES}
        report["results"][idx]["n_chunks"] = len(index.chunks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


def cmd_eval_qa(a):
    from .evaluate import load_qa, qa_eval, subsample
    from .pipeline import RAGPipeline
    from .vlm import QwenVLReader

    qa = subsample(load_qa(Path(a.qa)), a.n, a.seed)
    index = ChunkIndex.load(Path(a.index), load_encoder(a.embedder))
    pipe = RAGPipeline(index, QwenVLReader(a.vlm, load_in_4bit=not a.no_4bit, max_pixels=a.max_pixels),
                       k=a.k, mode=a.mode)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".predictions.jsonl")
    log.unlink(missing_ok=True)
    metrics = qa_eval(pipe, qa, log)
    report = {"config": {k: v for k, v in vars(a).items() if k != "func"}, "env": _env(), "metrics": metrics}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mmrag", description="Multimodal document RAG (FAISS + Qwen2.5-VL)")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("synth", help="generate the synthetic document corpus + QA pairs")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n-docs", type=int, default=45)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("index", help="ingest PDFs/images and build a FAISS + BM25 chunk index")
    p.add_argument("inputs", nargs="+", help="PDF/image files or directories")
    p.add_argument("--out", required=True)
    p.add_argument("--max-words", type=int, default=60)
    p.add_argument("--overlap", type=int, default=15)
    p.add_argument("--header-words", type=int, default=0,
                   help="prefix every non-first chunk of a page with the page's first N words")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="answer a question over an index")
    p.add_argument("question")
    p.add_argument("--index", required=True)
    p.add_argument("--no-vlm", action="store_true", help="only print the retrieved pages")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval-retrieval", help="page-level hit@k / MRR for dense, BM25 and hybrid")
    p.add_argument("--qa", required=True)
    p.add_argument("--index", required=True, action="append")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("eval-qa", help="end-to-end answer accuracy with the VLM")
    p.add_argument("--qa", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--n", type=int, default=30, help="random subsample size (VLM is slow on 4 GB GPUs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_eval_qa)

    for name, sp in sub.choices.items():
        sp.add_argument("--embedder", default=DEFAULT_EMBEDDER)
        if name in ("ask", "eval-qa"):
            sp.add_argument("--k", type=int, default=5)
            sp.add_argument("--mode", choices=MODES, default="bm25")
            sp.add_argument("--vlm", default="Qwen/Qwen2.5-VL-3B-Instruct")
            sp.add_argument("--no-4bit", action="store_true", help="load the VLM in bf16 (needs ~8 GB VRAM)")

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
