"""Retrieval and end-to-end QA evaluation against the synthetic question set."""

from __future__ import annotations

import json
import random
import re
import statistics
import time
from pathlib import Path

from .index import ChunkIndex
from .pipeline import RAGPipeline


def load_qa(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(s: str) -> str:
    """Lowercase, drop currency words, thousands separators, punctuation and spaces."""
    s = s.lower().replace("inr", "").replace("rs.", "").replace("₹", "")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"(\d+\.\d*?)0+\b", r"\1", s)  # 1250.50 -> 1250.5, 1250.00 -> 1250.
    s = re.sub(r"(\d)\.(?!\d)", r"\1", s)  # 1250. -> 1250
    return re.sub(r"[^a-z0-9.]", "", s)


def answer_match(pred: str, gold: str) -> bool:
    """Relaxed exact match: the normalised gold answer appears in the normalised prediction."""
    g, p = normalize(gold), normalize(pred)
    if not g:
        return False
    # guard numbers against prefix/suffix collisions (gold 12.5 must not match 112.55)
    return re.search(rf"(?<![0-9.]){re.escape(g)}(?![0-9])", p) is not None


def retrieval_metrics(index: ChunkIndex, qa: list[dict], mode: str, ks=(1, 3, 5)) -> dict:
    ranks: list[int | None] = []
    t0 = time.perf_counter()
    for q in qa:
        pages = index.search_pages(q["question"], k=10, mode=mode)
        gold = (q["doc_id"], q["page"])
        ranks.append(next((r for r, (d, p, _) in enumerate(pages, start=1) if (d, p) == gold), None))
    secs = time.perf_counter() - t0
    n = len(qa)
    out = {f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 4) for k in ks}
    out["mrr@10"] = round(sum(1 / r for r in ranks if r) / n, 4)
    out["n_questions"] = n
    out["ms_per_query"] = round(1000 * secs / n, 2)
    return out


def qa_eval(pipe: RAGPipeline, qa: list[dict], log_path: Path | None = None) -> dict:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows, latencies = [], []
    for q in qa:
        t0 = time.perf_counter()
        ans = pipe.ask(q["question"])
        dt = time.perf_counter() - t0
        latencies.append(dt)
        row = {"question": q["question"], "gold": q["answer"], "pred": ans.answer,
               "correct": answer_match(ans.answer, q["answer"]),
               "retrieved_page_correct": (ans.doc_id, ans.page) == (q["doc_id"], q["page"]),
               "gold_doc": q["doc_id"], "retrieved_doc": ans.doc_id, "seconds": round(dt, 2)}
        rows.append(row)
        print(f"[{len(rows)}/{len(qa)}] {'OK ' if row['correct'] else 'ERR'} {dt:5.1f}s "
              f"gold={q['answer']!r} pred={ans.answer!r}", flush=True)
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    n = len(rows)
    correct_page = [r for r in rows if r["retrieved_page_correct"]]
    return {
        "n_questions": n,
        "answer_accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "retrieval_page_hit@1": round(len(correct_page) / n, 4),
        "answer_accuracy_given_correct_page": round(
            sum(r["correct"] for r in correct_page) / len(correct_page), 4) if correct_page else None,
        "latency_s_mean": round(statistics.mean(latencies), 2),
        "latency_s_median": round(statistics.median(latencies), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
    }


def subsample(qa: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(qa):
        return qa
    return random.Random(seed).sample(qa, n)
