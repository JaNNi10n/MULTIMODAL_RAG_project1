"""Plot page-level hit@k from results/retrieval_metrics.json into docs/retrieval_hit_at_k.png."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/retrieval_metrics.json")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/retrieval_hit_at_k.png")
results = json.loads(src.read_text(encoding="utf-8"))["results"]

labels = {"indexes/pdf": "PDF text layer", "indexes/ocr": "OCR of scans",
          "indexes/pdf_hdr16": "PDF text + 16-word header", "indexes/ocr_hdr16": "OCR + 16-word header"}
modes, ks = ["dense", "bm25", "hybrid"], ["hit@1", "hit@3", "hit@5"]
colors = {"dense": "#4C78A8", "bm25": "#F58518", "hybrid": "#54A24B"}

cols = min(len(results), 2)
rows = -(-len(results) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), sharey=True, squeeze=False)
axes = list(axes.flat)
for ax, (name, res) in zip(axes, results.items(), strict=False):
    width = 0.26
    for j, mode in enumerate(modes):
        vals = [res[mode][k] for k in ks]
        bars = ax.bar([i + (j - 1) * width for i in range(len(ks))], vals, width, label=mode, color=colors[mode])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=2)
    ax.set_xticks(range(len(ks)), ks)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{labels.get(name, name)} ({res['n_chunks']} chunks)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[::cols]:
    ax.set_ylabel("fraction of questions")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.suptitle(f"Page retrieval on {res['bm25']['n_questions']} synthetic questions", fontsize=11)
for ax in axes[len(results):]:
    ax.set_visible(False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
dst.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(dst, dpi=150)
print(f"saved {dst}")
