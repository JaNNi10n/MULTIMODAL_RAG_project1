#!/usr/bin/env bash
# Reproduce every number in results/ from scratch. Run from the repo root with the venv active.
# Steps 1-4 need only a CPU; step 5 needs a CUDA GPU with ~3 GB free VRAM.
set -euo pipefail

python -m mmrag synth --out data/synthetic --n-docs 45 --seed 7
python -m mmrag index data/synthetic/pdf --out indexes/pdf
python -m mmrag index data/synthetic/scans --out indexes/ocr
# Ablation: prefix every later chunk of a page with the page's first 16 words (title / document ID)
python -m mmrag index data/synthetic/pdf --out indexes/pdf_hdr16 --header-words 16
python -m mmrag index data/synthetic/scans --out indexes/ocr_hdr16 --header-words 16
python -m mmrag eval-retrieval --qa data/synthetic/qa.jsonl --index indexes/pdf --index indexes/ocr \
  --index indexes/pdf_hdr16 --index indexes/ocr_hdr16 --out results/retrieval_metrics.json
python scripts/plot_retrieval.py results/retrieval_metrics.json docs/retrieval_hit_at_k.png

# End-to-end answers with Qwen2.5-VL-3B (4-bit) on a seeded random subsample of 40 questions.
python -m mmrag eval-qa --qa data/synthetic/qa.jsonl --index indexes/ocr --n 40 --seed 0 \
  --out results/qa_ocr_bm25.json
