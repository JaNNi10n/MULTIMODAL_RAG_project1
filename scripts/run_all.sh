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

# End-to-end answers with Qwen2.5-VL-3B (4-bit) on all 120 questions.
python -m mmrag eval-qa --qa data/synthetic/qa.jsonl --index indexes/ocr --n 120 --out results/qa_ocr_bm25_full.json
# Ablation: the VLM gets only the page image (no OCR text in the prompt).
python -m mmrag eval-qa --qa data/synthetic/qa.jsonl --index indexes/ocr --n 120 --context-chunks 0   --out results/qa_ocr_bm25_image_only_full.json
# The same two variants on the seeded 40-question subsample used in the earlier runs.
python -m mmrag eval-qa --qa data/synthetic/qa.jsonl --index indexes/ocr --n 40 --seed 0   --out results/qa_ocr_bm25_idle_gpu.json
python -m mmrag eval-qa --qa data/synthetic/qa.jsonl --index indexes/ocr --n 40 --seed 0 --context-chunks 0   --out results/qa_ocr_bm25_image_only_idle_gpu.json
