# Changelog

All notable changes to this project are listed here. Dates are commit dates.

## [1.0.0] - 2026-10-13

First working version, built on top of the original prototype commit.

### Removed
- `New folder/your_document.jpg`, a scanned bank statement with personal data. It is still in the
  history of the first commit and has to be purged before the repository is published again (see README).

### Added
- `mmrag` package with a CLI (`python -m mmrag synth | index | ask | eval-retrieval | eval-qa`).
- Ingestion of PDFs (text layer + page render via PyMuPDF) and page images (OCR via RapidOCR).
- Overlapping word-window chunking with an optional page-header prefix (`--header-words`).
- Retrieval: dense (all-MiniLM-L6-v2 + FAISS), BM25, and a reciprocal-rank-fusion hybrid. Index save/load.
- Qwen2.5-VL-3B-Instruct reader with the full chat-template / `process_vision_info` / processor pipeline,
  4-bit NF4 loading for 4 GB GPUs and blank-margin trimming.
- Pipeline that routes the best-matching page image and its text to the reader and returns sources.
- Seeded generator for a synthetic, privacy-safe corpus (45 invoices, statements and bills, 120 questions).
- Evaluation: page-level hit@k / MRR for every retrieval mode, end-to-end answer accuracy, latency and VRAM.
- End-to-end Qwen2.5-VL results on a seeded 40-question subsample (`results/qa_ocr_bm25.*`).
- `--context-chunks` option for `eval-qa` (`0` = the reader sees only the page image), and an image-only run on the
  same 40 questions (`results/qa_ocr_bm25_image_only.*`): also 40/40, peak VRAM 2.83 GB.
- Retrieval ablation with the 16-word page-header prefix on both the PDF and the OCR index.
- Results in `results/`, retrieval chart in `docs/`, pytest suite, ruff config, GitHub Actions CI,
  pinned `requirements.txt` / `requirements-vlm.txt`, README, MIT license.

### Fixed
- `faiss_retrieval.py` (now `examples/`): removed the notebook-only `!pip` line that caused a SyntaxError.
- `vlm_inference.py` (now `examples/`): fixed the IndentationError and the duplicated resize block,
  and built the missing `inputs` (chat message, `process_vision_info`, processor call) that `generate()` needed.
