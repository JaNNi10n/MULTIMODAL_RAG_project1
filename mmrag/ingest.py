"""Turn PDFs and page images into page records (text + page image) and retrieval chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Page:
    doc_id: str
    page: int  # 1-based
    text: str
    image_path: str  # page image handed to the VLM


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    page: int
    text: str
    image_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR  # optional dependency, imported only for images

    return RapidOCR()


def ocr_image(path: Path) -> str:
    """OCR an image with RapidOCR (PP-OCR models on ONNX Runtime, CPU). Returns lines top-to-bottom."""
    result, _ = _ocr_engine()(str(path))
    if not result:
        return ""
    # result rows: [box, text, score]; sort by the top-left y, then x, to keep reading order
    rows = sorted(result, key=lambda r: (round(r[0][0][1] / 10), r[0][0][0]))
    return "\n".join(r[1] for r in rows)


def load_pdf(path: Path, image_dir: Path, dpi: int = 130) -> list[Page]:
    """Extract the text layer of every page and render each page to PNG for the VLM."""
    import fitz  # PyMuPDF

    image_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    with fitz.open(path) as doc:
        for i, pg in enumerate(doc, start=1):
            img = image_dir / f"{path.stem}_p{i}.png"
            pg.get_pixmap(dpi=dpi).save(img)
            pages.append(Page(path.stem, i, pg.get_text(), img.as_posix()))
    return pages


def load_image(path: Path) -> list[Page]:
    return [Page(path.stem, 1, ocr_image(path), path.as_posix())]


def load_documents(paths: list[Path], image_dir: Path) -> list[Page]:
    pages: list[Page] = []
    for p in paths:
        ext = p.suffix.lower()
        if ext == ".pdf":
            pages += load_pdf(p, image_dir)
        elif ext in IMAGE_EXTS:
            pages += load_image(p)
        else:
            raise ValueError(f"Unsupported file type: {p}")
    return pages


def chunk_text(text: str, max_words: int = 60, overlap: int = 15) -> list[str]:
    """Split text into overlapping word windows. Short texts give a single chunk."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    words = text.split()
    if not words:
        return []
    step = max_words - overlap
    return [" ".join(words[i:i + max_words]) for i in range(0, max(len(words) - overlap, 1), step)]


def chunk_pages(pages: list[Page], max_words: int = 60, overlap: int = 15) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pg in pages:
        for t in chunk_text(pg.text, max_words, overlap):
            chunks.append(Chunk(len(chunks), pg.doc_id, pg.page, t, pg.image_path))
    return chunks
