from pathlib import Path

import pytest

from mmrag.evaluate import load_qa, normalize
from mmrag.ingest import Page, chunk_pages, chunk_text, load_documents


def test_chunk_text_windows_and_overlap():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), max_words=60, overlap=15)
    assert len(chunks) == 2
    assert chunks[0].split() == words[:60]
    assert chunks[1].split() == words[45:]  # 15-word overlap with the first window


def test_chunk_text_short_and_empty():
    assert chunk_text("just a few words") == ["just a few words"]
    assert chunk_text("   ") == []


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=5, overlap=5)


def test_chunk_pages_keeps_page_metadata():
    pages = [Page("doc_a", 1, "alpha " * 70, "a.png"), Page("doc_b", 2, "beta", "b.png")]
    chunks = chunk_pages(pages)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert {(c.doc_id, c.page, c.image_path) for c in chunks} == {("doc_a", 1, "a.png"), ("doc_b", 2, "b.png")}


def test_synthetic_pdfs_contain_their_answers(corpus, tmp_path):
    out, qa_path = corpus
    qa = load_qa(qa_path)
    assert len(qa) > 0 and all(q["question"].endswith("?") for q in qa)
    pages = {p.doc_id: p for p in load_documents(sorted((out / "pdf").glob("*.pdf")), tmp_path)}
    assert len(pages) == 6
    for q in qa:
        page = pages[q["doc_id"]]
        assert Path(page.image_path).exists()
        assert "SAMPLE - SYNTHETIC DATA" in page.text
        assert normalize(q["answer"]) in normalize(page.text)
    assert len(list((out / "scans").glob("*.jpg"))) == 6


def test_unsupported_file_type(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(ValueError):
        load_documents([f], tmp_path)


def test_ocr_reads_a_scan(corpus):
    pytest.importorskip("rapidocr_onnxruntime")
    from mmrag.ingest import ocr_image

    out, qa_path = corpus
    q = next(q for q in load_qa(qa_path) if q["kind"] == "invoice")
    text = ocr_image(out / "scans" / f"{q['doc_id']}.jpg")
    invoice_no = q["question"].split("invoice ")[-1].rstrip("?")
    assert "TAX INVOICE" in text.upper()
    assert invoice_no in text


def test_chunk_pages_header_words_prefix_later_chunks():
    text = "TAX INVOICE INV-1 " + " ".join(f"w{i}" for i in range(100))
    chunks = chunk_pages([Page("d", 1, text, "d.png")], max_words=60, overlap=15, header_words=3)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("TAX INVOICE INV-1 w0")
    assert chunks[1].text.startswith("TAX INVOICE INV-1 ... ")
