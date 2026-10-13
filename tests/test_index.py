import pytest

from mmrag.index import BM25, ChunkIndex, tokenize
from mmrag.ingest import Chunk

from .conftest import hash_encoder

CHUNKS = [
    Chunk(0, "invoice_1", 1, "TAX INVOICE INV-1111 Issued by: Lotus Print Works Total due: INR 500.00", "i1.png"),
    Chunk(1, "invoice_2", 1, "TAX INVOICE INV-2222 Issued by: Saffron Logistics Total due: INR 900.00", "i2.png"),
    Chunk(2, "bill_3", 1, "ELECTRICITY BILL Consumer number: 913-3945 Units consumed: 888 kWh", "b3.png"),
    Chunk(3, "bill_3", 1, "Amount payable: INR 5,700.00 Due date: 2025-09-12", "b3.png"),
]


@pytest.fixture()
def index():
    return ChunkIndex.build(CHUNKS, hash_encoder)


def test_tokenize_splits_identifiers():
    assert tokenize("XXXX-XXXX-1968, INV-2222!") == ["xxxx", "xxxx", "1968", "inv", "2222"]


def test_bm25_prefers_exact_identifier():
    bm = BM25([tokenize(c.text) for c in CHUNKS])
    scores = bm.scores(tokenize("total for invoice INV-2222"))
    assert scores.argmax() == 1
    assert bm.scores(["nonexistentterm"]).sum() == 0


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_finds_relevant_chunk(index, mode):
    hits = index.search("How many units were consumed for consumer number 913-3945?", k=2, mode=mode)
    assert hits[0].chunk.doc_id == "bill_3"


def test_search_pages_dedupes_pages(index):
    pages = index.search_pages("consumer 913-3945 amount payable", k=3)
    assert pages[0][:2] == ("bill_3", 1)
    assert len({(d, p) for d, p, _ in pages}) == len(pages)


def test_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = ChunkIndex.load(tmp_path / "idx", hash_encoder)
    assert [c.to_dict() for c in loaded.chunks] == [c.to_dict() for c in CHUNKS]
    q = "Which company issued invoice INV-1111?"
    for mode in ("dense", "bm25", "hybrid"):
        got = [h.chunk.chunk_id for h in loaded.search(q, 4, mode)]
        assert got == [h.chunk.chunk_id for h in index.search(q, 4, mode)]


def test_dense_without_encoder_errors(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = ChunkIndex.load(tmp_path / "idx")
    assert loaded.search("INV-1111", mode="bm25")[0].chunk.doc_id == "invoice_1"
    with pytest.raises(RuntimeError):
        loaded.search("INV-1111", mode="dense")


def test_invalid_mode_and_empty_build(index):
    with pytest.raises(ValueError):
        index.search("x", mode="magic")
    with pytest.raises(ValueError):
        ChunkIndex.build([], hash_encoder)
