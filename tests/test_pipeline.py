from PIL import Image, ImageDraw

from mmrag.evaluate import answer_match, load_qa, normalize, retrieval_metrics
from mmrag.index import ChunkIndex
from mmrag.ingest import chunk_pages, load_documents
from mmrag.pipeline import RAGPipeline
from mmrag.vlm import build_prompt, trim_margins

from .conftest import hash_encoder


class FakeReader:
    """Stands in for Qwen2.5-VL: records its inputs and returns a fixed answer."""

    def __init__(self):
        self.calls = []

    def answer(self, question, image_path, context):
        self.calls.append((question, image_path, context))
        return "INR 1,234.50"


def _corpus_index(corpus, tmp_path):
    out, qa_path = corpus
    pages = load_documents(sorted((out / "pdf").glob("*.pdf")), tmp_path)
    return ChunkIndex.build(chunk_pages(pages), hash_encoder), load_qa(qa_path)


def test_pipeline_routes_top_page_to_reader(corpus, tmp_path):
    index, qa = _corpus_index(corpus, tmp_path)
    reader = FakeReader()
    pipe = RAGPipeline(index, reader, k=3)
    q = qa[0]
    ans = pipe.ask(q["question"])
    assert (ans.doc_id, ans.page) == (q["doc_id"], q["page"])
    assert ans.answer == "INR 1,234.50"
    question, image_path, context = reader.calls[0]
    assert question == q["question"] and image_path == ans.image_path
    page_texts = [c.text for c in index.chunks if c.doc_id == q["doc_id"]]
    assert context == page_texts  # context comes only from the chosen page
    assert ans.sources and ans.sources[0]["doc_id"] == q["doc_id"]


def test_bm25_retrieval_on_synthetic_corpus(corpus, tmp_path):
    index, qa = _corpus_index(corpus, tmp_path)
    m = retrieval_metrics(index, qa, mode="bm25")
    assert m["n_questions"] == len(qa)
    assert m["hit@5"] >= m["hit@3"] >= m["hit@1"] >= 0.9


def test_answer_normalisation():
    assert normalize("INR 26,318.70") == "26318.7"
    assert normalize("₹ 1,000.00") == "1000"
    assert answer_match("The total is INR 26,318.72.", "26,318.72")
    assert answer_match("Lotus Print Works", "Lotus Print Works")
    assert answer_match("888 kWh", "888")
    assert not answer_match("1888 kWh", "888")
    assert not answer_match("112.55", "12.5")
    assert not answer_match("anything", "")


def test_prompt_contains_context_and_question():
    p = build_prompt("What is X?", ["first chunk", "second chunk"])
    assert "[1] first chunk" in p and "[2] second chunk" in p and "Question: What is X?" in p
    assert "(none)" in build_prompt("Q?", [])


def test_trim_margins_crops_blank_paper():
    img = Image.new("RGB", (1000, 1400), "white")
    ImageDraw.Draw(img).rectangle((100, 100, 600, 300), fill="black")
    cropped = trim_margins(img, pad=10)
    assert cropped.size == (521, 221)
    assert trim_margins(Image.new("RGB", (50, 50), "white")).size == (50, 50)
