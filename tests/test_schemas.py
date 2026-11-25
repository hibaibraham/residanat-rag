from src.common.schemas import Chunk, Question


def test_roundtrip_jsonl():
    c = Chunk(id="x#0", text="t", source="wikipedia_fr", doc_title="T")
    assert Chunk.model_validate_json(c.model_dump_json()) == c


def test_mcq_question():
    q = Question(id="q1", text="?", lang="fr", type="mcq",
                 options={"A": "a", "B": "b"}, gold_options=["A"])
    assert q.answerable and q.split == "dev"
