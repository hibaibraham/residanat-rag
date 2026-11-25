from src.common.schemas import Chunk, RetrievedChunk
from src.generation.prompts import (REFUSAL_TOKEN, extract_citations, is_refusal,
                                    mcq_prompt, open_prompt)

RC = [RetrievedChunk(chunk=Chunk(id="wiki:Asthme#2", text="Le salbutamol...",
                                 source="wikipedia_fr", doc_title="Asthme"),
                     score=1.0, rank=1)]


def test_open_prompt_contains_ids_and_lang():
    system, user = open_prompt("Traitement ?", RC, "ar")
    assert "[wiki:Asthme#2]" in user and "Arabic" in system and REFUSAL_TOKEN in system


def test_mcq_prompt_has_json_and_sorted_options():
    system, user = mcq_prompt("Q", {"B": "b", "A": "a"}, RC, "fr")
    assert '{"answer": ["A", "C"]' in system
    assert user.index("A. a") < user.index("B. b")


def test_extract_citations_filters_and_dedups():
    t = "X [wiki:Asthme#2]. Y [wiki:Asthme#2] [fake#9]."
    assert extract_citations(t, {"wiki:Asthme#2"}) == ["wiki:Asthme#2"]
    assert extract_citations(t) == ["wiki:Asthme#2", "fake#9"]


def test_refusal():
    assert is_refusal(REFUSAL_TOKEN) and is_refusal("  ") and not is_refusal("ok")
