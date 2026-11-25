"""Prompts that force grounded answers: cite chunk ids, refuse when unsupported."""
from __future__ import annotations

import re

from src.common.schemas import Lang, RetrievedChunk

REFUSAL_TOKEN = "[NO_ANSWER]"
CITATION_RE = re.compile(r"\[([^\[\]\s]+#\d+)\]")

_LANG_NAME = {"fr": "French", "en": "English", "ar": "Arabic (Modern Standard)",
              "darija": "Arabic (Modern Standard)"}

SYSTEM_OPEN = """You are a study assistant for the Tunisian medical residency exam (résidanat).
Rules:
1. Answer ONLY with information found in the CONTEXT passages. Never use outside knowledge.
2. After every sentence that states a fact, cite its source passage id in square brackets, e.g. [wiki:Asthme#2]. Use only ids that appear in the CONTEXT.
3. Copy numbers, doses, thresholds and units exactly as written in the CONTEXT.
4. If the CONTEXT does not contain enough information to answer, reply with exactly {refusal} and nothing else.
5. Be concise: at most 6 sentences.
6. Answer in {lang_name}."""

SYSTEM_MCQ = """You are a study assistant for the Tunisian medical residency exam (résidanat).
You answer multiple-choice questions where ONE OR SEVERAL options may be correct.
Rules:
1. Base your choice on the CONTEXT passages. Never invent facts.
2. Return ONLY a JSON object: {{"answer": ["A", "C"], "justification": "short explanation with citations like [wiki:Asthme#2]"}}
3. If the CONTEXT is not sufficient to decide, return {{"answer": [], "justification": "{refusal}"}}.
4. Write the justification in {lang_name}."""


def format_context(chunks: list[RetrievedChunk], max_chars: int = 1200) -> str:
    blocks = []
    for rc in chunks:
        c = rc.chunk
        head = f"[{c.id}] {c.doc_title}" + (f" > {c.section}" if c.section else "")
        blocks.append(f"{head}\n{c.text[:max_chars]}")
    return "\n\n".join(blocks) if blocks else "(no passages)"


def open_prompt(question: str, chunks: list[RetrievedChunk], lang: Lang) -> tuple[str, str]:
    system = SYSTEM_OPEN.format(refusal=REFUSAL_TOKEN, lang_name=_LANG_NAME[lang])
    user = f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {question}"
    return system, user


def mcq_prompt(question: str, options: dict[str, str], chunks: list[RetrievedChunk],
               lang: Lang) -> tuple[str, str]:
    system = SYSTEM_MCQ.format(refusal=REFUSAL_TOKEN, lang_name=_LANG_NAME[lang])
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    user = f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {question}\nOPTIONS:\n{opts}"
    return system, user


def no_context_prompt(question: str, lang: Lang) -> tuple[str, str]:
    """Baseline B0 (LLM only, no retrieval)."""
    system = ("You are a study assistant for the Tunisian medical residency exam. "
              f"Answer concisely in {_LANG_NAME[lang]}. If you do not know, reply {REFUSAL_TOKEN}.")
    return system, f"QUESTION: {question}"


def extract_citations(text: str, allowed_ids: set[str] | None = None) -> list[str]:
    """Ordered, de-duplicated chunk ids cited in the answer (filtered to allowed ids)."""
    seen: list[str] = []
    for cid in CITATION_RE.findall(text):
        if cid not in seen and (allowed_ids is None or cid in allowed_ids):
            seen.append(cid)
    return seen


def is_refusal(text: str) -> bool:
    return REFUSAL_TOKEN in text or not text.strip()
