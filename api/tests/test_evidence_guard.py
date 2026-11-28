"""Retrieval-level evidence check: abstain before generation when the retrieved
passages share no salient term with the question (A03 abstention correctness)."""

from __future__ import annotations

from app.config import Settings
from app.providers.llm.stub import StubProvider
from app.retrieval.stub import StubRetriever
from app.schemas import Passage
from app.services.answer import _has_evidence_overlap, _salient_terms, answer_question


class FixedRetriever:
    name = "fixed"

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def retrieve(self, question: str, k: int) -> list[Passage]:
        return [Passage(chunk_id=f"c{i}", text=t, source="rules.md", section=f"{i}", page=1,
                        score=0.6) for i, t in enumerate(self.texts)]


class CountingProvider(StubProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question, passages):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().generate(question, passages)


ATTENDANCE = ["A student must attend at least 80 percent of the lectures to sit the final "
              "examination.", "Students are assigned a university email account upon enrollment."]


def _ask(question: str, texts: list[str]):  # type: ignore[no-untyped-def]
    provider = CountingProvider()
    settings = Settings(retriever="chroma", relevance_floor=0.5)
    return answer_question(question, FixedRetriever(texts), provider, settings), provider


def test_unrelated_passages_abstain_without_calling_the_llm() -> None:
    resp, provider = _ask("How is eligibility for the honours roll determined?", ATTENDANCE)
    assert not resp.answered_from_context and resp.citations == []
    assert provider.calls == 0
    assert resp.retrieved, "retrieved refs are still reported for retrieval metrics"


def test_related_passages_reach_the_llm() -> None:
    resp, provider = _ask("What percentage of lectures must I attend?", ATTENDANCE)
    assert provider.calls == 1 and resp.answered_from_context


def test_inflections_count_as_evidence() -> None:
    passages = [Passage(chunk_id="c", text="Students eligible to sit the examination ...",
                        source="s", section="s", score=0.9)]
    assert _has_evidence_overlap("What is the exam eligibility rule?", passages)


def test_generic_words_are_not_evidence() -> None:
    assert _salient_terms("What is the university's policy for students on campus?") == set()
    passages = [Passage(chunk_id="c", text="University policy for all students.", source="s",
                        section="s", score=0.9)]
    assert not _has_evidence_overlap("What is the university's parking policy?", passages)


def test_stub_mode_is_unchanged() -> None:
    provider = CountingProvider()
    resp = answer_question("What is attendance policy?", StubRetriever(), provider, Settings())
    assert provider.calls == 1 and resp.citations
