"""État partagé entre les agents."""

import operator
from typing import Annotated, TypedDict

from assistant_rag.rag import Source


class AgentState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    queries: list[str]
    sources: list[Source]
    draft: str
    first_draft: str
    first_sources: list[Source]
    verdict: str
    problems: list[str]
    searches: list[str]
    revisions: int
    llm_calls: int
    trace: Annotated[list[dict], operator.add]
