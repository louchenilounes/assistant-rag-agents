from assistant_rag.citations import REFUSAL, is_refusal, parse_citations, strip_thinking


def test_parse_citations_reads_single_and_grouped_numbers():
    text = "Le droit à l'effacement existe [1]. Il a des exceptions [2, 3] et des limites [3][4]."

    assert parse_citations(text, source_count=5) == [1, 2, 3, 4]


def test_parse_citations_ignores_numbers_out_of_range():
    assert parse_citations("Réponse [0] [2] [9].", source_count=5) == [2]


def test_parse_citations_ignores_non_citation_brackets():
    assert parse_citations("Voir l'article 5 [paragraphe 1] du règlement.", source_count=5) == []


def test_strip_thinking_removes_reasoning_block():
    assert strip_thinking("<think>raisonnement</think>\n\nRéponse finale [1].") == "Réponse finale [1]."


def test_strip_thinking_handles_missing_opening_tag():
    assert strip_thinking("raisonnement\n</think>\nRéponse [2].") == "Réponse [2]."


def test_is_refusal_detects_expected_sentence():
    assert is_refusal(f"« {REFUSAL} »")
    assert is_refusal(REFUSAL.upper())
    assert not is_refusal("Le RGPD s'applique aux données personnelles [1].")
