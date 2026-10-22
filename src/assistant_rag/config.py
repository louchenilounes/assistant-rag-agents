"""Configuration centrale du projet : chemins et documents sources."""

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


@dataclass(frozen=True)
class SourceDocument:
    """Document du corpus et ses métadonnées de référence."""

    key: str
    name: str
    reference: str
    filename: str
    url: str
    expected_articles: int

    @property
    def path(self) -> Path:
        return RAW_DIR / self.filename


DOCUMENTS: dict[str, SourceDocument] = {
    "rgpd": SourceDocument(
        key="rgpd",
        name="RGPD",
        reference="Règlement (UE) 2016/679",
        filename="rgpd.pdf",
        url="https://eur-lex.europa.eu/legal-content/FR/TXT/PDF/?uri=CELEX:32016R0679",
        expected_articles=99,
    ),
    "ai_act": SourceDocument(
        key="ai_act",
        name="AI Act",
        reference="Règlement (UE) 2024/1689",
        filename="ai_act.pdf",
        url="https://eur-lex.europa.eu/legal-content/FR/TXT/PDF/?uri=CELEX:32024R1689",
        expected_articles=113,
    ),
}
