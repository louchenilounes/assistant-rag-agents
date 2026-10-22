from pathlib import Path

import pymupdf
import pytest

PAGES = [
    [
        "CONSIDÉRANT CE QUI SUIT :",
        "(1) La protection des personnes physiques est un droit fondamental.",
        "CHAPITRE I",
        "Dispositions générales",
        "Article premier",
        "Objet et objectifs",
        "1. Le présent règlement établit des règles.",
    ],
    [
        "2. Il protège les libertés et droits fondamentaux.",
        "Article 2",
        "Champ d'application matériel",
        "Le présent règlement s'applique au traitement de données.",
        "CHAPITRE II",
        "Principes",
        "Article 3",
        "Modification du règlement (CE) no 300/2008",
        "Le règlement est modifié comme suit :",
        "Article 12",
        "Texte cité d'un autre règlement.",
    ],
    [
        "Article 2",
        "Ce titre cité ne doit pas être pris en compte.",
        "Fait à Bruxelles, le 27 avril 2016.",
        "ANNEXE I",
        "Liste des législations.",
    ],
]


@pytest.fixture
def regulation_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "reglement.pdf"
    pdf = pymupdf.open()
    for number, lines in enumerate(PAGES, start=1):
        page = pdf.new_page()
        y = 40
        page.insert_text((50, y), f"4.5.2016 FR Journal officiel de l'Union européenne L 119/{number}")
        for line in lines:
            y += 20
            page.insert_text((50, y), line)
        page.insert_text((50, 800), f"Page {number} sur {len(PAGES)}")
    pdf.save(path)
    pdf.close()
    return path
