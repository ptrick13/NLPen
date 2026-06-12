import pytest


@pytest.fixture()
def sentences():
    return [
        "Das Produkt ist sehr gut und hilfreich.",
        "Leider hat der Service völlig versagt.",
        "Angela Merkel ist eine bekannte Politikerin.",
        "Berlin ist die Hauptstadt von Deutschland.",
        "Siemens ist ein großes deutsches Unternehmen.",
        "Die Qualität könnte besser sein.",
        "Ich bin mit dem Ergebnis sehr zufrieden.",
        "Der Support war eine einzige Katastrophe.",
    ]
