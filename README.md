# NLPen — NLP-Based PDF Annotation Tool

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

A desktop application that runs multiple NLP pipelines on German PDF documents and saves a colour-coded, annotated copy — built as my Bachelor's thesis project and refactored for publication.

---

## What it does

Upload any German-language PDF, select one or two analysis modes, and the app produces an annotated copy with highlights and a colour legend baked into the document.

| Mode | Highlights | Colour |
|------|-----------|--------|
| **Key Sentences** | Most informative sentences (TF-IDF + TextRank) | Yellow |
| **Sentiment** | Positive sentences | Green |
| | Negative sentences | Red |
| **Named Entities** | Persons | Mint green |
| | Organisations | Peach |
| | Locations | Lavender |
| | Miscellaneous names | Light blue |

Any two modes can be combined. When *Key Sentences* and *Sentiment* are combined, only the most important positive/negative sentences are highlighted — filtering noise from both pipelines simultaneously.

---

## Technical approach

### Extractive Summarisation
1. Lemmatise sentences and remove stopwords with **SpaCy** (`de_core_news_md`).
2. Build a TF-IDF matrix with **scikit-learn**.
3. Compute pairwise cosine similarity → weighted graph via **NetworkX**.
4. Run **PageRank** to score each sentence, return the top 20–30 %.

### Sentiment Analysis
- **XLM-RoBERTa** (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) via HuggingFace **Transformers**.
- The multilingual model handles German without language-specific fine-tuning.

### Named Entity Recognition
- **Flair** sequence tagger (`flair/ner-german`) with BIO tagging.
- Classifies four entity types: PER · LOC · ORG · MISC.

---

## Architecture

    nlpen/
    ├── config.py            # Model names, colours, thresholds — one place to change
    ├── app.py               # CustomTkinter GUI (class-based, no module globals)
    ├── pdf/
    │   ├── extractor.py     # PDF parsing & sentence tokenisation (TextBlobDE)
    │   └── highlighter.py   # Highlight & legend drawing with PyMuPDF
    └── nlp/
        ├── base.py          # Abstract BaseAnalyzer: analyze → annotate → save
        ├── summarizer.py    # SentenceSummarizer  (TF-IDF + TextRank)
        ├── sentiment.py     # SentimentAnalyzer   (XLM-RoBERTa)
        ├── ner.py           # NERAnalyzer         (Flair)
        └── combined.py      # CombinedAnalyzer    (coordinates two modes)

---

## Installation

**Python 3.11 required.**

```bash
# 1. Clone and install dependencies
git clone https://github.com/ptrick13/NLPen.git
cd NLPen
pip install -r requirements.txt

# 2. Download the SpaCy German model
python -m spacy download de_core_news_md
```

> **Note:** Flair (`flair/ner-german`, ~400 MB) and HuggingFace
> (`cardiffnlp/twitter-xlm-roberta-base-sentiment`, ~500 MB) models are
> downloaded automatically on first use.

---

## Usage

```bash
python main.py
```

1. Click **PDF auswählen** and pick a German-language PDF.
2. Tick one or two analysis options.
3. Click **PDF analysieren & downloaden**.
4. The annotated PDF is saved in the same folder as the source file (e.g. `report_Sentiment.pdf`).

---

## Bachelor's thesis context

This project was developed as part of my Bachelor's thesis on *combined NLP analysis of German documents*. The thesis explored how extractive summarisation, sentiment classification, and named entity recognition can be combined into a single document-annotation workflow accessible to non-technical users.

---

## Tech stack

| Library | Role | Version |
|---------|------|---------|
| [CustomTkinter](https://customtkinter.tomschimansky.com) | Desktop GUI | 5.2.0 |
| [PyMuPDF](https://pymupdf.readthedocs.io) | PDF reading & annotation | 1.22.5 |
| [SpaCy](https://spacy.io) | German lemmatisation & stopwords | 3.6.0 |
| [scikit-learn](https://scikit-learn.org) | TF-IDF vectorisation | 1.3.0 |
| [NetworkX](https://networkx.org) | TextRank / PageRank graph | 3.1 |
| [Transformers](https://huggingface.co/docs/transformers) | XLM-RoBERTa sentiment | 4.31.0 |
| [Flair](https://flairnlp.github.io) | German NER sequence tagger | 0.12.2 |
| [textblob-de](https://textblob-de.readthedocs.io) | German sentence tokenisation | 0.4.3 |

---

## License

MIT © Patrick Vorreiter