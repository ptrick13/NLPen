"""FastAPI web application for NLPen PDF analyser."""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from nlpen.nlp.combined import CombinedAnalyzer, CombinationMode
from nlpen.nlp.ner import NERAnalyzer
from nlpen.nlp.sentiment import SentimentAnalyzer
from nlpen.nlp.summarizer import SentenceSummarizer
from nlpen.pdf.extractor import extract_sentences

logger = logging.getLogger(__name__)

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="NLPen")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    summarize: bool = Form(False),
    sentiment: bool = Form(False),
    ner: bool = Form(False),
):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    output_path: str | None = None
    try:
        sentences = extract_sentences(tmp_path)
        analyzer = _build_analyzer(summarize, sentiment, ner)
        output_path = analyzer.run(sentences, tmp_path)
        with open(output_path, "rb") as f:
            pdf_bytes = f.read()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _remove_file(tmp_path)
        if output_path:
            _remove_file(output_path)

    stem = Path(file.filename or "document").stem
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{stem}_analysiert.pdf"'},
    )


def _build_analyzer(summarize: bool, sentiment: bool, ner: bool):
    match (summarize, sentiment, ner):
        case (True, False, False):
            return SentenceSummarizer()
        case (False, True, False):
            return SentimentAnalyzer()
        case (False, False, True):
            return NERAnalyzer()
        case (True, True, False):
            return CombinedAnalyzer(CombinationMode.SENTIMENT_AND_SUMMARIZATION)
        case (True, False, True):
            return CombinedAnalyzer(CombinationMode.NER_AND_SUMMARIZATION)
        case (False, True, True):
            return CombinedAnalyzer(CombinationMode.NER_AND_SENTIMENT)
        case _:
            raise ValueError(
                "Bitte wähle mindestens eine und höchstens zwei Analyseoptionen aus."
            )


def _remove_file(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
