"""Entry point: starts the NLPen web server."""

import logging

import nltk
import uvicorn

nltk.download("punkt_tab", quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
