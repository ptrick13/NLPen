"""Entry point for the NLPen PDF analyser application."""

import logging

from nlpen.app import App

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    App().run()
