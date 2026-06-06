"""Main application window."""

import logging
import os
import threading

import customtkinter
from CTkMessagebox import CTkMessagebox
from tkinter import filedialog

from nlpen.config import WINDOW_HEIGHT, WINDOW_WIDTH
from nlpen.nlp.combined import CombinedAnalyzer, CombinationMode
from nlpen.nlp.ner import NERAnalyzer
from nlpen.nlp.sentiment import SentimentAnalyzer
from nlpen.nlp.summarizer import SentenceSummarizer
from nlpen.pdf.extractor import extract_sentences

logger = logging.getLogger(__name__)

_CHECKBOX_LABELS = [
    "  Markiere die wichtigsten Sätze",
    "  Markiere Sätze mit positiver und negativer\n  Stimmung",
    "  Markiere Personen, Orte, Organisationen\n  und andere Namen",
]

_INFO_TEXT = (
    "Bei der Kombination von den wichtigsten Sätzen und positiven und negativen "
    "Sätzen werden die wichtigsten positiven und negativen Sätze markiert."
)


class App:
    """CustomTkinter desktop application for NLP-based PDF annotation.

    State (selected PDF path, extracted sentences) is encapsulated inside the
    instance instead of module-level globals.
    """

    _MAX_SELECTIONS = 2

    def __init__(self) -> None:
        self._pdf_path: str = ""
        self._sentences: list[str] = []
        self._checkbox_vars: list[customtkinter.IntVar] = []
        self._checkboxes: list[customtkinter.CTkCheckBox] = []
        self._btn_upload: customtkinter.CTkButton
        self._btn_analyze: customtkinter.CTkButton

        customtkinter.set_appearance_mode("System")
        self._root = customtkinter.CTk()
        self._setup_window()
        self._build_ui()

    def run(self) -> None:
        self._root.mainloop()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self._root.title("NLPen")
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        w = min(WINDOW_WIDTH, sw)
        h = min(WINDOW_HEIGHT, sh)
        x = (sw - w) // 2
        y = (sh - h) // 2
        self._root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self) -> None:
        customtkinter.CTkLabel(
            self._root,
            text="Analysiere deine PDF-Datei",
            font=("Helvetica", 32),
        ).pack(pady=(80, 20))

        desc_frame = customtkinter.CTkFrame(self._root, fg_color="transparent")
        desc_frame.pack(pady=10)
        customtkinter.CTkLabel(
            desc_frame,
            text=(
                "Lade hier deine PDF-Datei hoch. Achte darauf, dass "
                "sie \nmöglichst nur den zu analysierenden Text enthält. "
                "Wähle im \nAnschluss bis zu zwei Optionen aus, die du "
                "analysieren möchtest."
            ),
            justify="center",
        ).pack(pady=20)

        self._btn_upload = customtkinter.CTkButton(
            self._root, text="PDF auswählen", command=self._select_pdf
        )
        self._btn_upload.pack(pady=(20, 30))

        checkbox_frame = customtkinter.CTkFrame(self._root, fg_color="transparent")
        checkbox_frame.pack(pady=10)
        for i, label in enumerate(_CHECKBOX_LABELS):
            var = customtkinter.IntVar()
            self._checkbox_vars.append(var)
            cb = customtkinter.CTkCheckBox(
                checkbox_frame,
                text=label,
                variable=var,
                state="disabled",
                command=lambda idx=i: self._on_checkbox_toggle(idx),
            )
            self._checkboxes.append(cb)
            cb.pack(pady=10, anchor="w")

        self._btn_analyze = customtkinter.CTkButton(
            self._root,
            text="PDF analysieren & downloaden",
            command=self._analyze,
            state="disabled",
        )
        self._btn_analyze.pack(pady=(30, 50))

        customtkinter.CTkButton(
            self._root,
            text="Info",
            width=32,
            corner_radius=16,
            command=self._show_info,
        ).pack(pady=10, padx=10, anchor="e")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _select_pdf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        self._pdf_path = path
        self._sentences = extract_sentences(path)
        file_name = os.path.basename(path)
        for cb in self._checkboxes:
            cb.configure(state="normal")
        self._btn_upload.configure(text=f"{file_name} ausgewählt (ändern)")
        self._btn_analyze.configure(
            text="PDF analysieren & downloaden", state="normal"
        )
        logger.info("Loaded '%s' — %d sentences", file_name, len(self._sentences))

    def _on_checkbox_toggle(self, index: int) -> None:
        n_selected = sum(v.get() for v in self._checkbox_vars)
        # Deselect if the limit is exceeded
        if self._checkbox_vars[index].get() == 1 and n_selected > self._MAX_SELECTIONS:
            self._checkbox_vars[index].set(0)
        self._btn_analyze.configure(
            text="PDF analysieren & downloaden", state="normal"
        )

    def _analyze(self) -> None:
        self._btn_analyze.configure(state="disabled")
        self._btn_upload.configure(state="disabled")
        self._spinner_running = True
        self._spinner_dots = 0
        self._animate_spinner()
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _animate_spinner(self) -> None:
        if not self._spinner_running:
            return
        dots = "." * (self._spinner_dots % 4)
        self._btn_analyze.configure(text=f"Analysiere{dots}")
        self._spinner_dots += 1
        self._root.after(500, self._animate_spinner)

    def _run_analysis(self) -> None:
        try:
            analyzer = self._build_analyzer()
            analyzer.run(self._sentences, self._pdf_path)
            self._root.after(0, self._on_analysis_done)
        except Exception as exc:
            logger.exception("Analysis failed")
            self._root.after(0, lambda: self._on_analysis_failed(str(exc)))

    def _on_analysis_done(self) -> None:
        self._spinner_running = False
        self._btn_analyze.configure(text="Erneut herunterladen", state="normal")
        self._btn_upload.configure(state="normal")
        self._show_success()

    def _on_analysis_failed(self, message: str) -> None:
        self._spinner_running = False
        self._btn_analyze.configure(text="PDF analysieren & downloaden", state="normal")
        self._btn_upload.configure(state="normal")
        self._show_error(message)

    # ------------------------------------------------------------------
    # Analyzer factory
    # ------------------------------------------------------------------

    def _build_analyzer(self):
        sel = tuple(v.get() for v in self._checkbox_vars)
        match sel:
            case (1, 0, 0):
                return SentenceSummarizer()
            case (0, 1, 0):
                return SentimentAnalyzer()
            case (0, 0, 1):
                return NERAnalyzer()
            case (1, 1, 0):
                return CombinedAnalyzer(CombinationMode.SENTIMENT_AND_SUMMARIZATION)
            case (1, 0, 1):
                return CombinedAnalyzer(CombinationMode.NER_AND_SUMMARIZATION)
            case (0, 1, 1):
                return CombinedAnalyzer(CombinationMode.NER_AND_SENTIMENT)
            case _:
                raise ValueError("Bitte wähle mindestens eine Option aus.")

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _show_success(self) -> None:
        folder = os.path.basename(os.path.dirname(self._pdf_path))
        CTkMessagebox(
            title="Download erfolgreich",
            message=f"Die analysierte PDF wurde erfolgreich im Ordner {folder} gespeichert.",
            icon="check",
            icon_size=(40, 40),
        )

    def _show_error(self, message: str) -> None:
        CTkMessagebox(
            title="Fehler",
            message=f"{message}\n\nBitte probiere es erneut.",
            icon="cancel",
        )

    def _show_info(self) -> None:
        CTkMessagebox(title="Info", message=_INFO_TEXT, width=400)
