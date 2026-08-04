import os
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from translator import translate_file

DEFAULT_MATLAB_PATH = "/sample_matlab/fft_basic.m"

_REFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "reference_set"
)

_MARKER_STYLE = {
    "verified": (
        "background-color: #2ecc71; color: white; padding: 2px 10px;"
        " border-radius: 8px; font-weight: bold;"
    ),
    "unverified": (
        "background-color: #f1c40f; color: #333333; padding: 2px 10px;"
        " border-radius: 8px; font-weight: bold;"
    ),
    "flagged": (
        "background-color: #e74c3c; color: white; padding: 2px 10px;"
        " border-radius: 8px; font-weight: bold;"
    ),
    "pending": (
        "background-color: #95a5a6; color: white; padding: 2px 10px;"
        " border-radius: 8px; font-weight: bold;"
    ),
}


def section_marker(status):
    if status in ("ok", "verified", "none"):
        return "verified"
    if status in ("unresolved", "drafted", "review needed", "skipped"):
        return "unverified"
    if status in ("error", "failed"):
        return "flagged"
    return "unverified"


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class MinimalTranslatorWindow(QMainWindow):
    def __init__(self, matlab_path=DEFAULT_MATLAB_PATH):
        super().__init__()
        self.setWindowTitle("MATPY Translator")
        self.resize(1000, 640)
        self.matlab_path = matlab_path

        self.matlab_pane = QPlainTextEdit()
        self.matlab_pane.setReadOnly(True)
        self.python_pane = QPlainTextEdit()

        splitter = QSplitter()
        splitter.addWidget(self.matlab_pane)
        splitter.addWidget(self.python_pane)
        splitter.setSizes([500, 500])

        self.translate_button = QPushButton("Translate")
        self.translate_button.clicked.connect(self._translate)

        self.save_button = QPushButton("Save correction")
        self.save_button.clicked.connect(self._save_correction)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.translate_button)
        buttons.addWidget(self.save_button)

        self.section_labels = {}
        sections_row = QHBoxLayout()
        sections_row.addWidget(QLabel("Sections:"))
        for stage in ("reader", "rulebook", "assistant", "checker"):
            label = QLabel("%s: pending" % stage.title())
            label.setStyleSheet(_MARKER_STYLE["pending"])
            self.section_labels[stage] = label
            sections_row.addWidget(label)
        sections_row.addStretch(1)

        central = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(buttons)
        layout.addLayout(sections_row)
        layout.addWidget(splitter)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")
        if matlab_path:
            self.load_matlab(matlab_path)

    def load_matlab(self, path):
        self.matlab_path = path
        self.matlab_pane.setPlainText(_read_text(path))
        self.setWindowTitle("MATPY Translator - %s" % path)
        self.statusBar().showMessage("Loaded %s" % path)

    def reference_path(self):
        name = os.path.basename(self.matlab_path).rsplit(".", 1)[0] + ".py"
        return os.path.normpath(os.path.join(_REFERENCE_DIR, name))

    def _set_section_marker(self, stage, status):
        label = self.section_labels[stage]
        label.setText("%s: %s" % (stage.title(), status))
        label.setStyleSheet(_MARKER_STYLE[section_marker(status)])

    def _update_sections(self, sections):
        for stage, info in sections.items():
            self._set_section_marker(stage, info["status"])

    def _translate(self):
        if not self.matlab_path:
            self.statusBar().showMessage("No MATLAB file loaded")
            return
        try:
            result = translate_file(self.matlab_path)
        except Exception as exc:
            self.statusBar().showMessage("Translation failed: %s" % exc)
            self._set_section_marker("checker", "failed")
            return
        self.python_pane.setPlainText(result["python"])
        self._update_sections(result["sections"])
        self.statusBar().showMessage(
            "Translated %s (status=%s)" % (self.matlab_path, result["status"])
        )

    def _save_correction(self):
        if not self.matlab_path:
            self.statusBar().showMessage("No MATLAB file loaded")
            return
        path = self.reference_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.python_pane.toPlainText())
        except Exception as exc:
            self.statusBar().showMessage("Save failed: %s" % exc)
            return
        self.statusBar().showMessage("Saved correction to %s" % path)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MATLAB_PATH
    app = QApplication(sys.argv)
    window = MinimalTranslatorWindow(matlab_path=path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
