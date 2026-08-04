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

from reader import load_matlab_file
from translator import translate_file

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


class TranslatorWindow(QMainWindow):
    def __init__(self, matlab_path=None):
        super().__init__()
        self.setWindowTitle("MATPY Translator")
        self.resize(1000, 640)
        self.matlab_path = matlab_path

        self.matlab_pane = QPlainTextEdit()
        self.matlab_pane.setReadOnly(True)
        self.python_pane = QPlainTextEdit()
        self.python_pane.setReadOnly(True)

        splitter = QSplitter()
        splitter.addWidget(self.matlab_pane)
        splitter.addWidget(self.python_pane)
        splitter.setSizes([500, 500])

        self.translate_button = QPushButton("Translate")
        self.translate_button.clicked.connect(self._translate)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.translate_button)

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
        self.matlab_pane.setPlainText(load_matlab_file(path))
        self.statusBar().showMessage("Loaded %s" % path)

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
            "status=%s checker=%s" % (result["status"], result["sections"]["checker"]["status"])
        )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/sample_matlab/fft_basic.m"
    app = QApplication(sys.argv)
    window = TranslatorWindow(matlab_path=path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
