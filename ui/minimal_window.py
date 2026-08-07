import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from checker import accuracy, build_translation_report
from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB
from reference_store import save_reference_entry
from translator import translate_source
from ui.highlight import ProblemLineHighlighter
from ui.summary import accuracy_style, accuracy_text, report_text, status_line

_DIRECTION_ITEMS = (
    ("MATLAB -> Python", MATLAB_TO_PYTHON),
    ("Python -> MATLAB", PYTHON_TO_MATLAB),
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
    if status in ("unresolved", "drafted", "review needed", "skipped",
                  "inconclusive_no_matlab"):
        return "unverified"
    if status in ("error", "failed", "errored"):
        return "flagged"
    return "unverified"


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class MinimalTranslatorWindow(QMainWindow):
    def __init__(self, matlab_path=None):
        super().__init__()
        self.setWindowTitle("MATPY Translator")
        self.resize(1000, 640)
        self.matlab_path = matlab_path

        self.matlab_pane = QPlainTextEdit()
        self.matlab_pane.setPlaceholderText(
            "Type or paste source here, or load a file with Open file"
        )
        self.source_label = QLabel("Source: MATLAB")
        self.output_label = QLabel("Output: Python")
        self.python_pane = QPlainTextEdit()
        self.python_highlighter = ProblemLineHighlighter(self.python_pane.document())

        self.splitter = QSplitter()
        self.splitter.addWidget(self.matlab_pane)
        self.splitter.addWidget(self.python_pane)
        self.splitter.setSizes([500, 500])

        self.translate_button = QPushButton("Translate")
        self.translate_button.clicked.connect(self._translate)

        self.open_button = QPushButton("Open file")
        self.open_button.clicked.connect(self._open_file)

        self.save_button = QPushButton("Save correction")
        self.save_button.clicked.connect(self._save_correction)

        self.direction_combo = QComboBox()
        for label, direction in _DIRECTION_ITEMS:
            self.direction_combo.addItem(label, direction)
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)

        buttons = QHBoxLayout()
        buttons.addWidget(self.direction_combo)
        buttons.addWidget(self.translate_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        buttons.addWidget(self.save_button)

        pane_labels = QHBoxLayout()
        pane_labels.addWidget(self.source_label)
        pane_labels.addStretch(1)
        pane_labels.addWidget(self.output_label)

        self.status_line = QLabel("Not translated yet")
        self.details_button = QToolButton()
        self.details_button.setText("Details")
        self.details_button.setCheckable(True)
        self.details_button.setArrowType(Qt.RightArrow)
        self.details_button.toggled.connect(self._toggle_details)

        self.section_labels = {}
        self.details_widget = QWidget()
        details_row = QHBoxLayout()
        details_row.addWidget(QLabel("Stage status:"))
        for stage in ("reader", "rulebook", "assistant", "checker"):
            label = QLabel("%s: pending" % stage.title())
            label.setStyleSheet(_MARKER_STYLE["pending"])
            self.section_labels[stage] = label
            details_row.addWidget(label)
        details_row.addStretch(1)
        self.details_widget.setLayout(details_row)
        self.details_widget.setVisible(False)

        self.report_button = QToolButton()
        self.report_button.setText("Report")
        self.report_button.setCheckable(True)
        self.report_button.setArrowType(Qt.RightArrow)
        self.report_button.toggled.connect(self._toggle_report)

        self.report_pane = QPlainTextEdit()
        self.report_pane.setReadOnly(True)
        self.report_pane.setMaximumHeight(180)
        self.report_pane.setPlaceholderText(
            "Run a translation to populate the report."
        )
        self.report_widget = QWidget()
        report_layout = QVBoxLayout()
        report_layout.setContentsMargins(0, 0, 0, 0)
        report_title = QLabel("Translation Report")
        report_title.setStyleSheet("font-weight: bold;")
        report_layout.addWidget(report_title)
        report_layout.addWidget(self.report_pane)
        self.report_widget.setLayout(report_layout)
        self.report_widget.setVisible(False)

        summary_row = QHBoxLayout()
        summary_row.addWidget(self.status_line)
        summary_row.addStretch(1)
        self.accuracy_label = QLabel(accuracy_text(None))
        self.accuracy_label.setStyleSheet(accuracy_style(None))
        summary_row.addWidget(self.accuracy_label)
        summary_row.addWidget(self.details_button)
        summary_row.addWidget(self.report_button)

        central = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(buttons)
        layout.addLayout(summary_row)
        layout.addWidget(self.details_widget)
        layout.addWidget(self.report_widget)
        layout.addLayout(pane_labels)
        layout.addWidget(self.splitter)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")
        if matlab_path:
            self.load_matlab(matlab_path)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open file",
            "",
            "Source files (*.m *.py);;All files (*)",
        )
        if not path:
            return
        path = os.fspath(path)
        if path.endswith(".py"):
            index = self.direction_combo.findData(PYTHON_TO_MATLAB)
        else:
            index = self.direction_combo.findData(MATLAB_TO_PYTHON)
        self.direction_combo.setCurrentIndex(index)
        self.load_matlab(path)

    def load_matlab(self, path):
        self.matlab_path = path
        self.matlab_pane.setPlainText(_read_text(path))
        self.setWindowTitle("MATPY Translator - %s" % path)
        self.statusBar().showMessage("Loaded %s" % path)

    def _save_correction(self):
        matlab_source = self.matlab_pane.toPlainText()
        python_source = self.python_pane.toPlainText()
        if not matlab_source.strip():
            self.statusBar().showMessage("No source to save")
            return
        if not python_source.strip():
            self.statusBar().showMessage("No translated output to save")
            return
        if self.matlab_path:
            base_name = os.path.basename(self.matlab_path).rsplit(".", 1)[0]
        else:
            base_name, accepted = QInputDialog.getText(
                self,
                "Save correction",
                "Reference entry name (no extension):",
            )
            base_name = base_name.strip()
            if not accepted or not base_name:
                self.statusBar().showMessage("Save cancelled")
                return
        try:
            matlab_path, python_path = save_reference_entry(
                matlab_source, python_source, base_name
            )
        except Exception as exc:
            self.statusBar().showMessage("Save failed: %s" % exc)
            return
        self.statusBar().showMessage(
            "Saved correction to %s and %s" % (matlab_path, python_path)
        )

    def current_direction(self):
        return self.direction_combo.currentData()

    def _on_direction_changed(self, *_):
        reverse = self.current_direction() == PYTHON_TO_MATLAB
        self.source_label.setText("Source: Python" if reverse else "Source: MATLAB")
        self.output_label.setText("Output: MATLAB" if reverse else "Output: Python")
        self.python_pane.setPlainText("")
        self.python_highlighter.set_problem_lines([])
        self.accuracy_label.setText(accuracy_text(None))
        self.accuracy_label.setStyleSheet(accuracy_style(None))
        self.report_pane.clear()
        self.report_button.setText("Report")
        self.report_button.setChecked(False)

    def _set_section_marker(self, stage, status):
        label = self.section_labels[stage]
        label.setText("%s: %s" % (stage.title(), status))
        label.setStyleSheet(_MARKER_STYLE[section_marker(status)])

    def _toggle_details(self, checked):
        self.details_widget.setVisible(checked)
        self.details_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _toggle_report(self, checked):
        self.report_widget.setVisible(checked)
        self.report_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _update_report(self, result):
        entries = build_translation_report(result)
        lines = [report_text(entry) for entry in entries]
        self.report_pane.setPlainText(
            "\n".join(lines) if lines else "Nothing to report."
        )
        self.report_button.setText(
            "Report (%d)" % len(entries) if entries else "Report"
        )

    def _update_sections(self, sections):
        for stage, info in sections.items():
            self._set_section_marker(stage, info["status"])

    def _translate(self):
        source = self.matlab_pane.toPlainText()
        if not source.strip():
            self.statusBar().showMessage("Enter or load source first")
            return
        try:
            result = translate_source(
                source, direction=self.current_direction(), name=self.matlab_path
            )
        except Exception as exc:
            self.statusBar().showMessage("Translation failed: %s" % exc)
            self._set_section_marker("checker", "failed")
            return
        self.python_pane.setPlainText(result["python"])
        self.python_highlighter.set_problem_lines(result.get("problems", []))
        self._update_sections(result["sections"])
        self.status_line.setText(status_line(result))
        self._update_accuracy(result)
        self._update_report(result)
        self.statusBar().showMessage(
            "Translated %s (status=%s)" % (self.matlab_path, result["status"])
        )

    def _update_accuracy(self, result):
        if result.get("status") == "error":
            score = None
        else:
            score = accuracy(result)["score"]
        self.accuracy_label.setText(accuracy_text(score))
        self.accuracy_label.setStyleSheet(accuracy_style(score))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    app = QApplication(sys.argv)
    window = MinimalTranslatorWindow(matlab_path=path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
