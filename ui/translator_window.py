import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from repo_paths import sample_matlab
from checker import accuracy, build_translation_report
from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB, load_matlab_file
from translator import translate_file
from ui.highlight import ProblemLineHighlighter
from ui.summary import accuracy_style, report_text, summary_line

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


class TranslatorWindow(QMainWindow):
    def __init__(self, matlab_path=None):
        super().__init__()
        self.setWindowTitle("MATPY Translator")
        self.resize(1000, 640)
        self.matlab_path = matlab_path

        self.matlab_pane = QPlainTextEdit()
        self.matlab_pane.setReadOnly(True)
        self.source_label = QLabel("Source: MATLAB")
        self.output_label = QLabel("Output: Python")
        self.python_pane = QPlainTextEdit()
        self.python_pane.setReadOnly(True)
        self.python_highlighter = ProblemLineHighlighter(self.python_pane.document())

        self.splitter = QSplitter()
        self.splitter.addWidget(self.matlab_pane)
        self.splitter.addWidget(self.python_pane)
        self.splitter.setSizes([500, 500])

        self.placeholder = self._build_placeholder()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(self.splitter)

        self.translate_button = QPushButton("Translate")
        self.translate_button.clicked.connect(self._translate)

        self.direction_combo = QComboBox()
        for label, direction in _DIRECTION_ITEMS:
            self.direction_combo.addItem(label, direction)
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)

        buttons = QHBoxLayout()
        buttons.addWidget(self.direction_combo)
        buttons.addWidget(self.translate_button)
        buttons.addStretch(1)

        pane_labels = QHBoxLayout()
        pane_labels.addWidget(self.source_label)
        pane_labels.addStretch(1)
        pane_labels.addWidget(self.output_label)

        self.status_line = QLabel("Not translated yet")
        self.status_line.setStyleSheet(accuracy_style(None))
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
        summary_row.addWidget(self.details_button)
        summary_row.addWidget(self.report_button)

        central = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(buttons)
        layout.addLayout(summary_row)
        layout.addWidget(self.details_widget)
        layout.addLayout(pane_labels)
        layout.addWidget(self.stack)
        layout.addWidget(self.report_widget)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")
        if matlab_path:
            self.load_matlab(matlab_path)
        else:
            self.stack.setCurrentWidget(self.placeholder)

    def _build_placeholder(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addStretch(1)
        message = QLabel("Open a MATLAB or Python file to translate it")
        message.setAlignment(Qt.AlignCenter)
        layout.addWidget(message)
        open_button = QPushButton("Open")
        open_button.clicked.connect(self._open_file)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(open_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

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
        self.matlab_pane.setPlainText(load_matlab_file(path))
        self.stack.setCurrentWidget(self.splitter)
        self.statusBar().showMessage("Loaded %s" % path)

    def current_direction(self):
        return self.direction_combo.currentData()

    def _on_direction_changed(self, *_):
        reverse = self.current_direction() == PYTHON_TO_MATLAB
        self.source_label.setText("Source: Python" if reverse else "Source: MATLAB")
        self.output_label.setText("Output: MATLAB" if reverse else "Output: Python")
        self.python_pane.setPlainText("")
        self.python_highlighter.set_problem_lines([])
        self.status_line.setText("Not translated yet")
        self.status_line.setStyleSheet(accuracy_style(None))
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
        if not self.matlab_path:
            self.statusBar().showMessage("No MATLAB file loaded")
            return
        try:
            result = translate_file(self.matlab_path, direction=self.current_direction())
        except Exception as exc:
            self.statusBar().showMessage("Translation failed: %s" % exc)
            self._set_section_marker("checker", "failed")
            return
        self.python_pane.setPlainText(result["python"])
        self.python_highlighter.set_problem_lines(result.get("problems", []))
        self._update_sections(result["sections"])
        self._update_summary(result)
        self._update_report(result)
        self.statusBar().showMessage(
            "status=%s checker=%s" % (result["status"], result["sections"]["checker"]["status"])
        )

    def _update_summary(self, result):
        self.status_line.setText(summary_line(result))
        self.status_line.setStyleSheet(accuracy_style(accuracy(result)["score"]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sample_matlab("fft_basic.m")
    app = QApplication(sys.argv)
    window = TranslatorWindow(matlab_path=path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
