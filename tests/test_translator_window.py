import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from PyQt5.QtWidgets import QApplication

from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB
from tests.paths import sample_matlab, sample_python
from ui.summary import ACCURACY_STYLE, accuracy_style, accuracy_text, summary_line
from ui.translator_window import TranslatorWindow

FFT_MATLAB = sample_matlab("fft_basic.m")
INDEXING_PYTHON = sample_python("indexing_ops_py.py")

FAKE_RESPONSE = """CODE
import numpy as np

def f(x):
    return np.sum(x)
END CODE
CONFIDENCE
0.6
END CONFIDENCE
UNSURE
- assumed x is 1-D
END UNSURE
"""


class TestTranslatorWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_loads_matlab_into_left_pane(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        text = win.matlab_pane.toPlainText()
        self.assertIn("fs = 1000;", text)
        self.assertIn("plot(f, P1);", text)
        self.assertEqual(win.python_pane.toPlainText(), "")
        win.close()

    def test_window_title(self):
        win = TranslatorWindow()
        self.assertEqual(win.windowTitle(), "MATPY Translator")
        win.close()

    def test_translate_button_populates_python_pane(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        self.assertEqual(win.python_pane.toPlainText(), "")
        win.translate_button.click()
        self.assertIn("import numpy as np", win.python_pane.toPlainText())
        self.assertIn("plt.plot(f, P1)", win.python_pane.toPlainText())
        win.close()

    def test_translate_without_file_shows_message(self):
        win = TranslatorWindow()
        win.translate_button.click()
        self.assertIn("No MATLAB file loaded", win.statusBar().currentMessage())
        win.close()

    def test_sections_pending_before_translate(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        for stage, label in win.section_labels.items():
            self.assertIn("pending", label.text())
            self.assertIn("#95a5a6", label.styleSheet())
        win.close()

    def test_sections_marked_after_translate(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.translate_button.click()
        self.assertIn("Reader: ok", win.section_labels["reader"].text())
        self.assertIn("#2ecc71", win.section_labels["reader"].styleSheet())
        self.assertIn("Rulebook: ok", win.section_labels["rulebook"].text())
        self.assertIn("Assistant: none", win.section_labels["assistant"].text())
        self.assertIn("#2ecc71", win.section_labels["assistant"].styleSheet())
        self.assertIn("Checker: skipped", win.section_labels["checker"].text())
        self.assertIn("#f1c40f", win.section_labels["checker"].styleSheet())
        win.close()

    def test_direction_defaults_to_matlab_to_python(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        self.assertEqual(win.current_direction(), MATLAB_TO_PYTHON)
        self.assertIn("MATLAB -> Python", win.direction_combo.currentText())
        self.assertEqual(win.source_label.text(), "Source: MATLAB")
        self.assertEqual(win.output_label.text(), "Output: Python")
        win.close()

    def test_direction_toggle_relabels_panes(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.python_pane.setPlainText("stale output")
        reverse_index = win.direction_combo.findData(PYTHON_TO_MATLAB)
        win.direction_combo.setCurrentIndex(reverse_index)
        self.assertEqual(win.current_direction(), PYTHON_TO_MATLAB)
        self.assertEqual(win.source_label.text(), "Source: Python")
        self.assertEqual(win.output_label.text(), "Output: MATLAB")
        self.assertEqual(win.python_pane.toPlainText(), "")
        win.close()

    def test_translate_reverse_routes_to_rulebook_reverse(self):
        win = TranslatorWindow(matlab_path=INDEXING_PYTHON)
        reverse_index = win.direction_combo.findData(PYTHON_TO_MATLAB)
        win.direction_combo.setCurrentIndex(reverse_index)
        win.translate_button.click()
        output = win.python_pane.toPlainText()
        self.assertIn("A = [1 2 3; 4 5 6];", output)
        self.assertIn("disp(A(1, 1));", output)
        self.assertNotIn("import numpy as np", output)
        self.assertIn("Rulebook: ok", win.section_labels["rulebook"].text())
        win.close()

    def test_translate_forward_keeps_rulebook_forward(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.direction_combo.setCurrentIndex(0)
        win.translate_button.click()
        output = win.python_pane.toPlainText()
        self.assertIn("import numpy as np", output)
        self.assertIn("plt.plot(f, P1)", output)
        win.close()

    def test_highlighter_applies_background_on_problem_lines(self):
        from PyQt5.QtGui import QTextDocument

        from ui.highlight import ProblemLineHighlighter

        doc = QTextDocument()
        doc.setPlainText("clean\n# UNRESOLVED: x = find(a, b)\n")
        highlighter = ProblemLineHighlighter(doc, problem_lines=[1])
        highlighter.rehighlight()
        self.app.processEvents()
        block = doc.findBlockByNumber(1)
        fmts = block.layout().formats()
        self.assertTrue(fmts)
        self.assertEqual(fmts[0].format.background().color().name(), "#ffeb3b")

    def test_highlighter_empty_when_fully_resolved(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.translate_button.click()
        self.assertEqual(win.python_highlighter._problem_lines, set())
        win.close()

    def test_highlighter_marks_problem_lines_after_translate(self):
        from unittest import mock

        win = TranslatorWindow(matlab_path=sample_python("beamform_basic_py.py"))
        reverse_index = win.direction_combo.findData(PYTHON_TO_MATLAB)
        win.direction_combo.setCurrentIndex(reverse_index)
        with mock.patch(
            "assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE
        ):
            win.translate_button.click()
        self.assertTrue(win.python_highlighter._problem_lines)
        win.close()

    def test_empty_state_shows_placeholder_message_and_open_button(self):
        from PyQt5.QtWidgets import QLabel, QPushButton

        win = TranslatorWindow()
        self.assertIs(win.stack.currentWidget(), win.placeholder)
        messages = [w.text() for w in win.placeholder.findChildren(QLabel)]
        self.assertIn("Open a MATLAB or Python file to translate it", messages)
        buttons = win.placeholder.findChildren(QPushButton)
        self.assertEqual(len(buttons), 1)
        self.assertEqual(buttons[0].text(), "Open")
        win.close()

    def test_loaded_state_shows_splitter(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        self.assertIs(win.stack.currentWidget(), win.splitter)
        win.close()

    def test_open_python_file_sets_reverse_direction_and_loads(self):
        from unittest import mock

        from PyQt5.QtWidgets import QPushButton

        win = TranslatorWindow()
        with mock.patch(
            "ui.translator_window.QFileDialog.getOpenFileName",
            return_value=(INDEXING_PYTHON, ""),
        ):
            win.placeholder.findChildren(QPushButton)[0].click()
        self.assertIs(win.stack.currentWidget(), win.splitter)
        self.assertEqual(win.current_direction(), PYTHON_TO_MATLAB)
        self.assertIn("A = np.array", win.matlab_pane.toPlainText())
        win.close()

    def test_open_matlab_file_keeps_forward_direction_and_loads(self):
        from unittest import mock

        from PyQt5.QtWidgets import QPushButton

        win = TranslatorWindow()
        with mock.patch(
            "ui.translator_window.QFileDialog.getOpenFileName",
            return_value=(FFT_MATLAB, ""),
        ):
            win.placeholder.findChildren(QPushButton)[0].click()
        self.assertIs(win.stack.currentWidget(), win.splitter)
        self.assertEqual(win.current_direction(), MATLAB_TO_PYTHON)
        self.assertIn("fs = 1000;", win.matlab_pane.toPlainText())
        win.close()


class TestSummaryLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_summary_initially_unknown(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        self.assertEqual(win.status_line.text(), "Not translated yet")
        self.assertIn("#95a5a6", win.status_line.styleSheet())
        win.close()

    def test_summary_updates_after_translate(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.translate_button.click()
        text = win.status_line.text()
        self.assertIn("lines translated", text)
        self.assertIn("0 need review", text)
        self.assertIn("accuracy 100%", text)
        self.assertIn("#1e8e3e", win.status_line.styleSheet())
        win.close()

    def test_summary_resets_on_direction_change(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.translate_button.click()
        self.assertIn("accuracy 100%", win.status_line.text())
        reverse_index = win.direction_combo.findData(PYTHON_TO_MATLAB)
        win.direction_combo.setCurrentIndex(reverse_index)
        self.assertEqual(win.status_line.text(), "Not translated yet")
        self.assertIn("#95a5a6", win.status_line.styleSheet())
        win.close()

    def test_details_hidden_by_default(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.show()
        self.assertFalse(win.details_widget.isVisible())
        self.assertFalse(win.details_button.isChecked())
        win.close()

    def test_details_toggles(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.show()
        win.details_button.click()
        self.assertTrue(win.details_widget.isVisible())
        win.details_button.click()
        self.assertFalse(win.details_widget.isVisible())
        win.close()

    def test_style_green_above_90(self):
        self.assertEqual(accuracy_style(95), ACCURACY_STYLE["high"])
        self.assertEqual(accuracy_style(100), ACCURACY_STYLE["high"])

    def test_style_yellow_from_70_to_90(self):
        self.assertEqual(accuracy_style(70), ACCURACY_STYLE["mid"])
        self.assertEqual(accuracy_style(90), ACCURACY_STYLE["mid"])

    def test_style_red_below_70(self):
        self.assertEqual(accuracy_style(69), ACCURACY_STYLE["low"])
        self.assertEqual(accuracy_style(0), ACCURACY_STYLE["low"])

    def test_style_unknown_when_no_score(self):
        self.assertEqual(accuracy_style(None), ACCURACY_STYLE["unknown"])

    def test_text_formatting(self):
        self.assertEqual(accuracy_text(87), "Accuracy: 87%")
        self.assertEqual(accuracy_text(99.6), "Accuracy: 100%")
        self.assertEqual(accuracy_text(None), "Accuracy: --")


class TestSummaryLineFunction(unittest.TestCase):
    def test_clean_file_reports_full_summary(self):
        from translator import translate_file

        result = translate_file(FFT_MATLAB)
        text = summary_line(result)
        self.assertIn("lines translated", text)
        self.assertIn("0 need review", text)
        self.assertIn("accuracy 100%", text)
        self.assertTrue(text.endswith("."))

    def test_unresolved_file_reports_review_count(self):
        from unittest import mock

        from translator import translate_file

        with mock.patch(
            "assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE
        ):
            result = translate_file(
                sample_python("beamform_basic_py.py"), direction=PYTHON_TO_MATLAB
            )
        text = summary_line(result)
        self.assertIn("need review", text)
        self.assertNotIn("accuracy 100%", text)


class TestReportText(unittest.TestCase):
    def test_formats_like_expected_example(self):
        from ui.summary import report_text

        entry = {
            "line": 47,
            "source": "interp1 with 3 outputs",
            "reason": "not yet supported, left as TODO comment.",
            "issue": "unresolved",
            "stage": "rulebook",
        }
        self.assertEqual(
            report_text(entry),
            "Line 47: interp1 with 3 outputs - not yet supported, left as "
            "TODO comment.",
        )

    def test_checker_entry_uses_stage_prefix(self):
        from ui.summary import report_text

        entry = {
            "line": None,
            "source": "/path/to/file.m",
            "reason": "The checker could not decide whether the outputs match.",
            "issue": "review needed",
            "stage": "checker",
        }
        text = report_text(entry)
        self.assertTrue(text.startswith("Checker: "))
        self.assertIn("could not decide", text)

    def test_entry_without_source_uses_reason_only(self):
        from ui.summary import report_text

        entry = {
            "line": 3,
            "source": "",
            "reason": "No rule matches the function 'fft'.",
            "issue": "unresolved",
            "stage": "rulebook",
        }
        self.assertEqual(report_text(entry), "Line 3: No rule matches the function 'fft'.")


class TestReportPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_panel_collapsed_by_default(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.show()
        self.assertFalse(win.report_widget.isVisible())
        self.assertFalse(win.report_button.isChecked())
        self.assertEqual(win.report_button.text(), "Report")
        win.close()

    def test_toggle_expands_and_collapses(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.show()
        win.report_button.click()
        self.assertTrue(win.report_widget.isVisible())
        win.report_button.click()
        self.assertFalse(win.report_widget.isVisible())
        win.close()

    def test_report_populated_after_translate_with_issues(self):
        from unittest import mock

        win = TranslatorWindow(matlab_path=sample_python("beamform_basic_py.py"))
        reverse_index = win.direction_combo.findData(PYTHON_TO_MATLAB)
        win.direction_combo.setCurrentIndex(reverse_index)
        with mock.patch(
            "assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE
        ):
            win.translate_button.click()
        text = win.report_pane.toPlainText()
        self.assertIn("Line", text)
        self.assertIn("n = np.arange(N)", text)
        self.assertEqual(win.report_button.text(), "Report (%d)" % len(text.splitlines()))
        win.close()

    def test_report_empty_state_after_clean_translate(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.translate_button.click()
        self.assertEqual(win.report_pane.toPlainText(), "Nothing to report.")
        self.assertEqual(win.report_button.text(), "Report")
        win.close()

    def test_report_reset_on_direction_change(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.show()
        win.translate_button.click()
        self.assertEqual(win.report_pane.toPlainText(), "Nothing to report.")
        reverse_index = win.direction_combo.findData(PYTHON_TO_MATLAB)
        win.direction_combo.setCurrentIndex(reverse_index)
        self.assertEqual(win.report_pane.toPlainText(), "")
        self.assertFalse(win.report_widget.isVisible())
        self.assertEqual(win.report_button.text(), "Report")
        win.close()

    def test_clean_translate_reports_nothing(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.translate_button.click()
        self.assertEqual(win.report_pane.toPlainText(), "Nothing to report.")
        win.close()


if __name__ == "__main__":
    unittest.main()
