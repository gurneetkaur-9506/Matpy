import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from PyQt5.QtWidgets import QApplication

from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB
from tests.paths import sample_matlab, sample_python
from ui.summary import ACCURACY_STYLE, accuracy_style, accuracy_text
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


class TestAccuracyLabel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_label_initially_unknown(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        self.assertEqual(win.accuracy_label.text(), "Accuracy: --")
        self.assertIn("#95a5a6", win.accuracy_label.styleSheet())
        win.close()

    def test_label_updates_after_translate(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.translate_button.click()
        self.assertEqual(win.accuracy_label.text(), "Accuracy: 100%")
        self.assertIn("#1e8e3e", win.accuracy_label.styleSheet())
        win.close()

    def test_label_resets_on_direction_change(self):
        win = TranslatorWindow(matlab_path=FFT_MATLAB)
        win.translate_button.click()
        self.assertEqual(win.accuracy_label.text(), "Accuracy: 100%")
        reverse_index = win.direction_combo.findData(PYTHON_TO_MATLAB)
        win.direction_combo.setCurrentIndex(reverse_index)
        self.assertEqual(win.accuracy_label.text(), "Accuracy: --")
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


if __name__ == "__main__":
    unittest.main()
