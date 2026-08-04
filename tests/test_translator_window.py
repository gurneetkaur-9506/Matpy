import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from PyQt5.QtWidgets import QApplication

from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB
from ui.translator_window import TranslatorWindow

FFT_MATLAB = "/workspace/sample_matlab/fft_basic.m"
INDEXING_PYTHON = "/workspace/sample_python/indexing_ops_py.py"


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


if __name__ == "__main__":
    unittest.main()
