import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from PyQt5.QtWidgets import QApplication

from ui.translator_window import TranslatorWindow


class TestTranslatorWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_loads_matlab_into_left_pane(self):
        win = TranslatorWindow(matlab_path="/sample_matlab/fft_basic.m")
        text = win.matlab_pane.toPlainText()
        self.assertIn("fs = 1000;", text)
        self.assertIn("plot(f, P1);", text)
        self.assertEqual(win.python_pane.toPlainText(), "")

    def test_window_title(self):
        win = TranslatorWindow()
        self.assertEqual(win.windowTitle(), "MATPY Translator")
        win.close()

    def test_translate_button_populates_python_pane(self):
        win = TranslatorWindow(matlab_path="/sample_matlab/fft_basic.m")
        self.assertEqual(win.python_pane.toPlainText(), "")
        win.translate_button.click()
        self.assertIn("import numpy as np", win.python_pane.toPlainText())
        self.assertIn("plt.plot(f, P1)", win.python_pane.toPlainText())

    def test_translate_without_file_shows_message(self):
        win = TranslatorWindow()
        win.translate_button.click()
        self.assertIn("No MATLAB file loaded", win.statusBar().currentMessage())
        win.close()

    def test_sections_pending_before_translate(self):
        win = TranslatorWindow(matlab_path="/sample_matlab/fft_basic.m")
        for stage, label in win.section_labels.items():
            self.assertIn("pending", label.text())
            self.assertIn("#95a5a6", label.styleSheet())
        win.close()

    def test_sections_marked_after_translate(self):
        win = TranslatorWindow(matlab_path="/sample_matlab/fft_basic.m")
        win.translate_button.click()
        self.assertIn("Reader: ok", win.section_labels["reader"].text())
        self.assertIn("#2ecc71", win.section_labels["reader"].styleSheet())
        self.assertIn("Rulebook: ok", win.section_labels["rulebook"].text())
        self.assertIn("Assistant: none", win.section_labels["assistant"].text())
        self.assertIn("#2ecc71", win.section_labels["assistant"].styleSheet())
        self.assertIn("Checker: skipped", win.section_labels["checker"].text())
        self.assertIn("#f1c40f", win.section_labels["checker"].styleSheet())
        win.close()


if __name__ == "__main__":
    unittest.main()
