from PyQt5.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat

_PROBLEM_FORMAT = QTextCharFormat()
_PROBLEM_FORMAT.setBackground(QColor(255, 235, 59, 140))


class ProblemLineHighlighter(QSyntaxHighlighter):
    def __init__(self, document, problem_lines=()):
        super().__init__(document)
        self._problem_lines = set(problem_lines)

    def set_problem_lines(self, problem_lines):
        self._problem_lines = set(problem_lines)
        self.rehighlight()

    def highlightBlock(self, text):
        if self.currentBlock().blockNumber() in self._problem_lines:
            self.setFormat(0, max(len(text), 1), _PROBLEM_FORMAT)
