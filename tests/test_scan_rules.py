import unittest

from reader import MATLAB_TO_PYTHON, load_structure_from_source
from reader.structure import Statement
from rulebook import translate_with_rulebook
from rulebook.scan_rules import (
    translate_feof_loop,
    translate_feof_statement,
    translate_fopen,
    translate_fscanf,
)
from rulebook.translator import _translate_statement


def _fopen_io(target="fid", path="'data.txt'"):
    result = translate_fopen(target, "fopen(%s,'r')" % path)
    self = None
    assert result is not None
    return result[0], {target: result[1]}


class TestTranslateFopen(unittest.TestCase):
    def test_basic(self):
        line, record = translate_fopen("fid", "fopen('data.txt','r')")
        self.assertEqual(line, 'fid = open("data.txt", \'r\')')
        self.assertEqual(record["path"], '"data.txt"')
        self.assertEqual(record["mode"], "r")

    def test_write_mode(self):
        line, record = translate_fopen("fid", "fopen('out.bin','w')")
        self.assertEqual(line, 'fid = open("out.bin", \'w\')')

    def test_non_fopen_returns_none(self):
        self.assertIsNone(translate_fopen("fid", "max(x)"))

    def test_non_literal_path_returns_none(self):
        self.assertIsNone(translate_fopen("fid", "fopen(path_var, 'r')"))


class TestTranslateFscanf(unittest.TestCase):
    def test_with_known_handle(self):
        _, io = _fopen_io()
        line = translate_fscanf("amp", "fscanf(fid,'%f')", io)
        self.assertEqual(line, 'amp = read_matlab_scan_file("data.txt", "%f")')

    def test_unknown_handle_returns_none(self):
        self.assertIsNone(translate_fscanf("amp", "fscanf(fid,'%f')", {}))

    def test_format_variation(self):
        _, io = _fopen_io()
        for fmt in ("%f", "%d", "%x", "%s"):
            line = translate_fscanf("data", "fscanf(fid,'%s')" % fmt, io)
            self.assertIn('"%s"' % fmt, line)

    def test_non_fscanf_returns_none(self):
        _, io = _fopen_io()
        self.assertIsNone(translate_fscanf("amp", "fgetl(fid)", io))


class TestTranslateFeof(unittest.TestCase):
    def test_feof_loop_with_body(self):
        _, io = _fopen_io()
        body = [Statement("assignment", "amp=fscanf(fid,'%f')")]
        line = translate_feof_loop("~feof(fid)", body, io)
        self.assertEqual(line, 'amp = read_matlab_scan_file("data.txt", "%f")')

    def test_non_feof_header_returns_none(self):
        _, io = _fopen_io()
        body = [Statement("assignment", "amp=fscanf(fid,'%f')")]
        self.assertIsNone(translate_feof_loop("i < n", body, io))

    def test_feof_statement_raw_text(self):
        _, io = _fopen_io()
        text = "while ~feof(fid)\n    amp=fscanf(fid,'%f');\nend"
        line = translate_feof_statement(text, io)
        self.assertEqual(line, 'amp = read_matlab_scan_file("data.txt", "%f")')

    def test_feof_statement_unknown_handle(self):
        text = "while ~feof(fid)\n    amp=fscanf(fid,'%f');\nend"
        self.assertIsNone(translate_feof_statement(text, {}))


class TestScanPipeline(unittest.TestCase):
    def _translate(self, source):
        structure = load_structure_from_source(source, MATLAB_TO_PYTHON)
        result = translate_with_rulebook(structure)
        return [s["python"] for s in result["statements"]]

    def test_float_script(self):
        source = (
            "fid = fopen('amp.txt','r');\n"
            "while ~feof(fid)\n"
            "    amp=fscanf(fid,'%f');\n"
            "end\n"
            "amp1=reshape(amp,92,115);\n"
        )
        lines = self._translate(source)
        self.assertIn('fid = open("amp.txt", \'r\')', lines)
        self.assertIn('amp = read_matlab_scan_file("amp.txt", "%f")', lines)
        self.assertIn("amp1 = np.reshape(amp, (92, 115))", lines)

    def test_decimal_script(self):
        source = (
            "fid = fopen('n.txt','r');\n"
            "while ~feof(fid)\n"
            "    n=fscanf(fid,'%d');\n"
            "end\n"
        )
        lines = self._translate(source)
        self.assertIn('n = read_matlab_scan_file("n.txt", "%d")', lines)

    def test_hex_script(self):
        source = (
            "fid = fopen('iq.txt','r');\n"
            "while ~feof(fid)\n"
            "    iq=fscanf(fid,'%x');\n"
            "end\n"
        )
        lines = self._translate(source)
        self.assertIn('iq = read_matlab_scan_file("iq.txt", "%x")', lines)

    def test_string_script(self):
        source = (
            "fid = fopen('names.txt','r');\n"
            "while ~feof(fid)\n"
            "    names=fscanf(fid,'%s');\n"
            "end\n"
        )
        lines = self._translate(source)
        self.assertIn('names = read_matlab_scan_file("names.txt", "%s")', lines)

    def test_standalone_fscanf_without_loop(self):
        source = (
            "fid = fopen('amp.txt','r');\n"
            "amp=fscanf(fid,'%f');\n"
        )
        lines = self._translate(source)
        self.assertIn('amp = read_matlab_scan_file("amp.txt", "%f")', lines)

    def test_fscanf_before_fopen_remains_unresolved(self):
        source = "amp=fscanf(fid,'%f');\n"
        lines = self._translate(source)
        self.assertEqual(lines, ["UNRESOLVED"])

    def test_loop_inside_function(self):
        source = (
            "function out = f()\n"
            "    fid = fopen('in.dat','r')\n"
            "    while ~feof(fid)\n"
            "        s=fscanf(fid,'%s');\n"
            "    end\n"
            "end\n"
        )
        structure = load_structure_from_source(source, MATLAB_TO_PYTHON)
        result = translate_with_rulebook(structure)
        statements = [s["python"] for s in result["functions"][0]["statements"]]
        self.assertIn('s = read_matlab_scan_file("in.dat", "%s")', statements)

    def test_fclose_translation(self):
        source = "fclose(fid);\n"
        lines = self._translate(source)
        self.assertIn("fid.close()", lines)


class TestStatementLevel(unittest.TestCase):
    def test_fopen_statement_records_handle(self):
        io = {}
        out = _translate_statement(Statement("assignment", "fid = fopen('d.txt','r')"), io=io)
        self.assertIn("fid = open(", out["python"])
        self.assertIn("fid", io)

    def test_while_statement_kind_feof(self):
        io = {"fid": {"path": '"d.txt"', "mode": "r"}}
        stmt = Statement("while_statement", "while ~feof(fid)\n    x=fscanf(fid,'%d');\nend")
        out = _translate_statement(stmt, io=io)
        self.assertIn('x = read_matlab_scan_file("d.txt", "%d")', out["python"])

    def test_non_feof_while_stays_unresolved(self):
        io = {}
        stmt = Statement("while_statement", "while i < n\n    x = x + 1;\nend")
        out = _translate_statement(stmt, io=io)
        self.assertEqual(out["python"], "UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
