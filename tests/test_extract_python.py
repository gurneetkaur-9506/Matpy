import ast
import unittest

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from reader import extract_structure, load_matlab_file
from tests.paths import sample_matlab, sample_python


def _parse_py(name):
    with open(sample_python(name), "r", encoding="utf-8") as f:
        return ast.parse(f.read())


class TestExtractPythonStructure(unittest.TestCase):
    def test_unified_output_format(self):
        result = extract_structure(_parse_py("indexing_ops_py.py"))
        self.assertEqual(sorted(result.keys()), ["functions", "refs"])
        self.assertEqual(result["functions"], [])

    def test_index_refs_from_subscripts(self):
        result = extract_structure(_parse_py("indexing_ops_py.py"))
        index_refs = [r for r in result["refs"] if r["kind"] == "index"]
        a_indices = [r["indices"] for r in index_refs if r["name"] == "A"]
        self.assertIn(["0", "0"], a_indices)
        self.assertIn(["1", "2"], a_indices)
        self.assertIn(["0", ":"], a_indices)

    def test_plain_refs(self):
        result = extract_structure(_parse_py("indexing_ops_py.py"))
        plain = {r["name"] for r in result["refs"] if r["kind"] == "plain"}
        self.assertIn("A", plain)
        self.assertIn("B", plain)

    def test_python_function_extraction(self):
        result = extract_structure(_parse_py("beamform_basic_py.py"))
        self.assertEqual(len(result["functions"]), 1)
        func = result["functions"][0]
        self.assertEqual(func["name"], "beamform_basic")
        self.assertIn("return af", func["body"])
        self.assertIn("loops", func)

    def test_matlab_path_unchanged(self):
        parser = Parser(Language(language()))
        tree = parser.parse(load_matlab_file(sample_matlab("fft_basic.m")).encode("utf-8"))
        result = extract_structure(tree)
        self.assertEqual(result["functions"], [])
        index_refs = [r for r in result["refs"] if r["kind"] == "index"]
        self.assertIn(
            {"kind": "index", "name": "P2", "indices": ["1:length(P2)/2+1"]},
            index_refs,
        )


if __name__ == "__main__":
    unittest.main()
