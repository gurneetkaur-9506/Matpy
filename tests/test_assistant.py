import unittest
from unittest import mock

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from assistant import draft_translation, draft_unresolved_functions, parse_response
from reader import build_structure, load_matlab_file
from rulebook import UNRESOLVED, translate_with_rulebook

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
- not sure about broadcasting
END UNSURE
"""


class TestParseResponse(unittest.TestCase):
    def test_parses_structured_response(self):
        parsed = parse_response(FAKE_RESPONSE)
        self.assertEqual(set(parsed.keys()), {"code", "confidence", "notes"})
        self.assertIn("import numpy as np", parsed["code"])
        self.assertIn("np.sum(x)", parsed["code"])
        self.assertAlmostEqual(parsed["confidence"], 0.6)
        self.assertIn("uncertainty flagged: assumed x is 1-D", parsed["notes"])
        self.assertIn("uncertainty flagged: not sure about broadcasting", parsed["notes"])

    def test_strips_markdown_fences(self):
        text = (
            "CODE\n```python\nx = 1\n```\nEND CODE\n"
            "CONFIDENCE\n1.0\nEND CONFIDENCE\n"
            "UNSURE\nnone\nEND UNSURE"
        )
        parsed = parse_response(text)
        self.assertEqual(parsed["code"], "x = 1")
        self.assertEqual(parsed["confidence"], 1.0)
        self.assertEqual(parsed["notes"], [])

    def test_missing_confidence_defaults_low(self):
        text = (
            "CODE\nx = 1\nEND CODE\n"
            "UNSURE\nnone\nEND UNSURE"
        )
        parsed = parse_response(text)
        self.assertEqual(parsed["code"], "x = 1")
        self.assertEqual(parsed["confidence"], 0.0)
        self.assertIn("model did not state a CONFIDENCE value", parsed["notes"])

    def test_missing_code_reported(self):
        text = (
            "CONFIDENCE\n0.5\nEND CONFIDENCE\n"
            "UNSURE\nnone\nEND UNSURE"
        )
        parsed = parse_response(text)
        self.assertEqual(parsed["code"], "")
        self.assertIn("no CODE section found in model response", parsed["notes"])

    def test_uncertain_language_without_flags_reported(self):
        text = (
            "CODE\nx = 1\nEND CODE\n"
            "CONFIDENCE\n0.9\nEND CONFIDENCE\n"
            "UNSURE\nnone\nEND UNSURE\n"
            "I think this is right, maybe."
        )
        parsed = parse_response(text)
        self.assertIn(
            "model used uncertain language but did not flag items in UNSURE",
            parsed["notes"],
        )


class TestDraftTranslation(unittest.TestCase):
    @mock.patch("assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE)
    def test_draft_calls_llm_and_returns_parsed_shape(self, mock_call):
        draft = draft_translation({"name": "f"}, {"libs": []})
        self.assertEqual(set(draft.keys()), {"code", "confidence", "notes"})
        self.assertIn("np.sum(x)", draft["code"])
        self.assertAlmostEqual(draft["confidence"], 0.6)
        self.assertTrue(any("assumed x is 1-D" in n for n in draft["notes"]))
        mock_call.assert_called_once()


class TestDraftRouting(unittest.TestCase):
    @mock.patch("assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE)
    def test_only_unresolved_functions_routed(self, mock_call):
        result = {
            "functions": [
                {"name": "ok", "statements": [{"python": "x = 1"}]},
                {"name": "bad", "statements": [{"python": UNRESOLVED}]},
            ],
            "statements": [],
        }
        routed = draft_unresolved_functions(result, {"libs": []})
        self.assertNotIn("draft", routed["functions"][0])
        self.assertIn("draft", routed["functions"][1])

    @mock.patch("assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE)
    def test_beamform_basic_routed(self, mock_call):
        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file("/sample_matlab/beamform_basic.m").encode("utf-8")
        )
        result = translate_with_rulebook(build_structure(tree))
        routed = draft_unresolved_functions(result, {"libs": []})
        func = routed["functions"][0]
        self.assertEqual(func["name"], "beamform_basic")
        self.assertIn("draft", func)
        self.assertIn("np.sum(x)", func["draft"]["code"])


if __name__ == "__main__":
    unittest.main()
