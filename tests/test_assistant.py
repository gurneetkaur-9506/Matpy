import unittest
from unittest import mock

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from assistant import draft_translation, draft_unresolved_functions, parse_response
from reader import (
    MATLAB_TO_PYTHON,
    PYTHON_TO_MATLAB,
    build_structure,
    load_matlab_file,
    load_structure,
)
from rulebook import (
    UNRESOLVED,
    translate_with_rulebook,
    translate_with_rulebook_reverse,
)
from tests.paths import sample_matlab, sample_python

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

LOW_CONFIDENCE_MATLAB_RESPONSE = """CODE
function af = beamform_basic(N, d, lamb, theta, theta0)
    k = 2 * pi / lamb;
    phase = k * d * (sin(theta) - sin(theta0));
    n = 0:N-1;
    af = sum(exp(1i * n.' * phase), 1);
end
END CODE
CONFIDENCE
0.3
END CONFIDENCE
UNSURE
- np.exp maps to multiple candidates (steervec, phased.SteeringVector, phased.ULA); chose steervec
- np.arange has no direct MATLAB equivalent; used 0:N-1
- broadcasting n[:, np.newaxis] has several MATLAB forms
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

    def test_parses_delimiterless_response(self):
        text = (
            "CODE\nimport numpy as np\n"
            "def f(x):\n"
            "    return np.sum(x)\n"
            "\n"
            "CONFIDENCE\n0.5\n"
            "\n"
            "UNSURE\n"
            "- assumed x is 1-D\n"
            "- not sure about broadcasting\n"
        )
        parsed = parse_response(text)
        self.assertEqual(parsed["code"], "import numpy as np\ndef f(x):\n    return np.sum(x)")
        self.assertAlmostEqual(parsed["confidence"], 0.5)
        self.assertIn("uncertainty flagged: assumed x is 1-D", parsed["notes"])
        self.assertIn("uncertainty flagged: not sure about broadcasting", parsed["notes"])

    def test_parses_delimiterless_low_confidence_response(self):
        text = (
            "CODE\nfunction af = beamform_basic(N, d, lamb, theta, theta0)\n"
            "    k = 2 * pi / lamb;\n"
            "end\n"
            "\n"
            "CONFIDENCE\n0.3\n"
            "\n"
            "UNSURE\n"
            "- np.exp maps to multiple candidates; chose steervec\n"
        )
        parsed = parse_response(text)
        self.assertEqual(
            parsed["code"], "function af = beamform_basic(N, d, lamb, theta, theta0)\n    k = 2 * pi / lamb;\nend"
        )
        self.assertAlmostEqual(parsed["confidence"], 0.3)
        self.assertEqual(
            parsed["notes"], ["uncertainty flagged: np.exp maps to multiple candidates; chose steervec"]
        )

    def test_delimiterless_without_unsure_yields_no_notes(self):
        text = "CODE\nx = 1\n\nCONFIDENCE\n0.7\n"
        parsed = parse_response(text)
        self.assertEqual(parsed["code"], "x = 1")
        self.assertAlmostEqual(parsed["confidence"], 0.7)
        self.assertEqual(parsed["notes"], [])

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


class TestDraftTranslationDirection(unittest.TestCase):
    @mock.patch("assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE)
    def test_reverse_direction_uses_reverse_lookup_context(self, mock_call):
        func = {
            "name": "beamform_basic",
            "statements": [
                {"source": "af = np.exp(1j * n[:, np.newaxis] * phase).sum(axis=0)"}
            ],
        }
        draft_translation(func, {"specialist": "ignored"}, direction=PYTHON_TO_MATLAB)
        prompt = mock_call.call_args[0][0]
        self.assertIn("reverse-lookup candidates", prompt)
        self.assertIn("steervec", prompt)
        self.assertNotIn("Available specialist library", prompt)
        self.assertIn("Python-to-MATLAB", prompt)
        self.assertIn("PYTHON function", prompt)

    @mock.patch("assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE)
    def test_forward_direction_uses_specialist_library(self, mock_call):
        func = {"name": "f", "statements": [{"source": "y = x + 1;"}]}
        draft_translation(func, {"libs": ["steering_vector"]})
        prompt = mock_call.call_args[0][0]
        self.assertIn("specialist library", prompt)
        self.assertIn("steering_vector", prompt)
        self.assertIn("MATLAB function", prompt)
        self.assertIn("MATLAB-to-Python", prompt)

    @mock.patch("assistant.draft_translation._call_ollama", return_value=FAKE_RESPONSE)
    def test_forward_default_direction_matches_legacy(self, mock_call):
        func = {"name": "f", "statements": [{"source": "y = x + 1;"}]}
        draft_translation(func, {"libs": []})
        prompt = mock_call.call_args[0][0]
        self.assertIn("MATLAB-to-Python", prompt)
        self.assertIn("MATLAB function", prompt)


class TestDraftTranslationReverse(unittest.TestCase):
    def _beamform_basic_func(self):
        structure = load_structure(
            sample_python("beamform_basic_py.py"), PYTHON_TO_MATLAB
        )
        result = translate_with_rulebook_reverse(structure)
        self.assertEqual(result["functions"][0]["name"], "beamform_basic")
        return result["functions"][0]

    @mock.patch(
        "assistant.draft_translation._call_ollama",
        return_value=LOW_CONFIDENCE_MATLAB_RESPONSE,
    )
    def test_reverse_presents_candidates_not_a_single_answer(self, mock_call):
        func = self._beamform_basic_func()
        draft = draft_translation(func, None, direction=PYTHON_TO_MATLAB)
        prompt = mock_call.call_args[0][0]

        self.assertIn("reverse-lookup candidates", prompt)
        self.assertIn("PYTHON function", prompt)
        self.assertIn("Python-to-MATLAB", prompt)

        for op in ("np.exp", "np.arange", "np.sin", "np.sum", "np.newaxis"):
            self.assertIn(op, prompt)
        self.assertIn("steervec", prompt)
        self.assertIn("phased.SteeringVector", prompt)
        self.assertIn("phased.ULA", prompt)

        self.assertLess(draft["confidence"], 0.5)
        self.assertTrue(
            any("multiple candidates" in n.lower() for n in draft["notes"])
        )

    @mock.patch(
        "assistant.draft_translation._call_ollama",
        return_value=LOW_CONFIDENCE_MATLAB_RESPONSE,
    )
    def test_reverse_flags_ambiguity_in_notes(self, mock_call):
        func = self._beamform_basic_func()
        draft = draft_translation(func, None, direction=PYTHON_TO_MATLAB)
        self.assertEqual(draft["confidence"], 0.3)
        self.assertTrue(
            any("uncertainty flagged:" in n for n in draft["notes"])
        )
        self.assertTrue(
            any("phased.SteeringVector" in n for n in draft["notes"])
        )


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
    def test_beamform_basic_not_routed(self, mock_call):
        parser = Parser(Language(language()))
        tree = parser.parse(
            load_matlab_file(sample_matlab("beamform_basic.m")).encode("utf-8")
        )
        result = translate_with_rulebook(build_structure(tree))
        routed = draft_unresolved_functions(result, {"libs": []})
        func = routed["functions"][0]
        self.assertEqual(func["name"], "beamform_basic")
        self.assertNotIn("draft", func)
        mock_call.assert_not_called()


class TestOllamaUnavailableClassification(unittest.TestCase):
    @mock.patch(
        "assistant.draft_translation._call_ollama",
        side_effect=ConnectionError("connection refused"),
    )
    def test_connection_error_maps_to_unavailable_message(self, mock_call):
        result = {
            "functions": [{"name": "f", "statements": [{"python": UNRESOLVED}]}],
            "statements": [],
        }
        routed = draft_unresolved_functions(result, {"libs": []})
        error = routed["functions"][0]["draft_error"]
        self.assertIn("Ollama not running", error)
        self.assertIn("not a missing rule", error)

    @mock.patch(
        "assistant.draft_translation._call_ollama",
        side_effect=RuntimeError("model timed out"),
    )
    def test_runtime_error_keeps_original_message(self, mock_call):
        result = {
            "functions": [{"name": "f", "statements": [{"python": UNRESOLVED}]}],
            "statements": [],
        }
        routed = draft_unresolved_functions(result, {"libs": []})
        self.assertEqual(
            routed["functions"][0]["draft_error"], "model timed out"
        )


if __name__ == "__main__":
    unittest.main()
