import unittest
from validation.validator import validate_qualitative, validate_quantitative


def _chunk(source="company-policy.txt"):
    return {"content": "Some content.", "source": source, "chunk": 0}


class TestValidateQualitative(unittest.TestCase):
    def test_grounded_when_source_cited(self):
        result = validate_qualitative("See [Source 1] for details.", [_chunk()])
        self.assertTrue(result["is_grounded"])
        self.assertFalse(result["flag"])
        self.assertIsNone(result["warning"])

    def test_flagged_when_no_source_cited(self):
        result = validate_qualitative("The answer is 42.", [_chunk()])
        self.assertFalse(result["is_grounded"])
        self.assertTrue(result["flag"])
        self.assertIsNotNone(result["warning"])

    def test_refusal_not_flagged(self):
        result = validate_qualitative(
            "I cannot find this information in the provided documents.", [_chunk()]
        )
        self.assertFalse(result["flag"])
        self.assertTrue(result["refused_to_answer"])

    def test_sources_cited_list(self):
        chunks = [_chunk("doc1.txt"), _chunk("doc2.txt")]
        result = validate_qualitative("See [Source 1] and [Source 2].", chunks)
        self.assertEqual(result["sources_cited"], ["doc1.txt", "doc2.txt"])

    def test_only_cited_sources_included(self):
        chunks = [_chunk("doc1.txt"), _chunk("doc2.txt"), _chunk("doc3.txt")]
        result = validate_qualitative("See [Source 2].", chunks)
        self.assertEqual(result["sources_cited"], ["doc2.txt"])

    def test_empty_answer_flagged(self):
        result = validate_qualitative("", [_chunk()])
        self.assertFalse(result["is_grounded"])

    def test_no_chunks_unflagged_refusal(self):
        result = validate_qualitative("I cannot find this information in the provided documents.", [])
        self.assertFalse(result["flag"])


class TestValidateQuantitative(unittest.TestCase):
    def test_passed_status(self):
        result = validate_quantitative("SELECT * FROM sales", "PASSED")
        self.assertTrue(result["sql_validated"])
        self.assertFalse(result["flag"])
        self.assertIsNone(result["warning"])

    def test_failed_status(self):
        result = validate_quantitative("DROP TABLE sales", "FAILED")
        self.assertFalse(result["sql_validated"])
        self.assertTrue(result["sql_blocked"])
        self.assertTrue(result["flag"])
        self.assertIn("FAILED", result["warning"])

    def test_error_status(self):
        result = validate_quantitative("SELECT * FROM nonexistent", "ERROR")
        self.assertTrue(result["execution_error"])
        self.assertTrue(result["flag"])
        self.assertIn("ERROR", result["warning"])

    def test_unknown_status_is_flagged(self):
        result = validate_quantitative("SELECT 1", "UNKNOWN")
        self.assertTrue(result["flag"])
