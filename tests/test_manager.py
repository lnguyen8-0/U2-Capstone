import unittest
from unittest.mock import patch, MagicMock
from agents import manager


def _mock_response(content, prompt_tokens=80, completion_tokens=2):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


QUAL_RESULT = {
    "answer": "Reviews require two approvals. [Source 1: company-policy.txt]",
    "chunks": [{"content": "Two approvals required.", "source": "company-policy.txt", "chunk": 0}],
    "input_tokens": 300,
    "output_tokens": 80,
}

QUANT_RESULT = {
    "answer": "Total revenue is $1,234,567.",
    "sql": "SELECT SUM(revenue) FROM sales",
    "rows": [(1234567,)],
    "validation": "PASSED",
    "input_tokens": 200,
    "output_tokens": 50,
}

REVIEW_RESULT = {
    "verdict": "SUPPORTED",
    "reason": "Answer is backed by context.",
    "supported": True,
    "input_tokens": 120,
    "output_tokens": 25,
}

NO_FLAG = {"flag": False, "warning": None}
FLAG = {"flag": True, "warning": "Response may not be grounded"}


class TestClassify(unittest.TestCase):
    @patch("agents.manager.log")
    @patch("agents.manager.client")
    def test_qualitative(self, mock_client, mock_log):
        mock_client.chat.completions.create.return_value = _mock_response("qualitative")
        self.assertEqual(manager.classify("Explain the leave policy"), "qualitative")

    @patch("agents.manager.log")
    @patch("agents.manager.client")
    def test_quantitative(self, mock_client, mock_log):
        mock_client.chat.completions.create.return_value = _mock_response("quantitative")
        self.assertEqual(manager.classify("Show monthly revenue"), "quantitative")

    @patch("agents.manager.log")
    @patch("agents.manager.client")
    def test_both(self, mock_client, mock_log):
        mock_client.chat.completions.create.return_value = _mock_response("both")
        self.assertEqual(manager.classify("Revenue trends and related policies"), "both")

    @patch("agents.manager.log")
    @patch("agents.manager.client")
    def test_fallback_to_qualitative_on_invalid_response(self, mock_client, mock_log):
        mock_client.chat.completions.create.return_value = _mock_response("gibberish")
        self.assertEqual(manager.classify("hi there"), "qualitative")

    @patch("agents.manager.log")
    @patch("agents.manager.client")
    def test_classify_logs_tokens(self, mock_client, mock_log):
        mock_client.chat.completions.create.return_value = _mock_response("qualitative")
        manager.classify("Some query")
        mock_log.assert_called_once()
        call_args = mock_log.call_args[0]
        self.assertEqual(call_args[1], "manager-classifier")


class TestManagerRun(unittest.TestCase):
    @patch("agents.manager.log")
    @patch("agents.manager.validate_qualitative", return_value=NO_FLAG)
    @patch("agents.manager.qualitative")
    @patch("agents.manager.classify", return_value="qualitative")
    def test_routes_to_qualitative(self, mock_classify, mock_qual, mock_validate, mock_log):
        mock_qual.run.return_value = QUAL_RESULT
        mock_qual.review.return_value = REVIEW_RESULT
        result = manager.run("Explain the leave policy")
        mock_qual.run.assert_called_once()
        mock_qual.review.assert_called_once()
        self.assertIn("two approvals", result)

    @patch("agents.manager.log")
    @patch("agents.manager.validate_quantitative", return_value=NO_FLAG)
    @patch("agents.manager.quantitative")
    @patch("agents.manager.classify", return_value="quantitative")
    def test_routes_to_quantitative(self, mock_classify, mock_quant, mock_validate, mock_log):
        mock_quant.run.return_value = QUANT_RESULT
        result = manager.run("What is total revenue?")
        mock_quant.run.assert_called_once()
        self.assertIn("1,234,567", result)

    @patch("agents.manager.log")
    @patch("agents.manager.validate_quantitative", return_value=NO_FLAG)
    @patch("agents.manager.validate_qualitative", return_value=NO_FLAG)
    @patch("agents.manager.quantitative")
    @patch("agents.manager.qualitative")
    @patch("agents.manager.classify", return_value="both")
    def test_routes_to_both(self, mock_classify, mock_qual, mock_quant, mock_vq, mock_vquant, mock_log):
        mock_qual.run.return_value = QUAL_RESULT
        mock_qual.review.return_value = REVIEW_RESULT
        mock_quant.run.return_value = QUANT_RESULT
        result = manager.run("Revenue trends and approval policies")
        mock_qual.run.assert_called_once()
        mock_quant.run.assert_called_once()
        self.assertIn("two approvals", result)
        self.assertIn("1,234,567", result)

    @patch("agents.manager.log")
    @patch("agents.manager.validate_qualitative", return_value=NO_FLAG)
    @patch("agents.manager.qualitative")
    @patch("agents.manager.classify", return_value="qualitative")
    def test_passes_history_to_qualitative(self, mock_classify, mock_qual, mock_validate, mock_log):
        mock_qual.run.return_value = QUAL_RESULT
        mock_qual.review.return_value = REVIEW_RESULT
        history = [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer"},
        ]
        manager.run("Follow-up question", history=history)
        mock_qual.run.assert_called_once_with("Follow-up question", history)

    @patch("agents.manager.log")
    @patch("agents.manager.validate_qualitative", return_value=FLAG)
    @patch("agents.manager.qualitative")
    @patch("agents.manager.classify", return_value="qualitative")
    def test_validation_warning_does_not_raise(self, mock_classify, mock_qual, mock_validate, mock_log):
        mock_qual.run.return_value = QUAL_RESULT
        mock_qual.review.return_value = REVIEW_RESULT
        try:
            manager.run("Some query that triggers a warning")
        except Exception as e:
            self.fail(f"manager.run raised unexpectedly: {e}")

    @patch("agents.manager.log")
    @patch("agents.manager.validate_qualitative", return_value=NO_FLAG)
    @patch("agents.manager.qualitative")
    @patch("agents.manager.classify", return_value="qualitative")
    def test_logs_qualitative_and_reviewer(self, mock_classify, mock_qual, mock_validate, mock_log):
        mock_qual.run.return_value = QUAL_RESULT
        mock_qual.review.return_value = REVIEW_RESULT
        manager.run("Any query")
        agent_names = [call[0][1] for call in mock_log.call_args_list]
        self.assertIn("qualitative", agent_names)
        self.assertIn("qualitative-reviewer", agent_names)

    @patch("agents.manager.log")
    @patch("agents.manager.validate_quantitative", return_value=NO_FLAG)
    @patch("agents.manager.quantitative")
    @patch("agents.manager.classify", return_value="quantitative")
    def test_quantitative_only_does_not_call_qualitative(self, mock_classify, mock_quant, mock_validate, mock_log):
        mock_quant.run.return_value = QUANT_RESULT
        with patch("agents.manager.qualitative") as mock_qual:
            manager.run("How many customers?")
            mock_qual.run.assert_not_called()
