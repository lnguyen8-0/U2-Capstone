"""
Integration tests for multi-agent workflows.
All external I/O (OpenAI API, ChromaDB, SQLite) is mocked;
real module wiring (manager → qualitative/quantitative → validator) is exercised.
"""
import unittest
from unittest.mock import patch, MagicMock
from agents import manager


SAMPLE_CHROMA_RESULTS = {
    "documents": [["Two approvals required.", "Submit PRs via GitHub."]],
    "metadatas": [[
        {"source": "company-policy.txt", "chunk": 0},
        {"source": "company-policy.txt", "chunk": 1},
    ]],
}


def _api(content, prompt_tokens=200, completion_tokens=80):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _patch_encode(mock_encode):
    """embedding_model.encode() returns a numpy-array-like object; mock .tolist() accordingly."""
    embed = MagicMock()
    embed.tolist.return_value = [[0.1] * 384]
    mock_encode.return_value = embed


def _patch_chroma(mock_chroma):
    mock_chroma.return_value.get_collection.return_value.query.return_value = SAMPLE_CHROMA_RESULTS


def _patch_sqlite(mock_connect, rows, description):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    mock_cursor.description = [(col,) for col in description]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn


class TestQualitativeWorkflow(unittest.TestCase):
    """Classifier → qualitative agent → heuristic validator → LLM reviewer."""

    @patch("agents.manager.log")
    @patch("agents.qualitative.chromadb.PersistentClient")
    @patch("agents.qualitative.embedding_model.encode")
    @patch("agents.manager.client")
    @patch("agents.qualitative.client")
    def test_grounded_answer_passes_validation(
        self, qual_client, mgr_client, mock_encode, mock_chroma, mock_log
    ):
        _patch_encode(mock_encode)
        _patch_chroma(mock_chroma)
        mgr_client.chat.completions.create.return_value = _api("qualitative", 80, 2)
        qual_client.chat.completions.create.side_effect = [
            _api("Code reviews require two approvals. [Source 1: company-policy.txt]", 350, 90),
            _api("VERDICT: SUPPORTED\nREASON: Backed by retrieved context.", 150, 20),
        ]
        result = manager.run("Explain the code review process")
        self.assertIn("two approvals", result)

    @patch("agents.manager.log")
    @patch("agents.qualitative.chromadb.PersistentClient")
    @patch("agents.qualitative.embedding_model.encode")
    @patch("agents.manager.client")
    @patch("agents.qualitative.client")
    def test_refusal_not_flagged(self, qual_client, mgr_client, mock_encode, mock_chroma, mock_log):
        _patch_encode(mock_encode)
        _patch_chroma(mock_chroma)
        mgr_client.chat.completions.create.return_value = _api("qualitative", 80, 2)
        qual_client.chat.completions.create.side_effect = [
            _api("I cannot find this information in the provided documents.", 300, 10),
            _api("VERDICT: SUPPORTED\nREASON: Appropriate refusal.", 140, 15),
        ]
        result = manager.run("What are next quarter's revenue targets?")
        self.assertIn("cannot find", result.lower())

    @patch("agents.manager.log")
    @patch("agents.qualitative.chromadb.PersistentClient")
    @patch("agents.qualitative.embedding_model.encode")
    @patch("agents.manager.client")
    @patch("agents.qualitative.client")
    def test_conversation_history_flows_through(
        self, qual_client, mgr_client, mock_encode, mock_chroma, mock_log
    ):
        _patch_encode(mock_encode)
        _patch_chroma(mock_chroma)
        mgr_client.chat.completions.create.return_value = _api("qualitative", 80, 2)
        qual_client.chat.completions.create.side_effect = [
            _api("Follow-up answer. [Source 2: company-policy.txt]", 450, 60),
            _api("VERDICT: SUPPORTED\nREASON: Supported.", 150, 20),
        ]
        history = [
            {"role": "user", "content": "What is the leave policy?"},
            {"role": "assistant", "content": "Employees get 20 days. [Source 1: company-policy.txt]"},
        ]
        result = manager.run("Does that apply to part-time staff?", history=history)
        self.assertIn("Follow-up answer", result)

        messages = qual_client.chat.completions.create.call_args_list[0][1]["messages"]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["content"], "What is the leave policy?")


class TestQuantitativeWorkflow(unittest.TestCase):
    """Classifier → quantitative agent → SQL validator → SQLite → interpreter."""

    @patch("agents.manager.log")
    @patch("agents.quantitative.sqlite3.connect")
    @patch("agents.manager.client")
    @patch("agents.quantitative.client")
    def test_select_query_returns_interpretation(
        self, quant_client, mgr_client, mock_connect, mock_log
    ):
        mgr_client.chat.completions.create.return_value = _api("quantitative", 80, 2)
        quant_client.chat.completions.create.side_effect = [
            _api("SELECT region, SUM(revenue) FROM sales GROUP BY region", 150, 20),
            _api("North region leads with $1M.", 200, 45),
        ]
        _patch_sqlite(mock_connect, [("North", 1000000), ("South", 800000)], ["region", "SUM(revenue)"])
        result = manager.run("Show revenue by region")
        self.assertIn("North", result)

    @patch("agents.manager.log")
    @patch("agents.manager.client")
    @patch("agents.quantitative.client")
    def test_blocked_sql_surfaces_in_result(self, quant_client, mgr_client, mock_log):
        mgr_client.chat.completions.create.return_value = _api("quantitative", 80, 2)
        quant_client.chat.completions.create.return_value = _api("DELETE FROM sales WHERE id=1", 150, 15)
        result = manager.run("Remove old sales data")
        self.assertIn("blocked", result.lower())

    @patch("agents.manager.log")
    @patch("agents.quantitative.sqlite3.connect")
    @patch("agents.manager.client")
    @patch("agents.quantitative.client")
    def test_sql_error_surfaces_in_result(self, quant_client, mgr_client, mock_connect, mock_log):
        mgr_client.chat.completions.create.return_value = _api("quantitative", 80, 2)
        quant_client.chat.completions.create.return_value = _api("SELECT * FROM ghost_table", 150, 10)
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("no such table: ghost_table")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        result = manager.run("Query a missing table")
        self.assertIn("failed", result.lower())


class TestBothWorkflow(unittest.TestCase):
    """Classifier routes 'both': qualitative + quantitative agents run and results are combined."""

    @patch("agents.manager.log")
    @patch("agents.quantitative.sqlite3.connect")
    @patch("agents.qualitative.chromadb.PersistentClient")
    @patch("agents.qualitative.embedding_model.encode")
    @patch("agents.manager.client")
    @patch("agents.qualitative.client")
    @patch("agents.quantitative.client")
    def test_both_results_concatenated(
        self, quant_client, qual_client, mgr_client, mock_encode, mock_chroma, mock_connect, mock_log
    ):
        _patch_encode(mock_encode)
        _patch_chroma(mock_chroma)
        mgr_client.chat.completions.create.return_value = _api("both", 80, 2)
        qual_client.chat.completions.create.side_effect = [
            _api("Policy requires two approvals. [Source 1: company-policy.txt]", 350, 80),
            _api("VERDICT: SUPPORTED\nREASON: Backed by context.", 150, 20),
        ]
        quant_client.chat.completions.create.side_effect = [
            _api("SELECT AVG(satisfaction_score) FROM employees", 150, 15),
            _api("Average satisfaction score is 5.1.", 200, 40),
        ]
        _patch_sqlite(mock_connect, [(5.1,)], ["AVG(satisfaction_score)"])

        result = manager.run("What are our HR policies and employee satisfaction scores?")
        self.assertIn("two approvals", result)
        self.assertIn("5.1", result)

    @patch("agents.manager.log")
    @patch("agents.quantitative.sqlite3.connect")
    @patch("agents.qualitative.chromadb.PersistentClient")
    @patch("agents.qualitative.embedding_model.encode")
    @patch("agents.manager.client")
    @patch("agents.qualitative.client")
    @patch("agents.quantitative.client")
    def test_both_calls_each_agent_exactly_once(
        self, quant_client, qual_client, mgr_client, mock_encode, mock_chroma, mock_connect, mock_log
    ):
        _patch_encode(mock_encode)
        _patch_chroma(mock_chroma)
        mgr_client.chat.completions.create.return_value = _api("both", 80, 2)
        qual_client.chat.completions.create.side_effect = [
            _api("Policy info. [Source 1: company-policy.txt]", 350, 80),
            _api("VERDICT: SUPPORTED\nREASON: OK.", 150, 20),
        ]
        quant_client.chat.completions.create.side_effect = [
            _api("SELECT COUNT(*) FROM customers", 150, 10),
            _api("There are 100 customers.", 200, 30),
        ]
        _patch_sqlite(mock_connect, [(100,)], ["COUNT(*)"])

        manager.run("Policy and customer count")
        # qualitative: 1 answer call + 1 review call = 2; quantitative: 2 calls (SQL gen + interpret)
        self.assertEqual(qual_client.chat.completions.create.call_count, 2)
        self.assertEqual(quant_client.chat.completions.create.call_count, 2)
