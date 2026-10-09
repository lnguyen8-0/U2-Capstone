import unittest
from unittest.mock import patch, MagicMock
from agents.quantitative import strip_markdown_fences, validate_sql, generate_sql, run


def _mock_response(content, prompt_tokens=100, completion_tokens=50):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestStripMarkdownFences(unittest.TestCase):
    def test_strips_sql_fence(self):
        self.assertEqual(strip_markdown_fences("```sql\nSELECT 1\n```"), "SELECT 1")

    def test_strips_plain_fence(self):
        self.assertEqual(strip_markdown_fences("```\nSELECT 1\n```"), "SELECT 1")

    def test_no_fence_unchanged(self):
        self.assertEqual(strip_markdown_fences("SELECT * FROM sales"), "SELECT * FROM sales")

    def test_preserves_multiline(self):
        sql = "```sql\nSELECT region,\n  SUM(revenue)\nFROM sales\n```"
        result = strip_markdown_fences(sql)
        self.assertIn("SUM(revenue)", result)
        self.assertNotIn("```", result)

    def test_strips_leading_trailing_whitespace(self):
        result = strip_markdown_fences("  SELECT 1  ")
        self.assertEqual(result, "SELECT 1")


class TestValidateSQL(unittest.TestCase):
    def test_select_passes(self):
        result = validate_sql("SELECT * FROM sales")
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "OK")

    def test_drop_blocked(self):
        result = validate_sql("DROP TABLE sales")
        self.assertFalse(result["valid"])
        self.assertIn("DROP", result["reason"])

    def test_delete_blocked(self):
        result = validate_sql("DELETE FROM sales WHERE id=1")
        self.assertFalse(result["valid"])

    def test_update_blocked(self):
        result = validate_sql("UPDATE sales SET revenue=0")
        self.assertFalse(result["valid"])

    def test_insert_blocked(self):
        result = validate_sql("INSERT INTO sales VALUES (1)")
        self.assertFalse(result["valid"])

    def test_alter_blocked(self):
        result = validate_sql("ALTER TABLE sales ADD COLUMN x INT")
        self.assertFalse(result["valid"])

    def test_truncate_blocked(self):
        result = validate_sql("TRUNCATE TABLE sales")
        self.assertFalse(result["valid"])

    def test_non_select_start_blocked(self):
        result = validate_sql("EXPLAIN SELECT * FROM sales")
        self.assertFalse(result["valid"])

    def test_case_insensitive_blocking(self):
        self.assertFalse(validate_sql("drop table sales")["valid"])
        self.assertFalse(validate_sql("Delete FROM sales")["valid"])

    def test_select_with_where(self):
        result = validate_sql("SELECT id, revenue FROM sales WHERE region='North'")
        self.assertTrue(result["valid"])

    def test_select_with_aggregate(self):
        result = validate_sql("SELECT region, SUM(revenue) FROM sales GROUP BY region")
        self.assertTrue(result["valid"])


class TestGenerateSQL(unittest.TestCase):
    @patch("agents.quantitative.client")
    def test_returns_sql_and_token_counts(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response(
            "SELECT * FROM sales", 120, 10
        )
        result = generate_sql("Show all sales")
        self.assertEqual(result["sql"], "SELECT * FROM sales")
        self.assertEqual(result["input_tokens"], 120)
        self.assertEqual(result["output_tokens"], 10)

    @patch("agents.quantitative.client")
    def test_handles_zero_usage(self, mock_client):
        resp = _mock_response("SELECT 1")
        resp.usage = None
        mock_client.chat.completions.create.return_value = resp
        result = generate_sql("test")
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)


class TestQuantitativeRun(unittest.TestCase):
    @patch("agents.quantitative.client")
    @patch("agents.quantitative.sqlite3.connect")
    def test_successful_query(self, mock_connect, mock_client):
        mock_client.chat.completions.create.side_effect = [
            _mock_response("SELECT region, SUM(revenue) FROM sales GROUP BY region", 100, 20),
            _mock_response("North leads with $1M.", 200, 50),
        ]
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("North", 1000000), ("South", 800000)]
        mock_cursor.description = [("region",), ("SUM(revenue)",)]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = run("Revenue by region")
        self.assertEqual(result["validation"], "PASSED")
        self.assertIn("North", result["answer"])
        self.assertEqual(result["columns"], ["region", "SUM(revenue)"])
        self.assertEqual(result["rows"], [("North", 1000000), ("South", 800000)])

    @patch("agents.quantitative.client")
    def test_blocked_sql_returns_failed(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response("DROP TABLE sales", 100, 10)
        result = run("Delete all records")
        self.assertEqual(result["validation"], "FAILED")
        self.assertIn("blocked", result["answer"].lower())
        self.assertEqual(result["rows"], [])

    @patch("agents.quantitative.client")
    @patch("agents.quantitative.sqlite3.connect")
    def test_execution_error_returns_error(self, mock_connect, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response(
            "SELECT * FROM nonexistent", 100, 10
        )
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("no such table: nonexistent")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = run("Query nonexistent table")
        self.assertEqual(result["validation"], "ERROR")
        self.assertIn("failed", result["answer"].lower())

    @patch("agents.quantitative.client")
    @patch("agents.quantitative.sqlite3.connect")
    def test_token_counts_summed_across_both_calls(self, mock_connect, mock_client):
        mock_client.chat.completions.create.side_effect = [
            _mock_response("SELECT COUNT(*) FROM customers", 100, 10),
            _mock_response("There are 100 customers.", 200, 30),
        ]
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(100,)]
        mock_cursor.description = [("COUNT(*)",)]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = run("How many customers?")
        self.assertEqual(result["input_tokens"], 300)
        self.assertEqual(result["output_tokens"], 40)
