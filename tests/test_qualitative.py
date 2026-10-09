import unittest
from unittest.mock import patch, MagicMock
from agents.qualitative import build_prompt, run, review


SAMPLE_CHUNKS = [
    {"content": "Employees must submit code reviews via GitHub.", "source": "company-policy.txt", "chunk": 0},
    {"content": "All PRs require at least two approvals.", "source": "company-policy.txt", "chunk": 1},
]


def _mock_response(content, prompt_tokens=200, completion_tokens=80):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestBuildPrompt(unittest.TestCase):
    def test_includes_query(self):
        prompt = build_prompt("What is the review process?", SAMPLE_CHUNKS)
        self.assertIn("What is the review process?", prompt)

    def test_includes_chunk_content(self):
        prompt = build_prompt("test", SAMPLE_CHUNKS)
        self.assertIn("submit code reviews via GitHub", prompt)
        self.assertIn("two approvals", prompt)

    def test_includes_source_labels(self):
        prompt = build_prompt("test", SAMPLE_CHUNKS)
        self.assertIn("[Source 1: company-policy.txt]", prompt)
        self.assertIn("[Source 2: company-policy.txt]", prompt)

    def test_empty_chunks_still_valid(self):
        prompt = build_prompt("test query", [])
        self.assertIn("test query", prompt)
        self.assertIn("CONTEXT:", prompt)


class TestQualitativeRun(unittest.TestCase):
    @patch("agents.qualitative.retrieve")
    @patch("agents.qualitative.client")
    def test_returns_expected_keys(self, mock_client, mock_retrieve):
        mock_retrieve.return_value = SAMPLE_CHUNKS
        mock_client.chat.completions.create.return_value = _mock_response(
            "Submit via GitHub. [Source 1: company-policy.txt]", 300, 80
        )
        result = run("Explain the code review process")
        for key in ("answer", "chunks", "input_tokens", "output_tokens"):
            self.assertIn(key, result)

    @patch("agents.qualitative.retrieve")
    @patch("agents.qualitative.client")
    def test_answer_and_chunks(self, mock_client, mock_retrieve):
        mock_retrieve.return_value = SAMPLE_CHUNKS
        mock_client.chat.completions.create.return_value = _mock_response(
            "Submit via GitHub. [Source 1: company-policy.txt]", 300, 80
        )
        result = run("Explain the code review process")
        self.assertIn("GitHub", result["answer"])
        self.assertEqual(result["chunks"], SAMPLE_CHUNKS)
        self.assertEqual(result["input_tokens"], 300)
        self.assertEqual(result["output_tokens"], 80)

    @patch("agents.qualitative.retrieve")
    @patch("agents.qualitative.client")
    def test_no_history_sends_single_message(self, mock_client, mock_retrieve):
        mock_retrieve.return_value = SAMPLE_CHUNKS
        mock_client.chat.completions.create.return_value = _mock_response("Answer.", 200, 40)
        run("What is the policy?")
        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")

    @patch("agents.qualitative.retrieve")
    @patch("agents.qualitative.client")
    def test_history_prepended_to_messages(self, mock_client, mock_retrieve):
        mock_retrieve.return_value = SAMPLE_CHUNKS
        mock_client.chat.completions.create.return_value = _mock_response("Follow-up.", 400, 60)
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        run("Follow-up question", history=history)
        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["content"], "Previous question")
        self.assertEqual(messages[1]["content"], "Previous answer")
        self.assertEqual(messages[2]["role"], "user")

    @patch("agents.qualitative.retrieve")
    @patch("agents.qualitative.client")
    def test_history_not_mutated(self, mock_client, mock_retrieve):
        mock_retrieve.return_value = SAMPLE_CHUNKS
        mock_client.chat.completions.create.return_value = _mock_response("Answer.", 200, 40)
        history = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
        original_len = len(history)
        run("Follow-up", history=history)
        self.assertEqual(len(history), original_len)


class TestQualitativeReview(unittest.TestCase):
    @patch("agents.qualitative.client")
    def test_supported_verdict(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response(
            "VERDICT: SUPPORTED\nREASON: Answer is directly backed by context.", 150, 30
        )
        result = review("Two approvals required. [Source 1]", SAMPLE_CHUNKS)
        self.assertTrue(result["supported"])
        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertIn("directly backed", result["reason"])

    @patch("agents.qualitative.client")
    def test_unsupported_verdict(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response(
            "VERDICT: UNSUPPORTED\nREASON: Claim not found in context.", 150, 35
        )
        result = review("Reviews require ten approvals.", SAMPLE_CHUNKS)
        self.assertFalse(result["supported"])
        self.assertEqual(result["verdict"], "UNSUPPORTED")
        self.assertIn("not found", result["reason"])

    @patch("agents.qualitative.client")
    def test_returns_token_counts(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response(
            "VERDICT: SUPPORTED\nREASON: Supported.", 120, 25
        )
        result = review("Some answer.", SAMPLE_CHUNKS)
        self.assertEqual(result["input_tokens"], 120)
        self.assertEqual(result["output_tokens"], 25)

    @patch("agents.qualitative.client")
    def test_malformed_response_gracefully_handled(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response("No structured output here.", 100, 10)
        result = review("Some answer.", SAMPLE_CHUNKS)
        self.assertIn("verdict", result)
        self.assertIn("supported", result)
        self.assertFalse(result["supported"])
