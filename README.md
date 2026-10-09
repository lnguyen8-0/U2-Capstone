# U2-Capstone

A multi-agent system that answers questions about Northstar Solutions' policies (RAG over ChromaDB) and business data (NL to SQL over SQLite), with validation and token cost logging on every response. Uses Gemini 3.5 Flash-Lite via the OpenAI-compatible endpoint.

## Setup
- `python -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `echo "GEMINI_API_KEY=your-key" > .env`
- `python ingest.py`
- `python main.py`

`data/documents/database.sqlite` is included. `seed_db.py` rebuilds it with fixed seed data if needed.

## Architecture
- `agents/manager.py` classifies each question as qualitative, quantitative, or both, and routes it. Falls back to qualitative for anything unrecognized.
- `agents/qualitative.py` retrieves the top 5 chunks from ChromaDB using `all-MiniLM-L6-v2` embeddings and answers with source citations from `company-policy.txt`.
- `agents/quantitative.py` generates SQL, blocks anything that is not a SELECT, runs it against the SQLite database, and summarizes the results.
- `validation/validator.py` checks that qualitative answers cite at least one `[Source N]` reference, and checks that SQL passed, failed, or errored.
- `tokenomics/logger.py` logs input tokens, output tokens, and cost for every model call to `tokenomics_log.jsonl`.

## Model Selection
- Started on `gemini-3.5-flash-lite` and kept it throughout. Both `gemini-3.6-flash` and `gemini-3.8-flash` returned `503` high-demand errors during testing, so all calls stayed on 3.5 Flash-Lite, which passed every test query without error.
- The tokenomics logger uses a flat rate of $0.00075 per 1,000 tokens for both input and output, which matches the Gemini 3.5 Flash-Lite rate at the time of testing.

## Trust-but-Verify
- **"Show me monthly revenue trends"** The quantitative agent initially generated `DATE_TRUNC('month', date)`, which is PostgreSQL syntax. SQLite does not support that function, so execution threw an exception and the validation layer flagged `SQL validation status: ERROR`. I accepted the flag and updated the schema prompt in `quantitative.py` to explicitly say "Use `strftime('%Y-%m', date)` instead of `DATE_TRUNC`" and list the PostgreSQL functions that are not supported. After that fix the query passed validation and returned the correct monthly totals (January 2023: $777,717.15 through December 2023: $1,102,294.12).
- **"What is our customer churn rate?"** The quantitative agent generated `SELECT COUNT(*) AS total, COUNT(churn_date) AS churned FROM customers`, which passed validation. The answer came back as 24 out of 100 customers churned (24%). The validation layer returned no flag. I verified this by running the query directly in SQLite and got the same numbers, so I accepted the output unchanged.
- **"Explain the code review process"** The qualitative agent returned a ~430-token answer citing `[Source 1: company-policy.txt]` and `[Source 3: company-policy.txt]` covering the review and approval steps from the engineering section of the policy document. The validation layer found the citations and returned `is_grounded=True` with no flag. I accepted the answer after confirming the cited source filename matched the document that was ingested.
- **Output I did not immediately trust.** When I asked "How does our employee satisfaction compare to industry standards and what policies might impact this?", the qualitative agent returned only 10 output tokens: "I cannot find this information in the provided documents." The validation layer did not flag this — `refused_to_answer` was set to `True`, so `flag` stayed `False`, which is the correct behaviour. The quantitative agent ran in parallel and returned the internal satisfaction averages (Sales: 5.87, HR: 5.36, Engineering: 5.31, Support: 5.26, Marketing: 4.53, overall: 5.10/10). I did not immediately trust this as a complete answer because the question asked about industry benchmarks, which the system has no data for. I accepted the quantitative output as accurate for internal scores — I spot-checked the SQL used `AVG(satisfaction_score) GROUP BY department` against the database and the numbers matched — but I documented the gap: the system cannot answer the industry comparison half of the question without ingesting external benchmark data.

## Tokenomics
- Prices: $0.00075 per 1,000 tokens for both input and output (flat rate, Gemini 3.5 Flash-Lite).
- Average cost per qualitative query: about $0.0025, or about $2.50 per 1,000 queries.
- Average cost per quantitative query: about $0.0002–$0.0006, or about $0.20–$0.60 per 1,000 queries.
- Qualitative queries cost roughly 5–10× more than quantitative ones because the retrieved chunks add the bulk of input tokens per call.
- The most expensive query type is "both" routing, which runs the full RAG pipeline and the SQL pipeline in the same call.

### Tokenomics log analysis (`tokenomics_log.jsonl`)

Analysis over 51 logged calls (26 classifier, 12 qualitative, 13 quantitative):

| Agent | Calls | Avg input tokens | Avg output tokens | Total cost |
|---|---|---|---|---|
| manager-classifier | 26 | 86 | 1 | $0.0017 |
| qualitative | 12 | 3,060 | 143 | $0.0288 |
| quantitative | 13 | 239 | 233 | $0.0046 |

The qualitative agent consumed 88 % of total token cost despite handling fewer than a quarter of calls. Profiling the prompt showed that the five retrieved chunks contributed approximately 2,700 of the 3,060 average input tokens — about 88 % of the input budget — while the answer quality was typically determined by the top one or two matches.

**Concrete change made:** Reduced `top_k` from 5 to 3 in `qualitative.py:retrieve()`.

Back-of-envelope impact at the observed average of 540 tokens per chunk:
- Before: 3,060 input tokens/call → $0.00230/call
- After: ~1,980 input tokens/call → ~$0.00149/call (estimated ~35 % reduction)
- Projected saving at 1,000 qualitative queries/day: ~$0.81/day, ~$296/year

Output quality was not degraded: every test query that previously returned a grounded, cited answer continued to do so after the change, because the top-3 chunks consistently contained the relevant policy section. The two dropped chunks were lower-similarity matches that did not appear in any citation.

## Limitations
- `validate_qualitative` checks that a source was cited, not that the answer is correct or complete (see Trust-but-Verify).
- `validate_sql` uses substring matching, so a column name like `last_updated` would be falsely blocked because it contains `UPDATE`.
- The database connection in `quantitative.py` is not opened read-only, so `validate_sql` is the only barrier to write operations.
- Greetings and off-topic inputs are routed to the qualitative pipeline by the fallback in `manager.py`, consuming a full ChromaDB retrieval and LLM call before returning a refusal. The `hiiiii` test query cost $0.0023 this way.
- "Both" queries return two separate answers rather than one synthesised response.
