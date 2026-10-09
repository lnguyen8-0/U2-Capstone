from dotenv import load_dotenv
import os
import sqlite3
from openai import OpenAI
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
SCHEMA_CONTEXT = """
Available tables:
- sales(id, region, product, revenue, date, units_sold)
- customers(id, name, industry, churn_date, satisfaction_score)
- employees(id, department, satisfaction_score, tenure_years)
"""

def strip_markdown_fences(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.splitlines()
        # remove opening fence (```sql or ```)
        lines = lines[1:]
        # remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql = "\n".join(lines).strip()
    return sql

def validate_sql(query: str) -> dict:
    blocked = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    for word in blocked:
        if word in query.upper():
            return {"valid": False, "reason": f"Blocked keyword: {word}"}
    if not query.strip().upper().startswith("SELECT"):
        return {"valid": False, "reason": "Only SELECT queries are permitted"}
    return {"valid": True, "reason": "OK"}

def generate_sql(query: str) -> dict:
    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        max_tokens=256,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{SCHEMA_CONTEXT}\n\n"
                    f"Generate a SQLite-compatible SQL query for: {query}\n\n"
                    "IMPORTANT: This is SQLite. Use strftime('%Y-%m', date) instead of DATE_TRUNC. "
                    "Use strftime('%Y', date) for year grouping. Do NOT use DATE_TRUNC, EXTRACT, or other PostgreSQL-only functions.\n\n"
                    "Return ONLY the SQL query, nothing else."
                ),
            }
        ],
    )

    usage = response.usage

    return {
        "sql": response.choices[0].message.content.strip(),
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }

def run(query: str) -> dict:
    sql_result = generate_sql(query)
    sql = strip_markdown_fences(sql_result["sql"])
    validation = validate_sql(sql)

    if not validation["valid"]:
        return {
            "answer": f"Query blocked: {validation['reason']}",
            "sql": sql,
            "rows": [],
            "validation": "FAILED",
            "input_tokens": sql_result["input_tokens"],
            "output_tokens": sql_result["output_tokens"]
        }

    try:
        conn = sqlite3.connect("./data/documents/database.sqlite")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        conn.close()

        interpretation = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"The user asked: {query}\n\nSQL query used: {sql}\n\nResults:\nColumns: {cols}\nData: {rows[:20]}\n\nProvide a clear, concise interpretation of these results."
            }]
        )

        interp_usage = interpretation.usage
        return {
            "answer": interpretation.choices[0].message.content,
            "sql": sql,
            "columns": cols,
            "rows": rows,
            "validation": "PASSED",
            "input_tokens": sql_result["input_tokens"] + (getattr(interp_usage, "prompt_tokens", 0) or 0),
            "output_tokens": sql_result["output_tokens"] + (getattr(interp_usage, "completion_tokens", 0) or 0)
        }

    except Exception as e:
        return {
            "answer": f"Query execution failed: {str(e)}",
            "sql": sql,
            "rows": [],
            "validation": "ERROR",
            "input_tokens": sql_result["input_tokens"],
            "output_tokens": sql_result["output_tokens"]
        }
