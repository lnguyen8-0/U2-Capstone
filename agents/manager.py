# agents/manager.py

from dotenv import load_dotenv
import os
from openai import OpenAI
from agents import qualitative, quantitative
from validation.validator import validate_qualitative, validate_quantitative
from tokenomics.logger import log
load_dotenv()



client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

def classify(query: str) -> str:
    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": f"""Classify this query as exactly one of: qualitative, quantitative, both.

qualitative = questions about policies, processes, procedures, explanations, documentation
quantitative = questions about numbers, metrics, trends, comparisons, SQL-queryable data
both = questions that need both document search and data analysis

Query: {query}

Reply with one word only: qualitative, quantitative, or both."""
        }]
    )

    route = response.choices[0].message.content.strip().lower()

    usage = response.usage
    log(
        query,
        "manager-classifier",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )

    return route if route in ["qualitative", "quantitative", "both"] else "qualitative"

def run(query: str):
    print(f"\nQuery: {query}")
    route = classify(query)
    print(f"Route: {route}")

    qual_result = None
    quant_result = None

    if route in ["qualitative", "both"]:
        qual_result = qualitative.run(query)
        validation = validate_qualitative(qual_result["answer"], qual_result["chunks"])
        log(query, "qualitative", qual_result["input_tokens"], qual_result["output_tokens"])
        if validation["flag"]:
            print(f"\n⚠️  VALIDATION WARNING: {validation['warning']}")
        print(f"\n[Qualitative]\n{qual_result['answer']}")

    if route in ["quantitative", "both"]:
        quant_result = quantitative.run(query)
        validation = validate_quantitative(
            quant_result["sql"],
            quant_result["validation"]
        )
        log(query, "quantitative", quant_result["input_tokens"], quant_result["output_tokens"])
        if validation["flag"]:
            print(f"\n⚠️  VALIDATION WARNING: {validation['warning']}")
        print(f"\n[Quantitative]\n{quant_result['answer']}")
        print(f"SQL used: {quant_result['sql']}")