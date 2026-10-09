from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import chromadb
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Replace this with the exact model used during document ingestion.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    chroma = chromadb.PersistentClient(path="./data/chroma")
    collection = chroma.get_collection("enterprise-docs")

    embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
    )

    return [
        {
            "content": document,
            "source": metadata["source"],
            "chunk": metadata["chunk"],
        }
        for document, metadata in zip(
            results["documents"][0],
            results["metadatas"][0],
        )
    ]


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = ""

    for i, chunk in enumerate(chunks):
        context += (
            f"[Source {i + 1}: {chunk['source']}]\n"
            f"{chunk['content']}\n\n"
        )

    return f"""You are a helpful enterprise documentation assistant.

Answer the question using ONLY the context provided below.

If the answer is not in the context, say:
"I cannot find this information in the provided documents."

Always cite the source number(s) you used.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def run(query: str, history: list[dict] | None = None) -> dict:
    chunks = retrieve(query)
    prompt = build_prompt(query, chunks)

    messages: list[dict] = list(history) if history else []
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        max_tokens=1024,
        messages=messages,
    )

    usage = response.usage

    return {
        "answer": response.choices[0].message.content or "",
        "chunks": chunks,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


def review(answer: str, chunks: list[dict]) -> dict:
    """Second-pass LLM reviewer: checks if every claim in the answer is supported by the retrieved context."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1}: {chunk['source']}]\n{chunk['content']}\n\n"

    prompt = f"""You are a strict fact-checker for an enterprise Q&A system.

Given the retrieved context and the generated answer below, determine whether every claim in the answer is directly supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond in this exact format:
VERDICT: SUPPORTED or UNSUPPORTED
REASON: one sentence explaining your verdict"""

    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or ""
    verdict = "UNKNOWN"
    reason = ""
    for line in content.strip().splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    usage = response.usage
    return {
        "verdict": verdict,
        "reason": reason,
        "supported": verdict == "SUPPORTED",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }
