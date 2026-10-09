# main.py
from agents.manager import run

def main():
    print("Spoonful Enterprise RAG System")
    print("Type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue
        answer = run(query, history)
        history.append({"role": "user", "content": query})
        if answer:
            history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()