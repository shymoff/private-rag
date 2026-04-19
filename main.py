import sys

from src.graph import app


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Pytanie: ").strip()

    result = app.invoke({
        "question": question,
        "documents": [],
        "generation": "",
        "is_relevant": False,
    })

    print("\n--- ODPOWIEDŹ ---")
    if result.get("generation"):
        print(result["generation"])
    else:
        print("(brak odpowiedzi — dokumenty uznane za nieistotne)")


if __name__ == "__main__":
    main()
