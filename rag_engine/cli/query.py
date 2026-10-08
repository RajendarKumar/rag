import logging
import sys

from ._common import load_environment


def main() -> None:
    load_environment()

    if len(sys.argv) < 2:
        print("Error: Please provide a query string.")
        print('Example: rag-query "technical skill"')
        raise SystemExit(1)

    query_text = sys.argv[1]
    print(f"Sending semantic query to Chroma: {query_text!r}\n")

    from ..retrieval import LocalQueryEngine

    engine = LocalQueryEngine()
    matched_docs = engine.execute_similarity_search(query_text, top_k=3)
    if not matched_docs:
        print("No matching text blocks were returned from the vector database.")
        return

    print(f"Found {len(matched_docs)} relevant context blocks:\n")
    for idx, doc in enumerate(matched_docs, start=1):
        filter_prompt = f"""You are a precise data verification filter.
Analyze the following text chunk retrieved for the user query.

Text Chunk:
---
{doc.page_content.strip()}
---

Note: Do not predict from your side, answer only with YES or NO on the basis of provided text chunk.

User query:
------
{query_text}
------
Respond with exactly one word: YES or NO."""
        print(f"CHUNK MATCH #{idx} (Source: {doc.metadata.get('source', 'Unknown')})")
        print("-" * 60)
        print(doc.page_content.strip())
        response = engine.llm.invoke(filter_prompt)
        verdict = response.content.strip().upper()
        print(f"\nVerification verdict: {verdict}")
        print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
