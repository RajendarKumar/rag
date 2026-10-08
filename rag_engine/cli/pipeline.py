from langchain_core.documents import Document

from ._common import load_environment


def main() -> None:
    load_environment()

    from ..embeddings import HybridEmbeddingEngine
    from ..retrieval import HybridSearchOrchestrator
    from ..storage import MilvusStorageProvider

    documents = [
        Document(
            page_content=(
                "Maintained architectural standards and high trust environments "
                "across team boundaries."
            ),
            metadata={"source": "perf_review.txt"},
        ),
        Document(
            page_content=(
                "Designed high-performance microservices pipelines using the Rust "
                "systems programming language."
            ),
            metadata={"source": "resume.txt"},
        ),
    ]

    embedding_engine = HybridEmbeddingEngine(dense_model_name="text-embedding-004")
    storage_provider = MilvusStorageProvider(host="localhost", port="19530")
    collection_name = "enterprise_resume_rag"

    corpus_texts = [doc.page_content for doc in documents]
    embedding_engine.fit_sparse_matrix(corpus_texts)
    sample_dense, _ = embedding_engine.generate_vectors(corpus_texts[0])
    storage_provider.initialize_hybrid_collection(
        collection_name, dense_dim=len(sample_dense)
    )

    write_payload = []
    for doc in documents:
        dense_vector, sparse_vector = embedding_engine.generate_vectors(doc.page_content)
        write_payload.append(
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source"),
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
        )

    inserted = storage_provider.batch_insert(collection_name, write_payload)
    print(f"Successfully ingested {inserted} records into Milvus.")

    orchestrator = HybridSearchOrchestrator(embedding_engine, storage_provider)
    search_results = orchestrator.execute_hybrid_query(
        collection_name=collection_name,
        query="Rust",
        top_k=2,
    )
    print("\nFinal hybrid search results:")
    for hit in search_results:
        print(f"Score: {hit['score']:.4f} | Source: {hit['source']}")
        print(f"Content: {hit['text']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
