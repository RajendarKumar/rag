import os
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from pymilvus.model.sparse import BM25EmbeddingFunction


class HybridEmbeddingEngine:
    """
    Manages generation of dual-vector semantics (Dense + Sparse).
    """

    def __init__(self, dense_model_name: str = "embedding-001") -> None:
        # Initialize Google GenAI dense embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Production Error: GOOGLE_API_KEY environment variable is missing.")
        # Try the requested model first, then fallbacks. If a model is not available
        # the Google API returns NOT_FOUND; try other known model names.
        tried = []
        model_candidates = [
            os.getenv("GOOGLE_EMBEDDING_MODEL"),
            dense_model_name,
            # Google Gemini embedding models commonly available
            "gemini-embedding-001",
            "gemini-embedding-2",
            "gemini-embedding-2-preview",
            # Older/common fallbacks
            "text-embedding-3-large",
            "text-embedding-3-small",
            "embedding-001",
        ]
        # dedupe while preserving order
        seen: set[str] = set()
        model_candidates = [m for m in model_candidates if m and (m not in seen and not seen.add(m))]

        last_exc: Exception | None = None
        for model_name in model_candidates:
            tried.append(model_name)
            try:
                client = GoogleGenerativeAIEmbeddings(model=model_name, api_key=api_key)
                # probe the model with a tiny request to validate availability
                client.embed_query("test")
                self.dense_client = client
                break
            except GoogleGenerativeAIError as exc:
                last_exc = exc
            except Exception as exc:
                last_exc = exc

        if not hasattr(self, "dense_client"):
            raise RuntimeError(f"Failed to initialize Google embedding client. Tried: {tried}. Last error: {last_exc}")
        # Built-in sparse encoder for BM25; initialize directly with default analyzer
        self.sparse_client = BM25EmbeddingFunction()

    def fit_sparse_matrix(self, corpus: list[str]) -> None:
        """Fits the BM25 model on document frequencies before matrix calculation."""
        if not corpus:
            raise ValueError("Cannot fit an empty document corpus.")
        self.sparse_client.fit(corpus)

    def generate_vectors(self, text: str) -> tuple[list[float], Any]:
        """Transforms a single text block into both dense and sparse representations."""
        dense_vector = self.dense_client.embed_query(text)
        # Milvus expects sparse vectors as either a 1xN SciPy sparse row or a dict/list-of-pairs.
        raw_sparse = self.sparse_client.encode_documents([text])
        try:
            import scipy
            is_sparse = scipy.sparse.issparse(raw_sparse)
        except Exception:
            is_sparse = False

        if is_sparse:
            # ensure we return a 2D single-row sparse matrix (shape (1, N))
            if getattr(raw_sparse, "shape", (0, ))[0] == 1:
                sparse_vector = raw_sparse[0:1]
            else:
                sparse_vector = raw_sparse
        else:
            # list-like return (legacy): take first element
            sparse_vector = raw_sparse[0]
        return dense_vector, sparse_vector

    def generate_query_vectors(self, query: str) -> tuple[list[float], Any]:
        """Transforms an isolated user query into search targets."""
        dense_vector = self.dense_client.embed_query(query)
        raw_sparse = self.sparse_client.encode_queries([query])
        try:
            import scipy
            is_sparse = scipy.sparse.issparse(raw_sparse)
        except Exception:
            is_sparse = False

        if is_sparse:
            if getattr(raw_sparse, "shape", (0, ))[0] == 1:
                sparse_vector = raw_sparse[0:1]
            else:
                sparse_vector = raw_sparse
        else:
            sparse_vector = raw_sparse[0]
        return dense_vector, sparse_vector