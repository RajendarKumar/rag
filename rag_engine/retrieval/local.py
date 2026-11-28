"""Local retrieval and grounded answer generation using Chroma.

This module provides :class:`LocalQueryEngine`, which connects to the
project's persistent Chroma vector store, performs semantic similarity
searches, and uses Gemini to answer candidate-related questions from the
retrieved context.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ..storage.chroma import LocalVectorStoreManager


class LocalQueryEngine:
    """Query the local resume store and generate context-grounded answers.

    The engine initializes a persistent Chroma vector store, configures the
    Google Generative AI embedding and language model dependencies through
    :class:`~rag_engine.storage.chroma.LocalVectorStoreManager`, and applies a
    recruiter-oriented system prompt for answer generation.

    Attributes:
        store_manager (LocalVectorStoreManager): Manager used to validate the
            environment and configure the persistent vector store.
        db (Chroma): Persistent Chroma collection used for similarity searches.
        llm (ChatGoogleGenerativeAI): Gemini language model used to generate
            responses from retrieved resume context.
        prompt_template (ChatPromptTemplate): Prompt template containing the
            system instructions and human question placeholders.

    Note:
        Instantiation requires the environment variables validated by
        :class:`~rag_engine.storage.chroma.LocalVectorStoreManager`.
    """

    def __init__(self) -> None:
        """Initialize the local vector store and language model.

        The vector store is opened from the configured persistent directory.
        The configured Gemini model uses a temperature of ``0.0`` so generated
        answers are deterministic and conservative.

        The prompt instructs the model to use only supplied context and to
        return ``"Information not found in the resume."`` when the context does
        not contain a direct answer.
        """
        self.store_manager = LocalVectorStoreManager()
        self.db = self.store_manager.get_vector_store()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an expert, professional technical recruiter assistant. "
                        "Your objective is to answer questions about the candidate's "
                        "background using exclusively the provided text context sections "
                        "below.\n\n"
                        "CRITICAL RULES:\n"
                        "1. Base your answer strictly on the provided Context blocks.\n"
                        "2. If the context does not contain the answer, state explicitly: "
                        "'Information not found in the resume.'\n"
                        "3. Do not assume, extrapolate, or invent facts outside the text.\n\n"
                        "--- CONTEXT START ---\n"
                        "{context}\n"
                        "--- CONTEXT END ---"
                    ),
                ),
                ("human", "{question}"),
            ]
        )

    def execute_similarity_search(
        self, query_text: str, top_k: int = 3
    ) -> list[Document]:
        """Retrieve documents that are semantically similar to a query.

        Args:
            query_text: Natural-language text used to query the vector store.
                Whitespace-only input is treated as an empty query.
            top_k: Maximum number of documents to return. This value is passed
                directly to Chroma's ``similarity_search`` method.

        Returns:
            A list of the most similar LangChain
            :class:`~langchain_core.documents.Document` objects. An empty list
            is returned for whitespace-only input or when no matches exist.

        Raises:
            ValueError: If ``top_k`` is not accepted by the underlying Chroma
                similarity search implementation.
        """
        if not query_text.strip():
            return []
        return self.db.similarity_search(query_text, k=top_k)

    def generate_answer(self, query_text: str) -> str:
        """Generate an answer from the three most relevant local documents.

        Args:
            query_text: Candidate-related question to answer. The query must
                contain at least one non-whitespace character.

        Returns:
            A Gemini-generated answer grounded in the retrieved context. Returns
            a validation message for an empty query or a context-unavailable
            message when similarity search finds no documents.

        Raises:
            Exception: Propagates an exception raised by the vector store or
                Gemini language model.

        Note:
            The three matched documents are joined with blank lines to form the
            prompt context. Document metadata is not included in the prompt;
            only each document's ``page_content`` is used.
        """
        if not query_text.strip():
            return "Please provide a valid query."

        matched_docs = self.execute_similarity_search(query_text, top_k=3)
        if not matched_docs:
            return (
                "No matching context blocks found in the database. "
                "Unable to generate an answer."
            )

        compiled_context = "\n\n".join(doc.page_content for doc in matched_docs)
        formatted_messages = self.prompt_template.format_messages(
            context=compiled_context,
            question=query_text,
        )
        response = self.llm.invoke(formatted_messages)
        return response.content
