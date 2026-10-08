from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ..storage.chroma import LocalVectorStoreManager


class LocalQueryEngine:
    def __init__(self) -> None:
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
        """Look up the most similar documents in the vector store."""
        if not query_text.strip():
            return []
        return self.db.similarity_search(query_text, k=top_k)

    def generate_answer(self, query_text: str) -> str:
        """Generate a grounded response using retrieved context."""
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
