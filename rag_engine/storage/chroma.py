from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from .. import config


class LocalVectorStoreManager:
    def __init__(self) -> None:
        config.validate_environment()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME
        )
        self.db_path = str(config.DB_DIR)

    def get_vector_store(self) -> Chroma:
        """Establish a persistent disk connection to Chroma."""
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
        )
