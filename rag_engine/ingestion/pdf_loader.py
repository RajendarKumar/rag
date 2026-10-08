import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config
from ..storage.chroma import LocalVectorStoreManager

logger = logging.getLogger(__name__)


class LocalIngestWorker:
    def __init__(self) -> None:
        self.store_manager = LocalVectorStoreManager()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.PARENT_CHUNK_SIZE,
            chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        )

    def process_directory(self) -> int:
        """Load PDF files from the data directory and index their text chunks."""
        pdf_files = sorted(config.DATA_DIR.glob(config.FILE_EXTENSION_GLOB))
        if not pdf_files:
            return 0

        all_chunks = []
        for file_path in pdf_files:
            try:
                logger.info("Reading PDF: %s", file_path.name)
                pages = PyPDFLoader(str(file_path)).load()
                all_chunks.extend(self.splitter.split_documents(pages))
            except Exception:
                logger.exception("Failed to process PDF %s", file_path)

        if all_chunks:
            self.store_manager.get_vector_store().add_documents(all_chunks)

        return len(all_chunks)
