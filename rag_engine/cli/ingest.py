import logging

from ._common import load_environment


def main() -> None:
    load_environment()

    from ..ingestion import LocalIngestWorker

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Initializing local ingestion worker")
    total_chunks = LocalIngestWorker().process_directory()
    if total_chunks:
        logger.info("Ingestion complete. Indexed %s chunks.", total_chunks)
    else:
        logger.warning(
            "No PDF documents were processed. Add .pdf files to the data directory."
        )
