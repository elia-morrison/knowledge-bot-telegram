import asyncio
import logging
import uuid

import pymupdf
from knowledge_bot_telegram.schemas import Document
from knowledge_bot_telegram.vector_search import QdrantEngine
from knowledge_bot_telegram.embedder import Embedder
from dotenv import load_dotenv

load_dotenv()

embedder = Embedder()
qdrant_engine = QdrantEngine()

logger = logging.getLogger(__name__)


def pdf_to_text(path: str) -> str:
    with pymupdf.open(path) as doc:
        return chr(12).join([page.get_text() for page in doc])


async def main():
    logging.basicConfig(level=logging.INFO)
    if await qdrant_engine.check_collection_initialized():
        logger.info("Collection already initialized. Skipping initialization.")
        return

    logger.info("Initializing collection.")
    await qdrant_engine.initialize_collection(vector_size=embedder.embedding_dimension)

    logger.info("Embedding documents.")
    documents = [
        Document(
            doc_id=uuid.uuid4(),
            name="Daichi Инструкция по монтажу и эксплуатации сплит-система серия ICE модели ICE95AVQ1 / ICE95FV1",
            full_text=pdf_to_text("./data/daichi.pdf"),
        ),
        Document(
            doc_id=uuid.uuid4(),
            name="Dantex Руководство по эксплуатации сплит-системы RK-09SVGI/RK-09SVGIE RK-12SVGI/RK-12SVGIE RK-18SVGI/RK-18SVGIE RK-24SVGI/RK-24SVGIE",
            full_text=pdf_to_text("./data/dantex.pdf"),
        ),
    ]

    embedded_chunks_per_doc = [
        embedder.embed_document(document) for document in documents
    ]

    for embedded_chunks in embedded_chunks_per_doc:
        await qdrant_engine.upsert(embedded_chunks)


if __name__ == "__main__":
    asyncio.run(main())
