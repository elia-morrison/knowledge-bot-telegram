import os
from qdrant_client import AsyncQdrantClient
from qdrant_client import models

import logging

from knowledge_bot_telegram.schemas import EmbeddedRequest

logger = logging.getLogger(__name__)


class QdrantEngine:
    def __init__(self):
        self.qdrant_client = AsyncQdrantClient(url=os.environ["QDRANT_URL"])

    async def check_collection_initialized(self, collection_name: str) -> bool:
        return await self.qdrant_client.collection_exists(collection_name)

    async def initialize_collection(self, collection_name: str):
        await self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=1536, distance=models.Distance.COSINE)
            },
            sparse_vector_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

        await self.qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_id",
            field_schema="keyword",
        )

    async def search(self, request: EmbeddedRequest, top_k: int = 10):
        pass
