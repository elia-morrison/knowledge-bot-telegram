import itertools
import os
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client import models

import logging

from knowledge_bot_telegram.schemas import EmbeddedDocumentChunk, EmbeddedRequest

logger = logging.getLogger(__name__)


class QdrantEngine:
    def __init__(self):
        self.qdrant_client = AsyncQdrantClient(url=os.environ["QDRANT_URL"])

    async def check_collection_initialized(self, collection_name: str) -> bool:
        return await self.qdrant_client.collection_exists(collection_name)

    async def initialize_collection(self, collection_name: str, vector_size: int = 768):
        await self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

        await self.qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_id",
            field_schema="keyword",
        )

    async def upsert(self, collection_name: str, chunks: list[EmbeddedDocumentChunk]):
        for batched_chunks in itertools.batched(chunks, 100):
            await self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        payload={
                            "text": point.text,
                            "doc_id": point.doc_id,
                        },
                        vector={
                            "dense": point.dense_vector,
                            "bm25": models.SparseVector(
                                values=list(point.bm25_vector.values()),
                                indices=list(point.bm25_vector.keys()),
                            ),
                        },
                    )
                    for point in batched_chunks
                ],
            )

    async def search(self, request: EmbeddedRequest, top_k: int = 10):
        pass
