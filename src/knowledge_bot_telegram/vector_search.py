import itertools
import os
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client import models

import logging

from knowledge_bot_telegram.schemas import EmbeddedDocumentChunk, EmbeddedRequest

logger = logging.getLogger(__name__)


class QdrantEngine:
    def __init__(self, collection_name: str = "knowledge_bot_telegram"):
        self.collection_name = collection_name
        self.qdrant_client = AsyncQdrantClient(url=os.environ["QDRANT_URL"])

    async def check_collection_initialized(self) -> bool:
        return await self.qdrant_client.collection_exists(self.collection_name)

    async def initialize_collection(self, vector_size: int = 768):
        await self.qdrant_client.create_collection(
            collection_name=self.collection_name,
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
            collection_name=self.collection_name,
            field_name="doc_id",
            field_schema="keyword",
        )

    async def upsert(self, chunks: list[EmbeddedDocumentChunk]):
        for batched_chunks in itertools.batched(chunks, 100):
            await self.qdrant_client.upsert(
                collection_name=self.collection_name,
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

    async def search(self, request: EmbeddedRequest, top_k: int = 10) -> list[str]:
        results = await self.qdrant_client.query_points(
            collection_name=self.collection_name,
            with_payload=True,
            with_vectors=False,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(request.bm25_vector.keys()),
                        values=list(request.bm25_vector.values()),
                    ),
                    using="bm25",
                    limit=100,
                ),
                models.Prefetch(
                    query=request.dense_vector,
                    using="dense",
                    limit=100,
                ),
            ],
            limit=top_k,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )

        return [result.payload["text"] for result in results.points if result.payload]
