from qdrant_client import AsyncQdrantClient

import logging

logger = logging.getLogger(__name__)


class QdrantEngine:
    def __init__(self, url: str):
        self.qdrant_client = AsyncQdrantClient(url=url)

    def search(self, query: str, top_k: int = 10):
        pass
