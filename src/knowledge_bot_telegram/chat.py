from knowledge_bot_telegram.embedder import Embedder
from knowledge_bot_telegram.llm import LLMProvider
from knowledge_bot_telegram.schemas import Message
from knowledge_bot_telegram.vector_search import QdrantEngine


class ChatAgent:
    def __init__(
        self, embedder: Embedder, qdrant_engine: QdrantEngine, llm_provider: LLMProvider
    ):
        self.embedder = embedder
        self.qdrant_engine = qdrant_engine
        self.llm_provider = llm_provider

    async def process_message(self, *, history: list[Message], question: str) -> str:
        question_embed = self.embedder.embed_request(question)
        relevant_chunks = await self.qdrant_engine.search(question_embed)
        response = await self.llm_provider.generate_rag_response(
            history=history, question=question, relevant_chunks=relevant_chunks
        )
        return response
