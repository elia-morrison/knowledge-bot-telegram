import asyncio
import os

from dotenv import load_dotenv

from knowledge_bot_telegram.chat import ChatAgent
from knowledge_bot_telegram.embedder import Embedder
from knowledge_bot_telegram.llm import LLMProvider
from knowledge_bot_telegram.schemas import Message, Role
from knowledge_bot_telegram.vector_search import QdrantEngine


async def repl() -> None:
    load_dotenv()
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

    embedder = Embedder()
    qdrant_engine = QdrantEngine()
    llm_provider = LLMProvider()
    agent = ChatAgent(
        embedder=embedder, qdrant_engine=qdrant_engine, llm_provider=llm_provider
    )

    history: list[Message] = []

    while True:
        user_input = input("you> ").strip()

        try:
            answer = await agent.process_message(history=history, question=user_input)
        except Exception as e:
            print(f"[error] Failed to process message: {e}")
            continue

        history.append(Message(role=Role.user, text=user_input))
        history.append(Message(role=Role.agent, text=answer))

        print("bot> ", answer)


def main() -> None:
    asyncio.run(repl())


if __name__ == "__main__":
    main()
