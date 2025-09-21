import logging
from knowledge_bot_telegram.chat import TelegramBot
from knowledge_bot_telegram.chat import ChatAgent
from knowledge_bot_telegram.embedder import Embedder
from knowledge_bot_telegram.vector_search import QdrantEngine
from knowledge_bot_telegram.llm import LLMProvider
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # TODO: move to DI
    bot = TelegramBot(
        chat_agent=ChatAgent(
            embedder=Embedder(),
            qdrant_engine=QdrantEngine(),
            llm_provider=LLMProvider(),
        )
    )
    bot.run()


if __name__ == "__main__":
    main()
