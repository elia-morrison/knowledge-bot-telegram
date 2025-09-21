from collections import deque
import logging
import os
import random

import telegramify_markdown
from knowledge_bot_telegram.embedder import Embedder
from knowledge_bot_telegram.llm import LLMProvider
from knowledge_bot_telegram.schemas import Message, Role
from knowledge_bot_telegram.vector_search import QdrantEngine
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)


logger = logging.getLogger(__name__)


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


class TelegramBot:
    def __init__(self, chat_agent: ChatAgent):
        self.token = os.environ["TELEGRAM_BOT_TOKEN"]
        self.chat_agent = chat_agent
        self.reactions = ["👌", "👀", "🫡"]
        self.chat_histories: dict[int, deque[Message]] = {}
        self.max_history = 5
        self.app = Application.builder().token(self.token).build()

        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.info("New client started")
        if update.message:
            await update.message.reply_text(
                self.chat_agent.llm_provider.get_greeting_text()
            )

    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message:
            logger.warning("No message in update")
            return
        chat_id = update.message.chat_id
        user_message = update.message.text

        if not user_message:
            logger.warning("No text in message")
            return

        if chat_id not in self.chat_histories:
            self.chat_histories[chat_id] = deque(maxlen=self.max_history)

        self.chat_histories[chat_id].append(Message(role=Role.user, text=user_message))

        reaction = random.choice(self.reactions)

        try:
            await context.bot.set_message_reaction(
                chat_id=chat_id, message_id=update.message.message_id, reaction=reaction
            )

            response = await self.chat_agent.process_message(
                history=list(self.chat_histories[chat_id]), question=user_message
            )
            self.chat_histories[chat_id].append(Message(role=Role.agent, text=response))
            await update.message.reply_text(
                telegramify_markdown.markdownify(response), parse_mode="MarkdownV2"
            )
            await context.bot.set_message_reaction(
                chat_id=chat_id,
                message_id=update.message.message_id,
                reaction=None,
            )
        except Exception as e:
            print(f"Failed to process message: {e}")
            await update.message.reply_text(
                self.chat_agent.llm_provider.get_exception_text()
            )

    def run(self) -> None:
        logger.info("Serving the bot...")
        self.app.run_polling()
