import os

import yaml

from knowledge_bot_telegram.schemas import Message

yaml_file_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")

with open(yaml_file_path, "r") as file:
    PROMPTS = yaml.safe_load(file)


class LLMProvider:
    pass

    async def generate_rag_response(
        self, *, history: list[Message], question: str, relevant_chunks: list[str]
    ) -> str:
        raise NotImplementedError
