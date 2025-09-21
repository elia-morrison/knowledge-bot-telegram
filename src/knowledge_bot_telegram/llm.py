import os
import re
import traceback

import httpx
import yaml

from knowledge_bot_telegram.schemas import Message

yaml_file_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")

with open(yaml_file_path, "r") as file:
    PROMPTS = yaml.safe_load(file)


class LLMProvider:
    def __init__(self):
        self.model = "qwen3-14b"
        self.api_key = os.environ["GPTUNNEL_API_KEY"]
        self.url = "https://gptunnel.ru/v1/chat/completions"

    async def generate_rag_response(
        self, *, history: list[Message], question: str, relevant_chunks: list[str]
    ) -> str:
        prompt = PROMPTS["rag_response"].format(
            history=self._format_history(history=history),
            question=question,
            relevant_chunks=self._format_relevant_chunks(
                relevant_chunks=relevant_chunks
            ),
        )
        print(prompt)
        return await self._send_request(prompt=prompt)

    def _postprocess_prompt(self, prompt: str) -> str:
        return re.sub(r"\n+", "\n", prompt)

    def _format_history(self, *, history: list[Message]) -> str:
        return "\n".join([f"{msg.role}: {msg.text}" for msg in history])

    def _format_relevant_chunks(self, relevant_chunks: list[str]) -> str:
        return "\n======\n".join(relevant_chunks)

    async def _send_request(self, prompt: str) -> str:
        auth_header = (
            self.api_key
            if self.api_key.lower().startswith("bearer ")
            else f"Bearer {self.api_key}"
        )
        headers = {
            "Authorization": auth_header,
        }
        data = {
            "model": self.model,
            "useWalletBalance": True,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                }
            ],
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.url, headers=headers, json=data, timeout=60.0
                )
                print(f"HTTP {response.status_code} {response.reason_phrase}")
                print(f"Content-Type: {response.headers.get('content-type')}")

                response_json = response.json()

                if response.status_code >= 400:
                    print(f"HTTP error body: {response_json}")
                    response.raise_for_status()

                return response_json["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Exception type: {type(e).__name__}, repr: {repr(e)}")
                traceback.print_exc()
                return "Прости, что-то пошло не так. Попробуй позже."
