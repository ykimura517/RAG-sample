from abc import ABC, abstractmethod
from openai import OpenAI


class ChatBot(ABC):
    @abstractmethod
    def generate_response(self, user_query: str, refs: list[str]) -> str:
        pass


class GPTBasedChatBot(ChatBot):
    def __init__(self):
        self.client = OpenAI()

    def generate_response(self, user_query: str, refs: list[str]) -> str:

        # GPTによる応答を生成
        context = "\n".join(refs) + "\n"

        prompt = f"以下の情報に基づいてユーザーの質問に答えてください:\n\n{context}\n\n質問: {user_query}\n答え:"
        print("#" * 30)
        print("#" * 30)
        print(f"\nprompt:\n {prompt}\n")
        print("#" * 30)
        print("#" * 30)

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return completion.choices[0].message.content
