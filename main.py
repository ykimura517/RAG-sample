import os
from embedder import OpenAIEmbedder
from searcher import CosineNearestNeighborsFinder
from chatBot import GPTBasedChatBot

# OpenAI APIキーを事前に環境変数にセットしてください。
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("APIキーがセットされていません。")


def main():

    embedder = OpenAIEmbedder(api_key)
    searcher = CosineNearestNeighborsFinder("sample_data.json")

    user_query: str = "アートを教えられる先生を探しています"

    user_query_vector: list[float] = embedder.embed([user_query])[0]
    search_results: list[dict] = searcher.find_nearest(user_query_vector, topk=2)
    chat_bot = GPTBasedChatBot()
    response: str = chat_bot.generate_response(
        user_query, [search_result["text"] for search_result in search_results]
    )

    print("*" * 30)
    print("*" * 30)
    print("【AIの返答】")
    print(response)


if __name__ == "__main__":
    main()
