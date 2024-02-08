from abc import ABC, abstractmethod
import json
import openai


# データをベクトル化するモジュールのインターフェース
class Embedder(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, texts: list[str], filename: str) -> bool:
        raise NotImplementedError


# Embedderインターフェースの実装
class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        # openai 1.10.0 で動作確認
        response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        # レスポンスからベクトルを抽出
        return [data.embedding for data in response.data]

    def save(self, texts: list[str], filename: str) -> bool:
        vectors = self.embed(texts)
        data_to_save = [
            {"id": idx, "text": text, "vector": vector}
            for idx, (text, vector) in enumerate(zip(texts, vectors))
        ]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        print(f"{filename} に保存されました。")
        return True


if __name__ == "__main__":
    import os

    texts = [
        "佐藤一郎は、東京生まれの35歳のプログラマーです。趣味は写真撮影とハイキング。新しい技術を学ぶことに情熱を注いでいます。",
        "鈴木花子は、北海道出身の28歳のイラストレーターです。猫を二匹飼っており、自然と動物を愛する心優しい人物です。",
        "田中健二は、大阪で小さなカフェを経営する45歳の起業家です。コーヒーに対する深い知識と情熱を持ち、地域社会に貢献しています。",
        "山本美咲は、福岡県出身の22歳の大学生です。法律を専攻しており、将来は人権に関わる仕事に就きたいと考えています。",
        "伊藤高志は、長野県の山の中で育った30歳の写真家です。自然の美しさを捉えることに特化し、国内外で展示会を開催しています。",
        "小林由紀子は、沖縄県出身の40歳の小学校教師です。子どもたちに芸術と文化の大切さを教えることに生きがいを感じています。",
    ]

    # OpenAI APIキーを事前に環境変数にセットしてください。
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("APIキーがセットされていません。")

    embedder = OpenAIEmbedder(api_key)
    embedder.save(texts, "sample_data.json")
