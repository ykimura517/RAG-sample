from abc import ABC, abstractmethod
import json
import numpy as np


class NearestNeighborsFinder(ABC):
    @abstractmethod
    def find_nearest(self, vector: list[float], topk: int = 3) -> list[dict]:
        pass


class CosineNearestNeighborsFinder(NearestNeighborsFinder):
    def __init__(self, data_file: str):
        self.data = self._load_data(data_file)

    def _load_data(self, data_file: str) -> list[dict]:
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        # openAI embeddingのベクトルを対象にする場合は正規化されているため、np.dot(vec1, vec2) だけでも良い
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_nearest(self, vector: list[float], topk: int = 1) -> list[dict]:
        similarities = [
            (idx, self._cosine_similarity(vector, item["vector"]))
            for idx, item in enumerate(self.data)
        ]
        # 類似度が高い順にソート
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        # Top-Kの結果を返す
        return [self.data[idx] for idx, _ in sorted_similarities[:topk]]
