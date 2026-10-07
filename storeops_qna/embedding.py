"""임베딩. 한국어를 지원하는 BAAI/bge-m3 (명세서 6절 '임베딩: 미정'에 대한 제안). 벡터는 L2 정규화되어 내적 = 코사인 유사도."""
from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class BgeM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("pip install sentence-transformers 가 필요합니다") from e
            self._model = SentenceTransformer(self.name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    """글자 2~3-gram 해시. 의미 검색이 아니라 글자가 겹치는 정도만 봅니다. 테스트와 모의 연결 전용."""

    name = "hash-ngram"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = "".join(t.split())
            for n in (2, 3):
                for j in range(len(t) - n + 1):
                    out[i, zlib.crc32(t[j : j + n].encode("utf-8")) % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


def get_embedder(backend: str, model_name: str = "BAAI/bge-m3") -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    if backend == "bge-m3":
        return BgeM3Embedder(model_name)
    raise ValueError(f"알 수 없는 임베딩 backend: {backend}")
