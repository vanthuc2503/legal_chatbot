from __future__ import annotations

from typing import List
import os

import faiss  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from ..config import (
    CHUNKS_CSV_PATH,
    DF_FOLDER,
    DEFAULT_TOP_K,
    FAISS_INDEX_FOLDER,
    FAISS_INDEX_PATH,
)
from ..models.embeddings import EmbeddingModel


class FaissStore:
    def __init__(self, embedder: EmbeddingModel) -> None:
        self.embedder = embedder

        # Global index + chunks
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.df = pd.read_csv(CHUNKS_CSV_PATH)

        # Per-label indices
        dfs = sorted([f for f in os.listdir(DF_FOLDER) if not f.startswith(".")])
        index_files = sorted([f for f in os.listdir(FAISS_INDEX_FOLDER) if not f.startswith(".")])

        self.label_dfs = {}
        self.label_indexes = {}

        for i in range(1, len(dfs) + 1):
            self.label_dfs[i] = pd.read_csv(DF_FOLDER / dfs[i - 1])
            self.label_indexes[i] = faiss.read_index(str(FAISS_INDEX_FOLDER / index_files[i - 1]))

    def _encode(self, query: str) -> np.ndarray:
        return np.array(self.embedder.encode(query)).reshape(1, -1)

    def similarity_search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[str]:
        query_embedding = self._encode(query)
        _, indices = self.index.search(query_embedding, top_k)

        results: List[str] = []
        for i in indices[0]:
            row = self.df.iloc[i]
            raw_prefix = [row.get("Law"), row.get("Chapter"), row.get("Section")]
            prefix = " ".join(filter(pd.notna, raw_prefix))

            article = f"{prefix}\n{row['Article']}" if prefix else row["Article"]
            results.append(article)

        return results

    def similarity_search_with_filter(self, query: str, label: int, top_k: int = DEFAULT_TOP_K) -> List[str]:
        if label not in self.label_indexes:
            return []

        query_embedding = self._encode(query)
        index = self.label_indexes[label]
        df = self.label_dfs[label]

        _, indices = index.search(query_embedding, top_k)

        results: List[str] = []
        for i in indices[0]:
            row = df.iloc[i]
            raw_prefix = [row.get("Law"), row.get("Chapter"), row.get("Section")]
            prefix = " ".join(filter(pd.notna, raw_prefix))

            article = f"{prefix}\n{row['Article']}" if prefix else row["Article"]
            results.append(article)

        return results

