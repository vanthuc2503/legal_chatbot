from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def encode(self, text: str | list[str]) -> np.ndarray:
        return self.model.encode(text)

