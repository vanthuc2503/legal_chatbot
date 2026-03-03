from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import CLASSIFIER_MODEL_PATH


LABEL_MAPPING: Dict[int, str] = {
    0: "Trẻ em",
    1: "Bình đẳng giới",
    2: "Dân số",
    3: "Hôn nhân và gia đình",
    4: "Phòng, chống bạo lực gia đình",
    5: "Khác",
}


class LawClassifier:
    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        if model_path is None:
            model_path = str(CLASSIFIER_MODEL_PATH)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def predict_label(self, query: str) -> int:
        """Predict law label of a given query (1..6)."""
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()

        # Shift to 1-based labels to match existing notebook logic
        return predicted_label + 1

