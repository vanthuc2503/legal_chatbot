from __future__ import annotations

from typing import List
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import DECOMPOSER_MODEL_NAME


class QueryDecomposer:
    def __init__(self, model_name: str = DECOMPOSER_MODEL_NAME, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == "cuda" and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(device)

        self.model.eval()

    def decompose(self, query: str) -> List[str]:
        """Break down a query into up to 3 simpler sub-questions."""
        system_prompt = (
            "You are a helpful assistant that generates multiple sub-questions related to an input question.\n"
            "The goal is to break down the input into a set of 3 sub-problems / sub-questions that can be "
            "answered in isolation and must be understandable stand-alone questions.\n"
            f"Generate multiple search queries related to: {query}\n\n"
            "**Respond format:**\n"
            "Write a short analysis of why the query needs decomposition and how it will be done to derive the "
            "final 3 decomposed sub-questions.\n"
            "[Sub-question 1, Sub-question 2, Sub-question 3]"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        with torch.no_grad():
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
            ).to(self.device)

            output_ids = self.model.generate(
                input_ids=inputs,
                max_new_tokens=512,
                use_cache=True,
            )

            decoded_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        match = re.search(r"\[([^]]*)\](?!.*\[)", decoded_output, re.DOTALL)
        if not match:
            return [query]

        raw = match.group(1)
        questions = [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]

        return questions or [query]

