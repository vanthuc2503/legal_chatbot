from __future__ import annotations

from typing import List, Dict

import requests

from ..config import OLLAMA_BASE_URL, OLLAMA_MODEL_ID


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model_id: str = OLLAMA_MODEL_ID) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id

    def chat(
        self,
        messages: List[Dict[str, str]],
        num_ctx: int = 2048,
        temperature: float = 0.0,
        top_k: int = 64,
        top_p: float = 0.9,
    ) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "keep_alive": 0,
            "options": {
                "num_ctx": num_ctx,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
        }

        url = f"{self.base_url}/api/chat"

        try:
            response = requests.post(url, json=payload, timeout=120)
        except requests.RequestException:
            return "Xin lỗi, có lỗi xảy ra khi gọi API."

        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "")

        return "Xin lỗi, có lỗi xảy ra khi gọi API."

