from __future__ import annotations

from typing import List, Dict

from ..llm.ollama_client import OllamaClient
from ..models.classifier import LawClassifier
from ..models.decomposer import QueryDecomposer
from ..models.embeddings import EmbeddingModel
from ..retrieval.faiss_store import FaissStore


class ChatbotService:
    def __init__(self) -> None:
        self.decomposer = QueryDecomposer()
        self.classifier = LawClassifier()
        self.embedder = EmbeddingModel()
        self.retriever = FaissStore(self.embedder)
        self.llm = OllamaClient()

        self.chat_history: List[Dict[str, str]] = []

    def reset(self) -> None:
        self.chat_history = []

    def _build_initial_context(self, query: str) -> str:
        sub_questions = self.decomposer.decompose(query)
        label = self.classifier.predict_label(query)

        data1 = self.retriever.similarity_search_with_filter(query, label) if label != 6 else []
        data2 = self.retriever.similarity_search(query)

        data3: List[str] = []
        for q in sub_questions:
            data3.extend(self.retriever.similarity_search(q))

        context: set[str] = set(data1) | set(data2)
        for chunk in data3:
            context.add(chunk)

        return "\n".join(context)

    def chat(self, query: str) -> List[Dict[str, str]]:
        if not self.chat_history:
            context_text = self._build_initial_context(query)
            system_prompt = f"""Bạn là OpenAPI, một chatbot hỗ trợ pháp lý tại Việt Nam.
Nhiệm vụ của bạn là trả lời một câu hỏi về pháp luật dựa trên ngữ cảnh sau đây.

**Ngữ cảnh**: {context_text}

**Vui lòng tuân thủ các yêu cầu sau**:
    1. **Luôn trả lời bằng tiếng Việt.** 
    2. **Trả lời ngắn gọn không lan man DƯỚI 100 TỪ và trình bày rõ ràng, dễ hiểu.**    
    3. **Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp ở trên.** Tuyệt đối **không bịa đặt** thông tin. Nếu không thể trả lời dựa trên ngữ cảnh, hãy nói: "Xin lỗi, tôi không thể trả lời câu hỏi này của bạn."
    4. **Câu trả lời được dùng từ phần nào của ngữ cảnh thì phải trích dẫn nguồn**, quy tắc trích dẫn là nguồn bao gồm tên văn bản pháp luật (ví dụ như luật, nghị định, thông tư,...), chương, mục và điều khoản.  
        Ví dụ trích dẫn nguồn:  
            - *Luật Hôn nhân và Gia đình, Chương II, Điều 8.*  
            - *Bộ luật Lao động, Mục 3, Điều 34.*"""

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query + " RESPONSE UNDER 150 TOKENS"},
            ]

            response = self.llm.chat(
                messages,
                num_ctx=2048,
                temperature=0.0,
                top_k=64,
                top_p=0.9,
            )

            self.chat_history = messages + [{"role": "assistant", "content": response}]
        else:
            system_prompt = """Bạn là OpenAPI, một chatbot hỗ trợ pháp lý tại Việt Nam.
Bạn được được cung cấp một ngữ cảnh từ lịch sử cuộc trò chuyện với người dùng. Lịch sử trò chuyện này bao gồm các câu hỏi của người dùng, các Điều luật, bộ luật, thông tư etc. mà người dùng đã cung cấp kèm theo câu trả lời của chatbot cho mỗi câu hỏi.
Nhiệm vụ của bạn là trả lời câu hỏi tiếp theo mà người dùng hỏi trong cuộc trò chuyện đó.

**Vui lòng tuân thủ các yêu cầu sau**:
    1. **Luôn trả lời bằng tiếng Việt.**
    2. **Trả lời ngắn gọn không lan man DƯỚI 100 TỪ và trình bày rõ ràng, dễ hiểu.**     
    3. **Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp.** Tuyệt đối **không bịa đặt** thông tin. Nếu không thể trả lời dựa trên ngữ cảnh, hãy nói: "Xin lỗi, tôi không thể trả lời câu hỏi này của bạn."
    4. **Câu trả lời được dùng từ phần nào của ngữ cảnh thì phải trích dẫn nguồn**, quy tắc trích dẫn là nguồn bao gồm tên văn bản pháp luật, chương, mục và điều khoản."""

            messages = self.chat_history + [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query + " RESPONSE UNDER 150 TOKENS"},
            ]

            response = self.llm.chat(
                messages,
                num_ctx=3072,
                temperature=0.2,
                top_k=64,
                top_p=0.9,
            )

            self.chat_history = messages + [{"role": "assistant", "content": response}]

        return self.chat_history


_service: ChatbotService | None = None


def get_chatbot_service() -> ChatbotService:
    global _service
    if _service is None:
        _service = ChatbotService()
    return _service

