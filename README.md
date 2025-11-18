## Trợ lý Pháp luật Việt Nam

Dự án xây dựng một chatbot hỗ trợ tra cứu pháp luật Việt Nam theo thời gian thực. Hệ thống sử dụng kho văn bản pháp luật được thu thập từ cổng **Pháp điển**, sau đó tiền xử lý thành các đoạn ngắn, nhúng vector và lập chỉ mục bằng FAISS để truy vấn nhanh. Giao diện người dùng được triển khai bằng Gradio, mô hình sinh trả lời chạy qua Ollama với `gemma3:27b`.

---

### 1. Tính năng chính

- Hiểu câu hỏi tiếng Việt liên quan đến 5 chủ đề pháp luật trọng điểm: **Trẻ em, Bình đẳng giới, Dân số, Hôn nhân & Gia đình, Phòng chống bạo lực gia đình** và nhóm **Khác**.
- Phân tách câu hỏi phức tạp thành các truy vấn nhỏ để tăng độ bao phủ khi tìm kiếm.
- Kết hợp phân loại chủ đề + tìm kiếm ngữ nghĩa trên FAISS để thu thập đúng điều luật.
- Sinh câu trả lời ngắn gọn (<100 từ) và luôn kèm trích dẫn nguồn (Luật/Chương/Mục/Điều).
- Giao diện chat trực quan, hỗ trợ reset lịch sử và xuất bản công khai thông qua Gradio share link.

---

### 2. Kiến trúc tổng quan

1. **Xử lý truy vấn**
   - `phmtung/dpl302m-model` (Hugging Face) dùng để tách 1 câu hỏi thành tối đa 3 truy vấn phụ.
   - Bộ phân loại PhoBERT fine-tuned (`./phobert_finetuned`) dự đoán chủ đề luật để chọn FAISS index chuyên biệt.
2. **Nhúng & tìm kiếm**
   - `dangvantuan/vietnamese-document-embedding` tạo vector cho câu hỏi.
   - Bộ chỉ mục `faiss.index` (tổng hợp) + `faiss_index/index1-5` (theo từng chủ đề) trả về các điều luật gần nhất.
3. **Sinh câu trả lời**
   - Ollama phục vụ mô hình `gemma3:27b` (trả lời) và `gemma3:12b` (tùy chọn dự phòng).
   - Ngữ cảnh cung cấp cho LLM gồm: kết quả tìm kiếm toàn cục, theo chủ đề, và từng truy vấn phụ.
4. **Giao diện**
   - `gr.Blocks` dựng UI chat; mỗi input được giới hạn <150 tokens để đảm bảo latency và chi phí.

Sơ đồ ngắn: **Query → Decompose → Classify → Retrieve (FAISS) → Compose context → Ollama → Gradio UI**.

---

### 2.1 Flow xử lý end-to-end

1. **Người dùng đặt câu hỏi** trong giao diện Gradio (ví dụ về quyền nuôi con).
2. **Tách truy vấn**: câu hỏi được gửi cho `phmtung/dpl302m-model`, model trả về tối đa 3 truy vấn nhỏ độc lập để bao phủ đầy đủ ý hỏi.
3. **Phân loại chủ đề**: câu hỏi gốc đi qua PhoBERT fine-tuned để xác định thuộc 1 trong 5 chủ đề (hoặc Khác). Kết quả này giúp chọn đúng FAISS index chuyên biệt (ví dụ chỉ mục “Hôn nhân & Gia đình”).
4. **Nhúng vector**: cả câu hỏi gốc và từng truy vấn phụ được mã hóa bằng `dangvantuan/vietnamese-document-embedding`.
5. **Truy vấn FAISS**:
   - **Theo chủ đề**: dùng index tương ứng (`index{label}`) để lấy 3 đoạn liên quan nhất.
   - **Toàn cục**: dùng `faiss.index` để phòng trường hợp câu hỏi khớp nhiều chủ đề.
   - **Truy vấn phụ**: mỗi sub-query tiếp tục truy vấn FAISS để bổ sung ngữ cảnh chi tiết.
6. **Tổng hợp ngữ cảnh**: ghép các đoạn luật, bỏ trùng, định dạng dạng “Luật – Chương – Mục – Điều”.
7. **Sinh câu trả lời**: hệ thống tạo system prompt (yêu cầu trả lời <100 từ, bắt buộc trích nguồn) rồi gửi toàn bộ ngữ cảnh + câu hỏi đến Ollama (`gemma3:27b`).
8. **Trả kết quả**: nội dung trả lời kèm trích dẫn được hiển thị trong Gradio Chatbot và lưu vào `chat_history` để phục vụ câu hỏi tiếp theo trong cùng session.

Flow trên giúp người dùng hiểu chính xác dữ liệu được xử lý thế nào từ lúc đặt câu hỏi cho đến khi nhận câu trả lời đầy đủ trích dẫn.

---

### 3. Yêu cầu hệ thống

- **OS**: Linux/WSL/Cloud máy ảo (đã kiểm thử trên Ubuntu 22.04).
- **Python**: 3.10+ (khuyến nghị tạo virtualenv).
- **GPU**: >=24 GB VRAM cho `gemma3:27b` (có thể hạ xuống 12B nếu tài nguyên hạn chế).
- **Dung lượng lưu trữ**: ~60 GB (mô hình + dữ liệu FAISS).
- **Phụ thuộc chính**: xem `requirements.txt`.

---

### 4. Chuẩn bị môi trường

```bash
# 1. Tạo virtualenv (khuyến nghị)
python3 -m venv .venv && source .venv/bin/activate

# 2. Cài thư viện Python
pip install -r requirements.txt

# 3. Cài Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 4. Khởi chạy dịch vụ Ollama và nạp model
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup ollama serve &
ollama pull gemma3:12b
ollama pull gemma3:27b
```

> Các bước 3–4 được tóm tắt trong `setup.txt`.

---

### 5. Chuẩn bị dữ liệu & FAISS

- `chunks.csv`: toàn bộ dữ liệu sau bước chia nhỏ và làm sạch.
- `data_with_label/`: 5 tệp CSV tương ứng từng chủ đề luật.
- `faiss.index`: chỉ mục tổng hợp.
- `faiss_index/`: gồm `index1 … index5` (mỗi index ứng với một nhãn).
- `phobert_finetuned/`: thư mục checkpoint mô hình phân loại.

Nếu repository không chứa các thư mục trên, hãy tải từ nguồn lưu trữ riêng (Google Drive/S3) rồi đặt đúng đường dẫn trước khi chạy notebook.

---

### 6. Chạy chatbot

1. Đảm bảo Ollama service đang chạy (`ps aux | grep ollama`).
2. Mở notebook `Chatbot.ipynb` (VS Code, Jupyter Lab…).
3. Chạy tuần tự các cell:
   - Cài yêu cầu, cài Ollama (nếu chưa có).
   - Thiết lập biến môi trường GPU.
   - Nạp mô hình decomposition, phân loại, embedder và FAISS index.
   - Khởi tạo Gradio Blocks (`demo.launch()`).
4. Copy đường dẫn `http://127.0.0.1:7861` (local) hoặc link `gradio.live` để truy cập UI.

---

### 7. Sử dụng

- Nhập câu hỏi pháp luật tiếng Việt, ví dụ: _“Thủ tục đăng ký kết hôn cần giấy tờ gì?”_
- Bot trả lời ngắn gọn, cuối câu có trích dẫn dạng _Luật Hôn nhân và Gia đình, Chương II, Điều 8_.
- Nút **🔄 Đoạn chat mới** để xóa lịch sử và giải phóng bộ nhớ GPU.

---

### 8. Đánh giá & theo dõi chất lượng

- File `chatbot_evaluation_results.xlsx` tổng hợp điểm đánh giá thủ công nhiều tình huống hỏi đáp (độ chính xác, mức liên quan, cách trích dẫn).
- Có thể bổ sung kịch bản `multiple choice` trong phần cuối notebook để kiểm tra tự động.

---

### 9. Cấu trúc thư mục

- `Chatbot.ipynb`: notebook chính chứa toàn bộ pipeline.
- `requirements.txt`: danh sách thư viện Python.
- `setup.txt`: hướng dẫn tóm tắt cài đặt môi trường đám mây.
- `faiss.index`, `faiss_index/`, `chunks.csv`, `data_with_label/`: dữ liệu tri thức.
- `chatbot_evaluation_results.xlsx`: kết quả đánh giá.

---

### 10. Định hướng mở rộng

- Tự động hóa pipeline ETL để cập nhật luật mới.
- Tối ưu bộ trích dẫn (ví dụ chuẩn hóa tên văn bản).
- Đóng gói dưới dạng API Flask/FastAPI để tích hợp hệ thống khác.
- Bổ sung kiểm thử (unit & integration) cho các bước decomposition, classification, retrieval.

---

**Liên hệ**: vui lòng mở issue hoặc PR nếu bạn muốn đóng góp thêm tính năng/nguồn dữ liệu mới. Cảm ơn bạn đã quan tâm!
