# Vietnamese Legal Assistant

A chatbot that supports real-time lookup of Vietnamese law. The system uses a legal text corpus collected from the **Pháp điển** portal, preprocessed into short chunks, embedded as vectors, and indexed with FAISS for fast retrieval. The UI can be run via **Streamlit** (`app.py`) or **Gradio** in the notebook; answer generation is powered by Ollama with `gemma3:27b`.

---

## 1. Main features

- Handles Vietnamese questions on 5 core legal topics: **Children**, **Gender Equality**, **Population**, **Marriage & Family**, **Prevention of Domestic Violence**, plus an **Other** category.
- Decomposes complex questions into smaller sub-queries to improve search coverage.
- Combines topic classification with FAISS semantic search to retrieve relevant legal articles.
- Generates concise answers (<100 words) with mandatory source citations (Law / Chapter / Section / Article).
- Chat UI with session reset and optional public sharing (Gradio share link when using the notebook).

---

## 2. Architecture overview

1. **Query processing**
   - `phmtung/dpl302m-model` (Hugging Face) decomposes one question into up to 3 sub-queries.
   - Fine-tuned PhoBERT classifier (`./phobert_finetuned`) predicts the legal topic and selects the topic-specific FAISS index.
2. **Embedding & search**
   - `dangvantuan/vietnamese-document-embedding` produces query vectors.
   - Index set: global `faiss.index` plus per-topic `faiss_index/index1–5` return the nearest legal passages.
3. **Answer generation**
   - Ollama serves `gemma3:27b` (primary) and optionally `gemma3:12b` (fallback).
   - Context for the LLM combines: global search results, topic-specific results, and results from each sub-query.
4. **Interface**
   - Streamlit (`app.py`) or Gradio in the notebook; inputs are limited to &lt;150 tokens for latency and cost.

**Pipeline**: **Query → Decompose → Classify → Retrieve (FAISS) → Compose context → Ollama → UI**.

---

## 2.1 End-to-end flow

1. **User asks a question** in the Streamlit or Gradio UI (e.g. about child custody).
2. **Query decomposition**: the question is sent to `phmtung/dpl302m-model`, which returns up to 3 independent sub-queries for broader coverage.
3. **Topic classification**: the original question is classified by fine-tuned PhoBERT into one of 5 topics (or Other), and the corresponding FAISS index is selected (e.g. “Marriage & Family”).
4. **Vector encoding**: the main query and each sub-query are embedded with `dangvantuan/vietnamese-document-embedding`.
5. **FAISS retrieval**:
   - **By topic**: use `index{label}` to fetch the top 3 relevant passages.
   - **Global**: use `faiss.index` for cross-topic coverage.
   - **Sub-queries**: each sub-query is also run against FAISS to enrich context.
6. **Context assembly**: retrieved passages are merged, deduplicated, and formatted as “Law – Chapter – Section – Article”.
7. **Answer generation**: a system prompt (answer &lt;100 words, mandatory citations) plus context and question are sent to Ollama (`gemma3:27b`).
8. **Response**: the answer with citations is shown in the chat UI and stored in `chat_history` for follow-up questions in the same session.

---

## 3. System requirements

- **OS**: Linux / WSL / cloud VM (tested on Ubuntu 22.04).
- **Python**: 3.10+ (virtualenv recommended).
- **GPU**: ≥24 GB VRAM for `gemma3:27b` (or use 12B if resources are limited).
- **Storage**: ~60 GB (models + FAISS data).
- **Dependencies**: see `requirements.txt`.

---

## 4. Environment setup

```bash
# 1. Create virtualenv (recommended)
python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS
# On Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt
pip install streamlit requests   # for Streamlit app

# 3. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 4. Start Ollama and pull models
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup ollama serve &
ollama pull gemma3:12b
ollama pull gemma3:27b
```

> Steps 3–4 are summarized in `setup.txt`.

---

## 5. Data & FAISS

- **`chunks.csv`**: full corpus of legal chunks after cleaning and segmentation.
- **`data_with_label/`**: 5 CSV files, one per legal topic.
- **`faiss.index`**: global FAISS index.
- **`faiss_index/`**: `index1` … `index5`, one per topic label.
- **`phobert_finetuned/`**: classifier checkpoint directory.

If these assets are not in the repo, download them from the project’s storage (e.g. Google Drive / S3) and place them at the expected paths before running. See **`data/DATA.md`** for a detailed data description.

---

## 6. Running the chatbot

### Option A: Streamlit (recommended)

```bash
# Ensure Ollama is running (e.g. ps aux | grep ollama)
streamlit run app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`).

### Option B: Jupyter notebook (Gradio)

1. Ensure Ollama is running (`ps aux | grep ollama`).
2. Open `notebooks/Chatbot.ipynb` in VS Code, Jupyter Lab, etc.
3. Run the cells in order: install deps, set GPU env, load models and FAISS, then launch Gradio (`demo.launch()`).
4. Use the local URL (e.g. `http://127.0.0.1:7861`) or the Gradio share link to access the UI.

---

## 7. Usage

- Type a Vietnamese legal question, e.g. *“Thủ tục đăng ký kết hôn cần giấy tờ gì?”* (What documents are needed for marriage registration?).
- The bot replies concisely with citations such as *Luật Hôn nhân và Gia đình, Chương II, Điều 8*.
- Use **🔄 Đoạn chat mới** (New chat) to clear history and free GPU memory.

---

## 8. Evaluation & quality

- **`chatbot_evaluation_results.xlsx`** summarizes manual scores for multiple Q&A scenarios (accuracy, relevance, citation quality).
- The notebook’s “multiple choice” section can be used for automated evaluation.

---

## 9. Project structure

```
legal_chatbot/
├── app.py                    # Streamlit entrypoint
├── requirements.txt
├── setup.txt                 # Short cloud/env setup notes
├── data/
│   └── DATA.md               # Data & FAISS artifact description
├── notebooks/
│   └── Chatbot.ipynb         # Full pipeline + Gradio UI
├── src/
│   └── legal_chatbot/       # Package: config, models, retrieval, llm, chat
│       ├── config.py
│       ├── models/          # decomposer, classifier, embeddings
│       ├── retrieval/       # FAISS store
│       ├── llm/             # Ollama client
│       └── chat/            # ChatbotService
├── faiss.index, chunks.csv, data_with_label/, faiss_index/, phobert_finetuned/  # Data (see data/DATA.md)
└── chatbot_evaluation_results.xlsx   # Optional evaluation file
```

---

## 10. Future work

- Automate ETL to ingest new legislation.
- Improve citation formatting (e.g. normalize law names).
- Expose a Flask/FastAPI API for integration with other systems.
- Add unit and integration tests for decomposition, classification, and retrieval.

---

**Contact**: open an issue or PR if you want to contribute features or new data sources. Thank you for your interest!
