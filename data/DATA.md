# Legal Knowledge Base & Data

This project uses a curated legal knowledge base built from Vietnamese legislative documents collected from the official **Pháp điển** portal. The raw texts (laws, decrees, circulars, etc.) are cleaned, segmented, labeled, and indexed to support fast semantic search for the chatbot.

---

## Source and Pre-processing

- **Source**: Vietnamese legal documents downloaded from the Pháp điển portal.
- **Cleaning & normalization**:
  - Remove boilerplate content, formatting artifacts and duplicated sections.
  - Normalize headings and sections to a consistent hierarchy: *Law → Chapter → Section → Article*.
- **Chunking**:
  - Long legal texts are split into shorter, coherent chunks at article/paragraph level.
  - Each chunk keeps its structural metadata (law name, chapter, section, article).
- **Labeling**:
  - Each chunk is assigned to one of 5 key legal domains (with an additional “Other” group):
    - Children
    - Gender Equality
    - Population
    - Marriage & Family
    - Prevention of Domestic Violence
    - Other
- **Vectorization & indexing**:
  - Each chunk is embedded using `dangvantuan/vietnamese-document-embedding`.
  - Embeddings are stored in FAISS indexes for efficient similarity search.

---

## Data Artifacts

The main processed artifacts used by the chatbot are:

| Artifact | Description |
|----------|-------------|
| **`chunks.csv`** | Global corpus of all legal chunks after cleaning and segmentation. Columns typically include: `Law`, `Chapter`, `Section`, `Article`, plus any additional metadata. Used with the global FAISS index for broad semantic search across all domains. |
| **`data_with_label/`** | Folder containing 5 CSV files, one per legal domain. Each file is a filtered subset of `chunks.csv` for a specific label. Supports domain-specific retrieval when the classifier predicts a topic. |
| **`faiss.index`** | Global FAISS index built on all chunks in `chunks.csv`. Supports domain-agnostic similarity search. |
| **`faiss_index/`** | Directory containing **`index1` … `index5`**, one FAISS index per labeled domain. Each `index{i}` aligns with the corresponding CSV in `data_with_label/`. Used when the classifier predicts a specific legal topic. |
| **`phobert_finetuned/`** | Directory containing the fine-tuned PhoBERT checkpoint for topic classification. Outputs a label in `{1..5, 6}` (6 = “Other”) to select the appropriate FAISS index. |

---

## Availability & Setup

In some environments, large data assets are not committed directly to the repository (due to size and licensing). If the following are missing:

- `chunks.csv`
- `data_with_label/`
- `faiss.index`
- `faiss_index/`
- `phobert_finetuned/`

download them from the project’s designated storage (e.g. Google Drive / S3) and place them at the expected paths (project root or as configured in `src/legal_chatbot/config.py`) before running the chatbot.

---

## Usage in the Chatbot Pipeline

At inference time:

1. The user query is classified by the fine-tuned PhoBERT model into one of the legal domains.
2. The query (and its decomposed sub-queries) are embedded with the sentence-embedding model.
3. FAISS is queried both **per-domain** (via `faiss_index/index{label}` and `data_with_label/`) and **globally** (via `faiss.index` and `chunks.csv`).
4. Retrieved chunks are de-duplicated and merged into a single context string (Law name, Chapter, Section, Article text).
5. This context is passed to the generative model (Ollama), which produces grounded answers with explicit legal citations.
