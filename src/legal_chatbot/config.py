from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths for FAISS and data
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss.index"
CHUNKS_CSV_PATH = PROJECT_ROOT / "chunks.csv"
DF_FOLDER = PROJECT_ROOT / "data_with_label"
FAISS_INDEX_FOLDER = PROJECT_ROOT / "faiss_index"

# Local classifier path
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "phobert_finetuned"

# Hugging Face models
DECOMPOSER_MODEL_NAME = "phmtung/dpl302m-model"
EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_ID = "gemma3:27b"

# Retrieval defaults
DEFAULT_TOP_K = 3

