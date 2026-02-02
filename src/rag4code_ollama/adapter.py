from typing import Any, Dict, List
from rag4code.interfaces.embedder import EmbedderInterface

class OllamaEmbedder(EmbedderInterface):
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "qwen3-embedding", **kwargs):
        self.base_url = base_url
        self.model_name = model_name
        print(f"DEBUG: OllamaEmbedder инициализирован для {base_url}, модель: {model_name}")

    def embed_text(self, text: str) -> List[float]:
        print(f"DEBUG: OllamaEmbedder генерирует эмбеддинг для текста: {text[:20]}...")
        return [0.1, 0.2, 0.3]  # Dummy vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"DEBUG: OllamaEmbedder генерирует эмбеддинги для {len(texts)} документов")
        return [[0.1, 0.2, 0.3] for _ in texts]