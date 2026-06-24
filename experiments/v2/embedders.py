"""
embedders.py — pluggable embedding backends with a disk cache.

Three backends, exposed behind one tiny interface so the harness can swap them
by name in Stage C without touching any other code:

  * gemini-embedding-2  : Google API, latest Gemini embedder. No `task_type`
                          parameter (unlike the older `001`), so query/document
                          intent is supplied as an INLINE instruction prefix.
  * neodictabert        : dicta-il/neodictabert-bilingual-embed, Hebrew-
                          specialized, local (sentence-transformers).
  * bge-m3              : BAAI/bge-m3, local; also exposes learned-sparse weights
                          used by the sparse_rrf retrieval method.

All heavy imports are lazy. Document embeddings are cached on disk keyed by
(embedder name, chunk-set hash) so re-running a stage never re-pays embedding
cost. NOTHING here runs at import time.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "embeddings"

# Inline task instructions for gemini-embedding-2 (no task_type param exists).
_GEMINI_DOC_INSTRUCTION = "task: search document | "
_GEMINI_QUERY_INSTRUCTION = "task: search query | "


def chunkset_hash(texts: list[str]) -> str:
    """Stable short hash of a chunk set (order-sensitive) for cache keys."""
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _cache_path(embedder_name: str, texts: list[str]) -> Path:
    safe = embedder_name.replace("/", "_")
    return CACHE_DIR / f"{safe}__{chunkset_hash(texts)}.npy"


class BaseEmbedder:
    """Interface: embed_documents / embed_query, plus a `dim` and `name`."""

    name: str = "base"
    dim: int = 0

    def _encode_documents(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - backend specific
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:  # pragma: no cover - backend specific
        raise NotImplementedError

    def embed_documents(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """Embed documents, reading/writing the disk cache when enabled."""
        if use_cache:
            path = _cache_path(self.name, texts)
            if path.exists():
                print(f"    [emb-cache] {path.name}")
                return np.load(str(path)).astype("float32")
        vecs = self._encode_documents(texts).astype("float32")
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(str(_cache_path(self.name, texts)), vecs)
        return vecs


class GeminiEmbedder(BaseEmbedder):
    """gemini-embedding-2 via google-generativeai (inline task instructions)."""

    name = "gemini-embedding-2"
    dim = 3072

    def __init__(self, model: str = "models/gemini-embedding-2", sleep_s: float = 0.06,
                 output_dim: int | None = None):
        self.model = model
        self.sleep_s = sleep_s
        self.output_dim = output_dim  # Matryoshka truncation (e.g. 768/1536); None = default 3072
        if output_dim:
            self.dim = output_dim

    def _embed_one(self, text: str, instruction: str) -> list[float]:
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        kwargs = {}
        if self.output_dim:
            kwargs["output_dimensionality"] = self.output_dim
        result = genai.embed_content(model=self.model, content=instruction + text, **kwargs)
        return result["embedding"]

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for i, t in enumerate(texts):
            vecs.append(self._embed_one(t, _GEMINI_DOC_INSTRUCTION))
            if (i + 1) % 100 == 0:
                print(f"      gemini-embedding-2 {i + 1}/{len(texts)}")
            time.sleep(self.sleep_s)
        return np.array(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(self._embed_one(text, _GEMINI_QUERY_INSTRUCTION), dtype="float32")


class NeoDictaBertEmbedder(BaseEmbedder):
    """dicta-il/neodictabert-bilingual-embed — Hebrew-specialized, local."""

    name = "neodictabert"
    dim = 768

    def __init__(self, model_id: str = "dicta-il/neodictabert-bilingual-embed"):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_id, trust_remote_code=True)
        return self._model

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        # NeoDictaBERT exposes encode_document/encode_query; fall back to encode.
        if hasattr(model, "encode_document"):
            vecs = model.encode_document(texts)
        else:
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32)
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        if hasattr(model, "encode_query"):
            vec = model.encode_query([text])[0]
        else:
            vec = model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype="float32")


class BgeM3Embedder(BaseEmbedder):
    """BAAI/bge-m3 — multilingual, local. Dense + learned-sparse in one model."""

    name = "bge-m3"
    dim = 1024

    def __init__(self, model_id: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.model_id = model_id
        self.use_fp16 = use_fp16
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(self.model_id, use_fp16=self.use_fp16)
        return self._model

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        out = model.encode(texts, return_dense=True, return_sparse=False)
        return np.asarray(out["dense_vecs"], dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        out = model.encode([text], return_dense=True, return_sparse=False)
        return np.asarray(out["dense_vecs"][0], dtype="float32")

    def encode_sparse(self, texts: list[str]) -> list[dict]:
        """Return learned-sparse lexical weights (token_id -> weight) for sparse_rrf."""
        model = self._load()
        out = model.encode(texts, return_dense=False, return_sparse=True)
        return out["lexical_weights"]


_REGISTRY = {
    "gemini-embedding-2": GeminiEmbedder,
    "neodictabert": NeoDictaBertEmbedder,
    "bge-m3": BgeM3Embedder,
}


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    """Factory: return an (unloaded) embedder backend by name."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown embedder '{name}'. Choose from {list(_REGISTRY)}.")
    return _REGISTRY[name](**kwargs)
