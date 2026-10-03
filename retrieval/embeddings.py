import importlib.machinery
import logging
import sys
import types
from typing import List

import torch
import torch.nn.functional as functional
from config.settings import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


def _load_transformer_classes():
    """Load Transformers without importing blocked, inference-unused sklearn DLLs.

    Transformers imports ``sklearn.metrics.roc_curve`` from an optional text
    generation helper. BERT embedding inference never calls that function.
    Some managed Windows machines block sklearn's compiled extension, so
    provide only that optional import as a Python stub when sklearn is blocked.
    """
    needs_stub = False
    try:
        import sklearn.metrics  # noqa: F401
    except ImportError as error:
        missing_optional_sklearn = isinstance(error, ModuleNotFoundError) and error.name == "sklearn"
        if not missing_optional_sklearn and "sparsefuncs_fast" not in str(error) and "Application Control" not in str(error):
            raise
        needs_stub = True

    if needs_stub:
        sklearn_stub = types.ModuleType("sklearn")
        sklearn_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        metrics_stub = types.ModuleType("sklearn.metrics")
        metrics_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
        metrics_stub.roc_curve = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("roc_curve is unavailable in embedding-only mode")
        )
        sklearn_stub.metrics = metrics_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.metrics"] = metrics_stub
    try:
        from transformers import AutoModel, AutoTokenizer
        return AutoModel, AutoTokenizer
    finally:
        # The optional symbol has already been imported by Transformers. Do not
        # leave fake sklearn modules in the process for unrelated application code.
        if needs_stub:
            sys.modules.pop("sklearn.metrics", None)
            sys.modules.pop("sklearn", None)


class LegalEmbeddings:
    """BGE embeddings with the model's CLS pooling and L2 normalization."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        auto_model, auto_tokenizer = _load_transformer_classes()
        logger.info("Loading local Hugging Face embedding model: %s", model_name)
        self.tokenizer = auto_tokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = auto_model.from_pretrained(model_name, local_files_only=True)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def _encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                output = self.model(**encoded)
                # BAAI/bge-small-en-v1.5 uses CLS pooling followed by L2 norm.
                pooled = output.last_hidden_state[:, 0]
                vectors.append(functional.normalize(pooled, p=2, dim=1).cpu())
        if not vectors:
            return torch.empty((0, self.model.config.hidden_size), dtype=torch.float32)
        return torch.cat(vectors).to(dtype=torch.float32)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0].tolist()
