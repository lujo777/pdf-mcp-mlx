"""
Thin wrapper around fastembed for lazy model loading and text embedding.

The embedding model is loaded once per process (singleton). If the configured
model name changes mid-process, the singleton reloads automatically.
fastembed is an optional dependency; calling encode() when it is not installed
raises ImportError with an actionable install hint.

Note: _get_model is not thread-safe. This is intentional — FastMCP uses
asyncio with a single thread for STDIO transport, so concurrent access cannot
occur in normal operation.

E5 prefix handling
------------------
Models in the intfloat E5 family are trained with asymmetric instruction
prefixes: passages must be embedded as "passage: <text>" and queries as
"query: <text>". fastembed's .embed() does NOT apply these automatically, so
encode()/encode_query() add them based on the model name. Omitting the prefixes
degrades E5 retrieval sharply (the same failure mode that collapsed
arctic-embed-m in the project benchmark). Non-E5 models are embedded unchanged,
so existing behaviour for bge / snowflake / sentence-transformers is preserved
byte-for-byte.

Caveat: this covers the standard E5 checkpoints (e5-* and multilingual-e5-*).
The "-instruct" variants use a different query prompt format and are NOT
handled here.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Asymmetric prefixes required by the intfloat E5 family.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "

# Module-level singleton. None until the first encode() call.
_model: Any = None
_model_name_loaded: str | None = None


def _needs_e5_prefix(model_name: str) -> bool:
    """True for intfloat E5 checkpoints, which require query:/passage: prefixes.

    Matches names like 'intfloat/e5-large-v2' and
    'intfloat/multilingual-e5-large'. The '-instruct' variants are deliberately
    excluded — they use a different prompt format.
    """
    low = model_name.lower()
    return "e5-" in low and "instruct" not in low


def _prepare(texts: list[str], model_name: str, *, is_query: bool) -> list[str]:
    """Apply the E5 prefix when required; otherwise return texts unchanged."""
    if not _needs_e5_prefix(model_name):
        return texts
    prefix = _E5_QUERY_PREFIX if is_query else _E5_PASSAGE_PREFIX
    return [prefix + t for t in texts]


# ---------------------------------------------------------------------------
# Optional MLX backend (Apple-Silicon GPU)
# ---------------------------------------------------------------------------
# By default embeddings run through fastembed (ONNX Runtime, CPU on macOS).
# Set PDF_MCP_EMBED_BACKEND=mlx to run them on the Apple GPU via the
# 'mlx-embeddings' package instead. The E5 query:/passage: prefixes above are
# applied identically in both backends, so retrieval quality is unchanged.
#
# Model selection for the MLX backend:
#   * config.toml `model` stays the logical name (e.g.
#     "intfloat/multilingual-e5-large") — it keeps the E5 prefixes active and
#     keeps the embedding cache keyed consistently.
#   * PDF_MCP_MLX_MODEL_PATH may point to the actual MLX weights to load
#     (a local converted dir, or an MLX repo like
#     "mlx-community/multilingual-e5-large"). If unset, the logical name is
#     passed to mlx-embeddings directly.
#
# MLX requires Apple Silicon (M1+); on Intel Macs this backend is unavailable.

_mlx_model: Any = None
_mlx_tokenizer: Any = None
_mlx_target_loaded: str | None = None


def _backend() -> str:
    """Active embedding backend: 'fastembed' (default) or 'mlx'."""
    return os.environ.get("PDF_MCP_EMBED_BACKEND", "fastembed").strip().lower()


def _get_mlx_model(model_name: str) -> Any:
    """Load the MLX model + tokenizer once; reload if the target changes.

    The on-disk target is PDF_MCP_MLX_MODEL_PATH when set, else model_name.
    """
    global _mlx_model, _mlx_tokenizer, _mlx_target_loaded
    target = os.environ.get("PDF_MCP_MLX_MODEL_PATH", "").strip() or model_name
    if _mlx_model is None or _mlx_target_loaded != target:
        try:
            from mlx_embeddings.utils import load
        except ImportError as exc:
            raise ImportError(
                "PDF_MCP_EMBED_BACKEND=mlx requires the 'mlx-embeddings' "
                "package. Install it with: pip install mlx-embeddings"
            ) from exc
        _mlx_model, _mlx_tokenizer = load(target)
        _mlx_target_loaded = target
    return _mlx_model, _mlx_tokenizer


def _mlx_encode(texts: list[str], model_name: str) -> Any:
    """Embed already-prepared texts on the Apple GPU. Returns (N, D) float32.

    `text_embeds` is mean-pooled and L2-normalized for BERT/XLM-RoBERTa models
    (the E5 family), matching the fastembed contract (dot product == cosine).
    """
    import numpy as np  # type: ignore[import-untyped]
    import mlx.core as mx  # type: ignore[import-untyped]

    model, tokenizer = _get_mlx_model(model_name)
    inputs = tokenizer.batch_encode_plus(
        list(texts),
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=512,
    )
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    embeds = outputs.text_embeds
    mx.eval(embeds)
    return np.array(embeds, dtype=np.float32)


def check_available(model_name: str) -> None:
    """
    Raise ImportError (fastembed missing) or ValueError (unknown model name).

    Call this before running semantic search to surface config errors
    before any expensive PDF work begins.
    """
    if _backend() == "mlx":
        try:
            import mlx.core  # noqa: F401  # type: ignore[import-untyped]
            import mlx_embeddings  # noqa: F401  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PDF_MCP_EMBED_BACKEND=mlx requires the 'mlx-embeddings' "
                "package. Install it with: pip install mlx-embeddings"
            ) from exc
        return
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError(
            "pdf_search semantic mode requires the 'fastembed' package. "
            "Install it with: pip install 'pdf-mcp[semantic]'"
        ) from exc
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    if model_name not in supported:
        names = ", ".join(sorted(supported))
        raise ValueError(
            f"Unknown embedding model '{model_name}'. "
            f"Supported fastembed models: {names}"
        )


def _get_model(model_name: str) -> Any:
    """Load embedding model on first call; reload if model_name changed."""
    global _model, _model_name_loaded
    if _model is None or _model_name_loaded != model_name:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "pdf_search semantic mode requires the 'fastembed' package. "
                "Install it with: pip install 'pdf-mcp[semantic]'"
            ) from exc
        _model = TextEmbedding(model_name)
        _model_name_loaded = model_name
    return _model


def encode(texts: list[str], model_name: str) -> Any:
    """
    Encode a list of passages/documents into embedding vectors.

    Returns an ndarray of shape (N, D), dtype float32.
    Vectors are L2-normalized by fastembed (dot product == cosine similarity).
    For E5 models each text is prefixed with "passage: " before embedding.
    """
    import numpy as np  # type: ignore[import-untyped]

    prepared = _prepare(texts, model_name, is_query=False)
    if _backend() == "mlx":
        return _mlx_encode(prepared, model_name)
    model = _get_model(model_name)
    embeddings = list(model.embed(prepared))
    return np.array(embeddings, dtype=np.float32)


def encode_query(text: str, model_name: str) -> Any:
    """
    Encode a single query string.

    Returns an ndarray of shape (D,), dtype float32.
    For E5 models the text is prefixed with "query: " before embedding.
    """
    import numpy as np  # type: ignore[import-untyped]

    prepared = _prepare([text], model_name, is_query=True)
    if _backend() == "mlx":
        return _mlx_encode(prepared, model_name)[0]
    model = _get_model(model_name)
    embeddings = list(model.embed(prepared))
    return np.array(embeddings, dtype=np.float32)[0]
