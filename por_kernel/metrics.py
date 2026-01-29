# PoR Kernel - Metrics Module
# Practical implementations of drift and coherence metrics

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import math


# =============================================================================
# Metric Result Types
# =============================================================================


@dataclass
class MetricResult:
    """
    Result of a metric computation.

    Attributes:
        value: The computed metric value (0.0 to 1.0)
        confidence: Confidence in the measurement (0.0 to 1.0)
        details: Additional computation details
    """

    value: float
    confidence: float = 1.0
    details: dict | None = None

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}

    def __float__(self) -> float:
        return self.value


# =============================================================================
# Drift Metrics
# =============================================================================


def compute_drift_embedding(
    current_embedding: Sequence[float],
    reference_embedding: Sequence[float],
    *,
    metric: str = "cosine",
) -> MetricResult:
    """
    Compute drift using embedding similarity.

    Drift = 1 - similarity(current, reference)

    Args:
        current_embedding: Current context embedding
        reference_embedding: Reference/original context embedding
        metric: Similarity metric ("cosine", "euclidean", "dot")

    Returns:
        MetricResult with drift value (0 = no drift, 1 = maximum drift)

    Example:
        >>> current = [0.1, 0.2, 0.3]
        >>> reference = [0.1, 0.2, 0.3]
        >>> result = compute_drift_embedding(current, reference)
        >>> result.value  # Should be ~0.0 (no drift)
    """
    if len(current_embedding) != len(reference_embedding):
        return MetricResult(
            value=1.0,
            confidence=0.0,
            details={"error": "embedding dimension mismatch"},
        )

    if len(current_embedding) == 0:
        return MetricResult(value=0.0, confidence=0.0, details={"error": "empty embeddings"})

    if metric == "cosine":
        similarity = _cosine_similarity(current_embedding, reference_embedding)
    elif metric == "euclidean":
        distance = _euclidean_distance(current_embedding, reference_embedding)
        similarity = 1.0 / (1.0 + distance)
    elif metric == "dot":
        similarity = max(-1.0, min(1.0, _dot_product(current_embedding, reference_embedding)))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    drift = 1.0 - similarity
    drift = max(0.0, min(1.0, drift))

    return MetricResult(
        value=drift,
        confidence=1.0,
        details={"metric": metric, "similarity": similarity},
    )


def compute_drift_token_overlap(
    current_tokens: Sequence[str],
    reference_tokens: Sequence[str],
    *,
    use_idf: bool = False,
) -> MetricResult:
    """
    Compute drift using token/keyword overlap.

    Drift = 1 - Jaccard similarity of token sets

    Args:
        current_tokens: Tokens from current context
        reference_tokens: Tokens from reference context
        use_idf: If True, weight tokens by inverse document frequency

    Returns:
        MetricResult with drift value
    """
    _ = use_idf

    if not current_tokens and not reference_tokens:
        return MetricResult(value=0.0, confidence=0.5, details={"reason": "empty token sets"})

    current_set = set(current_tokens)
    reference_set = set(reference_tokens)

    intersection = len(current_set & reference_set)
    union = len(current_set | reference_set)

    if union == 0:
        return MetricResult(value=0.0, confidence=0.5)

    jaccard = intersection / union
    drift = 1.0 - jaccard

    return MetricResult(
        value=drift,
        confidence=min(1.0, len(reference_tokens) / 10),
        details={
            "jaccard_similarity": jaccard,
            "intersection_size": intersection,
            "union_size": union,
        },
    )


def compute_drift_sliding_window(
    history: Sequence[Sequence[float]],
    current: Sequence[float],
    *,
    window_size: int = 5,
) -> MetricResult:
    """
    Compute drift using sliding window average of recent embeddings.

    Useful for detecting gradual drift over conversation turns.

    Args:
        history: List of previous embeddings
        current: Current embedding
        window_size: Number of recent embeddings to average

    Returns:
        MetricResult with drift value
    """
    if not history:
        return MetricResult(value=0.0, confidence=0.3, details={"reason": "no history"})

    recent = list(history[-window_size:])

    dim = len(current)
    avg_embedding = [0.0] * dim
    for emb in recent:
        for i in range(min(dim, len(emb))):
            avg_embedding[i] += emb[i]

    for i in range(dim):
        avg_embedding[i] /= len(recent)

    result = compute_drift_embedding(current, avg_embedding)

    return MetricResult(
        value=result.value,
        confidence=min(1.0, len(recent) / window_size),
        details={
            **result.details,
            "window_size": len(recent),
            "target_window": window_size,
        },
    )


# =============================================================================
# Coherence Metrics
# =============================================================================


def compute_coherence_perplexity(
    perplexity: float,
    *,
    low_ppl: float = 10.0,
    high_ppl: float = 100.0,
) -> MetricResult:
    """
    Compute coherence from language model perplexity.

    Lower perplexity = higher coherence.

    Args:
        perplexity: Model perplexity score
        low_ppl: Perplexity at which coherence = 1.0
        high_ppl: Perplexity at which coherence = 0.0

    Returns:
        MetricResult with coherence value (0 = incoherent, 1 = highly coherent)
    """
    if perplexity <= low_ppl:
        coherence = 1.0
    elif perplexity >= high_ppl:
        coherence = 0.0
    else:
        coherence = 1.0 - (perplexity - low_ppl) / (high_ppl - low_ppl)

    return MetricResult(
        value=coherence,
        confidence=1.0,
        details={
            "perplexity": perplexity,
            "low_threshold": low_ppl,
            "high_threshold": high_ppl,
        },
    )


def compute_coherence_self_consistency(
    responses: Sequence[str],
    *,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> MetricResult:
    """
    Compute coherence via self-consistency across multiple samples.

    High agreement between samples = high coherence.

    Args:
        responses: Multiple sampled responses to the same prompt
        similarity_fn: Function to compare response similarity (default: exact match ratio)

    Returns:
        MetricResult with coherence value
    """
    if len(responses) < 2:
        return MetricResult(value=1.0, confidence=0.3, details={"reason": "insufficient samples"})

    if similarity_fn is None:
        similarity_fn = _default_string_similarity

    similarities = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = similarity_fn(responses[i], responses[j])
            similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities)

    return MetricResult(
        value=avg_similarity,
        confidence=min(1.0, len(responses) / 5),
        details={
            "num_samples": len(responses),
            "pairwise_comparisons": len(similarities),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities),
        },
    )


def compute_coherence_entropy(
    token_probabilities: Sequence[Sequence[float]],
    *,
    max_entropy: float = 5.0,
) -> MetricResult:
    """
    Compute coherence from token prediction entropy.

    Low entropy = high confidence predictions = high coherence.

    Args:
        token_probabilities: Probability distributions for each predicted token
        max_entropy: Entropy value at which coherence = 0

    Returns:
        MetricResult with coherence value
    """
    if not token_probabilities:
        return MetricResult(value=1.0, confidence=0.0, details={"reason": "no probabilities"})

    entropies = []
    for probs in token_probabilities:
        entropy = _compute_entropy(probs)
        entropies.append(entropy)

    avg_entropy = sum(entropies) / len(entropies)

    coherence = max(0.0, 1.0 - (avg_entropy / max_entropy))

    return MetricResult(
        value=coherence,
        confidence=min(1.0, len(entropies) / 10),
        details={
            "avg_entropy": avg_entropy,
            "max_entropy_threshold": max_entropy,
            "num_tokens": len(entropies),
        },
    )


def compute_coherence_factual_grounding(
    response: str,
    source_documents: Sequence[str],
    *,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> MetricResult:
    """
    Compute coherence based on factual grounding to source documents.

    High grounding = response is consistent with sources = high coherence.

    Args:
        response: Generated response text
        source_documents: Source documents for fact-checking
        similarity_fn: Function to compute response-source similarity

    Returns:
        MetricResult with coherence value
    """
    if not source_documents:
        return MetricResult(value=0.5, confidence=0.3, details={"reason": "no sources"})

    if similarity_fn is None:
        similarity_fn = _default_string_similarity

    similarities = [similarity_fn(response, doc) for doc in source_documents]
    max_sim = max(similarities)
    avg_sim = sum(similarities) / len(similarities)

    return MetricResult(
        value=max_sim,
        confidence=min(1.0, len(source_documents) / 3),
        details={
            "max_similarity": max_sim,
            "avg_similarity": avg_sim,
            "num_sources": len(source_documents),
        },
    )


# =============================================================================
# Composite Metrics
# =============================================================================


def compute_stability_score(
    drift: float,
    coherence: float,
    *,
    drift_weight: float = 0.5,
    coherence_weight: float = 0.5,
) -> MetricResult:
    """
    Compute overall stability score combining drift and coherence.

    Stability = weighted combination of (1 - drift) and coherence.

    Args:
        drift: Drift value (0 to 1)
        coherence: Coherence value (0 to 1)
        drift_weight: Weight for drift component
        coherence_weight: Weight for coherence component

    Returns:
        MetricResult with stability score (higher = more stable)
    """
    total_weight = drift_weight + coherence_weight
    if total_weight == 0:
        return MetricResult(value=0.5, confidence=0.0)

    stability = (
        drift_weight * (1.0 - drift) + coherence_weight * coherence
    ) / total_weight

    return MetricResult(
        value=stability,
        confidence=1.0,
        details={
            "drift": drift,
            "coherence": coherence,
            "drift_contribution": drift_weight * (1.0 - drift) / total_weight,
            "coherence_contribution": coherence_weight * coherence / total_weight,
        },
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def _compute_entropy(probs: Sequence[float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _default_string_similarity(a: str, b: str) -> float:
    """Simple string similarity based on character overlap."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    def ngrams(s: str, n: int = 3) -> set[str]:
        return set(s[i : i + n] for i in range(len(s) - n + 1))

    a_ngrams = ngrams(a.lower())
    b_ngrams = ngrams(b.lower())

    if not a_ngrams or not b_ngrams:
        return 1.0 if a.lower() == b.lower() else 0.0

    intersection = len(a_ngrams & b_ngrams)
    union = len(a_ngrams | b_ngrams)

    return intersection / union if union > 0 else 0.0


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MetricResult",
    "compute_drift_embedding",
    "compute_drift_token_overlap",
    "compute_drift_sliding_window",
    "compute_coherence_perplexity",
    "compute_coherence_self_consistency",
    "compute_coherence_entropy",
    "compute_coherence_factual_grounding",
    "compute_stability_score",
]
