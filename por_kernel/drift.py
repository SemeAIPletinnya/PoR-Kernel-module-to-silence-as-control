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
    high_ppl: float = 50.0,
) -> MetricResult:
    """
    Compute coherence from model perplexity.

    Lower perplexity implies higher coherence.

    Args:
        perplexity: Model perplexity score
        low_ppl: Perplexity value considered highly coherent
        high_ppl: Perplexity value considered incoherent

    Returns:
        MetricResult with coherence value
    """
    if perplexity <= 0:
        return MetricResult(value=0.0, confidence=0.0, details={"error": "invalid perplexity"})

    if perplexity <= low_ppl:
        coherence = 1.0
    elif perplexity >= high_ppl:
        coherence = 0.0
    else:
        coherence = 1.0 - (perplexity - low_ppl) / (high_ppl - low_ppl)

    return MetricResult(
        value=coherence,
        confidence=0.8,
        details={"perplexity": perplexity, "low_ppl": low_ppl, "high_ppl": high_ppl},
    )


def compute_coherence_self_consistency(
    samples: Sequence[str],
    *,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> MetricResult:
    """
    Compute coherence based on self-consistency among multiple samples.

    Args:
        samples: Multiple model outputs for the same prompt
        similarity_fn: Optional similarity function between two samples

    Returns:
        MetricResult with coherence value
    """
    if len(samples) < 2:
        return MetricResult(value=1.0, confidence=0.3, details={"reason": "single sample"})

    if similarity_fn is None:
        similarity_fn = _simple_similarity

    total_similarity = 0.0
    comparisons = 0
    for i, sample_a in enumerate(samples):
        for sample_b in samples[i + 1 :]:
            total_similarity += similarity_fn(sample_a, sample_b)
            comparisons += 1

    avg_similarity = total_similarity / comparisons if comparisons else 0.0
    return MetricResult(
        value=avg_similarity,
        confidence=min(1.0, comparisons / 10),
        details={"comparisons": comparisons},
    )


def compute_coherence_entropy(
    token_probs: Sequence[float],
    *,
    max_entropy: float = 5.0,
) -> MetricResult:
    """
    Compute coherence based on entropy of token probabilities.

    Lower entropy implies higher coherence.

    Args:
        token_probs: Token probability distribution
        max_entropy: Maximum expected entropy for normalization

    Returns:
        MetricResult with coherence value
    """
    if not token_probs:
        return MetricResult(value=0.0, confidence=0.0, details={"error": "empty distribution"})

    entropy = -sum(p * math.log(max(p, 1e-12)) for p in token_probs)
    normalized_entropy = min(entropy / max_entropy, 1.0)
    coherence = 1.0 - normalized_entropy

    return MetricResult(
        value=coherence,
        confidence=0.7,
        details={"entropy": entropy, "normalized_entropy": normalized_entropy},
    )


def compute_coherence_factual_grounding(
    grounded_facts: Sequence[str],
    total_facts: Sequence[str],
) -> MetricResult:
    """
    Compute coherence based on factual grounding.

    Args:
        grounded_facts: Facts verified or grounded in context
        total_facts: All facts claimed in output

    Returns:
        MetricResult with coherence value
    """
    if not total_facts:
        return MetricResult(value=1.0, confidence=0.4, details={"reason": "no facts"})

    grounded = len(grounded_facts)
    total = len(total_facts)
    coherence = grounded / total if total else 0.0

    return MetricResult(
        value=coherence,
        confidence=min(1.0, total / 10),
        details={"grounded": grounded, "total": total},
    )


# =============================================================================
# Combined stability score
# =============================================================================


def compute_stability_score(
    drift: MetricResult,
    coherence: MetricResult,
    *,
    drift_weight: float = 0.5,
    coherence_weight: float = 0.5,
) -> MetricResult:
    """
    Combine drift and coherence into a single stability score.

    Args:
        drift: Drift metric result
        coherence: Coherence metric result
        drift_weight: Weight for drift (higher drift decreases stability)
        coherence_weight: Weight for coherence (higher coherence increases stability)

    Returns:
        MetricResult with stability score (0 = unstable, 1 = stable)
    """
    total_weight = drift_weight + coherence_weight
    if total_weight == 0:
        return MetricResult(value=0.0, confidence=0.0, details={"error": "zero weights"})

    normalized_drift_weight = drift_weight / total_weight
    normalized_coherence_weight = coherence_weight / total_weight

    stability = (
        (1.0 - drift.value) * normalized_drift_weight
        + coherence.value * normalized_coherence_weight
    )
    stability = max(0.0, min(1.0, stability))

    confidence = min(drift.confidence, coherence.confidence)
    return MetricResult(
        value=stability,
        confidence=confidence,
        details={
            "drift_weight": drift_weight,
            "coherence_weight": coherence_weight,
            "drift_value": drift.value,
            "coherence_value": coherence.value,
        },
    )


# =============================================================================
# Helper functions
# =============================================================================


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = _dot_product(a, b)
    norm_a = math.sqrt(_dot_product(a, a))
    norm_b = math.sqrt(_dot_product(b, b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _simple_similarity(a: str, b: str) -> float:
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
