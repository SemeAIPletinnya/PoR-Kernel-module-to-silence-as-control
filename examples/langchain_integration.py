"""
LangChain Integration with PoR Kernel
======================================

This example demonstrates how to integrate PoR Kernel into a LangChain
pipeline to enable silence-as-control for coherence gating.

The SilenceGatedChain wraps any LangChain chain and applies the PoR Kernel
decision logic before emitting output.

Requirements:
    pip install langchain langchain-openai

Usage:
    from examples.langchain_integration import SilenceGatedChain

    chain = SilenceGatedChain(
        base_chain=your_chain,
        coherence_estimator=your_estimator,
        drift_calculator=your_drift_calc
    )
    result = chain.invoke({"input": "your query"})
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from por_kernel.kernel import SilenceToken, por_kernel


# =============================================================================
# Coherence & Drift Estimators (Example Implementations)
# =============================================================================

def simple_coherence_estimator(query: str, response: str) -> float:
    """
    Simple coherence estimator based on response characteristics.

    In production, replace with:
    - Embedding similarity (query vs response)
    - NLI-based entailment scores
    - Custom fine-tuned coherence models

    Returns:
        float: Coherence score between 0.0 and 1.0
    """
    if not response or len(response.strip()) < 10:
        return 0.2  # Very short responses are likely incoherent

    # Check for common incoherence signals
    incoherence_signals = [
        "I don't know",
        "I'm not sure",
        "I cannot",
        "as an AI",
        "I apologize",
        "unclear",
        "ambiguous",
    ]

    response_lower = response.lower()
    signal_count = sum(1 for sig in incoherence_signals if sig.lower() in response_lower)

    # Base coherence from response length and structure
    base_coherence = min(1.0, len(response) / 500)

    # Penalize for incoherence signals
    coherence = base_coherence * (1.0 - signal_count * 0.15)

    return max(0.0, min(1.0, coherence))


def conversation_drift_calculator(
    original_query: str,
    current_context: str,
    conversation_history: list[str],
) -> float:
    """
    Calculate semantic drift from original query.

    In production, replace with:
    - Embedding distance from original query
    - Topic modeling divergence
    - Attention pattern analysis

    Returns:
        float: Drift score between 0.0 (no drift) and 1.0 (complete drift)
    """
    if not conversation_history:
        return 0.0

    # Simple heuristic: longer conversations tend to drift more
    history_length = len(conversation_history)
    base_drift = min(0.5, history_length * 0.05)

    # Check if original query terms still appear in recent context
    query_terms = set(original_query.lower().split())
    context_terms = set(current_context.lower().split())

    if query_terms:
        overlap = len(query_terms & context_terms) / len(query_terms)
        term_drift = 1.0 - overlap
    else:
        term_drift = 0.5

    # Combine factors
    drift = (base_drift + term_drift) / 2

    return max(0.0, min(1.0, drift))


# =============================================================================
# SilenceGatedChain - LangChain Integration
# =============================================================================


@dataclass
class SilenceResult:
    """Result when kernel decides to silence."""

    reason: str
    drift: float
    coherence: float

    def __repr__(self) -> str:
        return (
            f"<SILENCE: {self.reason} (drift={self.drift:.2f}, "
            f"coherence={self.coherence:.2f})>"
        )


class SilenceGatedChain:
    """
    A LangChain wrapper that applies PoR Kernel gating to chain outputs.

    Before emitting any response, this chain:
    1. Measures coherence of the generated response
    2. Calculates drift from the original query
    3. Applies por_kernel() to decide: PROCEED or SILENCE

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain.prompts import ChatPromptTemplate
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> prompt = ChatPromptTemplate.from_template("Answer: {input}")
        >>> base_chain = prompt | llm
        >>>
        >>> gated_chain = SilenceGatedChain(
        ...     base_chain=base_chain,
        ...     coherence_threshold=0.7,
        ...     drift_tolerance=0.3
        ... )
        >>>
        >>> result = gated_chain.invoke({"input": "What is quantum computing?"})
        >>> if isinstance(result, SilenceResult):
        ...     print("System chose to remain silent:", result.reason)
        ... else:
        ...     print("Response:", result)
    """

    def __init__(
        self,
        base_chain: Any,
        coherence_estimator: Callable[[str, str], float] | None = None,
        drift_calculator: Callable[[str, str, list[str]], float] | None = None,
        coherence_threshold: float = 0.7,
        drift_tolerance: float = 0.1,
        silence_message: Optional[str] = None,
    ) -> None:
        """
        Initialize the silence-gated chain.

        Args:
            base_chain: The underlying LangChain chain to wrap
            coherence_estimator: Function(query, response) -> coherence score
            drift_calculator: Function(query, context, history) -> drift score
            coherence_threshold: Minimum coherence to proceed (default: 0.7)
            drift_tolerance: Maximum drift to proceed (default: 0.1)
            silence_message: Optional message to return instead of None on silence
        """
        self.base_chain = base_chain
        self.coherence_estimator = coherence_estimator or simple_coherence_estimator
        self.drift_calculator = drift_calculator or conversation_drift_calculator
        self.coherence_threshold = coherence_threshold
        self.drift_tolerance = drift_tolerance
        self.silence_message = silence_message
        self.conversation_history: list[str] = []

    def invoke(self, inputs: dict[str, Any]) -> Any | SilenceResult:
        """
        Invoke the chain with silence gating.

        Args:
            inputs: Input dictionary for the chain

        Returns:
            Chain output if PROCEED, SilenceResult if SILENCE
        """
        query = inputs.get("input", str(inputs))

        # Get response from base chain
        response = self.base_chain.invoke(inputs)
        response_text = str(response.content if hasattr(response, "content") else response)

        # Measure coherence and drift
        coherence = self.coherence_estimator(query, response_text)
        drift = self.drift_calculator(query, response_text, self.conversation_history)

        # Apply PoR Kernel decision
        decision = por_kernel(
            drift=drift,
            coherence=coherence,
            tol=self.drift_tolerance,
            thresh=self.coherence_threshold,
        )

        if isinstance(decision, SilenceToken):
            reason = []
            if drift > self.drift_tolerance:
                reason.append(f"high drift ({drift:.2f} > {self.drift_tolerance})")
            if coherence < self.coherence_threshold:
                reason.append(
                    f"low coherence ({coherence:.2f} < {self.coherence_threshold})"
                )

            return SilenceResult(
                reason=" and ".join(reason),
                drift=drift,
                coherence=coherence,
            )

        # PROCEED - update history and return response
        self.conversation_history.append(response_text)
        return response

    def reset_history(self) -> None:
        """Clear conversation history and reset drift tracking."""
        self.conversation_history = []


# =============================================================================
# Convenience Functions
# =============================================================================


def create_gated_chain(base_chain: Any, strict: bool = False) -> SilenceGatedChain:
    """
    Factory function to create a SilenceGatedChain with preset configurations.

    Args:
        base_chain: The LangChain chain to wrap
        strict: If True, use stricter thresholds (coherence=0.8, drift=0.05)

    Returns:
        Configured SilenceGatedChain
    """
    if strict:
        return SilenceGatedChain(
            base_chain=base_chain,
            coherence_threshold=0.8,
            drift_tolerance=0.05,
        )
    return SilenceGatedChain(
        base_chain=base_chain,
        coherence_threshold=0.7,
        drift_tolerance=0.1,
    )


# =============================================================================
# Example Usage
# =============================================================================


if __name__ == "__main__":
    print("LangChain + PoR Kernel Integration Example")
    print("=" * 50)

    # Simulate a chain response for demonstration
    class MockChain:
        def invoke(self, inputs: dict[str, Any]) -> Any:
            query = inputs.get("input", "")
            if "quantum" in query.lower():
                return type(
                    "Response",
                    (),
                    {
                        "content": (
                            "Quantum computing uses quantum mechanical phenomena "
                            "like superposition and entanglement to perform computations. "
                            "Unlike classical computers that use bits (0 or 1), quantum "
                            "computers use qubits which can exist in multiple states "
                            "simultaneously, enabling parallel processing of complex problems."
                        )
                    },
                )()
            return type("Response", (), {"content": "I'm not sure about that."})()

    # Create gated chain
    mock_chain = MockChain()
    gated = SilenceGatedChain(
        base_chain=mock_chain,
        coherence_threshold=0.7,
        drift_tolerance=0.1,
    )

    # Test 1: Good query - should PROCEED
    print("\n[Test 1] Query: 'What is quantum computing?'")
    result = gated.invoke({"input": "What is quantum computing?"})
    if isinstance(result, SilenceResult):
        print(f"  → SILENCE: {result}")
    else:
        print(f"  → PROCEED: {result.content[:80]}...")

    # Test 2: Vague query - should SILENCE
    print("\n[Test 2] Query: 'Tell me about xyz'")
    result = gated.invoke({"input": "Tell me about xyz"})
    if isinstance(result, SilenceResult):
        print(f"  → SILENCE: {result}")
    else:
        print(f"  → PROCEED: {result.content[:80]}...")

    print("\n" + "=" * 50)
    print("Integration complete. See docstrings for usage details.")
