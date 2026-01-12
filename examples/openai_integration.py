"""
OpenAI API Integration with PoR Kernel
=======================================

This example shows how to integrate PoR Kernel with the OpenAI API
by adding coherence gating to chat completions.

The SilenceGatedOpenAI wraps the OpenAI client and applies PoR Kernel
decision logic before returning responses.

Requirements:
    pip install openai

Usage:
    from examples.openai_integration import SilenceGatedOpenAI

    client = SilenceGatedOpenAI(
        api_key="your-api-key",
        coherence_threshold=0.75,
    )

    result = client.chat("Explain machine learning")
    if result.silenced:
        print("System chose silence:", result.reason)
    else:
        print("Response:", result.content)
"""

from dataclasses import dataclass
from typing import Any, Optional

from por_kernel.kernel import SilenceToken, por_kernel


# =============================================================================
# Response Types
# =============================================================================


@dataclass
class GatedResponse:
    """Response from the silence-gated OpenAI client."""

    content: Optional[str]
    silenced: bool
    reason: Optional[str] = None
    coherence: float = 0.0
    drift: float = 0.0
    raw_response: Any = None

    def __bool__(self) -> bool:
        return not self.silenced


# =============================================================================
# Coherence Estimation
# =============================================================================


def estimate_response_coherence(
    messages: list[dict[str, str]],
    response_content: str,
    model_confidence: Optional[float] = None,
) -> float:
    """
    Estimate coherence of the response relative to the conversation.

    This is a heuristic implementation. For production, consider:
    - Using the OpenAI embeddings API to measure semantic similarity
    - Fine-tuned coherence classifiers
    - Entropy-based measures from logprobs

    Args:
        messages: The conversation history
        response_content: The model's response
        model_confidence: Optional confidence score from the model

    Returns:
        Coherence score between 0.0 and 1.0
    """
    if model_confidence is not None:
        return model_confidence

    # Extract the last user message for comparison
    user_messages = [m for m in messages if m.get("role") == "user"]
    last_query = user_messages[-1]["content"] if user_messages else ""

    # Heuristic coherence factors
    factors = []

    # 1. Response length factor (too short or too long suggests issues)
    response_len = len(response_content)
    if response_len < 20:
        factors.append(0.3)
    elif response_len < 100:
        factors.append(0.6)
    elif response_len < 1000:
        factors.append(0.9)
    else:
        factors.append(0.8)

    # 2. Query term overlap (basic relevance)
    query_terms = set(last_query.lower().split())
    response_terms = set(response_content.lower().split())
    if query_terms:
        overlap = len(query_terms & response_terms) / len(query_terms)
        factors.append(0.5 + overlap * 0.5)
    else:
        factors.append(0.7)

    # 3. Uncertainty markers (lower coherence if uncertain)
    uncertainty_markers = [
        "i'm not sure",
        "i don't know",
        "it's unclear",
        "i cannot",
        "i apologize",
        "as an ai",
        "i'm unable",
        "i cannot provide",
    ]
    response_lower = response_content.lower()
    uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response_lower)
    factors.append(max(0.3, 1.0 - uncertainty_count * 0.2))

    # 4. Structure factor (well-structured responses are more coherent)
    has_structure = any(
        [
            "\n" in response_content,  # Multi-line
            ":" in response_content,  # Contains explanations
            "1." in response_content or "•" in response_content,  # Lists
        ]
    )
    factors.append(0.9 if has_structure else 0.7)

    return sum(factors) / len(factors)


def calculate_conversation_drift(messages: list[dict[str, str]]) -> float:
    """
    Calculate how much the conversation has drifted from the original topic.

    Args:
        messages: Full conversation history

    Returns:
        Drift score between 0.0 (no drift) and 1.0 (complete drift)
    """
    if len(messages) <= 2:
        return 0.0

    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if len(user_messages) < 2:
        return 0.0

    first_query = user_messages[0].lower()
    last_query = user_messages[-1].lower()

    # Term overlap between first and last queries
    first_terms = set(first_query.split())
    last_terms = set(last_query.split())

    if first_terms:
        overlap = len(first_terms & last_terms) / len(first_terms)
        term_drift = 1.0 - overlap
    else:
        term_drift = 0.0

    # Conversation length factor
    length_factor = min(0.3, len(messages) * 0.02)

    return min(1.0, term_drift * 0.7 + length_factor)


# =============================================================================
# SilenceGatedOpenAI Client
# =============================================================================


class SilenceGatedOpenAI:
    """
    OpenAI client with PoR Kernel silence gating.

    Wraps the OpenAI chat completions API and applies coherence/drift
    checks before returning responses.

    Example:
        >>> client = SilenceGatedOpenAI(api_key="sk-...")
        >>>
        >>> # Single query
        >>> result = client.chat("What is photosynthesis?")
        >>> if result:
        ...     print(result.content)
        ... else:
        ...     print(f"Silenced: {result.reason}")
        >>>
        >>> # Conversation with context
        >>> client.add_message("user", "What is DNA?")
        >>> client.add_message("assistant", "DNA is...")
        >>> result = client.chat("How does it relate to proteins?")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        coherence_threshold: float = 0.7,
        drift_tolerance: float = 0.1,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize the silence-gated OpenAI client.

        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            model: Model to use for completions
            coherence_threshold: Minimum coherence to proceed
            drift_tolerance: Maximum drift to proceed
            system_prompt: Optional system prompt for all conversations
        """
        self.api_key = api_key
        self.model = model
        self.coherence_threshold = coherence_threshold
        self.drift_tolerance = drift_tolerance
        self.messages: list[dict[str, str]] = []

        if system_prompt:
            self.messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        # Note: In production, you would initialize the actual client:
        # from openai import OpenAI
        # self.client = OpenAI(api_key=api_key)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear conversation history (keeps system prompt if any)."""
        self.messages = [m for m in self.messages if m.get("role") == "system"]

    def chat(
        self,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> GatedResponse:
        """
        Send a chat message with silence gating.

        Args:
            user_message: The user's message
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            GatedResponse with content (if proceeded) or reason (if silenced)
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        # In production, this would call the actual API:
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=self.messages,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        # )
        # response_content = response.choices[0].message.content

        # For demonstration, we'll simulate a response
        response_content = self._simulate_response(user_message)

        # Calculate coherence and drift
        coherence = estimate_response_coherence(self.messages, response_content)
        drift = calculate_conversation_drift(self.messages)

        # Apply PoR Kernel decision
        decision = por_kernel(
            drift=drift,
            coherence=coherence,
            tol=self.drift_tolerance,
            thresh=self.coherence_threshold,
        )

        if isinstance(decision, SilenceToken):
            # Remove the user message since we're not proceeding
            self.messages.pop()

            reasons = []
            if coherence < self.coherence_threshold:
                reasons.append(f"low coherence ({coherence:.2f})")
            if drift > self.drift_tolerance:
                reasons.append(f"high drift ({drift:.2f})")

            return GatedResponse(
                content=None,
                silenced=True,
                reason=" and ".join(reasons),
                coherence=coherence,
                drift=drift,
            )

        # PROCEED - add response to history
        self.messages.append({"role": "assistant", "content": response_content})

        return GatedResponse(
            content=response_content,
            silenced=False,
            coherence=coherence,
            drift=drift,
        )

    def _simulate_response(self, query: str) -> str:
        """Simulate an API response for demonstration purposes."""
        query_lower = query.lower()

        if "photosynthesis" in query_lower:
            return (
                "Photosynthesis is the process by which plants, algae, and some "
                "bacteria convert light energy into chemical energy. The process occurs "
                "in chloroplasts and involves capturing sunlight to convert carbon dioxide "
                "and water into glucose and oxygen. This is fundamental to life on Earth "
                "as it produces oxygen and forms the base of most food chains."
            )

        if "machine learning" in query_lower:
            return (
                "Machine learning is a subset of artificial intelligence that "
                "enables systems to learn and improve from experience without being "
                "explicitly programmed. It focuses on developing algorithms that can "
                "access data, learn from it, and make predictions or decisions."
            )

        if "dna" in query_lower or "protein" in query_lower:
            return (
                "DNA (deoxyribonucleic acid) contains the genetic instructions "
                "for the development and function of living organisms. It encodes "
                "proteins through a process called gene expression, where DNA is "
                "first transcribed into RNA, then translated into proteins."
            )

        return "I'm not sure I can provide a reliable answer to that question."


# =============================================================================
# High-Level API
# =============================================================================


def gated_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4",
    api_key: str | None = None,
    coherence_threshold: float = 0.7,
    drift_tolerance: float = 0.1,
) -> GatedResponse:
    """
    One-shot gated completion without maintaining state.

    Args:
        messages: OpenAI-format messages list
        model: Model to use
        api_key: OpenAI API key
        coherence_threshold: Minimum coherence to proceed
        drift_tolerance: Maximum drift to proceed

    Returns:
        GatedResponse
    """
    client = SilenceGatedOpenAI(
        api_key=api_key,
        model=model,
        coherence_threshold=coherence_threshold,
        drift_tolerance=drift_tolerance,
    )

    # Add all but last message to history
    for msg in messages[:-1]:
        client.add_message(msg["role"], msg["content"])

    # Chat with the last message
    last_msg = messages[-1]
    return client.chat(last_msg["content"])


# =============================================================================
# Example Usage
# =============================================================================


if __name__ == "__main__":
    print("OpenAI + PoR Kernel Integration Example")
    print("=" * 50)

    client = SilenceGatedOpenAI(
        api_key="demo-key",  # Would be real key in production
        model="gpt-4",
        coherence_threshold=0.7,
        drift_tolerance=0.2,
    )

    # Test 1: Good query - should PROCEED
    print("\n[Test 1] Query: 'What is photosynthesis?'")
    result = client.chat("What is photosynthesis?")
    print(f"  Silenced: {result.silenced}")
    print(f"  Coherence: {result.coherence:.2f}, Drift: {result.drift:.2f}")
    if result.content:
        print(f"  Response: {result.content[:100]}...")

    # Test 2: Vague query - should SILENCE
    print("\n[Test 2] Query: 'Tell me about xyz'")
    result = client.chat("Tell me about xyz")
    print(f"  Silenced: {result.silenced}")
    print(f"  Coherence: {result.coherence:.2f}, Drift: {result.drift:.2f}")
    if result.silenced:
        print(f"  Reason: {result.reason}")

    # Test 3: Follow-up in conversation
    print("\n[Test 3] Follow-up: 'How does it work?'")
    result = client.chat("How does it work?")
    print(f"  Silenced: {result.silenced}")
    print(f"  Coherence: {result.coherence:.2f}, Drift: {result.drift:.2f}")

    print("\n" + "=" * 50)
    print("Integration complete.")
