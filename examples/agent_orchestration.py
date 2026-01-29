"""
Multi-Agent Orchestration with PoR Kernel
==========================================

This example demonstrates how to use PoR Kernel in a multi-agent system
where agents coordinate through a shared silence protocol.

Key concepts:
- Each agent has its own coherence/drift thresholds
- Agents can vote on whether to proceed (consensus-based silence)
- The orchestrator aggregates decisions using PoR Kernel

Requirements:
    pip install asyncio (built-in)

Usage:
    from examples.agent_orchestration import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    orchestrator.add_agent(ResearcherAgent(coherence_threshold=0.8))
    orchestrator.add_agent(CriticAgent(coherence_threshold=0.9))

    result = await orchestrator.process("Complex query here")
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from por_kernel.kernel import SilenceToken, por_kernel


# =============================================================================
# Agent Types and Results
# =============================================================================


class AgentVote(Enum):
    """Agent's vote on whether to proceed."""

    PROCEED = "proceed"
    SILENCE = "silence"
    ABSTAIN = "abstain"


@dataclass
class AgentResult:
    """Result from a single agent's processing."""

    agent_name: str
    vote: AgentVote
    content: Optional[str] = None
    coherence: float = 0.0
    drift: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def proceeded(self) -> bool:
        return self.vote == AgentVote.PROCEED


@dataclass
class OrchestratorResult:
    """Aggregated result from all agents."""

    proceeded: bool
    content: Optional[str] = None
    agent_results: list[AgentResult] = field(default_factory=list)
    consensus_score: float = 0.0
    reason: Optional[str] = None

    def __repr__(self) -> str:
        status = "PROCEED" if self.proceeded else "SILENCE"
        return (
            f"<{status}: consensus={self.consensus_score:.2f}, "
            f"agents={len(self.agent_results)}>")


# =============================================================================
# Base Agent Class
# =============================================================================


class BaseAgent(ABC):
    """
    Abstract base class for agents in the orchestration system.

    Each agent implements its own processing logic and applies
    PoR Kernel gating to its output.
    """

    def __init__(
        self,
        name: str,
        coherence_threshold: float = 0.7,
        drift_tolerance: float = 0.1,
        weight: float = 1.0,
    ) -> None:
        self.name = name
        self.coherence_threshold = coherence_threshold
        self.drift_tolerance = drift_tolerance
        self.weight = weight
        self.context: list[str] = []

    @abstractmethod
    async def process(self, query: str, shared_context: dict[str, list[str]]) -> str:
        """
        Process the query and return a response.

        Args:
            query: The input query
            shared_context: Context shared between agents

        Returns:
            The agent's response string
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_coherence(self, query: str, response: str) -> float:
        """Estimate coherence of the response."""
        raise NotImplementedError

    @abstractmethod
    def estimate_drift(self, query: str) -> float:
        """Estimate drift from original context."""
        raise NotImplementedError

    async def run(self, query: str, shared_context: dict[str, list[str]]) -> AgentResult:
        """
        Run the agent with PoR Kernel gating.

        Returns:
            AgentResult with vote and optional content
        """
        try:
            response = await self.process(query, shared_context)
        except Exception as exc:  # pragma: no cover - demo error path
            return AgentResult(
                agent_name=self.name,
                vote=AgentVote.ABSTAIN,
                metadata={"error": str(exc)},
            )

        coherence = self.estimate_coherence(query, response)
        drift = self.estimate_drift(query)

        # Apply PoR Kernel
        decision = por_kernel(
            drift=drift,
            coherence=coherence,
            tol=self.drift_tolerance,
            thresh=self.coherence_threshold,
        )

        if isinstance(decision, SilenceToken):
            return AgentResult(
                agent_name=self.name,
                vote=AgentVote.SILENCE,
                coherence=coherence,
                drift=drift,
                metadata={"raw_response": response},
            )

        self.context.append(response)
        return AgentResult(
            agent_name=self.name,
            vote=AgentVote.PROCEED,
            content=response,
            coherence=coherence,
            drift=drift,
        )


# =============================================================================
# Example Agent Implementations
# =============================================================================


class ResearcherAgent(BaseAgent):
    """
    Agent that researches and provides detailed information.
    Higher coherence threshold for accuracy.
    """

    def __init__(self, coherence_threshold: float = 0.8) -> None:
        super().__init__(
            name="researcher",
            coherence_threshold=coherence_threshold,
            drift_tolerance=0.15,
            weight=1.5,  # Higher weight for research output
        )

    async def process(self, query: str, shared_context: dict[str, list[str]]) -> str:
        # Simulate research processing
        await asyncio.sleep(0.1)  # Simulate API call

        # In production, this would call an LLM or knowledge base
        if "ai safety" in query.lower():
            return (
                "AI Safety research focuses on ensuring artificial intelligence "
                "systems behave as intended and remain beneficial. Key areas include:\n"
                "1. Alignment - ensuring AI goals match human values\n"
                "2. Robustness - handling edge cases and adversarial inputs\n"
                "3. Interpretability - understanding AI decision-making\n"
                "4. Control - maintaining human oversight of AI systems"
            )
        if "quantum" in query.lower():
            return (
                "Quantum computing leverages quantum mechanical phenomena "
                "to process information. Key concepts include superposition, "
                "entanglement, and quantum interference. Current applications "
                "focus on optimization, cryptography, and simulation."
            )
        return "I need more specific context to provide accurate research."

    def estimate_coherence(self, query: str, response: str) -> float:
        # Simple heuristic based on response quality indicators
        if len(response) < 50:
            return 0.4
        if "need more" in response.lower() or "not sure" in response.lower():
            return 0.5
        if any(marker in response for marker in ["1.", "2.", "-", "•"]):
            return 0.9  # Structured response
        return 0.75

    def estimate_drift(self, query: str) -> float:
        if not self.context:
            return 0.0
        # Check if query relates to previous context
        query_terms = set(query.lower().split())
        context_terms = set(" ".join(self.context).lower().split())
        if query_terms & context_terms:
            return 0.05
        return 0.2


class CriticAgent(BaseAgent):
    """
    Agent that evaluates and critiques responses.
    Very high coherence threshold to ensure quality feedback.
    """

    def __init__(self, coherence_threshold: float = 0.85) -> None:
        super().__init__(
            name="critic",
            coherence_threshold=coherence_threshold,
            drift_tolerance=0.1,
            weight=1.0,
        )

    async def process(self, query: str, shared_context: dict[str, list[str]]) -> str:
        await asyncio.sleep(0.05)

        # Evaluate content from other agents
        other_responses = shared_context.get("responses", [])
        if not other_responses:
            return "No content to evaluate."

        # Simulate critique
        total_length = sum(len(response) for response in other_responses)
        if total_length > 200:
            return (
                "The provided research appears comprehensive and well-structured.\n"
                "Key points are clearly articulated with supporting details.\n"
                "Recommendation: ACCEPT with minor clarifications on scope."
            )
        return "The response lacks sufficient detail for evaluation."

    def estimate_coherence(self, query: str, response: str) -> float:
        if "ACCEPT" in response:
            return 0.95
        if "lacks" in response.lower():
            return 0.6
        return 0.7

    def estimate_drift(self, query: str) -> float:
        return 0.05  # Critic stays focused on evaluation task


class SynthesizerAgent(BaseAgent):
    """
    Agent that synthesizes multiple inputs into a coherent output.
    Balanced thresholds for flexibility and quality.
    """

    def __init__(self, coherence_threshold: float = 0.75) -> None:
        super().__init__(
            name="synthesizer",
            coherence_threshold=coherence_threshold,
            drift_tolerance=0.2,
            weight=1.2,
        )

    async def process(self, query: str, shared_context: dict[str, list[str]]) -> str:
        await asyncio.sleep(0.05)

        responses = shared_context.get("responses", [])
        critiques = shared_context.get("critiques", [])

        if not responses:
            return "Nothing to synthesize."

        # Combine and synthesize
        synthesis = f"Based on {len(responses)} research inputs"
        if critiques:
            synthesis += f" and {len(critiques)} evaluations"
        synthesis += ", the synthesized response addresses the key aspects of the query."

        return synthesis

    def estimate_coherence(self, query: str, response: str) -> float:
        if "Nothing to" in response:
            return 0.4
        return 0.8

    def estimate_drift(self, query: str) -> float:
        return len(self.context) * 0.03  # Gradual drift with conversation length


# =============================================================================
# Agent Orchestrator
# =============================================================================


class AgentOrchestrator:
    """
    Orchestrates multiple agents with consensus-based silence protocol.

    The orchestrator:
    1. Runs all agents in parallel
    2. Collects their votes (PROCEED/SILENCE/ABSTAIN)
    3. Applies consensus rules to determine final action
    4. Uses PoR Kernel for the final aggregated decision

    Example:
        >>> orchestrator = AgentOrchestrator(consensus_threshold=0.6)
        >>> orchestrator.add_agent(ResearcherAgent())
        >>> orchestrator.add_agent(CriticAgent())
        >>>
        >>> result = await orchestrator.process("What is AI safety?")
        >>> if result.proceeded:
        ...     print(result.content)
        ... else:
        ...     print(f"Silenced: {result.reason}")
    """

    def __init__(self, consensus_threshold: float = 0.5, require_unanimous: bool = False) -> None:
        """
        Initialize the orchestrator.

        Args:
            consensus_threshold: Minimum weighted vote ratio to proceed
            require_unanimous: If True, all agents must vote PROCEED
        """
        self.agents: list[BaseAgent] = []
        self.consensus_threshold = consensus_threshold
        self.require_unanimous = require_unanimous

    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the orchestration."""
        self.agents.append(agent)

    def remove_agent(self, name: str) -> None:
        """Remove an agent by name."""
        self.agents = [agent for agent in self.agents if agent.name != name]

    async def process(self, query: str) -> OrchestratorResult:
        """
        Process a query through all agents.

        Args:
            query: The input query

        Returns:
            OrchestratorResult with aggregated decision and content
        """
        if not self.agents:
            return OrchestratorResult(
                proceeded=False,
                reason="No agents configured",
            )

        shared_context: dict[str, list[str]] = {
            "query": [query],
            "responses": [],
            "critiques": [],
        }

        # Run all agents in parallel
        tasks = [agent.run(query, shared_context) for agent in self.agents]
        results = await asyncio.gather(*tasks)

        # Aggregate responses for context
        for result in results:
            if result.proceeded and result.content:
                if "critic" in result.agent_name.lower():
                    shared_context["critiques"].append(result.content)
                else:
                    shared_context["responses"].append(result.content)

        # Calculate consensus
        consensus_score = self._calculate_consensus(results)

        # Determine final decision
        if self.require_unanimous:
            all_proceed = all(result.vote != AgentVote.SILENCE for result in results)
            proceeded = all_proceed
            if not proceeded:
                silencing_agents = [
                    result.agent_name for result in results if result.vote == AgentVote.SILENCE
                ]
                reason = (
                    "Unanimous consent failed: "
                    f"{', '.join(silencing_agents)} voted SILENCE"
                )
            else:
                reason = None
        else:
            proceeded = consensus_score >= self.consensus_threshold
            if not proceeded:
                reason = (
                    f"Consensus {consensus_score:.2f} below threshold "
                    f"{self.consensus_threshold}"
                )
            else:
                reason = None

        # Aggregate content from proceeding agents
        contents = [result.content for result in results if result.proceeded and result.content]
        final_content = "\n\n---\n\n".join(contents) if contents else None

        return OrchestratorResult(
            proceeded=proceeded,
            content=final_content,
            agent_results=results,
            consensus_score=consensus_score,
            reason=reason if not proceeded else None,
        )

    def _calculate_consensus(self, results: list[AgentResult]) -> float:
        """Calculate weighted consensus score."""
        total_weight = 0.0
        proceed_weight = 0.0

        for agent, result in zip(self.agents, results, strict=True):
            if result.vote != AgentVote.ABSTAIN:
                total_weight += agent.weight
                if result.vote == AgentVote.PROCEED:
                    proceed_weight += agent.weight

        if total_weight == 0:
            return 0.0

        return proceed_weight / total_weight


# =============================================================================
# Example Usage
# =============================================================================


async def main() -> None:
    print("Multi-Agent Orchestration with PoR Kernel")
    print("=" * 50)

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        consensus_threshold=0.5,
        require_unanimous=False,
    )

    # Add agents with different thresholds
    orchestrator.add_agent(ResearcherAgent(coherence_threshold=0.75))
    orchestrator.add_agent(CriticAgent(coherence_threshold=0.85))
    orchestrator.add_agent(SynthesizerAgent(coherence_threshold=0.7))

    # Test 1: Clear query - should PROCEED
    print("\n[Test 1] Query: 'What is AI safety?'")
    result = await orchestrator.process("What is AI safety?")
    print(f"  Result: {result}")
    print(f"  Consensus: {result.consensus_score:.2f}")
    for agent_result in result.agent_results:
        print(
            "    - "
            f"{agent_result.agent_name}: {agent_result.vote.value} "
            f"(coh={agent_result.coherence:.2f})"
        )

    # Test 2: Vague query - might trigger silence
    print("\n[Test 2] Query: 'Tell me about stuff'")
    result = await orchestrator.process("Tell me about stuff")
    print(f"  Result: {result}")
    print(f"  Consensus: {result.consensus_score:.2f}")
    for agent_result in result.agent_results:
        print(
            "    - "
            f"{agent_result.agent_name}: {agent_result.vote.value} "
            f"(coh={agent_result.coherence:.2f})"
        )

    # Test 3: Specific technical query
    print("\n[Test 3] Query: 'Explain quantum computing applications'")
    result = await orchestrator.process("Explain quantum computing applications")
    print(f"  Result: {result}")
    if result.proceeded and result.content:
        print(f"  Content preview: {result.content[:150]}...")

    print("\n" + "=" * 50)
    print("Integration complete.")


if __name__ == "__main__":
    asyncio.run(main())
