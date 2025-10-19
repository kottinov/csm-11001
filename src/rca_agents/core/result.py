"""Structured result types for agent coordination."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AgentResult:
    """Result from a LATS agent investigation with coordination metadata.

    This enables reflection-based multi-agent coordination as described
    in the LATS-RCA thesis: agents use reflection scores to assess their
    confidence and determine whether they need collaboration from other agents.
    """

    agent_name: str
    """Which agent produced this result."""

    summary: str
    """Human-readable conclusion from the investigation."""

    confidence: float
    """Overall confidence from reflection.normalized_score (0.0 to 1.0)."""

    evidence: List[str]
    """Key evidence pointers (file paths, metric names, etc.)."""

    reflections: str
    """The agent's self-assessment reasoning."""

    needs_collaboration: bool
    """Whether this agent believes it needs help from other agents."""

    suggested_next_agent: Optional[str] = None
    """Which agent should investigate next, if any."""

    confidence_breakdown: Optional[dict] = None
    """Detailed confidence components for coordination decisions."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "summary": self.summary,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reflections": self.reflections,
            "needs_collaboration": self.needs_collaboration,
            "suggested_next_agent": self.suggested_next_agent,
            "confidence_breakdown": self.confidence_breakdown,
        }

    @property
    def is_high_confidence(self) -> bool:
        """Check if this result has high confidence (>= 0.7)."""
        return self.confidence >= 0.7

    @property
    def is_complete(self) -> bool:
        """Check if this result is complete enough to stand alone."""
        if self.confidence_breakdown:
            completeness = self.confidence_breakdown.get("completeness", 0.0)
            return completeness >= 0.7
        return self.confidence >= 0.7


__all__ = ["AgentResult"]
