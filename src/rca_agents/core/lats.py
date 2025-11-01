from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
from typing import Iterable, List, Optional, Sequence, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain as as_runnable
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError


class Reflection(BaseModel):
    reflections: str = Field(
        description="Reflections on the sufficiency and quality of the candidate response."
    )
    score: int = Field(
        description="Score from 0-10 on candidate quality.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        default=False,
        description="Whether the candidate fully answered the task."
    )
    evidence_strength: Optional[float] = Field(
        default=None,
        description="Quality and relevance of evidence gathered (0.0-1.0).",
        ge=0.0,
        le=1.0,
    )
    completeness: Optional[float] = Field(
        default=None,
        description="How fully the investigation covered the problem space (0.0-1.0).",
        ge=0.0,
        le=1.0,
    )
    consistency: Optional[float] = Field(
        default=None,
        description="Internal consistency of findings and reasoning (0.0-1.0).",
        ge=0.0,
        le=1.0,
    )

    def as_message(self) -> HumanMessage:
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0

    @property
    def confidence_breakdown(self) -> dict:
        """Return structured confidence components for coordination decisions."""
        return {
            "evidence_strength": self.evidence_strength or self.normalized_score,
            "completeness": self.completeness or self.normalized_score,
            "consistency": self.consistency or self.normalized_score,
        }

    @property
    def needs_escalation(self) -> bool:
        """Determine if agent should request help from other agents.

        Escalation is needed if:
        - Overall confidence is low (< 0.7)
        - Completeness is low (< 0.6) even if confidence is moderate
        """
        if self.normalized_score < 0.7:
            return True
        if self.completeness is not None and self.completeness < 0.6:
            return True
        return False


class Node:
    def __init__(
        self,
        messages: List[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
        *,
        action_signature: Optional[str] = None,
        reward_override: Optional[float] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children: List["Node"] = []
        self.value = 0.0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent else 1
        self.action_signature = action_signature
        self._is_solved = bool(reflection and reflection.found_solution)
        if self._is_solved:
            self._mark_tree_as_solved()
        self._last_reward = reward_override if reward_override is not None else reflection.normalized_score
        self.backpropagate(self._last_reward)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value:.3f}, visits={self.visits}, "
            f"solved={self._is_solved}, depth={self.depth}>"
        )

    @property
    def is_solved(self) -> bool:
        return self._is_solved

    @property
    def is_terminal(self) -> bool:
        return not self.children

    @property
    def height(self) -> int:
        if self.children:
            return 1 + max(child.height for child in self.children)
        return 1

    def upper_confidence_bound(self, exploration_weight: float = 1.0) -> float:
        if self.parent is None:
            raise ValueError("Cannot compute UCB for root node.")
        if self.visits == 0:
            return self.value
        average_reward = self.value / self.visits
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float) -> None:
        node: Optional["Node"] = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflection: bool = True) -> List[BaseMessage]:
        if include_reflection:
            return self.messages + [self.reflection.as_message()]
        return list(self.messages)

    def get_trajectory(self, include_reflections: bool = True) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        node: Optional["Node"] = self
        while node:
            chunk = node.get_messages(include_reflection=include_reflections)
            messages.extend(reversed(chunk))
            node = node.parent
        return list(reversed(messages))

    def _mark_tree_as_solved(self) -> None:
        node = self.parent
        while node:
            node._is_solved = True
            node = node.parent

    def _collect_all_nodes(self) -> Iterable["Node"]:
        nodes = deque([self])
        while nodes:
            node = nodes.popleft()
            yield node
            nodes.extend(node.children)

    def best_solution(self) -> "Node":
        candidates = list(self._collect_all_nodes())
        return max(
            candidates,
            key=lambda node: (node.is_terminal and node.is_solved, node.value),
        )


class TreeState(TypedDict):
    root: Node
    input: str
    iterations: int


@dataclass(frozen=True)
class LATSAgentSpec:
    name: str
    llm: ChatAnthropic
    tools: Sequence
    initial_prompt: ChatPromptTemplate
    reflection_prompt: ChatPromptTemplate
    max_height: int
    max_iterations: int = 50
    candidate_batch_size: int = 5
    temperature: float = 0.0
    exploration_weight: float = 1.0
    self_consistency_weight: float = 0.5


def _build_reflection_chain(spec: LATSAgentSpec):
    reflection_llm_chain = (
        spec.reflection_prompt
        | spec.llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
            run_name=f"{spec.name}.Reflection"
        )
        | PydanticToolsParser(tools=[Reflection])
    )

    @as_runnable
    def reflection_chain(inputs) -> Reflection:
        try:
            tool_choices = reflection_llm_chain.invoke(inputs)
            reflection = tool_choices[0]
            if not isinstance(inputs["candidate"][-1], AIMessage):
                reflection.found_solution = False
            return reflection
        except (ValidationError, IndexError) as exc:
            return Reflection(
                reflections=f"Reflection incomplete due to error: {type(exc).__name__}. "
                           "Candidate response likely too verbose or complex.",
                score=3,
                found_solution=False,
            )

    return reflection_chain


def build_agent_graph(spec: LATSAgentSpec):
    """Compile a LangGraph state machine for a LATS agent."""
    tool_node = ToolNode(tools=list(spec.tools))
    parser = JsonOutputToolsParser(return_id=True)
    reflection_chain = _build_reflection_chain(spec)

    initial_answer_chain = spec.initial_prompt | spec.llm.bind_tools(
        tools=list(spec.tools),
        temperature=spec.temperature,
    ).with_config(run_name=f"{spec.name}.InitialCandidate")

    def generate_initial_response(state: TreeState) -> TreeState:
        res = initial_answer_chain.invoke({"input": state["input"]})
        parsed = parser.invoke(res)

        tool_responses = []
        for tool_call in parsed:
            tool_responses.append(
                tool_node.invoke(
                    {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": tool_call["type"],
                                        "args": tool_call["args"],
                                        "id": tool_call["id"],
                                    }
                                ],
                            )
                        ]
                    }
                )
            )

        output_messages: List[BaseMessage] = [res]
        for response in tool_responses:
            output_messages.append(response["messages"][0])

        action_signature = _extract_action_signature(res)

        reflection = reflection_chain.invoke(
            {"input": state["input"], "candidate": output_messages}
        )
        reward = _combine_reward(
            reflection.normalized_score,
            1.0,
            spec.self_consistency_weight,
        )
        root = Node(
            output_messages,
            reflection=reflection,
            action_signature=action_signature,
            reward_override=reward,
        )
        return {"root": root, "input": state["input"], "iterations": 1}

    def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
        batch_size = config.get("configurable", {}).get(
            "N", spec.candidate_batch_size
        )
        bound_llm = spec.llm.bind_tools(
            tools=list(spec.tools), temperature=spec.temperature
        )
        candidates = []
        for index in range(batch_size):
            response = bound_llm.invoke(
                messages.to_messages(),
                config={
                    "callbacks": config.get("callbacks"),
                    "run_name": f"{spec.name}.Candidate_{index}",
                },
            )
            candidates.append(response)
        return candidates

    expansion_chain = spec.initial_prompt | generate_candidates

    def select(node: Node) -> Node:
        current = node
        while current.children:
            current = max(
                current.children,
                key=lambda child: child.upper_confidence_bound(
                    exploration_weight=spec.exploration_weight
                ),
            )
        return current

    def expand(state: TreeState, config: RunnableConfig) -> TreeState:
        root = state["root"]
        best_candidate = select(root)
        trajectory = best_candidate.get_trajectory()

        new_candidates = expansion_chain.invoke(
            {"input": state["input"], "messages": trajectory}, config
        )
        parsed_calls = parser.batch(new_candidates)

        collected_responses: dict[int, List[BaseMessage]] = defaultdict(list)
        for index, tool_calls in enumerate(parsed_calls):
            for tool_call in tool_calls:
                tool_response = tool_node.invoke(
                    {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": tool_call["type"],
                                        "args": tool_call["args"],
                                        "id": tool_call["id"],
                                    }
                                ],
                            )
                        ]
                    }
                )
                collected_responses[index].append(tool_response["messages"][0])

        candidate_messages = []
        for index, candidate in enumerate(new_candidates):
            candidate_messages.append([candidate] + collected_responses[index])

        signatures: List[str] = [
            _extract_action_signature(messages[0])
            for messages in candidate_messages
        ]
        signature_counts = Counter(signatures)
        batch_size = len(candidate_messages)

        reflections = []
        for messages in candidate_messages:
            try:
                reflection = reflection_chain.invoke(
                    {"input": state["input"], "candidate": messages},
                    config,
                )
                reflections.append(reflection)
            except (ValidationError, IndexError, Exception) as exc:
                reflections.append(
                    Reflection(
                        reflections=f"Reflection failed: {type(exc).__name__}",
                        score=3,
                        found_solution=False,
                    )
                )

        children: List[Node] = []
        for messages, reflection, signature in zip(
            candidate_messages, reflections, signatures
        ):
            consistency = (
                signature_counts[signature] / batch_size if batch_size else 0.0
            )
            reward = _combine_reward(
                reflection.normalized_score,
                consistency,
                spec.self_consistency_weight,
            )
            child = Node(
                messages,
                parent=best_candidate,
                reflection=reflection,
                action_signature=signature,
                reward_override=reward,
            )
            children.append(child)

        best_candidate.children.extend(children)
        state["iterations"] += 1
        return state

    def should_continue(state: TreeState):
        root = state["root"]
        if root.is_solved:
            return END
        if root.height > spec.max_height:
            return END
        if state.get("iterations", 0) >= spec.max_iterations:
            return END
        return "expand"

    builder = StateGraph(TreeState)
    builder.add_node("start", generate_initial_response)
    builder.add_node("expand", expand)
    builder.add_edge(START, "start")
    builder.add_conditional_edges("start", should_continue, ["expand", END])
    builder.add_conditional_edges("expand", should_continue, ["expand", END])
    return builder.compile()


def _extract_action_signature(ai_message: BaseMessage) -> str:
    """Identify the action signature for self-consistency calculations."""
    if isinstance(ai_message, AIMessage):
        tool_calls = getattr(ai_message, "tool_calls", None)
        if tool_calls:
            serialized_calls = []
            for call in tool_calls:
                name = call.get("name", "")
                try:
                    args = json.dumps(call.get("args", {}), sort_keys=True)
                except (TypeError, ValueError):
                    args = str(call.get("args", {}))
                serialized_calls.append(f"{name}:{args}")
            return "|".join(serialized_calls)
        content = getattr(ai_message, "content", "")
        if isinstance(content, str):
            return content.strip()[:2000]
    return repr(ai_message)[:2000]


def _combine_reward(
    reflection_score: float, consistency: float, weight: float
) -> float:
    """Blend reflection score with self-consistency weighting."""
    weight = max(0.0, min(1.0, weight))
    return weight * reflection_score + (1 - weight) * consistency


__all__ = ["Reflection", "Node", "TreeState", "LATSAgentSpec", "build_agent_graph"]
