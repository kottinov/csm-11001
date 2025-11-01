from langchain_core.messages import AIMessage

from rca_agents.core.lats import _combine_reward, _extract_action_signature, Reflection, Node


def test_combine_reward_balances_components():
    assert _combine_reward(1.0, 0.0, 1.0) == 1.0
    assert _combine_reward(0.0, 1.0, 0.0) == 1.0
    blended = _combine_reward(0.8, 0.2, 0.5)
    assert abs(blended - 0.5) < 1e-6


def test_extract_action_signature_prefers_tool_calls():
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "search_directory", "args": {"keyword": "error"}}],
    )
    signature = _extract_action_signature(ai_message)
    assert "search_directory" in signature
    assert "keyword" in signature


def test_node_uses_reward_override():
    reflection = Reflection(reflections="ok", score=6, found_solution=False)
    node = Node(
        messages=[AIMessage(content="candidate")],
        reflection=reflection,
        reward_override=0.9,
    )
    assert abs(node.value - 0.9) < 1e-6
