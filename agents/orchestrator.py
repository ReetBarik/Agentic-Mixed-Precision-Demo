"""LangGraph orchestrator — wires the top-level agents.

Topology (locked with the Strategy/Patcher design, docs/strategy_patcher_design.md
Q5): Strategy owns the whole remediation loop internally — it drives the Patcher
and Validator as callables on the state, not as separate graph nodes.  So the
graph is a straight line:

    characterize → strategy → END

The Patcher and Validator are no longer graph nodes; they are injected into the
state as ``patcher_fn`` / ``validator_fn`` (see ``agents/patcher/agent.py`` and
``agents/validator/agent.py`` for the adapters that build them).  The old
strategy→patcher→validate→patcher loop edges were vestigial once Strategy took
ownership of the loop, and have been removed.
"""

from langgraph.graph import END, StateGraph

from agents.characterizer import agent as characterizer_agent
from agents.strategy import agent as strategy_agent
from agents.state import PipelineState


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("characterize", characterizer_agent.run)
    g.add_node("strategy", strategy_agent.run)

    g.set_entry_point("characterize")
    g.add_edge("characterize", "strategy")
    g.add_edge("strategy", END)

    return g.compile()
