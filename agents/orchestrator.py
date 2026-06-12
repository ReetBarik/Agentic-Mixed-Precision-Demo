"""LangGraph orchestrator — wires the four top-level agents.

v1 topology:
    characterize → strategy → (queue empty?) END
                                    ↓
                              patcher → validate → (queue empty?) END
                                                        ↓
                                                    patcher  (loop)

In this slice strategy always returns an empty queue, so the graph exits
immediately after characterization.
"""

from langgraph.graph import END, StateGraph

from agents.characterizer import agent as characterizer_agent
from agents.patcher import agent as patcher_agent
from agents.strategy import agent as strategy_agent
from agents.validator import agent as validator_agent
from agents.state import PipelineState


def _route_after_strategy(state: PipelineState) -> str:
    return END if not state.get("strategy_queue") else "patcher"


def _route_after_validate(state: PipelineState) -> str:
    return END if not state.get("strategy_queue") else "patcher"


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("characterize", characterizer_agent.run)
    g.add_node("strategy", strategy_agent.run)
    g.add_node("patcher", patcher_agent.run)
    g.add_node("validate", validator_agent.run)

    g.set_entry_point("characterize")
    g.add_edge("characterize", "strategy")
    g.add_conditional_edges("strategy", _route_after_strategy)
    g.add_edge("patcher", "validate")
    g.add_conditional_edges("validate", _route_after_validate)

    return g.compile()
