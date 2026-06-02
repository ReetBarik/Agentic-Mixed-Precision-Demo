"""LangGraph subgraph for the downcast proposer.

Pure proposer: given a variable, call the LLM once and return a PatchProposal
(or propose_error). The orchestrator owns the per-variable retry loop, policy
check, build, and verify.

Flow:
    propose → END
"""

from langgraph.graph import END, StateGraph

from llm_agent import config
from llm_agent.client import make_client
from llm_agent.skills.downcast.prompts import (
    PROPOSE_PATCH_TOOL,
    SYSTEM_PROMPT,
    build_initial_user_message,
)
from llm_agent.state import DowncastProposerState, PatchProposal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_to_dict(block) -> dict:
    """Convert an Anthropic SDK content block to a plain dict for state storage."""
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": dict(block.input),
        }
    if block.type == "text":
        return {"type": "text", "text": block.text}
    return {"type": block.type}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def propose(state: DowncastProposerState) -> dict:
    """Call the LLM once to get a patch proposal for the current variable.

    If messages is empty this is the first attempt; build the initial user
    turn.  If messages is non-empty the orchestrator has already appended a
    tool-result feedback message, so we continue the conversation.
    """
    var = state["current_variable"]
    messages = list(state["messages"])

    if not messages:
        initial_msg = build_initial_user_message(
            spec=state["spec"],
            impl_source=state["impl_source"],
            current_variable=var,
            accepted_patches=state["accepted_patches"],
            accepted_variables=state["accepted_variables"],
            min_digits=state["min_digits"],
        )
        messages = [initial_msg]

    try:
        client = make_client(base_url=state.get("base_url"))
        response = client.messages.create(
            model=config.DEFAULT_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[PROPOSE_PATCH_TOOL],
            tool_choice={"type": "tool", "name": "propose_patch"},
            messages=messages,
        )
    except Exception as exc:
        return {
            "messages": messages,
            "propose_error": "API error: {0}".format(str(exc)[:400]),
            "proposal": None,
            "current_tool_use_id": None,
        }

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        return {
            "messages": messages,
            "propose_error": "LLM did not return a tool call",
            "proposal": None,
            "current_tool_use_id": None,
        }

    proposal = PatchProposal(
        file_path=tool_use_block.input.get("file_path", ""),
        old_line=tool_use_block.input.get("old_line", ""),
        new_line=tool_use_block.input.get("new_line", ""),
        reasoning=tool_use_block.input.get("reasoning", ""),
    )

    assistant_msg = {
        "role": "assistant",
        "content": [_block_to_dict(b) for b in response.content],
    }

    return {
        "proposal": proposal,
        "current_tool_use_id": tool_use_block.id,
        "messages": messages + [assistant_msg],
        "propose_error": None,
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_downcast_graph():
    g = StateGraph(DowncastProposerState)
    g.add_node("propose", propose)
    g.set_entry_point("propose")
    g.add_edge("propose", END)
    return g.compile()
