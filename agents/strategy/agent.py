"""Strategy agent — stub for v1 vertical slice.

In v1 the graph always exits after characterization.  The real strategy agent
will inspect the sensitivity profile and populate strategy_queue with patch
proposals; until then this returns an empty queue so the conditional edge
routes to END.
"""

from agents.state import PipelineState


def run(state: PipelineState) -> dict:
    # single writer in v1; if you parallelize the strategy loop, add a reducer
    return {"strategy_queue": []}
