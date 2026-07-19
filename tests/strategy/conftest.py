"""Shared fixtures/helpers for strategy tests."""

from agents.strategy.characterization import RegionRecord
from agents.strategy.models import RegionTarget


def make_region(integral, file, line, signal_class, *, max_cond=1.0,
                max_rel_err=1e-16, pred_float=1e-16, pred_ff=None, op_count=1,
                variables=None):
    """Build a RegionRecord without going through report JSON.

    ``pred_ff`` defaults to ``pred_float`` (the conservative float-fallback the
    report loader applies to reports predating the ff signal), so existing tests
    that set only ``pred_float`` keep their admission behavior.
    """
    return RegionRecord(
        integral=integral,
        target=RegionTarget(file=file, line_start=line, line_end=line,
                            variables=list(variables or ["x"])),
        signal_class=signal_class,
        max_cond=max_cond,
        max_rel_err=max_rel_err,
        predicted_rel_err_if_float=pred_float,
        predicted_rel_err_if_ff=pred_float if pred_ff is None else pred_ff,
        op_count=op_count,
        n=100,
    )
