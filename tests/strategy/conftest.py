"""Shared fixtures/helpers for strategy tests."""

from agents.strategy.characterization import RegionRecord
from agents.strategy.models import RegionTarget


def make_region(integral, file, line, signal_class, *, max_cond=1.0,
                max_rel_err=1e-16, pred_float=1e-16, op_count=1, variables=None):
    """Build a RegionRecord without going through report JSON."""
    return RegionRecord(
        integral=integral,
        target=RegionTarget(file=file, line_start=line, line_end=line,
                            variables=list(variables or ["x"])),
        signal_class=signal_class,
        max_cond=max_cond,
        max_rel_err=max_rel_err,
        predicted_rel_err_if_float=pred_float,
        op_count=op_count,
        n=100,
    )
