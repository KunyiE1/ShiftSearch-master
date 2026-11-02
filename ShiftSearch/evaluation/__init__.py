"""Evaluation procedures used for the paper; independent of legacy search FDR."""

from .fdr import add_tie_aware_q_values

__all__ = ["add_tie_aware_q_values"]
