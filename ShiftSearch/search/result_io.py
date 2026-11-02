"""Incremental search output; never retain the complete set of PSM records."""
from __future__ import annotations

import csv
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from ShiftSearch.search.fdr import compute_psm_q_values


RESULT_COLUMNS = (
    "spectrum_index", "title", "charge", "pre_mz", "sequence",
    "search_result", "target_sequence", "neutral_mass", "ppm_error",
    "score", "logit", "protein_id", "is_decoy", "rank",
)


def check_new_output_paths(*paths: Path) -> None:
    """A new search must not truncate an existing or partially completed run."""
    for path in paths:
        if path.exists():
            raise FileExistsError(
                f"Search output already exists: {path}. Use a new --output name; "
                "automatic resume/overwrite is not supported."
            )


class StreamingSearchWriter:
    """Write each spectrum immediately, and sync both files every N spectra.

    Only the standard file buffers are retained. A Python exception (including
    Ctrl+C) closes and flushes both files. A forced process kill/power loss can
    lose the not-yet-synced tail; this is not a transactional resume mechanism.
    """

    def __init__(self, output_path: Path, flush_every: int = 1000):
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.output_path = Path(output_path)
        self.best_path = self.output_path.with_name(self.output_path.stem + ".best.tsv")
        self.flush_every = flush_every
        self.n_spectra = 0
        self.n_rows = 0
        self.n_best = 0

    def __enter__(self):
        check_new_output_paths(self.output_path, self.best_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stack = ExitStack()
        try:
            self._files = [
                self._stack.enter_context(path.open("x", encoding="utf-8", newline=""))
                for path in (self.output_path, self.best_path)
            ]
            self._writers = [
                csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t", lineterminator="\n")
                for handle in self._files
            ]
            for writer in self._writers:
                writer.writeheader()
            self.flush()
        except BaseException:
            self._stack.close()
            raise
        return self

    def write_spectrum(self, ranked_rows, top_k: int):
        for rank, row in enumerate(ranked_rows[:top_k], start=1):
            record = dict(row, rank=rank)
            self._writers[0].writerow(record)
            self.n_rows += 1
            if rank == 1:
                self._writers[1].writerow(record)
                self.n_best += 1
        self.n_spectra += 1  # Includes spectra without a reported candidate.
        if self.n_spectra % self.flush_every == 0:
            self.flush()

    def flush(self):
        for handle in self._files:
            handle.flush()
            os.fsync(handle.fileno())

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.flush()
        finally:
            self._stack.close()
        return False


def write_fdr_filtered_best(best_path: Path, output_path: Path, threshold: float) -> int:
    """Global legacy D/T q-values, without loading PSM strings into memory.

    FDR still needs O(number of best hits) numeric memory for sorting. Full
    records are streamed from the completed best TSV in their original order.
    Do not compute independent FDRs for output batches.
    """
    check_new_output_paths(output_path)
    numeric = pd.read_csv(
        best_path, sep="\t", usecols=["score", "is_decoy"],
        dtype={"score": np.float64, "is_decoy": bool}, float_precision="round_trip",
    )
    numeric["_row_number"] = np.arange(len(numeric), dtype=np.int64)
    ordered = compute_psm_q_values(numeric)
    q_values = np.empty(len(numeric), dtype=np.float64)
    q_values[ordered["_row_number"].to_numpy()] = ordered["q_value"].to_numpy()
    del ordered, numeric

    kept = 0
    with best_path.open("r", encoding="utf-8", newline="") as inp, output_path.open(
        "x", encoding="utf-8", newline=""
    ) as out:
        reader = csv.DictReader(inp, delimiter="\t")
        writer = csv.DictWriter(
            out, fieldnames=[*RESULT_COLUMNS, "q_value"], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        seen = 0
        for index, row in enumerate(reader):
            seen += 1
            if q_values[index] <= threshold:
                row["q_value"] = q_values[index]
                writer.writerow(row)
                kept += 1
        if seen != len(q_values):
            raise RuntimeError("Best-hit TSV changed during FDR processing")
        out.flush()
        os.fsync(out.fileno())
    return kept
