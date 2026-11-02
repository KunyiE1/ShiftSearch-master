from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ShiftSearch.search.peptide_utils import (
    add_shift_to_sequence,
    compute_neutral_mass,
    digest_tryptic,
    enumerate_mod_site_combinations,
    has_unsupported_residue,
    parse_fasta,
)

# Lightweight tuple for in-memory candidate storage
# (sequence, search_result, neutral_mass, protein_id, is_decoy)
CandidateMeta = Tuple[str, str, float, str, bool]


@dataclass(frozen=True)
class CandidatePeptide:
    sequence: str
    search_result: str
    prefix_masses: np.ndarray
    neutral_mass: float
    protein_id: str
    is_decoy: bool


def _enumerate_one_peptide(
    peptide: str,
    protein_id: str,
    fixed_mods: Optional[Dict[str, List[float]]],
    variable_mods: Optional[Dict[str, List[float]]],
    label_mods: Optional[Dict[str, List[float]]],
    min_len: int,
    max_len: int,
    max_variable_mods: int,
    decoy_prefix: str,
) -> List[CandidateMeta]:
    if not (min_len <= len(peptide) <= max_len):
        return []
    if has_unsupported_residue(peptide):
        return []

    is_decoy = str(protein_id).startswith(decoy_prefix)

    out: List[CandidateMeta] = []
    label_mods = label_mods or {}

    for _, nterm_shift, residue_shifts in enumerate_mod_site_combinations(
        peptide,
        fixed_mods=fixed_mods,
        variable_mods=variable_mods,
        max_variable_mods=max_variable_mods,
    ):
        # Every fixed/variable-modification state always contributes its light
        # candidate.  A configured isotope label contributes one additional
        # candidate in which every matching residue is labeled simultaneously.
        # Label shifts are global and do not consume max_variable_mods slots.
        label_states = [(nterm_shift, list(residue_shifts))]

        labeled_nterm_shift = nterm_shift + sum(label_mods.get("NTERM", []))
        labeled_residue_shifts = list(residue_shifts)
        label_applied = labeled_nterm_shift != nterm_shift
        for idx, aa in enumerate(peptide):
            residue_label_shift = sum(label_mods.get(aa, []))
            if residue_label_shift != 0.0:
                labeled_residue_shifts[idx] += residue_label_shift
                label_applied = True

        # Peptides without a configured label site keep only the light state,
        # preventing an identical duplicate candidate.
        if label_applied:
            label_states.append((labeled_nterm_shift, labeled_residue_shifts))

        for state_nterm_shift, state_residue_shifts in label_states:
            search_result = add_shift_to_sequence(
                peptide,
                state_residue_shifts,
                nterm_shift=state_nterm_shift,
            )
            neutral_mass = compute_neutral_mass(
                peptide,
                state_nterm_shift,
                state_residue_shifts,
            )
            out.append((
                peptide,
                search_result,
                neutral_mass,
                protein_id,
                is_decoy,
            ))
    return out


def enumerate_peptides_from_fasta(
    fasta_path: str | Path,
    fixed_mods: Optional[Dict[str, List[float]]],
    variable_mods: Optional[Dict[str, List[float]]],
    missed_cleavages: int,
    min_len: int,
    max_len: int,
    max_variable_mods: int,
    decoy_prefix: str,
    max_pep_len: int,
    label_mods: Optional[Dict[str, List[float]]] = None,
) -> Iterable[CandidateMeta]:
    for protein_id, protein_seq in tqdm(parse_fasta(fasta_path), desc="digest proteins", unit="protein"):
        for peptide in digest_tryptic(protein_seq, missed_cleavages=missed_cleavages, min_len=min_len, max_len=max_len):
            yield from _enumerate_one_peptide(
                peptide=peptide,
                protein_id=protein_id,
                fixed_mods=fixed_mods,
                variable_mods=variable_mods,
                label_mods=label_mods,
                min_len=min_len,
                max_len=max_len,
                max_variable_mods=max_variable_mods,
                decoy_prefix=decoy_prefix,
            )


def _iter_peptide_db_rows(peptide_db_path: str | Path):
    for header, sequence in parse_fasta(peptide_db_path):
        yield sequence, header


def enumerate_peptides_from_peptide_db(
    peptide_db_path: str | Path,
    fixed_mods: Optional[Dict[str, List[float]]],
    variable_mods: Optional[Dict[str, List[float]]],
    min_len: int,
    max_len: int,
    max_variable_mods: int,
    decoy_prefix: str,
    max_pep_len: int,
    label_mods: Optional[Dict[str, List[float]]] = None,
) -> Iterable[CandidateMeta]:
    for peptide, protein_id in tqdm(_iter_peptide_db_rows(peptide_db_path), desc="read peptide DB", unit="peptide"):
        yield from _enumerate_one_peptide(
            peptide=peptide,
            protein_id=protein_id,
            fixed_mods=fixed_mods,
            variable_mods=variable_mods,
            label_mods=label_mods,
            min_len=min_len,
            max_len=max_len,
            max_variable_mods=max_variable_mods,
            decoy_prefix=decoy_prefix,
        )


def _bucket_key(neutral_mass: float) -> str:
    """Bucket key in 0.1 Da resolution (matches DeepSearch)."""
    return f"{round(neutral_mass, 1):06.1f}"


def build_candidate_bucket_index(
    candidate_iter: Iterable[CandidateMeta],
) -> Tuple[List[CandidateMeta], Dict[str, List[int]]]:
    """Build in-memory candidate list + 0.1-Da bucket index.

    Returns:
        metas: list of candidate tuples, index = meta index
        buckets: bucket_str -> list of meta indices in the bucket
    """
    metas: List[CandidateMeta] = []
    buckets: Dict[str, List[int]] = {}
    for cand in tqdm(candidate_iter, desc="build bucket index", unit="cand"):
        idx = len(metas)
        metas.append(cand)
        bucket = _bucket_key(cand[2])
        buckets.setdefault(bucket, []).append(idx)
    return metas, buckets


def retrieve_candidates_by_mass_inmem(
    metas: List[CandidateMeta],
    buckets: Dict[str, List[int]],
    neutral_mass: float,
    ppm_tolerance: float,
) -> List[CandidateMeta]:
    """Query in-memory bucket index for candidates within mass tolerance.

    Uses DeepSearch strategy: check 3 adjacent 0.1-Da buckets, then ppm filter.
    """
    window = neutral_mass * ppm_tolerance * 1e-6
    lower = neutral_mass - window
    upper = neutral_mass + window

    center_val = float(_bucket_key(neutral_mass))
    offsets = [-0.1, 0.0, 0.1]
    out: List[CandidateMeta] = []
    for d in offsets:
        bucket = _bucket_key(center_val + d)
        for idx in buckets.get(bucket, []):
            cand = metas[idx]
            if lower <= cand[2] <= upper:
                out.append(cand)
    return out
