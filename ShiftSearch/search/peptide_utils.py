from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ShiftSearch.constants import H2O, PROTON, RESIDUE_MASS

AA_MASS = RESIDUE_MASS


def has_unsupported_residue(peptide: str) -> bool:
    """Return whether a peptide contains a residue with no supported mass."""
    return any(aa not in RESIDUE_MASS for aa in peptide)


def parse_fasta(fasta_path: str | Path) -> Iterator[Tuple[str, str]]:
    header = None
    chunks: List[str] = []
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.upper())
        if header is not None:
            yield header, "".join(chunks)


def digest_tryptic(sequence: str, missed_cleavages: int = 2, min_len: int = 6, max_len: int = 32, enzyme: str = "trypsin") -> Iterator[str]:
    enzyme_lc = enzyme.lower().replace("-", "_").replace(" ", "")
    if enzyme_lc == "lysc":
        sites = {"K"}
        proline_blocks = False
    elif enzyme_lc in ("trypsin_p", "trypsinp", "trypsin/p"):
        sites = {"K", "R"}
        proline_blocks = False
    else:
        sites = {"K", "R"}
        proline_blocks = True
    cleavages = [-1]
    for i, aa in enumerate(sequence):
        if aa not in sites:
            continue
        if proline_blocks and i + 1 < len(sequence) and sequence[i + 1] == "P":
            continue
        cleavages.append(i)
    if cleavages[-1] != len(sequence) - 1:
        cleavages.append(len(sequence) - 1)

    for start_idx in range(len(cleavages) - 1):
        for missed in range(missed_cleavages + 1):
            end_idx = start_idx + missed + 1
            if end_idx >= len(cleavages):
                continue
            start = cleavages[start_idx] + 1
            end = cleavages[end_idx] + 1
            pep = sequence[start:end]
            if min_len <= len(pep) <= max_len:
                yield pep


def reverse_decoy(peptide: str) -> str:
    if len(peptide) <= 1:
        return peptide
    return peptide[:-1][::-1] + peptide[-1]


def reverse_protein_decoy(protein: str) -> str:
    """Reverse a protein sequence for decoy generation.

    The entire sequence is reversed (no end is fixed), matching the
    MS-GF+ ReverseDB strategy. Both terminal tryptic peptides are
    preserved as a composition (they appear at the opposite end of the
    decoy protein) but their position/cleavage context changes, so the
    decoy set mirrors the target set in mass distribution.
    """
    return protein[::-1]


def parse_mod_spec(spec: Optional[str]) -> Dict[str, List[float]]:
    if not spec:
        return {}
    mods: Dict[str, List[float]] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        norm_key = key.strip().upper()
        if norm_key in {"NTERM", "N_TERM", "N-TERM", "X"}:
            norm_key = "NTERM"
        mods.setdefault(norm_key, []).append(float(value))
    return mods


def mod_sites_for_peptide(peptide: str, mod_map: Dict[str, List[float]], include_nterm: bool = True) -> List[Tuple[int, float]]:
    sites: List[Tuple[int, float]] = []
    for idx, aa in enumerate(peptide):
        for shift in mod_map.get(aa, []):
            sites.append((idx, shift))
    if include_nterm:
        for shift in mod_map.get("NTERM", []):
            sites.append((-1, shift))
    return sites


def enumerate_mod_site_combinations(
    peptide: str,
    fixed_mods: Optional[Dict[str, List[float]]],
    variable_mods: Optional[Dict[str, List[float]]],
    max_variable_mods: int,
) -> Iterator[Tuple[List[Tuple[int, float]], float, List[float]]]:
    fixed_mods = fixed_mods or {}
    variable_mods = variable_mods or {}

    fixed_sites = mod_sites_for_peptide(peptide, fixed_mods, include_nterm=False)
    fixed_nterm_shift = sum(fixed_mods.get("NTERM", []))

    # Non-Nterm variable sites, with first/last residue positions excluded
    variable_sites = mod_sites_for_peptide(peptide, variable_mods, include_nterm=False)
    if peptide:
        variable_sites = [(idx, shift) for idx, shift in variable_sites if 0 < idx < len(peptide) - 1]

    nterm_mods = variable_mods.get("NTERM", [])
    nterm_choices = [None] + nterm_mods  # None means no Nterm mod

    for nterm_shift_extra in nterm_choices:
        # When an Nterm mod is picked, that consumes one of the max_variable_mods slots
        remaining_slots = max_variable_mods - (1 if nterm_shift_extra is not None else 0)
        n_choices = min(remaining_slots, len(variable_sites))
        variable_combos = [()]
        if n_choices > 0:
            for r in range(1, n_choices + 1):
                variable_combos.extend(combinations(variable_sites, r))

        for combo in variable_combos:
            combo_sites = list(fixed_sites)
            combo_sites.extend(combo)
            if nterm_shift_extra is not None:
                combo_sites.append((-1, nterm_shift_extra))
            used_sites = [idx for idx, _ in combo_sites]
            if len(used_sites) != len(set(used_sites)):
                continue

            residue_shifts = [0.0] * len(peptide)
            nterm_shift = fixed_nterm_shift + (nterm_shift_extra if nterm_shift_extra is not None else 0.0)
            for idx, shift in combo_sites:
                if idx >= 0:
                    residue_shifts[idx] += shift
            yield combo_sites, nterm_shift, residue_shifts


def add_shift_to_sequence(peptide: str, residue_shifts: Sequence[float], nterm_shift: float = 0.0) -> str:
    parts: List[str] = []
    if nterm_shift != 0.0:
        parts.append(f"{nterm_shift:+.3f}")
    for i, aa in enumerate(peptide):
        parts.append(aa)
        shift = residue_shifts[i]
        if shift != 0.0:
            parts.append(f"{shift:+.3f}")
    return "".join(parts)


def build_prefix_masses(annotated_sequence: str, max_pep_len: int) -> Tuple[str, np.ndarray]:
    residues: List[str] = []
    shifts: List[float] = []
    pending_nterm_shift = 0.0
    idx = 0
    while idx < len(annotated_sequence):
        token = annotated_sequence[idx]
        if token.isalpha():
            is_first = len(residues) == 0
            residues.append(token)
            shifts.append(0.0)
            if is_first and pending_nterm_shift != 0.0:
                shifts[0] += pending_nterm_shift
                pending_nterm_shift = 0.0
            idx += 1
            continue
        if token in "+-":
            end = idx + 1
            while end < len(annotated_sequence) and (annotated_sequence[end].isdigit() or annotated_sequence[end] == "."):
                end += 1
            if end == idx + 1:
                raise ValueError(f"Malformed mass shift in sequence: {annotated_sequence}")
            shift = float(annotated_sequence[idx:end])
            if residues:
                shifts[-1] += shift
            else:
                pending_nterm_shift += shift
            idx = end
            continue
        raise ValueError(f"Unsupported token {token!r} in sequence: {annotated_sequence}")

    seq = "".join(residues)
    prefix_masses = np.zeros(max_pep_len + 1, dtype=np.float32)
    running = 0.0
    for i, aa in enumerate(seq[:max_pep_len]):
        running += AA_MASS[aa] + float(shifts[i])
        prefix_masses[i + 1] = running
    return seq, prefix_masses


def neutral_mass_from_prefix(prefix_masses: np.ndarray) -> float:
    pep_len = int(np.count_nonzero(prefix_masses))
    return float(prefix_masses[pep_len] + H2O)


def compute_neutral_mass(peptide: str, nterm_shift: float, residue_shifts: Sequence[float]) -> float:
    """Compute neutral mass directly from peptide + shifts (no prefix_masses needed)."""
    mass = H2O + nterm_shift
    for i, aa in enumerate(peptide):
        if aa not in RESIDUE_MASS:
            raise KeyError(f"Unsupported residue {aa!r} in peptide {peptide!r}")
        mass += RESIDUE_MASS[aa] + float(residue_shifts[i])
    return float(mass)


def write_digested_peptide_fasta(
    fasta_path: str | Path,
    output_path: str | Path,
    enzyme: str = "trypsin",
    missed_cleavages: int = 2,
    min_len: int = 6,
    max_len: int = 32,
) -> int:
    if enzyme.lower() != "trypsin":
        raise NotImplementedError("Only trypsin digestion is supported")

    fasta_path = Path(fasta_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    total_written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for protein_id, protein_seq in parse_fasta(fasta_path):
            for peptide in digest_tryptic(
                protein_seq,
                missed_cleavages=missed_cleavages,
                min_len=min_len,
                max_len=max_len,
            ):
                if peptide in seen:
                    continue
                seen.add(peptide)
                total_written += 1
                out.write(f">pep_{total_written} {protein_id}\n{peptide}\n")
    return total_written
