"""Fast streaming search for the hierarchical b/y PSM scorer.

Candidate enumeration, precursor-mass retrieval, MGF parsing, result writing,
and FDR calculation are shared with the legacy ShiftSearch search pipeline.
Only model loading and PSM scoring are architecture-specific.
"""
from __future__ import annotations

import heapq
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from ShiftSearch.constants import PROTON
from ShiftSearch.psm.hierarchical_by_module import PSMHierarchicalBY
from ShiftSearch.search.database import (
    CandidateMeta,
    build_candidate_bucket_index,
    enumerate_peptides_from_peptide_db,
    retrieve_candidates_by_mass_inmem,
)
from ShiftSearch.search.peptide_utils import build_prefix_masses, parse_mod_spec
from ShiftSearch.search.result_io import (
    StreamingSearchWriter,
    check_new_output_paths,
    write_fdr_filtered_best,
)
from ShiftSearch.search.spectrum_io import SpectrumRecord, load_mgf_spectra


SpectrumCandidates = Tuple[SpectrumRecord, Sequence[CandidateMeta]]
PrefixBuilder = Callable[[str], np.ndarray]


def load_hierarchical_by_model(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cuda",
) -> PSMHierarchicalBY:
    print(f"[search] loading hierarchical b/y config: {config_path}", flush=True)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = PSMHierarchicalBY(
        model_params=dict(config["Model"]["params"]),
        train_params=dict(config.get("Train", {})),
        meta_params=dict(config["Meta"]),
    )
    print(f"[search] loading checkpoint: {checkpoint_path}", flush=True)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:  # PyTorch versions predating the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _make_prefix_builder(max_pep_len: int, cache_size: int) -> tuple[PrefixBuilder, Callable]:
    def build(annotated_sequence: str) -> np.ndarray:
        return build_prefix_masses(annotated_sequence, max_pep_len)[1]

    if cache_size <= 0:
        return build, lambda: None

    cached = lru_cache(maxsize=cache_size)(build)
    return cached, cached.cache_info


def _pair_chunks(
    group: Sequence[SpectrumCandidates],
    batch_size: int,
) -> Iterable[List[Tuple[int, int, CandidateMeta]]]:
    chunk: List[Tuple[int, int, CandidateMeta]] = []
    for spectrum_local_index, (_, candidates) in enumerate(group):
        for candidate_order, candidate in enumerate(candidates):
            chunk.append((spectrum_local_index, candidate_order, candidate))
            if len(chunk) == batch_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def score_spectrum_group(
    model: PSMHierarchicalBY,
    group: Sequence[SpectrumCandidates],
    batch_size: int,
    device: torch.device,
    max_pep_len: int,
    top_k: int,
    min_score: float,
    prefix_builder: PrefixBuilder,
    amp: bool = False,
) -> List[List[Dict[str, object]]]:
    """Score several spectra together while retaining only top-k PSMs.

    Each distinct spectrum is peak-selected and normalized once. Candidate
    pairs from all spectra in the group are then packed into large GPU batches.
    This keeps the GPU occupied without retaining every scored PSM in memory.
    """
    if not group:
        return []

    raw_mz = torch.from_numpy(np.stack([item[0].raw_mz for item in group]))
    raw_intensity = torch.from_numpy(
        np.stack([item[0].raw_intensity for item in group])
    )
    raw_peak_mask = torch.from_numpy(
        np.stack([item[0].raw_peak_mask for item in group])
    )
    top_mz, normalized_intensity, top_valid = model.prepare_spectrum_batch(
        raw_mz,
        raw_intensity,
        raw_peak_mask,
    )

    # Heap key is (score, -candidate_order). Thus the lowest-scoring and, for
    # exact ties, latest-enumerated candidate is discarded first. This matches
    # the old stable descending sort while keeping only O(spectra * top_k) rows.
    heaps: List[List[Tuple[float, int, Dict[str, object]]]] = [
        [] for _ in group
    ]
    observed_neutral_mass = [
        spectrum.pre_mz * spectrum.charge - spectrum.charge * PROTON
        for spectrum, _ in group
    ]
    use_amp = bool(amp and device.type == "cuda")

    with torch.inference_mode():
        for chunk in _pair_chunks(group, batch_size):
            spectrum_indices_np = np.fromiter(
                (item[0] for item in chunk),
                dtype=np.int64,
                count=len(chunk),
            )
            spectrum_indices = torch.from_numpy(spectrum_indices_np).to(device)
            candidate_prefixes = np.stack(
                [prefix_builder(item[2][1]) for item in chunk]
            )
            prefix_masses = torch.from_numpy(candidate_prefixes).to(device)
            peptide_length = torch.from_numpy(
                np.count_nonzero(candidate_prefixes, axis=1).astype(np.int64)
            ).to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                (
                    b_cell_features,
                    y_cell_features,
                    cell_mask,
                    cleavage_mask,
                ) = model.build_candidate_features(
                    top_mz.index_select(0, spectrum_indices),
                    normalized_intensity.index_select(0, spectrum_indices),
                    top_valid.index_select(0, spectrum_indices),
                    prefix_masses,
                    peptide_length,
                )
                logits = model(
                    b_cell_features,
                    y_cell_features,
                    cell_mask,
                    cleavage_mask,
                )

            logits_cpu = logits.float().cpu().numpy()
            scores_cpu = torch.sigmoid(logits.float()).cpu().numpy()
            for item, score_value, logit_value in zip(
                chunk,
                scores_cpu,
                logits_cpu,
            ):
                spectrum_local_index, candidate_order, candidate = item
                score = float(score_value)
                logit = float(logit_value)
                if not np.isfinite(score) or not np.isfinite(logit):
                    raise RuntimeError(
                        "Non-finite hierarchical b/y score for "
                        f"spectrum={group[spectrum_local_index][0].title!r}, "
                        f"candidate={candidate[1]!r}"
                    )
                if score < min_score:
                    continue
                spectrum = group[spectrum_local_index][0]
                neutral_mass = float(candidate[2])
                precursor_mass = observed_neutral_mass[spectrum_local_index]
                ppm_error = (
                    (neutral_mass - precursor_mass)
                    / max(precursor_mass, 1e-8)
                    * 1e6
                )
                row: Dict[str, object] = {
                    "spectrum_index": spectrum.spectrum_index,
                    "title": spectrum.title,
                    "charge": spectrum.charge,
                    "pre_mz": spectrum.pre_mz,
                    "sequence": candidate[0],
                    "search_result": candidate[1],
                    "target_sequence": spectrum.target_sequence,
                    "neutral_mass": neutral_mass,
                    "ppm_error": ppm_error,
                    "score": score,
                    "logit": logit,
                    "protein_id": candidate[3],
                    "is_decoy": candidate[4],
                }
                heap = heaps[spectrum_local_index]
                heap_item = (score, -candidate_order, row)
                if len(heap) < top_k:
                    heapq.heappush(heap, heap_item)
                elif heap_item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, heap_item)

    ranked: List[List[Dict[str, object]]] = []
    for heap in heaps:
        ordered = sorted(heap, key=lambda item: item[:2], reverse=True)
        ranked.append([item[2] for item in ordered])
    return ranked


def search_mgf_hierarchical_by(
    peptide_db_path: str | Path,
    mgf_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    precursor_ppm: float,
    fixed_mods: Optional[str],
    variable_mods: Optional[str],
    max_variable_mods: int,
    decoy_prefix: str,
    batch_size: int,
    spectra_per_batch: int,
    top_k: int,
    output_path: str | Path,
    device: str | torch.device = "cuda",
    min_pep_len: Optional[int] = None,
    max_pep_len: Optional[int] = None,
    min_score: float = 0.0,
    fdr_threshold: float = 1.0,
    label_mods: Optional[str] = None,
    flush_every: int = 1000,
    prefix_cache_size: int = 100000,
    amp: bool = False,
    return_dataframe: bool = False,
) -> Path | pd.DataFrame:
    """Run a bounded-memory hierarchical b/y search on an MGF file."""
    if min(batch_size, spectra_per_batch, top_k, flush_every) < 1:
        raise ValueError(
            "batch_size, spectra_per_batch, top_k and flush_every must be at least 1"
        )

    output_path = Path(output_path)
    best_path = output_path.with_name(output_path.stem + ".best.tsv")
    fdr_best_path = output_path.with_name(
        output_path.stem + f".fdr_{int(fdr_threshold * 100):02d}.best.tsv"
    )
    check_new_output_paths(
        output_path,
        best_path,
        *([fdr_best_path] if fdr_threshold < 1.0 else []),
    )

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    meta = config["Meta"]
    cpu_pre_topk = int(meta.get("cpu_pre_topk", 1024))
    max_mz = float(meta["max_mz"])
    if max_pep_len is None:
        max_pep_len = int(meta["max_pep_len"])
    if min_pep_len is None:
        min_pep_len = int(meta.get("min_pep_len", 0))
    min_n_peaks = int(meta.get("min_npeaks", 0))

    fixed_mod_map = parse_mod_spec(fixed_mods)
    variable_mod_map = parse_mod_spec(variable_mods)
    label_mod_map = parse_mod_spec(label_mods)
    print(
        f"[search] precursor_ppm={precursor_ppm} top_k={top_k} "
        f"batch_size={batch_size} spectra_per_batch={spectra_per_batch} "
        f"amp={amp}",
        flush=True,
    )
    print(
        f"[search] fixed_mods={fixed_mod_map} "
        f"variable_mods={variable_mod_map} label_mods={label_mod_map}",
        flush=True,
    )

    candidate_iter = enumerate_peptides_from_peptide_db(
        peptide_db_path=peptide_db_path,
        fixed_mods=fixed_mod_map,
        variable_mods=variable_mod_map,
        min_len=min_pep_len,
        max_len=max_pep_len,
        max_variable_mods=max_variable_mods,
        decoy_prefix=decoy_prefix,
        max_pep_len=max_pep_len,
        label_mods=label_mod_map,
    )
    print("[search] building in-memory candidate index...", flush=True)
    metas, buckets = build_candidate_bucket_index(candidate_iter)
    decoy_counts = Counter(candidate[4] for candidate in metas)
    print(
        f"[search] indexed: {len(metas)} candidates "
        f"({decoy_counts.get(False, 0)} target, "
        f"{decoy_counts.get(True, 0)} decoy) in {len(buckets)} mass buckets",
        flush=True,
    )

    device = torch.device(device)
    model = load_hierarchical_by_model(checkpoint_path, config_path, device=device)
    prefix_builder, prefix_cache_info = _make_prefix_builder(
        max_pep_len=max_pep_len,
        cache_size=prefix_cache_size,
    )

    n_valid_spectra = 0
    n_candidate_pairs = 0
    print(f"[search] streaming top-k results to: {output_path}", flush=True)
    print(f"[search] streaming best-hit results to: {best_path}", flush=True)
    print(f"[search] scoring spectra from MGF: {mgf_path}", flush=True)

    def process_group(
        writer: StreamingSearchWriter,
        group: List[SpectrumCandidates],
    ) -> None:
        nonlocal n_candidate_pairs
        n_candidate_pairs += sum(len(candidates) for _, candidates in group)
        ranked_group = score_spectrum_group(
            model=model,
            group=group,
            batch_size=batch_size,
            device=device,
            max_pep_len=max_pep_len,
            top_k=top_k,
            min_score=min_score,
            prefix_builder=prefix_builder,
            amp=amp,
        )
        for ranked_rows in ranked_group:
            writer.write_spectrum(ranked_rows, top_k=top_k)

    with StreamingSearchWriter(output_path, flush_every=flush_every) as writer:
        group: List[SpectrumCandidates] = []
        for spectrum in tqdm(
            load_mgf_spectra(
                mgf_path,
                max_mz=max_mz,
                cpu_pre_topk=cpu_pre_topk,
                min_peaks=min_n_peaks,
            ),
            desc="search spectra",
            unit="spectrum",
        ):
            neutral_mass = (
                spectrum.pre_mz * spectrum.charge - spectrum.charge * PROTON
            )
            candidates = retrieve_candidates_by_mass_inmem(
                metas,
                buckets,
                neutral_mass,
                precursor_ppm,
            )
            group.append((spectrum, candidates))
            n_valid_spectra += 1
            if len(group) >= spectra_per_batch:
                process_group(writer, group)
                group = []
        if group:
            process_group(writer, group)

    print(f"[search] total valid spectra searched: {n_valid_spectra}", flush=True)
    print(f"[search] total candidate PSMs scored: {n_candidate_pairs}", flush=True)
    print(
        f"[search] saved {writer.n_rows} top-k rows and {writer.n_best} best hits",
        flush=True,
    )
    cache_info = prefix_cache_info()
    if cache_info is not None:
        print(f"[search] prefix-mass cache: {cache_info}", flush=True)

    del model, metas, buckets, candidate_iter
    if writer.n_best and fdr_threshold < 1.0:
        print("[search] computing global FDR from best-hit numeric columns only", flush=True)
        kept = write_fdr_filtered_best(best_path, fdr_best_path, fdr_threshold)
        print(
            f"[search] saved FDR-controlled results to: {fdr_best_path} "
            f"(spectra passing: {kept})",
            flush=True,
        )

    if return_dataframe:
        return pd.read_csv(output_path, sep="\t", float_precision="round_trip")
    return output_path


__all__ = [
    "load_hierarchical_by_model",
    "score_spectrum_group",
    "search_mgf_hierarchical_by",
]
