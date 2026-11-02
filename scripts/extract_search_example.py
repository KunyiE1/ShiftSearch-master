"""Extract real spectra and preserve their complete mass-compatible search space.

This preparation utility uses the existing candidate enumeration and retrieval
functions. Model scores and peptide assignments are never used for selection.
The distributed example is already prepared; users do not need the full inputs
unless they want to repeat extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import yaml

from ShiftSearch.constants import PROTON
from ShiftSearch.search.database import (
    build_candidate_bucket_index,
    enumerate_peptides_from_peptide_db,
    retrieve_candidates_by_mass_inmem,
)
from ShiftSearch.search.peptide_utils import parse_mod_spec
from ShiftSearch.search.spectrum_io import load_mgf_spectra


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mgf_blocks(path):
    block = None
    source_index = -1
    with Path(path).open("rb") as handle:
        for line in handle:
            marker = line.strip().upper()
            if marker == b"BEGIN IONS":
                if block is not None:
                    raise ValueError("Nested MGF spectrum")
                source_index += 1
                block = [line]
            elif block is not None:
                block.append(line)
                if marker == b"END IONS":
                    yield source_index, b"".join(block)
                    block = None
    if block is not None:
        raise ValueError("Unterminated MGF spectrum")


def spectrum_metadata(block):
    params = {}
    peaks = 0
    for line in block.splitlines():
        line = line.strip()
        if not line or line.upper() in (b"BEGIN IONS", b"END IONS"):
            continue
        if b"=" in line:
            key, value = line.split(b"=", 1)
            params[key.upper()] = value.decode("utf-8")
        elif line[:1] not in (b"#", b";", b"!"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    float(fields[0])
                    float(fields[1])
                    peaks += 1
                except ValueError:
                    pass
    charge = re.match(r"\s*(\d+)", params.get(b"CHARGE", ""))
    return {"title": params.get(b"TITLE", ""), "charge": int(charge[1]) if charge else 0,
            "raw_peak_count": peaks}


def fasta_blocks(path):
    block = []
    with Path(path).open("rb") as handle:
        for line in handle:
            if line.lstrip().startswith(b">") and block:
                yield b"".join(block)
                block = []
            if block or line.lstrip().startswith(b">"):
                block.append(line)
        if block:
            yield b"".join(block)


def candidate_signature(candidates):
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(json.dumps(candidate, separators=(",", ":")).encode("utf-8") + b"\n")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mgf", type=Path, required=True)
    parser.add_argument("--source-peptide-fasta", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-spectra", type=int, default=100)
    parser.add_argument("--min-peaks", type=int, default=64)
    parser.add_argument("--min-charge", type=int, default=2)
    parser.add_argument("--max-charge", type=int, default=4)
    parser.add_argument("--dataset-accession", default="")
    args = parser.parse_args()
    if args.max_spectra < 1 or args.min_peaks < 1 or not 1 <= args.min_charge <= args.max_charge:
        parser.error("Invalid spectrum selection limits")
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing files are never replaced")
    for path in (args.source_mgf, args.source_peptide_fasta, args.model_config, args.search_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    search_config = yaml.safe_load(args.search_config.read_text(encoding="utf-8"))
    search = dict(search_config["Search"])

    def eligible(metadata):
        return (metadata["raw_peak_count"] >= args.min_peaks and
                args.min_charge <= metadata["charge"] <= args.max_charge)

    print("[example] counting eligible real spectra in source order", flush=True)
    total = sum(eligible(spectrum_metadata(block)) for _, block in mgf_blocks(args.source_mgf))
    if not total:
        raise ValueError("No eligible spectra")
    count = min(total, args.max_spectra)
    ordinals = {i * (total - 1) // (count - 1) for i in range(count)} if count > 1 else {0}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mgf_output = args.output_dir / "spectra.mgf"
    selected = []
    eligible_index = 0
    with mgf_output.open("xb") as handle:
        for source_index, block in mgf_blocks(args.source_mgf):
            metadata = spectrum_metadata(block)
            if not eligible(metadata):
                continue
            if eligible_index in ordinals:
                # Preserve headers, experimental peaks and intensities verbatim.
                handle.write(block)
                if not block.endswith(b"\n"):
                    handle.write(b"\n")
                selected.append(dict(metadata, demo_index=len(selected), source_index=source_index,
                                     source_eligible_index=eligible_index,
                                     original_block_sha256=hashlib.sha256(block).hexdigest()))
            eligible_index += 1
    spectra = list(load_mgf_spectra(mgf_output, max_mz=model_config["Meta"]["max_mz"],
                                  cpu_pre_topk=model_config["Meta"]["cpu_pre_topk"],
                                  min_peaks=model_config["Meta"]["min_npeaks"]))
    if len(spectra) != count or len({(s.title, s.charge) for s in spectra}) != count:
        raise ValueError("Extracted spectrum IDs must be unique and valid for search")
    kwargs = dict(
        fixed_mods=parse_mod_spec(search_config.get("Mods", {}).get("fixed")),
        variable_mods=parse_mod_spec(search_config.get("Mods", {}).get("variable")),
        label_mods=parse_mod_spec(search_config.get("Mods", {}).get("label")),
        min_len=search["min_pep_len"], max_len=search["max_pep_len"],
        max_variable_mods=search["max_variable_mods"], decoy_prefix=search["decoy_prefix"],
        max_pep_len=model_config["Meta"]["max_pep_len"],
    )
    print("[example] enumerating the original complete peptide database", flush=True)
    metas, buckets = build_candidate_bucket_index(enumerate_peptides_from_peptide_db(args.source_peptide_fasta, **kwargs))
    chosen = set()
    queries = []
    for record, spectrum in zip(selected, spectra):
        mass = spectrum.pre_mz * spectrum.charge - spectrum.charge * PROTON
        candidates = retrieve_candidates_by_mass_inmem(metas, buckets, mass, search["precursor_ppm"])
        queries.append(candidates)
        chosen.update((candidate[0], candidate[3]) for candidate in candidates)
        record.update(candidate_count=len(candidates), target_candidates=sum(not c[4] for c in candidates),
                      decoy_candidates=sum(c[4] for c in candidates),
                      complete_candidate_sha256=candidate_signature(candidates))
    fasta_output = args.output_dir / "peptides.fasta"
    retained_entries = 0
    with fasta_output.open("xb") as handle:
        for block in fasta_blocks(args.source_peptide_fasta):
            lines = block.decode("utf-8").splitlines()
            header = lines[0].strip()[1:].strip()
            sequence = "".join(line.strip().upper() for line in lines[1:] if line.strip())
            if (sequence, header) in chosen:
                handle.write(block)
                if not block.endswith(b"\n"):
                    handle.write(b"\n")
                retained_entries += 1
    mini_metas, mini_buckets = build_candidate_bucket_index(enumerate_peptides_from_peptide_db(fasta_output, **kwargs))
    for record, spectrum, expected in zip(selected, spectra, queries):
        mass = spectrum.pre_mz * spectrum.charge - spectrum.charge * PROTON
        actual = retrieve_candidates_by_mass_inmem(mini_metas, mini_buckets, mass, search["precursor_ppm"])
        if actual != expected:
            raise RuntimeError(f"Reduced database changes the ordered candidate list: {spectrum.title}")
        record["subset_candidate_sha256"] = candidate_signature(actual)
        record["ordered_candidates_identical"] = True
    # Only computational batching/output settings change; scientific search
    # settings and all per-query candidate lists remain the same.
    search_config["Search"].update(batch_size=64, spectra_per_batch=8, fdr=1.0, flush_every=100)
    config_output = args.output_dir / "search.yaml"
    with config_output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(search_config, handle, sort_keys=False)
    files = {p.name: {"sha256": checksum(p), "bytes": p.stat().st_size}
             for p in (mgf_output, fasta_output, config_output)}
    manifest = {
        "dataset_accession": args.dataset_accession, "data_kind": "real experimental spectra and real target-decoy peptide database subset",
        "selection": {"rule": "uniformly spaced eligible spectra in source order, including first and last; integer index floor(i*(N-1)/(K-1))",
                      "uses_scores_or_search_assignments": False, "min_peaks": args.min_peaks,
                      "min_charge": args.min_charge, "max_charge": args.max_charge,
                      "eligible_source_spectra": total},
        "n_spectra": count, "expected_best_hits": sum(bool(candidates) for candidates in queries),
        "spectra_without_candidates": sum(not candidates for candidates in queries),
        "original_candidate_forms": len(metas), "retained_fasta_entries": retained_entries,
        "subset_candidate_forms": len(mini_metas), "candidate_psms": sum(map(len, queries)),
        "ordered_candidates_identical_for_every_query": True,
        "sources": {"mgf": {"filename": args.source_mgf.name, "sha256": checksum(args.source_mgf)},
                    "peptide_fasta": {"filename": args.source_peptide_fasta.name, "sha256": checksum(args.source_peptide_fasta)}},
        "source_search_settings": search, "files": files, "spectra": selected,
        "limits": ["Search assignments are not ground-truth annotations", "Small-example FDR is not a performance estimate", "The reduced FASTA preserves candidates only for these selected spectra and settings"],
    }
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: manifest[key] for key in ("n_spectra", "expected_best_hits", "spectra_without_candidates", "retained_fasta_entries", "candidate_psms", "ordered_candidates_identical_for_every_query")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
