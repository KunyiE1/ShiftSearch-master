from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ShiftSearch.search.hierarchical_by_search import search_mgf_hierarchical_by


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search an MGF with the ShiftSearch [g, I] scorer."
    )
    parser.add_argument("--mgf", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="Hierarchical model YAML")
    parser.add_argument("--search-config", default=None)
    parser.add_argument(
        "--peptide-fasta",
        required=True,
        help="Pre-digested peptide FASTA; target/decoy status comes from its IDs.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fdr", type=float, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="PSM pairs per GPU forward pass; overrides Search.batch_size.",
    )
    parser.add_argument(
        "--spectra-per-batch",
        type=int,
        default=None,
        help="Distinct spectra preprocessed together; overrides YAML.",
    )
    parser.add_argument("--flush-every", type=int, default=None)
    parser.add_argument("--prefix-cache-size", type=int, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable FP16 autocast. Float32 is recommended for FDR ranking.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    search_config_path = Path(
        args.search_config
        or Path(args.config).with_name("search_oxi_max2.yaml")
    )
    with search_config_path.open("r", encoding="utf-8") as handle:
        search_cfg = yaml.safe_load(handle)
    search_params = dict(search_cfg["Search"])
    mods_cfg = dict(search_cfg.get("Mods", {}))

    fdr_threshold = (
        args.fdr
        if args.fdr is not None
        else float(search_params.get("fdr", 1.0))
    )
    search_mgf_hierarchical_by(
        peptide_db_path=args.peptide_fasta,
        mgf_path=args.mgf,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        precursor_ppm=float(search_params.get("precursor_ppm", 10.0)),
        fixed_mods=mods_cfg.get("fixed"),
        variable_mods=mods_cfg.get("variable"),
        max_variable_mods=int(search_params.get("max_variable_mods", 2)),
        decoy_prefix=str(search_params.get("decoy_prefix", "DECOY_")),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else int(search_params.get("batch_size", 1024))
        ),
        spectra_per_batch=(
            args.spectra_per_batch
            if args.spectra_per_batch is not None
            else int(search_params.get("spectra_per_batch", 32))
        ),
        top_k=int(search_params.get("top_k", 10)),
        output_path=args.output,
        device=args.device,
        min_pep_len=(
            int(search_params["min_pep_len"])
            if search_params.get("min_pep_len") is not None
            else None
        ),
        max_pep_len=(
            int(search_params["max_pep_len"])
            if search_params.get("max_pep_len") is not None
            else None
        ),
        min_score=float(search_params.get("min_score", 0.0)),
        fdr_threshold=fdr_threshold,
        label_mods=mods_cfg.get("label"),
        flush_every=(
            args.flush_every
            if args.flush_every is not None
            else int(search_params.get("flush_every", 1000))
        ),
        prefix_cache_size=(
            args.prefix_cache_size
            if args.prefix_cache_size is not None
            else int(search_params.get("prefix_cache_size", 100000))
        ),
        amp=(
            args.amp
            if args.amp is not None
            else bool(search_params.get("amp", False))
        ),
    )


if __name__ == "__main__":
    main()
