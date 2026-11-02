"""Filter and merge input MGF files using the evaluation criteria.

The peak-count, charge parsing and SCANS handling are adapted from the original
filter_and_merge_mgfs.py. Filtering is streamed to avoid retaining every peak
array in memory. Peak m/z and intensity values are not normalized here.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SCAN_RE = re.compile(r"scan=(\d+)")


def ensure_scan_param(spectrum):
    params = spectrum.get("params")
    if params is None or params.get("scans") is not None or params.get("scan") is not None:
        return
    match = SCAN_RE.search(str(params.get("title", "") or ""))
    if match:
        params["scans"] = match.group(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", nargs="+", type=Path, help="One or more original MGF files.")
    inputs.add_argument("--input-dir", type=Path, help="Directory containing original *.mgf files, non-recursively.")
    parser.add_argument("--output", required=True, type=Path, help="New merged MGF file.")
    parser.add_argument("--summary", type=Path, help="New summary TSV; defaults to <output>.summary.tsv.")
    parser.add_argument("--min-peaks", type=int, default=64)
    parser.add_argument("--min-charge", type=int, default=2)
    parser.add_argument("--max-charge", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    from pyteomics import mgf

    if args.min_peaks < 0 or args.min_charge < 1 or args.max_charge < args.min_charge:
        raise ValueError("Invalid peak-count or charge limits")
    paths = sorted(args.input_dir.glob("*.mgf")) if args.input_dir is not None else args.input
    paths = [path.resolve() for path in paths]
    output = args.output.resolve()
    summary = (args.summary or args.output.with_suffix(".summary.tsv")).resolve()
    if not paths:
        raise ValueError("No input MGF files found")
    if len(set(paths)) != len(paths):
        raise ValueError("An input MGF was supplied more than once")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output in paths or summary in paths or output == summary:
        raise ValueError("Inputs, output and summary must be different files")
    if output.exists() or summary.exists():
        raise FileExistsError("Choose new output and summary paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with output.open("x", encoding="utf-8", newline="") as handle:
        for path in paths:
            counts = {"total": 0, "kept": 0}

            def retained():
                with mgf.MGF(str(path)) as reader:
                    for spectrum in reader:
                        counts["total"] += 1
                        peaks = spectrum.get("m/z array")
                        n_peaks = 0 if peaks is None else len(peaks)
                        charge = (spectrum.get("params") or {}).get("charge")
                        if isinstance(charge, (list, tuple)):
                            charge = charge[0] if charge else None
                        try:
                            charge_value = int(str(charge).replace("+", "").replace("-", ""))
                        except (TypeError, ValueError):
                            charge_value = 0
                        if n_peaks < args.min_peaks or not args.min_charge <= charge_value <= args.max_charge:
                            continue
                        ensure_scan_param(spectrum)
                        counts["kept"] += 1
                        yield spectrum

            mgf.write(retained(), output=handle)
            summaries.append((path.name, counts["total"], counts["total"] - counts["kept"], counts["kept"]))
            print(f"{path.name}: total={counts['total']} kept={counts['kept']}")
    with summary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("file", "total_spectra", "dropped_spectra", "kept_spectra"))
        writer.writerows(summaries)
        writer.writerow(("COMBINED", sum(row[1] for row in summaries), sum(row[2] for row in summaries), sum(row[3] for row in summaries)))
    print(f"output={output}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
