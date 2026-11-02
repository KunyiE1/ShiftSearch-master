"""Check a small database-search example against recorded best-hit results.

Spectrum titles and precursor charges identify records; spectrum_index is not
used because extracting an MGF subset changes that positional index.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_decoy(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ValueError(f"Unsupported is_decoy value: {value!r}")


def read_results(path: Path):
    records = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        title_column = "spectrum_title" if "spectrum_title" in columns else "title"
        required = {title_column, "charge", "search_result", "is_decoy", "score"}
        if not required.issubset(columns):
            raise ValueError(f"{path.name}: missing columns {sorted(required - columns)}")
        for line, row in enumerate(reader, 2):
            title = row[title_column]
            if not title or not title.strip():
                raise ValueError(f"{path.name}:{line}: empty spectrum title")
            try:
                charge = int(row["charge"])
                score = float(row["score"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path.name}:{line}: invalid charge or score") from error
            if charge <= 0 or not math.isfinite(score):
                raise ValueError(f"{path.name}:{line}: charge must be positive and score finite")
            key = (title, charge)
            if key in records:
                raise ValueError(f"{path.name}:{line}: duplicate spectrum title/charge {key!r}")
            peptide = row["search_result"]
            if not peptide or not peptide.strip():
                raise ValueError(f"{path.name}:{line}: empty search_result")
            records[key] = {"search_result": peptide, "is_decoy": parse_decoy(row["is_decoy"]), "score": score}
            for name in ("sequence", "protein_id", "rank"):
                if name in columns:
                    if name == "rank":
                        try:
                            records[key][name] = int(row[name])
                        except (TypeError, ValueError) as error:
                            raise ValueError(f"{path.name}:{line}: invalid rank") from error
                    else:
                        records[key][name] = row[name]
    return records


def check_results(results: Path, expected: Path, score_atol: float = 0.0005):
    if not math.isfinite(score_atol) or score_atol < 0:
        raise ValueError("--score-atol must be finite and nonnegative")
    actual = read_results(Path(results))
    reference = read_results(Path(expected))
    if len(actual) != len(reference):
        raise ValueError(f"Best-hit count differs: actual={len(actual)}, expected={len(reference)}")
    missing = set(reference) - set(actual)
    unexpected = set(actual) - set(reference)
    if missing or unexpected:
        raise ValueError(f"Spectrum title/charge keys differ: missing={len(missing)}, unexpected={len(unexpected)}")
    differences = []
    maximum_score_difference = 0.0
    for key, original in reference.items():
        observed = actual[key]
        delta = abs(observed["score"] - original["score"])
        maximum_score_difference = max(maximum_score_difference, delta)
        if observed["search_result"] != original["search_result"]:
            differences.append(f"{key!r}: peptide differs ({observed['search_result']!r} vs {original['search_result']!r})")
        if observed["is_decoy"] != original["is_decoy"]:
            differences.append(f"{key!r}: target/decoy label differs")
        for name in ("sequence", "protein_id", "rank"):
            if name in original and observed.get(name) != original[name]:
                differences.append(f"{key!r}: {name} differs")
        if not math.isclose(observed["score"], original["score"], rel_tol=0.0, abs_tol=score_atol):
            differences.append(f"{key!r}: score difference {delta:.9g} exceeds {score_atol:.9g}")
    if differences:
        raise ValueError(f"{len(differences)} result differences:\n" + "\n".join(differences[:10]))
    return {"status": "PASS", "best_hits": len(reference),
            "key": "spectrum title and precursor charge", "peptidoforms_identical": True,
            "target_decoy_flags_identical": True, "score_atol": score_atol,
            "optional_metadata_checked": sorted({name for row in reference.values() for name in ("sequence", "protein_id", "rank") if name in row}),
            "maximum_absolute_score_difference": maximum_score_difference}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--expected", type=Path, default=Path("examples/celegans/expected_best.tsv"))
    parser.add_argument("--score-atol", type=float, default=0.0005)
    args = parser.parse_args(argv)
    try:
        report = check_results(args.results, args.expected, args.score_atol)
    except (OSError, ValueError) as error:
        parser.exit(1, f"[example-check] FAIL: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
