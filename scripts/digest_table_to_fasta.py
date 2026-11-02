"""Convert an externally digested peptide table to a search peptide FASTA.

No digestion or modification enumeration is performed. Length and neutral-mass
checks use the same residue masses and limits as the original peptide builder.
Every retained table row is written; target/decoy IDs are preserved.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Protein Digestion Simulator tab-delimited result.")
    parser.add_argument("--output", required=True, type=Path, help="New peptide FASTA for --peptide-fasta.")
    parser.add_argument("--protein-column", default="Protein_Name")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--min-length", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--min-mass", type=float, default=500.0)
    parser.add_argument("--max-mass", type=float, default=5000.0)
    args = parser.parse_args()
    from ShiftSearch.constants import H2O, RESIDUE_MASS

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must differ")
    if args.min_length < 1 or args.max_length < args.min_length or args.max_mass < args.min_mass:
        raise ValueError("Invalid length or mass limits")
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames or any(col not in reader.fieldnames for col in (args.protein_column, args.sequence_column)):
            raise ValueError("Digest table must contain protein and sequence columns")
        with args.output.open("x", encoding="utf-8", newline="") as handle:
            for number, row in enumerate(reader, 2):
                protein = (row.get(args.protein_column) or "").strip()
                sequence = (row.get(args.sequence_column) or "").strip().upper()
                if not protein or not sequence:
                    raise ValueError(f"Empty protein or sequence at table row {number}")
                counts["input_rows"] += 1
                if any(aa not in RESIDUE_MASS for aa in sequence):
                    counts["unsupported_sequence_rows"] += 1
                    continue
                if not args.min_length <= len(sequence) <= args.max_length:
                    counts["outside_length_rows"] += 1
                    continue
                mass = sum(RESIDUE_MASS[aa] for aa in sequence) + H2O
                if not args.min_mass <= mass <= args.max_mass:
                    counts["outside_mass_rows"] += 1
                    continue
                handle.write(f">{protein}\n{sequence}\n")
                counts["written_rows"] += 1
    for key, value in sorted(counts.items()):
        print(f"{key}={value}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
