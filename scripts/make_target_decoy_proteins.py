"""Write original proteins followed by fully reversed protein decoys.

Uses the existing FASTA reader and reverse_protein_decoy function. Digest
the resulting protein FASTA using the external Protein Digestion Simulator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path, help="Original target protein FASTA.")
    parser.add_argument("--output", required=True, type=Path, help="New target+decoy protein FASTA.")
    parser.add_argument("--decoy-prefix", default="DECOY_")
    args = parser.parse_args()
    from ShiftSearch.search.peptide_utils import parse_fasta, reverse_protein_decoy

    if not args.fasta.is_file():
        raise FileNotFoundError(args.fasta)
    if args.fasta.resolve() == args.output.resolve():
        raise ValueError("Input and output must differ")
    if not args.decoy_prefix or any(char.isspace() for char in args.decoy_prefix):
        raise ValueError("Decoy prefix must be nonempty and contain no whitespace")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        for header, sequence in parse_fasta(args.fasta):
            if not header or not sequence or header.startswith(args.decoy_prefix):
                raise ValueError("Input must contain nonempty target protein records only")
            handle.write(f">{header}\n{sequence}\n")
            count += 1
        for header, sequence in parse_fasta(args.fasta):
            handle.write(f">{args.decoy_prefix}{header}\n{reverse_protein_decoy(sequence)}\n")
    if not count:
        raise ValueError("Input FASTA contains no proteins")
    print(f"target_proteins={count}")
    print(f"decoy_proteins={count}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
