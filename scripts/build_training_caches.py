"""Build spectrum and pair LMDB caches from prepared data.

Calls the production dataset classes directly. Existing caches are reused;
negative-pair construction and dataset splitting are not changed here.
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
    parser.add_argument("--data", required=True, type=Path, help="Annotated MGF or an existing spectrum LMDB file.")
    parser.add_argument("--train-pairs", required=True, type=Path, help="Prepared train TSV or pair LMDB.")
    parser.add_argument("--val-pairs", type=Path, help="Prepared validation TSV or pair LMDB.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--spectrum-cache", type=Path, help="Optional spectrum-cache location; default replaces --data suffix by .lmdb.")
    parser.add_argument("--map-size-gb", type=float, help="LMDB maximum map size in GiB for new caches; use sufficient headroom for large datasets.")
    parser.add_argument("--check-items", type=int, default=8, help="Read this many initial pairs from each split, plus its last pair, to check spectrum-key links.")
    args = parser.parse_args()
    if args.check_items < 0 or (args.map_size_gb is not None and args.map_size_gb <= 0):
        raise ValueError("Invalid check count or LMDB map size")
    for path in (args.data, args.train_pairs, args.val_pairs, args.config):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    import yaml
    import ShiftSearch.psm.dataset as dataset_module

    if args.map_size_gb is not None:
        dataset_module.LMDB_MAP_SIZE = int(args.map_size_gb * 1024 ** 3)
    with args.config.open(encoding="utf-8") as handle:
        meta = yaml.safe_load(handle)["Meta"]
    cache = args.spectrum_cache or args.data.with_suffix(".lmdb")
    if args.data.suffix.lower() == ".lmdb" and cache.resolve() != args.data.resolve():
        raise ValueError("For --data LMDB, --spectrum-cache must refer to that same file")
    if args.data.suffix.lower() != ".lmdb" and cache.resolve() == args.data.resolve():
        raise ValueError("The spectrum cache must differ from the input MGF")
    cache.parent.mkdir(parents=True, exist_ok=True)
    spectra = dataset_module.PositivePairPSMDataset(str(args.data), meta, cache_path=str(cache), rebuild_cache=False)
    datasets = []
    for split, path in (("train", args.train_pairs), ("validation", args.val_pairs)):
        if path is None:
            continue
        data = dataset_module.PairTablePSMDataset(str(path), str(args.data), meta, cache_path=str(cache), rebuild_cache=False)
        datasets.append((split, data))
        if len(data) == 0:
            raise ValueError(f"The {split} pair table has no supported rows")
    # Finish opening/validating metadata before indexed reads open the shared
    # spectrum environment. Python LMDB rejects reopening an already-open path
    # in the same process, including a second metadata-only open.
    for split, data in datasets:
        indices = list(range(min(args.check_items, len(data))))
        if args.check_items and len(data) - 1 not in indices:
            indices.append(len(data) - 1)
        for index in indices:
            data[index]
        print(f"{split}_pairs={len(data)} checked={len(indices)} cache={data.pair_lmdb_path}")
    print(f"spectrum_records={len(spectra)}")
    print(f"spectrum_cache={cache.resolve()}")
    print("Caches are ready; training can use the .lmdb paths directly.")


if __name__ == "__main__":
    main()
