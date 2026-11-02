import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import lmdb
except ImportError:
    lmdb = None

import numpy as np
import pandas as pd
import torch
from pyteomics import mgf
from torch.utils.data import Dataset
from tqdm import tqdm

from ShiftSearch.constants import H2O, PROTON, RESIDUE_MASS


LMDB_MAP_SIZE = 3687091200
LMDB_CACHE_VERSION = "3"
PAIR_LMDB_CACHE_VERSION = "2"
LMDB_COMMIT_INTERVAL = 1000
LMDB_ENV_CACHE: Dict[str, Any] = {}


class PositivePairPSMDataset(Dataset):
    def __init__(
        self,
        mgf_path: str,
        config: Dict,
        norm: bool = True,
        cache_backend: str = "lmdb",
        cache_path: Optional[str] = None,
        rebuild_cache: bool = False,
    ) -> None:
        super().__init__()
        self.mgf_path = Path(mgf_path).resolve()
        self.config = config
        self.max_n_peaks = config["max_npeaks"]
        self.cpu_pre_topk = max(int(config.get("cpu_pre_topk", 1024)), self.max_n_peaks)
        self.max_mz = config["max_mz"]
        self.max_pep_len = config["max_pep_len"]
        self.min_pep_len = config.get("min_pep_len", 0)
        self.min_n_peaks = config.get("min_npeaks", 0)
        self.convert_y_to_b = config.get("convert_y_to_b", True)
        self.merge_tolerance_ppm = float(config.get("merge_tolerance_ppm", 10.0))
        self.norm = norm
        self.cache_backend = cache_backend.lower()
        self.lmdb_path: Optional[Path] = None
        self._env = None

        if self.cache_backend != "lmdb":
            raise ValueError("PositivePairPSMDataset now supports only cache_backend='lmdb'")

        if lmdb is None:
            raise ImportError("lmdb is required when cache_backend='lmdb'")

        self.lmdb_path = Path(cache_path).resolve() if cache_path is not None else self.mgf_path.with_suffix(".lmdb")
        if rebuild_cache:
            self._remove_cache_files(self.lmdb_path)
        if not self.lmdb_path.exists():
            self._build_lmdb_cache()

        metadata = self._load_lmdb_metadata()
        source = metadata["source"]
        if source != str(self.mgf_path):
            import warnings
            warnings.warn(
                f"LMDB source mismatch (expected {self.mgf_path}, found {source}). "
                "This is expected when using pre-built LMDB caches."
            )
        if metadata["cache_version"] != LMDB_CACHE_VERSION:
            import warnings
            warnings.warn(
                f"LMDB cache version mismatch (expected {LMDB_CACHE_VERSION}, found {metadata['cache_version']}). "
                "Using existing cache — if issues occur, set rebuild_cache=true."
            )
        self.length = metadata["length"]

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    def __del__(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _parse_annotated_seq(self, annotated_seq: str) -> Tuple[str, np.ndarray]:
        residues: List[str] = []
        residue_shifts: List[float] = []
        pending_nterm_shift = 0.0
        idx = 0

        while idx < len(annotated_seq):
            token = annotated_seq[idx]
            if token.isalpha():
                is_first_residue = len(residues) == 0
                residues.append(token)
                residue_shifts.append(0.0)
                if is_first_residue and pending_nterm_shift != 0.0:
                    residue_shifts[0] += pending_nterm_shift
                    pending_nterm_shift = 0.0
                idx += 1
                continue

            if token in "+-":
                end = idx + 1
                while end < len(annotated_seq) and (annotated_seq[end].isdigit() or annotated_seq[end] == "."):
                    end += 1
                if end == idx + 1:
                    raise ValueError(f"Malformed mass shift in sequence: {annotated_seq}")
                shift = float(annotated_seq[idx:end])
                if residues:
                    residue_shifts[-1] += shift
                else:
                    pending_nterm_shift += shift
                idx = end
                continue

            raise ValueError(f"Unsupported token {token!r} in sequence: {annotated_seq}")

        if not residues:
            raise ValueError(f"No residues parsed from sequence: {annotated_seq}")

        return "".join(residues), np.asarray(residue_shifts, dtype=np.float32)

    def _iter_filtered_records(self) -> Iterator[Dict[str, Any]]:
        with mgf.MGF(str(self.mgf_path)) as spectra:
            for spectrum in spectra:
                params = spectrum.get("params") or {}
                record = self._build_spectrum_record_from_params(params, spectrum)
                if record is None:
                    continue

                annotated_seq = record["annotated_seq"]
                seq, residue_shifts = self._parse_annotated_seq(annotated_seq)
                if len(seq) < self.min_pep_len:
                    continue

                record["seq"] = seq
                record["residue_shifts"] = residue_shifts
                yield record

    def _build_spectrum_record_from_params(self, params: Dict[str, Any], spectrum: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = params.get("title")
        scan = params.get("scan") or params.get("scans") or title or ""
        mz_array = np.asarray(spectrum.get("m/z array"), dtype=np.float32)
        intensity_array = np.asarray(spectrum.get("intensity array"), dtype=np.float32)
        if mz_array.size < self.min_n_peaks:
            return None

        pre_mz = float(params["pepmass"][0])
        charge = int(params["charge"][0])
        annotated_seq = str(params.get("seq", "")).strip()
        if not annotated_seq:
            raise KeyError("Could not find peptide annotation in MGF params['seq']")

        filename = ""
        if title is not None:
            title_text = str(title)
            if title_text.startswith("File:") and "_Scan:" in title_text:
                filename = title_text[len("File:") :].rsplit("_Scan:", 1)[0]

        return {
            "mz_array": mz_array,
            "intensity_array": intensity_array,
            "charge": charge,
            "pre_mz": np.float32(pre_mz),
            "annotated_seq": annotated_seq,
            "scan": str(scan),
            "title": "" if title is None else str(title),
            "filename": filename,
        }

    def _build_spectrum_key(self, title: str, filename: str, scan: str) -> str:
        if title:
            return f"title::{title}"
        return f"file_scan::{filename}::{scan}"

    def _remove_cache_files(self, cache_path: Path) -> None:
        for path in (cache_path, Path(f"{cache_path}-lock")):
            if path.exists():
                path.unlink()

    def _build_lmdb_cache(self) -> None:
        assert self.lmdb_path is not None
        self.lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(str(self.lmdb_path), map_size=LMDB_MAP_SIZE, subdir=False, readonly=False, lock=True)
        txn = env.begin(write=True)
        count = 0

        try:
            with mgf.MGF(str(self.mgf_path)) as spectra:
                progress = tqdm(spectra, desc=f"build spectrum lmdb {self.mgf_path.name}", unit="spectra")
                for spectrum in progress:
                    params = spectrum.get("params") or {}
                    record = self._build_spectrum_record_from_params(params, spectrum)
                    if record is None:
                        continue

                    lmdb_record = {
                        "mz_array": record["mz_array"],
                        "intensity_array": record["intensity_array"],
                        "charge": record["charge"],
                        "pre_mz": record["pre_mz"],
                    }

                    key = self._build_spectrum_key(record["title"], record["filename"], record["scan"])
                    txn.put(key.encode(), pickle.dumps(lmdb_record, protocol=pickle.HIGHEST_PROTOCOL))
                    count += 1
                    if count % LMDB_COMMIT_INTERVAL == 0:
                        txn.commit()
                        txn = env.begin(write=True)

            txn.put(b"__length__", str(count).encode())
            txn.put(b"__source__", str(self.mgf_path).encode())
            txn.put(b"__cache_version__", LMDB_CACHE_VERSION.encode())
            txn.commit()
        finally:
            env.close()

    def _load_lmdb_metadata(self) -> Dict[str, Any]:
        assert self.lmdb_path is not None
        env = lmdb.open(str(self.lmdb_path), subdir=False, readonly=True, lock=False)
        try:
            with env.begin() as txn:
                length = txn.get(b"__length__")
                source = txn.get(b"__source__")
                cache_version = txn.get(b"__cache_version__")
        finally:
            env.close()

        if length is None or source is None or cache_version is None:
            raise ValueError("Invalid LMDB cache metadata; rebuild the cache")

        return {
            "length": int(length.decode()),
            "source": source.decode(),
            "cache_version": cache_version.decode(),
        }

    def _get_env(self):
        if self._env is None:
            assert self.lmdb_path is not None
            env_key = str(self.lmdb_path)
            if env_key not in LMDB_ENV_CACHE:
                LMDB_ENV_CACHE[env_key] = lmdb.open(env_key, subdir=False, readonly=True, lock=False)
            self._env = LMDB_ENV_CACHE[env_key]
        return self._env

    def _get_record(self, index: int) -> Dict[str, Any]:
        raise RuntimeError("PositivePairPSMDataset does not support indexed reads; use PairTablePSMDataset for training data")

    def _build_prefix_masses(self, peptide: str, residue_shifts: np.ndarray) -> np.ndarray:
        prefix_masses = np.zeros(self.max_pep_len + 1, dtype=np.float32)
        running_mass = 0.0
        peptide_len = min(len(peptide), self.max_pep_len)
        for idx in range(peptide_len):
            residue = peptide[idx]
            if residue not in RESIDUE_MASS:
                raise KeyError(f"Unsupported residue {residue!r} in peptide {peptide!r}")
            running_mass += RESIDUE_MASS[residue] + float(residue_shifts[idx])
            prefix_masses[idx + 1] = running_mass
        return prefix_masses

    def _merge_close_peaks(self, peak_masses: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if peak_masses.size <= 1:
            return peak_masses, intensities

        order = np.argsort(peak_masses)
        peak_masses = peak_masses[order]
        intensities = intensities[order]

        kept_masses = [float(peak_masses[0])]
        kept_intensities = [float(intensities[0])]

        for mass, intensity in zip(peak_masses[1:], intensities[1:]):
            prev_mass = kept_masses[-1]
            ppm = abs(float(mass) - prev_mass) / prev_mass * 1e6 if prev_mass > 0 else np.inf
            if ppm <= self.merge_tolerance_ppm:
                if float(intensity) > kept_intensities[-1]:
                    kept_masses[-1] = float(mass)
                    kept_intensities[-1] = float(intensity)
            else:
                kept_masses.append(float(mass))
                kept_intensities.append(float(intensity))

        return np.asarray(kept_masses, dtype=np.float32), np.asarray(kept_intensities, dtype=np.float32)

    def _select_top_peaks(self, peak_masses: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        valid = (peak_masses > 0) & (peak_masses < self.max_mz) & (intensities > 0)
        peak_masses = peak_masses[valid]
        intensities = intensities[valid]
        if peak_masses.size == 0:
            return peak_masses.astype(np.float32), intensities.astype(np.float32)

        if peak_masses.size > self.max_n_peaks:
            top_idx = np.argsort(intensities)[-self.max_n_peaks:]
            peak_masses = peak_masses[top_idx]
            intensities = intensities[top_idx]

        order = np.argsort(peak_masses)
        peak_masses = peak_masses[order]
        intensities = intensities[order]
        return peak_masses.astype(np.float32), intensities.astype(np.float32)

    def _build_peak_representation(self, mz_array: np.ndarray, intensities: np.ndarray, total_residue_mass: float) -> Tuple[np.ndarray, np.ndarray]:
        valid = mz_array < self.max_mz
        mz_array = np.asarray(mz_array[valid], dtype=np.float32)
        intensities = np.asarray(intensities[valid], dtype=np.float32)
        if mz_array.size == 0:
            return mz_array, intensities

        if self.norm:
            intensities = np.sqrt(np.clip(intensities, a_min=0.0, a_max=None))
            intensity_sum = float(intensities.sum())
            if intensity_sum > 0:
                intensities = intensities / intensity_sum

        original_b_like = mz_array - PROTON
        converted_b_like = total_residue_mass + H2O - original_b_like

        original_b_like, original_intensities = self._merge_close_peaks(original_b_like, intensities)
        peak_masses = original_b_like
        peak_intensities = original_intensities

        if self.convert_y_to_b:
            converted_b_like, converted_intensities = self._merge_close_peaks(converted_b_like, intensities)
            peak_masses = np.concatenate([original_b_like, converted_b_like])
            peak_intensities = np.concatenate([original_intensities, converted_intensities])

        return self._select_top_peaks(peak_masses, peak_intensities)

    def _preprocess_peaks_for_cache(self, mz_array: np.ndarray, intensities: np.ndarray, pre_mz: float, charge: int) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess peaks: filter by mz, y<->b conversion, sort by mass. No merge. Normalization done after top-k."""
        valid = mz_array < self.max_mz
        mz_array = np.asarray(mz_array[valid], dtype=np.float32)
        intensities = np.asarray(intensities[valid], dtype=np.float32)
        if mz_array.size == 0:
            return mz_array, intensities

        b_like = mz_array - PROTON
        peak_masses = b_like
        peak_intensities = intensities

        if self.convert_y_to_b:
            neutral_mass = pre_mz * charge - charge * PROTON
            y_like = neutral_mass - mz_array + PROTON
            peak_masses = np.concatenate([b_like, y_like])
            peak_intensities = np.concatenate([intensities, intensities])

        valid = (peak_masses > 0) & (peak_masses < self.max_mz)
        peak_masses = peak_masses[valid]
        peak_intensities = peak_intensities[valid]

        order = np.argsort(peak_masses)
        return peak_masses[order].astype(np.float32), peak_intensities[order].astype(np.float32)

    def _build_pair_tensor(
        self,
        prefix_masses: np.ndarray,
        peak_masses: np.ndarray,
        peak_intensities: np.ndarray,
        peptide_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pair_tensor = np.zeros((3, self.max_pep_len - 1, self.max_n_peaks), dtype=np.float32)
        pair_mask = np.zeros((self.max_pep_len - 1, self.max_n_peaks), dtype=bool)

        n_rows = max(0, min(peptide_len - 1, self.max_pep_len - 1))
        n_cols = min(len(peak_masses), self.max_n_peaks)
        if n_rows == 0 or n_cols == 0:
            return pair_tensor, pair_mask

        internal_prefix_masses = prefix_masses[1 : n_rows + 1]
        mass_diff = peak_masses[:n_cols][None, :] - internal_prefix_masses[:, None]
        intensity_channel = np.broadcast_to(peak_intensities[:n_cols][None, :], (n_rows, n_cols)).astype(np.float32)
        mask_channel = np.ones((n_rows, n_cols), dtype=np.float32)

        pair_tensor[0, :n_rows, :n_cols] = mass_diff.astype(np.float32)
        pair_tensor[1, :n_rows, :n_cols] = intensity_channel
        pair_tensor[2, :n_rows, :n_cols] = mask_channel
        pair_mask[:n_rows, :n_cols] = True
        return pair_tensor, pair_mask

    def __getitem__(self, index: int) -> Dict:
        record = self._get_record(index)
        peptide = record["seq"][: self.max_pep_len]
        residue_shifts = record["residue_shifts"][: self.max_pep_len]
        prefix_masses = self._build_prefix_masses(peptide, residue_shifts)
        peak_masses, peak_intensities = self._build_peak_representation(
            record["mz_array"],
            record["intensity_array"],
            float(prefix_masses[min(len(peptide), self.max_pep_len)]),
        )
        pair_tensor, pair_mask = self._build_pair_tensor(prefix_masses, peak_masses, peak_intensities, len(peptide))

        return {
            "pair_tensor": torch.from_numpy(pair_tensor),
            "pair_mask": torch.from_numpy(pair_mask),
            "label": torch.tensor([1.0], dtype=torch.float32),
            "seq": peptide,
            "annotated_seq": record["annotated_seq"],
            "charge": torch.tensor(record["charge"], dtype=torch.long),
            "pre_mz": torch.tensor(record["pre_mz"], dtype=torch.float32),
            "scan": record["scan"],
            "title": record["title"],
            "npeaks": torch.tensor(len(peak_masses), dtype=torch.long),
            "peptide_length": torch.tensor(len(peptide), dtype=torch.long),
        }

    def __len__(self) -> int:
        return self.length


class PairTablePSMDataset(PositivePairPSMDataset):
    def __init__(
        self,
        pair_table_path: str,
        mgf_path: str,
        config: Dict,
        norm: bool = True,
        cache_backend: str = "lmdb",
        cache_path: Optional[str] = None,
        rebuild_cache: bool = False,
    ) -> None:
        self.pair_table_path = Path(pair_table_path).resolve()
        self.pair_lmdb_path = self.pair_table_path.with_suffix(".lmdb")
        self._pair_env = None
        super().__init__(
            mgf_path=mgf_path,
            config=config,
            norm=norm,
            cache_backend=cache_backend,
            cache_path=cache_path,
            rebuild_cache=rebuild_cache,
        )
        if rebuild_cache and self.pair_lmdb_path.exists():
            self._remove_cache_files(self.pair_lmdb_path)
        if not self.pair_lmdb_path.exists():
            self._build_pair_lmdb()
        pair_metadata = self._load_pair_lmdb_metadata()
        if pair_metadata["source"] != str(self.pair_table_path):
            import warnings
            warnings.warn(
                f"Pair LMDB source mismatch (expected {self.pair_table_path}, found {pair_metadata['source']}). "
                "This is expected when using pre-built LMDB caches."
            )
        if pair_metadata["cache_version"] != PAIR_LMDB_CACHE_VERSION:
            raise ValueError("Pair LMDB cache version mismatch; rebuild the cache")
        self.length = pair_metadata["length"]

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["_pair_env"] = None
        return state

    def __del__(self) -> None:
        super().__del__()
        if self._pair_env is not None:
            self._pair_env.close()
            self._pair_env = None

    def _build_pair_lmdb(self) -> None:
        env = lmdb.open(str(self.pair_lmdb_path), map_size=LMDB_MAP_SIZE, subdir=False, readonly=False, lock=True)
        txn = env.begin(write=True)
        count = 0
        try:
            df = pd.read_csv(self.pair_table_path, sep="\t", usecols=["title", "filename", "scan", "peptide_sequence", "label"])
            progress = tqdm(df.itertuples(index=False), total=len(df), desc=f"build pair lmdb {self.pair_table_path.name}", unit="rows")
            for row in progress:
                peptide_sequence = str(row.peptide_sequence)
                seq, residue_shifts = self._parse_annotated_seq(peptide_sequence)
                if len(seq) < self.min_pep_len or len(seq) > self.max_pep_len:
                    continue
                prefix_masses = self._build_prefix_masses(seq, residue_shifts)
                record = {
                    "title": "" if pd.isna(row.title) else str(row.title),
                    "filename": "" if pd.isna(row.filename) else str(row.filename),
                    "scan": "" if pd.isna(row.scan) else str(row.scan),
                    "peptide_sequence": peptide_sequence,
                    "label": float(row.label),
                    "prefix_masses": prefix_masses,
                }
                txn.put(str(count).encode(), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                count += 1
                if count % LMDB_COMMIT_INTERVAL == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            txn.put(b"__length__", str(count).encode())
            txn.put(b"__source__", str(self.pair_table_path).encode())
            txn.put(b"__cache_version__", PAIR_LMDB_CACHE_VERSION.encode())
            txn.commit()
        finally:
            env.close()

    def _load_pair_lmdb_metadata(self) -> Dict[str, Any]:
        env = lmdb.open(str(self.pair_lmdb_path), subdir=False, readonly=True, lock=False)
        try:
            with env.begin() as txn:
                length = txn.get(b"__length__")
                source = txn.get(b"__source__")
                cache_version = txn.get(b"__cache_version__")
        finally:
            env.close()

        if length is None or source is None or cache_version is None:
            raise ValueError("Invalid pair LMDB metadata; rebuild the cache")

        return {
            "length": int(length.decode()),
            "source": source.decode(),
            "cache_version": cache_version.decode(),
        }

    def _get_pair_env(self):
        if self._pair_env is None:
            env_key = str(self.pair_lmdb_path)
            if env_key not in LMDB_ENV_CACHE:
                LMDB_ENV_CACHE[env_key] = lmdb.open(env_key, subdir=False, readonly=True, lock=False)
            self._pair_env = LMDB_ENV_CACHE[env_key]
        return self._pair_env

    def _get_pair_row(self, index: int) -> Dict[str, Any]:
        env = self._get_pair_env()
        with env.begin() as txn:
            buffer = txn.get(str(index).encode())
        if buffer is None:
            raise IndexError(index)
        return pickle.loads(buffer)

    def _get_record(self, index: int) -> Dict[str, Any]:
        pair_row = self._get_pair_row(index)
        title = pair_row["title"]
        filename = pair_row["filename"]
        scan = pair_row["scan"]
        key = self._build_spectrum_key(title, filename, scan)

        env = self._get_env()
        with env.begin() as txn:
            buffer = txn.get(key.encode())
        if buffer is None:
            raise KeyError(f"Spectrum key not found in LMDB: {key}")

        spectrum_record = pickle.loads(buffer)
        spectrum_record["annotated_seq"] = str(pair_row["peptide_sequence"])
        spectrum_record["label"] = float(pair_row["label"])
        spectrum_record["prefix_masses"] = pair_row["prefix_masses"]
        spectrum_record["scan"] = pair_row["scan"]
        spectrum_record["title"] = pair_row["title"]
        return spectrum_record

    def __getitem__(self, index: int) -> Dict:
        record = self._get_record(index)
        peptide_text = record["annotated_seq"]
        prefix_masses = record["prefix_masses"][: self.max_pep_len + 1]
        peptide_length = int(np.count_nonzero(prefix_masses))

        mz_array = np.asarray(record["mz_array"], dtype=np.float32)
        intensity_array = np.asarray(record["intensity_array"], dtype=np.float32)
        valid = mz_array < self.max_mz
        mz_array = mz_array[valid]
        intensity_array = intensity_array[valid]

        if mz_array.size > self.cpu_pre_topk:
            top_idx = np.argpartition(intensity_array, -self.cpu_pre_topk)[-self.cpu_pre_topk:]
            top_idx = np.sort(top_idx)
            mz_array = mz_array[top_idx]
            intensity_array = intensity_array[top_idx]

        raw_peak_count = mz_array.size
        padded_mz = np.zeros(self.cpu_pre_topk, dtype=np.float32)
        padded_intensity = np.zeros(self.cpu_pre_topk, dtype=np.float32)
        raw_peak_mask = np.zeros(self.cpu_pre_topk, dtype=bool)
        padded_mz[:raw_peak_count] = mz_array
        padded_intensity[:raw_peak_count] = intensity_array
        raw_peak_mask[:raw_peak_count] = True

        return {
            "label": torch.tensor([record["label"]], dtype=torch.float32),
            "seq": peptide_text,
            "annotated_seq": record["annotated_seq"],
            "mz_array": torch.from_numpy(padded_mz),
            "intensity_array": torch.from_numpy(padded_intensity),
            "raw_peak_mask": torch.from_numpy(raw_peak_mask),
            "charge": torch.tensor(record["charge"], dtype=torch.long),
            "pre_mz": torch.tensor(record["pre_mz"], dtype=torch.float32),
            "scan": record["scan"],
            "title": record["title"],
            "prefix_masses": torch.from_numpy(prefix_masses.astype(np.float32, copy=False)),
            "peptide_length": torch.tensor(peptide_length, dtype=torch.long),
        }
