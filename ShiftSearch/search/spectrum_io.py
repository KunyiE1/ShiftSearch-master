from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
from pyteomics import mgf


@dataclass
class SpectrumRecord:
    spectrum_index: int
    title: str
    scan: str
    charge: int
    pre_mz: float
    target_sequence: str
    raw_mz: np.ndarray
    raw_intensity: np.ndarray
    raw_peak_mask: np.ndarray


def _parse_charge(raw_charge) -> int:
    if isinstance(raw_charge, (list, tuple)) and len(raw_charge) > 0:
        raw_charge = raw_charge[0]
    text = str(raw_charge).strip()
    text = text.replace("+", "").replace("-", "")
    return int(text)


def prepare_raw_peaks(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    max_mz: float,
    cpu_pre_topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mz_array = np.asarray(mz_array, dtype=np.float32)
    intensity_array = np.asarray(intensity_array, dtype=np.float32)
    valid = mz_array < max_mz
    mz_array = mz_array[valid]
    intensity_array = intensity_array[valid]

    if mz_array.size > cpu_pre_topk:
        top_idx = np.argpartition(intensity_array, -cpu_pre_topk)[-cpu_pre_topk:]
        top_idx = np.sort(top_idx)
        mz_array = mz_array[top_idx]
        intensity_array = intensity_array[top_idx]

    padded_mz = np.zeros(cpu_pre_topk, dtype=np.float32)
    padded_intensity = np.zeros(cpu_pre_topk, dtype=np.float32)
    raw_peak_mask = np.zeros(cpu_pre_topk, dtype=bool)
    raw_peak_count = mz_array.size
    padded_mz[:raw_peak_count] = mz_array
    padded_intensity[:raw_peak_count] = intensity_array
    raw_peak_mask[:raw_peak_count] = True
    return padded_mz, padded_intensity, raw_peak_mask


def load_mgf_spectra(mgf_path: str | Path, max_mz: float, cpu_pre_topk: int, min_peaks: int = 0) -> Iterator[SpectrumRecord]:
    with mgf.MGF(str(mgf_path)) as reader:
        for spectrum_index, spectrum in enumerate(reader):
            params = spectrum.get("params") or {}
            title = str(params.get("title", ""))
            scan = str(params.get("scan") or params.get("scans") or title or spectrum_index)
            charge = _parse_charge(params["charge"])
            pre_mz = float(params["pepmass"][0])
            target_sequence = str(params.get("seq") or params.get("sequence") or params.get("peptide") or "").strip()
            raw_mz, raw_intensity, raw_mask = prepare_raw_peaks(
                np.asarray(spectrum.get("m/z array"), dtype=np.float32),
                np.asarray(spectrum.get("intensity array"), dtype=np.float32),
                max_mz=max_mz,
                cpu_pre_topk=cpu_pre_topk,
            )
            if int(raw_mask.sum()) < min_peaks:
                continue
            yield SpectrumRecord(
                spectrum_index=spectrum_index,
                title=title,
                scan=scan,
                charge=charge,
                pre_mz=pre_mz,
                target_sequence=target_sequence,
                raw_mz=raw_mz,
                raw_intensity=raw_intensity,
                raw_peak_mask=raw_mask,
            )
