"""Read prepared pilot files independently and summarize data consistency.

This audits stored data and arithmetic, not ecological accuracy. Requires the
project CPU environment; use run_windows_cpu.ps1 on Windows to isolate PROJ.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_grid(path):
    with rasterio.open(path) as ds:
        return ds.read(1, masked=True), (ds.shape, ds.crs.to_epsg(), tuple(ds.transform)), ds.nodata


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--maxent-inputs", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    for folder in (args.observations, args.maxent_inputs):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for name, item in manifest["outputs"].items():
            require(digest(folder / name) == item["sha256"], "Changed prepared CSV: " + name)
    source = pd.read_csv(args.observations / "waterbird_source_records.csv", low_memory=False)
    candidate = pd.read_csv(args.observations / "waterbird_occurrences_candidate.csv", low_memory=False)
    require(source.record_id.is_unique and candidate.record_id.is_unique, "Duplicate source record IDs")
    require(set(candidate.record_id).issubset(source.record_id), "Candidate source linkage missing")
    require(not source.eligible_for_gate_training.any(), "Unexpected gate eligibility")
    manifest = json.loads((args.baselines / "manifest.json").read_text(encoding="utf-8"))
    require(not manifest["gate_trained"] and not manifest["restoration_outputs_generated"], "Unexpected training/restoration output")
    hq, grid, _ = load_grid(args.baselines / "quality_c_ref_utm49_100m.tif")
    require(grid[1] == 32649, "Wrong reference projection")
    reports = {}
    for season in ("spring", "summer", "autumn", "winter"):
        m, mgrid, _ = load_grid(args.baselines / ("m_" + season + "_utm49_100m.tif"))
        require(mgrid == grid, "MaxEnt/HQ alignment differs")
        valid = ~(np.ma.getmaskarray(m) | np.ma.getmaskarray(hq))
        require(int(valid.sum()) == manifest["season_results"][season]["common_valid_pixels"], "Manifest valid count differs")
        expected = {"geometric_050": np.sqrt(np.clip(m.data, 0, None) * np.clip(hq.data, 0, None)),
                    "linear_050": 0.5 * (m.data + hq.data), "minimum": np.minimum(m.data, hq.data)}
        methods = {}
        for method, values in expected.items():
            path = args.baselines / ("conservation_" + season + "_" + method + ".tif")
            actual, actualgrid, nodata = load_grid(path)
            require(actualgrid == grid and nodata == -9999, "Wrong output grid/NoData")
            require(np.array_equal(~np.ma.getmaskarray(actual), valid), "Output valid mask differs")
            require(np.allclose(actual.data[valid], values[valid], atol=1e-7, rtol=1e-6), "Baseline arithmetic differs")
            stored = next(item for item in manifest["season_results"][season]["outputs"] if Path(item["path"]).name == path.name)
            require(digest(path) == stored["sha256"], "Baseline hash differs")
            methods[method] = {"pixels": int(valid.sum()), "quantiles_05_50_95": np.quantile(actual.data[valid], [0.05, 0.5, 0.95]).tolist()}
        presence = pd.read_csv(args.maxent_inputs / ("occurrence_cells_" + season + ".csv"))
        background = pd.read_csv(args.maxent_inputs / ("target_group_background_cells_" + season + ".csv"))
        require(presence.native_cell_id.is_unique and background.native_cell_id.is_unique, "Duplicated analysis cells")
        require(set(presence.native_cell_id).issubset(background.native_cell_id), "Positive waterbird cells should be in all-bird candidate background")
        require(presence.split_role.eq("NOT_ASSIGNED").all(), "Unexpected final split assignment")
        reports[season] = {"unique_presence_cells": len(presence), "background_candidate_cells": len(background), "baselines": methods}
    result = {
        "status": "PREPARED_DATA_HASH_GRID_MASK_AND_ARITHMETIC_AUDIT_PASSED",
        "ecological_accuracy_established": False, "real_models_retrained": False,
        "source_rows": len(source), "positive_candidates": len(candidate), "seasons": reports,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "torch", "rasterio", "affine")}},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "seasons": {key: value["unique_presence_cells"] for key, value in reports.items()}}))


if __name__ == "__main__":
    main()
