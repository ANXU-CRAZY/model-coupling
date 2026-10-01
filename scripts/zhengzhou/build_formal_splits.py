"""Freeze one shared full-grid spatial partition before any MaxEnt fitting.

Run with the project CPU environment. The prepared-input directory contains
grid.json/common_valid_mask.npy and season/{presence,B0,B1}.csv. Output must be
new; locked test values are never evaluated or used to balance the partitions.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wetland_coupling.maxent_splits import (
    SEASONS, build_spatial_split_plan, file_hash, save_split_plan,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--site-cells", type=Path,
                        help="Optional reviewed source-site links: site_key,raster_row,raster_col")
    parser.add_argument("--spatial-audit", type=Path,
                        help="Read-only diagnostic evidence directory, recorded as lineage")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    started = datetime.now(timezone.utc).isoformat()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    grid_path, mask_path = args.inputs / "grid.json", args.inputs / "common_valid_mask.npy"
    input_manifest_path = args.inputs / "manifest.json"
    if not input_manifest_path.exists():
        raise ValueError("Prepared-input manifest is required before freezing a split")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    declared_outputs = input_manifest.get("outputs", {})
    if isinstance(declared_outputs, dict):
        for relative, evidence in declared_outputs.items():
            if isinstance(evidence, dict) and "sha256" in evidence:
                path = args.inputs / relative
                if not path.exists() or file_hash(path) != evidence["sha256"]:
                    raise ValueError("Prepared-input output hash mismatch: " + relative)
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    valid = np.load(mask_path, allow_pickle=False)
    presence, backgrounds, lineage = {}, {}, {}
    for path in (grid_path, mask_path, input_manifest_path):
        lineage[str(path.resolve())] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    for season in SEASONS:
        folder = args.inputs / season
        names = json.loads((folder / "names.json").read_text(encoding="utf-8"))
        if not isinstance(names, list) or len(names) != len(set(names)) or not names:
            raise ValueError("Predictor names must be a nonempty unique list")
        presence[season] = pd.read_csv(folder / "presence.csv", low_memory=False)
        backgrounds[season] = {scheme: pd.read_csv(folder / (scheme + ".csv"), low_memory=False)
                               for scheme in ("B0", "B1")}
        for table in [presence[season], *backgrounds[season].values()]:
            if not set(names).issubset(table.columns) or not np.isfinite(table[names].to_numpy(float)).all():
                raise ValueError("Missing/nonfinite predictor in " + season)
        for name in ("presence.csv", "B0.csv", "B1.csv", "names.json"):
            path = folder / name
            lineage[str(path.resolve())] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    site_cells = pd.read_csv(args.site_cells) if args.site_cells else None
    if args.site_cells:
        lineage[str(args.site_cells.resolve())] = {"bytes": args.site_cells.stat().st_size, "sha256": file_hash(args.site_cells)}
    plan = build_spatial_split_plan(presence, backgrounds, grid, valid, config, site_cells)
    plan["source_table_inventory"] = lineage
    plan["spatial_audit_directory"] = str(args.spatial_audit.resolve()) if args.spatial_audit else None
    plan["start_utc"] = started
    plan["end_utc"] = datetime.now(timezone.utc).isoformat()
    # New lineage fields must be part of the frozen split hash. Runtime arrays
    # and duplicated convenience fields are intentionally excluded.
    from wetland_coupling.maxent_splits import canonical_hash, _manifest_view
    frozen = _manifest_view(plan); frozen.pop("split_hash")
    plan["split_hash"] = canonical_hash(frozen)
    manifest = save_split_plan(plan, args.out, args.config, args.inputs)
    print(json.dumps({"status": manifest["status"], "split_hash": plan["split_hash"],
                      "locked_counts": plan["final_counts"],
                      "outer_counts": {fold["fold"]: fold["counts"] for fold in plan["outer_folds"]},
                      "locked_test_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
