"""Independent data checks for the 2026-10-02 WorkBuddy handoff.

No empirical model fitting, locked-test score use or ecological parameter
assignment. The four historical phase outputs are preserved as audit inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio

from scripts.zhengzhou.export_maxent_oof import verify_split_inventory
from wetland_coupling.parameter_diagnostics import ols_vif, rank_correlation, record_input


SEASONS = ("spring", "summer", "autumn", "winter")
THREATS = ("urban_structure", "human_activity", "night_light")


def read_layer(path, reference, inventory):
    record_input(path, inventory)
    with rasterio.open(path) as ds:
        grid = (ds.shape, ds.transform, ds.crs)
        if reference is not None and grid != reference:
            raise ValueError("Diagnostic raster alignment differs: " + str(path))
        data = ds.read(1).astype(float)
        mask = (ds.read_masks(1) != 0) & np.isfinite(data)
    return data, mask, grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pilot = root / "local_work/zhengzhou_pilot"
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    inputs = {}
    handoff = root / "HANDOFF_WORKBUDDY_20261002.md"
    record_input(handoff, inputs)
    split = pilot / "maxent_formal_v1_001/splits"
    verify_split_inventory(split)
    for name in ("manifest.json", "split_plan.json", "split_masks.npz"):
        record_input(split / name, inputs)
    with np.load(split / "split_masks.npz", allow_pickle=False) as z:
        common = z["common_valid"].astype(bool)
    base = pilot / "archival_baselines_003"
    lulc, lulc_valid, grid = read_layer(base / "lulc_cur_utm49_100m.tif", None, inputs)
    area_per_pixel = abs(grid[1].a * grid[1].e - grid[1].b * grid[1].d) / 1.e6
    arrays, masks = {}, {}
    for threat in THREATS:
        arrays[threat], masks[threat], _ = read_layer(base / (threat + "_utm49_100m.tif"), grid, inputs)
    joint = common & np.logical_and.reduce(list(masks.values()))
    design = np.column_stack([arrays[name][joint] for name in THREATS])
    vif = dict(zip(THREATS, ols_vif(design)))
    overlap = []
    env = pilot / "supermap/environment_exports/spr_select"
    for threat, filename in (("night_light", "nightlight_spring.tif"),
                             ("urban_structure", "builtarea_spring.tif"),
                             ("urban_structure", "largebuildings_spring.tif"),
                             ("human_activity", "builtarea_spring.tif")):
        data, valid, _ = read_layer(env / filename, grid, inputs)
        use = common & masks[threat] & valid
        overlap.append({"threat": threat, "environment_variable": filename,
                        "paired_pixels": int(use.sum()),
                        "spearman_tie_aware": rank_correlation(arrays[threat][use], data[use]),
                        "causal_redundancy_established": False})
    events_by_code = {code: set() for code in range(1, 10)}
    raw_development_cells = set()
    rows_by_season = []
    for season in SEASONS:
        path = split / ("presence_" + season + ".csv")
        record_input(path, inputs)
        frame = pd.read_csv(path)
        dev = frame[frame.split_role == "development"]
        raw_development_cells.update(dev.native_cell_id.astype(str))
        for code in range(1, 10):
            selected = []
            for row in dev.itertuples():
                rr, cc = int(row.raster_row), int(row.raster_col)
                if lulc_valid[rr, cc] and int(lulc[rr, cc]) == code:
                    selected.append(str(row.native_cell_id))
            events_by_code[code].update(selected)
            rows_by_season.append({"season": season, "lucode": code,
                                   "unique_development_presence_cells": len(set(selected))})
    codes = []
    for code, cells in events_by_code.items():
        area = float((common & lulc_valid & (lulc == code)).sum() * area_per_pixel)
        codes.append({"lucode": code, "unique_development_presence_cells_across_seasons": len(cells),
                      "area_km2_in_common": area,
                      "cells_per_1000_km2": 1000 * len(cells) / area if area > 0 else None,
                      "cells_are_statistically_independent_visits": False,
                      "sample_size_guarantees_hj_identifiability": False})
    phase = pilot / "parameter_derivation_audit_001"
    historical = {}
    for filename in ("phase1_data_self_audit.json", "phase2_derived_parameters.json",
                     "phase2c_identifiability_audit.json", "phase3_profile_hj.json"):
        path = phase / filename
        record_input(path, inputs)
        historical[filename] = json.loads(path.read_text(encoding="utf-8"))
    profile = historical["phase3_profile_hj.json"]
    discrepancies = []
    profile_variants = {variant: {"fits": 0, "with_lst_columns": 0, "variables_differ_from_own_frozen_fit": 0}
                        for variant in ("no_lst", "static_lst")}
    run = pilot / "maxent_formal_v1_003"
    for row in profile["per_fit"]:
        path = run / row["variant"] / row["season"] / f"outer_{row['fold']}" / "refit" / row["scheme"] / "train.csv"
        record_input(path, inputs)
        columns = pd.read_csv(path, nrows=0).columns.tolist()[3:]
        stats = profile_variants[row["variant"]]
        stats["fits"] += 1
        stats["with_lst_columns"] += int(any("lst" in v.lower() for v in row["variables"]))
        if set(columns) != set(row["variables"]):
            stats["variables_differ_from_own_frozen_fit"] += 1
            discrepancies.append({"season": row["season"], "variant": row["variant"], "fold": row["fold"],
                                  "background": row["scheme"], "own_fit_columns": columns,
                                  "historical_profile_columns": row["variables"]})
    cross_season = []
    for scheme in ("B0_target_group", "B1_uniform", "B2_visit_density_proxy"):
        def vector(season):
            rows = [r for r in profile["per_fit"] if r["variant"] == "no_lst" and r["scheme"] == scheme and r["season"] == season]
            return np.array([np.mean([r["H_j_profile"][str(c)] for r in rows]) for c in range(1, 10)])
        for season in SEASONS[1:]:
            cross_season.append({"background": scheme, "comparison": season + "_vs_spring",
                                 "spearman_tie_aware": rank_correlation(vector(season), vector("spring"))})
    sdk = Path("D:/SuperMap/SuperMap iDesktopX 2025")
    sdk_records = []
    for name in ("bin/com.supermap.data.jar", "bin/com.supermap.data.conversion.jar", "jre/bin/java.exe"):
        path = sdk / name
        if path.exists():
            record_input(path, inputs)
        sdk_records.append({"component": name, "exists": path.is_file()})
    boundary = pilot / "aoi_boundary_001/zhengzhou_410100_full.geojson"
    record_input(boundary, inputs)
    boundary_summary = pilot / "aoi_boundary_001/aoi_coverage_summary.json"
    record_input(boundary_summary, inputs)
    source = json.loads(boundary_summary.read_text(encoding="utf-8"))["source"]
    boundary_hash_matches = inputs[str(boundary.resolve())]["sha256"] == source["sha256"]
    summary = {"status": "HANDOFF_ACCEPTED_AS_REVIEWED_EXPLORATION_NOT_PARAMETER_APPROVAL",
               "historical_outputs_unchanged": True, "models_fitted": 0, "locked_test_scores_read": False,
               "true_ols_vif": vif, "threat_overlap": overlap, "historical_profile_variant_checks": profile_variants,
               "historical_cross_season_rank_checks": cross_season,
               "historical_cross_scheme_min_rank": profile["verdict"]["min_cross_scheme_rank_corr"],
               "unique_development_presence_cells_before_lulc_filter": len(raw_development_cells),
               "unique_development_presence_cells_mapped_to_lulc": sum(len(cells) for cells in events_by_code.values()),
               "boundary_download_hash_matches": boundary_hash_matches,
               "datum_confirmed_by_iou": False, "sdk_components": sdk_records,
               "survey_77_is_validated_power_requirement": False,
               "class_profile_is_intrinsic_invest_habitat": False,
               "real_supervision_created": 0, "official_invest_eligible": False,
               "adopt": ["Threat/environment overlap diagnostics with tie-aware ranks and true OLS VIF",
                         "Rare-class survey coverage review with season/location/effort and zero-bird visits",
                         "Administrative AOI as research intent, with boundary/datum provenance reviewed separately"],
               "reject_as_unproven": ["Positive profile rank implies intrinsic H_j identifiability",
                                      "77 positive sightings guarantee estimability",
                                      "LULC numeric computability establishes ecological class semantics",
                                      "Only third-party teams can provide independent supervision"]}
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(codes).to_csv(out / "unique_development_class_coverage.csv", index=False)
    pd.DataFrame(rows_by_season).to_csv(out / "season_class_coverage.csv", index=False)
    (out / "profile_feature_discrepancies.json").write_text(json.dumps(discrepancies, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = {}
    for path in out.iterdir():
        record_input(path, outputs)
    import subprocess
    config_snapshot = {"common_domain": "unchanged frozen 100m UTM49 domain",
                       "class_codes": list(range(1, 10)), "correlation": "Spearman average tied ranks",
                       "vif": "joint OLS with intercept", "ecological_parameters_assigned": False}
    (out / "audit_config.snapshot.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")
    record_input(out / "audit_config.snapshot.json", outputs)
    manifest = {"status": summary["status"], "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit_sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                "python": sys.version, "argv": sys.argv, "inputs": inputs, "outputs": outputs, "models_fitted": 0,
                "locked_test_scores_used": False, "official_invest_executed": False}
    record_input(__file__, manifest.setdefault("source_code", {}))
    record_input(root / "wetland_coupling/parameter_diagnostics.py", manifest["source_code"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
