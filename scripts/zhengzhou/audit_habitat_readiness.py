"""Audit HQ data readiness without assigning ecological parameters or running HQ.

Coverage masks describe available raster support; they are neither management
AOIs nor restoration feasibility A. No bird response or model scores are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

from scripts.zhengzhou.export_maxent_oof import (
    read_json, sha256, verify_split_inventory, write_json,
)
from wetland_coupling.parameters import validate_prior


THREATS = ("urban_structure", "human_activity", "night_light")
SOURCE = "https://raw.githubusercontent.com/natcap/invest/3.16.1/src/natcap/invest/habitat_quality.py"


def check_values(values, valid, kind):
    values, valid = np.asarray(values), np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or not valid.any():
        raise ValueError("Empty or inconsistent raster mask")
    observed = values[valid]
    if not np.isfinite(observed).all():
        raise ValueError("Nonfinite raster value inside valid mask")
    if kind == "lulc":
        if not np.equal(observed, np.rint(observed)).all():
            raise ValueError("LULC must contain integer category values")
    elif kind == "threat":
        if np.any((observed < 0) | (observed > 1)):
            raise ValueError("Threat intensity outside [0,1]")
    else:
        raise ValueError("Unknown raster type")
    return {"valid_pixels": int(valid.sum()), "min": float(observed.min()),
            "max": float(observed.max()), "zero_pixels": int(np.sum(observed == 0))}


def conservative_support_distance(valid, resolution_m):
    """Lower bound from complete target cell to unknown raster cells/outside.

    EDT measures centre distances. Subtract two square-cell half diagonals,
    then clamp at zero. Padding makes the outside unknown even for a completely
    filled raster. This is a conservative geometric diagnostic, not ecological
    evidence that the source raster really observed the whole neighbourhood.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("Need a 2D mask and positive metric square-cell resolution")
    centre = distance_transform_edt(np.pad(valid, 1, constant_values=False),
                                   sampling=resolution_m)[1:-1, 1:-1]
    return np.maximum(0., centre - np.sqrt(2.) * resolution_m)


def coverage_row(name, support, common, development):
    return {"support": name, "available_in_common": int((support & common).sum()),
            "missing_in_common": int((~support & common).sum()),
            "available_in_development_oof": int((support & development).sum()),
            "missing_in_development_oof": int((~support & development).sum())}


def crosswalk_missing(codes, registry):
    rows = registry.get("class_crosswalk", [])
    lookup = {}
    for row in rows:
        code = int(row["lucode"])
        if code in lookup:
            raise ValueError("Duplicate LULC code in evidence registry")
        lookup[code] = row
    # A guessed class name alone cannot pass the evidence requirement.
    return [int(code) for code in codes if code not in lookup or
            lookup[code].get("status") != "SOURCE_CROSSWALK_CONFIRMED" or
            not lookup[code].get("name") or not lookup[code].get("evidence")]


def audit(args):
    import rasterio

    start = datetime.now(timezone.utc).isoformat()
    root = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    split = args.splits.resolve()
    verify_split_inventory(split)
    plan = read_json(split / "split_plan.json")
    with np.load(split / "split_masks.npz", allow_pickle=False) as z:
        common = z["common_valid"].astype(bool)
        development = np.logical_or.reduce([z[f"outer_{f['fold']}_validation"].astype(bool)
                                            for f in plan["outer_folds"]])
        locked = z["locked_test"].astype(bool)
        if np.any(development & locked) or np.any(development & ~common):
            raise ValueError("Frozen development domain is invalid")
    profile = read_json(args.profiles / "manifest.json")
    registry = read_json(args.registry)
    prior = read_json(args.prior)
    pilot = read_json(args.pilot_config)
    supervision = read_json(args.supervision / "manifest.json")
    inputs = [args.registry, args.prior, args.pilot_config, args.profiles / "manifest.json",
              args.supervision / "manifest.json", split / "manifest.json",
              split / "split_plan.json", split / "split_masks.npz"]
    arrays, masks, metadata = {}, {}, {}
    resolution = float(plan["grid"]["resolution_m"])
    if plan["grid"]["crs"] != "EPSG:32649":
        raise ValueError("Expected the frozen UTM49 metric grid")
    from affine import Affine
    expected_transform = Affine.from_gdal(*plan["grid"]["transform_gdal"])
    if expected_transform.b != 0 or expected_transform.d != 0 or expected_transform.a != resolution or expected_transform.e != -resolution:
        raise ValueError("Only the frozen unrotated square metric grid is supported")
    for name in ("lulc", *THREATS):
        record = profile["sources"][name]
        # The profile is historical evidence; always verify its recorded bytes.
        path = Path(record["path"])
        if not path.is_absolute():
            path = root / path
        if sha256(path) != record["sha256"]:
            raise ValueError("Source raster changed since profiling: " + name)
        with rasterio.open(path) as ds:
            if ds.count != 1 or ds.shape != common.shape or ds.crs is None or ds.crs.to_epsg() != 32649 or ds.transform != expected_transform:
                raise ValueError("Grid, band or CRS mismatch: " + name)
            array = ds.read(1)
            valid = ds.read_masks(1) != 0
            # Nonfinite pixels are unknown support, never zero threat.
            nonfinite = int((valid & ~np.isfinite(array)).sum())
            valid &= np.isfinite(array)
            metadata[name] = {"path": str(path.resolve()), "sha256": sha256(path),
                              "nodata": ds.nodata, "nonfinite_unmasked_pixels": nonfinite,
                              **check_values(array, valid, "lulc" if name == "lulc" else "threat")}
        inputs.append(path)
        arrays[name], masks[name] = array, valid
    joint = np.logical_and.reduce(list(masks.values()))
    coverage = [coverage_row(name, mask, common, development) for name, mask in masks.items()]
    coverage.append(coverage_row("lulc_and_all_threats", joint, common, development))
    codes = np.unique(arrays["lulc"][masks["lulc"]]).astype(int)
    class_rows = [{"lucode": int(code),
                   "available_raster_pixels": int((masks["lulc"] & (arrays["lulc"] == code)).sum()),
                   "common_domain_pixels": int((joint & common & (arrays["lulc"] == code)).sum()),
                   "class_semantics_verified": int(code) not in crosswalk_missing(codes, registry)}
                  for code in codes]
    distances = {name: conservative_support_distance(masks[name], resolution) for name in THREATS}
    design = registry.get("sensitivity_design", {})
    maximum = float(design.get("maximum_threat_distance_m", 5000))
    historical = {row["threat"]: float(row["max_dist"]) for row in registry["historical_threats"]}
    if set(historical) != set(THREATS) or registry.get("distance_unit") != "m":
        raise ValueError("Historical threat schema/name/unit mismatch")
    radii = sorted({500., 1000., 1500., 2500., maximum, *historical.values()})
    if not np.isfinite(radii).all() or min(radii) < resolution:
        raise ValueError("Threat distances must be finite and at least one LULC pixel")
    buffers = []
    for radius in radii:
        for name in THREATS:
            row = coverage_row(name, joint & (distances[name] >= radius), common, development)
            buffers.append({"radius_m": radius, "diagnostic": "conservative_complete_cell_support", **row})
        supported = joint & np.logical_and.reduce([distances[name] >= radius for name in THREATS])
        buffers.append({"radius_m": radius, "diagnostic": "conservative_complete_cell_support",
                        **coverage_row("all_threats", supported, common, development)})
    historical_support = joint & np.logical_and.reduce([distances[name] >= historical[name] for name in THREATS])
    maximum_support = joint & np.logical_and.reduce([distances[name] >= maximum for name in THREATS])
    try:
        validate_prior(prior)
        prior_problem = None
    except (ValueError, KeyError) as exc:
        prior_problem = str(exc)
    missing_crosswalk = crosswalk_missing(codes, registry)
    blockers = []
    if missing_crosswalk:
        blockers.append({"code": "LULC_SEMANTICS_UNVERIFIED", "detail": "Source crosswalk missing", "lucodes": missing_crosswalk})
    if registry.get("habitat_prior_strategy", {}).get("local_habitat_and_sensitivity_values") is None:
        blockers.append({"code": "LOCAL_HABITAT_SENSITIVITY_MISSING"})
    if prior_problem:
        blockers.append({"code": "PRIOR_NOT_EXECUTABLE", "detail": prior_problem})
    if not pilot["grid"].get("municipal_or_management_aoi_accepted", False):
        blockers.append({"code": "MANAGEMENT_AOI_UNCONFIRMED"})
    if not registry.get("threat_scale_comparability_confirmed", False):
        blockers.append({"code": "THREAT_PRODUCT_TIME_AND_SCALE_REVIEW_PENDING"})
    if np.any(common & ~joint):
        blockers.append({"code": "HQ_INPUTS_MISSING_ON_MAXENT_DOMAIN", "pixels": int((common & ~joint).sum())})
    blockers.append({"code": "SOURCE_BUFFER_PROVENANCE_UNVERIFIED",
                     "detail": "Raster interior geometry cannot establish observed source buffer coverage"})
    labels = int(supervision["labels_created"])
    # Never infer independent calibration from event counts or pending config flags.
    if pilot["invest"].get("independent_calibration_targets") is None:
        blockers.append({"code": "INDEPENDENT_HQ_CALIBRATION_TARGETS_MISSING"})
    if labels == 0:
        blockers.append({"code": "REAL_GATE_SUPERVISION_MISSING", "labels": 0})
    try:
        invest_version = importlib.metadata.version("natcap.invest")
    except importlib.metadata.PackageNotFoundError:
        invest_version = None
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        commit = "UNAVAILABLE"
    out.mkdir(parents=True, exist_ok=False)
    snapshots = out / "config_snapshots"
    snapshots.mkdir()
    snapshot_names = {}
    for name, path in (("registry.json", args.registry), ("prior.json", args.prior),
                       ("pilot_config.json", args.pilot_config)):
        target = snapshots / name
        target.write_bytes(path.read_bytes())
        snapshot_names[str(path.resolve())] = str(target.relative_to(out))
    import pandas as pd
    pd.DataFrame(class_rows).to_csv(out / "lulc_observed_codes.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "domain_coverage.csv", index=False)
    pd.DataFrame(buffers).to_csv(out / "threat_buffer_support.csv", index=False)
    np.savez_compressed(out / "coverage_masks.npz", lulc_and_threats_available=joint,
                        historical_radius_geometric_support=historical_support,
                        maximum_design_radius_geometric_support=maximum_support)
    summary = {"status": "INPUT_READINESS_AUDITED_FORMAL_HQ_CALIBRATION_BLOCKED" if blockers else "INPUT_AUDIT_COMPLETE_REVIEW_REQUIRED",
               "common_maxent_pixels": int(common.sum()), "development_oof_pixels": int(development.sum()),
               "joint_available": coverage[-1], "historical_radius_support": coverage_row("historical_radii", historical_support, common, development),
               "maximum_radius_support": coverage_row("maximum_design_radius", maximum_support, common, development),
               "maximum_design_radius_m": maximum, "observed_lulc_codes": codes.tolist(),
               "candidate_events": int(supervision["total_candidate_events"]), "real_labels": labels,
               "installed_invest_in_this_interpreter": invest_version, "target_invest_version": "3.16.1",
               "blockers": blockers, "official_invest_executed": False, "parameters_assigned": False,
               "gate_eligible": False, "source_data_modified": False, "formal_run_authorized_by_this_audit": False,
               "aoi_or_feasibility_inferred_from_coverage": False,
               "support_distance_method": "EDT on each valid threat mask with outside-unknown padding; subtract full cell diagonal for complete target/source pixel footprints",
               "buffer_support_is_source_provenance_confirmation": False,
               "next_actions": ["Recover source-verified LULC code legend and product years", "Review guild-specific habitat/sensitivity evidence without M/Q maps", "Resolve source-buffer/missing-pixel policy before clipping", "Obtain genuine independent calibration and management assessment targets"]}
    write_json(out / "summary.json", summary)
    manifest = {"status": summary["status"], "started_at_utc": start,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(), "command_line": sys.argv,
                "git_commit_sha": commit, "python": sys.version, "platform": platform.platform(),
                "source_code_sha256": sha256(__file__), "random_seed": None,
                "helper_source_sha256": {str(path.relative_to(root)): sha256(path) for path in
                                         (root / "wetland_coupling/parameters.py",
                                          root / "scripts/zhengzhou/export_maxent_oof.py")},
                "config_snapshots_saved": True,
                "versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "rasterio", "pandas")},
                "deterministic_geometric_audit": True, "split_manifest_sha256": sha256(split / "manifest.json"),
                "official_interface_source": SOURCE, "official_interface_checked_date": "2026-10-02",
                "input_rasters": metadata, "locked_test_predictions_or_metrics_read": False,
                "models_fitted": 0, "labels_created": 0, "official_invest_executed": False,
                "inputs": {str(p.resolve()): {"sha256": sha256(p), "bytes": p.stat().st_size,
                           **({"snapshot_relative_path": snapshot_names[str(p.resolve())]}
                              if str(p.resolve()) in snapshot_names else {})} for p in inputs},
                "outputs": {str(p.relative_to(out)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                            for p in out.rglob("*") if p.is_file()}}
    write_json(out / "manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/zhengzhou_habitat_evidence.draft.json"))
    parser.add_argument("--prior", type=Path, default=Path("configs/invest_prior.pending.json"))
    parser.add_argument("--pilot-config", type=Path, default=Path("configs/zhengzhou_pilot.pending.json"))
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
