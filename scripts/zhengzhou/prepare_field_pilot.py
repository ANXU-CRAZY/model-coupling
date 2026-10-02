"""Recover source evidence and prepare a local field-verification pilot.

No bird predictions, locked test scores, ecological parameter values or fitted
models are used. Candidate pixel centers are not approved observation stations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform, transform_geom

from scripts.zhengzhou.export_maxent_oof import verify_split_inventory
from wetland_coupling.parameter_diagnostics import record_input


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def verified_csv(directory, name, inventory):
    manifest_path = directory / "manifest.json"
    record_input(manifest_path, inventory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / name
    record_input(path, inventory)
    entries = {str(k).replace("\\", "/"): v for k, v in manifest["outputs"].items()}
    if inventory[str(path.resolve())]["sha256"] != entries[name]["sha256"]:
        raise ValueError("Source hash differs: " + name)
    return pd.read_csv(path, low_memory=False)


def recover_mlx(path, out, inventory):
    """Read saved table evidence; never execute attached MATLAB instructions."""
    record_input(path, inventory)
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("matlab/document.xml"))
        output = ET.fromstring(archive.read("matlab/output.xml"))
    code = "\n".join(node.text for node in document.iter() if node.text and not len(node))
    (out / "historical_lulc_code.txt").write_text(code, encoding="utf-8")
    tables = []
    for node in output.iter():
        if not node.text or not node.text.startswith('[["\'water\'"'):
            continue
        table = json.loads(node.text)
        if len(table) == 9 and all(len(row) == 6 for row in table):
            tables.append(table)
    if len(tables) != 1:
        raise ValueError("Saved MATLAB LULC table was not uniquely recovered")
    names = ["water", "trees", "grass", "aquatic", "crops", "shrub", "built", "bare", "snow"]
    rows = []
    for index, row in enumerate(tables[0]):
        if row[0].strip("'") != names[index] or int(row[1]) != index:
            raise ValueError("Unexpected historical class table")
        rows.append({"historical_code": index, "historical_name": names[index],
                     "count_2018": int(row[2]), "percent_2018": float(row[3]),
                     "count_2020": int(row[4]), "percent_2020": float(row[5]),
                     "applies_to_current_lulc_cur": False})
    pd.DataFrame(rows).to_csv(out / "historical_code_table.csv", index=False, encoding="utf-8-sig")
    result = {"status": "HISTORICAL_CODE_TABLE_RECOVERED_CURRENT_CROSSWALK_UNPROVEN",
              "source": str(path.resolve()), "codes": list(range(9)), "names": names,
              "source_inputs_named": ["2018_label.tif", "2020_label.tif"],
              "saved_output_not_new_execution": True,
              "matches_dynamic_world_class_order": True,
              "dynamic_world_product_provenance_confirmed": False,
              "current_reclassification_or_source_pixel_match_available": False,
              "current_crosswalk_confirmed": False,
              "historical_code_8_is": "snow, not large buildings",
              "counts_2018_sum": sum(row["count_2018"] for row in rows),
              "counts_2020_sum": sum(row["count_2020"] for row in rows)}
    write_json(out / "class_source_findings.json", result)
    return result


def audit_events(events, lineage, out):
    if not events.event_candidate_id.is_unique or not lineage.event_candidate_id.is_unique:
        raise ValueError("Duplicate inferred event IDs")
    if set(events.event_candidate_id) != set(lineage.event_candidate_id):
        raise ValueError("Event and lineage universes differ")
    if events[["target_protection", "target_restoration"]].notna().any().any():
        raise ValueError("Unexpected existing targets")
    if not events.eligible_for_supervised_training.astype(str).str.lower().eq("false").all():
        raise ValueError("Unexpected eligible supervision")
    rows = []
    for source, frame in events.groupby("source_kind", sort=True):
        duration = pd.to_numeric(frame.reported_duration_minutes, errors="coerce")
        positive = pd.to_numeric(frame.observed_waterbird_species_names_n, errors="raise").gt(0)
        rows.append({"source_kind": source, "candidate_events": len(frame),
                     "with_recorded_waterbirds": int(positive.sum()),
                     "no_recorded_waterbirds_not_confirmed_nondetection": int((~positive).sum()),
                     "reported_time_span_positive": int(duration.gt(0).sum()),
                     "reported_time_span_zero": int(duration.eq(0).sum()),
                     "reported_time_span_negative": int(duration.lt(0).sum()),
                     "reported_time_span_missing": int(duration.isna().sum()),
                     "reported_time_span_over_24h": int(duration.gt(1440).sum()),
                     "verified_effort_events": int(frame.effort_minutes_verified.notna().sum()),
                     "observer_count_available": int(frame.observer_count.notna().sum()),
                     "distance_available": int(frame.distance_km.notna().sum()),
                     "confirmed_complete_target_checklists": int(frame.complete_target_checklist.astype(str).str.lower().eq("true").sum()),
                     "confirmed_source_events": int(frame.event_id_status.eq("confirmed").sum()),
                     "sites_inferred": int(frame.site_candidate_id.nunique()),
                     "events_with_possible_duplicate_rows": int(frame.possible_duplicate_rows.gt(0).sum()),
                     "coordinates_confirmed_WGS84": int(frame.source_crs.eq("EPSG:4326").sum())})
    pd.DataFrame(rows).to_csv(out / "source_response_readiness.csv", index=False, encoding="utf-8-sig")
    # Select only documented development positive memberships for detailed summaries.
    # Historical locked/buffer event responses are not used to estimate an endpoint.
    development_ids = set(lineage.loc[lineage.presence_role_unique.eq("development"), "event_candidate_id"])
    dev = events.loc[events.event_candidate_id.isin(development_ids)].copy()
    dev["date"] = pd.to_datetime(dev.start_time, errors="coerce").dt.strftime("%Y-%m-%d")
    summaries = dev.groupby("season").agg(
        positive_development_candidate_events=("event_candidate_id", "size"),
        inferred_sites=("site_candidate_id", "nunique"),
        years=("calendar_year", "nunique"),
        source_rows=("source_record_count", "sum"),
        reported_richness_median=("observed_waterbird_species_names_n", "median"),
    ).reset_index()
    summaries.to_csv(out / "development_positive_event_summary.csv", index=False, encoding="utf-8-sig")
    repeat_rows = []
    for (site, year, season), group in dev.groupby(["site_candidate_id", "season_year", "season"]):
        repeat_rows.append({"site_id_local": site, "season_year": int(year), "season": season,
                            "events_inferred": len(group), "distinct_dates": int(group.date.nunique())})
    repeats = pd.DataFrame(repeat_rows)
    # No sites/coordinates are placed in the public summary.
    repeats.groupby("season").agg(site_season_year_groups=("site_id_local", "size"),
                                  with_two_distinct_dates=("distinct_dates", lambda x: int(x.ge(2).sum())),
                                  with_three_distinct_dates=("distinct_dates", lambda x: int(x.ge(3).sum()))
                                  ).reset_index().to_csv(out / "repeat_visit_opportunity.csv", index=False, encoding="utf-8-sig")
    return {"source_readiness": rows, "development_positive_events": len(dev),
            "confirmed_independent_targets": 0,
            "repeat_groups_are_occupancy_replicates": False,
            "existing_records_support": "Descriptive recorded-use summaries and existing presence/background models",
            "not_admitted": ["Historical missing species -> nondetection", "Reported span -> verified effort",
                             "Counts per minute without protocol", "Historical records -> independent HQ labels"]}


def read_raster(path, inventory, grid=None):
    record_input(path, inventory)
    with rasterio.open(path) as ds:
        actual = (ds.shape, ds.transform, ds.crs)
        if grid is not None and actual != grid:
            raise ValueError("Grid alignment differs: " + str(path))
        return ds.read(1), (ds.read_masks(1) != 0), actual


def sample_frame(root, config, out, inventory):
    pilot = root / "local_work/zhengzhou_pilot"
    splits = pilot / "maxent_formal_v1_001/splits"
    verify_split_inventory(splits)
    for name in ("manifest.json", "split_plan.json", "split_masks.npz"):
        record_input(splits / name, inventory)
    with np.load(splits / "split_masks.npz", allow_pickle=False) as data:
        common = data["common_valid"].astype(bool)
    base = pilot / "archival_baselines_003"
    lulc, valid_lulc, grid = read_raster(base / "lulc_cur_utm49_100m.tif", inventory)
    pressure, valid_pressure, _ = read_raster(base / "human_activity_utm49_100m.tif", inventory, grid)
    role, role_valid, _ = read_raster(splits / "role_raster.tif", inventory, grid)
    boundary_path = pilot / "aoi_boundary_001/zhengzhou_410100_full_wgs84.geojson"
    record_input(boundary_path, inventory)
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    geometries = [transform_geom("EPSG:4326", grid[2], item["geometry"]) for item in boundary["features"]]
    city = rasterize([(geometry, 1) for geometry in geometries], out_shape=grid[0], transform=grid[1], all_touched=False).astype(bool)
    if common.shape != lulc.shape or grid[2].to_epsg() != 32649:
        raise ValueError("Unexpected pilot grid")
    domain = city & common & valid_lulc & np.isin(lulc, config["classes"]) & valid_pressure & np.isfinite(pressure)
    rng = np.random.default_rng(config["seed"])
    rows, census = [], []
    width = lulc.shape[1]
    per_stratum = config["primary_blocks_per_class_pressure_stratum"]
    reserve = config["reserve_blocks_per_class_pressure_stratum"]
    for code in config["classes"]:
        indices = np.flatnonzero(domain & (lulc == code))
        if not len(indices):
            raise ValueError("No candidate pixels for code " + str(code))
        median = float(np.median(pressure.ravel()[indices]))
        for stratum in ("low", "high"):
            keep = pressure.ravel()[indices] <= median if stratum == "low" else pressure.ravel()[indices] > median
            cells = indices[keep]
            rr, cc = np.divmod(cells, width)
            xs, ys = rasterio.transform.xy(grid[1], rr, cc)
            xs, ys = np.asarray(xs), np.asarray(ys)
            bx = np.floor(xs / config["block_size_m"]).astype(int)
            by = np.floor(ys / config["block_size_m"]).astype(int)
            block_ids = np.array([f"{x}_{y}" for x, y in zip(bx, by)])
            blocks = np.unique(block_ids)
            if len(blocks) < per_stratum + reserve:
                raise ValueError("Insufficient distinct blocks for planned stratum")
            selected = rng.permutation(blocks)[:per_stratum + reserve]
            census.append({"lucode": code, "pressure_stratum": stratum, "median_human_activity": median,
                           "candidate_pixels": len(cells), "candidate_blocks": len(blocks),
                           "area_km2": float(len(cells) * abs(grid[1].a * grid[1].e) / 1.e6)})
            for priority, block in enumerate(selected, 1):
                available = np.flatnonzero(block_ids == block)
                chosen = int(rng.choice(available))
                flat = int(cells[chosen]); row, col = divmod(flat, width)
                x, y = float(xs[chosen]), float(ys[chosen])
                longitude, latitude = transform(grid[2], "EPSG:4326", [x], [y])
                primary = priority <= per_stratum
                rows.append({"candidate_id": f"C{code}_{stratum}_{priority:02d}", "lucode": code,
                             "class_name": "UNVERIFIED", "pressure_stratum": stratum,
                             "selection_status": "primary" if primary else "reserve",
                             "random_priority": priority, "block_1km_id": block,
                             "native_cell_id": flat, "row": row, "col": col,
                             "easting_32649": x, "northing_32649": y,
                             "longitude_WGS84": longitude[0], "latitude_WGS84": latitude[0],
                             "human_activity": float(pressure[row, col]),
                             "frozen_role_code_context_only": int(role[row, col]) if role_valid[row, col] else None,
                             "initial_primary_pixel_inclusion_probability": per_stratum / len(blocks) / len(available),
                             "initial_primary_probability_applies_to": "Initial design only, not activated reserves or replacements",
                             "accessibility": "UNVERIFIED", "approved_bird_survey_station": False})
    frame = pd.DataFrame(rows)
    if not frame.native_cell_id.is_unique or len(frame) != len(config["classes"]) * 2 * (per_stratum + reserve):
        raise ValueError("Unexpected candidate design count or duplicate cell")
    frame.to_csv(out / "landcover_verification_candidates_LOCAL_ONLY.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(census).to_csv(out / "sampling_stratum_census.csv", index=False, encoding="utf-8-sig")
    features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [row["longitude_WGS84"], row["latitude_WGS84"]]},
                 "properties": {key: row[key] for key in ("candidate_id", "lucode", "selection_status", "pressure_stratum", "accessibility")}}
                for row in rows]
    write_json(out / "landcover_verification_candidates_LOCAL_ONLY.geojson", {"type": "FeatureCollection", "features": features})
    return {"city_common_pixels": int((city & common).sum()), "sampling_frame_pixels": int(domain.sum()),
            "city_common_excluded_for_lulc_or_pressure": int((city & common & ~domain).sum()),
            "candidate_primary": int(frame.selection_status.eq("primary").sum()),
            "candidate_reserve": int(frame.selection_status.eq("reserve").sum()),
            "unique_1km_blocks_all_candidates": int(frame.block_1km_id.nunique()),
            "unique_1km_blocks_primary": int(frame.loc[frame.selection_status.eq("primary"), "block_1km_id"].nunique()),
            "candidate_boundary_is_authoritative": False,
            "power_requirement": False, "model_scores_used_for_selection": False,
            "probabilities_apply_after_accessibility_replacement": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--historical-mlx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.relative_to(root / "local_work")
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    inventory = {}
    record_input(args.config, inventory)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    write_json(out / "config.snapshot.json", config)
    source = recover_mlx(args.historical_mlx, out, inventory)
    pilot = root / "local_work/zhengzhou_pilot"
    events = verified_csv(pilot / "supervision_candidates_003", "survey_events_candidate.csv", inventory)
    lineage = verified_csv(pilot / "supervision_lineage_audit_002", "event_review_queue.csv", inventory)
    responses = audit_events(events, lineage, out)
    frame = sample_frame(root, config, out, inventory)
    sensitivity = [{"assumed_detection_probability_given_use": p, "visits": k,
                    "probability_at_least_one_detection_given_use": 1 - (1 - p) ** k,
                    "basis": "Hypothetical independent constant-p repeat visits, not observed detection rate or statistical power"}
                   for p in config["detection_sensitivity_grid"] for k in config["repeat_sensitivity_grid"]]
    pd.DataFrame(sensitivity).to_csv(out / "detection_repeat_sensitivity.csv", index=False, encoding="utf-8-sig")
    summary = {"status": "FIELD_VERIFICATION_AND_RESPONSE_PREPARATION_NO_NEW_LABELS",
               "class_source": source, "response_readiness": responses, "pilot_sampling": frame,
               "models_fitted": 0, "independent_supervision_created": 0,
               "invest_executed": False, "locked_test_metrics_read": False,
               "eligible_for_official_run": False, "gate_eligible": False}
    write_json(out / "summary.json", summary)
    sources, outputs = {}, {}
    record_input(__file__, sources)
    record_input(root / "wetland_coupling/parameter_diagnostics.py", sources)
    record_input(root / "scripts/zhengzhou/export_maxent_oof.py", sources)
    for path in sorted(out.iterdir()):
        if path.is_file():
            record_input(path, outputs)
    manifest = {"status": summary["status"], "started_at_utc": started,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
                "git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                "python": sys.version, "seed": config["seed"], "inputs": inventory,
                "source_code": sources, "outputs": outputs, "models_fitted": 0,
                "independent_supervision_created": 0, "locked_test_metrics_read": False}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": summary["status"], "pilot_sampling": frame,
                      "development_positive_events": responses["development_positive_events"]}))


if __name__ == "__main__":
    main()
