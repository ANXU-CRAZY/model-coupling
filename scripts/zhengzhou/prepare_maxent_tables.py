"""Extract confirmed WGS84 ND observations to native seasonal environmental grids.

Outputs are candidate MaxEnt presence/SWD tables and target-group backgrounds.
No final split, ecological absence label, gate target or fitted model is produced.
Run with GDAL, NumPy and pandas (the installed GeoScene Python can be used).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal, osr

gdal.UseExceptions()
SEASON_DIR = {"spring": "spr_select", "summer": "sum_select", "autumn": "aut_select", "winter": "win_select"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def srs(epsg):
    result = osr.SpatialReference()
    result.ImportFromEPSG(epsg)
    # Set explicit EPSG authority-axis mappings. The installed ESRI GDAL can
    # report a northing/easting mapping for UTM under the generic GIS strategy.
    result.SetDataAxisToSRSAxisMapping([2, 1] if epsg == 4326 else [1, 2])
    return result


def locate(data, reference):
    lon = pd.to_numeric(data.longitude_numeric_unconfirmed_crs, errors="coerce")
    lat = pd.to_numeric(data.latitude_numeric_unconfirmed_crs, errors="coerce")
    usable = lon.between(-180, 180) & lat.between(-90, 90)
    if not usable.all():
        raise ValueError("Candidate coordinates must be finite WGS84 longitude/latitude")
    xy = np.asarray(osr.CoordinateTransformation(srs(4326), srs(32649)).TransformPoints(list(zip(lon, lat))))[:, :2]
    gt = reference.GetGeoTransform()
    rows = np.floor((xy[:, 1] - gt[3]) / gt[5]).astype(int)
    cols = np.floor((xy[:, 0] - gt[0]) / gt[1]).astype(int)
    result = data.copy()
    result["raster_row"] = rows
    result["raster_col"] = cols
    result["inside_native_extent"] = (rows >= 0) & (rows < reference.RasterYSize) & (cols >= 0) & (cols < reference.RasterXSize)
    result["native_cell_id"] = [f"utm49_r{r}_c{c}" for r, c in zip(rows, cols)]
    return result


def sample_environment(records, paths, reference):
    selected = records.loc[records.inside_native_extent].copy()
    selected["environment_valid"] = True
    sources = []
    for path in paths:
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        layer_srs = osr.SpatialReference(wkt=ds.GetProjection())
        layer_srs.AutoIdentifyEPSG()
        if (layer_srs.GetAuthorityCode(None) != "32649" or ds.GetGeoTransform() != reference.GetGeoTransform()
                or (ds.RasterYSize, ds.RasterXSize) != (reference.RasterYSize, reference.RasterXSize)):
            raise ValueError("Environment alignment mismatch: " + str(path))
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()
        mask = band.GetMaskBand().ReadAsArray() != 0
        row, col = selected.raster_row.to_numpy(), selected.raster_col.to_numpy()
        values = array[row, col]
        valid = np.isfinite(values) & mask[row, col]
        nodata = band.GetNoDataValue()
        if nodata is not None:
            valid &= values != nodata
        selected["environment_valid"] &= valid
        selected[path.stem] = np.where(valid, values, np.nan)
        sources.append({"path": str(path), "sha256": digest(path)})
    return selected, sources


def cell_table(records, variable_names, reference):
    eligible = records.loc[records.environment_valid].copy()
    if eligible.empty:
        raise ValueError("No valid occurrence/background cells")
    keys = ["native_cell_id", "raster_row", "raster_col"]
    cell = eligible.groupby(keys, sort=True).agg(
        source_record_count=("record_id", "size"),
        source_species_names=("species_cn", "nunique"),
        first_year=("year_parsed", "min"), last_year=("year_parsed", "max"),
    ).reset_index()
    cell = cell.merge(eligible[keys + variable_names].drop_duplicates(keys), on=keys, validate="one_to_one")
    gt = reference.GetGeoTransform()
    cell["x_utm49"] = gt[0] + (cell.raster_col + 0.5) * gt[1]
    cell["y_utm49"] = gt[3] + (cell.raster_row + 0.5) * gt[5]
    ll = np.asarray(osr.CoordinateTransformation(srs(32649), srs(4326)).TransformPoints(list(zip(cell.x_utm49, cell.y_utm49))))[:, :2]
    cell["longitude"] = ll[:, 0]
    cell["latitude"] = ll[:, 1]
    for size in (5000, 10000, 20000):
        bx, by = np.floor(cell.x_utm49 / size).astype(int), np.floor(cell.y_utm49 / size).astype(int)
        cell[f"diagnostic_block_{size}m"] = [f"b{x}_{y}" for x, y in zip(bx, by)]
    cell["split_role"] = "NOT_ASSIGNED"
    return cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    observation_manifest = json.loads((args.observations / "manifest.json").read_text(encoding="utf-8"))
    if not observation_manifest["coordinate_reference_confirmed"] or observation_manifest["confirmed_source_crs"] != "EPSG:4326":
        raise ValueError("Confirm source CRS before environmental extraction")
    if digest(args.ledger) != observation_manifest["input_ledger_sha256"]:
        raise ValueError("Ledger hash differs from the candidate-observation source")
    occurrence_path = args.observations / "waterbird_occurrences_candidate.csv"
    if digest(occurrence_path) != observation_manifest["outputs"][occurrence_path.name]["sha256"]:
        raise ValueError("Candidate occurrence hash changed")
    core = pd.read_csv(occurrence_path, low_memory=False)
    ledger = pd.read_csv(args.ledger, low_memory=False)
    positive = pd.to_numeric(ledger.abundance_numeric, errors="coerce").gt(0)
    dated = pd.to_datetime(ledger.date_start_parsed, errors="coerce").notna()
    lon = pd.to_numeric(ledger.longitude_numeric_unconfirmed_crs, errors="coerce")
    lat = pd.to_numeric(ledger.latitude_numeric_unconfirmed_crs, errors="coerce")
    allbirds = ledger.loc[positive & dated & lon.between(-180, 180) & lat.between(-90, 90)].copy()
    reference = gdal.Open(str(args.environment_root / "spr_select" / "dem.tif"))
    ref_srs = osr.SpatialReference(wkt=reference.GetProjection())
    ref_srs.AutoIdentifyEPSG()
    gt = reference.GetGeoTransform()
    if ref_srs.GetAuthorityCode(None) != "32649" or (gt[1], gt[2], gt[4], gt[5]) != (100.0, 0.0, 0.0, -100.0):
        raise ValueError("Expected native EPSG:32649 north-up 100 m reference")
    core, allbirds = locate(core, reference), locate(allbirds, reference)
    args.out.mkdir(parents=True)
    seasons = {}
    for season, directory in SEASON_DIR.items():
        paths = sorted((args.environment_root / directory).glob("*.tif"))
        if not paths:
            raise ValueError("No environmental variables for " + season)
        variables = [path.stem for path in paths]
        records = core.loc[core.season_calendar == season]
        sampled, sources = sample_environment(records, paths, reference)
        background_sampled, _ = sample_environment(allbirds.loc[allbirds.season_calendar == season], paths, reference)
        presence = cell_table(sampled, variables, reference)
        background = cell_table(background_sampled, variables, reference)
        presence.to_csv(args.out / ("occurrence_cells_" + season + ".csv"), index=False, encoding="utf-8-sig")
        background.to_csv(args.out / ("target_group_background_cells_" + season + ".csv"), index=False, encoding="utf-8-sig")
        for species, table, kind in (("waterbird_community_candidate", presence, "presence"), ("background", background, "background")):
            swd = table[["longitude", "latitude"] + variables].copy()
            swd.insert(0, "species", species)
            swd.to_csv(args.out / ("maxent_swd_" + kind + "_" + season + ".csv"), index=False)
        membership = records[["record_id", "native_cell_id", "inside_native_extent"]].copy()
        membership = membership.merge(sampled[["record_id", "environment_valid"]], on="record_id", how="left", validate="one_to_one")
        membership.to_csv(args.out / ("source_membership_" + season + ".csv"), index=False, encoding="utf-8-sig")
        seasons[season] = {
            "candidate_records": len(records), "outside_native_extent": int((~records.inside_native_extent).sum()),
            "records_with_complete_environment": int(sampled.environment_valid.sum()),
            "unique_presence_cells": len(presence), "target_group_candidate_cells": len(background),
            "diagnostic_block_counts": {str(size): int(presence[f"diagnostic_block_{size}m"].nunique()) for size in (5000, 10000, 20000)},
            "environment_sources": sources,
        }
    manifest = {
        "status": "MAXENT_INPUT_CANDIDATES_ENVIRONMENT_EXTRACTED_NOT_FITTED",
        "source_observation_manifest_sha256": digest(args.observations / "manifest.json"),
        "source_crs": "EPSG:4326", "source_crs_evidence": observation_manifest["source_crs_evidence"],
        "grid_crs": "EPSG:32649", "pixel_size_m": 100, "seasons": seasons,
        "background_is_absence": False, "background_strategy_finalized": False,
        "final_split_assigned": False, "gate_targets_created": False,
        "environment_input_years_confirmed": False,
        "records_pooled_across_years": True,
        "cell_center_coordinates_are_analysis_units": True,
        "caveats": [
            "Family/species/synonym and survey QA are pending; source records are retained separately",
            "Grid-cell pooling prevents repeated sightings from acting as independent presence cells; counts are not effort-adjusted targets",
            "Candidate background comprises all positive ND bird records and may include waterbird cells; it is not a set of confirmed absences",
            "Environmental layers are historical literature-selected predictors; their sampling dates and selection provenance must be established",
            "The native valid raster extent is a candidate sampling domain, not an accepted Zhengzhou/management boundary",
            "5/10/20 km blocks are diagnostics only; do not describe them as validated spatial independence",
        ],
        "outputs": {},
    }
    for path in args.out.glob("*.csv"):
        manifest["outputs"][path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "seasons": {k: {f: v[f] for f in ("unique_presence_cells", "target_group_candidate_cells", "records_with_complete_environment")} for k, v in seasons.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
