"""Prepare auditable archival grids and exploratory conservation baselines.

Run with a Python environment containing GDAL and NumPy. No gate training or
restoration score is generated: the archival models lack fold provenance and
independent supervision, H_j and management feasibility remain unresolved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
NODATA = -9999.0
SEASONS = ("spring", "summer", "autumn", "winter")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def values(dataset):
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    valid = np.isfinite(array) & (band.GetMaskBand().ReadAsArray() != 0)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        valid &= array != nodata
    return array, valid


def metadata(path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    reference = osr.SpatialReference(wkt=ds.GetProjection())
    reference.AutoIdentifyEPSG()
    array, valid = values(ds)
    data = array[valid]
    return {
        "path": str(path), "sha256": digest(path),
        "width": ds.RasterXSize, "height": ds.RasterYSize,
        "epsg": reference.GetAuthorityCode(None), "transform": list(ds.GetGeoTransform()),
        "nodata": ds.GetRasterBand(1).GetNoDataValue(), "valid_pixels": int(valid.sum()),
        "min": float(data.min()) if data.size else None,
        "max": float(data.max()) if data.size else None,
    }


def write_array(path, array, valid, reference):
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), reference.RasterXSize, reference.RasterYSize, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    ds.SetProjection(reference.GetProjection())
    ds.SetGeoTransform(reference.GetGeoTransform())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(NODATA)
    band.WriteArray(np.where(valid, array, NODATA).astype(np.float32))
    ds.SetMetadataItem("RESULT_STATUS", "EXPLORATORY_ARCHIVAL_BASELINE_NOT_VALIDATED")
    ds.FlushCache()
    ds = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--maxent-root", type=Path, required=True)
    parser.add_argument("--invest-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assign-unreferenced-asc-from-grid", action="store_true",
                        help="Explicitly permit inferred CRS assignment to ASC with no CRS, after exact native-grid matching")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    reference_path = args.environment_root / "spr_select" / "dem.tif"
    reference = gdal.Open(str(reference_path), gdal.GA_ReadOnly)
    reference_info = metadata(reference_path)
    if reference_info["epsg"] != "32649":
        raise ValueError("Reference CRS must be the documented WGS84/UTM49N grid")
    transform = reference.GetGeoTransform()
    if transform[1] != 100 or transform[5] != -100 or transform[2] or transform[4]:
        raise ValueError("Expected the original north-up 100 m grid")
    environment = []
    for path in sorted(args.environment_root.glob("*/*.tif")):
        item = metadata(path)
        if any(item[key] != reference_info[key] for key in ("width", "height", "epsg", "transform")):
            raise ValueError("Environment alignment mismatch: " + str(path))
        item["source_observation_year"] = "NOT_ESTABLISHED_FROM_FILE_METADATA"
        environment.append(item)
    if not environment:
        raise ValueError("No exported environmental rasters")
    asc_crs_audit = {}
    reference_srs = osr.SpatialReference(wkt=reference.GetProjection())
    for season in SEASONS:
        path = args.maxent_root / ("Waterbirds_" + season + "_avg.asc")
        source = gdal.Open(str(path), gdal.GA_ReadOnly)
        if (source.RasterXSize, source.RasterYSize, source.GetGeoTransform()) != (
            reference.RasterXSize, reference.RasterYSize, reference.GetGeoTransform()
        ):
            raise ValueError("ASC does not match documented source grid: " + str(path))
        projection = source.GetProjection()
        if projection:
            source_srs = osr.SpatialReference(wkt=projection)
            if not source_srs.IsSame(reference_srs):
                raise ValueError("Existing ASC CRS conflicts with the native reference: " + str(path))
            status = "SOURCE_CRS_MATCHES_REFERENCE"
        else:
            if not args.assign_unreferenced_asc_from_grid:
                raise ValueError("ASC has no CRS; explicit assignment flag required: " + str(path))
            status = "INFERRED_FROM_EXACT_NATIVE_GRID_PENDING_PRODUCER_CONFIRMATION"
        asc_crs_audit[season] = {
            "source_path": str(path), "source_sha256": digest(path),
            "original_projection_wkt": projection, "status": status,
            "reference_path": str(reference_path), "reference_sha256": reference_info["sha256"],
            "reference_projection_wkt": reference.GetProjection(),
            "dimensions_and_geotransform_match": True,
        }
    args.out.mkdir(parents=True)
    left, top = transform[0], transform[3]
    right = left + reference.RasterXSize * transform[1]
    bottom = top + reference.RasterYSize * transform[5]
    aligned_inputs = []
    for name in ("quality_c_ref", "lulc_cur", "human_activity", "night_light", "urban_structure"):
        source = args.invest_root / (name + ".tif")
        target = args.out / (name + "_utm49_100m.tif")
        result = gdal.Warp(
            str(target), str(source), dstSRS=reference.GetProjection(),
            outputBounds=(left, bottom, right, top), width=reference.RasterXSize,
            height=reference.RasterYSize, resampleAlg="near" if name == "lulc_cur" else "bilinear",
            dstNodata=255 if name == "lulc_cur" else NODATA,
            outputType=gdal.GDT_Byte if name == "lulc_cur" else gdal.GDT_Float32,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
        )
        if result is None:
            raise RuntimeError("GDAL Warp failed: " + str(source))
        result = None
        aligned_inputs.append({"source": metadata(source), "output": metadata(target)})
    hq_ds = gdal.Open(str(args.out / "quality_c_ref_utm49_100m.tif"))
    hq, hq_valid = values(hq_ds)
    if ((hq[hq_valid] < 0) | (hq[hq_valid] > 1)).any():
        raise ValueError("HQ outside 0-1")
    results = {}
    for season in SEASONS:
        source_path = args.maxent_root / ("Waterbirds_" + season + "_avg.asc")
        source = gdal.Open(str(source_path))
        m, m_valid = values(source)
        if ((m[m_valid] < 0) | (m[m_valid] > 1)).any():
            raise ValueError("MaxEnt outside 0-1")
        target = args.out / ("m_" + season + "_utm49_100m.tif")
        translated = gdal.Translate(str(target), source, outputSRS=reference.GetProjection(),
                                    creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
        translated = None
        valid = m_valid & hq_valid
        if not valid.any():
            raise ValueError("No common MaxEnt/HQ pixels")
        controls = {"geometric_050": np.sqrt(np.maximum(m, 0) * np.maximum(hq, 0)),
                    "linear_050": 0.5 * m + 0.5 * hq, "minimum": np.minimum(m, hq)}
        outputs = []
        for method, scores in controls.items():
            output = args.out / ("conservation_" + season + "_" + method + ".tif")
            write_array(output, scores, valid, reference)
            outputs.append(metadata(output))
        results[season] = {
            "source_asc_sha256": digest(source_path), "m_output": metadata(target),
            "source_crs_audit": asc_crs_audit[season],
            "common_valid_pixels": int(valid.sum()), "outputs": outputs,
        }
    manifest = {
        "status": "EXPLORATORY_ARCHIVAL_CONSERVATION_BASELINES_NOT_VALIDATED",
        "reference": reference_info, "environmental_raster_count": len(environment),
        "environmental_rasters": environment, "aligned_invest_inputs": aligned_inputs,
        "season_results": results,
        "analysis_extent": "Valid input intersection; no municipal/wetland management boundary has been accepted",
        "base_predictions_are_group_out_of_fold": False,
        "maxent_output_transform": "NOT_ESTABLISHED_FROM_ARCHIVE_LOGS",
        "hq_is_static_across_seasons": True,
        "restoration_outputs_generated": False,
        "gate_trained": False,
        "missing_for_restoration": ["Matched sensitivity/H_j table", "Independent restoration feasibility A"],
        "missing_for_gate": ["Independent target definitions/data", "Fold-wise base predictions and producer logs", "M/Q ensemble uncertainty"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "environmental_rasters": len(environment),
                      "common_pixels_by_season": {k: v["common_valid_pixels"] for k, v in results.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
