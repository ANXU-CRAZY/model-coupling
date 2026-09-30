"""Windowed GeoTIFF fusion. Strict alignment; no silent reprojection or rescaling."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio

from .audit import file_sha256, save_json
from .fusion import fuse, quadrants, unit_interval


def check_alignment(datasets):
    ref = next(iter(datasets.values()))
    if ref.crs is None or not ref.crs.is_projected:
        raise ValueError("A projected reference CRS is required")
    if ref.crs.linear_units not in ("metre", "meter"):
        raise ValueError("Reference CRS linear units must be metres")
    for name, ds in datasets.items():
        if ds.count != 1:
            raise ValueError(f"{name}: expected a single-band raster")
        if ds.crs != ref.crs or ds.width != ref.width or ds.height != ref.height:
            raise ValueError(f"{name}: mismatched CRS or shape; align upstream")
        if not ds.transform.almost_equals(ref.transform, precision=1e-9):
            raise ValueError(f"{name}: mismatched pixel origin/resolution; align upstream")
    return ref


def fuse_rasters(paths, output, method="geometric", alpha=0.5, gamma=0.5,
                 m_high=0.7, h_high=0.7):
    required = {"m", "h", "feasible", "habitat_suitability"}
    if set(paths) != required:
        raise ValueError(f"Raster input keys must be {sorted(required)}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    summary = {"status": "BASELINE_RESEARCH_SCORES", "method": method, "alpha": alpha, "gamma": gamma,
               "thresholds": {"m_high": m_high, "h_high": h_high}, "zone_counts": {},
               "valid_pixels": 0, "mask_disagreement_pixels": 0,
               "interpretation": "Restoration candidate ranking; not measured restoration benefit.",
               "inputs": {k: {"path": str(Path(v).resolve()), "sha256": file_sha256(v)} for k,v in paths.items()}}
    with ExitStack() as stack:
        sources = {k: stack.enter_context(rasterio.open(v)) for k,v in paths.items()}
        ref = check_alignment(sources)
        output.mkdir(parents=True, exist_ok=False)
        profile = ref.profile.copy()
        profile.update(driver="GTiff", dtype="float32", count=1, nodata=-9999., compress="deflate")
        writers = {k: stack.enter_context(rasterio.open(output/f"{k}.tif", "w", **profile))
                   for k in ("conservation_score", "restoration_candidate_score")}
        zprofile = profile.copy()
        zprofile.update(dtype="uint8", nodata=0)
        zwriter = stack.enter_context(rasterio.open(output/"management_reference_zones.tif", "w", **zprofile))
        zwriter.update_tags(zone_1="protection_reference", zone_2="restoration_candidate_reference",
                            zone_3="maintenance_reference", zone_4="general_management_reference",
                            zone_5="requires_feasibility_or_conversion_scenario_review")
        for writer in writers.values():
            writer.update_tags(score_type="dimensionless_research_index",
                               scientific_status="unvalidated_on_real_observations")
        for _, window in ref.block_windows(1):
            arrays = {k: ds.read(1, window=window, masked=True).astype(float) for k,ds in sources.items()}
            masks = [np.ma.getmaskarray(a) | ~np.isfinite(a.data) for a in arrays.values()]
            invalid = np.logical_or.reduce(masks)
            summary["mask_disagreement_pixels"] += int((np.logical_or.reduce(masks) ^ np.logical_and.reduce(masks)).sum())
            values = {k: np.where(invalid, np.nan, a.data) for k,a in arrays.items()}
            unit_interval(values["habitat_suitability"], "H_j")
            if np.any(values["h"] > values["habitat_suitability"] + 1e-6):
                raise ValueError("HQ exceeds H_j; check the sensitivity/LULC lookup")
            scores = fuse(values["m"], values["h"], alpha, gamma, method=method,
                          feasible=values["feasible"], habitat_suitability=values["habitat_suitability"])
            for source_key, target_key in (("conservation", "conservation_score"),
                                           ("restoration_candidate", "restoration_candidate_score")):
                data = scores[source_key]
                writers[target_key].write(np.where(np.isfinite(data), data, -9999).astype("float32"), 1, window=window)
            zones = quadrants(values["m"], values["h"], values["feasible"], values["habitat_suitability"], m_high, h_high)
            zwriter.write(zones, 1, window=window)
            codes, counts = np.unique(zones, return_counts=True)
            for code, count in zip(codes, counts):
                key = str(int(code))
                summary["zone_counts"][key] = summary["zone_counts"].get(key, 0) + int(count)
            summary["valid_pixels"] += int((~invalid).sum())
        summary["reference"] = {"crs": ref.crs.to_string(), "width": ref.width, "height": ref.height,
                                "transform": list(ref.transform), "resolution": list(ref.res)}
    summary["outputs_sha256"] = {p.name: file_sha256(p) for p in output.glob("*.tif")}
    save_json(output/"manifest.json", summary)
    return summary
