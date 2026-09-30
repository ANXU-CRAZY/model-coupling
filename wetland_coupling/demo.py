"""Synthetic engineering fixtures. No fitted MaxEnt or InVEST output is fabricated."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .audit import SEASONS, file_sha256, save_json


def make_demo(output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(20):
        role = "train" if g<12 else "validation" if g<16 else "test"
        for i in range(24):
            season = i%4
            potential, pressure = rng.uniform(0.05,0.95,2)
            true_m = 0.12 + 0.8*potential
            true_h = 0.94 - 0.78*pressure
            tau_m = 0.03 + 0.10*(season in (1,2))
            tau_h = 0.03 + 0.07*(season in (0,3))
            # Observations come from latent environmental variables, not the exported m/h.
            yc = np.clip(0.10 + 0.76*potential - 0.20*pressure + rng.normal(0,0.025), 0,1)
            yr = np.clip(0.04 + 0.57*potential + 0.25*pressure + rng.normal(0,0.025), 0,1)
            feasible = int(rng.random()>0.1)
            rows.append({"sample_id": f"demo_{g:02d}_{i:02d}", "site_id": f"site_{g:02d}_{i:02d}",
                         "group_id": f"block_{g:02d}", "year": 2025 if role=="test" else 2023+i%2,
                         "season": SEASONS[season], "role": role,
                         "m": np.clip(true_m+rng.normal(0,tau_m),0.001,0.999),
                         "h": np.clip(true_h+rng.normal(0,tau_h),0.001,0.999),
                         "u_m": tau_m, "u_h": tau_h, "feasible": feasible, "habitat_suitability": 1.,
                         "target_protection": yc, "target_restoration": yr if feasible else np.nan,
                         "m_artifact_id": f"M_{g:02d}" if role=="train" else "M_outer",
                         "h_artifact_id": f"H_{g:02d}" if role=="train" else "H_outer"})
    df = pd.DataFrame(rows)
    df.to_csv(output/"samples.csv", index=False, encoding="utf-8")
    artifacts = {}
    for g in list(range(12))+[None]:
        source = df[(df.role=="train") & (df.group_id != f"block_{g:02d}" if g is not None else True)]
        for prefix, kind in (("M", "maxent"), ("H", "invest")):
            key = f"{prefix}_{g:02d}" if g is not None else f"{prefix}_outer"
            artifacts[key] = {"model_type": kind, "output_type": "cloglog" if kind=="maxent" else "HQ",
                              "software": "synthetic_fixture_no_ecological_model_run",
                              "evidence_uri": "synthetic://demo.py",
                              "fit_sample_ids": source.sample_id.tolist(), "fit_group_ids": sorted(set(source.group_id)),
                              "tune_sample_ids": [], "tune_group_ids": [],
                              "calibrate_sample_ids": [], "calibrate_group_ids": []}
    provenance = {"data_kind": "synthetic", "test_scope": "spatiotemporal", "artifacts": artifacts,
                  "table_sha256": file_sha256(output/"samples.csv"),
                  "supervision": {task: {"source_kind": "synthetic", "evidence_uri": "synthetic://demo.py",
                                          "definition": "Independent latent-environment synthetic response; engineering fixture only",
                                          "normalization": "Explicit synthetic [0,1] clipping", "derived_from_m_or_h": False}
                                  for task in ("protection", "restoration")}}
    save_json(output/"provenance.json", provenance)
    make_raster_fixtures(output/"raster_inputs", rng)
    return {"status": "SYNTHETIC_FIXTURE_ONLY", "rows": len(df), "path": str(output.resolve())}


def make_raster_fixtures(output, rng):
    import rasterio
    from rasterio.transform import from_origin
    output.mkdir()
    y, x = np.mgrid[:32,:40]
    m = np.clip(0.1 + x/50 + 0.1*np.sin(y/4),0.001,0.99)
    h = np.clip(0.95 - y/38 + 0.02*rng.normal(size=m.shape),0.001,0.999)
    arrays = {"m": m, "h": h, "feasible": np.ones_like(m), "habitat_suitability": np.ones_like(m)}
    arrays["feasible"][:4,20:30] = 0
    arrays["habitat_suitability"][27:,30:] = 0
    arrays["h"][27:,30:] = 0
    profile = {"driver": "GTiff", "height": 32, "width": 40, "count": 1,
               "dtype": "float32", "crs": "EPSG:32649", "transform": from_origin(350000,3850000,100,100),
               "nodata": -9999., "compress": "deflate"}
    for key,a in arrays.items():
        a = a.copy()
        a[0,:] = -9999.
        with rasterio.open(output/f"{key}.tif", "w", **profile) as dst:
            dst.write(a.astype("float32"),1)
            dst.update_tags(data_kind="synthetic", geographic_location="fictional_fixture")
