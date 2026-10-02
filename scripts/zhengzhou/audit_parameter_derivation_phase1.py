# -*- coding: utf-8 -*-
"""Exploratory presence/background diagnostics, independently reviewed 2026-10-02.

Historical WorkBuddy outputs are not approved InVEST parameters. A stable
classifier surface does not establish intrinsic habitat identifiability,
independent supervision, calibrated occurrence probability or causal threats.
Use a NEW --out directory; original phase outputs must remain immutable.
"""
import collections
import csv
import argparse
from wetland_coupling.parameter_diagnostics import (fit_logistic, rank_correlation, ols_vif, record_input, save_report)
INPUT_INVENTORY = {}
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "local_work" / "zhengzhou_pilot" / "maxent_formal_v1_003"
LULC = ROOT / "local_work" / "zhengzhou_pilot" / "archival_baselines_003" / "lulc_cur_utm49_100m.tif"
THREATS = ROOT / "local_work" / "zhengzhou_pilot" / "archival_baselines_003"
OUT = ROOT / "local_work" / "zhengzhou_pilot" / "parameter_derivation_audit_001"

THREAT_NAMES = ("urban_structure", "human_activity", "night_light")
MAXENT_VARS = ("waterfrequency", "ndvi", "dem", "nightlight", "builtarea",
               "largebuildings", "crops", "grass", "trees", "bareground")


def read_raster(path):
    record_input(path, INPUT_INVENTORY)
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None:
            a = np.where(a == ds.nodata, np.nan, a)
        return a, ds.crs, ds.shape


def load_table(path):
    record_input(path, INPUT_INVENTORY)
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def phase_a_inventory():
    rows = []
    for train_path in sorted(RUN.rglob("train.csv")):
        bg_path = train_path.parent / "background.csv"
        if not bg_path.exists():
            continue
        rel = train_path.parent.relative_to(RUN).as_posix()
        parts = rel.split("/")
        variant, season = parts[0], parts[1]
        scope = parts[2]
        scheme = parts[-1]
        tr = load_table(train_path)
        bg = load_table(bg_path)
        tr_keys = {(r["longitude"], r["latitude"]) for r in tr}
        bg_keys = {(r["longitude"], r["latitude"]) for r in bg}
        bg_only = len(bg_keys - tr_keys)
        rows.append({
            "variant": variant, "season": season, "scope": scope,
            "scheme": scheme,
            "presence_n": len(tr_keys),
            "background_total_n": len(bg_keys),
            "background_independent_n": bg_only,
            "presence_in_background_n": len(tr_keys & bg_keys),
        })
    return rows


def phase_b_lulc_join():
    lulc, crs, shape = read_raster(LULC)
    code_mask = np.isfinite(lulc) & (lulc >= 1) & (lulc <= 9)

    per_table = []
    presence_by_code = collections.Counter()
    background_by_code = collections.Counter()
    for train_path in sorted(RUN.rglob("train.csv")):
        bg_path = train_path.parent / "background.csv"
        if not bg_path.exists():
            continue
        rel = train_path.parent.relative_to(RUN).as_posix()
        parts = rel.split("/")
        tr = load_table(train_path)
        bg = load_table(bg_path)
        tr_keys = {(r["longitude"], r["latitude"]) for r in tr}
        bg_only_rows = [r for r in bg
                        if (r["longitude"], r["latitude"]) not in tr_keys]

        def codes_of(rows):
            if not rows:
                return [], 0
            lons = [float(r["longitude"]) for r in rows]
            lats = [float(r["latitude"]) for r in rows]
            xs, ys = warp_transform("EPSG:4326", str(crs), lons, lats)
            out, miss = [], 0
            with rasterio.open(LULC) as ds:
                arr = ds.read(1)
                T = ds.transform
            for x, y in zip(xs, ys):
                rr, cc = rowcol(T, x, y)
                if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]:
                    v = int(arr[rr, cc])
                    out.append(v if 1 <= v <= 9 else 0)
                else:
                    out.append(0)
                    miss += 1
            return out, miss

        tr_codes, tr_miss = codes_of(tr)
        bg_codes, bg_miss = codes_of(bg_only_rows)
        for c in tr_codes:
            if c:
                presence_by_code[c] += 1
        for c in bg_codes:
            if c:
                background_by_code[c] += 1
        per_table.append({
            "table": rel,
            "presence_n": len(tr_codes), "presence_unmapped": tr_miss,
            "background_independent_n": len(bg_codes),
            "background_unmapped": bg_miss,
            "presence_codes": dict(sorted(collections.Counter(
                c for c in tr_codes if c).items())),
        })

    ratios = {}
    for c in range(1, 10):
        p = presence_by_code.get(c, 0)
        b = background_by_code.get(c, 0)
        ratios[c] = {
            "presence_observations": p,
            "background_observations": b,
            "presence_to_background": (round(p / b, 3) if b else None),
        }
    return {
        "lulc": str(LULC),
        "code_mask_pixels": int(code_mask.sum()),
        "per_table": per_table,
        "aggregated_by_code": ratios,
        "total_presence_observations": int(sum(presence_by_code.values())),
        "total_background_observations": int(sum(background_by_code.values())),
    }


def phase_c_threat_overlap():
    arrays = {}
    for name in THREAT_NAMES:
        a, _, _ = read_raster(THREATS / f"{name}_utm49_100m.tif")
        arrays[name] = a
    env = ROOT / "local_work" / "zhengzhou_pilot" / "supermap" / "environment_exports"
    for name, path in (("env_nightlight", env / "spr_select" / "nightlight_spring.tif"),
                       ("env_builtarea", env / "spr_select" / "builtarea_spring.tif"),
                       ("env_largebuildings", env / "spr_select" / "largebuildings_spring.tif"),
                       ("env_crops", env / "spr_select" / "crops_spring.tif"),
                       ("env_water", env / "spr_select" / "water_spring.tif")):
        if path.exists():
            a, _, _ = read_raster(path)
            arrays[name] = a

    mask = np.ones_like(next(iter(arrays.values())), dtype=bool)
    for a in arrays.values():
        mask &= np.isfinite(a)
    names = list(arrays)
    vecs = {n: arrays[n][mask] for n in names}

    def spearman(x, y):
        return rank_correlation(x, y)

    threat_block = {}
    for i, a in enumerate(THREAT_NAMES):
        for b in THREAT_NAMES[i + 1:]:
            threat_block[f"{a}|{b}"] = round(spearman(vecs[a], vecs[b]), 4)
    cross_block = {}
    for t in THREAT_NAMES:
        for e in names:
            if e in THREAT_NAMES:
                continue
            cross_block[f"{t}|{e}"] = round(spearman(vecs[t], vecs[e]), 4)

    stats = {}
    for n in THREAT_NAMES:
        v = vecs[n]
        stats[n] = {
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "mean": round(float(v.mean()), 4),
            "within_0_1_pct": round(100.0 * float(((v >= 0) & (v <= 1)).mean()), 2),
            "positive_pct": round(100.0 * float((v > 0).mean()), 2),
            "std": round(float(v.std()), 4),
        }
    return {
        "valid_pixels": int(mask.sum()),
        "threat_rasters": stats,
        "threat_pair_spearman": threat_block,
        "threat_vs_environment_spearman": cross_block,
        "finding": ("Correlation is descriptive. High positive correlation "
                    "between threat rasters means InVEST's additive threat "
                    "term can count the same urbanisation pressure more than "
                    "once; it does not by itself prove redundancy because "
                    "the rasters may differ in how they were constructed."),
    }


def phase_d_lambdas():
    per_var = collections.defaultdict(list)
    n_models = 0
    for path in RUN.rglob("*.lambdas"):
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                name = parts[0]
                if name in ("linearPredictorNormalizer", "densityNormalizer",
                            "numBackgroundPoints", "numSamples",
                            "numSamplesWithResidual", "equivalentSampleSize"):
                    continue
                try:
                    per_var[name].append(float(parts[1]))
                except ValueError:
                    continue
            n_models += 1
        except Exception:
            continue
    summary = {}
    for name, vals in per_var.items():
        arr = np.array(vals, dtype="float64")
        summary[name] = {
            "models": int(arr.size),
            "mean": round(float(arr.mean()), 5),
            "median": round(float(np.median(arr)), 5),
            "sd": round(float(arr.std()), 5),
            "abs_mean": round(float(np.abs(arr).mean()), 5),
            "share_near_zero_pct": round(100.0 * float((np.abs(arr) < 1e-6).mean()), 2),
            "share_positive_pct": round(100.0 * float((arr > 0).mean()), 2),
        }
    return {"models_scanned": n_models, "variables": summary}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    result = {
        "audit": "parameter_derivation_audit_001",
        "phase": "0+1 data self-audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "declaration": {
            "purpose": ("pre-derivation self-audit; no parameters derived, "
                        "no config written, no official InVEST run"),
            "independence": ("all inputs are MaxEnt development artifacts; "
                             "any H_j or threat weight later derived from them "
                             "is model-derived and NOT independent evidence"),
            "locked_test": "not read in this phase",
        },
        "phase_a_sample_inventory": phase_a_inventory(),
        "phase_b_lulc_join": phase_b_lulc_join(),
        "phase_c_threat_overlap": phase_c_threat_overlap(),
        "phase_d_lambda_audit": phase_d_lambdas(),
    }

    inv = result["phase_a_sample_inventory"]
    schemes = collections.Counter((r["scheme"]) for r in inv)
    schemes_ind = collections.Counter()
    for r in inv:
        schemes_ind[r["scheme"]] += r["background_independent_n"]
    result["phase_a_summary"] = {
        "tables": len(inv),
        "by_scheme_table_count": dict(schemes),
        "by_scheme_total_independent_background": dict(schemes_ind),
        "note": ("addsamplestobackground=true means each background file "
                 "contains every presence row; only the residual rows are "
                 "independent background."),
    }

    save_report(OUT, 'phase1_data_self_audit.json', result, __file__, INPUT_INVENTORY)

    print("== A 样本清单 ==")
    print("表数量:", len(inv), "| 按方案:", dict(schemes))
    print("独立背景合计:", dict(schemes_ind))
    b = result["phase_b_lulc_join"]
    print("\n== B LULC 连接 ==")
    print("presence 观测:", b["total_presence_observations"],
          "| background 观测:", b["total_background_observations"])
    for c, v in b["aggregated_by_code"].items():
        print("  code %s: presence=%4d background=%5d p/b=%s" % (
            c, v["presence_observations"], v["background_observations"],
            v["presence_to_background"]))
    c_ = result["phase_c_threat_overlap"]
    print("\n== C 威胁重叠 ==")
    for k, v in c_["threat_pair_spearman"].items():
        print("  %s = %.4f" % (k, v))
    print("  值域:", {k: (v["min"], v["max"], v["within_0_1_pct"]) for k, v in c_["threat_rasters"].items()})
    d = result["phase_d_lambda_audit"]
    print("\n== D 变量贡献（%d 个模型）==" % d["models_scanned"])
    for name, v in sorted(d["variables"].items(), key=lambda t: -abs(t[1]["mean"])):
        print("  %-22s mean=%9.5f  |mean|=%8.5f  近零=%5.1f%%  正=%5.1f%%" % (
            name, v["mean"], v["abs_mean"], v["share_near_zero_pct"],
            v["share_positive_pct"]))


if __name__ == "__main__":
    main()
