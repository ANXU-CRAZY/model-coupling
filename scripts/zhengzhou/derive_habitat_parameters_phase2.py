# -*- coding: utf-8 -*-
"""Exploratory presence/background diagnostics, independently reviewed 2026-10-02.

Historical WorkBuddy outputs are not approved InVEST parameters. A stable
classifier surface does not establish intrinsic habitat identifiability,
independent supervision, calibrated occurrence probability or causal threats.
Use a NEW --out directory; original phase outputs must remain immutable.
"""
import csv
import argparse
from wetland_coupling.parameter_diagnostics import (fit_logistic, rank_correlation, ols_vif, record_input, save_report)
INPUT_INVENTORY = {}
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "local_work" / "zhengzhou_pilot" / "maxent_formal_v1_003"
BASE = ROOT / "local_work" / "zhengzhou_pilot" / "archival_baselines_003"
OUT = ROOT / "local_work" / "zhengzhou_pilot" / "parameter_derivation_audit_001"

THREATS = ("urban_structure", "human_activity", "night_light")
CODES = tuple(range(1, 10))
SEASONS = ("spring", "summer", "autumn", "winter")
SCHEMES = ("B0_target_group", "B1_uniform", "B2_visit_density_proxy")
FOLDS = (0, 1, 2)
RIDGE = 1e-6


# ----------------------------------------------------------------- io helpers
def load_table(path):
    record_input(path, INPUT_INVENTORY)
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_raster(path):
    record_input(path, INPUT_INVENTORY)
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None:
            a = np.where(a == ds.nodata, np.nan, a)
        return a, str(ds.crs), ds.transform


def sample_rasters(rows, layers):
    """layers: dict name -> (array, crs, transform). Returns dict name -> values."""
    lons = [float(r["longitude"]) for r in rows]
    lats = [float(r["latitude"]) for r in rows]
    out = {}
    for name, (arr, crs, T) in layers.items():
        xs, ys = warp_transform("EPSG:4326", crs, lons, lats)
        vals = np.empty(len(xs), dtype="float64")
        for i, (x, y) in enumerate(zip(xs, ys)):
            rr, cc = rowcol(T, x, y)
            if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]:
                vals[i] = arr[rr, cc]
            else:
                vals[i] = np.nan
        out[name] = vals
    return out


# ------------------------------------------------------------- logistic irls
def logistic_irls(X, y, ridge=RIDGE, max_iter=200, tol=1e-9):
    return fit_logistic(X, y, ridge=ridge, max_iter=max_iter, tol=tol)


def design_code(code):
    """Intercept + 8 dummies (code 1 as reference)."""
    cols = [np.ones(len(code))]
    for c in CODES[1:]:
        cols.append((code == c).astype("float64"))
    return np.column_stack(cols)


def design_threats(t):
    cols = [np.ones(len(t[0]))]
    for v in t:
        cols.append(v)
    return np.column_stack(cols)


def design_joint(code, t):
    return np.hstack([design_code(code), design_threats(t)[:, 1:]])


def vif(X_names, X):
    values = ols_vif(np.asarray(X)[:, 1:])
    return {name: (None if value is None else round(value, 3))
            for name, value in zip(X_names[1:], values)}


# ------------------------------------------------------------------- main run
def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    lulc_arr, lulc_crs, lulc_T = read_raster(BASE / "lulc_cur_utm49_100m.tif")
    layers = {"lulc": (lulc_arr, lulc_crs, lulc_T)}
    for name in THREATS:
        arr, crs, T = read_raster(BASE / f"{name}_utm49_100m.tif")
        layers[name] = (arr, crs, T)

    results = []
    skipped = []
    for variant in ("no_lst", "static_lst"):
        for season in SEASONS:
            for scheme in SCHEMES:
                for fold in FOLDS:
                    d = RUN / variant / season / f"outer_{fold}" / "refit" / scheme
                    tr_p, bg_p = d / "train.csv", d / "background.csv"
                    if not (tr_p.exists() and bg_p.exists()):
                        skipped.append(d.relative_to(RUN).as_posix())
                        continue
                    tr, bg = load_table(tr_p), load_table(bg_p)
                    tr_keys = {(r["longitude"], r["latitude"]) for r in tr}
                    bg_only = [r for r in bg
                               if (r["longitude"], r["latitude"]) not in tr_keys]
                    rows = tr + bg_only
                    is_pres = np.array([1.0] * len(tr) + [0.0] * len(bg_only))
                    vals = sample_rasters(rows, layers)

                    code = vals["lulc"]
                    ok = np.isfinite(code) & (code >= 1) & (code <= 9) & (code == np.rint(code))
                    for name in THREATS:
                        ok &= np.isfinite(vals[name])
                    for r in is_pres[~ok]:
                        pass
                    if ok.sum() < 30 or len(set(code[ok])) < 2:
                        skipped.append(d.relative_to(RUN).as_posix() + " (thin)")
                        continue
                    code_i = np.array([int(c) for c in code[ok]])
                    y = is_pres[ok]

                    # standardised threats on the valid subset
                    tcols = []
                    tstats = {}
                    for name in THREATS:
                        v = vals[name][ok]
                        mu, sd = float(np.nanmean(v)), float(np.nanstd(v))
                        tstats[name] = {"mean": round(mu, 5), "sd": round(sd, 5)}
                        tcols.append(np.nan_to_num((v - mu) / (sd if sd else 1.0)))
                    t = tuple(tcols)

                    # Model A: habitat suitability by LULC code
                    XA = design_code(code_i)
                    bA = logistic_irls(XA, y)
                    # log-odds ratio relative to code 1, then normalise max -> 1
                    lo = {CODES[0]: 0.0}
                    for idx, c in enumerate(CODES[1:], start=1):
                        lo[c] = float(bA[idx])
                    odds = {c: math.exp(v) for c, v in lo.items()}
                    mx = max(odds.values())
                    hj = {c: round(odds[c] / mx, 4) for c in CODES}

                    # Model B: conditional threat weights
                    XB = design_threats(t)
                    bB = logistic_irls(XB, y)
                    threat_coef = {name: round(float(bB[i + 1]), 4)
                                   for i, name in enumerate(THREATS)}

                    # Model C: joint stability
                    XC = design_joint(code_i, t)
                    bC = logistic_irls(XC, y)
                    hjC_raw = {CODES[0]: 0.0}
                    for idx, c in enumerate(CODES[1:], start=1):
                        hjC_raw[c] = float(bC[idx])
                    oddsC = {c: math.exp(v) for c, v in hjC_raw.items()}
                    mxC = max(oddsC.values())
                    hjC = {c: round(oddsC[c] / mxC, 4) for c in CODES}
                    threat_coefC = {name: round(float(bC[len(CODES) + i]), 4)
                                    for i, name in enumerate(THREATS)}

                    # non-parametric check: presence/background density ratio
                    p_codes, b_codes = code_i[y == 1], code_i[y == 0]
                    pb = {}
                    for c in CODES:
                        n_p = int((p_codes == c).sum())
                        n_b = int((b_codes == c).sum())
                        pb[c] = {
                            "presence": n_p, "background": n_b,
                            "ratio": round(n_p / n_b, 4) if n_b else None,
                        }

                    results.append({
                        "variant": variant, "season": season,
                        "scheme": scheme, "fold": fold,
                        "n_presence": int((y == 1).sum()),
                        "n_background": int((y == 0).sum()),
                        "threat_scaling": tstats,
                        "H_j_model_A": hj,
                        "H_j_log_odds_A": {c: round(lo[c], 4) for c in CODES},
                        "threat_coef_model_B": threat_coef,
                        "H_j_model_C_joint": hjC,
                        "threat_coef_model_C_joint": threat_coefC,
                        "presence_background_ratio": pb,
                    })

    # ---------------- aggregate
    def agg(field, sub=None):
        out = {}
        for c in CODES:
            vals = []
            for r in results:
                if sub and r[sub[0]] != sub[1]:
                    continue
                vals.append(r[field][c])
            if vals:
                out[c] = {"mean": round(float(np.mean(vals)), 4),
                          "sd": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
                          "n": len(vals)}
        return out

    summary = {
        "H_j_by_scheme": {s: agg("H_j_model_A", ("scheme", s)) for s in SCHEMES},
        "H_j_by_season": {s: agg("H_j_model_A", ("season", s)) for s in SEASONS},
        "threat_coef_by_scheme": {
            s: {name: {
                "mean": round(float(np.mean([r["threat_coef_model_B"][name] for r in results
                                             if r["scheme"] == s])), 4),
                "sd": round(float(np.std([r["threat_coef_model_B"][name] for r in results
                                          if r["scheme"] == s], ddof=1)), 4)}
                for name in THREATS} for s in SCHEMES},
        "threat_coef_by_season": {
            s: {name: {
                "mean": round(float(np.mean([r["threat_coef_model_B"][name] for r in results
                                             if r["season"] == s])), 4),
                "sd": round(float(np.std([r["threat_coef_model_B"][name] for r in results
                                          if r["season"] == s], ddof=1)), 4)}
                for name in THREATS} for s in SEASONS},
        "H_j_joint_vs_A_max_abs_diff": round(float(max(
            abs(a - b) for c in CODES
            for a, b in zip(
                [r["H_j_model_C_joint"][c] for r in results],
                [r["H_j_model_A"][c] for r in results]))), 4),
        "VIF_threats_in_joint_design": None,
    }

    # VIF on the pooled joint design (one representative fit)
    if results:
        d = RUN / "no_lst" / "spring" / "outer_0" / "refit" / "B1_uniform"
        tr, bg = load_table(d / "train.csv"), load_table(d / "background.csv")
        tr_keys = {(r["longitude"], r["latitude"]) for r in tr}
        rows = tr + [r for r in bg if (r["longitude"], r["latitude"]) not in tr_keys]
        vals = sample_rasters(rows, layers)
        code = np.array([int(c) if 1 <= c <= 9 else 0 for c in vals["lulc"]])
        ok = code > 0
        t = []
        for name in THREATS:
            v = vals[name][ok]
            t.append(np.nan_to_num((v - np.nanmean(v)) / (np.nanstd(v) or 1.0)))
        names = ["intercept"] + [f"code_{c}" for c in CODES[1:]] + list(THREATS)
        summary["VIF_threats_in_joint_design"] = vif(
            names, design_joint(code[ok], tuple(t)))

    report = {
        "audit": "parameter_derivation_audit_002",
        "phase": "2 quantitative derivation of H_j and threat weights",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "model_A": "logistic presence ~ C(lulc_code); H_j = odds ratio, max-normalised",
            "model_B": "logistic presence ~ 3 standardised threats; coefficients are conditional",
            "model_C": "joint model for stability and VIF",
            "case_control": ("slopes interpreted as log-odds ratios; absolute "
                             "presence probability is not claimed"),
            "ridge": RIDGE,
            "locked_test": "not read",
            "fold_structure": "fitted separately per season x outer fold x background scheme",
        },
        "declaration": {
            "independence": ("H_j and threat weights are derived from the same "
                             "MaxEnt development observations, therefore they "
                             "are NOT independent evidence and M->Q dependency "
                             "is structural, not incidental"),
            "status": "exploratory model-derived prior; eligible_for_official_run=false",
            "class_semantics": ("H_j is indexed by LULC code 1..9. Code names "
                                "are not required for HQ execution and remain "
                                "PENDING_SOURCE_CROSSWALK for interpretation"),
        },
        "fits": len(results),
        "skipped": len(skipped),
        "summary": summary,
        "per_fit": results,
    }
    save_report(OUT, 'phase2_derived_parameters.json', report, __file__, INPUT_INVENTORY)

    print("完成拟合:", len(results), "| 跳过:", len(skipped))
    print("\nH_j 按背景方案（编码 1-9，均值）:")
    print("  code  " + "  ".join(f"{s:>18}" for s in SCHEMES))
    for c in CODES:
        cells = []
        for s in SCHEMES:
            e = summary["H_j_by_scheme"][s].get(c)
            cells.append(f"{e['mean']:.3f}±{e['sd']:.3f}" if e else "n/a")
        print(f"   {c}    " + "  ".join(f"{x:>18}" for x in cells))
    print("\n威胁系数（条件效应，按方案）:")
    for s in SCHEMES:
        e = summary["threat_coef_by_scheme"][s]
        print(f"  {s:24s} " + "  ".join(
            f"{n}={e[n]['mean']:+.3f}±{e[n]['sd']:.3f}" for n in THREATS))
    print("\n联合模型与A模型H_j最大差异:", summary["H_j_joint_vs_A_max_abs_diff"])
    print("VIF:", json.dumps(summary["VIF_threats_in_joint_design"], ensure_ascii=False))


if __name__ == "__main__":
    main()
