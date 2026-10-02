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
from rasterio.warp import transform as warp_transform
from rasterio.transform import rowcol

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "local_work" / "zhengzhou_pilot" / "maxent_formal_v1_003"
BASE = ROOT / "local_work" / "zhengzhou_pilot" / "archival_baselines_003"
ENV = ROOT / "local_work" / "zhengzhou_pilot" / "supermap" / "environment_exports"
OUT = ROOT / "local_work" / "zhengzhou_pilot" / "parameter_derivation_audit_001"

SEASON_DIR = {"spring": "spr_select", "summer": "sum_select",
              "autumn": "aut_select", "winter": "win_select"}
SEASONS = ("spring", "summer", "autumn", "winter")
SCHEMES = ("B0_target_group", "B1_uniform", "B2_visit_density_proxy")
FOLDS = (0, 1, 2)
VARIANTS = ("no_lst", "static_lst")
CODES = tuple(range(1, 10))
MAXENT_DOMAIN = RUN.parent / "maxent_formal_v1_001" / "inputs" / "common_valid_mask.npy"


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
        return a


def logistic(X, y, ridge=1e-4, max_iter=300, tol=1e-10):
    return fit_logistic(X, y, ridge=ridge, max_iter=max_iter, tol=tol)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison-audit", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    lulc = read_raster(BASE / "lulc_cur_utm49_100m.tif")
    record_input(MAXENT_DOMAIN, INPUT_INVENTORY)
    domain = np.load(MAXENT_DOMAIN).astype(bool)
    split_path = RUN.parent / "maxent_formal_v1_001/splits/split_masks.npz"
    record_input(split_path, INPUT_INVENTORY)
    with np.load(split_path, allow_pickle=False) as masks:
        fit_masks = {fold: masks[f"outer_{fold}_fit"].copy() for fold in FOLDS}
    code_mask = domain & np.isfinite(lulc) & (lulc >= 1) & (lulc <= 9)
    codes_flat = lulc[code_mask].astype(int)

    per_season = {}
    for season in SEASONS:
        # Cache the union only for IO; each fit uses its OWN frozen columns.
        tables = [RUN / variant / season / f"outer_{fold}" / "refit" / scheme / "train.csv"
                  for variant in VARIANTS for scheme in SCHEMES for fold in FOLDS]
        cols = sorted({c for table in tables for c in load_table(table)[0]
                       if c not in ("species", "longitude", "latitude")})
        rasters, kept = {}, []
        for c in cols:
            path = ENV / SEASON_DIR[season] / f"{c}.tif"
            if not path.exists():
                raise FileNotFoundError(path)
            arr = read_raster(path)
            if arr.shape != lulc.shape:
                raise ValueError("Predictor grid shape differs: " + str(path))
            with rasterio.open(path) as ds, rasterio.open(BASE / "lulc_cur_utm49_100m.tif") as ref:
                if ds.crs != ref.crs or ds.transform != ref.transform:
                    raise ValueError("Predictor CRS/transform differs: " + str(path))
            rasters[c] = arr
            kept.append(c)
        if not kept:
            continue
        stack = np.column_stack([rasters[c][code_mask] for c in kept])
        per_season[season] = {"columns": kept, "stack": stack,
                              "domain_pixels": int(code_mask.sum())}
        print(f"{season}: {len(kept)}/{len(cols)} 变量可用, 有效像元 {code_mask.sum()}")

    results = []
    for season, info in per_season.items():
        cols, stack = info["columns"], info["stack"]
        for variant in VARIANTS:
            for scheme in SCHEMES:
                for fold in FOLDS:
                    d = RUN / variant / season / f"outer_{fold}" / "refit" / scheme
                    if not (d / "train.csv").exists():
                        continue
                    tr = load_table(d / "train.csv")
                    bg = load_table(d / "background.csv")
                    keys = {(r["longitude"], r["latitude"]) for r in tr}
                    rows = tr + [r for r in bg
                                 if (r["longitude"], r["latitude"]) not in keys]
                    use = [c for c in rows[0] if c not in ("species", "longitude", "latitude")]
                    if not set(use).issubset(cols):
                        raise ValueError("Selected predictor missing from raster cache")
                    if len(use) < 3:
                        continue
                    Xs = np.array([[float(r[c]) for c in use] for r in rows])
                    keep = np.isfinite(Xs).all(axis=1)
                    Xs = Xs[keep]
                    n_pres = len(tr)
                    y = np.array([1.0] * n_pres + [0.0] * (len(rows) - n_pres))[keep]
                    if len(np.unique(y)) < 2:
                        continue
                    mu, sd = Xs.mean(axis=0), Xs.std(axis=0)
                    sd[sd == 0] = 1.0
                    idx = [cols.index(c) for c in use]
                    P = stack[:, idx]
                    ok = np.isfinite(P).all(axis=1) & fit_masks[fold][code_mask]
                    Z = np.zeros((int(ok.sum()), len(use) + 1))
                    Z[:, 0] = 1.0
                    Z[:, 1:] = (P[ok] - mu) / sd
                    Xn = np.column_stack([np.ones(len(Xs)), (Xs - mu) / sd])
                    beta = logistic(Xn, y)
                    pred = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -35, 35)))
                    cls = codes_flat[ok]
                    prof = {c: (float(pred[cls == c].mean())
                                if (cls == c).any() else None) for c in CODES}
                    vals = [v for v in prof.values() if v is not None]
                    mx = max(vals) if vals else 1.0
                    hj = {c: (round(prof[c] / mx, 4) if prof[c] is not None else None)
                          for c in CODES}
                    raw = {c: (round(prof[c], 6) if prof[c] is not None else None)
                           for c in CODES}
                    results.append({
                        "variant": variant, "season": season, "scheme": scheme,
                        "fold": fold, "variables": use,
                        "n_presence": int((y == 1).sum()),
                        "n_background": int((y == 0).sum()),
                        "H_j_profile": hj,
                        "profile_raw_mean_prediction": raw,
                    })
        print(f"{season}: 累计 {len([r for r in results if r['season']==season])} 次拟合")

    # aggregation
    def agg(season=None, scheme=None, variant=None):
        rows = [r for r in results
                if (season is None or r["season"] == season)
                and (scheme is None or r["scheme"] == scheme)
                and (variant is None or r["variant"] == variant)]
        if not rows:
            return None
        out = {}
        for c in CODES:
            v = [r["H_j_profile"][c] for r in rows if r["H_j_profile"][c] is not None]
            out[c] = {"mean": round(float(np.mean(v)), 4),
                      "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0,
                      "n": len(v)}
        return out

    def rank_corr(a, b):
        value = rank_correlation(a, b)
        return None if value is None else round(value, 3)

    # stability: within season, across schemes (no_lst only)
    stability = {}
    for season in per_season:
        vecs = {}
        for scheme in SCHEMES:
            e = agg(season=season, scheme=scheme, variant="no_lst")
            if e:
                vecs[scheme] = np.array([e[c]["mean"] for c in CODES])
        for a in SCHEMES:
            for b in SCHEMES:
                if a < b and a in vecs and b in vecs:
                    stability[f"{season}:{a} vs {b}"] = rank_corr(vecs[a], vecs[b])

    # agreement with the non-parametric presence/area ratio
    record_input(args.comparison_audit, INPUT_INVENTORY)
    ratio = json.loads(args.comparison_audit.read_text(encoding="utf-8"))
    ratio_by_season = {}
    for r in ratio["per_fit"]:
        if r["variant"] != "no_lst" or r["scheme"] != "B2_visit_density_proxy":
            continue
        ratio_by_season.setdefault(r["season"], []).append(r["presence_background_ratio"])
    agreement = {}
    for season, lst in ratio_by_season.items():
        e = agg(season=season, scheme="B2_visit_density_proxy", variant="no_lst")
        if not e:
            continue
        a = np.array([e[c]["mean"] for c in CODES])
        b = np.array([float(np.mean([x[str(c)] for x in lst if x[str(c)] is not None]))
                      for c in CODES])
        agreement[season] = {
            "spearman_profile_vs_presence_ratio": rank_corr(a, b),
            "pearson": round(float(np.corrcoef(a, b)[0, 1]), 3),
        }

    seasons_agree = [v for v in stability.values() if v is not None]
    verdict = {
        "cross_scheme_rank_agreement_all_positive": bool(len(seasons_agree) == 12 and all(v > 0 for v in seasons_agree)),
        "min_cross_scheme_rank_corr": min(seasons_agree) if seasons_agree else None,
        "profile_vs_ratio_agreement": agreement,
        "mean_profile_vs_ratio_spearman": (round(float(np.mean(
            [v["spearman_profile_vs_presence_ratio"] for v in agreement.values()
             if v["spearman_profile_vs_presence_ratio"] is not None])), 3)
            if agreement else None),
    }

    report = {
        "audit": "parameter_derivation_audit_005",
        "phase": "3 profile-based H_j (shared surface, no per-class coefficient)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "model": "logistic presence ~ continuous predictors of that season",
            "prediction": "class profiling only within each outer fit mask; no held-out/locked distribution used",
            "H_j": "mean prediction inside each LULC code, max-normalised to 1",
            "why": "Shared surface smooths classes but cannot restore missing class evidence or establish intrinsic H_j",
            "season_variables": {s: per_season[s]["columns"] for s in per_season},
            "locked_test": "not read",
        },
        "declaration": {
            "meaning": ("relative occurrence intensity, NOT a field-measured "
                        "suitability and NOT a calibrated probability"),
            "independence": ("derived from MaxEnt development observations; "
                             "M->Q dependency is structural"),
            "status": "exploratory model-derived prior; eligible_for_official_run=false",
        },
        "fits": len(results),
        "summary_by_season_scheme": {
            season: {scheme: agg(season=season, scheme=scheme, variant="no_lst")
                     for scheme in SCHEMES}
            for season in per_season},
        "summary_no_lst_all_schemes": agg(variant="no_lst"),
        "cross_scheme_stability": stability,
        "verdict": verdict,
        "per_fit": results,
    }
    save_report(OUT, 'phase3_profile_hj.json', report, __file__, INPUT_INVENTORY)

    print("\nH_j 剖面（no_lst，按方案）:")
    for season in per_season:
        print(f"\n{season}:")
        print("  code  " + "".join(f"{s:>20}" for s in SCHEMES))
        for c in CODES:
            cells = []
            for s in SCHEMES:
                e = agg(season=season, scheme=s, variant="no_lst")
                cells.append(f"{e[c]['mean']:.3f}±{e[c]['sd']:.3f}" if e else "n/a")
            print(f"   {c}    " + "".join(f"{x:>20}" for x in cells))
    print("\n跨方案秩相关稳定性:", json.dumps(stability, ensure_ascii=False, indent=1))
    print("\n与presence/面积比的一致性:", json.dumps(agreement, ensure_ascii=False))
    print("\n判定:", json.dumps(verdict, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
