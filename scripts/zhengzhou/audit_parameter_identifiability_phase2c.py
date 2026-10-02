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
MIN_PRESENCE_PER_CLASS = 30   # exploratory count screen only; not a universal estimability bound
CAP = 8.0                     # numerical guard against perfect separation


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


def sample(rows, layers):
    lons = [float(r["longitude"]) for r in rows]
    lats = [float(r["latitude"]) for r in rows]
    out = {}
    for name, (arr, crs, T) in layers.items():
        xs, ys = warp_transform("EPSG:4326", crs, lons, lats)
        v = np.empty(len(xs))
        for i, (x, y) in enumerate(zip(xs, ys)):
            rr, cc = rowcol(T, x, y)
            v[i] = arr[rr, cc] if (0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]) else np.nan
        out[name] = v
    return out


def logistic_w(X, y, w=None, ridge=1e-3, max_iter=300, tol=1e-10):
    return fit_logistic(X, y, weights=w, ridge=ridge, max_iter=max_iter, tol=tol)


def dummies(code, levels):
    cols = [np.ones(len(code))]
    for c in levels[1:]:
        cols.append((code == c).astype(float))
    return np.column_stack(cols)


def norm_hj(lo, levels, cap=CAP):
    clipped = {c: (max(-cap, min(cap, lo[c])), bool(abs(lo[c]) > cap)) for c in levels}
    odds = {c: math.exp(clipped[c][0]) for c in levels}
    mx = max(odds.values())
    return ({c: odds[c] / mx for c in levels},
            sorted([c for c in levels if clipped[c][1]]))


def rank_corr(a, b):
    value = rank_correlation(a, b)
    return None if value is None else round(value, 3)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    layers = {"lulc": read_raster(BASE / "lulc_cur_utm49_100m.tif")}
    lulc = layers["lulc"][0]
    for n in THREATS:
        layers[n] = read_raster(BASE / f"{n}_utm49_100m.tif")

    # threat separability over the whole valid domain (D6)
    domain = np.isfinite(lulc) & (lulc >= 1) & (lulc <= 9)
    for n in THREATS:
        domain &= np.isfinite(layers[n][0])
    names = list(THREATS)
    M = np.column_stack([layers[n][0][domain] for n in names])
    corr = np.corrcoef(M, rowvar=False)
    vif = {name: (None if value is None else round(value, 3))
           for name, value in zip(names, ols_vif(M))}
    if any(value is None for value in vif.values()):
        raise ValueError("Threat VIF is undefined due to constant/perfect dependence")

    fits = []
    for variant in ("no_lst", "static_lst"):
        for season in SEASONS:
            for scheme in SCHEMES:
                for fold in FOLDS:
                    d = RUN / variant / season / f"outer_{fold}" / "refit" / scheme
                    if not (d / "train.csv").exists():
                        continue
                    tr = load_table(d / "train.csv")
                    bg = load_table(d / "background.csv")
                    keys = {(r["longitude"], r["latitude"]) for r in tr}
                    bg_only = [r for r in bg
                               if (r["longitude"], r["latitude"]) not in keys]
                    rows = tr + bg_only
                    y = np.array([1.0] * len(tr) + [0.0] * len(bg_only))
                    vals = sample(rows, layers)
                    code = np.array([int(c) if 1 <= c <= 9 else 0 for c in vals["lulc"]])
                    ok = code > 0
                    for name in THREATS:
                        ok &= np.isfinite(vals[name])
                    code, y = code[ok], y[ok]
                    if len(np.unique(y)) < 2 or len(set(code.tolist())) < 2:
                        continue

                    counts = {c: int(((code == c) & (y == 1)).sum()) for c in CODES}
                    bcounts = {c: int(((code == c) & (y == 0)).sum()) for c in CODES}
                    ratio = {c: (round(counts[c] / bcounts[c], 4) if bcounts[c] else None)
                             for c in CODES}

                    w = np.zeros(len(y))
                    for c in CODES:
                        for cls in (1.0, 0.0):
                            m = (code == c) & (y == cls)
                            if m.sum():
                                w[m] = 0.5 / m.sum()
                    w *= len(y)

                    beta = logistic_w(dummies(code, CODES), y, w)
                    lo = {CODES[0]: 0.0}
                    for i, c in enumerate(CODES[1:], start=1):
                        lo[c] = float(beta[i])
                    hj, clipped = norm_hj(lo, CODES)

                    t = []
                    tdesc = {}
                    for n in THREATS:
                        v = vals[n][ok]
                        sd = float(np.nanstd(v)) or 1.0
                        t.append(np.nan_to_num((v - np.nanmean(v)) / sd))
                        tdesc[n] = {"mean": round(float(np.nanmean(v)), 4),
                                    "sd": round(sd, 4),
                                    "zero_pct": round(100 * float(np.nanmean(v == 0)), 2)}
                    bt = logistic_w(np.column_stack([np.ones(len(y))] + t), y, w)
                    tc = {n: round(float(bt[i + 1]), 4) for i, n in enumerate(THREATS)}

                    fits.append({
                        "variant": variant, "season": season, "scheme": scheme,
                        "fold": fold,
                        "n_presence": int((y == 1).sum()),
                        "n_background": int((y == 0).sum()),
                        "presence_per_code": counts,
                        "background_per_code": bcounts,
                        "presence_background_ratio": ratio,
                        "H_j_balanced": {c: round(hj[c], 4) for c in CODES},
                        "clipped_codes": clipped,
                        "threat_coef_conditional": tc,
                        "threat_distributions": tdesc,
                    })

    # ---- D3/D4 rank agreement
    def scheme_vec(s, season=None):
        rows = [r for r in fits if r["scheme"] == s
                and (season is None or r["season"] == season)]
        if not rows:
            return None
        return np.array([float(np.mean([r["H_j_balanced"][c] for r in rows]))
                         for c in CODES])

    d3 = {}
    for s in SCHEMES:
        per_season = {}
        for season in SEASONS:
            a, b = scheme_vec(s, season), None
            others = [x for x in SCHEMES if x != s]
            key = others[0]
            b = scheme_vec(key, season)
            per_season[season] = {"compare": f"{s} vs {key}",
                                  "rank_corr": rank_corr(a, b)}
        d3[s] = per_season

    d4 = {}
    for s in SCHEMES:
        for season in SEASONS[1:]:
            a = scheme_vec(s, season)
            b = scheme_vec(s, SEASONS[0])
            d4[f"{s}:{season} vs {SEASONS[0]}"] = rank_corr(a, b)

    # ---- D1 estimability
    per_code_presence = {c: sorted([r["presence_per_code"][c] for r in fits])
                         for c in CODES}
    d1 = {c: {"min": per_code_presence[c][0],
              "median": int(np.median(per_code_presence[c])),
              "max": per_code_presence[c][-1],
              "estimable": bool(np.median(per_code_presence[c]) >= MIN_PRESENCE_PER_CLASS)}
          for c in CODES}

    # ---- verdict
    scheme_corrs = [v["rank_corr"] for per in d3.values()
                    for v in per.values() if v["rank_corr"] is not None]
    n_estimable = sum(1 for c in CODES if d1[c]["estimable"])
    max_threat_vif = max(vif.values())
    identifiable = (len(scheme_corrs) == 12 and all(c > 0 for c in scheme_corrs)
                    and n_estimable == len(CODES)
                    and max_threat_vif < 5)

    report = {
        "audit": "parameter_derivation_audit_004",
        "phase": "2c identifiability audit (supersedes pooled attempt 2b)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "why_superseded": ("pooling outer folds is invalid for presence-background "
                           "data: the same presence rows appear in every fold's "
                           "train.csv, so coordinate-membership labelling "
                           "misassigns other folds' presence as background. "
                           "Phase 2 per-fold structure is the correct one."),
        "min_presence_per_class": MIN_PRESENCE_PER_CLASS,
        "log_odds_cap": CAP,
        "D1_class_estimability": d1,
        "D3_rank_agreement_between_background_schemes": d3,
        "D4_rank_agreement_between_seasons": d4,
        "D6_threat_separability": {
            "pearson_correlation": {names[i]: {names[j]: round(float(corr[i, j]), 4)
                                              for j in range(len(names))}
                                    for i in range(len(names))},
            "vif": vif,
            "max_vif": max_threat_vif,
        },
        "verdict": {
            "H_j_identifiable": False,
            "heuristic_screen_passed": bool(identifiable),
            "class_balancing_erases_membership_prevalence": True,
            "reasons": [
                f"codes with median presence >= {MIN_PRESENCE_PER_CLASS}: {n_estimable}/9",
                f"between-scheme rank correlations: {scheme_corrs}",
                f"max threat VIF: {max_threat_vif}",
            ],
            "consequence": ("H_j may be reported as an exploratory relative "
                            "ordering with uncertainty. It is NOT a calibrated "
                            "suitability vector and must not enter "
                            "invest_prior.pending.json as a value."),
        },
        "declaration": {
            "independence": "derived from MaxEnt development observations; not independent evidence",
            "locked_test": "not read",
            "config_written": False,
        },
        "fits": len(fits),
        "per_fit": fits,
    }
    save_report(OUT, 'phase2c_identifiability_audit.json', report, __file__, INPUT_INVENTORY)

    print("拟合数:", len(fits))
    print("\nD1 每类 presence 计数（最小/中位/最大，是否可估）:")
    for c in CODES:
        e = d1[c]
        print(f"  code {c}: {e['min']:3d} / {e['median']:3d} / {e['max']:3d}  "
              f"{'可估' if e['estimable'] else '不可估'}")
    print("\nD3 背景方案间秩相关:")
    for s, per in d3.items():
        print(f"  {s:24s} " + "  ".join(f"{k}:{v['rank_corr']}" for k, v in per.items()))
    print("\nD4 季节间秩相关（同方案）:")
    for k, v in d4.items():
        print(f"  {k:44s} {v}")
    print("\nD6 威胁可分性: VIF =", json.dumps(vif), "| max =", max_threat_vif)
    print("\nHeuristic screening only, NOT H_j identifiability:", identifiable)
    for r in report["verdict"]["reasons"]:
        print("   -", r)


if __name__ == "__main__":
    main()
