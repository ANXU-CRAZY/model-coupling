"""Evidence-constrained InVEST run plans; uses official InVEST rather than a proxy."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .audit import file_sha256, save_json
from .fusion import unit_interval


def validate_prior(spec):
    if spec.get("status") != "evidence_reviewed":
        raise ValueError("Real prior is not approved by the project evidence review; fill evidence before sampling")
    if spec.get("target_invest_version") != "3.16.1":
        raise ValueError("This adapter is source-checked for InVEST 3.16.1; other versions require an adapter review")
    if spec.get("distance_unit") != "m":
        raise ValueError("Use metres. InVEST >=3.15 changed max_dist from km to m")
    parameters = spec.get("parameters", [])
    if not parameters:
        raise ValueError("No prior parameters")
    keys = []
    for p in parameters:
        key, lo, hi = p["key"], p["low"], p["high"]
        if not p.get("evidence") or lo is None or hi is None:
            raise ValueError(f"Fill a defensible bound and evidence for {key}")
        if not np.isfinite([lo,hi]).all() or lo > hi:
            raise ValueError(f"Invalid bounds: {key}")
        if key.startswith(("weight:", "sensitivity:", "habitat:")):
            unit_interval([lo,hi], key, allow_nan=False)
        elif key.startswith("distance:") or key == "half_saturation_constant":
            if lo <= 0:
                raise ValueError(f"Positive parameter required: {key}")
        else:
            raise ValueError(f"Unsupported parameter key: {key}")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate parameter keys")
    return parameters


def sample_prior(spec, members=32, seed=42):
    params = validate_prior(spec)
    if members < 2:
        raise ValueError("At least two members are needed for uncertainty exploration")
    uniform = qmc.LatinHypercube(d=len(params), seed=seed).random(members)
    lo, hi = np.asarray([p["low"] for p in params]), np.asarray([p["high"] for p in params])
    values = lo + uniform*(hi-lo)
    df = pd.DataFrame(values, columns=[p["key"] for p in params])
    df.insert(0, "member_id", [f"invest_{i:04d}" for i in range(members)])
    return df


def make_run_plan(spec_path, threats_path, sensitivity_path, lulc_path, output, members=32, seed=42):
    """Materialize each member's CSVs and args. No run or calibration is implied."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    samples = sample_prior(spec, members, seed)
    threats = pd.read_csv(threats_path)
    sensitivity = pd.read_csv(sensitivity_path)
    if not {"threat", "max_dist", "weight", "decay", "cur_path"}.issubset(threats.columns):
        raise ValueError("Incomplete threat table")
    if not {"lucode", "habitat"}.issubset(sensitivity.columns):
        raise ValueError("Sensitivity table must use lucode and habitat")
    if threats.threat.duplicated().any() or sensitivity.lucode.duplicated().any():
        raise ValueError("Duplicate threat/LULC codes")
    for p in spec["parameters"]:
        parts = p["key"].split(":")
        if parts[0] in ("weight", "distance") and parts[1] not in set(threats.threat):
            raise ValueError(f"Unknown threat: {p['key']}")
        if parts[0] == "habitat" and int(parts[1]) not in set(sensitivity.lucode):
            raise ValueError(f"Unknown LULC: {p['key']}")
        if parts[0] == "sensitivity" and (int(parts[1]) not in set(sensitivity.lucode) or parts[2] not in sensitivity.columns):
            raise ValueError(f"Unknown sensitivity: {p['key']}")
    # Keep absolute threat paths when writing tables into member-specific directories.
    for col in ("cur_path", "base_path", "fut_path"):
        if col in threats:
            threats[col] = threats[col].map(lambda p: str((Path(threats_path).resolve().parent / str(p)).resolve())
                                          if pd.notna(p) and str(p).strip() else "")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for row in samples.to_dict("records"):
        member = output/row.pop("member_id")
        member.mkdir()
        tt, ss = threats.copy(), sensitivity.copy()
        half = spec.get("fixed_half_saturation_constant")
        for key, value in row.items():
            parts = key.split(":")
            if parts[0] == "weight":
                tt.loc[tt.threat==parts[1], "weight"] = value
            elif parts[0] == "distance":
                tt.loc[tt.threat==parts[1], "max_dist"] = value
            elif parts[0] == "habitat":
                ss.loc[ss.lucode==int(parts[1]), "habitat"] = value
            elif parts[0] == "sensitivity":
                ss.loc[ss.lucode==int(parts[1]), parts[2]] = value
            elif key == "half_saturation_constant":
                half = value
        unit_interval(tt.weight, "threat weight", allow_nan=False)
        if tt.weight.sum() <= 0 or not np.isfinite(tt.max_dist).all() or (tt.max_dist<=0).any():
            raise ValueError("Invalid threat weights/distances")
        if not tt.decay.isin(["linear", "exponential"]).all():
            raise ValueError("Invalid threat decay")
        unit_interval(ss.habitat, "H_j", allow_nan=False)
        for threat in tt.threat:
            if threat not in ss:
                raise ValueError(f"Missing sensitivity column: {threat}")
            unit_interval(ss[threat], f"sensitivity:{threat}", allow_nan=False)
        if half is None or not np.isfinite(half) or half<=0:
            raise ValueError("Specify a positive half saturation constant with evidence")
        tt.to_csv(member/"threats.csv", index=False)
        ss.to_csv(member/"sensitivity.csv", index=False)
        args = {"workspace_dir": str((member/"official_output").resolve()), "results_suffix": "",
                "lulc_cur_path": str(Path(lulc_path).resolve()),
                "threats_table_path": str((member/"threats.csv").resolve()),
                "sensitivity_table_path": str((member/"sensitivity.csv").resolve()),
                "half_saturation_constant": float(half), "n_workers": -1}
        save_json(member/"args.json", args)
        entries.append({"member_id": member.name, "args_path": str((member/"args.json").resolve())})
    samples.to_csv(output/"parameter_samples.csv", index=False)
    manifest = {"status": "PRIOR_RUN_PLAN_ONLY", "target_invest_version": "3.16.1", "distance_unit": "m",
                "members": entries, "seed": seed,
                "prior_sha256": file_sha256(spec_path),
                "uncertainty_type": "Uncalibrated prior exploration, not a Bayesian posterior",
                "calibration": "NOT_RUN; reserve calibration/tuning data within base training groups"}
    save_json(output/"plan.json", manifest)
    return manifest


def run_official_invest(args_path):
    version = importlib.metadata.version("natcap.invest")
    if version != "3.16.1":
        raise ValueError(f"Adapter pinned to InVEST 3.16.1; installed version is {version}")
    module = importlib.import_module("natcap.invest.habitat_quality")
    args = json.loads(Path(args_path).read_text(encoding="utf-8"))
    workspace = Path(args["workspace_dir"])
    if workspace.exists():
        raise FileExistsError(f"Official workspace already exists: {workspace}")
    warnings = module.validate(args)
    if warnings:
        raise ValueError(f"InVEST validation messages: {warnings}")
    module.execute(args)
    save_json(workspace/"coupling_adapter_manifest.json", {
        "status": "OFFICIAL_INVEST_COMPLETED", "version": version,
        "args_sha256": file_sha256(args_path), "outputs": sorted(p.name for p in workspace.glob("*.tif"))})


def rank_calibration_candidates(csv_path, output):
    """Rank externally cross-validated parameter candidates; never label them posterior samples."""
    df = pd.read_csv(csv_path)
    required = {"member_id", "fold", "score", "metric", "higher_is_better", "split_role"}
    if not required.issubset(df):
        raise ValueError(f"Required calibration fields: {sorted(required)}")
    if not df.split_role.isin(["inner_calibration_validation"]).all():
        raise ValueError("Only base inner calibration validation is allowed; never outer/test scores")
    if df.duplicated(["member_id", "fold"]).any() or df.metric.nunique()!=1 or df.higher_is_better.nunique()!=1:
        raise ValueError("Inconsistent or duplicate calibration measurements")
    if not np.isfinite(df.score).all():
        raise ValueError("Non-finite calibration scores")
    folds = df.groupby("member_id").fold.agg(set)
    if len(folds)<2 or any(x != folds.iloc[0] for x in folds) or len(folds.iloc[0])<3:
        raise ValueError("Require >=2 candidates evaluated on the same >=3 spatial calibration folds")
    direction = df.higher_is_better.iloc[0]
    if direction not in (0,1):
        raise ValueError("higher_is_better must be 0/1")
    summary = df.groupby("member_id").score.agg(["mean", "std", "count"])
    summary = summary.sort_values("mean", ascending=not bool(direction))
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary.to_csv(output)
    return {"status": "CALIBRATION_CANDIDATE_RANKING_ONLY", "metric": df.metric.iloc[0],
            "best_member": summary.index[0], "candidate_count": len(summary),
            "caveat": "No evidence of posterior probabilities or independent final ecological validity."}
