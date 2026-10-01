"""Frozen MaxEnt selection, presence-background metrics and final-test guards."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_predictions(values):
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Expected finite official cloglog scores in [0,1]; these are not calibrated probabilities")
    return result


def background_auc(presence, background):
    """Mann-Whitney ROC AUC against available background, not true absences."""
    p, b = validate_predictions(presence), validate_predictions(background)
    if not len(p) or not len(b):
        raise ValueError("AUC requires held-out presences and background")
    ranks = rankdata(np.concatenate([p, b]), method="average")
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(b)))


def metrics(train_p, train_b, val_p, val_b, complexity):
    tp, vp = validate_predictions(train_p), validate_predictions(val_p)
    if not len(tp) or not len(vp):
        raise ValueError("Empty fit or validation presence prediction")
    train_auc, val_auc = background_auc(tp, train_b), background_auc(vp, val_b)
    threshold = float(np.quantile(tp, 0.1))
    return {"train_auc": train_auc, "validation_auc": val_auc,
            "auc_gap": train_auc - val_auc, "omission_10": float(np.mean(vp < threshold)),
            "omission_min": float(np.mean(vp < tp.min())), "threshold_10": threshold,
            "threshold_min": float(tp.min()), "prediction_min": float(vp.min()),
            "prediction_max": float(vp.max()), "complexity": int(complexity),
            "validation_presence_n": len(vp), "validation_background_n": len(val_b),
            "background_is_absence": False, "cbi": None}


def select_predictors(frame, names, cutoff=0.7):
    """Fit-background-only filtering; order has no label/model-based priority."""
    selected, dropped = [], {}
    for name in sorted(names):
        if name not in frame or not np.isfinite(frame[name].to_numpy(float)).all():
            raise ValueError("Missing/invalid fit predictor: " + name)
        values = frame[name].to_numpy(float)
        if np.ptp(values) <= 1e-12:
            dropped[name] = "constant_on_fit_background"
            continue
        correlated = None
        for previous in selected:
            rho = float(spearmanr(values, frame[previous].to_numpy(float)).statistic)
            if not np.isfinite(rho):
                raise ValueError("Undefined fit-background correlation")
            if abs(rho) >= cutoff:
                correlated = previous
                break
        if correlated is None:
            selected.append(name)
        else:
            dropped[name] = "correlated_with_" + correlated
    if not selected:
        raise ValueError("No variable predictors remain in fit scope")
    return selected, dropped


def aggregate_candidates(rows):
    frame = pd.DataFrame(rows)
    keys = ["variant", "background", "rm", "fc"]
    if frame.empty or not set(keys + ["inner_fold", "validation_auc", "omission_10", "complexity", "auc_gap"]).issubset(frame):
        raise ValueError("Incomplete inner candidate metrics")
    outputs = []
    for group, f in frame.groupby(keys, sort=True):
        if f.inner_fold.duplicated().any() or len(f) < 2:
            raise ValueError("Candidate requires distinct spatial inner folds")
        outputs.append(dict(zip(keys, group), mean_auc=float(f.validation_auc.mean()),
                            auc_sd=float(f.validation_auc.std(ddof=1)),
                            auc_se=float(f.validation_auc.std(ddof=1)/np.sqrt(len(f))),
                            mean_omission=float(f.omission_10.mean()),
                            mean_complexity=float(f.complexity.mean()),
                            mean_abs_auc_gap=float(f.auc_gap.abs().mean()), inner_folds=len(f)))
    return outputs


def choose_candidate(candidates, omission_limit=0.2):
    if not candidates:
        raise ValueError("No tuning candidates")
    eligible = [c for c in candidates if c["mean_omission"] <= omission_limit]
    failed = not bool(eligible)
    if failed:
        lowest = min(c["mean_omission"] for c in candidates)
        eligible = [c for c in candidates if c["mean_omission"] <= lowest + 1e-12]
    best = max(eligible, key=lambda c: (c["mean_auc"], -c["auc_se"]))
    band = [c for c in eligible if c["mean_auc"] >= best["mean_auc"] - best["auc_se"] - 1e-12]
    fc_order = {"L": 0, "LQ": 1, "LQH": 2}
    chosen = min(band, key=lambda c: (c["mean_complexity"], c["mean_abs_auc_gap"], -c["rm"], fc_order[c["fc"]], c["background"], c["variant"]))
    return {**chosen, "omission_constraint_failed": failed, "selection_auc_band_lower": best["mean_auc"]-best["auc_se"]}


def assert_oof_groups(predicted_groups, fit_groups, tune_groups, locked_groups):
    predicted, used = set(predicted_groups), set(fit_groups) | set(tune_groups)
    if predicted & used:
        raise ValueError("OOF prediction group participated in fit/tune")
    if used & set(locked_groups):
        raise ValueError("Locked test group participated in model selection or fit")


def freeze_selection(path, selection, split_hash, config_hash):
    p = Path(path)
    if p.exists():
        raise FileExistsError("Frozen selection already exists")
    write_json(p, {"status": "FROZEN_BEFORE_LOCKED_TEST", "selection": selection,
                   "split_sha256": split_hash, "config_sha256": config_hash})
    return sha256(p)


def claim_locked_test(directory, frozen_selection_path, selection_hash, split_hash, config_hash):
    """Exclusive attempt marker is created before any final-test prediction."""
    p = Path(frozen_selection_path)
    value = json.loads(p.read_text(encoding="utf-8"))
    if sha256(p) != selection_hash or value["status"] != "FROZEN_BEFORE_LOCKED_TEST":
        raise ValueError("Frozen selection hash/status changed")
    if value["split_sha256"] != split_hash or value["config_sha256"] != config_hash:
        raise ValueError("Split/config changed after selection freeze")
    marker = Path(directory) / "LOCKED_TEST_ATTEMPT.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"selection_sha256": selection_hash, "split_sha256": split_hash,
                   "config_sha256": config_hash, "status": "CLAIMED_DO_NOT_REPEAT"}, stream)
    return marker
