"""Numerical diagnostics for exploratory presence/background contrasts.

These functions do not estimate approved InVEST habitat values or causality.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr


def rank_correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite([a, b]).all():
        raise ValueError("Require aligned finite rank diagnostic vectors")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def ols_vif(matrix):
    """VIF from each column's OLS projection on ALL other columns + intercept."""
    matrix = np.asarray(matrix, float)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or not np.isfinite(matrix).all():
        raise ValueError("Require a finite diagnostic design matrix")
    result = []
    for index in range(matrix.shape[1]):
        y = matrix[:, index]
        total = float(np.sum((y - y.mean()) ** 2))
        if total == 0:
            result.append(None)
            continue
        other = np.column_stack([np.ones(len(y)), np.delete(matrix, index, axis=1)])
        fitted = other @ np.linalg.lstsq(other, y, rcond=None)[0]
        residual = float(np.sum((y - fitted) ** 2))
        # Singular/perfect dependence is explicit, never a fabricated finite VIF.
        result.append(None if residual <= total * 1.e-12 else max(1., total / residual))
    return result


def logistic_objective(beta, matrix, y, weights, ridge):
    eta = matrix @ beta
    penalty = beta.copy()
    penalty[0] = 0.
    value = float(np.sum(weights * (np.logaddexp(0., eta) - y * eta))
                  + .5 * ridge * (penalty @ penalty))
    gradient = matrix.T @ (weights * (expit(eta) - y)) + ridge * penalty
    return value, gradient


def fit_logistic(matrix, y, weights=None, ridge=1.e-4, max_iter=300, tol=1.e-9):
    """Checked penalised logistic contrast; background class is not absence.

    Weights multiply the likelihood/score. They must not divide the working
    residual a second time, which would cancel them in weighted IRLS.
    """
    matrix, y = np.asarray(matrix, float), np.asarray(y, float)
    if matrix.ndim != 2 or y.shape != (len(matrix),) or not np.isfinite(matrix).all():
        raise ValueError("Invalid logistic design/response")
    if not np.array_equal(np.unique(y), [0., 1.]) or not np.all(matrix[:, 0] == 1):
        raise ValueError("Require intercept and both sample-membership classes")
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid likelihood weights")
    if any(weights[y == cls].sum() <= 0 for cls in (0., 1.)) or ridge < 0:
        raise ValueError("Both membership classes need positive total weight")
    weights = weights / weights.mean()
    initial = np.zeros(matrix.shape[1])
    fraction = float(np.average(y, weights=weights))
    initial[0] = np.log(fraction / (1. - fraction))
    result = minimize(logistic_objective, initial, args=(matrix, y, weights, ridge),
                      jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": max(tol, 1.e-7), "ftol": 1.e-13})
    _, gradient = logistic_objective(result.x, matrix, y, weights, ridge)
    if not result.success or not np.isfinite(result.x).all() or np.max(np.abs(gradient)) > 1.e-4:
        raise ValueError("Exploratory logistic fit did not establish numerical convergence: " + str(result.message))
    return result.x


def record_input(path, inventory):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    inventory[str(path)] = {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def save_report(out, name, report, script, inventory):
    """New output only; preserve original WorkBuddy calculations as history."""
    out, script = Path(out), Path(script).resolve()
    target = out / name
    if target.exists():
        raise FileExistsError(target)
    report["audit_status"] = "EXPLORATORY_DIAGNOSTIC_NOT_HQ_PARAMETER_CALIBRATION"
    report["eligible_for_official_run"] = False
    report["statistical_identifiability_established"] = False
    report["background_membership_is_true_absence"] = False
    report["legacy_hj_fields_are_approved_invest_parameters"] = False
    report["lambda_coefficients_are_variable_contributions"] = False
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    sources = {}
    record_input(script, sources)
    record_input(__file__, sources)
    root = script.parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    snapshot = out / "command.snapshot.json"
    snapshot.write_text(json.dumps({"argv": sys.argv, "ecological_parameters_approved": False}, indent=2), encoding="utf-8")
    outputs = {}
    record_input(target, outputs)
    record_input(snapshot, outputs)
    manifest = {"status": report["audit_status"], "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit_sha": commit, "python": sys.version, "platform": platform.platform(),
                "random_seed": None, "inputs": inventory, "source_code": sources, "outputs": outputs,
                "official_invest_executed": False, "eligible_for_official_run": False,
                "independent_supervision_created": 0, "argv": sys.argv}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
