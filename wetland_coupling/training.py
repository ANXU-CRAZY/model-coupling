"""Masked independent supervision; validation chooses epoch, final test is read once."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
import torch

from .audit import (FEATURES, TARGETS, audit_provenance, build_features, file_sha256,
                    load_table, save_json, validate_table)
from .fusion import fuse, geometric
from .model import DualGate


def metrics(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = np.asarray(y)[mask], np.asarray(p)[mask]
    if not len(y):
        return {"n": 0, "rmse": None, "mae": None, "spearman": None}
    rho = None
    if len(y) > 2 and np.ptp(y) > 0 and np.ptp(p) > 0:
        rho = float(spearmanr(y, p).statistic)
    return {"n": len(y), "rmse": float(np.sqrt(np.mean((p-y)**2))),
            "mae": float(np.mean(np.abs(p-y))), "spearman": rho}


def masked_loss(scores, targets):
    losses = []
    for task in range(2):
        mask = torch.isfinite(targets[:, task])
        if mask.any():
            losses.append(((scores[mask, task] - targets[mask, task]) ** 2).mean())
    if not losses:
        raise ValueError("No valid supervision in this split")
    return torch.stack(losses).mean()


def tensors(df, mean, scale):
    x = (build_features(df) - mean) / scale
    eligible = (df.feasible.to_numpy() == 1) & (df.habitat_suitability.to_numpy() > 0)
    return (torch.tensor(x, dtype=torch.float32), torch.tensor(df.m.to_numpy(), dtype=torch.float32),
            torch.tensor(df.h.to_numpy(), dtype=torch.float32), torch.tensor(eligible))


def predict_with_model(model, df, mean, scale):
    model.eval()
    with torch.no_grad():
        scores, weights = model(*tensors(df, mean, scale))
    return scores.numpy(), weights.numpy()


def fit_constant(df, method, active, min_weight=0.1):
    chosen = []
    for t, key in enumerate(TARGETS):
        mask = df[key].notna().to_numpy()
        if not active[t] or not mask.any():
            chosen.append(0.5)
            continue
        m, h, y = df.m.to_numpy()[mask], df.h.to_numpy()[mask], df[key].to_numpy()[mask]
        b = h if t == 0 else 1-h
        if method == "geometric":
            objective = lambda w: np.mean((geometric(m, b, w)-y)**2)
        else:
            objective = lambda w: np.mean((w*m+(1-w)*b-y)**2)
        opt = minimize_scalar(objective, bounds=(min_weight, 1-min_weight), method="bounded")
        chosen.append(float(opt.x))
    return chosen


def baseline_predictions(df, constant_geo=(0.5, 0.5), constant_linear=(0.5, 0.5)):
    args = {"feasible": df.feasible.to_numpy(), "habitat_suitability": df.habitat_suitability.to_numpy()}
    result = {}
    eligible = (df.feasible.to_numpy()==1) & (df.habitat_suitability.to_numpy()>0)
    # Single inputs are component controls, not complete management objectives.
    result["M_only_component"] = np.column_stack([df.m, np.where(eligible, df.m, 0.)])
    result["HQ_only_component"] = np.column_stack([df.h, np.where(eligible, 1-df.h, 0.)])
    relative = fuse(df.m,df.h,feasible=args["feasible"],habitat_suitability=args["habitat_suitability"],
                    deficit_mode="relative_degradation")
    result["relative_degradation_geometric_050"] = np.column_stack([relative["conservation"],relative["restoration_candidate"]])
    for name, method, weights in (("fixed_geometric_050", "geometric", (0.5, 0.5)),
                                  ("fixed_linear_050", "linear", (0.5, 0.5)),
                                  ("fitted_constant_geometric", "geometric", constant_geo),
                                  ("fitted_constant_linear", "linear", constant_linear),
                                  ("minimum", "minimum", (0.5, 0.5))):
        r = fuse(df.m, df.h, *weights, method=method, **args)
        result[name] = np.column_stack([r["conservation"], r["restoration_candidate"]])
    return result


def envelope_audit(df, min_weight):
    m, h = df.m.to_numpy(), df.h.to_numpy()
    result = {}
    for t, key in enumerate(TARGETS):
        b = h if t == 0 else 1-h
        p1, p2 = geometric(m, b, min_weight), geometric(m, b, 1-min_weight)
        low, high = np.minimum(p1,p2), np.maximum(p1,p2)
        if t == 1:
            eligible = (df.feasible.to_numpy()==1) & (df.habitat_suitability.to_numpy()>0)
            low, high = np.where(eligible, low, 0.), np.where(eligible, high, 0.)
        y = df[key].to_numpy(float)
        valid = np.isfinite(y)
        residual = np.maximum(low-y, 0.) + np.maximum(y-high, 0.)
        result[key] = {"n": int(valid.sum()),
                       "outside_feasible_score_envelope_fraction": float(np.mean(residual[valid]>1e-6)) if valid.any() else None,
                       "minimum_possible_rmse": float(np.sqrt(np.mean(residual[valid]**2))) if valid.any() else None}
    return result


def train(csv_path, provenance_path, output, config=None, demo=False):
    config = dict(config or {})
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    df = load_table(csv_path)
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    audit = audit_provenance(df, provenance, demo=demo)
    if provenance.get("table_sha256") != file_sha256(csv_path):
        raise ValueError("Provenance table_sha256 does not match input CSV")
    train_df, val_df = df[df.role=="train"], df[df.role=="validation"]
    targets = df[list(TARGETS)].to_numpy(float)
    # Missing repair labels disable that head; they never become zeros or bird absences.
    active = []
    for key in TARGETS:
        ntrain, nval = train_df[key].notna().sum(), val_df[key].notna().sum()
        if ntrain == 0 and nval == 0:
            active.append(False)
        elif ntrain < config.get("min_train_labels", 20) or nval < config.get("min_val_labels", 5):
            raise ValueError(f"Insufficient {key} labels to train/validate; retain the fixed baseline")
        else:
            active.append(True)
    if not any(active):
        raise ValueError("No independent labels: use baseline, not train")
    ineligible = (df.feasible != 1) | (df.habitat_suitability <= 0)
    if (df.loc[ineligible, "target_restoration"].fillna(0)>0).any():
        raise ValueError("Positive restoration labels outside current eligible habitat need a separate scenario model")
    output.mkdir(parents=True, exist_ok=False)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x_train = build_features(train_df)
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale < 1e-6, 1., scale)
    model_config = {"n_features": len(FEATURES), "width": int(config.get("width", 32)),
                    "dropout": float(config.get("dropout", 0.1)),
                    "min_weight": float(config.get("min_weight", 0.1)), "active_heads": active}
    model = DualGate(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 0.003),
                                  weight_decay=config.get("weight_decay", 0.001))
    tr = tensors(train_df, mean, scale)
    va = tensors(val_df, mean, scale)
    ytr = torch.tensor(train_df[list(TARGETS)].to_numpy(), dtype=torch.float32)
    yva = torch.tensor(val_df[list(TARGETS)].to_numpy(), dtype=torch.float32)
    best, best_epoch, best_state, stale, history = float("inf"), -1, None, 0, []
    for epoch in range(int(config.get("epochs", 300))):
        model.train()
        optimizer.zero_grad()
        pred, weights = model(*tr)
        data_loss = masked_loss(pred, ytr)
        active_tensor = torch.tensor(active, dtype=torch.bool)
        regularizer = ((weights[:, active_tensor] - 0.5)**2).mean()
        loss = data_loss + config.get("gate_prior_strength", 0.001) * regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            vp, _ = model(*va)
            val_loss = float(masked_loss(vp, yva))
        history.append({"epoch": epoch+1, "train_mse": float(data_loss.detach()), "validation_mse": val_loss})
        if val_loss < best - 1e-7:
            best, best_epoch = val_loss, epoch+1
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= config.get("patience", 40):
            break
    if best_state is None:
        raise RuntimeError("No finite best model")
    model.load_state_dict(best_state)
    fitted_geo = fit_constant(train_df,"geometric",active,model_config["min_weight"])
    fitted_linear = fit_constant(train_df,"linear",active,model_config["min_weight"])
    # Final test targets are evaluated only after every choice above is frozen.
    pred, weights = predict_with_model(model, df, mean, scale)
    controls = baseline_predictions(df, fitted_geo, fitted_linear)
    controls["dual_gate"] = pred
    evaluation = {}
    for role in ("train", "validation", "test"):
        mask = (df.role == role).to_numpy()
        evaluation[role] = {name: {key: metrics(targets[mask,t], values[mask,t])
                                  for t,key in enumerate(TARGETS)} for name,values in controls.items()}
    public = df[["sample_id", "site_id", "group_id", "year", "season", "role"]].copy()
    public["conservation_score"], public["restoration_candidate_score"] = pred[:,0], pred[:,1]
    for column, vals in zip(("alpha", "beta", "gamma", "delta"),
                             (weights[:,0,0], weights[:,0,1], weights[:,1,0], weights[:,1,1])):
        public[column] = vals
    public.to_csv(output/"predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(history).to_csv(output/"training_history.csv", index=False)
    checkpoint = {"state_dict": best_state, "model_config": model_config,
                  "feature_names": list(FEATURES), "mean": mean.tolist(), "scale": scale.tolist(),
                  "data_kind": provenance.get("data_kind", "real"), "table_sha256": file_sha256(csv_path)}
    torch.save(checkpoint, output/"gate.pt")
    report = {"status": "SYNTHETIC_DEMO_ONLY" if demo else "EXPERIMENTAL_GATE_FITTED",
              "best_epoch": best_epoch, "epochs_run": len(history), "active_heads": active,
              "independent_ecological_validation": False, "audit": audit, "metrics": evaluation,
              "fitted_constant_geometric": fitted_geo, "fitted_constant_linear": fitted_linear,
              "score_envelope": envelope_audit(df, model_config["min_weight"]),
              "config": config, "model_config": model_config,
              "input_sha256": file_sha256(csv_path), "provenance_sha256": file_sha256(provenance_path),
              "gate_interpretation": "Statistical allocation conditional on input scaling and calibration; not causal ecology.",
              "uncertainty": "Input ensemble SD features only; no predictive interval is claimed."}
    save_json(output/"report.json", report)
    return report


def load_gate(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["feature_names"] != list(FEATURES):
        raise ValueError("Feature schema mismatch")
    model = DualGate(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["scale"]), checkpoint


def propagate_gate_paired(checkpoint_path, df, m_draws, h_draws, habitat_draws=None, demo=False):
    """Recompute the frozen gate for every supplied paired base-model member.

    Conditional on one gate checkpoint: excludes gate-training uncertainty and
    does not assert that ensemble percentiles have frequentist coverage.
    """
    from .fusion import propagate_paired, unit_interval
    model,mean,scale,meta = load_gate(checkpoint_path)
    if meta["data_kind"] == "synthetic" and not demo:
        raise ValueError("Synthetic gate requires demo=True")
    m,h = unit_interval(m_draws),unit_interval(h_draws)
    if m.shape != h.shape or m.ndim!=2 or m.shape[1]!=len(df) or m.shape[0]<2:
        raise ValueError("Supply matched [members,table_rows] arrays")
    hj = np.broadcast_to(df.habitat_suitability.to_numpy() if habitat_draws is None else habitat_draws, m.shape)
    weights = []
    for i in range(m.shape[0]):
        member = df.copy()
        member["m"],member["h"],member["habitat_suitability"] = m[i],h[i],hj[i]
        _,w = predict_with_model(model,member,mean,scale)
        weights.append(w)
    w = np.asarray(weights)
    result = propagate_paired(m,h,df.feasible.to_numpy(),hj,alpha=w[:,:,0,0],gamma=w[:,:,1,0])
    result["uncertainty_scope"] = "Paired base-input uncertainty conditional on one frozen gate"
    return result
