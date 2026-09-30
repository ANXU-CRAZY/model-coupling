"""Fail-closed declared provenance checks for an outer gate train/validation/test split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fusion import unit_interval

SEASONS = ("spring", "summer", "autumn", "winter")
FEATURES = ("m", "h", "u_m", "u_h", "season_sin", "season_cos")
TARGETS = ("target_protection", "target_restoration")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(path):
    # Read identifiers as strings; preserve leading zeros in real observation IDs.
    return pd.read_csv(path, dtype={k: str for k in (
        "sample_id", "site_id", "group_id", "species_id", "m_artifact_id", "h_artifact_id")})


def validate_table(df, training=False):
    required = {"sample_id", "site_id", "group_id", "year", "season", "m", "h",
                "u_m", "u_h", "feasible", "habitat_suitability"}
    if training:
        required |= {"role", "m_artifact_id", "h_artifact_id", *TARGETS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    for k in ("sample_id", "site_id", "group_id", "year", "season"):
        if df[k].isna().any() or (df[k].astype(str).str.strip() == "").any():
            raise ValueError(f"Missing identifier: {k}")
    if df.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique")
    unit_keys = ["site_id", "year", "season"] + (["species_id"] if "species_id" in df else [])
    if df.duplicated(unit_keys).any():
        raise ValueError("Duplicate site/year/season/species units: aggregate observations first")
    if not df.season.isin(SEASONS).all():
        raise ValueError(f"season must be one of {SEASONS}")
    for k in ("m", "h", "feasible", "habitat_suitability"):
        unit_interval(df[k].to_numpy(), k, allow_nan=False)
    if not df.feasible.isin([0, 1]).all():
        raise ValueError("feasible must be a documented binary mask")
    for k in ("u_m", "u_h"):
        a = df[k].to_numpy(float)
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"{k}: expected nonnegative, finite ensemble SD")
    if (df.h > df.habitat_suitability + 1e-6).any():
        raise ValueError("HQ exceeds H_j; confirm parameter member and LULC lookup match")
    if training:
        if not df.role.isin(["train", "validation", "test"]).all():
            raise ValueError("role must be train/validation/test")
        for k in TARGETS:
            unit_interval(df[k].to_numpy(), k)


def build_features(df):
    validate_table(df)
    idx = df.season.map({s: i for i, s in enumerate(SEASONS)}).to_numpy(float)
    angle = 2 * np.pi * idx / 4
    return np.column_stack([df[k].to_numpy(float) for k in ("m", "h", "u_m", "u_h")]
                            + [np.sin(angle), np.cos(angle)]).astype(np.float32)


def audit_provenance(df, manifest, demo=False):
    """Check declared fit/tune/calibration footprints, including gate validation isolation.

    This checks declarations and matching hashes, not the truth of external logs.
    Producers must attach actual fitting logs to evidence_uri for substantive audit.
    """
    validate_table(df, training=True)
    if manifest.get("data_kind") == "synthetic" and not demo:
        raise ValueError("Synthetic provenance is accepted only through --demo")
    if manifest.get("test_scope") not in ("spatial", "temporal", "spatiotemporal"):
        raise ValueError("Declare test_scope")
    train, val, test = [df[df.role == k] for k in ("train", "validation", "test")]
    if train.empty or val.empty or test.empty:
        raise ValueError("A declared train/validation/test split is required")
    for a, b in ((train, val),):
        if set(a.group_id) & set(b.group_id) or set(a.site_id) & set(b.site_id):
            raise ValueError("Gate train/validation sites and groups must be disjoint")
    if manifest["test_scope"] in ("spatial", "spatiotemporal"):
        if set(test.group_id) & set(df[df.role != "test"].group_id):
            raise ValueError("Spatial final test groups overlap development")
        if set(test.site_id) & set(df[df.role != "test"].site_id):
            raise ValueError("Spatial final test sites overlap development")
    if manifest["test_scope"] in ("temporal", "spatiotemporal"):
        if test.year.min() <= df[df.role != "test"].year.max():
            raise ValueError("Temporal test years must follow development years")
    artifacts = manifest.get("artifacts", {})
    maxent_transforms = {artifacts[key].get("output_type") for key in set(df.m_artifact_id) if key in artifacts}
    if len(maxent_transforms)>1:
        raise ValueError("Mixed MaxEnt output transforms: use a consistent, explicitly validated scale")
    protected_ids = set(df.loc[df.role != "train", "sample_id"])
    protected_groups = set(val.group_id)
    if manifest["test_scope"] in ("spatial", "spatiotemporal"):
        protected_groups |= set(test.group_id)
    for row in df.itertuples():
        for key, model_type in ((row.m_artifact_id, "maxent"), (row.h_artifact_id, "invest")):
            if key not in artifacts:
                raise ValueError(f"Undeclared base artifact: {key}")
            a = artifacts[key]
            if a.get("model_type") != model_type or not a.get("evidence_uri") or not a.get("software"):
                raise ValueError(f"Incomplete artifact provenance: {key}")
            if model_type == "maxent" and a.get("output_type") not in ("cloglog", "logistic"):
                raise ValueError("Record MaxEnt output transform; raw/logistic/cloglog cannot be mixed")
            if model_type == "invest" and a.get("output_type") != "HQ":
                raise ValueError("InVEST input must declare HQ; raw degradation is a different quantity")
            used_ids, used_groups = set(), set()
            for stage in ("fit", "tune", "calibrate"):
                ids, groups = a.get(f"{stage}_sample_ids"), a.get(f"{stage}_group_ids")
                if not isinstance(ids, list) or not isinstance(groups, list):
                    raise ValueError(f"Missing {stage} footprint for {key}")
                used_ids.update(ids)
                used_groups.update(groups)
                known = df[df.sample_id.isin(ids)]
                if not set(known.group_id).issubset(set(groups)):
                    raise ValueError(f"Inconsistent ID/group footprint for {key}")
            if used_ids & protected_ids or used_groups & protected_groups:
                raise ValueError(f"Validation/test labels leaked into base fitting/tuning: {key}")
            if row.sample_id in used_ids:
                raise ValueError(f"In-sample base prediction used at {row.sample_id}")
            if row.role == "train" and row.group_id in used_groups:
                raise ValueError(f"Gate training needs group-out-of-fold base predictions: {row.sample_id}")
    for task, target in zip(("protection", "restoration"), TARGETS):
        if df.loc[df.role != "test", target].notna().any():
            source = manifest.get("supervision", {}).get(task, {})
            allowed = ("independent_ecological_measurement", "independent_management_assessment",
                       "measured_restoration_response", "synthetic")
            if source.get("source_kind") not in allowed or not source.get("evidence_uri"):
                raise ValueError(f"Document independent {task} supervision")
            if source.get("source_kind") == "synthetic" and not demo:
                raise ValueError("Synthetic supervision is demonstration only")
            if not source.get("definition") or not source.get("normalization"):
                raise ValueError(f"Document target meaning and scale: {task}")
            if source.get("derived_from_m_or_h") is not False:
                raise ValueError("Do not use MaxEnt/HQ/fusion-derived pseudo-labels to prove adaptive fusion")
    return {"status": "DECLARED_PROVENANCE_PASS", "rows": len(df),
            "roles": df.role.value_counts().to_dict(), "test_scope": manifest["test_scope"],
            "caveat": "Declarations require external fitting-log verification; this is not ecological validation."}


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
