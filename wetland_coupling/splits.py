"""Plan spatial group cross-fitting before base-model tuning or gate fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def spatial_groups(x_m, y_m, block_size_m, origin=(0.,0.)):
    """Caller supplies metre coordinates and a block size justified by autocorrelation.

    Pixel size (e.g. 100 m) is NOT an appropriate automatic block-size choice.
    This generator groups cells but does not by itself remove boundary correlation.
    """
    x,y = np.asarray(x_m,float),np.asarray(y_m,float)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or block_size_m<=0:
        raise ValueError("Supply finite metre coordinates and a positive block size")
    gx = np.floor((x-origin[0])/block_size_m).astype(np.int64)
    gy = np.floor((y-origin[1])/block_size_m).astype(np.int64)
    return np.asarray([f"b{a}_{b}" for a,b in zip(gx.flat,gy.flat)]).reshape(x.shape)


def base_crossfit_plan(df, n_inner_folds=3, seed=42):
    """Gate validation/final test are excluded from ALL base fit/tune/calibration.

    Every gate-train group gets a separate base fit with that group withheld.
    RM/FC and InVEST calibration are performed using inner folds inside that fit.
    Group buffers can be applied by a source-data producer before executing this plan.
    """
    required = {"sample_id","group_id","role"}
    if not required.issubset(df):
        raise ValueError(f"Required fields: {sorted(required)}")
    if df.sample_id.duplicated().any() or not df.role.isin(["train","validation","test"]).all():
        raise ValueError("Invalid sample IDs/roles")
    development = df[df.role=="train"]
    groups = sorted(development.group_id.unique())
    if n_inner_folds<2 or len(groups)<n_inner_folds+1:
        raise ValueError("Need enough independent gate-train groups for nested base tuning")
    protected_groups = set(df.loc[df.role=="validation","group_id"])
    if protected_groups & set(groups):
        raise ValueError("Gate validation groups overlap training")
    result = []
    for outer in groups + [None]:
        source = development if outer is None else development[development.group_id!=outer]
        available = sorted(source.group_id.unique())
        shuffled = np.random.default_rng(seed).permutation(available)
        bins = {str(g):i%n_inner_folds for i,g in enumerate(shuffled)}
        folds = []
        for fold in range(n_inner_folds):
            val_groups = {g for g,i in bins.items() if i==fold}
            folds.append({"fold":fold,"fit_sample_ids":source.loc[~source.group_id.isin(val_groups),"sample_id"].tolist(),
                          "validation_sample_ids":source.loc[source.group_id.isin(val_groups),"sample_id"].tolist()})
        predict = df[df.role!="train"] if outer is None else development[development.group_id==outer]
        result.append({"artifact_suffix":"outer" if outer is None else outer,
                       "base_source_sample_ids":source.sample_id.tolist(),
                       "base_source_group_ids":sorted(source.group_id.unique()),
                       "prediction_sample_ids":predict.sample_id.tolist(), "inner_folds":folds})
    return {"status":"NESTED_BASE_EXECUTION_PLAN_ONLY", "fits":result,
            "caveat":"Producers must implement spatial buffers, sampling-bias correction and fitting-log evidence."}
