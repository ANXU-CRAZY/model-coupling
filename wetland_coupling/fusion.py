"""Explicit, bounded fusion; NoData never becomes a zero-valued habitat."""
from __future__ import annotations

import numpy as np


def unit_interval(values, name="value", allow_nan=True):
    a = np.asarray(values, dtype=float)
    if np.isinf(a).any() or (not allow_nan and np.isnan(a).any()):
        raise ValueError(f"{name}: non-finite values")
    finite = np.isfinite(a)
    if ((a[finite] < 0) | (a[finite] > 1)).any():
        raise ValueError(f"{name}: expected [0,1]; fix the inputs rather than rescale silently")
    return a


def geometric(a, b, weight=0.5, epsilon=1e-8):
    """Weighted geometric mean with exact zeros and meaningful endpoint weights."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0,1)")
    a, b, w = np.broadcast_arrays(unit_interval(a, "a"), unit_interval(b, "b"),
                                   unit_interval(weight, "weight", allow_nan=False))
    valid = np.isfinite(a) & np.isfinite(b)
    score = np.exp(w * np.log(np.maximum(a, epsilon)) +
                   (1 - w) * np.log(np.maximum(b, epsilon)))
    score = np.where(((a == 0) & (w > 0)) | ((b == 0) & (w < 1)), 0., score)
    score = np.where(w == 1, a, np.where(w == 0, b, score))
    return np.where(valid, score, np.nan)


def relative_degradation(h, habitat_suitability):
    """1-Q/H_j: within-type relative quality loss; distinct from raw degradation D.

    H_j=0 is inapplicable and returns NaN, not a maximally degraded habitat.
    This remains an InVEST-derived model quantity, not field-measured damage.
    """
    h,hj = np.broadcast_arrays(unit_interval(h,"HQ"),unit_interval(habitat_suitability,"H_j"))
    if np.any(h>hj+1e-6):
        raise ValueError("HQ must not exceed H_j")
    ratio = np.clip(np.divide(h,hj,out=np.zeros_like(h),where=hj>0),0.,1.)
    return np.where((hj>0) & np.isfinite(h) & np.isfinite(hj),1-ratio,np.nan)


def fuse(m, h, alpha=0.5, gamma=0.5, method="geometric", feasible=None,
         habitat_suitability=None, deficit_mode="low_hq"):
    """Return conservation score and a provisional low-HQ restoration candidate score.

    Restoration requires both an independent feasibility mask and H_j > 0.
    Areas requiring conversion from non-habitat need a separate restoration scenario.
    """
    m, h = np.broadcast_arrays(unit_interval(m, "M"), unit_interval(h, "HQ"))
    alpha = unit_interval(alpha, "alpha", allow_nan=False)
    gamma = unit_interval(gamma, "gamma", allow_nan=False)
    if feasible is None or habitat_suitability is None:
        raise ValueError("Restoration candidate scoring requires feasible and habitat_suitability")
    if deficit_mode == "low_hq":
        deficit = 1-h
    elif deficit_mode == "relative_degradation":
        rel = relative_degradation(h,habitat_suitability)
        deficit = np.where(np.asarray(habitat_suitability)==0,0.,rel)
    else:
        raise ValueError("Unknown deficit mode")
    if method == "geometric":
        conservation, restoration = geometric(m, h, alpha), geometric(m, deficit, gamma)
    elif method == "linear":
        conservation = alpha * m + (1 - alpha) * h
        restoration = gamma * m + (1 - gamma) * deficit
    elif method == "minimum":
        conservation, restoration = np.minimum(m, h), np.minimum(m, deficit)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    f, hj = np.broadcast_arrays(unit_interval(feasible, "feasible"),
                                unit_interval(habitat_suitability, "H_j"))
    if f.shape != m.shape:
        raise ValueError("Mask shapes must match the scores")
    if not np.isin(f[np.isfinite(f)], [0, 1]).all():
        raise ValueError("Feasibility must be a 0/1 mask, established outside this formula")
    invalid = ~np.isfinite(m) | ~np.isfinite(h) | ~np.isfinite(f) | ~np.isfinite(hj)
    eligible = (f == 1) & (hj > 0) & ~invalid
    return {
        "conservation": np.where(invalid, np.nan, conservation),
        "restoration_candidate": np.where(invalid, np.nan, np.where(eligible, restoration, 0.)),
        "restoration_eligible": eligible,
    }


def quadrants(m, h, feasible, habitat_suitability, m_high=0.7, h_high=0.7):
    """0=NoData; 1=protect; 2=restore candidate; 3=maintain; 4=general; 5=review.

    Thresholds are management settings, not fitted universal ecological constants.
    They must be frozen without consulting the final test data.
    """
    unit_interval([m_high, h_high], "thresholds", allow_nan=False)
    m, h, f, hj = np.broadcast_arrays(unit_interval(m), unit_interval(h),
                                     unit_interval(feasible), unit_interval(habitat_suitability))
    if not np.isin(f[np.isfinite(f)], [0, 1]).all():
        raise ValueError("Feasibility must be 0/1")
    valid = np.isfinite(m) & np.isfinite(h) & np.isfinite(f) & np.isfinite(hj)
    high_m, high_h = m >= m_high, h >= h_high
    zone = np.full(m.shape, 4, dtype=np.uint8)
    zone[high_m & high_h] = 1
    zone[~high_m & high_h] = 3
    zone[high_m & ~high_h] = 5
    zone[high_m & ~high_h & (f == 1) & (hj > 0)] = 2
    zone[~valid] = 0
    return zone


def propagate_paired(m_draws, h_draws, feasible, habitat_suitability,
                     alpha=0.5, gamma=0.5, m_high=0.7, h_high=0.7):
    """Propagate paired input members; quantiles are ensemble ranges, not coverage CIs.

    The pairing must be supplied by the experiment design. This function does not
    invent independence or a calibrated Bayesian posterior.
    """
    m, h = unit_interval(m_draws, "M draws"), unit_interval(h_draws, "HQ draws")
    if m.shape != h.shape or m.ndim < 2 or m.shape[0] < 2:
        raise ValueError("Paired draws must have the same [members,...] shape, with >=2 members")
    f = np.broadcast_to(unit_interval(feasible), m.shape)
    hj = np.broadcast_to(unit_interval(habitat_suitability), m.shape)
    if not np.array_equal(f, np.broadcast_to(f[0],f.shape), equal_nan=True):
        raise ValueError("Feasibility must remain fixed across uncertainty members")
    scores = fuse(m, h, alpha, gamma, feasible=f, habitat_suitability=hj)
    # Do not silently aggregate incomplete member sets: mark affected cells NoData.
    complete = np.isfinite(m).all(axis=0) & np.isfinite(h).all(axis=0)
    complete &= np.isfinite(f).all(axis=0) & np.isfinite(hj).all(axis=0)
    result = {}
    for key in ("conservation", "restoration_candidate"):
        q = np.quantile(np.where(np.isfinite(scores[key]), scores[key], 0.), [0.05, 0.5, 0.95], axis=0)
        for label, val in zip(("q05", "median", "q95"), q):
            result[f"{key}_{label}"] = np.where(complete, val, np.nan)
    zones = quadrants(m, h, f, hj, m_high, h_high)
    center_zone = quadrants(np.median(m, axis=0), np.median(h, axis=0),
                            f[0], np.median(hj,axis=0), m_high, h_high)
    agreement = np.mean(zones == center_zone, axis=0)
    result["zone_agreement"] = np.where(complete, agreement, np.nan)
    return result
