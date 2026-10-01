"""Frozen, shared spatial partitions for presence-only MaxEnt evaluation.

Groups cover the complete native grid, including unsurveyed and NoData cells.
Every fitting/tuning mask excludes the locked groups, their buffer, and the
current validation groups and buffer. Backgrounds never receive absence labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

SEASONS = ("spring", "summer", "autumn", "winter")
CELL_FIELDS = {"native_cell_id", "raster_row", "raster_col", "x_utm49", "y_utm49"}


def canonical_hash(value):
    # JSON object keys become strings on disk; normalize before sorting so a
    # saved/reloaded group-code mapping hashes identically.
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _settings(config):
    spatial = config.get("spatial", config.get("spatial_split", config))
    def get(*keys, default=None):
        return next((spatial[k] for k in keys if k in spatial), default)
    result = {
        "block_size_m": get("block_size_m"), "buffer_m": get("buffer_m"),
        "n_outer_folds": get("outer_folds", "n_outer_folds", default=3),
        "n_inner_folds": get("inner_folds", "n_inner_folds", default=3),
        "locked_fraction": get("locked_test_fraction_target", "locked_fraction", default=.2),
        "assignment_seed_trials": get("assignment_seed_trials", default=64),
        "minimum_fit_presence": get("minimum_fit_presences", "minimum_fit_presence", default=20),
        "minimum_validation_presence": get("minimum_validation_presences", "minimum_validation_presence", default=10),
        "minimum_background": get("minimum_background", default=20),
        "seed": config.get("seed", get("seed", default=20261001)),
        "domain_balance_weight": .5,
        "buffer_conservative_margin": "one pixel diagonal; complete fit cells lie beyond buffered held-out cell polygons",
        "assignment_uses_model_outputs": False,
    }
    if result["block_size_m"] is None or result["buffer_m"] is None:
        raise ValueError("Freeze an evidence-based block size and buffer before splitting")
    if result["block_size_m"] <= 0 or result["buffer_m"] < 0 or not 0 < result["locked_fraction"] < .5:
        raise ValueError("Invalid spatial size, buffer, or locked fraction")
    for key in ("n_outer_folds", "n_inner_folds", "assignment_seed_trials"):
        if int(result[key]) != result[key] or result[key] < (2 if "folds" in key else 1):
            raise ValueError("Invalid split count: " + key)
        result[key] = int(result[key])
    return result


def _grid(grid, common_valid_mask):
    if str(grid.get("crs", grid.get("crs_epsg", ""))) not in ("EPSG:32649", "32649"):
        raise ValueError("Formal native method grid must declare EPSG:32649")
    gt = grid.get("transform_gdal", grid.get("affine_gdal", grid.get("geotransform")))
    if gt is None and grid.get("transform") is not None:
        a, b, c, d, e, f = grid["transform"]
        gt = (c, a, b, f, d, e)
    if gt is None or len(gt) != 6:
        raise ValueError("Supply an explicit GDAL affine transform")
    gt = tuple(float(v) for v in gt)
    if not np.isfinite(gt).all() or (gt[1], gt[2], gt[4], gt[5]) != (100., 0., 0., -100.):
        raise ValueError("Expected north-up, unrotated 100 m native grid")
    mask = np.asarray(common_valid_mask)
    if mask.ndim != 2 or mask.dtype != bool or not mask.any():
        raise ValueError("Supply a nonempty boolean common predictor-valid mask")
    height = int(grid.get("height", grid.get("shape_yx", mask.shape)[0]))
    width = int(grid.get("width", grid.get("shape_yx", mask.shape)[1]))
    if mask.shape != (height, width):
        raise ValueError("Common mask/grid shape mismatch")
    return {"crs": "EPSG:32649", "height": height, "width": width,
            "transform_gdal": list(gt), "resolution_m": 100.}, mask.copy()


def _check_table(table, grid, valid):
    if not CELL_FIELDS.issubset(table.columns):
        raise ValueError("Missing spatial cell fields: " + str(sorted(CELL_FIELDS - set(table.columns))))
    result = table.reset_index(drop=True).copy()
    if result.native_cell_id.isna().any() or result.native_cell_id.duplicated().any():
        raise ValueError("Cell IDs must be nonmissing and unique within each table")
    pos = result[["raster_row", "raster_col", "x_utm49", "y_utm49"]].to_numpy(dtype=float)
    if not np.isfinite(pos).all() or not np.equal(pos[:, :2], np.floor(pos[:, :2])).all():
        raise ValueError("Cell positions must be finite with integer rows/columns")
    rows, cols = pos[:, 0].astype(int), pos[:, 1].astype(int)
    if ((rows < 0) | (rows >= grid["height"]) | (cols < 0) | (cols >= grid["width"])).any():
        raise ValueError("Cell outside native grid")
    if not valid[rows, cols].all():
        raise ValueError("Missing predictor/NoData cell outside common valid domain")
    gt = grid["transform_gdal"]
    expected = np.column_stack((gt[0] + (cols + .5) * gt[1], gt[3] + (rows + .5) * gt[5]))
    if not np.allclose(pos[:, 2:], expected, atol=.01, rtol=0):
        raise ValueError("Cell centers conflict with grid/CRS")
    if result[["raster_row", "raster_col"]].duplicated().any():
        raise ValueError("Repeated raster cell under different IDs")
    return result


def rows_in_mask(table, mask):
    return np.flatnonzero(mask[table.raster_row.to_numpy(dtype=int), table.raster_col.to_numpy(dtype=int)])


def buffered_group_exclusion(group_raster, heldout_groups, buffer_m, resolution_m=100.):
    held = np.isin(group_raster, list(heldout_groups))
    if not held.any():
        return np.zeros(group_raster.shape, dtype=bool)
    if buffer_m == 0:
        return held
    # EDT measures cell-center distance. Adding a full diagonal is conservative
    # for the distance between complete source/held-out pixel polygons, so every
    # original source coordinate within a fit cell also satisfies the exclusion.
    margin = np.sqrt(2) * resolution_m
    return distance_transform_edt(~held, sampling=resolution_m) <= buffer_m + margin


def balanced_group_assignment(groups, counts, fractions, seed, trials=64, domain_weight=.5):
    """Balance fixed counts/domain mass; never consume predictors or scores."""
    groups = np.asarray(groups, dtype=int)
    counts = np.asarray(counts, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != len(groups) or len(groups) < len(fractions):
        raise ValueError("Too few spatial groups for requested partitions")
    if (counts < 0).any() or not np.isfinite(counts).all() or not np.isclose(fractions.sum(), 1):
        raise ValueError("Invalid balancing counts/fractions")
    total = np.maximum(counts.sum(axis=0), 1)
    weights = np.ones(counts.shape[1]); weights[-1] = domain_weight
    best = None
    for attempt in range(trials):
        rng = np.random.default_rng(int(seed) + attempt)
        noise = rng.random(len(groups))
        order = sorted(range(len(groups)), key=lambda i: (-float(np.max(counts[i] / total)),
                       -float(np.sum(counts[i] / total)), noise[i]))
        acc = np.zeros((len(fractions), counts.shape[1])); mapping = {}
        for i in order:
            cost = []
            for role in range(len(fractions)):
                cand = acc.copy(); cand[role] += counts[i]
                cost.append(float(np.sum((cand / total - fractions[:, None]) ** 2 /
                                         fractions[:, None] * weights)))
            role = int(np.argmin(cost)) if counts[i].sum() else int(rng.choice(len(fractions), p=fractions))
            mapping[int(groups[i])] = role; acc[role] += counts[i]
        score = float(np.sum((acc / total - fractions[:, None]) ** 2 / fractions[:, None] * weights))
        if best is None or score < best[0]: best = (score, mapping, acc, attempt)
    return {"mapping": best[1], "objective": best[0], "counts": best[2].tolist(),
            "selected_trial": best[3], "trials": trials, "seed": int(seed)}


def _groups_on_grid(grid, size, site_cells):
    rr, cc = np.indices((grid["height"], grid["width"]))
    gt = grid["transform_gdal"]
    bx = np.floor((gt[0] + (cc + .5) * 100) / size).astype(int)
    by = np.floor((gt[3] - (rr + .5) * 100) / size).astype(int)
    pairs = [tuple(pair) for pair in np.unique(np.column_stack((bx.ravel(), by.ravel())), axis=0).tolist()]
    pair_code = {p: i + 1 for i, p in enumerate(pairs)}
    base = np.empty(rr.shape, dtype=np.int32)
    for pair, code in pair_code.items(): base[(bx == pair[0]) & (by == pair[1])] = code
    parent = {code: code for code in pair_code.values()}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    merged = 0
    if site_cells is not None and len(site_cells):
        required = {"site_key", "raster_row", "raster_col"}
        if not required.issubset(site_cells): raise ValueError("Site links lack identity or grid cells")
        for _, rows in site_cells.groupby("site_key"):
            r = rows.raster_row.to_numpy(dtype=int); c = rows.raster_col.to_numpy(dtype=int)
            if ((r < 0) | (r >= grid["height"]) | (c < 0) | (c >= grid["width"])).any():
                raise ValueError("Source site outside native grid")
            codes = np.unique(base[r, c])
            for code in codes[1:]:
                a, b = find(int(codes[0])), find(int(code))
                if a != b: parent[max(a, b)] = min(a, b); merged += 1
    lookup = np.arange(len(pairs) + 1, dtype=np.int32)
    for code in parent: lookup[code] = find(code)
    raster = lookup[base]
    groups = sorted(set(lookup[1:].tolist()))
    records = [{"group_code": code, "group_id": "g_b%d_%d" % pairs[code - 1],
                "base_blocks": [{"block_id": "b%d_%d" % pair,
                                  "block_x": pair[0], "block_y": pair[1]}
                                 for pair, bcode in pair_code.items() if lookup[bcode] == code]}
               for code in groups]
    return raster, records, merged


def _count_matrix(groups, group_raster, mask, presence, backgrounds):
    matrix = []
    for group in groups:
        row = []
        for season in sorted(presence):
            p = presence[season]
            row.append(int(np.sum(mask[p.raster_row.to_numpy(int), p.raster_col.to_numpy(int)] &
                                   (p.group_code.to_numpy(int) == group))))
        for season in sorted(backgrounds):
            # B0 is the observed target-group candidate; if unavailable, balance
            # the first declared scheme. Other schemes share this frozen map.
            schemes = backgrounds[season]; b = schemes.get("B0", schemes[sorted(schemes)[0]])
            row.append(int(np.sum(mask[b.raster_row.to_numpy(int), b.raster_col.to_numpy(int)] &
                                   (b.group_code.to_numpy(int) == group))))
        row.append(int(np.sum(mask & (group_raster == group)))); matrix.append(row)
    return np.asarray(matrix, dtype=float)


def _indices_and_counts(fit_mask, val_mask, presence, backgrounds, settings, context):
    indices = {}; counts = {}
    for season, p in presence.items():
        fit = rows_in_mask(p, fit_mask).tolist(); val = rows_in_mask(p, val_mask).tolist()
        counts[season] = {"fit_presence": len(fit), "validation_presence": len(val), "background": {}}
        indices[season] = {"presence": {"fit": fit, "validation": val}, "background": {}}
        if len(fit) < settings["minimum_fit_presence"] or len(val) < settings["minimum_validation_presence"]:
            raise ValueError(f"Insufficient presence counts in {context}/{season}: fit={len(fit)}, val={len(val)}")
        for scheme, b in backgrounds[season].items():
            bf = rows_in_mask(b, fit_mask).tolist(); bv = rows_in_mask(b, val_mask).tolist()
            if min(len(bf), len(bv)) < settings["minimum_background"]:
                raise ValueError(f"Insufficient background counts in {context}/{season}/{scheme}")
            indices[season]["background"][scheme] = {"fit": bf, "validation": bv}
            counts[season]["background"][scheme] = {"fit": len(bf), "validation": len(bv)}
    return indices, counts


def build_spatial_split_plan(presence_by_season, background_by_season, grid,
                             common_valid_mask, config, site_cells=None):
    settings = _settings(config); grid, valid = _grid(grid, common_valid_mask)
    if settings["block_size_m"] % grid["resolution_m"]:
        raise ValueError("Block boundaries must align with the native 100 m grid")
    if set(presence_by_season) != set(background_by_season) or not presence_by_season:
        raise ValueError("Presence/background seasons must match")
    presence = {s: _check_table(p, grid, valid) for s, p in presence_by_season.items()}
    backgrounds = {s: {k: _check_table(b, grid, valid) for k, b in schemes.items()}
                   for s, schemes in background_by_season.items()}
    if any(not schemes for schemes in backgrounds.values()): raise ValueError("No declared background scheme")
    for schemes in backgrounds.values():
        for b in schemes.values():
            if any(name in b.columns for name in ("absence", "is_absence", "confirmed_absence")):
                raise ValueError("Presence-only backgrounds must not be labelled absence")
    group_raster, group_records, merged = _groups_on_grid(grid, settings["block_size_m"], site_cells)
    group_names = {g["group_code"]: g["group_id"] for g in group_records}
    for table in list(presence.values()) + [b for schemes in backgrounds.values() for b in schemes.values()]:
        table["group_code"] = group_raster[table.raster_row.to_numpy(int), table.raster_col.to_numpy(int)]
        table["group_id"] = table.group_code.map(group_names); table["spatial_group"] = table.group_id
    groups = sorted(group_names); counts = _count_matrix(groups, group_raster, valid, presence, backgrounds)
    fractions = [settings["locked_fraction"]] + [(1 - settings["locked_fraction"]) / settings["n_outer_folds"]] * settings["n_outer_folds"]
    assignment = balanced_group_assignment(groups, counts, fractions, settings["seed"],
                                           settings["assignment_seed_trials"], settings["domain_balance_weight"])
    roles = assignment["mapping"]; role_raster = np.empty(group_raster.shape, dtype=np.int16)
    for g, role in roles.items(): role_raster[group_raster == g] = role
    locked_groups = sorted(g for g, role in roles.items() if role == 0)
    locked_mask = valid & (role_raster == 0)
    locked_exclusion = buffered_group_exclusion(group_raster, locked_groups, settings["buffer_m"])
    development_mask = valid & ~locked_exclusion
    final_indices, final_counts = _indices_and_counts(development_mask, locked_mask, presence, backgrounds, settings, "final_locked_internal_test")
    masks = {"common_valid": valid, "group_raster": group_raster, "role_raster": role_raster,
             "locked_test": locked_mask, "locked_group_buffer": locked_exclusion,
             "development_fit": development_mask}
    outer = []
    for fold in range(settings["n_outer_folds"]):
        held_groups = sorted(g for g, role in roles.items() if role == fold + 1)
        held_exclusion = buffered_group_exclusion(group_raster, held_groups, settings["buffer_m"])
        fit_mask = development_mask & ~held_exclusion
        val_mask = development_mask & (role_raster == fold + 1)
        fit_groups = sorted(np.unique(group_raster[fit_mask]).tolist())
        indices, summary = _indices_and_counts(fit_mask, val_mask, presence, backgrounds, settings, f"outer_{fold}")
        fit_key, val_key = f"outer_{fold}_fit", f"outer_{fold}_validation"
        masks[fit_key] = fit_mask; masks[val_key] = val_mask
        inner_counts = _count_matrix(fit_groups, group_raster, fit_mask, presence, backgrounds)
        inner_assignment = balanced_group_assignment(fit_groups, inner_counts,
                            [1 / settings["n_inner_folds"]] * settings["n_inner_folds"],
                            settings["seed"] + 10000 + fold * 1000,
                            settings["assignment_seed_trials"], settings["domain_balance_weight"])
        inner = []
        for infold in range(settings["n_inner_folds"]):
            ih = sorted(g for g, f in inner_assignment["mapping"].items() if f == infold)
            iv = fit_mask & np.isin(group_raster, ih)
            it = fit_mask & ~buffered_group_exclusion(group_raster, ih, settings["buffer_m"])
            ii, ic = _indices_and_counts(it, iv, presence, backgrounds, settings, f"outer_{fold}/inner_{infold}")
            ikf, ikv = f"outer_{fold}_inner_{infold}_fit", f"outer_{fold}_inner_{infold}_validation"
            masks[ikf] = it; masks[ikv] = iv
            inner.append({"fold": infold, "fit_groups": sorted(np.unique(group_raster[it]).tolist()),
                          "validation_groups": ih, "fit_mask_key": ikf, "validation_mask_key": ikv,
                          "indices": ii, "counts": ic})
        outer.append({"fold": fold, "fit_groups": fit_groups, "validation_groups": held_groups,
                      "tune_groups": fit_groups, "fit_mask_key": fit_key, "validation_mask_key": val_key,
                      "indices": indices, "counts": summary, "inner_assignment": inner_assignment,
                      "inner_folds": inner})
    for record in group_records:
        record["role_code"] = roles[record["group_code"]]
        record["role"] = "locked_internal_test" if record["role_code"] == 0 else "development"
        record["outer_fold"] = None if record["role_code"] == 0 else record["role_code"] - 1
    for table in list(presence.values()) + [b for schemes in backgrounds.values() for b in schemes.values()]:
        r, c = table.raster_row.to_numpy(int), table.raster_col.to_numpy(int)
        table["outer_fold"] = role_raster[r, c] - 1
        table["split_role"] = np.where(locked_mask[r, c], "locked_internal_test",
                                np.where(locked_exclusion[r, c], "locked_buffer_excluded", "development"))
    final_groups = sorted(np.unique(group_raster[development_mask]).tolist())
    final_assignment = balanced_group_assignment(final_groups,
                       _count_matrix(final_groups, group_raster, development_mask, presence, backgrounds),
                       [1 / settings["n_inner_folds"]] * settings["n_inner_folds"],
                       settings["seed"] + 20000, settings["assignment_seed_trials"], settings["domain_balance_weight"])
    final_inner = []
    for infold in range(settings["n_inner_folds"]):
        ih = sorted(g for g, role in final_assignment["mapping"].items() if role == infold)
        iv = development_mask & np.isin(group_raster, ih)
        it = development_mask & ~buffered_group_exclusion(group_raster, ih, settings["buffer_m"])
        ii, ic = _indices_and_counts(it, iv, presence, backgrounds, settings, f"final_development_inner_{infold}")
        ikf, ikv = f"final_inner_{infold}_fit", f"final_inner_{infold}_validation"
        masks[ikf] = it; masks[ikv] = iv
        final_inner.append({"fold": infold, "fit_groups": sorted(np.unique(group_raster[it]).tolist()),
                            "validation_groups": ih, "fit_mask_key": ikf, "validation_mask_key": ikv,
                            "indices": ii, "counts": ic})
    plan = {"status": "FROZEN_SHARED_NESTED_SPATIAL_SPLIT", "grid": grid, "settings": settings,
            "config_sha256": canonical_hash(config), "group_records": group_records,
            "assignment": assignment, "named_site_block_merges": merged,
            "locked_groups": locked_groups, "final_indices": final_indices, "final_counts": final_counts,
            "outer_folds": outer, "final_inner_assignment": final_assignment, "final_inner_folds": final_inner,
            "mask_hashes": {k: array_hash(a) for k, a in masks.items()},
            "test_name": "Internal spatial test locked after this round's preregistration",
            "background_is_absence": False, "lst_variants_share_all_spatial_partitions": True,
            "minimum_counts_are_engineering_guards_not_sample_adequacy_proof": True}
    plan["split_hash"] = canonical_hash(plan)
    plan["masks"] = masks; plan["presence_by_season"] = presence; plan["background_by_season"] = backgrounds
    validate_split_plan(plan)
    return _runtime_views(plan)


def _runtime_views(plan):
    """Expose masks used by the runner without changing the frozen manifest."""
    for outer in plan["outer_folds"]:
        for fold in [outer] + outer["inner_folds"]:
            fold["fit_mask"] = plan["masks"][fold["fit_mask_key"]]
            fold["validation_mask"] = plan["masks"][fold["validation_mask_key"]]
    for fold in plan["final_inner_folds"]:
        fold["fit_mask"] = plan["masks"][fold["fit_mask_key"]]
        fold["validation_mask"] = plan["masks"][fold["validation_mask_key"]]
    plan["development_fit_mask"] = plan["masks"]["development_fit"]
    plan["development_groups"] = sorted(np.unique(plan["masks"]["group_raster"][plan["masks"]["development_fit"]]).tolist())
    plan["locked_mask"] = plan["masks"]["locked_test"]
    plan["group_raster"] = plan["masks"]["group_raster"]
    plan["group_manifest"] = plan["group_records"]
    return plan


def _manifest_view(plan):
    ignored = {"masks", "presence_by_season", "background_by_season", "development_fit_mask", "development_groups",
               "locked_mask", "group_raster", "group_manifest"}
    def strip(value):
        if isinstance(value, dict): return {k: strip(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
        if isinstance(value, list): return [strip(v) for v in value]
        return value
    return strip({k: v for k, v in plan.items() if k not in ignored})


def validate_split_plan(plan):
    masks = plan["masks"]; group = masks["group_raster"]; protected = masks["locked_group_buffer"]
    locked = set(plan["locked_groups"])
    if not masks["locked_test"].any() or np.any(masks["development_fit"] & protected):
        raise ValueError("Locked groups/buffer leaked into development")
    for outer in plan["outer_folds"]:
        held = set(outer["validation_groups"])
        if held & set(outer["fit_groups"]) or locked & set(outer["fit_groups"]) or locked & set(outer["tune_groups"]):
            raise ValueError("Held-out or locked group leaked into fit/tune provenance")
        for fold in [outer] + outer["inner_folds"]:
            fit = masks[fold["fit_mask_key"]]; val = masks[fold["validation_mask_key"]]
            if "fit_mask" in fold and (not np.array_equal(fold["fit_mask"], fit) or
                                        not np.array_equal(fold["validation_mask"], val)):
                raise ValueError("Runtime fold masks conflict with frozen mask inventory")
            if np.any(fit & val) or np.any(fit & protected) or np.any(val & protected):
                raise ValueError("Locked/test/validation mask leak")
            if set(np.unique(group[fit])) & set(fold["validation_groups"]):
                raise ValueError("Own prediction group included in model fitting")
            if set(np.unique(group[fit])) != set(fold["fit_groups"]):
                raise ValueError("Fit group provenance conflicts with grid mask")
            if fold is not outer and np.any((fit | val) & ~masks[outer["fit_mask_key"]]):
                raise ValueError("Inner tuning leaked outside outer fit source")
    for fold in plan["final_inner_folds"]:
        fit = masks[fold["fit_mask_key"]]; val = masks[fold["validation_mask_key"]]
        if "fit_mask" in fold and (not np.array_equal(fold["fit_mask"], fit) or
                                    not np.array_equal(fold["validation_mask"], val)):
            raise ValueError("Runtime final-fold masks conflict with frozen mask inventory")
        if np.any(fit & val) or np.any((fit | val) & protected) or not np.all(~fit | masks["development_fit"]):
            raise ValueError("Final-development inner fold leaked locked buffer or held-out data")
        if set(np.unique(group[fit])) & set(fold["validation_groups"]) or locked & set(fold["fit_groups"]):
            raise ValueError("Final-development tuning group leakage")
    for key, expected in plan["mask_hashes"].items():
        if array_hash(masks[key]) != expected: raise ValueError("Split mask hash mismatch: " + key)
    for key, canonical in (("development_fit_mask", "development_fit"), ("locked_mask", "locked_test"),
                           ("group_raster", "group_raster")):
        if key in plan and not np.array_equal(plan[key], masks[canonical]):
            raise ValueError("Runtime grid mask conflicts with frozen mask inventory")
    frozen = _manifest_view(plan); frozen.pop("split_hash", None)
    if canonical_hash(frozen) != plan["split_hash"]: raise ValueError("Split manifest hash mismatch")
    return True


def save_spatial_split_plan(plan, out, config, provenance=None):
    validate_split_plan(plan); out = Path(out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    frozen = _manifest_view(plan)
    (out / "split_plan.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config_snapshot.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "split_masks.npz", **plan["masks"])
    for season, p in plan["presence_by_season"].items():
        p.to_csv(out / f"presence_{season}.csv", index=False, encoding="utf-8-sig")
    for season, schemes in plan["background_by_season"].items():
        for scheme, b in schemes.items(): b.to_csv(out / f"background_{scheme}_{season}.csv", index=False, encoding="utf-8-sig")
    import rasterio
    from rasterio.transform import Affine
    for name in ("group_raster", "role_raster", "locked_group_buffer", "development_fit"):
        a = plan["masks"][name].astype(np.int32)
        with rasterio.open(out / (name + ".tif"), "w", driver="GTiff", width=a.shape[1], height=a.shape[0],
                           count=1, dtype="int32", crs=plan["grid"]["crs"],
                           transform=Affine.from_gdal(*plan["grid"]["transform_gdal"]), compress="deflate") as dst:
            dst.write(a, 1)
    manifest = {"status": plan["status"], "split_hash": plan["split_hash"],
                "config_sha256": plan["config_sha256"], "provenance": provenance or {},
                "outputs": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                            for p in out.iterdir() if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_spatial_split_plan(out):
    out = Path(out); manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for name, evidence in manifest["outputs"].items():
        if file_hash(out / name) != evidence["sha256"]: raise ValueError("Saved split output hash changed: " + name)
    plan = json.loads((out / "split_plan.json").read_text(encoding="utf-8"))
    with np.load(out / "split_masks.npz", allow_pickle=False) as archive:
        plan["masks"] = {k: archive[k] for k in archive.files}
    plan["presence_by_season"] = {p.stem.removeprefix("presence_"): pd.read_csv(p)
                                  for p in out.glob("presence_*.csv")}
    plan["background_by_season"] = {}
    for p in out.glob("background_*.csv"):
        scheme, season = p.stem.removeprefix("background_").split("_", 1)
        plan["background_by_season"].setdefault(season, {})[scheme] = pd.read_csv(p)
    validate_split_plan(plan)
    config = json.loads((out / "config_snapshot.json").read_text(encoding="utf-8"))
    if canonical_hash(config) != plan["config_sha256"] or manifest["split_hash"] != plan["split_hash"]:
        raise ValueError("Config/split hash changed")
    return _runtime_views(plan)


def save_split_plan(plan, out, config_path, inputs_path):
    """Runner-compatible save entry point with explicit config/input lineage."""
    import platform, subprocess, sys
    from datetime import datetime, timezone
    config_path, inputs_path = Path(config_path), Path(inputs_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if canonical_hash(config) != plan["config_sha256"]: raise ValueError("Config changed after split creation")
    input_manifest = inputs_path / "manifest.json"
    root = Path(__file__).resolve().parents[1]
    provenance = {"saved_utc": datetime.now(timezone.utc).isoformat(), "command_line": sys.argv,
                  "python": sys.version, "platform": platform.platform(),
                  "git_commit_sha": subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  "config_path": str(config_path.resolve()), "config_file_sha256": file_hash(config_path),
                  "inputs_path": str(inputs_path.resolve()),
                  "input_manifest_sha256": file_hash(input_manifest) if input_manifest.exists() else None,
                  "seed": plan["settings"]["seed"], "original_inputs_modified": False,
                  "fitted_models": 0, "test_predictions_evaluated": False}
    return save_spatial_split_plan(plan, out, config, provenance)


def load_split_plan(out):
    return load_spatial_split_plan(out)
