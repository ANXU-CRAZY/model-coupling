"""Auditable reference-surface inputs and fit-only presence-background sampling.

Historical response-based candidate screening remains an explicit limitation.
Background rows describe available environmental reference, never absences.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import transform
from scipy.ndimage import gaussian_filter

from .maxent_protocol import sha256, write_json

SEASONS = {"spring": "spr_select", "summer": "sum_select", "autumn": "aut_select", "winter": "win_select"}
EXPECTED_COUNTS = {"spring": 14, "summer": 17, "autumn": 15, "winter": 14}
CELL_COLUMNS = ["native_cell_id", "raster_row", "raster_col", "x_utm49", "y_utm49", "longitude", "latitude"]


def verify_hash(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if not isinstance(expected, str) or observed.lower() != expected.lower():
        raise ValueError("Input SHA256 mismatch: " + str(path))
    return observed


def validate_raster(path, reference_grid=None, expected_sha256=None):
    """Require declared CRS, explicit NoData, a north-up 100 m UTM49N grid."""
    if expected_sha256 is not None:
        verify_hash(path, expected_sha256)
    with rasterio.open(path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 32649:
            raise ValueError("Expected explicit EPSG:32649 CRS: " + str(path))
        if ds.count != 1 or ds.nodata is None or not np.isfinite(ds.nodata):
            raise ValueError("Expected one band and finite explicit NoData: " + str(path))
        t = ds.transform
        if (t.a, t.b, t.d, t.e) != (100., 0., 0., -100.):
            raise ValueError("Expected native north-up 100 m grid: " + str(path))
        grid = {"crs": ds.crs.to_string(), "epsg": 32649, "height": ds.height, "width": ds.width,
                "resolution_m": 100, "transform": list(t)[:6], "bounds": list(ds.bounds), "index_order": "row_major"}
        if reference_grid is not None:
            for field in ("crs", "height", "width", "transform"):
                if grid[field] != reference_grid[field]:
                    raise ValueError("Predictor grid mismatch in " + field + ": " + str(path))
        array = ds.read(1)
        valid = (ds.read_masks(1) != 0) & np.isfinite(array) & (array != ds.nodata)
    if not valid.any():
        raise ValueError("Predictor has no valid cells: " + str(path))
    return grid, valid


def _shape(grid):
    return int(grid["height"]), int(grid["width"])


def _mask(mask, grid):
    result = np.asarray(mask)
    if result.dtype != np.dtype(bool) or result.shape != _shape(grid):
        raise ValueError("Mask must be bool and match predictor raster shape")
    return result


def dataframe_cells(indices, env, names, grid):
    """Read row-major cell indices from a seasonal (height,width,predictor) array."""
    if str(grid.get("crs")) != "EPSG:32649" or int(grid.get("epsg", 32649)) != 32649:
        raise ValueError("Cell conversion requires the verified EPSG:32649 grid")
    if len(set(names)) != len(names) or not names:
        raise ValueError("Missing or duplicate predictor names")
    if env.shape != (*_shape(grid), len(names)):
        raise ValueError("Predictor array shape differs from grid/names")
    idx = np.asarray(indices)
    if idx.ndim != 1 or idx.dtype.kind not in "iu" or np.any(idx < 0) or np.any(idx >= grid["height"] * grid["width"]):
        raise ValueError("Supply in-range integer row-major cell indices")
    row, col = np.divmod(idx.astype(np.int64), grid["width"])
    values = np.asarray(env[row, col, :], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/NoData/nonfinite predictor at requested cell")
    a = Affine(*grid["transform"])
    x = a.c + (col + .5) * a.a + (row + .5) * a.b
    y = a.f + (col + .5) * a.d + (row + .5) * a.e
    if len(idx):
        longitude, latitude = transform("EPSG:32649", "EPSG:4326", x.tolist(), y.tolist())
    else:
        longitude, latitude = [], []
    frame = pd.DataFrame({"native_cell_id": [f"utm49_r{r}_c{c}" for r,c in zip(row,col)],
        "raster_row": row, "raster_col": col, "x_utm49": x, "y_utm49": y,
        "longitude": longitude, "latitude": latitude})
    for j,name in enumerate(names):
        frame[name] = values[:,j]
    return frame


def _indices(frame, grid):
    if not {"raster_row", "raster_col"}.issubset(frame):
        raise ValueError("Cell table missing raster_row/raster_col")
    r = pd.to_numeric(frame.raster_row, errors="coerce").to_numpy(float)
    c = pd.to_numeric(frame.raster_col, errors="coerce").to_numpy(float)
    if (not np.isfinite(r).all() or not np.isfinite(c).all() or np.any(r != np.floor(r))
            or np.any(c != np.floor(c)) or np.any(r < 0) or np.any(c < 0)
            or np.any(r >= grid["height"]) or np.any(c >= grid["width"])):
        raise ValueError("Cell table contains noninteger/out-of-grid cells")
    return r.astype(np.int64) * grid["width"] + c.astype(np.int64)


def _background_config(config):
    background = config.get("background", config)
    if background.get("is_absence", False) is not False:
        raise ValueError("Background cannot be designated as absence")
    return background


def sample_background(scheme, fit_mask, season_arrays, visits, B0, config, seed):
    """Sample a fit domain using only visits whose cells belong to that domain.

    season_arrays is {'env', 'names', 'grid', 'valid_mask'}. B0/visits are cell
    dataframes; visits use inferred_visit_count, never species or record counts.
    The B2 KDE is built anew for every fit. No held-out visit informs the kernel.
    """
    grid, env, names = season_arrays["grid"], season_arrays["env"], season_arrays["names"]
    fit = _mask(fit_mask, grid)
    valid = _mask(season_arrays["valid_mask"], grid)
    if np.any(fit & ~valid):
        raise ValueError("Fit mask includes cells outside common valid predictor domain")
    available = np.flatnonzero(fit)
    if not len(available):
        raise ValueError("Empty fit background domain")
    background = _background_config(config)
    requested = int(background.get("uniform_points_per_fit", 3000))
    if requested < 1:
        raise ValueError("Background sample size must be positive")
    audit = {"scheme": scheme, "background_is_absence": False, "seed": int(seed),
             "fit_valid_cells": len(available), "requested_rows": requested,
             "presence_background_overlap_allowed": True, "visit_effort_verified": False,
             "validation_or_locked_visits_used_for_bias": False}
    rng = np.random.default_rng(seed)
    count = min(requested, len(available))
    if scheme in ("B0", "B0_target_group"):
        candidate_indices = _indices(B0, grid)
        if len(np.unique(candidate_indices)) != len(candidate_indices):
            raise ValueError("B0 cells must be unique")
        idx = candidate_indices[fit.ravel()[candidate_indices]]
        if not len(idx):
            raise ValueError("B0 has no candidate cells in fit scope")
        audit["source_B0_cells"] = len(candidate_indices)
        audit["excluded_nonfit_B0_cells"] = len(candidate_indices) - len(idx)
    elif scheme in ("B1", "B1_uniform"):
        idx = np.sort(rng.choice(available, count, replace=False))
    elif scheme in ("B2", "B2_visit_density_proxy"):
        if "inferred_visit_count" not in visits:
            raise ValueError("B2 requires inferred event-cell visit counts")
        visit_indices = _indices(visits, grid)
        if len(np.unique(visit_indices)) != len(visit_indices):
            raise ValueError("Visit cells must be aggregated and unique")
        # Filter BEFORE reading counts: changes to held-out visit weights do not
        # influence fit density or random sample.
        include = fit.ravel()[visit_indices]
        fitted = visits.loc[include].copy()
        fit_indices = visit_indices[include]
        counts = pd.to_numeric(fitted.inferred_visit_count, errors="coerce").to_numpy(float)
        if not np.isfinite(counts).all() or np.any(counts <= 0) or np.any(counts != np.floor(counts)):
            raise ValueError("Fit inferred visit counts must be positive finite integers")
        density = np.zeros(_shape(grid), dtype=np.float64)
        density.ravel()[fit_indices] = counts
        sigma_m = float(background.get("kde_bandwidth_m", 1000))
        mixture = float(background.get("uniform_mixture_fraction", .1))
        if not np.isfinite(sigma_m) or sigma_m <= 0 or not 0 < mixture <= 1:
            raise ValueError("Require positive KDE bandwidth and uniform mixture in (0,1]")
        density = gaussian_filter(density, sigma=sigma_m / grid["resolution_m"], mode="constant", cval=0.)
        weights = density.ravel()[available]
        total = float(weights.sum())
        fallback = total <= 0
        probability = (1-mixture)*weights/total + mixture/len(available) if not fallback else None
        idx = np.sort(rng.choice(available, count, replace=False, p=probability))
        digest = hashlib.sha256()
        for cell,weight in sorted(zip(fit_indices.tolist(), counts.astype(int).tolist())):
            digest.update(f"{cell}:{weight}\n".encode("ascii"))
        audit.update(fit_visit_cells=len(fit_indices), fit_inferred_visits=int(counts.sum()),
                     excluded_nonfit_visit_cells=int((~include).sum()), fit_visit_digest=digest.hexdigest(),
                     kde_bandwidth_m=sigma_m, uniform_mixture_fraction=mixture,
                     no_fit_visits_fallback_uniform=fallback, bias_proxy="inferred_all_bird_visit_density_not_verified_effort")
    else:
        raise ValueError("Unknown background scheme: " + str(scheme))
    result = dataframe_cells(np.asarray(idx, dtype=np.int64), env, names, grid)
    result.insert(0, "species", "background")
    result["background_scheme"] = scheme
    result["background_is_absence"] = False
    audit.update(sampled_rows=len(result), available_domain_smaller_than_request=len(available) < requested,
                 sampled_cell_ids_sha256=hashlib.sha256("\n".join(result.native_cell_id).encode("ascii")).hexdigest())
    result.attrs["background_audit"] = audit
    return result


def _visit_cells(ledger_path, mask, grid):
    """Count unique inferred event-cell visits from positive original ND records."""
    columns = ["survey_event_key_unconfirmed", "abundance_numeric", "date_start_parsed", "season_calendar",
               "longitude_numeric_unconfirmed_crs", "latitude_numeric_unconfirmed_crs"]
    events = {season: set() for season in SEASONS}
    summary = Counter()
    for chunk in pd.read_csv(ledger_path, usecols=columns, chunksize=20000, low_memory=False):
        counts = pd.to_numeric(chunk.abundance_numeric, errors="coerce")
        lon = pd.to_numeric(chunk.longitude_numeric_unconfirmed_crs, errors="coerce")
        lat = pd.to_numeric(chunk.latitude_numeric_unconfirmed_crs, errors="coerce")
        dates = pd.to_datetime(chunk.date_start_parsed, errors="coerce")
        good = counts.gt(0) & dates.notna() & lon.between(-180,180) & lat.between(-90,90) & chunk.season_calendar.isin(SEASONS)
        selected = chunk.loc[good].copy()
        if selected.survey_event_key_unconfirmed.isna().any() or selected.survey_event_key_unconfirmed.astype(str).str.strip().eq("").any():
            raise ValueError("Positive ND visit source is missing inferred event identity")
        summary["eligible_positive_source_records"] += len(selected)
        if selected.empty:
            continue
        x,y = transform("EPSG:4326", "EPSG:32649", lon.loc[good].tolist(), lat.loc[good].tolist())
        a = Affine(*grid["transform"])
        col = np.floor((np.asarray(x)-a.c)/a.a).astype(int)
        row = np.floor((np.asarray(y)-a.f)/a.e).astype(int)
        inside = (row>=0) & (row<grid["height"]) & (col>=0) & (col<grid["width"])
        summary["positive_records_outside_native_extent"] += int((~inside).sum())
        within = np.zeros(len(selected), dtype=bool)
        within[inside] = mask[row[inside],col[inside]]
        summary["positive_records_inside_extent_outside_common_mask"] += int((inside & ~within).sum())
        for i,(season,event) in enumerate(zip(selected.season_calendar,selected.survey_event_key_unconfirmed.astype(str))):
            if within[i]:
                events[season].add((event, int(row[i])*grid["width"]+int(col[i])))
    result = {}
    for season,pairs in events.items():
        counter = Counter(cell for event,cell in pairs)
        indices = np.array(sorted(counter),dtype=np.int64)
        result[season] = (indices, np.array([counter[int(i)] for i in indices],dtype=np.int64))
        summary[season + "_unique_inferred_event_cell_pairs"] = len(pairs)
        summary[season + "_visit_cells"] = len(counter)
    return result, dict(summary)


def _filtered_candidate(frame, mask, grid, env, names, species):
    indices = _indices(frame, grid)
    if len(indices) != len(np.unique(indices)):
        raise ValueError("Candidate cell table must be deduplicated")
    if "native_cell_id" not in frame:
        raise ValueError("Candidate table is missing native cell identity")
    expected_ids = [f"utm49_r{i//grid['width']}_c{i%grid['width']}" for i in indices]
    if frame.native_cell_id.tolist() != expected_ids:
        raise ValueError("Candidate native_cell_id disagrees with row/column")
    keep = mask.ravel()[indices]
    retained = dataframe_cells(indices[keep], env, names, grid)
    for name in names:
        if name not in frame or not np.isfinite(pd.to_numeric(frame[name],errors="coerce").to_numpy()).all():
            raise ValueError("Candidate table missing/nonfinite predictor: " + name)
        values = frame.loc[keep,name].to_numpy(float)
        if not np.allclose(values, retained[name].to_numpy(float), rtol=1e-6, atol=1e-7):
            raise ValueError("Candidate raster samples disagree for predictor " + name)
    retained.insert(0,"species",species)
    if species == "background":
        retained["background_scheme"] = "B0_target_group"
        retained["background_is_absence"] = False
    excluded = frame.loc[~keep,CELL_COLUMNS[:3]].copy()
    excluded["reason"] = "outside_all_four_seasons_all_predictors_common_valid_domain"
    return retained, excluded


def build_formal_inputs(config_path, out_path, project_root=None):
    """Materialize a new private input run; every original hash/grid is checked."""
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    config_path = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else config_path.parent.parent
    out = Path(out_path).resolve()
    if out.exists():
        raise FileExistsError("Refusing existing formal-input output: " + str(out))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate = root / config["inputs"]["candidate_tables"]
    environment = root / config["inputs"]["environment"]
    ledger = root / config["inputs"]["bird_ledger"]
    source_manifest_path = candidate / "manifest.json"
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    observations = root / config["inputs"].get("prepared_observations", str(candidate.parent / "prepared_observations_002"))
    observation_manifest_path = observations / "manifest.json"
    verify_hash(observation_manifest_path, source["source_observation_manifest_sha256"])
    observation = json.loads(observation_manifest_path.read_text(encoding="utf-8"))
    if not observation.get("coordinate_reference_confirmed") or observation.get("confirmed_source_crs") != "EPSG:4326":
        raise ValueError("Require confirmed original WGS84 observations")
    verify_hash(ledger, observation["input_ledger_sha256"])
    scope_path = root / config["scope_policy"]
    verify_hash(scope_path, observation["scope_sha256"])
    _background_config(config)
    input_inventory = {str(p.resolve()): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in (config_path, source_manifest_path, observation_manifest_path, ledger, scope_path)}
    sources, grid, common = {}, None, None
    for season,directory in SEASONS.items():
        season_sources = source["seasons"][season]["environment_sources"]
        expected_names = {Path(item["path"]).name for item in season_sources}
        actual_names = {p.name for p in (environment/directory).glob("*.tif")}
        if len(expected_names) != EXPECTED_COUNTS[season] or actual_names != expected_names:
            raise ValueError("Missing/unexpected predictor raster in " + season)
        sources[season] = []
        for item in sorted(season_sources,key=lambda item: Path(item["path"]).name):
            path = environment / directory / Path(item["path"]).name
            checked_grid,valid = validate_raster(path,grid,item["sha256"])
            if grid is None:
                grid = checked_grid
                common = valid.copy()
            else:
                common &= valid
            sources[season].append(path)
            input_inventory[str(path.resolve())] = {"sha256": item["sha256"], "bytes": path.stat().st_size}
        # Every original candidate CSV in the supplying manifest is hash-checked.
        for filename,detail in source["outputs"].items():
            if filename.endswith("_"+season+".csv"):
                p = candidate / filename
                verify_hash(p,detail["sha256"])
                input_inventory[str(p.resolve())] = {"sha256": detail["sha256"], "bytes": p.stat().st_size}
    if not common.any():
        raise ValueError("Empty all-season predictor intersection")
    configured_grid = config.get("grid",{})
    if configured_grid.get("epsg",32649) != 32649 or configured_grid.get("resolution_m",100) != 100:
        raise ValueError("Config grid differs from verified native grid")
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"config.snapshot.json",config)
    write_json(out/"grid.json",grid)
    np.save(out/"common_valid_mask.npy",common,allow_pickle=False)
    a = Affine(*grid["transform"])
    with rasterio.open(out/"common_valid_mask.tif","w",driver="GTiff",width=grid["width"],height=grid["height"],
          count=1,dtype="uint8",nodata=0,crs=grid["crs"],transform=a,compress="deflate") as ds:
        ds.write(common.astype("uint8"),1)
    visits_by_season, visit_summary = _visit_cells(ledger,common,grid)
    season_summary = {}
    for season,paths in sources.items():
        directory = out/season
        directory.mkdir()
        names = [p.stem for p in paths]
        write_json(directory/"names.json",names)
        env = np.lib.format.open_memmap(directory/"env.npy",mode="w+",dtype="float32",shape=(*_shape(grid),len(names)))
        for j,path in enumerate(paths):
            with rasterio.open(path) as ds:
                array = ds.read(1)
                band = np.full(_shape(grid),np.nan,dtype="float32")
                band[common] = array[common]
                if not np.isfinite(band[common]).all():
                    raise ValueError("Predictor cannot be represented safely as float32: " + str(path))
                env[:,:,j] = band
            del array,band
        env.flush()
        tables = {}
        for kind,original in (("presence","occurrence_cells_"),("B0","target_group_background_cells_")):
            frame = pd.read_csv(candidate/(original+season+".csv"),low_memory=False)
            retained,excluded = _filtered_candidate(frame,common,grid,env,names,
                             "waterbird_community_candidate" if kind=="presence" else "background")
            retained.to_csv(directory/(kind+".csv"),index=False)
            excluded.to_csv(directory/(kind+"_excluded.csv"),index=False)
            tables[kind] = retained
            season_summary.setdefault(season,{})[kind] = {"source_rows":len(frame),"retained_rows":len(retained),"excluded_rows":len(excluded)}
        visit_indices,visit_counts = visits_by_season[season]
        visits = dataframe_cells(visit_indices,env,names,grid)[CELL_COLUMNS].copy()
        visits["inferred_visit_count"] = visit_counts
        visits["source_event_identity_verified"] = False
        visits["visit_effort_verified"] = False
        visits.to_csv(directory/"visit_cells.csv",index=False)
        arrays = {"env":env,"names":names,"grid":grid,"valid_mask":common}
        season_seed = int(config["seed"]) + list(SEASONS).index(season)
        B1 = sample_background("B1_uniform",common,arrays,visits,tables["B0"],config,season_seed)
        B1.to_csv(directory/"B1.csv",index=False)
        write_json(directory/"B1_sampling_audit.json",B1.attrs["background_audit"])
        season_summary[season].update(predictor_count=len(names),visit_cells=len(visits),
           inferred_event_cell_visits=int(visit_counts.sum()),B1_rows=len(B1),B1_seed=season_seed)
        del env,arrays,tables,frame,retained,excluded,visits,B1
    git = subprocess.check_output(["git","-c","safe.directory="+root.as_posix(),"rev-parse","HEAD"],cwd=root,text=True).strip()
    output_inventory = {str(p.relative_to(out)): {"sha256":sha256(p),"bytes":p.stat().st_size}
                        for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {"status":"FORMAL_REFERENCE_INPUTS_PREPARED_NO_MODEL_FITS","protocol_version":config["protocol_version"],
        "started_at_utc":started,"ended_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"git_commit_sha":git,
        "python_executable":sys.executable,"python_version":sys.version,"platform":platform.platform(),
        "versions":{"numpy":np.__version__,"pandas":pd.__version__,"rasterio":rasterio.__version__},
        "command_line":sys.argv,"random_seed":config["seed"],"split_hash":None,"split_not_yet_assigned":True,
        "config_sha256":sha256(config_path),"grid":grid,"common_valid_cells":int(common.sum()),
        "common_domain_all_four_seasons_and_all_predictors":True,"seasons":season_summary,"visit_source_summary":visit_summary,
        "background_is_absence":False,"presence_background_overlap_allowed":True,
        "B2_generated_globally":False,"B2_requires_fit_scope_visits_only":True,"visit_effort_verified":False,
        "scope_status":json.loads(scope_path.read_text(encoding="utf-8"))["status"],"species_synonym_review_complete":False,
        "candidate_predictor_provenance":"CANDIDATE_SET_PRESELECTED_FROM_PRIOR_STUDY",
        "oof_scope":"CONDITIONAL_ON_PRIOR_PRESELECTED_CANDIDATES","strict_end_to_end_oof":False,"gate_eligible":False,
        "static_lst_status":"STATIC_REFERENCE_THERMAL_COVARIATE_UNVERIFIED_PERIOD",
        "original_files_modified":False,"private_sensitive_data_do_not_commit":True,"inputs":input_inventory,"outputs":output_inventory}
    write_json(out/"manifest.json",manifest)
    return manifest
