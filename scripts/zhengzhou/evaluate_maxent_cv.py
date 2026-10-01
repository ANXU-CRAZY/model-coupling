"""Read-only development-CV summaries and figures; never choose with locked test.

Use the project CPU environment for --prepare-only (GeoTIFF -> plot NPZ), then
an environment with matplotlib for --render-only. No locked-test metrics are
read by this program. Full-resolution OOF arrays determine paired statistics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from scripts.zhengzhou.export_maxent_oof import SEASONS, VARIANTS, SCHEMES, sha256, read_json, write_json, verify_run_file, verify_split_inventory, resolve_migrated_project_path
except ModuleNotFoundError:
    from export_maxent_oof import SEASONS, VARIANTS, SCHEMES, sha256, read_json, write_json, verify_run_file, verify_split_inventory, resolve_migrated_project_path


def paired_prediction_metrics(a, b, fraction=.1):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not 0 < fraction < 1:
        raise ValueError("Paired score arrays must be one dimensional and aligned")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any((a<0)|(a>1)|(b<0)|(b>1)):
        raise ValueError("Require finite paired cloglog scores in [0,1]")
    if len(a)<2:
        raise ValueError("Insufficient paired prediction cells")
    constant = np.ptp(a)==0 or np.ptp(b)==0
    rho = None if constant else float(spearmanr(a,b).statistic)
    k = max(1,int(np.ceil(fraction*len(a))))
    if constant:
        intersection,jaccard,overlap = None,None,None
    else:
        aa = set(np.argsort(-a,kind="stable")[:k].tolist())
        bb = set(np.argsort(-b,kind="stable")[:k].tolist())
        intersection = len(aa & bb)
        jaccard = intersection/len(aa | bb)
        overlap = intersection/k
    return {"paired_cells":len(a),"spearman_rho":rho,"mean_static_minus_no_lst":float((b-a).mean()),
        "mae_static_vs_no_lst":float(np.abs(b-a).mean()),"top_fraction":fraction,"top_cells_per_variant":k,
        "top_intersection":intersection,"top_jaccard":jaccard,"top_overlap_fraction":overlap,
        "tie_rule":"row-major cell order breaks score ties; unavailable for constant predictions",
        "constant_prediction":constant,"locked_test_used_for_comparison":False}


def paired_outer_lst_metrics(outer, measures):
    """Pair LST variants on common outer folds and the fixed B1 background."""
    if outer.empty:
        return []
    required={"season","fold","variant","background",*measures}
    if not required.issubset(outer.columns):
        raise ValueError("Missing fields for fixed-background paired outer metrics")
    fixed=outer[outer.background=="B1_uniform"]
    if fixed.duplicated(["season","variant","fold"]).any():
        raise ValueError("Repeated B1-specific outer-fold metric")
    rows=[]
    for season,all_season in outer.groupby("season",sort=True):
        group=fixed[fixed.season==season]
        aa=group[group.variant=="no_lst"].set_index("fold")
        bb=group[group.variant=="static_lst"].set_index("fold")
        expected=set(all_season.fold.unique())
        if set(aa.index)!=expected or set(bb.index)!=expected:
            raise ValueError("Incomplete paired B1-specific outer-fold metrics: "+str(season))
        for fold in sorted(expected):
            row={"season":season,"fold":int(fold),"reference_background":"B1_uniform",
                 "selection_source":"variant-specific inner CV within B1_uniform; common held-out spatial fold",
                 "metric_scope":"B1-specific outer refit, not per-variant primary-background selection",
                 "locked_test_used":False}
            for metric in measures:
                row[metric+"_no_lst"]=float(aa.loc[fold,metric])
                row[metric+"_static_lst"]=float(bb.loc[fold,metric])
                row[metric+"_static_minus_no_lst"]=float(bb.loc[fold,metric]-aa.loc[fold,metric])
            rows.append(row)
    return rows


def predictor_jaccards(frame):
    rows,frequency = [],[]
    if frame.empty:
        return rows,frequency
    for keys,group in frame.groupby(["season","variant","background"],sort=True):
        sets = {int(r.fold):set(str(r.predictors).split(";"))-{"—","","nan"} for r in group.itertuples()}
        if len(sets)!=len(group):
            raise ValueError("Repeated outer fold predictor record")
        for a,b in itertools.combinations(sorted(sets),2):
            union = sets[a]|sets[b]
            rows.append(dict(zip(("season","variant","background"),keys),fold_a=a,fold_b=b,
                jaccard=len(sets[a]&sets[b])/len(union) if union else None,
                predictor_count_a=len(sets[a]),predictor_count_b=len(sets[b]),
                source="outer_refit_fit_background_only_selection"))
        all_names = sorted(set().union(*sets.values()))
        for name in all_names:
            n = sum(name in s for s in sets.values())
            frequency.append(dict(zip(("season","variant","background"),keys),predictor=name,
                folds_selected=n,folds_available=len(sets),selection_frequency=n/len(sets)))
    return rows,frequency


def _summary(frame, keys, measures):
    rows = []
    if frame.empty:
        return rows
    for group_key,group in frame.groupby(keys,sort=True):
        if not isinstance(group_key,tuple): group_key=(group_key,)
        row = dict(zip(keys,group_key));row["folds_available"]=len(group)
        for name in measures:
            values = pd.to_numeric(group[name],errors="raise").to_numpy(float)
            if not np.isfinite(values).all(): raise ValueError("Nonfinite development metric: "+name)
            row[name+"_mean"] = float(values.mean())
            row[name+"_sd"] = float(values.std(ddof=1)) if len(values)>1 else None
            row[name+"_variance"] = float(values.var(ddof=1)) if len(values)>1 else None
        rows.append(row)
    return rows


def _save_csv(path, rows, fallback_columns=None):
    frame = pd.DataFrame(rows)
    if frame.empty and fallback_columns: frame = pd.DataFrame(columns=fallback_columns)
    frame.to_csv(path,index=False)


def parse_convergence(html, iterations_csv=None, iteration_limit=None):
    """Read official termination text; this diagnostic never reranks models."""
    match=re.search(r"Algorithm (converged|terminated) after (\d+) iterations \((\d+) seconds\)",html)
    if match is None:
        return {"official_html_termination":"NOT_FOUND","official_html_iterations":None,
                "official_html_seconds":None,"convergence_verified":False,"iteration_limit_reached":None}
    termination,iterations,seconds=match.group(1),int(match.group(2)),int(match.group(3))
    if iterations_csv is not None and float(iterations_csv)!=iterations:
        raise ValueError("Official HTML and MaxEnt CSV iteration counts disagree")
    reached=termination=="terminated"
    if iteration_limit is not None and reached!=(iterations>=int(iteration_limit)):
        raise ValueError("Official HTML termination disagrees with frozen iteration limit")
    return {"official_html_termination":termination,"official_html_iterations":iterations,
            "official_html_seconds":seconds,"convergence_verified":True,"iteration_limit_reached":reached}


def convergence_inventory(run, outer):
    primary_paths=set()
    if not outer.empty:
        primary_paths={str(resolve_migrated_project_path(p,run).resolve()) for p in outer.loc[outer.chosen_background,"artifact"]}
    rows,consumed=[],[]
    for variant in VARIANTS:
        for path in sorted((run/variant).rglob("manifest.json")):
            engine=read_json(path)
            if engine.get("status")!="OFFICIAL_MAXENT_FITTED":continue
            relative=path.relative_to(run).parts
            if len(relative)<6:raise ValueError("Unrecognized official model path")
            season,scope=relative[1:3]
            if relative[3].startswith("inner_"):
                category="inner_candidate";background=relative[4]
            elif scope=="frozen_final":
                category="frozen_final_selected";background=relative[4]
            elif relative[3]=="refit":
                category="outer_selected_primary" if str(path.resolve()) in primary_paths else "outer_background_member"
                background=relative[4]
            else:raise ValueError("Unrecognized fitting scope in convergence audit")
            html_paths=list(path.parent.glob("*.html"))
            if len(html_paths)!=1:raise ValueError("Need one official model HTML for convergence audit")
            html_path=html_paths[0];html=html_path.read_text(encoding="utf-8",errors="replace")
            record=engine.get("outputs",{}).get(html_path.name)
            if record and sha256(html_path)!=record["sha256"]:raise ValueError("Official HTML changed since model completion")
            limit=next((int(a.split("=",1)[1]) for a in engine["maxent_arguments"] if a.startswith("maximumiterations=")),None)
            csv_iterations=engine.get("java_metrics",{}).get("Iterations")
            rows.append({"season":season,"variant":variant,"scope":scope,"category":category,"background":background,
                "rm":engine["rm"],"fc":engine["fc"],"engine_manifest":str(path),"html_path":str(html_path),
                "official_csv_iterations":csv_iterations,"frozen_maximumiterations":limit,
                **parse_convergence(html,csv_iterations,limit),"selection_rule_modified":False})
            consumed.extend((path,html_path))
    summary=[]
    if rows:
        frame=pd.DataFrame(rows)
        for key,f in frame.groupby(["season","variant","category"],sort=True):
            summary.append(dict(zip(("season","variant","category"),key),models=len(f),
                converged=int(f.official_html_termination.eq("converged").sum()),
                terminated_at_limit=int(f.official_html_termination.eq("terminated").sum()),
                convergence_unverified=int(f.official_html_termination.eq("NOT_FOUND").sum())))
    selected_final=[r for r in rows if r["category"]=="frozen_final_selected"]
    selected_outer=[r for r in rows if r["category"]=="outer_selected_primary"]
    limitations=[]
    if any(r["iteration_limit_reached"] for r in selected_final):limitations.append("FINAL_SELECTED_MODEL_REACHED_ITERATION_LIMIT_NOT_FULLY_CONVERGED")
    if any(r["iteration_limit_reached"] for r in selected_outer):limitations.append("OUTER_PRIMARY_MODEL_REACHED_ITERATION_LIMIT_CONDITIONAL_OOF_NUMERICAL_LIMITATION")
    if any(r["official_html_termination"]=="NOT_FOUND" for r in selected_final+selected_outer):limitations.append("SELECTED_MODEL_CONVERGENCE_UNVERIFIED")
    if len(selected_final)!=len(SEASONS)*len(VARIANTS):limitations.append("FINAL_SELECTED_MODEL_CONVERGENCE_AUDIT_INCOMPLETE")
    status={"completed_models_inspected":len(rows),"selected_final_models":len(selected_final),"selected_outer_primary_models":len(selected_outer),
        "numerical_limitations":limitations,"candidate_selection_rule_modified":False,
        "iterations_or_candidates_rerun":False,"selected_models_numerical_convergence_verified":not bool(limitations),
        "interpretation":"Termination at the frozen maximum is a numerical limitation; no new fitting or selection was performed"}
    return rows,summary,status,consumed


def prepare_reports(run,inputs,splits,out,allow_incomplete=False,plot_stride=3):
    start = datetime.now(timezone.utc).isoformat()
    run,inputs,splits,out = map(lambda p:Path(p).resolve(),(run,inputs,splits,out))
    if out.exists(): raise FileExistsError("Refusing existing CV postprocessing output: "+str(out))
    if plot_stride<1: raise ValueError("Positive plotting stride required")
    state = read_json(run/"manifests/run_manifest.json")
    verify_split_inventory(splits)
    plan = read_json(splits/"split_plan.json")
    if state.get("split_manifest_sha256") != sha256(splits/"manifest.json"):
        raise ValueError("Evaluation split differs from fitted run")
    if state.get("strict_end_to_end_oof") is not False or state.get("gate_eligible") is not False:
        raise ValueError("Historical preselection limitations must be preserved")
    with np.load(splits/"split_masks.npz",allow_pickle=False) as archive:
        masks = {k:archive[k] for k in ("common_valid","group_raster","role_raster","locked_test","locked_group_buffer",
                    *[f"outer_{f['fold']}_validation" for f in plan["outer_folds"]])}
    consumed = [run/"manifests/run_manifest.json",inputs/"manifest.json",splits/"manifest.json",splits/"split_plan.json",splits/"split_masks.npz"]
    outer_file = run/"reports/outer_metrics.csv"
    if outer_file.exists():
        verify_run_file(run,outer_file,state,not allow_incomplete);consumed.append(outer_file)
        outer = pd.read_csv(outer_file)
        required = {"season","fold","variant","background","chosen_background","predictors","validation_auc","omission_10","complexity","auc_gap"}
        if not required.issubset(outer): raise ValueError("Missing official outer metric fields")
        if outer.duplicated(["season","fold","variant","background"]).any(): raise ValueError("Repeated outer metric member")
        truth = outer.chosen_background.astype(str).str.lower().map({"true":True,"false":False})
        if truth.isna().any(): raise ValueError("Invalid chosen-background flag")
        outer["chosen_background"]=truth
    else:
        outer = pd.DataFrame()
    expected_rows=len(SEASONS)*len(VARIANTS)*len(SCHEMES)*len(plan["outer_folds"])
    missing = []
    if len(outer)!=expected_rows: missing.append(f"outer_metrics_complete_rows_{len(outer)}_of_{expected_rows}")
    measures = ["train_auc","validation_auc","auc_gap","omission_10","omission_min","complexity","prediction_min","prediction_max"]
    if not outer.empty and not set(measures).issubset(outer): raise ValueError("Missing outer metrics")
    scheme_summary = _summary(outer,["season","variant","background"],measures)
    primary = outer.loc[outer.chosen_background].copy() if not outer.empty else pd.DataFrame()
    primary_summary = _summary(primary,["season","variant"],measures)
    jaccards,frequency = predictor_jaccards(outer)
    paired_scores = paired_outer_lst_metrics(outer, measures)
    background_pairs=[]
    if not outer.empty:
        for (season,variant,fold),group in outer.groupby(["season","variant","fold"]):
            indexed=group.set_index("background")
            for a,b in itertools.combinations(SCHEMES,2):
                if a in indexed.index and b in indexed.index:
                    background_pairs.append({"season":season,"variant":variant,"fold":int(fold),"background_a":a,"background_b":b,
                        "validation_auc_b_minus_a":float(indexed.loc[b,"validation_auc"]-indexed.loc[a,"validation_auc"]),
                        "omission_10_b_minus_a":float(indexed.loc[b,"omission_10"]-indexed.loc[a,"omission_10"]),
                        "common_validation_reference":"B1_uniform","background_is_absence":False})
    plot_data={"role":masks["role_raster"][::plot_stride,::plot_stride],"common":masks["common_valid"][::plot_stride,::plot_stride],
               "locked_buffer":masks["locked_group_buffer"][::plot_stride,::plot_stride]}
    paired_rasters=[]
    expected=np.logical_or.reduce([masks[f"outer_{fold['fold']}_validation"] for fold in plan["outer_folds"]]).ravel()
    shape=masks["common_valid"].shape
    for season in SEASONS:
        oof_values={}
        for variant in VARIANTS:
            oof_path=run/"oof"/f"{season}_{variant}.npz"
            if oof_path.exists():
                verify_run_file(run,oof_path,state,not allow_incomplete);consumed.append(oof_path)
                with np.load(oof_path,allow_pickle=False) as z:
                    values=z["M_oof"].ravel();std=z["std"].ravel();count=z["member_count"].ravel()
                if not np.array_equal(np.isfinite(values),expected) or not np.array_equal(np.isfinite(std),expected):
                    raise ValueError("Partial/contaminated OOF prediction domain")
                if np.any(count[expected]!=len(SCHEMES)) or np.any(std[expected]<0): raise ValueError("OOF member uncertainty invalid")
                oof_values[variant]=values
                plot_data[f"{season}_{variant}_oof_std"]=std.reshape(shape)[::plot_stride,::plot_stride].astype("float32")
                plot_data[f"{season}_{variant}_oof_M"]=values.reshape(shape)[::plot_stride,::plot_stride].astype("float32")
            else: missing.append(f"oof_{season}_{variant}")
            final_npy=run/"predictions"/f"{season}_{variant}_cloglog.npy"
            final_path=run/"predictions"/f"{season}_{variant}_cloglog.tif"
            if final_npy.exists():
                verify_run_file(run,final_npy,state,not allow_incomplete);consumed.append(final_npy)
                a=np.load(final_npy,allow_pickle=False,mmap_mode="r").reshape(shape)
                valid=np.isfinite(a)
                if not np.array_equal(valid,masks["common_valid"]):raise ValueError("Final NPY prediction domain differs from common domain")
                if np.any((a[valid]<0)|(a[valid]>1)):raise ValueError("Final cloglog score outside [0,1]")
                plot_data[f"{season}_{variant}_final"]=a[::plot_stride,::plot_stride].astype("float32")
                continue
            if final_path.exists():
                import rasterio
                verify_run_file(run,final_path,state,not allow_incomplete);consumed.append(final_path)
                with rasterio.open(final_path) as ds:
                    if ds.crs is None or ds.crs.to_epsg()!=32649 or ds.shape!=shape: raise ValueError("Final prediction grid/CRS mismatch")
                    a=ds.read(1);valid=(ds.read_masks(1)!=0)&np.isfinite(a)
                    if ds.nodata is not None: valid &= a!=ds.nodata
                    if not np.array_equal(valid,masks["common_valid"]): raise ValueError("Final prediction missing common domain cells")
                    if np.any((a[valid]<0)|(a[valid]>1)): raise ValueError("Final cloglog score outside [0,1]")
                    plot_data[f"{season}_{variant}_final"]=np.where(valid,a,np.nan)[::plot_stride,::plot_stride].astype("float32")
            else: missing.append(f"final_prediction_{season}_{variant}")
        if set(oof_values)==set(VARIANTS):
            paired_rasters.append({"season":season,**paired_prediction_metrics(oof_values["no_lst"][expected],oof_values["static_lst"][expected])})
        del oof_values
    inner=[]
    for path in sorted((run/"reports").glob("inner_*_*.csv")):
        verify_run_file(run,path,state,not allow_incomplete);consumed.append(path)
        frame=pd.read_csv(path)
        inner.extend(frame.to_dict("records"))
    tuning=[]
    for season in SEASONS:
        path=run/"reports"/f"tuning_{season}_full_development.json"
        if path.exists():
            verify_run_file(run,path,state,not allow_incomplete);consumed.append(path)
            tuning.extend({"season":season,**r} for r in read_json(path)["all_candidates"])
        else: missing.append(f"development_tuning_{season}")
    convergence,convergence_summary,numerical_status,convergence_sources=convergence_inventory(run,outer)
    consumed.extend(convergence_sources)
    if missing and not allow_incomplete: raise ValueError("Completed four-season run required: "+", ".join(missing))
    out.mkdir(parents=True,exist_ok=False)
    _save_csv(out/"outer_background_summary.csv",scheme_summary,["season","variant","background"])
    _save_csv(out/"outer_primary_summary.csv",primary_summary,["season","variant"])
    _save_csv(out/"outer_member_metrics.csv",outer.to_dict("records"),["season","variant","fold"])
    _save_csv(out/"paired_lst_outer_metrics.csv",paired_scores,["season","fold"])
    _save_csv(out/"paired_lst_oof_spatial_metrics.csv",paired_rasters,["season"])
    _save_csv(out/"background_sensitivity_pairs.csv",background_pairs,["season","variant"])
    _save_csv(out/"variable_jaccard_stability.csv",jaccards,["season","variant","background"])
    _save_csv(out/"variable_selection_frequency.csv",frequency,["season","variant","background","predictor"])
    _save_csv(out/"development_rm_fc_candidates.csv",tuning,["season","variant","background","rm","fc","mean_auc"])
    _save_csv(out/"inner_development_metrics.csv",inner,["season","variant"])
    _save_csv(out/"model_numerical_convergence.csv",convergence,["season","variant","category"])
    _save_csv(out/"model_numerical_convergence_summary.csv",convergence_summary,["season","variant","category"])
    write_json(out/"numerical_limitations.json",numerical_status)
    for season in SEASONS:
        for kind,name in (("presence",f"presence_{season}.csv"),("B0",f"background_B0_{season}.csv"),("B1",f"background_B1_{season}.csv")):
            p=splits/name;consumed.append(p)
            f=pd.read_csv(p)
            f[["x_utm49","y_utm49","raster_row","raster_col","split_role"]].to_csv(out/f"plot_{season}_{kind}.csv",index=False)
    np.savez_compressed(out/"plot_data.npz",**plot_data)
    gt=plan["grid"]["transform_gdal"]
    bounds=[gt[0]/1000,(gt[0]+plan["grid"]["width"]*gt[1])/1000,
            (gt[3]+plan["grid"]["height"]*gt[5])/1000,gt[3]/1000]
    meta={"bounds_km_utm49":bounds,"plot_stride":plot_stride,"full_resolution_statistics":True,"grid":plan["grid"],
          "role_labels":{0:"Locked internal test",1:"Outer fold 0",2:"Outer fold 1",3:"Outer fold 2"},
          "expected_outer_folds":len(plan["outer_folds"]),"missing_artifacts":missing,
          "scope":"family-based candidate waterbird community","oof_condition":"prior-preselected candidates",
          "gate_eligible":False,"uncertainty_is_confidence_interval":False}
    write_json(out/"plot_metadata.json",meta)
    inputs_inventory={str(p):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in dict.fromkeys(consumed)}
    manifest={"status":"CV_POSTPROCESSING_DATA_PREPARED" if not missing else "PARTIAL_CV_POSTPROCESSING_DATA_PREPARED",
        "started_at_utc":start,"prepared_at_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"command_line":sys.argv,
        "git_commit":state.get("git_commit"),"seed":state.get("seed"),"split_manifest_sha256":sha256(splits/"manifest.json"),
        "config_sha256":state.get("config_sha256"),"maxent_runtime":state.get("runtime"),"locked_test_metrics_read":False,
        "model_selection_performed":False,"strict_end_to_end_oof":False,"gate_eligible":False,"missing_artifacts":missing,
        "numerical_limitations":numerical_status["numerical_limitations"],
        "source_files_modified":False,"private_coordinates_do_not_commit":True,"inputs":inputs_inventory,
        "outputs":{str(p.relative_to(out)):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in out.iterdir() if p.is_file()}}
    write_json(out/"manifest.json",manifest)
    return manifest


def render_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    out=Path(out).resolve();figure_dir=out/"figures"
    if figure_dir.exists(): raise FileExistsError("Refusing existing figure output")
    manifest=read_json(out/"manifest.json")
    for name,record in manifest["outputs"].items():
        if sha256(out/name)!=record["sha256"]: raise ValueError("Prepared plot data changed: "+name)
    meta=read_json(out/"plot_metadata.json");extent=meta["bounds_km_utm49"]
    figure_dir.mkdir()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,
                         "savefig.dpi":180,"figure.dpi":110,"axes.spines.top":False,"axes.spines.right":False})
    palette=["#E9B44C","#437F97","#9A67AC","#6EAA6D"]
    with np.load(out/"plot_data.npz",allow_pickle=False) as z:
        roles=z["role"];common=z["common"];buffer=z["locked_buffer"]
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        axes[0].imshow(np.ma.array(roles,mask=~common),extent=extent,origin="upper",cmap=ListedColormap(palette),norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),interpolation="nearest")
        axes[0].set_title("Frozen 10 km spatial groups")
        axes[0].legend(handles=[Patch(color=palette[i],label=meta["role_labels"][str(i)]) for i in range(4)],fontsize=8,loc="lower left")
        axes[1].imshow(np.ma.array(buffer.astype(float),mask=~common),extent=extent,origin="upper",cmap=ListedColormap(["#D6E4D1","#E7B9A0"]),vmin=0,vmax=1,interpolation="nearest")
        axes[1].set_title("Locked groups and conservative 1 km exclusion buffer")
        axes[1].legend(handles=[Patch(color="#D6E4D1",label="Outside locked buffer"),Patch(color="#E7B9A0",label="Excluded from all development fits")],fontsize=8,loc="lower left")
        for ax in axes: ax.set_xlabel("UTM49N easting (km)");ax.set_ylabel("UTM49N northing (km)")
        fig.suptitle("Internal spatial validation; native common valid domain")
        fig.savefig(figure_dir/"01_spatial_folds.png");plt.close(fig)

        fig,axes=plt.subplots(4,3,figsize=(13,12),constrained_layout=True)
        for i,season in enumerate(SEASONS):
            for j,kind in enumerate(("presence","B0","B1")):
                ax=axes[i,j];table=pd.read_csv(out/f"plot_{season}_{kind}.csv")
                ax.imshow(np.ma.array(np.zeros_like(common,dtype=float),mask=~common),extent=extent,origin="upper",cmap=ListedColormap(["#F1F2F1"]),interpolation="nearest")
                locked=table.split_role.eq("locked_internal_test")
                ax.scatter(table.loc[~locked,"x_utm49"]/1000,table.loc[~locked,"y_utm49"]/1000,s=6 if kind=="presence" else 2,color="#315B76",alpha=.5)
                ax.scatter(table.loc[locked,"x_utm49"]/1000,table.loc[locked,"y_utm49"]/1000,s=9 if kind=="presence" else 3,color="#C18A22",alpha=.8)
                ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_title(f"{season.title()} | {kind} | n={len(table)}")
                if i==3: ax.set_xlabel("Easting (km)")
                if j==0: ax.set_ylabel("Northing (km)")
        fig.suptitle("Presences and background candidates; amber = locked groups\nB0: all-bird observed cells; B1: common-domain uniform background; backgrounds are not absences",fontsize=12)
        fig.savefig(figure_dir/"02_presence_background.png");plt.close(fig)

        tuning=pd.read_csv(out/"development_rm_fc_candidates.csv")
        fig,axes=plt.subplots(4,6,figsize=(18,11),constrained_layout=True)
        for i,season in enumerate(SEASONS):
            for j,(variant,scheme) in enumerate(itertools.product(VARIANTS,SCHEMES)):
                ax=axes[i,j]
                if not tuning.empty:
                    t=tuning[(tuning.season==season)&(tuning.variant==variant)&(tuning.background==scheme)]
                else:t=pd.DataFrame()
                if not t.empty:
                    matrix=t.pivot(index="rm",columns="fc",values="mean_auc").reindex(index=[.5,1,2,4],columns=["L","LQ","LQH"])
                    im=ax.imshow(matrix.to_numpy(float),vmin=.3,vmax=1,cmap="viridis",aspect="auto")
                    ax.set_xticks(range(3),["L","LQ","LQH"]);ax.set_yticks(range(4),[".5","1","2","4"])
                    for y,x in itertools.product(range(4),range(3)):
                        value=matrix.iloc[y,x]
                        if pd.notna(value):ax.text(x,y,f"{value:.2f}",ha="center",va="center",fontsize=8,color="white" if value<.7 else "black")
                else:ax.text(.5,.5,"No completed tuning",ha="center",va="center",transform=ax.transAxes);ax.set_xticks([]);ax.set_yticks([])
                ax.set_title(f"{season[:3]} | {variant} | {scheme[:2]}",fontsize=9)
                if j==0:ax.set_ylabel("RM")
        if "im" in locals():fig.colorbar(im,ax=axes.ravel().tolist(),label="Mean full-development inner held-out/background AUC",shrink=.7)
        fig.suptitle("Predeclared RM x FC development grid; each LST/background combination independently tuned")
        fig.savefig(figure_dir/"03_rm_fc_development.png");plt.close(fig)

        paired=pd.read_csv(out/"paired_lst_outer_metrics.csv")
        spatial=pd.read_csv(out/"paired_lst_oof_spatial_metrics.csv")
        fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax,metric,label in zip(axes[:2],("validation_auc","omission_10"),("Held-out/background AUC","Held-out omission at fit 10% threshold")):
            for i,season in enumerate(SEASONS):
                if not paired.empty:
                    t=paired[paired.season==season]
                    for offset,row in enumerate(t.itertuples()):
                        x=i+(offset-(len(t)-1)/2)*.12
                        ax.plot([x-.1,x+.1],[getattr(row,metric+"_no_lst"),getattr(row,metric+"_static_lst")],color="#AAAAAA",lw=1)
                        ax.scatter(x-.1,getattr(row,metric+"_no_lst"),color="#437F97",s=22)
                        ax.scatter(x+.1,getattr(row,metric+"_static_lst"),color="#9A67AC",s=22)
            ax.set_xticks(range(4),[s[:3].title() for s in SEASONS]);ax.set_ylabel(label)
        axes[0].legend(handles=[Patch(color="#437F97",label="No LST"),Patch(color="#9A67AC",label="Static LST")],loc="best",fontsize=8)
        if not spatial.empty:
            indexed=spatial.set_index("season").reindex(SEASONS)
            axes[2].bar(np.arange(4)-.18,indexed.spearman_rho,width=.35,color="#437F97",label="OOF Spearman")
            axes[2].bar(np.arange(4)+.18,indexed.top_jaccard,width=.35,color="#9A67AC",label="OOF top 10% Jaccard")
        axes[2].set_xticks(range(4),[s[:3].title() for s in SEASONS]);axes[2].set_ylim(-1,1);axes[2].legend(fontsize=8)
        fig.suptitle("Static vs no LST: same outer spatial folds; conditional OOF prediction agreement")
        fig.savefig(figure_dir/"04_lst_paired_comparison.png");plt.close(fig)

        for kind,filename,title,label in (("final","05_final_prediction_maps.png","Frozen development-fit reference-surface predictions","Cloglog suitability; not calibrated probability"),
                                         ("oof_std","06_oof_uncertainty_maps.png","Conditional OOF member dispersion across three backgrounds","Member standard deviation; not a confidence interval")):
            fig,axes=plt.subplots(4,2,figsize=(11,13),constrained_layout=True)
            limit=1.
            if kind=="oof_std":
                maxima=[float(np.nanmax(z[f"{s}_{v}_{kind}"])) for s,v in itertools.product(SEASONS,VARIANTS) if f"{s}_{v}_{kind}" in z.files]
                limit=max(maxima,default=.1) or .1
            images=[]
            for i,season in enumerate(SEASONS):
                for j,variant in enumerate(VARIANTS):
                    ax=axes[i,j];key=f"{season}_{variant}_{kind}"
                    if key in z.files:
                        images.append(ax.imshow(z[key],extent=extent,origin="upper",cmap="viridis" if kind=="final" else "magma",vmin=0,vmax=limit,interpolation="nearest"))
                    else:ax.text(.5,.5,"No completed prediction",ha="center",va="center",transform=ax.transAxes)
                    ax.set_title(f"{season.title()} | {variant}");ax.set_xlabel("Easting (km)");ax.set_ylabel("Northing (km)")
            if images:fig.colorbar(images[0],ax=axes.ravel().tolist(),label=label,shrink=.55)
            fig.suptitle(title+"\nFamily-based community candidate scope; historical predictor preselection retained",fontsize=12)
            fig.savefig(figure_dir/filename);plt.close(fig)

        summary=pd.read_csv(out/"outer_background_summary.csv")
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        colors=["#437F97","#9A67AC","#6EAA6D"]
        for ax,variant in zip(axes,VARIANTS):
            for j,scheme in enumerate(SCHEMES):
                if not summary.empty:
                    f=summary[(summary.variant==variant)&(summary.background==scheme)].set_index("season").reindex(SEASONS)
                    ax.errorbar(np.arange(4)+(j-1)*.16,f.validation_auc_mean,yerr=f.validation_auc_sd,fmt="o-",markersize=4,color=colors[j],label=scheme,capsize=3)
            ax.set_xticks(range(4),[s[:3].title() for s in SEASONS]);ax.set_title(variant);ax.set_ylabel("Outer held-out/background AUC (mean +/- fold SD)");ax.set_ylim(0,1);ax.legend(fontsize=7)
        fig.suptitle("Background sensitivity on a common B1 held-out reference; fold SD is not a confidence interval")
        fig.savefig(figure_dir/"07_background_sensitivity.png");plt.close(fig)
    manifest.update(status="CV_POSTPROCESSING_COMPLETE" if not meta["missing_artifacts"] else "PARTIAL_CV_POSTPROCESSING_COMPLETE",
        rendered_at_utc=datetime.now(timezone.utc).isoformat(),render_python=sys.version,matplotlib_version=matplotlib.__version__,
        figure_classes=["spatial_fold_map","presence_background_distribution","RM_FC_development","paired_LST","final_predictions","OOF_uncertainty","background_sensitivity"])
    manifest["outputs"]={str(p.relative_to(out)):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in out.rglob("*") if p.is_file() and p.name!="manifest.json"}
    write_json(out/"manifest.json",manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("run","inputs","splits"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--out",type=Path,required=True)
    modes=p.add_mutually_exclusive_group();modes.add_argument("--prepare-only",action="store_true");modes.add_argument("--render-only",action="store_true")
    p.add_argument("--allow-incomplete",action="store_true")
    p.add_argument("--plot-stride",type=int,default=3)
    a=p.parse_args()
    if a.render_only: result=render_figures(a.out)
    else:
        if not all((a.run,a.inputs,a.splits)):p.error("--run --inputs --splits are required for preparation")
        result=prepare_reports(a.run,a.inputs,a.splits,a.out,a.allow_incomplete,a.plot_stride)
        if not a.prepare_only:result=render_figures(a.out)
    print(json.dumps({"status":result["status"],"out":str(a.out),"locked_test_metrics_read":False,"gate_eligible":False}))


if __name__=="__main__":
    main()
