"""Execute frozen, buffered, nested official-MaxEnt community pipeline V1.

No gate, HQ fitting, ecological absence labels or calibrated probabilities.
OOF is conditional on historically preselected covariates, explicitly recorded.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
from wetland_coupling.maxent_engine import check_jar, run_jobs
from wetland_coupling.maxent_inputs import dataframe_cells, sample_background
from wetland_coupling.maxent_splits import load_split_plan
from wetland_coupling.maxent_protocol import (sha256, write_json, metrics, select_predictors,
    aggregate_candidates, choose_candidate, assert_oof_groups, freeze_selection, claim_locked_test)

SEASONS = ("spring", "summer", "autumn", "winter")
VARIANTS = ("no_lst", "static_lst")
SCHEMES = ("B0_target_group", "B1_uniform", "B2_visit_density_proxy")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def seeded(base, name):
    return (int(base) + int(hashlib.sha256(name.encode()).hexdigest()[:8],16)) % (2**32)


def mask_rows(frame, mask):
    return frame.loc[mask[frame.raster_row.to_numpy(int),frame.raster_col.to_numpy(int)]].reset_index(drop=True)


def swd(path, frame, names, species):
    table=frame[["longitude","latitude",*names]].copy()
    table.insert(0,"species",species)
    table.to_csv(path,index=False,float_format="%.12g")


def context(inputs, season):
    root=inputs/season
    return {"env":np.load(root/"env.npy",mmap_mode="r"),
            "names":json.loads((root/"names.json").read_text(encoding="utf-8")),
            "grid":json.loads((inputs/"grid.json").read_text(encoding="utf-8")),
            "valid_mask":np.load(inputs/"common_valid_mask.npy",mmap_mode="r"),
            "presence":pd.read_csv(root/"presence.csv"),"B0":pd.read_csv(root/"B0.csv"),
            "reference":pd.read_csv(root/"B1.csv"),"visits":pd.read_csv(root/"visit_cells.csv")}


def prepare_fit(scope, ctx, fit_mask, val_mask, background, variant, config, seed, full_projection=False):
    scope.mkdir(parents=True,exist_ok=False)
    train_p=mask_rows(ctx["presence"],fit_mask)
    train_b=sample_background(background,fit_mask,ctx,ctx["visits"],ctx["B0"],config,seed)
    min_counts=config["spatial"]
    if len(train_p)<min_counts["minimum_fit_presences"] or len(train_b)<min_counts["minimum_background"]:
        raise ValueError("Insufficient buffered fit counts: "+str(scope))
    names=[n for n in ctx["names"] if variant=="static_lst" or not n.startswith("lst_")]
    selected,dropped=select_predictors(train_b,names,config["predictors"]["absolute_spearman_cutoff"])
    swd(scope/"train.csv",train_p,selected,"waterbird_community_candidate")
    swd(scope/"background.csv",train_b,selected,"background")
    train_reference=mask_rows(ctx["reference"],fit_mask)
    val_p=mask_rows(ctx["presence"],val_mask)
    val_b=mask_rows(ctx["reference"],val_mask)
    if len(val_p)<min_counts["minimum_validation_presences"] or len(val_b)<min_counts["minimum_background"]:
        raise ValueError("Insufficient validation counts: "+str(scope))
    pieces=[train_p,train_reference,val_p,val_b]
    offsets=np.cumsum([0,*[len(f) for f in pieces]]).tolist()
    swd(scope/"projection.csv",pd.concat(pieces,ignore_index=True),selected,"evaluation")
    projection_start=offsets[-1]
    indices=np.flatnonzero(val_mask) if full_projection else np.empty(0,dtype=np.int64)
    if full_projection:
        with (scope/"projection.csv").open("a",encoding="utf-8",newline="") as stream:
            for start in range(0,len(indices),50_000):
                frame=dataframe_cells(indices[start:start+50_000],ctx["env"],ctx["names"],ctx["grid"])
                frame=frame[["longitude","latitude",*selected]].copy()
                frame.insert(0,"species","projection")
                frame.to_csv(stream,index=False,header=False,float_format="%.12g")
    meta={"predictors":selected,"dropped_predictors":dropped,"offsets":offsets,
          "projection_start":projection_start,"raster_indices":indices,
          "background_audit":train_b.attrs.get("background_audit",{}),
          "train_presence_n":len(train_p),"train_background_n":len(train_b)}
    write_json(scope/"selection.json",{k:v for k,v in meta.items() if k!="raster_indices"})
    return meta


def final_feature_plan(ctx, fit_mask, background, variant, config, seed):
    """Freeze the exact final fit-only predictor and background choices before test."""
    train_b=sample_background(background,fit_mask,ctx,ctx["visits"],ctx["B0"],config,seed)
    names=[n for n in ctx["names"] if variant=="static_lst" or not n.startswith("lst_")]
    selected,dropped=select_predictors(train_b,names,config["predictors"]["absolute_spearman_cutoff"])
    digest=hashlib.sha256(pd.util.hash_pandas_object(train_b,index=False).to_numpy().tobytes()).hexdigest()
    return {"predictors":selected,"dropped_predictors":dropped,"background_frame_sha256":digest,
            "background_audit":train_b.attrs.get("background_audit",{}),"sampling_seed":seed}


def job(scope,rm,fc,seed,args,config,suffix):
    return dict(java=args.java,jar=args.jar,train_csv=scope/"train.csv",background_csv=scope/"background.csv",
                projection_csv=scope/"projection.csv",out_dir=scope/suffix,rm=rm,fc=fc,seed=seed,
                max_iterations=config["maxent"]["max_iterations"],timeout=300)


def result_metrics(result,meta):
    values=result["predictions"];o=meta["offsets"]
    return metrics(values[o[0]:o[1]],values[o[1]:o[2]],values[o[2]:o[3]],values[o[3]:o[4]],result["complexity"])


def tune_scope(run,season,label,ctx,inner_folds,config,args):
    rows=[]
    for inner_index,fold in enumerate(inner_folds):
        print(json.dumps({"phase":"inner_tuning","season":season,"scope":label,"inner":inner_index,"time":stamp()}),flush=True)
        for background in SCHEMES:
            for variant in VARIANTS:
                seed=seeded(config["seed"],f"{season}/{label}/{inner_index}/{background}")
                path=run/variant/season/label/f"inner_{inner_index}"/background
                meta=prepare_fit(path,ctx,fold["fit_mask"],fold["validation_mask"],background,variant,config,seed)
                jobs=[job(path,rm,fc,seed,args,config,f"rm{rm:g}_{fc}") for rm in config["maxent"]["rm"] for fc in config["maxent"]["fc"]]
                results=run_jobs(jobs,max_workers=config["maxent"]["parallel_workers"])
                for item,result in zip(jobs,results):
                    row={"season":season,"scope":label,"inner_fold":inner_index,"variant":variant,
                         "background":background,"rm":item["rm"],"fc":item["fc"],**result_metrics(result,meta),
                         "predictors":";".join(meta["predictors"]),"fit_groups":fold["fit_groups"],
                         "validation_groups":fold["validation_groups"],"artifact":result["manifest_path"]}
                    rows.append(row)
                # Save each completed scope so interruption never invents completeness.
                pd.DataFrame(rows).to_csv(run/"reports"/f"inner_{season}_{label}.csv",index=False)
    aggregate=aggregate_candidates(rows)
    winners={variant:{background:choose_candidate([r for r in aggregate if r["variant"]==variant and r["background"]==background],config["selection"]["mean_omission_eligibility_max"])
                      for background in SCHEMES} for variant in VARIANTS}
    chosen={variant:choose_candidate([r for r in aggregate if r["variant"]==variant],config["selection"]["mean_omission_eligibility_max"]) for variant in VARIANTS}
    write_json(run/"reports"/f"tuning_{season}_{label}.json",{"all_candidates":aggregate,"scheme_winners":winners,"variant_winners":chosen})
    return winners,chosen,aggregate


def fitted_member(run,season,label,ctx,fold,variant,candidate,config,args,full_projection):
    background=candidate["background"]
    seed=seeded(config["seed"],f"{season}/{label}/refit/{background}")
    path=run/variant/season/label/"refit"/background
    meta=prepare_fit(path,ctx,fold["fit_mask"],fold["validation_mask"],background,variant,config,seed,full_projection)
    result=run_jobs([job(path,candidate["rm"],candidate["fc"],seed,args,config,"official_model")],max_workers=1)[0]
    provenance={"season":season,"scope":label,"variant":variant,"background":background,
                "rm":candidate["rm"],"fc":candidate["fc"],"predictors":meta["predictors"],
                "fit_groups":fold["fit_groups"],"tune_groups":fold["fit_groups"],
                "validation_groups":fold["validation_groups"],"calibrate_groups":[],
                "engine_manifest":result["manifest_path"],"engine_manifest_sha256":sha256(result["manifest_path"]),
                "output_scale":"cloglog","background_is_absence":False,
                "candidate_provenance":config["predictors"]["candidate_provenance"],
                "strict_end_to_end_oof":False,"gate_eligible":False}
    write_json(path/"provenance.json",provenance)
    return result,meta,provenance


def save_raster(path,flat_values,ctx):
    import rasterio
    from affine import Affine
    grid=ctx["grid"]
    # Rasterio six-coefficient transform is recorded by formal-input builder.
    transform=Affine(*grid["transform"][:6])
    profile=dict(driver="GTiff",height=ctx["env"].shape[0],width=ctx["env"].shape[1],count=1,
                 dtype="float32",crs="EPSG:32649",transform=transform,nodata=-9999,
                 compress="deflate",tiled=True)
    arr=np.asarray(flat_values,dtype=np.float32).reshape(ctx["env"].shape[:2])
    with rasterio.open(path,"w",**profile) as dst:
        dst.write(np.where(np.isfinite(arr),arr,-9999).astype(np.float32),1)
        dst.update_tags(output_scale="cloglog",calibrated_probability="false",scope="candidate_community",
                        oof_condition="prior_preselected_covariates",gate_eligible="false")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True);parser.add_argument("--splits",type=Path,required=True)
    parser.add_argument("--java",required=True);parser.add_argument("--jar",required=True)
    args=parser.parse_args();run=args.run;config=json.loads(args.config.read_text(encoding="utf-8"))
    marker=run/"manifests"/"TRAINING_ATTEMPT.json"
    marker.parent.mkdir(parents=True,exist_ok=True)
    with marker.open("x",encoding="utf-8") as stream:json.dump({"started_at_utc":stamp()},stream)
    for name in (*VARIANTS,"oof","predictions","reports","figures"):(run/name).mkdir(exist_ok=True)
    inputs_manifest=json.loads((args.inputs/"manifest.json").read_text(encoding="utf-8"))
    # Fail closed if any formally prepared input has been edited since preparation.
    for name,item in inputs_manifest["outputs"].items():
        if sha256(args.inputs/name)!=item["sha256"]:raise ValueError("Formal input hash changed: "+name)
    plan=load_split_plan(args.splits)
    split_hash=sha256(args.splits/"manifest.json");config_hash=sha256(args.config)
    split_manifest=json.loads((args.splits/"manifest.json").read_text(encoding="utf-8"))
    if config_hash!=split_manifest["provenance"]["config_file_sha256"] or config_hash!=inputs_manifest["config_sha256"]:
        raise ValueError("Current protocol differs from frozen split or formal-input configuration")
    if sha256(args.inputs/"manifest.json")!=split_manifest["provenance"]["input_manifest_sha256"]:
        raise ValueError("Frozen split was built from a different formal-input manifest")
    runtime=check_jar(args.java,args.jar)
    git=subprocess.run(["git","-c","safe.directory=F:/model_coupling","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    write_json(run/"manifests"/"config_snapshot.json",config)
    state={"status":"RUNNING","started_at_utc":stamp(),"git_commit":git,"python":platform.python_version(),
           "argv":sys.argv,"runtime":runtime,"seed":config["seed"],"config_sha256":config_hash,
           "split_manifest_sha256":split_hash,"input_manifest_sha256":sha256(args.inputs/"manifest.json"),
           "strict_end_to_end_oof":False,"gate_eligible":False,"locked_test_used":False,"seasons_complete":[]}
    state["source_raster_hashes"]={name:item for name,item in inputs_manifest["inputs"].items() if name.lower().endswith(".tif")}
    if len(state["source_raster_hashes"])!=60:
        raise ValueError("Expected all 60 verified source-raster hashes")
    state["canonical_split_sha256"]=split_manifest["split_hash"]
    project_root=Path(__file__).resolve().parents[2]
    state["source_code_sha256"]={str(p.relative_to(project_root)):sha256(p) for p in
        [Path(__file__).resolve(),project_root/"scripts/zhengzhou/MaxentBatch.java",*
         (project_root/"wetland_coupling").glob("maxent_*.py")]}
    state["input_preparation_config_sha256"]=inputs_manifest["config_sha256"]
    state["input_preparation_preceded_spatial_decision"]=inputs_manifest["config_sha256"]!=config_hash
    write_json(run/"manifests"/"run_manifest.json",state)
    outer_rows=[];final_selections={};final_contexts={};artifacts=[]
    for season in SEASONS:
        ctx=context(args.inputs,season);size=ctx["env"].shape[0]*ctx["env"].shape[1]
        oof={v:{"M_oof":np.full(size,np.nan),"q05":np.full(size,np.nan),"median":np.full(size,np.nan),
                "q95":np.full(size,np.nan),"std":np.full(size,np.nan),"member_count":np.zeros(size,dtype=np.int16),
                "fold":np.full(size,-1,dtype=np.int16)} for v in VARIANTS}
        for fold in plan["outer_folds"]:
            label=f"outer_{fold['fold']}"
            assert_oof_groups(fold["validation_groups"],fold["fit_groups"],fold["fit_groups"],plan["locked_groups"])
            winners,chosen,_=tune_scope(run,season,label,ctx,fold["inner_folds"],config,args)
            for variant in VARIANTS:
                members=[];member_ids=[];indices=None
                for background in SCHEMES:
                    result,meta,provenance=fitted_member(run,season,label,ctx,fold,variant,winners[variant][background],config,args,True)
                    if indices is not None and not np.array_equal(indices,meta["raster_indices"]):raise ValueError("Member projection domains differ")
                    indices=meta["raster_indices"];members.append(result["predictions"][meta["projection_start"]:]);member_ids.append(provenance["engine_manifest"])
                    measure=result_metrics(result,meta)
                    outer_rows.append({"season":season,"fold":fold["fold"],"variant":variant,"background":background,
                                       "rm":winners[variant][background]["rm"],"fc":winners[variant][background]["fc"],
                                       "chosen_background":background==chosen[variant]["background"],**measure,
                                       "predictors":";".join(meta["predictors"]),"artifact":provenance["engine_manifest"]})
                    artifacts.append(provenance)
                values=np.stack(members)
                out=oof[variant];primary_index=list(SCHEMES).index(chosen[variant]["background"])
                out["M_oof"][indices]=values[primary_index]
                for key,vals in zip(("q05","median","q95"),np.quantile(values,[.05,.5,.95],axis=0)):out[key][indices]=vals
                out["std"][indices]=values.std(axis=0,ddof=0);out["member_count"][indices]=len(members);out["fold"][indices]=fold["fold"]
                write_json(run/"oof"/f"{season}_{variant}_{label}_members.json",{"members":member_ids,"primary":member_ids[primary_index],
                    "validation_groups":fold["validation_groups"],"fit_and_tune_groups":fold["fit_groups"],
                    "split_sha256":split_hash,"strict_end_to_end_oof":False,"gate_eligible":False})
            pd.DataFrame(outer_rows).to_csv(run/"reports"/"outer_metrics.csv",index=False)
        for variant,values in oof.items():
            np.savez_compressed(run/"oof"/f"{season}_{variant}.npz",**values)
            save_raster(run/"oof"/f"{season}_{variant}_M_oof.tif",values["M_oof"],ctx)
            save_raster(run/"oof"/f"{season}_{variant}_U_M.tif",values["std"],ctx)
        _,chosen,aggregate=tune_scope(run,season,"full_development",ctx,plan["final_inner_folds"],config,args)
        joint=choose_candidate(aggregate,config["selection"]["mean_omission_eligibility_max"])
        final_features={}
        for variant in VARIANTS:
            candidate=chosen[variant]
            seed=seeded(config["seed"],f"{season}/frozen_final/{candidate['background']}")
            final_features[variant]=final_feature_plan(ctx,plan["development_fit_mask"],candidate["background"],variant,config,seed)
        final_selections[season]={"variant_winners":chosen,"deployment_choice":joint,"final_features":final_features}
        state["seasons_complete"].append(season);write_json(run/"manifests"/"run_manifest.json",state)
    frozen=run/"manifests"/"frozen_selection.json"
    selection_hash=freeze_selection(frozen,final_selections,split_hash,config_hash)
    # This atomically claims the only final test batch, before creating any test predictions.
    claim_locked_test(run/"manifests",frozen,selection_hash,split_hash,config_hash)
    final_rows=[]
    for season in SEASONS:
        ctx=context(args.inputs,season)
        fold={"fit_mask":plan["development_fit_mask"],"validation_mask":plan["locked_mask"],
              "fit_groups":plan["development_groups"],"validation_groups":plan["locked_groups"]}
        for variant in VARIANTS:
            candidate=final_selections[season]["variant_winners"][variant]
            print(json.dumps({"phase":"frozen_final_test_batch","season":season,"variant":variant,"time":stamp()}),flush=True)
            # Evaluation rows only locked test; append full common domain for the final map.
            scope=run/variant/season/"frozen_final"/"refit"/candidate["background"]
            seed=seeded(config["seed"],f"{season}/frozen_final/{candidate['background']}")
            meta=prepare_fit(scope,ctx,fold["fit_mask"],fold["validation_mask"],candidate["background"],variant,config,seed,False)
            actual=final_feature_plan(ctx,fold["fit_mask"],candidate["background"],variant,config,seed)
            if actual!=final_selections[season]["final_features"][variant] or meta["predictors"]!=actual["predictors"]:
                raise ValueError("Final fit-only features/background changed after freeze")
            indices=np.flatnonzero(ctx["valid_mask"])
            with (scope/"projection.csv").open("a",encoding="utf-8",newline="") as stream:
                for start in range(0,len(indices),50_000):
                    f=dataframe_cells(indices[start:start+50_000],ctx["env"],ctx["names"],ctx["grid"])
                    f=f[["longitude","latitude",*meta["predictors"]]].copy();f.insert(0,"species","projection")
                    f.to_csv(stream,index=False,header=False,float_format="%.12g")
            result=run_jobs([job(scope,candidate["rm"],candidate["fc"],seed,args,config,"official_model")],max_workers=1)[0]
            final_rows.append({"season":season,"variant":variant,"background":candidate["background"],"rm":candidate["rm"],"fc":candidate["fc"],**result_metrics(result,meta)})
            arr=np.full(ctx["env"].shape[0]*ctx["env"].shape[1],np.nan);arr[indices]=result["predictions"][meta["projection_start"]:]
            save_raster(run/"predictions"/f"{season}_{variant}_cloglog.tif",arr,ctx)
            np.save(run/"predictions"/f"{season}_{variant}_cloglog.npy",arr.astype(np.float32))
            artifacts.append({"season":season,"scope":"frozen_final","variant":variant,"background":candidate["background"],"rm":candidate["rm"],"fc":candidate["fc"],
                              "predictors":meta["predictors"],"fit_groups":fold["fit_groups"],"tune_groups":fold["fit_groups"],"validation_groups":fold["validation_groups"],
                              "engine_manifest":result["manifest_path"],"strict_end_to_end_oof":False,"gate_eligible":False})
    pd.DataFrame(final_rows).to_csv(run/"reports"/"locked_test_metrics.csv",index=False)
    write_json(run/"manifests"/"model_provenance.json",{"artifacts":artifacts,"source_rasters":state["source_raster_hashes"],
                                                       "split_sha256":split_hash,"config_sha256":config_hash})
    state.update(status="OFFICIAL_MAXENT_NESTED_PIPELINE_COMPLETED_WITH_LIMITATIONS",ended_at_utc=stamp(),
                 locked_test_used=True,frozen_selection_sha256=selection_hash,outer_metrics=outer_rows,locked_metrics=final_rows,
                 oof_ready="CONDITIONAL_ON_PRIOR_PRESELECTED_CANDIDATES",u_m_ready=True,confidence_intervals=False)
    state["outputs"]={str(p.relative_to(run)):{"sha256":sha256(p),"bytes":p.stat().st_size} for folder in ("oof","predictions","reports") for p in (run/folder).rglob("*") if p.is_file()}
    write_json(run/"manifests"/"run_manifest.json",state)
    print(json.dumps({"status":state["status"],"locked_test_used":True,"oof_kind":state["oof_ready"]}),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        # Never remove failed artifacts or the exclusive final-test attempt marker.
        if "--run" in sys.argv:
            failed_run=Path(sys.argv[sys.argv.index("--run")+1])
            manifest=failed_run/"manifests"/"run_manifest.json"
            if manifest.parent.exists():
                failed=json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
                failed.update(status="FAILED_OR_INCOMPLETE",failed_at_utc=stamp(),failure=repr(error),
                              locked_test_attempted=(failed_run/"manifests"/"LOCKED_TEST_ATTEMPT.json").exists())
                write_json(manifest,failed)
        raise
