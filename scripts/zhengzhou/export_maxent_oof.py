"""Export auditable conditional OOF rows without fitting or model selection."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import os
import platform
import re
import sys

import numpy as np

SEASONS = ("spring", "summer", "autumn", "winter")
VARIANTS = ("no_lst", "static_lst")
SCHEMES = ("B0_target_group", "B1_uniform", "B2_visit_density_proxy")
OFFICIAL_VERSION = "3.4.4"
OFFICIAL_JAR_SHA256 = "4c856e55412f70c5597b03cf9aaaf27e0782e0921f937262273b68bdcb8fee5e"
COMPLETE_STATUS = "OFFICIAL_MAXENT_NESTED_PIPELINE_COMPLETED_WITH_LIMITATIONS"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def verify_file(path, expected):
    if sha256(path) != expected:
        raise ValueError("Postprocessing input hash mismatch: " + str(path))


def verify_run_file(run, path, state, require_inventory=True):
    key = Path(path).relative_to(run).as_posix()
    records = {str(k).replace("\\", "/"): v for k,v in state.get("outputs", {}).items()}
    if key in records:
        verify_file(path, records[key]["sha256"])
    elif require_inventory:
        raise ValueError("Postprocessing source is missing from frozen run output inventory: " + key)


def verify_split_inventory(splits):
    manifest=read_json(splits/"manifest.json")
    required={"split_plan.json","split_masks.npz","config_snapshot.json"}
    records={str(k).replace("\\","/"):v for k,v in manifest.get("outputs",{}).items()}
    if not required.issubset(records):raise ValueError("Frozen split manifest is missing required output hashes")
    for key,record in records.items():
        path=(splits/key).resolve()
        path.relative_to(splits.resolve())
        verify_file(path,record["sha256"])
    return manifest


def resolve_migrated_project_path(recorded_path, project_root):
    """Open a historical absolute F: reference from the relocated D: project."""
    original=Path(recorded_path)
    if original.is_file():return original
    stored=str(recorded_path).replace("/","\\")
    prefix="F:\\model_coupling\\"
    if not stored.casefold().startswith(prefix.casefold()):
        raise FileNotFoundError("Recorded path is missing and outside the migrated project: "+stored)
    relative=stored[len(prefix):]
    migrated=(project_root/Path(relative.replace("\\",os.sep))).resolve()
    migrated.relative_to(project_root.resolve())
    if not migrated.is_file():
        raise FileNotFoundError("Migrated project file is missing: "+str(migrated))
    return migrated


def resolve_migrated_source_path(recorded_path, inputs):
    """Resolve pre-migration environmental raster paths for source hash checks."""
    return resolve_migrated_project_path(recorded_path,inputs.resolve().parents[3])


def verify_prepared_inputs(inputs, expected_manifest_hash):
    manifest_path=inputs/"manifest.json"
    verify_file(manifest_path,expected_manifest_hash)
    source=read_json(manifest_path)
    if not source.get("outputs"):raise ValueError("Prepared-input output hash inventory is missing")
    for key,record in source["outputs"].items():
        path=(inputs/key).resolve();path.relative_to(inputs.resolve())
        verify_file(path,record["sha256"])
    raster_records={str(path):record for path,record in source.get("inputs",{}).items() if Path(path).suffix.lower()==".tif"}
    if len(raster_records)!=60:raise ValueError("Need 60 actual source raster hash records")
    for path,record in raster_records.items():verify_file(resolve_migrated_source_path(path,inputs),record["sha256"])
    return source


def validate_oof_arrays(arrays, masks, plan):
    """The scored domain must equal outer validation union, with correct folds."""
    required = {"M_oof", "q05", "median", "q95", "std", "member_count", "fold"}
    if not required.issubset(arrays):
        raise ValueError("OOF archive is missing uncertainty/fold fields")
    shape = masks["common_valid"].shape
    size = int(np.prod(shape))
    expected_fold = np.full(size, -1, dtype=np.int16)
    for fold in plan["outer_folds"]:
        valid = masks[fold.get("validation_mask_key", f"outer_{fold['fold']}_validation")].ravel()
        if np.any((expected_fold >= 0) & valid):
            raise ValueError("Overlapping outer validation masks")
        expected_fold[valid] = int(fold["fold"])
    expected = expected_fold >= 0
    if np.any(expected & (masks["locked_test"].ravel() | ~masks["common_valid"].ravel())):
        raise ValueError("OOF domain contains locked or invalid cells")
    for key in required:
        values = np.asarray(arrays[key])
        if values.shape not in ((size,), shape):
            raise ValueError("OOF array/grid shape mismatch: " + key)
        values = values.ravel()
        if key not in ("member_count", "fold"):
            if not np.array_equal(np.isfinite(values), expected):
                raise ValueError("Incomplete/contaminated OOF domain: " + key)
            if key == "std":
                if np.any(values[expected] < 0):
                    raise ValueError("Negative member standard deviation")
            elif np.any((values[expected] < 0) | (values[expected] > 1)):
                raise ValueError("OOF cloglog score outside [0,1]")
    fold_values = np.asarray(arrays["fold"]).ravel()
    count = np.asarray(arrays["member_count"]).ravel()
    if not np.array_equal(fold_values, expected_fold):
        raise ValueError("OOF fold does not match frozen spatial split")
    if np.any(count[expected] != len(SCHEMES)) or np.any(count[~expected] != 0):
        raise ValueError("OOF member count/domain mismatch")
    q05, median, q95 = (np.asarray(arrays[k]).ravel()[expected] for k in ("q05","median","q95"))
    if np.any(q05 > median) or np.any(median > q95):
        raise ValueError("OOF member quantiles out of order")
    return expected_fold


def validate_member(provenance, fold, locked_groups, predicted_groups):
    if provenance.get("strict_end_to_end_oof") is not False or provenance.get("gate_eligible") is not False:
        raise ValueError("Historical preselection limitation must be retained")
    fit = set(provenance["fit_groups"])
    tune = set(provenance["tune_groups"])
    calibration = set(provenance.get("calibrate_groups", []))
    used = fit | tune | calibration
    declared_groups = set(fold["validation_groups"])
    if used & declared_groups or used & set(locked_groups):
        raise ValueError("OOF predicted/locked group participated in fit/tune/calibration")
    if fit != set(fold["fit_groups"]) or not tune.issubset(set(fold["fit_groups"])):
        raise ValueError("OOF provenance disagrees with frozen outer fit groups")
    if set(provenance["validation_groups"]) != declared_groups:
        raise ValueError("OOF provenance disagrees with frozen declared validation groups")
    if not set(predicted_groups).issubset(declared_groups):
        raise ValueError("OOF prediction includes a group outside the frozen validation assignment")
    if provenance.get("output_scale", "cloglog") != "cloglog" or provenance.get("background_is_absence", False) is not False:
        raise ValueError("Invalid score/background semantics")


def build_catalog(run, inputs, splits, plan, masks, seasons=SEASONS):
    model_file = run / "manifests/model_provenance.json"
    artifacts = read_json(model_file)["artifacts"]
    state=read_json(run/"manifests/run_manifest.json")
    runtime=state["runtime"]
    if runtime.get("maxent_version")!=OFFICIAL_VERSION or runtime.get("jar_sha256")!=OFFICIAL_JAR_SHA256:
        raise ValueError("Run runtime is not the pinned official MaxEnt implementation")
    source_inputs = read_json(inputs / "manifest.json")
    raster_hashes = {str(path): item["sha256"] for path,item in source_inputs["inputs"].items()
                     if Path(path).suffix.lower() == ".tif"}
    if len(raster_hashes) != 60:
        raise ValueError("Need all 60 source raster hashes for OOF lineage")
    split_hash = sha256(splits / "manifest.json")
    project_root=run.resolve()
    artifact_map = {str(resolve_migrated_project_path(a["engine_manifest"],project_root).resolve()): a
                    for a in artifacts if a["scope"].startswith("outer_")}
    catalog, refs, consumed = {}, {}, [model_file, inputs/"manifest.json", splits/"manifest.json", splits/"split_plan.json", splits/"split_masks.npz"]
    for season in seasons:
        for variant in VARIANTS:
            for fold in plan["outer_folds"]:
                fold_no = int(fold["fold"])
                member_file = run / "oof" / f"{season}_{variant}_outer_{fold_no}_members.json"
                member_info = read_json(member_file)
                verify_run_file(run,member_file,state)
                consumed.append(member_file)
                if member_info["split_sha256"] != split_hash or member_info.get("strict_end_to_end_oof") is not False or member_info.get("gate_eligible") is not False:
                    raise ValueError("OOF members use a changed split or unsupported independence claim")
                if set(member_info["validation_groups"]) != set(fold["validation_groups"]) or set(member_info["fit_and_tune_groups"]) != set(fold["fit_groups"]):
                    raise ValueError("Member group inventory differs from frozen split")
                member_paths = [str(resolve_migrated_project_path(p,project_root).resolve()) for p in member_info["members"]]
                primary = str(resolve_migrated_project_path(member_info["primary"],project_root).resolve())
                if len(member_paths) != len(SCHEMES) or len(set(member_paths)) != len(SCHEMES) or primary not in member_paths:
                    raise ValueError("Missing/duplicate OOF background members")
                declared_codes = set(map(int, fold["validation_groups"]))
                predicted_codes = sorted(map(int, np.unique(
                    masks["group_raster"][masks[f"outer_{fold_no}_validation"]]).tolist()))
                development_codes = set(map(int, np.unique(
                    masks["group_raster"][masks["development_fit"]]).tolist()))
                effective_declared_codes = declared_codes & development_codes
                if set(predicted_codes) != effective_declared_codes:
                    raise ValueError("Outer validation raster does not cover every declared group with development-domain cells")
                absent_codes = sorted(declared_codes - set(predicted_codes))
                if absent_codes:
                    # Empty/fully buffer-excluded blocks may be assigned to a fold, but
                    # they cannot silently contain held-out development samples.
                    for season_name in seasons:
                        presence_path = splits / f"presence_{season_name}.csv"
                        presence = __import__("pandas").read_csv(presence_path)
                        missing_rows = presence[
                            presence["group_code"].isin(absent_codes)
                            & (presence["outer_fold"] == fold_no)
                            & (presence["split_role"] == "development")
                        ]
                        if not missing_rows.empty:
                            raise ValueError("Declared validation group without raster domain has development presences: "
                                             + season_name + "/outer_" + str(fold_no))
                        for background_path in sorted(splits.glob(f"background_*_{season_name}.csv")):
                            background = __import__("pandas").read_csv(background_path)
                            missing_background = background[
                                background["group_code"].isin(absent_codes)
                                & (background["outer_fold"] == fold_no)
                                & (background["split_role"] == "development")
                            ]
                            if not missing_background.empty:
                                raise ValueError("Declared validation group without raster domain has development background samples: "
                                                 + background_path.name + "/outer_" + str(fold_no))
                member_records = []
                for engine_path in member_paths:
                    if engine_path not in artifact_map:
                        raise ValueError("OOF member missing from model provenance: " + engine_path)
                    artifact = artifact_map[engine_path]
                    validate_member(artifact, fold, plan["locked_groups"], predicted_codes)
                    engine_file = Path(engine_path)
                    engine_file.resolve().relative_to(run.resolve())
                    consumed.append(engine_file)
                    if artifact.get("engine_manifest_sha256"):
                        verify_file(engine_file,artifact["engine_manifest_sha256"])
                    engine = read_json(engine_file)
                    if engine["status"] != "OFFICIAL_MAXENT_FITTED" or engine["output_scale"] != "cloglog" or engine["background_is_absence"] is not False:
                        raise ValueError("OOF member is not a completed official cloglog model")
                    if engine.get("maxent_version")!=runtime["maxent_version"] or engine.get("jar_sha256")!=runtime["jar_sha256"]:
                        raise ValueError("OOF member differs from pinned run MaxEnt version/hash")
                    if engine.get("output_is_calibrated_probability") is not False:
                        raise ValueError("OOF cannot claim calibrated occurrence probability")
                    if engine["rm"] != artifact["rm"] or engine["fc"] != artifact["fc"]:
                        raise ValueError("OOF parameter provenance differs from official model")
                    selected = list(engine["inputs"]["train"]["columns"][3:])
                    if selected != artifact["predictors"]:
                        raise ValueError("OOF selected variables differ from official fit inputs")
                    html_files=list(engine_file.parent.glob("*.html"))
                    termination,iterations=None,None
                    if len(html_files)==1:
                        consumed.append(html_files[0])
                        html_record=engine.get("outputs",{}).get(html_files[0].name)
                        if html_record:verify_file(html_files[0],html_record["sha256"])
                        match=re.search(r"Algorithm (converged|terminated) after (\d+) iterations",html_files[0].read_text(encoding="utf-8",errors="replace"))
                        if match:termination,iterations=match.group(1),int(match.group(2))
                    member_records.append({**artifact, "engine_manifest_sha256":sha256(engine_file),
                        "maxent_version":engine["maxent_version"],"maxent_jar_sha256":engine["jar_sha256"],
                        "java_version":engine["java_version"],"fit_input_hashes":engine["inputs"],
                        "output_is_calibrated_probability":False,"source_raster_hashes":raster_hashes,
                        "official_html_termination":termination,"official_html_iterations":iterations,
                        "numerical_limitations":[] if termination=="converged" else ["ITERATION_LIMIT_REACHED" if termination=="terminated" else "CONVERGENCE_UNVERIFIED"]})
                if {r["background"] for r in member_records} != set(SCHEMES):
                    raise ValueError("OOF ensemble lacks the predeclared three backgrounds")
                ref = f"{season}/{variant}/outer_{fold_no}"
                catalog[ref] = {"season":season,"variant":variant,"fold":fold_no,
                    "primary":next(r for r in member_records if str(Path(r["engine_manifest"]).resolve())==primary),
                    "members":member_records,"fit_groups":fold["fit_groups"],"tune_groups":fold["fit_groups"],
                    "declared_validation_groups":sorted(declared_codes),
                    "effective_prediction_groups":predicted_codes,
                    "declared_groups_without_effective_development_pixels":absent_codes,
                    "split_manifest_sha256":split_hash,
                    "oof_scope":"CONDITIONAL_ON_PRIOR_PRESELECTED_CANDIDATES","strict_end_to_end_oof":False,
                    "gate_eligible":False,"uncertainty_is_confidence_interval":False}
                refs[(season,variant,fold_no)] = ref
    return catalog, refs, consumed


def export_oof(run, inputs, splits, out, chunk_rows=50000, seasons=SEASONS):
    started = datetime.now(timezone.utc).isoformat()
    run,inputs,splits,out = map(lambda p:Path(p).resolve(),(run,inputs,splits,out))
    if out.exists():
        raise FileExistsError("Refusing existing OOF export directory: " + str(out))
    if chunk_rows < 1:
        raise ValueError("Positive export chunk size required")
    if not seasons or len(set(seasons))!=len(seasons) or not set(seasons).issubset(SEASONS):
        raise ValueError("Export seasons must be unique declared seasons")
    state = read_json(run/"manifests/run_manifest.json")
    if state.get("status")!=COMPLETE_STATUS or set(state.get("seasons_complete",[]))!=set(SEASONS):
        raise ValueError("OOF export requires the completed four-season run; incomplete data are not full-ready")
    if state.get("strict_end_to_end_oof") is not False or state.get("gate_eligible") is not False:
        raise ValueError("Run lacks required prior-preselection limitations")
    if state.get("split_manifest_sha256") != sha256(splits/"manifest.json"):
        raise ValueError("Run split hash differs from supplied frozen split")
    verify_prepared_inputs(inputs,state.get("input_manifest_sha256"))
    verify_split_inventory(splits)
    plan = read_json(splits/"split_plan.json")
    with np.load(splits/"split_masks.npz",allow_pickle=False) as archive:
        masks = {k:archive[k] for k in ("common_valid","development_fit","group_raster","locked_group_buffer","locked_test",*[f"outer_{f['fold']}_validation" for f in plan["outer_folds"]])}
    catalog, refs, consumed = build_catalog(run,inputs,splits,plan,masks,seasons)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"provenance_catalog.json",catalog)
    group_names = {int(r["group_code"]):r["group_id"] for r in plan["group_records"]}
    counts = {}
    for season in seasons:
        for variant in VARIANTS:
            path = run/"oof"/f"{season}_{variant}.npz"
            verify_run_file(run,path,state)
            consumed.append(path)
            with np.load(path,allow_pickle=False) as archive:
                arrays = {k:archive[k].ravel() for k in ("M_oof","q05","median","q95","std","member_count","fold")}
            expected_fold = validate_oof_arrays(arrays,masks,plan)
            ids = np.flatnonzero(expected_fold>=0)
            destination = out/f"{season}_{variant}_conditional_oof.csv.gz"
            with gzip.open(destination,"wt",encoding="utf-8",newline="",compresslevel=6) as stream:
                writer = csv.writer(stream)
                writer.writerow(["native_cell_id","season","variant","spatial_group","group_code","fold",
                    "raster_row","raster_col","M_oof","U_M","q05","median","q95","std","member_count",
                    "primary_provenance_ref","oof_scope","gate_eligible"])
                for offset in range(0,len(ids),chunk_rows):
                    block = ids[offset:offset+chunk_rows]
                    rows,cols = np.divmod(block,plan["grid"]["width"])
                    codes = masks["group_raster"].ravel()[block]
                    fold = expected_fold[block]
                    records = ([f"utm49_r{r}_c{c}",season,variant,group_names[int(g)],int(g),int(f),int(r),int(c),
                        *(format(float(arrays[k][i]),".10g") for k in ("M_oof","std","q05","median","q95","std")),
                        int(arrays["member_count"][i]),refs[(season,variant,int(f))],
                        "CONDITIONAL_ON_PRIOR_PRESELECTED_CANDIDATES",False]
                        for i,r,c,g,f in zip(block,rows,cols,codes,fold))
                    writer.writerows(records)
            counts[season+"/"+variant] = len(ids)
            del arrays,expected_fold
    consumed.append(run/"manifests/run_manifest.json")
    inventory = {str(p):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in dict.fromkeys(consumed)}
    manifest = {"status":"CONDITIONAL_OOF_GZ_CSV_EXPORTED_AND_GROUP_LINEAGE_VALIDATED",
        "started_at_utc":started,"ended_at_utc":datetime.now(timezone.utc).isoformat(),
        "python":sys.version,"platform":platform.platform(),"command_line":sys.argv,
        "git_commit":state.get("git_commit"),"random_seed":state.get("seed"),
        "split_manifest_sha256":sha256(splits/"manifest.json"),"config_sha256":state.get("config_sha256"),
        "maxent_runtime":state.get("runtime"),"rows":counts,"seasons_exported":list(seasons),
        "full_four_seasons_complete":set(seasons)==set(SEASONS),"strict_end_to_end_oof":False,"gate_eligible":False,
        "background_is_absence":False,"output_scale":"cloglog","calibrated_occurrence_probability":False,
        "uncertainty_is_confidence_interval":False,"source_files_modified":False,"sensitive_coordinates_private":True,
        "inputs":inventory,"outputs":{str(p.relative_to(out)):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in out.iterdir() if p.is_file()}}
    write_json(out/"manifest.json",manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run","inputs","splits","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--chunk-rows",type=int,default=50000)
    p.add_argument("--seasons",nargs="+",choices=SEASONS,default=list(SEASONS))
    args = p.parse_args()
    result = export_oof(args.run,args.inputs,args.splits,args.out,args.chunk_rows,tuple(args.seasons))
    print(json.dumps({"status":result["status"],"rows":result["rows"],"gate_eligible":False}))


if __name__=="__main__":
    main()
