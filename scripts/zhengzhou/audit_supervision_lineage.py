"""Audit candidate survey lineage against frozen inputs; create no labels.

No predictions or test metrics are opened. ND records may inform a future
cross-fitted response after protocol review; they are not new independent tests.
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


SEASONS = ("spring", "summer", "autumn", "winter")
LEDGERS = {
    "nd": "zhengzhou_nd_original_records_with_audit_flags.csv",
    "monitoring": "monitoring_csv_records_with_audit_flags.csv",
    "zhuque": "zhuque_xlsx_original_records_with_audit_flags.csv",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def verify_file(path, expected, inventory):
    path = Path(path).resolve()
    actual = digest(path)
    if actual != expected:
        raise ValueError("Supervision lineage input hash mismatch: " + str(path))
    inventory[str(path)] = {"sha256": actual, "bytes": path.stat().st_size}


def verified_output(directory, manifest, name, inventory):
    records = {str(k).replace("\\", "/"): v for k, v in manifest["outputs"].items()}
    if name not in records:
        raise ValueError("Required source output has no hash: " + name)
    path = (Path(directory) / name).resolve()
    path.relative_to(Path(directory).resolve())
    verify_file(path, records[name]["sha256"], inventory)
    return path


def verify_recorded_source(manifest, path, inventory):
    """Match relocated sources by their project-local suffix, never filename alone."""
    path = Path(path).resolve()
    project = Path(__file__).resolve().parents[2]
    suffix = path.relative_to(project).as_posix().casefold()
    matching = [v for k, v in manifest["inputs"].items()
                if str(k).replace("\\", "/").casefold().endswith("/" + suffix)]
    if len(matching) != 1:
        raise ValueError("Source is not uniquely anchored in frozen input inventory: " + str(path))
    verify_file(path, matching[0]["sha256"], inventory)


def validate_links(events, links, source_frames):
    if not events.event_candidate_id.is_unique:
        raise ValueError("Candidate event IDs are not unique")
    if links.record_id.isna().any() or not links.record_id.is_unique:
        raise ValueError("Record links must assign each original record to one event")
    event_source = events.set_index("event_candidate_id").source_kind
    if not set(links.event_candidate_id).issubset(event_source.index):
        raise ValueError("Record link points to an unknown event")
    for source, frame in source_frames.items():
        if frame.record_id.isna().any() or not frame.record_id.is_unique:
            raise ValueError("Original ledger record IDs are missing or duplicated")
        used = links.loc[links.event_candidate_id.map(event_source).eq(source)]
        if set(used.record_id) != set(frame.record_id):
            raise ValueError("Candidate links differ from original ledger IDs: " + source)
        joined = used.merge(frame[["record_id", "source_file", "source_sheet", "source_row"]],
                            on="record_id", suffixes=("_link", "_ledger"), validate="one_to_one")
        for field in ("source_file", "source_sheet", "source_row"):
            if not joined[field + "_link"].fillna("").astype(str).equals(
                    joined[field + "_ledger"].fillna("").astype(str)):
                raise ValueError("Original source-row lineage mismatch: " + field)
        counts = used.groupby("event_candidate_id").size()
        expected = events.loc[events.source_kind.eq(source)].set_index("event_candidate_id").source_record_count
        if not counts.sort_index().equals(expected.sort_index().astype("int64")):
            raise ValueError("Candidate event record count mismatch: " + source)


def join_memberships(links, ledger, members, presences):
    """Keep record-level roles, including unmatched cells; never pool seasons."""
    chunks = []
    metadata = ledger[["record_id", "season_calendar"]]
    for season in SEASONS:
        member, presence = members[season], presences[season]
        if not member.record_id.is_unique or not presence.native_cell_id.is_unique:
            raise ValueError("Duplicate record/cell in frozen membership or presence: " + season)
        if not set(member.record_id).issubset(ledger.record_id):
            raise ValueError("Membership has an unknown original record ID")
        joined = member.merge(metadata, on="record_id", validate="one_to_one")
        if not joined.season_calendar.eq(season).all():
            raise ValueError("Membership season differs from original observation season")
        joined = joined.merge(links[["record_id", "event_candidate_id"]], on="record_id", validate="one_to_one")
        if len(joined) != len(member):
            raise ValueError("Original member record has no candidate event link")
        joined = joined.merge(presence[["native_cell_id", "split_role", "group_id", "outer_fold"]],
                              on="native_cell_id", how="left", validate="many_to_one")
        joined["season"] = season
        joined["split_role"] = joined.split_role.fillna("outside_frozen_presence")
        chunks.append(joined)
    return pd.concat(chunks, ignore_index=True)


def unique_values(values):
    return sorted(set(values.dropna().astype(str)))


def summarize_memberships(records):
    rows = []
    for event_id, group in records.groupby("event_candidate_id", sort=True):
        cells = unique_values(group.native_cell_id)
        roles = unique_values(group.split_role)
        spatial_groups = unique_values(group.group_id)
        folds = sorted(set(int(v) for v in group.outer_fold.dropna()))
        conflict = len(cells) != 1 or len(roles) != 1 or len(spatial_groups) > 1 or len(folds) > 1
        rows.append({
            "event_candidate_id": event_id,
            "presence_member_record_count": len(group),
            "presence_native_cells_json": json.dumps(cells),
            "presence_roles_json": json.dumps(roles),
            "presence_groups_json": json.dumps(spatial_groups),
            "presence_outer_folds_json": json.dumps(folds),
            "presence_role_unique": roles[0] if not conflict else "MULTIPLE_OR_CONFLICTING_REVIEW_REQUIRED",
            "presence_lineage_conflict": conflict,
            "has_development_presence_source": "development" in roles,
            "has_locked_test_presence_source": "locked_internal_test" in roles,
            "has_buffer_excluded_presence_source": "locked_buffer_excluded" in roles,
        })
    return pd.DataFrame(rows)


def build_review_queue(events, membership_summary):
    columns = ["event_candidate_id", "source_kind", "source_file", "source_sheet", "source_record_count",
               "site_candidate_id", "site_name_raw", "start_time", "end_time", "calendar_year", "season", "season_year",
               "source_crs", "crs_evidence", "event_id_status", "protocol", "complete_target_checklist",
               "observed_waterbird_species_names_n", "taxonomy_unmatched_rows", "possible_duplicate_rows",
               "cross_event_collision_status"]
    result = events[columns].merge(membership_summary, on="event_candidate_id", how="left", validate="one_to_one")
    for field in ("has_development_presence_source", "has_locked_test_presence_source",
                  "has_buffer_excluded_presence_source", "presence_lineage_conflict"):
        result[field] = result[field].astype("boolean").fillna(False).astype(bool)
    result["presence_member_record_count"] = pd.to_numeric(result.presence_member_record_count).fillna(0).astype(int)
    for field in ("presence_native_cells_json", "presence_roles_json", "presence_groups_json", "presence_outer_folds_json"):
        result[field] = result[field].fillna("[]")
    result["presence_role_unique"] = result.presence_role_unique.fillna("NO_POSITIVE_WATERBIRD_PRESENCE_MEMBERSHIP")
    result.loc[~result.source_kind.eq("nd"), "presence_role_unique"] = "NOT_APPLICABLE_NON_ND_BASE_MEMBERSHIP"
    result["same_nd_source_used_by_base_model"] = result.source_kind.eq("nd")
    result["source_independence_status"] = np.where(result.source_kind.eq("nd"),
            "SAME_ORIGINAL_ND_LEDGER_AS_BASE_MODEL", "UNKNOWN_SOURCE_INDEPENDENCE_REQUIRES_REVIEW")
    result["historical_2025_already_viewed"] = result.calendar_year.eq(2025)
    result["temporal_status"] = np.where(result.calendar_year.eq(2025), "2025_ALREADY_VIEWED_NOT_FRESH_BLIND_TEST",
            np.where(result.source_kind.eq("zhuque"), "2026_APRIL_ONLY_PROTOCOL_AND_INDEPENDENCE_UNCONFIRMED",
                     "NOT_REGISTERED_AS_NEW_INDEPENDENT_TEST"))
    result["no_recorded_waterbirds_is_confirmed_non_detection"] = False
    result["eligible_for_supervised_training"] = False
    result["target_protection"] = np.nan
    result["target_restoration"] = np.nan
    result["required_next_review"] = np.where(result.source_kind.eq("nd"),
            "event_and_duplicates;taxonomy;verified_protocol_effort;independent_response_definition;future_cross_fit_exclusions",
            "source_independence;source_crs;event_and_duplicates;taxonomy;verified_protocol_effort;independent_response_definition")
    result.loc[result.presence_lineage_conflict, "required_next_review"] += ";multiple_cells_or_roles_do_not_force_unique_group"
    return result


def audit(args):
    started = datetime.now(timezone.utc).isoformat()
    if args.out.exists():
        raise FileExistsError(args.out)
    inventory = {}
    source = read_json(args.candidates / "manifest.json")
    if source.get("labels_created") != 0:
        raise ValueError("This audit expects unlabeled evidence candidates")
    inventory[str((args.candidates / "manifest.json").resolve())] = {
        "sha256": digest(args.candidates / "manifest.json"), "bytes": (args.candidates / "manifest.json").stat().st_size}
    for name in source["outputs"]:
        verified_output(args.candidates, source, name, inventory)
    frames = {}
    ledger_columns = ["record_id", "source_file", "source_sheet", "source_row", "season_calendar"]
    for kind, filename in LEDGERS.items():
        path = args.birds / filename
        verify_file(path, source["sources"][kind]["input_sha256"], inventory)
        frames[kind] = pd.read_csv(path, usecols=ledger_columns, low_memory=False)
    events = pd.read_csv(args.candidates / "survey_events_candidate.csv", low_memory=False)
    links = pd.read_csv(args.candidates / "event_record_links.csv", low_memory=False)
    if events[["target_protection", "target_restoration"]].notna().any().any() or events.eligible_for_supervised_training.any():
        raise ValueError("Candidate table contains targets or training eligibility; review separately")
    if len(events) != source["total_candidate_events"]:
        raise ValueError("Candidate manifest event count mismatch")
    validate_links(events, links, frames)
    split = read_json(args.splits / "manifest.json")
    verify_file(args.inputs / "manifest.json", split["provenance"]["input_manifest_sha256"], inventory)
    inputs = read_json(args.inputs / "manifest.json")
    verify_recorded_source(inputs, args.members / "manifest.json", inventory)
    verify_recorded_source(inputs, args.observations / "manifest.json", inventory)
    verify_recorded_source(inputs, args.birds / LEDGERS["nd"], inventory)
    members_manifest = read_json(args.members / "manifest.json")
    observation = read_json(args.observations / "manifest.json")
    verify_file(args.observations / "manifest.json", members_manifest["source_observation_manifest_sha256"], inventory)
    verify_file(args.birds / LEDGERS["nd"], observation["input_ledger_sha256"], inventory)
    if observation.get("confirmed_source_crs") != "EPSG:4326" or not observation.get("coordinate_reference_confirmed"):
        raise ValueError("ND coordinate confirmation is missing")
    # Verify the entire small frozen split inventory, including masks; read only presence role columns.
    inventory[str((args.splits / "manifest.json").resolve())] = {
        "sha256": digest(args.splits / "manifest.json"), "bytes": (args.splits / "manifest.json").stat().st_size}
    for name in split["outputs"]:
        verified_output(args.splits, split, name, inventory)
    member_frames, presence_frames = {}, {}
    for season in SEASONS:
        path = verified_output(args.members, members_manifest, "source_membership_" + season + ".csv", inventory)
        member_frames[season] = pd.read_csv(path, low_memory=False)
        presence_frames[season] = pd.read_csv(args.splits / ("presence_" + season + ".csv"),
                    usecols=["native_cell_id", "split_role", "group_id", "outer_fold"], low_memory=False)
    records = join_memberships(links, frames["nd"], member_frames, presence_frames)
    grouped = summarize_memberships(records)
    queue = build_review_queue(events, grouped)
    args.out.mkdir(parents=True)
    command_config = {
        "parameters": {key: str(value.resolve()) for key, value in vars(args).items()},
        "audit_schema_version": "supervision_lineage_v2",
        "membership_scope": "ORIGINAL_ND_POSITIVE_WATERBIRD_SOURCE_RECORDS_ONLY",
        "non_nd_membership_status": "NOT_APPLICABLE_NON_ND_BASE_MEMBERSHIP",
        "labels_created": 0, "training_eligible_events": 0,
        "source_crs_for_non_nd": "UNCONFIRMED_NOT_TRANSFORMED",
        "model_outputs_read": False, "locked_test_metrics_read": False,
    }
    write_json(args.out / "audit_config.snapshot.json", command_config)
    queue.to_csv(args.out / "event_review_queue.csv", index=False, encoding="utf-8-sig")
    records[["event_candidate_id", "record_id", "season", "native_cell_id", "environment_valid", "split_role",
             "group_id", "outer_fold"]].to_csv(args.out / "nd_presence_record_lineage.csv", index=False, encoding="utf-8-sig")
    counts = {}
    for kind, source_queue in queue.groupby("source_kind", sort=True):
        counts[kind] = {
            "events": len(source_queue), "same_base_model_ledger_events": int(source_queue.same_nd_source_used_by_base_model.sum()),
            "events_linked_to_nd_positive_waterbird_membership": int(source_queue.presence_member_record_count.gt(0).sum()),
            "presence_source_role_event_counts": source_queue.presence_role_unique.value_counts().to_dict(),
            "presence_source_role_flags_can_overlap": {field: int(source_queue[field].sum()) for field in (
                "has_development_presence_source", "has_locked_test_presence_source", "has_buffer_excluded_presence_source")},
            "lineage_conflicts": int(source_queue.presence_lineage_conflict.sum()),
            "events_with_possible_duplicate_rows": int(source_queue.possible_duplicate_rows.gt(0).sum()),
            "events_in_rounded_coordinate_collision_queue": int(source_queue.cross_event_collision_status.eq("review_required").sum()),
            "historical_2025_already_viewed_events": int(source_queue.historical_2025_already_viewed.sum()),
            "source_crs_status_counts": source_queue.source_crs.value_counts().to_dict(),
            "season_event_counts": source_queue.season.value_counts().to_dict(),
        }
    summary = {
        "status": "SUPERVISION_SOURCE_LINEAGE_AUDITED_NO_TARGETS_CREATED", "candidate_events": len(queue),
        "labels_created": 0, "training_eligible_events": 0, "sources": counts,
        "positive_waterbird_source_records_joined": len(records),
        "record_roles": records.split_role.value_counts().to_dict(),
        "models_fitted": 0, "model_outputs_read": False, "locked_test_metrics_read": False,
        "new_independent_test_registered": False,
        "caveats": [
            "ND is the same original ledger used for base-model presence, target-group background and inferred visit-density sources.",
            "Presence-role links identify source usage, not a newly observed target or verified survey effort.",
            "ND events without positive waterbird members can still contribute all-bird background/visit sources; they are not independent absences.",
            "ND can be reconsidered only after response/protocol review and a future cross-fitting design with recorded fit/tune/calibration exclusions.",
            "An outer fold ID alone does not make current preselected-predictor outputs eligible for gate training.",
            "Non-ND coordinate systems and source independence remain unknown; no geographic transform is applied to them.",
            "2025 monitoring was historically viewed and cannot be relabeled a fresh blind final test.",
            "Zhuque records cover April 2026 only; they do not independently validate four seasons.",
            "Exact/rounded duplicate flags are review screens, not automatic proof that visits are identical or independent.",
            "No waterbird rows is not a verified nondetection. No count, richness or blank rating is converted to a target.",
        ],
    }
    write_json(args.out / "summary.json", summary)
    report = ["# 独立监督来源血缘审计", "", f"候选事件 {len(queue):,}；真实标签 0；监督训练合格事件 0。", "",
              "ND 与底模使用同一原始台账，不能称新增独立测试。逐原始 record_id 和季节连接正数量水鸟源记录、原生像元及冻结空间角色。",
              "多像元、角色或外层折冲突保留为复核项，不强行指定单一组。其余来源保留坐标系和来源独立性未知。", "",
              "| 来源 | 事件 | 关联 ND 正数量水鸟成员事件 | 开发 presence | 锁定 presence | 缓冲排除 presence | 冲突 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for kind, value in counts.items():
        flags = value["presence_source_role_flags_can_overlap"]
        report.append(f"| {kind} | {value['events']} | {value['events_linked_to_nd_positive_waterbird_membership']} | {flags['has_development_presence_source']} | {flags['has_locked_test_presence_source']} | {flags['has_buffer_excluded_presence_source']} | {value['lineage_conflicts']} |")
    report.extend(["", "下一步先核实真实事件、重复记录、分类、调查协议与努力量，以及独立响应定义。ND 若将来用于监督，须重新设计严格交叉拟合并保存每折排除记录；当前审计不授予训练资格。",
                   "2025 已被历史分析查看；朱雀只有 2026 年 4 月。模型图、HQ 和排名均未读入，不能供真实评价者的盲评材料使用。",
                   "", "详细角色、折与复核任务见 event_review_queue.csv；原始记录链接见 nd_presence_record_lineage.csv。精确坐标未复制到审核表，本地来源文件和原生像元索引仍按敏感数据保存。"])
    (args.out / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    outputs = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(args.out.iterdir()) if p.is_file()}
    project = Path(__file__).resolve().parents[2]
    git_prefix = ["git", "-c", "safe.directory=" + project.as_posix(), "-C", str(project)]
    git_commit = subprocess.check_output(git_prefix + ["rev-parse", "HEAD"], text=True).strip()
    working_tree_dirty = bool(subprocess.check_output(git_prefix + ["status", "--porcelain"], text=True).strip())
    manifest = {**{k: summary[k] for k in ("status", "labels_created", "training_eligible_events", "models_fitted",
                    "model_outputs_read", "locked_test_metrics_read", "new_independent_test_registered")},
        "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "command_line": sys.argv,
        "script_sha256": digest(__file__), "git_commit_sha": git_commit, "git_working_tree_dirty": working_tree_dirty,
        "audit_schema_version": command_config["audit_schema_version"],
        "config_snapshot": "audit_config.snapshot.json", "cli_parameters": command_config["parameters"],
        "split_hash": split["split_hash"],
        "original_sources_modified": False, "private_sensitive_data_do_not_commit": True,
        "verified_consumed_inputs": inventory, "outputs": outputs}
    write_json(args.out / "manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("candidates", "birds", "members", "observations", "inputs", "splits", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    summary = audit(parser.parse_args())
    print(json.dumps({k: summary[k] for k in ("status", "candidate_events", "labels_created", "training_eligible_events",
                                             "positive_waterbird_source_records_joined", "record_roles")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
