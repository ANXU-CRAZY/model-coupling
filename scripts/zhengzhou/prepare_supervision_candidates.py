"""Build event evidence from original ledgers without inventing supervision.

Event hashes and taxonomic matches are candidates. Empty labels, unverified
effort and source independence remain explicit until real evidence is reviewed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(value):
    return re.sub(r"\s+", "", str(value)).casefold() if pd.notna(value) else ""


def identifier(prefix, value):
    return prefix + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--birds", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    policy = json.loads(args.scope.read_text(encoding="utf-8"))
    names = {"nd": "zhengzhou_nd_original_records_with_audit_flags.csv",
             "monitoring": "monitoring_csv_records_with_audit_flags.csv",
             "zhuque": "zhuque_xlsx_original_records_with_audit_flags.csv"}
    columns = ["record_id", "source_file", "source_sheet", "source_row", "species_cn", "species_latin", "family",
               "site_raw", "date_start_parsed", "date_end_parsed", "year_parsed", "season_calendar",
               "longitude_raw", "latitude_raw", "longitude_numeric_unconfirmed_crs", "latitude_numeric_unconfirmed_crs",
               "abundance_numeric", "survey_duration_minutes_from_times", "duration_status", "survey_event_key_unconfirmed"]
    frames = {source: pd.read_csv(args.birds / name, usecols=columns, low_memory=False) for source, name in names.items()}
    for source, frame in frames.items():
        for field in ("record_id", "survey_event_key_unconfirmed"):
            if frame[field].isna().any() or frame[field].astype(str).str.strip().eq("").any():
                raise ValueError(f"Missing {field} in {source}; restore source lineage first")
        if not frame.record_id.is_unique:
            raise ValueError(f"Repeated record_id in {source}")
        for field in ("source_file", "date_start_parsed", "date_end_parsed",
                      "longitude_numeric_unconfirmed_crs", "latitude_numeric_unconfirmed_crs"):
            if frame.groupby("survey_event_key_unconfirmed")[field].nunique(dropna=False).gt(1).any():
                raise ValueError(f"Inconsistent {field} within an inferred event in {source}")
    tax = frames["nd"][["species_cn", "species_latin", "family"]].drop_duplicates().copy()
    tax["tax_key"] = list(zip(tax.species_cn.map(key), tax.species_latin.map(key)))
    tax["core"] = tax.family.fillna("").astype(str).str.strip().isin(policy["families"])
    tax_map = tax.groupby("tax_key").core.agg(lambda values: bool(values.iloc[0]) if values.nunique() == 1 else None).to_dict()
    events, links, summaries = [], [], {}
    for source, data in frames.items():
        data["family_scope_core_candidate"] = data.family.fillna("").astype(str).str.strip().isin(policy["families"])
        data["taxonomy_scope_status"] = "source_family_pending_species_review"
        if source == "zhuque":
            matches = pd.Series([tax_map.get((key(cn), key(latin))) for cn, latin in zip(data.species_cn, data.species_latin)], index=data.index)
            data["family_scope_core_candidate"] = matches.eq(True)
            data["taxonomy_scope_status"] = np.where(matches.isna(), "unmatched_or_ambiguous", "exact_name_pair_scope_match_pending_taxonomy_review")
        data["event_candidate_id"] = data.survey_event_key_unconfirmed.map(lambda value: identifier(source + "_event_", value))
        counts = pd.to_numeric(data.abundance_numeric, errors="coerce")
        data["core_reported_count"] = counts.where(data.family_scope_core_candidate)
        duplicate_unit = ["species_cn", "species_latin", "date_start_parsed", "date_end_parsed", "site_raw",
                          "longitude_numeric_unconfirmed_crs", "latitude_numeric_unconfirmed_crs", "abundance_numeric"]
        data["possible_duplicate_record"] = data.duplicated(duplicate_unit, keep=False)
        for event_id, visit in data.groupby("event_candidate_id", sort=True):
            first = visit.iloc[0]
            core = visit.loc[visit.family_scope_core_candidate]
            lon, lat = first.longitude_numeric_unconfirmed_crs, first.latitude_numeric_unconfirmed_crs
            date = pd.to_datetime(first.date_start_parsed, errors="coerce")
            # Rounded same-day keys expose review collisions. They are not final event/site IDs.
            collision = f"{float(lon):.3f}|{float(lat):.3f}|{date.strftime('%Y-%m-%d')}" if pd.notna(lon) and pd.notna(lat) and pd.notna(date) else None
            events.append({
                "event_candidate_id": event_id, "event_id_status": "inferred_pending_confirmation",
                "source_kind": source, "source_file": first.source_file, "source_sheet": first.source_sheet,
                "source_event_key_original": first.survey_event_key_unconfirmed, "source_record_count": len(visit),
                "site_candidate_id": identifier(source + "_site_", f"{first.site_raw}|{lon}|{lat}"),
                "site_name_raw": first.site_raw, "start_time": first.date_start_parsed, "end_time": first.date_end_parsed,
                "calendar_year": first.year_parsed, "season": first.season_calendar,
                "season_year": date.year + int(date.month == 12) if pd.notna(date) else None,
                "longitude_numeric": lon, "latitude_numeric": lat,
                "source_crs": "EPSG:4326" if source == "nd" else "UNCONFIRMED",
                "crs_evidence": "user_confirmed_WGS84" if source == "nd" else "pending_source_confirmation",
                "same_day_rounded_coordinate_review_key": collision,
                "reported_duration_minutes": first.survey_duration_minutes_from_times,
                "duration_quality_flag": first.duration_status,
                "protocol": "unknown", "complete_target_checklist": "unknown",
                "effort_minutes_verified": None, "observer_count": None, "distance_km": None, "survey_area_ha": None,
                "observed_waterbird_species_names_n": int(core.species_cn.nunique()),
                "reported_waterbird_count_sum_unadjusted": float(core.core_reported_count.sum(min_count=1)) if len(core) else None,
                "no_recorded_waterbirds_is_confirmed_non_detection": False,
                "taxonomy_unmatched_rows": int(visit.taxonomy_scope_status.eq("unmatched_or_ambiguous").sum()),
                "possible_duplicate_rows": int(visit.possible_duplicate_record.sum()),
                "lineage_status": "pending_source_and_event_review",
                "target_protection": None, "target_restoration": None,
                "eligible_for_supervised_training": False,
                "blockers": "protocol_effort_unknown;event_identity_unconfirmed;source_independence_unconfirmed;targets_unmeasured",
            })
        links.append(data[["event_candidate_id", "record_id", "source_file", "source_sheet", "source_row"]])
        summaries[source] = {"records": len(data), "candidate_events": int(data.event_candidate_id.nunique()),
                             "core_waterbird_candidate_records": int(data.family_scope_core_candidate.sum()),
                             "unmatched_taxonomy_rows": int(data.taxonomy_scope_status.eq("unmatched_or_ambiguous").sum()),
                             "input_sha256": digest(args.birds / names[source])}
    frame = pd.DataFrame(events)
    require = frame.event_candidate_id.is_unique
    if not require:
        raise ValueError("Candidate event identifier collision")
    collisions = frame.dropna(subset=["same_day_rounded_coordinate_review_key"]).groupby("same_day_rounded_coordinate_review_key").filter(lambda group: len(group) > 1)
    collision_ids = set(collisions.event_candidate_id)
    frame["cross_event_collision_status"] = np.where(frame.event_candidate_id.isin(collision_ids), "review_required", "no_collision_found_by_rounded_same_day_screen")
    args.out.mkdir(parents=True)
    frame.to_csv(args.out / "survey_events_candidate.csv", index=False, encoding="utf-8-sig")
    pd.concat(links, ignore_index=True).to_csv(args.out / "event_record_links.csv", index=False, encoding="utf-8-sig")
    collisions.to_csv(args.out / "same_day_coordinate_collisions_review.csv", index=False, encoding="utf-8-sig")
    # A source-level review registers genuine protocol information; it does not create event labels.
    review = frame.groupby(["source_kind", "source_file"], dropna=False).agg(events=("event_candidate_id", "size"), records=("source_record_count", "sum")).reset_index()
    for field in ("protocol", "complete_target_checklist", "fixed_duration_confirmed", "source_independence"):
        review[field] = "unknown"
    for field in ("verified_duration_minutes", "verified_observer_count", "evidence_reference", "reviewer_id", "review_date"):
        review[field] = None
    review.to_csv(args.out / "source_protocol_review.csv", index=False, encoding="utf-8-sig")
    manifest = {"status": "SUPERVISION_EVIDENCE_CANDIDATES_NO_TARGETS_CREATED", "sources": summaries,
                "total_candidate_events": len(frame), "labels_created": 0,
                "same_day_coordinate_collision_events": len(collisions),
                "coordinate_collision_grid_degrees": 0.001,
                "coordinate_collision_is_proof_of_independence": False,
                "caveats": ["Event IDs are inferred and include source file; aliases and adjacent-cell collisions remain possible",
                            "Rounded coordinate screening is candidate-only; non-ND coordinate references remain unconfirmed",
                            "Zero-row visits are absent from record-only sources; empty waterbird summaries are not absences",
                            "Reported time span is not verified survey effort; no counts/richness were converted into targets",
                            "Zero unmatched taxonomy rows for ND/monitoring means no paired-name mapping was attempted; species review is still pending",
                            "Zhuque covers April 2026 only and cannot independently validate all seasons"],
                "outputs": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in args.out.glob("*.csv")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "events": len(frame), "sources": {k: {f: v[f] for f in ("records", "candidate_events", "core_waterbird_candidate_records")} for k, v in summaries.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
