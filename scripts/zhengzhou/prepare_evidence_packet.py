"""Assemble parameter references and blank supervision forms; create no labels.

Threat scenarios are proposed sensitivity designs, not model runs or confidence
intervals. Local habitat values remain empty until classes/guilds are reviewed.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    supervision_manifest = json.loads((args.supervision / "manifest.json").read_text(encoding="utf-8"))
    for name, meta in supervision_manifest["outputs"].items():
        if digest(args.supervision / name) != meta["sha256"]:
            raise ValueError("Supervision input hash differs: " + name)
    source_review = pd.read_csv(args.supervision / "source_protocol_review.csv", keep_default_na=False)
    events = pd.read_csv(args.supervision / "survey_events_candidate.csv", low_memory=False)
    if len(events) != supervision_manifest["total_candidate_events"]:
        raise ValueError("Event count differs from supervision manifest")
    if supervision_manifest.get("labels_created") != 0:
        raise ValueError("This preparation step accepts only unlabeled supervision candidates")
    if events[["target_protection", "target_restoration"]].notna().to_numpy().any():
        raise ValueError("Targets must remain empty in this candidate evidence packet")
    if not events.eligible_for_supervised_training.astype(str).str.casefold().eq("false").all():
        raise ValueError("Candidate supervision cannot yet be eligible for training")
    dates = pd.to_datetime(events.start_time, errors="coerce")
    events["actual_date"] = dates.dt.strftime("%Y-%m-%d")
    ranges = events.groupby(["source_kind", "source_file"]).actual_date.agg(["min", "max"])
    review_records = source_review.to_dict("records")
    for row in review_records:
        bounds = ranges.loc[(row["source_kind"], row["source_file"])]
        row["actual_first_date"], row["actual_last_date"] = bounds["min"], bounds["max"]

    base = registry["historical_threats"]
    scenarios = []

    def append(name, threats, k):
        raw_max = max(t["weight"] for t in threats)
        if raw_max <= 0:
            raise ValueError("All-zero threat weights")
        for threat in threats:
            threat["weight"] /= max(1.0, raw_max)
        total = sum(t["weight"] for t in threats)
        for threat in threats:
            threat["normalized_weight"] = threat["weight"] / total
        scenarios.append({"scenario_id": name, "threats": threats,
                          "half_saturation_constant": k,
                          "minimum_boundary_buffer_m": registry["sensitivity_design"]["required_minimum_boundary_buffer_m"],
                          "status": "DESIGN_ONLY_LOCAL_HABITAT_AND_SENSITIVITY_PENDING"})

    k = registry["historical_half_saturation_constant"]
    append("historical_reference", copy.deepcopy(base), k)
    for index, threat in enumerate(base):
        for factor in (0.5, 2.0):
            for field in ("max_dist", "weight"):
                altered = copy.deepcopy(base)
                altered[index][field] *= factor
                append(f"{threat['threat']}_{field}_x{factor:g}", altered, k)
        append(f"omit_{threat['threat']}", copy.deepcopy([t for i, t in enumerate(base) if i != index]), k)
    for factor in (0.5, 2.0):
        append(f"half_saturation_x{factor:g}", copy.deepcopy(base), k * factor)
    local = [{"lucode": item["lucode"], "class_name": "", "guild": guild,
              "habitat": "", "sensitivity_urban_structure": "", "sensitivity_human_activity": "",
              "sensitivity_night_light": "", "evidence_reference": "", "reviewer_id": "",
              "status": "pending_class_crosswalk_and_local_ecological_review"}
             for guild in registry["habitat_prior_strategy"]["guilds_to_review"]
             for item in registry["class_crosswalk"]]
    reference = []
    for scheme, literature in registry["literature_reference"].items():
        for cover, value in literature["habitat_values"].items():
            reference.append({"scheme": scheme, "class_original": cover, "habitat": value,
                              "sensitivity_urban_original": literature.get("urban_sensitivities", {}).get(cover, ""),
                              "table": literature["table"], "source": literature["source"],
                              "transfer_limit": literature["transfer_limit"],
                              "applied_to_local_lucode": False})
    args.out.mkdir(parents=True)
    write_csv(args.out / "habitat_local_review.csv", local, list(local[0]))
    write_csv(args.out / "habitat_literature_reference.csv", reference, list(reference[0]))
    write_csv(args.out / "source_protocol_review.csv", review_records, list(review_records[0]))
    scenario_rows = [{"scenario_id": s["scenario_id"], "half_saturation_constant": s["half_saturation_constant"],
                      "minimum_boundary_buffer_m": s["minimum_boundary_buffer_m"], "status": s["status"], **t}
                     for s in scenarios for t in s["threats"]]
    write_csv(args.out / "threat_scenario_designs.csv", scenario_rows, list(scenario_rows[0]))
    forms = {}
    forms_dir = args.out / "blank_forms"
    forms_dir.mkdir()
    for template in sorted(args.templates.glob("*.csv")):
        text = template.read_text(encoding="utf-8-sig")
        template_rows = list(csv.reader(text.splitlines()))
        if len(template_rows) != 1:
            raise ValueError("Expected a header-only blank form: " + template.name)
        fields = template_rows[0]
        (forms_dir / template.name).write_text(text, encoding="utf-8-sig")
        forms[template.stem] = fields
    payload = {"local_habitat": local, "literature": reference, "historical_threats": base,
               "historical_k": k, "historical_evidence": registry["historical_evidence"],
               "source_protocol_review": review_records, "blank_form_fields": forms,
               "candidate_events": len(events), "labels_created": 0,
               "scenario_design_count": len(scenarios)}
    (args.out / "workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "threat_scenario_designs.json").write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"status": "EVIDENCE_PACKET_READY_FOR_REAL_REVIEW_NO_LABELS_OR_OFFICIAL_RUN",
                "local_parameter_rows": len(local), "reference_rows": len(reference),
                "scenario_design_count": len(scenarios), "candidate_events": len(events),
                "labels_created": 0, "official_invest_executed": False,
                "registry_sha256": digest(args.registry),
                "supervision_manifest_sha256": digest(args.supervision / "manifest.json"),
                "outputs": {str(p.relative_to(args.out)): {"sha256": digest(p), "bytes": p.stat().st_size}
                            for p in args.out.rglob("*") if p.is_file()}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "local_parameter_rows", "reference_rows", "scenario_design_count", "candidate_events", "labels_created")}))


if __name__ == "__main__":
    main()
