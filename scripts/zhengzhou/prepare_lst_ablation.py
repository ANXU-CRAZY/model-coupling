"""Materialize static-LST and no-LST SWD variants with identical observation rows.

Uses CSV strings to preserve all remaining predictor/coordinate values exactly.
No model fitting, feature tuning, final split or ecological comparison is run.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source_manifest = json.loads((args.inputs / "manifest.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True)
    for variant in ("static_lst", "no_lst"):
        (args.out / variant).mkdir()
    files = []
    for season in ("spring", "summer", "autumn", "winter"):
        for kind in ("presence", "background"):
            name = f"maxent_swd_{kind}_{season}.csv"
            path = args.inputs / name
            if digest(path) != source_manifest["outputs"][name]["sha256"]:
                raise ValueError("SWD source hash differs: " + name)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            header = rows[0]
            removed = [i for i, column in enumerate(header) if column.startswith("lst_")]
            if len(removed) != 2:
                raise ValueError("Expected day and night LST predictors: " + name)
            keep = [i for i in range(len(header)) if i not in removed]
            static = args.out / "static_lst" / name
            without = args.out / "no_lst" / name
            shutil.copyfile(path, static)
            with without.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows([[row[i] for i in keep] for row in rows])
            files.append({"name": name, "rows": len(rows) - 1,
                          "removed_predictors": [header[i] for i in removed],
                          "source_sha256": digest(path), "static_sha256": digest(static), "no_lst_sha256": digest(without)})
    manifest = {"status": "ABLATION_INPUTS_PREPARED_MODELS_NOT_FITTED",
                "source_manifest_sha256": digest(args.inputs / "manifest.json"), "files": files,
                "retained_cell_coordinate_predictor_strings_unchanged": True,
                "final_folds_assigned": False, "model_performance_compared": False,
                "third_variant": {"name": "seasonal_lst", "status": "PENDING_TIME_WINDOW_AND_SEASONAL_SURFACES"},
                "paired_evaluation_domain": "common_valid_samples_and_background",
                "execution_rule": "Share outer groups, locked test and common valid domain; select features and retune RM/FC separately inside each variant's base training folds"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "paired_inputs": len(files)}))


if __name__ == "__main__":
    main()
