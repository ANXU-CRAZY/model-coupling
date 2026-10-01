"""Pinned official Java MaxEnt, isolated serial jobs and auditable SWD predictions.

Background represents availability, never measured absences. Cloglog predictions
remain suitability scores; no ecological probability calibration is asserted.
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

MAXENT_VERSION = "3.4.4"
MAXENT_SHA256 = "4c856e55412f70c5597b03cf9aaaf27e0782e0921f937262273b68bdcb8fee5e"
OUTPUT_SCALE = "cloglog"
PROJECTION_CHUNK_ROWS = 50_000
FC_FLAGS = {"L": (True, False, False), "LQ": (True, True, False), "LQH": (True, True, True)}
LAMBDA_METADATA = {"linearPredictorNormalizer", "densityNormalizer", "numBackgroundPoints", "entropy"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _time():
    return datetime.now(timezone.utc).isoformat()


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


@lru_cache(maxsize=16)
def _version(java, jar, expected_hash):
    actual = subprocess.run([java, "-Djava.awt.headless=true", "-jar", jar, "printversion=true"],
                            capture_output=True, text=True, timeout=30, check=True)
    if actual.stdout.strip() != "MaxEnt version " + MAXENT_VERSION:
        raise ValueError("Unexpected official MaxEnt version")
    runtime = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=30, check=True)
    return {"maxent_version": MAXENT_VERSION, "jar_sha256": expected_hash,
            "java_version": (runtime.stdout + runtime.stderr).strip()}


def check_jar(java, jar):
    java, jar = str(Path(java).resolve()), str(Path(jar).resolve())
    if not Path(java).is_file() or not Path(jar).is_file():
        raise FileNotFoundError("Java or pinned official MaxEnt jar missing")
    actual = sha256(jar)
    if actual != MAXENT_SHA256:
        raise ValueError("MaxEnt jar does not match the source-verified 3.4.4 SHA256")
    return dict(_version(java, jar, actual))


def validate_scale(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Official cloglog output must be finite [0,1]; it is not a calibrated probability")
    return values


def lambda_complexity(path):
    count = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError("Malformed official lambda row")
            name, coefficient = row[0].strip(), float(row[1])
            if not np.isfinite(coefficient):
                raise ValueError("Non-finite lambda coefficient")
            if name not in LAMBDA_METADATA and coefficient != 0:
                count += 1
    return count


def inspect_swd(path, expected_columns=None, require_one_species=False):
    """Stream input checks. SWD has no embedded CRS: producer must audit it upstream."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        original_header = next(csv.reader(stream), [])
    if len(original_header) != len(set(original_header)):
        raise ValueError("Duplicate SWD predictor header")
    columns, rows, species = None, 0, set()
    with pd.read_csv(path, chunksize=PROJECTION_CHUNK_ROWS, keep_default_na=False) as reader:
        for frame in reader:
            if columns is None:
                columns = list(frame.columns)
                if columns[:3] != ["species", "longitude", "latitude"] or len(columns) < 4:
                    raise ValueError("Expected SWD species,longitude,latitude,predictor... schema")
                if len(columns) != len(set(columns)) or any(not field.strip() for field in columns):
                    raise ValueError("Invalid SWD header")
                if expected_columns is not None and columns != expected_columns:
                    raise ValueError("Missing/mismatched predictor columns or ordering")
            values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
            if not np.isfinite(values).all() or (values[:, 2:] == -9999).any():
                raise ValueError("SWD contains missing/nonfinite/NoData predictor values")
            if np.any(np.abs(values[:, 0]) > 180) or np.any(np.abs(values[:, 1]) > 90):
                raise ValueError("Expected producer-confirmed longitude/latitude coordinates")
            if frame.species.astype(str).str.strip().eq("").any():
                raise ValueError("Missing SWD species name")
            species.update(frame.species.astype(str))
            rows += len(frame)
    if not rows:
        raise ValueError("Empty SWD input")
    if require_one_species and len(species) != 1:
        raise ValueError("One seasonal community model per job is required")
    return {"columns": columns, "rows": rows, "species": sorted(species), "sha256": sha256(path)}


def _prepare(job):
    job = dict(job)
    out = Path(job["out_dir"]).resolve()
    if out.exists():
        raise FileExistsError(out)
    if job["fc"] not in FC_FLAGS or float(job["rm"]) <= 0 or not np.isfinite(job["rm"]):
        raise ValueError("Invalid RM or FC")
    if int(job.get("max_iterations", 500)) < 1:
        raise ValueError("Positive iteration limit required")
    if float(job.get("timeout", 180)) <= 0:
        raise ValueError("Positive execution timeout required")
    job["runtime"] = check_jar(job["java"], job["jar"])
    for key in ("java", "jar", "train_csv", "background_csv", "projection_csv"):
        job[key] = str(Path(job[key]).resolve())
        if any(token in job[key] for token in ("\t", "\n", "\r", ",")):
            raise ValueError("Paths cannot contain TSV delimiters or projection-list commas")
    train = inspect_swd(job["train_csv"], require_one_species=True)
    background = inspect_swd(job["background_csv"], train["columns"])
    projection = inspect_swd(job["projection_csv"], train["columns"])
    species = train["species"][0]
    if any(character in species for character in '/\\:*?"<>|'):
        raise ValueError("Unsafe MaxEnt species output filename")
    out.mkdir(parents=True, exist_ok=False)
    job["out_dir"], job["species"] = str(out), species
    projection_files = []
    if projection["rows"] > PROJECTION_CHUNK_ROWS:
        chunks_dir = out / "projection_inputs"
        chunks_dir.mkdir()
        with pd.read_csv(job["projection_csv"], chunksize=PROJECTION_CHUNK_ROWS) as reader:
            for index, chunk in enumerate(reader):
                path = chunks_dir / f"projection_part{index:05d}.csv"
                chunk.to_csv(path, index=False)
                projection_files.append(path)
    else:
        projection_files.append(Path(job["projection_csv"]))
    linear, quadratic, hinge = FC_FLAGS[job["fc"]]
    boolean = lambda value: str(value).lower()
    args = ["visible=false", "warnings=false", "askoverwrite=false", "threads=1",
            "randomseed=false", "randomtestpoints=0", "replicates=1", "autofeature=false",
            f"linear={boolean(linear)}", f"quadratic={boolean(quadratic)}", f"hinge={boolean(hinge)}",
            "product=false", "threshold=false", "polyhedral=false", "outputformat=cloglog",
            f"betamultiplier={float(job['rm']):g}", "addsamplestobackground=true",
            "addallsamplestobackground=false", "allowpartialdata=false", "removeduplicates=true",
            "pictures=false", "plots=false", "responsecurves=false", "jackknife=false",
            "writebackgroundpredictions=true", "writeclampgrid=false", "writemess=false",
            "outputgrids=true", "cache=false", "appendtoresultsfile=false",
            f"maximumiterations={int(job.get('max_iterations', 500))}",
            f"samplesfile={job['train_csv']}", f"environmentallayers={job['background_csv']}",
            f"projectionlayers={','.join(str(path) for path in projection_files)}", f"outputdirectory={out}"]
    manifest = {"status": "PREPARED_NOT_FITTED", **job["runtime"], "rm": float(job["rm"]), "fc": job["fc"],
                "output_scale": OUTPUT_SCALE, "output_is_calibrated_probability": False,
                "background_is_absence": False, "requested_seed": int(job["seed"]), "effective_maxent_rng_seed": 0,
                "rng_note": "Official randomseed=false initializes java.util.Random(0); no arbitrary numeric seed CLI",
                "source_crs_audit": "REQUIRED_UPSTREAM_SWD_HAS_NO_CRS_METADATA",
                "inputs": {"train": train, "background": background, "projection": projection},
                "input_paths": {key: job[key] for key in ("train_csv", "background_csv", "projection_csv")},
                "command": [job["java"], "-Xmx384m", "-Djava.awt.headless=true", "-jar", job["jar"], *args],
                "prepared_at_utc": _time(), "maxent_arguments": args}
    job["args"], job["manifest"], job["projection_files"] = args, manifest, projection_files
    _save(out / "manifest.json", manifest)
    return job


def _finish(job, start, end):
    out = Path(job["out_dir"])
    predictions = []
    predicted_files = []
    for path in job["projection_files"]:
        result_path = out / (job["species"] + "_" + path.stem + ".csv")
        if not result_path.is_file():
            raise ValueError("Official projection output missing: " + str(result_path))
        original = pd.read_csv(path)
        result = pd.read_csv(result_path)
        if len(original) != len(result) or result.shape[1] != 3:
            raise ValueError("Official SWD projection row count/schema mismatch")
        if "cloglog" not in result.columns[-1].lower():
            raise ValueError("Official projection does not declare cloglog")
        if not np.allclose(result.iloc[:, :2], original.iloc[:, 1:3], rtol=0, atol=1e-10):
            raise ValueError("Official prediction coordinates do not match input row order")
        predictions.append(validate_scale(result.iloc[:, -1].to_numpy(float)))
        predicted_files.append(str(result_path))
    predictions = np.concatenate(predictions)
    lambda_path = out / (job["species"] + ".lambdas")
    complexity = lambda_complexity(lambda_path)
    summary = pd.read_csv(out / "maxentResults.csv")
    if len(summary) != 1:
        raise ValueError("Expected one MaxEnt model summary")
    java_metrics = {str(key): (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
                    for key, value in summary.iloc[0].items()}
    if sha256(job["jar"]) != MAXENT_SHA256:
        raise ValueError("Jar changed during job")
    for key in ("train", "background", "projection"):
        input_key = {"train": "train_csv", "background": "background_csv", "projection": "projection_csv"}[key]
        if sha256(job[input_key]) != job["manifest"]["inputs"][key]["sha256"]:
            raise ValueError("Input changed during MaxEnt execution")
    manifest = job["manifest"]
    manifest.update(status="OFFICIAL_MAXENT_FITTED", started_at_utc=start, ended_at_utc=end,
                    prediction_rows=len(predictions), prediction_min=float(predictions.min()),
                    prediction_max=float(predictions.max()), nonzero_feature_count=complexity,
                    java_metrics=java_metrics, prediction_files=predicted_files,
                    uncertainty_interpretation="A single fitted member; no confidence interval")
    manifest["outputs"] = {str(path.relative_to(out)): {"sha256": sha256(path), "bytes": path.stat().st_size}
                           for path in out.rglob("*") if path.is_file() and path.name != "manifest.json"}
    _save(out / "manifest.json", manifest)
    return {"predictions": predictions, "prediction_csv": predicted_files[0] if len(predicted_files) == 1 else predicted_files,
            "lambda_file": str(lambda_path), "complexity": complexity, "java_metrics": java_metrics,
            "command": manifest["command"], "manifest_path": str(out / "manifest.json")}


def _worker(indexed):
    first = indexed[0][1]
    build = Path(first["out_dir"]) / "_java_batch"
    build.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts" / "zhengzhou" / "MaxentBatch.java"
    javac = Path(first["java"]).with_name("javac.exe" if Path(first["java"]).suffix == ".exe" else "javac")
    tasks = build / "tasks.tsv"
    tasks.write_text("\n".join("\t".join([str(index), *job["args"]]) for index, job in indexed) + "\n", encoding="utf-8")
    command = [first["java"], "-Xmx384m", "-Djava.awt.headless=true", "-cp", str(build), "MaxentBatch", first["jar"], str(tasks)]
    batch_start = _time()
    for _, job in indexed:
        job["manifest"].update(status="BATCH_EXECUTION_PENDING", batch_command=command,
                               batch_java_source_sha256=sha256(source))
        _save(Path(job["out_dir"]) / "manifest.json", job["manifest"])
    try:
        compile_process = subprocess.run(
            [str(javac), "--release", "11", "-encoding", "UTF-8", "-d", str(build), str(source)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
        (build / "compile_stdout.txt").write_text(compile_process.stdout, encoding="utf-8")
        (build / "compile_stderr.txt").write_text(compile_process.stderr, encoding="utf-8")
        if compile_process.returncode != 0:
            raise RuntimeError("MaxEnt batch helper compilation failed; inspect " + str(build / "compile_stderr.txt"))
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=sum(float(job.get("timeout", 180)) for _, job in indexed), check=False)
        (build / "stdout.txt").write_text(process.stdout, encoding="utf-8")
        (build / "stderr.txt").write_text(process.stderr, encoding="utf-8")
        markers = {}
        for line in process.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) == 3 and fields[0] in ("MAXENT_BATCH_START", "MAXENT_BATCH_OK"):
                markers[(fields[0], int(fields[1]))] = datetime.fromtimestamp(int(fields[2])/1000, timezone.utc).isoformat()
        if process.returncode != 0:
            raise RuntimeError("Official MaxEnt batch failed; inspect " + str(build / "stderr.txt"))
        results = {}
        for index, job in indexed:
            if ("MAXENT_BATCH_OK", index) not in markers:
                raise ValueError("Missing official batch completion marker")
            results[index] = _finish(job, markers[("MAXENT_BATCH_START", index)], markers[("MAXENT_BATCH_OK", index)])
        return results
    except Exception as error:
        if isinstance(error, subprocess.TimeoutExpired):
            for name, captured in (("timeout_stdout.txt", error.stdout), ("timeout_stderr.txt", error.stderr)):
                if captured is not None:
                    captured = captured.decode("utf-8", errors="replace") if isinstance(captured, bytes) else captured
                    (build / name).write_text(captured, encoding="utf-8")
        for _, job in indexed:
            if job["manifest"]["status"] != "OFFICIAL_MAXENT_FITTED":
                job["manifest"].update(status="FAILED_OR_NOT_COMPLETED", failure=str(error),
                                       batch_started_at_utc=batch_start, failed_at_utc=_time())
                _save(Path(job["out_dir"]) / "manifest.json", job["manifest"])
        raise


def run_jobs(jobs, max_workers=1):
    """Run same-Java/jar jobs in deterministic input order, one isolated loader/job.

    One worker is the memory-conscious default. Batch timeout is the sum of job
    budgets; a worker stops at its first error and keeps failed artifacts.
    """
    if not jobs:
        return []
    if int(max_workers) < 1:
        raise ValueError("Positive worker count required")
    outputs = [str(Path(job["out_dir"]).resolve()) for job in jobs]
    if len(outputs) != len(set(outputs)) or any(Path(path).exists() for path in outputs):
        raise FileExistsError("Every job needs a distinct new output directory")
    prepared = [_prepare(job) for job in jobs]
    if len({(job["java"], job["jar"]) for job in prepared}) != 1:
        raise ValueError("A batch requires the same Java and pinned jar")
    workers = min(int(max_workers), len(prepared))
    shards = [[(index, job) for index, job in enumerate(prepared) if index % workers == shard] for shard in range(workers)]
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_worker, shards):
            results.update(result)
    return [results[index] for index in range(len(prepared))]


def run_job(java, jar, train_csv, background_csv, projection_csv, out_dir, rm, fc, seed,
            max_iterations=500, timeout=180):
    return run_jobs([dict(java=java, jar=jar, train_csv=train_csv, background_csv=background_csv,
                         projection_csv=projection_csv, out_dir=out_dir, rm=rm, fc=fc, seed=seed,
                         max_iterations=max_iterations, timeout=timeout)], max_workers=1)[0]
