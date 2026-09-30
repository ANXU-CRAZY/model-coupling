from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit import file_sha256, load_table, save_json, validate_table
from .fusion import fuse, quadrants


def main():
    parser = argparse.ArgumentParser(description="MaxEnt–InVEST auditable coupling research foundation")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create synthetic fixtures, never ecological results")
    demo.add_argument("--out", required=True)
    demo.add_argument("--seed", type=int, default=42)
    baseline = commands.add_parser("baseline", help="Fuse a validated CSV without learned weights")
    baseline.add_argument("--csv", required=True)
    baseline.add_argument("--out", required=True)
    baseline.add_argument("--method", choices=["geometric","linear","minimum"], default="geometric")
    baseline.add_argument("--alpha", type=float, default=0.5)
    baseline.add_argument("--gamma", type=float, default=0.5)
    baseline.add_argument("--m-high", type=float, default=0.7)
    baseline.add_argument("--h-high", type=float, default=0.7)
    baseline.add_argument("--deficit-mode", choices=["low_hq","relative_degradation"], default="low_hq")
    fit = commands.add_parser("train", help="Train with independent targets and audited base provenance")
    fit.add_argument("--csv", required=True)
    fit.add_argument("--provenance", required=True)
    fit.add_argument("--config")
    fit.add_argument("--out", required=True)
    fit.add_argument("--demo", action="store_true")
    pred = commands.add_parser("predict", help="Predict with a previously frozen gate")
    pred.add_argument("--csv", required=True)
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--out", required=True)
    pred.add_argument("--demo", action="store_true")
    raster = commands.add_parser("raster", help="Fuse strictly aligned GeoTIFFs, window by window")
    for key in ("m","h","feasible","habitat-suitability"):
        raster.add_argument("--"+key, required=True)
    raster.add_argument("--out", required=True)
    raster.add_argument("--method", choices=["geometric","linear","minimum"], default="geometric")
    raster.add_argument("--alpha", type=float, default=0.5)
    raster.add_argument("--gamma", type=float, default=0.5)
    prior = commands.add_parser("invest-plan", help="Generate official-model ensemble run configurations")
    for key in ("prior","threats","sensitivity","lulc","out"):
        prior.add_argument("--"+key, required=True)
    prior.add_argument("--members", type=int, default=32)
    prior.add_argument("--seed", type=int, default=42)
    official = commands.add_parser("invest-run", help="Call pinned official InVEST 3.16.1")
    official.add_argument("--args", required=True)
    calibrate = commands.add_parser("calibration-rank", help="Rank parameter candidates from inner spatial folds")
    calibrate.add_argument("--csv", required=True)
    calibrate.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import make_demo
        result = make_demo(args.out, args.seed)
    elif args.command == "baseline":
        df = load_table(args.csv)
        validate_table(df)
        scores = fuse(df.m, df.h, args.alpha, args.gamma, method=args.method,
                      feasible=df.feasible, habitat_suitability=df.habitat_suitability,deficit_mode=args.deficit_mode)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        df["conservation_score"] = scores["conservation"]
        df["restoration_candidate_score"] = scores["restoration_candidate"]
        df["restoration_eligible"] = scores["restoration_eligible"]
        df["zone_reference"] = quadrants(df.m, df.h, df.feasible, df.habitat_suitability, args.m_high, args.h_high)
        df.to_csv(out/"scores.csv", index=False)
        result = {"status": "BASELINE_RESEARCH_SCORES", "rows": len(df), "method": args.method,
                  "alpha": args.alpha, "gamma": args.gamma, "m_high": args.m_high, "h_high": args.h_high,
                  "deficit_mode": args.deficit_mode,
                  "input_sha256": file_sha256(args.csv), "uncertainty": "NOT_PROPAGATED_IN_POINT_BASELINE"}
        save_json(out/"manifest.json", result)
    elif args.command == "train":
        from .training import train
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
        result = train(args.csv,args.provenance,args.out,cfg,args.demo)
        result = {k:result[k] for k in ("status","best_epoch","epochs_run","active_heads")}
    elif args.command == "predict":
        from .training import load_gate, predict_with_model
        model,mean,scale,meta = load_gate(args.checkpoint)
        if meta["data_kind"] == "synthetic" and not args.demo:
            raise ValueError("Synthetic checkpoint requires --demo; never apply it as a real ecological model")
        df = load_table(args.csv)
        scores, weights = predict_with_model(model,df,mean,scale)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        df["conservation_score"],df["restoration_candidate_score"] = scores[:,0],scores[:,1]
        for key,values in zip(("alpha","beta","gamma","delta"),
                               (weights[:,0,0],weights[:,0,1],weights[:,1,0],weights[:,1,1])):
            df[key] = values
        df.to_csv(out/"predictions.csv",index=False)
        result = {"status": "SYNTHETIC_DEMO_ONLY" if args.demo else "EXPERIMENTAL_GATE_PREDICTIONS",
                  "rows": len(df), "checkpoint_sha256": file_sha256(args.checkpoint),
                  "input_sha256": file_sha256(args.csv), "active_heads": list(model.active_heads)}
        save_json(out/"manifest.json",result)
    elif args.command == "raster":
        from .rasters import fuse_rasters
        result = fuse_rasters({key:getattr(args,key) for key in ("m","h","feasible","habitat_suitability")},
                              args.out,args.method,args.alpha,args.gamma)
    elif args.command == "invest-plan":
        from .parameters import make_run_plan
        result = make_run_plan(args.prior,args.threats,args.sensitivity,args.lulc,args.out,args.members,args.seed)
        result = {"status":result["status"],"members":len(result["members"])}
    elif args.command == "invest-run":
        from .parameters import run_official_invest
        run_official_invest(args.args)
        result = {"status":"OFFICIAL_INVEST_COMPLETED"}
    else:
        from .parameters import rank_calibration_candidates
        if Path(args.out).exists():
            raise FileExistsError(args.out)
        result = rank_calibration_candidates(args.csv,args.out)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))


if __name__ == "__main__":
    main()
