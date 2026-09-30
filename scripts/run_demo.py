"""One-command cross-platform synthetic exercise, without pip installation of this project."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from wetland_coupling.demo import make_demo
from wetland_coupling.rasters import fuse_rasters
from wetland_coupling.training import train
from wetland_coupling.audit import load_table,save_json
from wetland_coupling.splits import base_crossfit_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",default="runs/demo_001")
    a = p.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True,exist_ok=False)
    make_demo(root/"input")
    save_json(root/"base_crossfit_plan.json",base_crossfit_plan(load_table(root/"input/samples.csv")))
    cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/gate.json").read_text())
    report = train(root/"input/samples.csv",root/"input/provenance.json",root/"gate",cfg,demo=True)
    paths = {k:root/f"input/raster_inputs/{k}.tif" for k in ("m","h","feasible","habitat_suitability")}
    raster = fuse_rasters(paths,root/"raster_baseline")
    print(json.dumps({"status":"SYNTHETIC_ENGINEERING_DEMO_ONLY","rows":report["audit"]["rows"],
                      "epochs_run":report["epochs_run"],"best_epoch":report["best_epoch"],
                      "valid_raster_pixels":raster["valid_pixels"],"directory":str(root.resolve())},indent=2))


if __name__ == "__main__":
    main()
