"""Render a local overview of exploratory archival conservation scores.

Requires GDAL, NumPy and matplotlib. This is a visual inventory of historical
inputs, not evidence of model accuracy or an accepted management zoning map.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads((args.baselines / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "EXPLORATORY_ARCHIVAL_CONSERVATION_BASELINES_NOT_VALIDATED":
        raise ValueError("Expected exploratory archival manifest")
    ref = manifest["reference"]
    gt = ref["transform"]
    extent = [gt[0] / 1000, (gt[0] + gt[1] * ref["width"]) / 1000,
              (gt[3] + gt[5] * ref["height"]) / 1000, gt[3] / 1000]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, season in zip(axes.flat, ("spring", "summer", "autumn", "winter")):
        path = args.baselines / ("conservation_" + season + "_geometric_050.tif")
        dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
        array = dataset.GetRasterBand(1).ReadAsArray()
        dataset = None
        scores = np.ma.masked_where((array == -9999) | ~np.isfinite(array), array)
        im = ax.imshow(scores, origin="upper", extent=extent, vmin=0, vmax=1, cmap="YlGnBu")
        ax.set_title(season.capitalize())
        ax.set_xlabel("UTM 49N easting (km)")
        ax.set_ylabel("UTM 49N northing (km)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Fixed geometric score: sqrt(M x HQ)")
    fig.suptitle("Zhengzhou | archival conservation baselines\nExploratory, unvalidated; static HQ; source MaxEnt ASC CRS inferred from native grid", fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(str(args.out.resolve()))


if __name__ == "__main__":
    main()
