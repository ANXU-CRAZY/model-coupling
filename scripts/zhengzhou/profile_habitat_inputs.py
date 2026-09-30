"""Describe LULC and pressure rasters; profiles do not confirm class names or H_j."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    paths = {"lulc": args.baselines / "lulc_cur_utm49_100m.tif",
             "urban_structure": args.baselines / "urban_structure_utm49_100m.tif",
             "human_activity": args.baselines / "human_activity_utm49_100m.tif",
             "night_light": args.baselines / "night_light_utm49_100m.tif"}
    cover = {"water_proxy": "spr_select/water_spring.tif", "trees_proxy": "spr_select/trees_spring.tif",
             "grass_proxy": "spr_select/grass_spring.tif", "crops_proxy": "spr_select/crops_spring.tif",
             "built_proxy": "spr_select/builtarea_spring.tif", "shrubs_proxy": "sum_select/shrubscrub_summer.tif",
             "flooded_vegetation_proxy": "sum_select/floodedvegetation_summer.tif",
             "bare_proxy": "aut_select/bareground_autumn.tif"}
    paths.update({key: args.environment / path for key, path in cover.items()})
    arrays, grid, metadata = {}, None, {}
    for name, path in paths.items():
        with rasterio.open(path) as ds:
            current = (ds.shape, tuple(ds.transform), ds.crs.to_epsg())
            if grid is not None and current != grid:
                raise ValueError("Grid mismatch: " + str(path))
            grid = current
            arrays[name] = ds.read(1, masked=True)
            metadata[name] = {"sha256": digest(path), "path": str(path), "tags": ds.tags(), "valid_pixels": int(arrays[name].count())}
    rows = []
    lulc = arrays["lulc"]
    for code in np.unique(lulc.compressed()):
        area = (~np.ma.getmaskarray(lulc)) & (lulc.data == code)
        row = {"lucode": int(code), "pixels": int(area.sum()), "class_name_confirmed": False}
        for name, array in arrays.items():
            if name == "lulc":
                continue
            valid = area & ~np.ma.getmaskarray(array) & np.isfinite(array.data)
            row[name + "_mean"] = float(array.data[valid].mean()) if valid.any() else None
        rows.append(row)
    threats = ("urban_structure", "human_activity", "night_light")
    correlations = []
    for i, first in enumerate(threats):
        for second in threats[i + 1:]:
            a, b = arrays[first], arrays[second]
            valid = ~(np.ma.getmaskarray(a) | np.ma.getmaskarray(b)) & np.isfinite(a.data) & np.isfinite(b.data)
            av, bv = a.data[valid], b.data[valid]
            correlations.append({"first": first, "second": second, "pixels": int(valid.sum()),
                                 "pearson_r": float(np.corrcoef(av, bv)[0, 1]),
                                 "spearman_rho": float(spearmanr(av, bv).statistic)})
    args.out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.out / "lulc_code_profiles_pending_crosswalk.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(correlations).to_csv(args.out / "threat_correlations.csv", index=False)
    output = {"status": "INPUT_DIAGNOSTICS_NOT_PARAMETER_CALIBRATION", "sources": metadata,
              "lulc_code_mapping_confirmed": False, "threat_intensity_scale_comparability_confirmed": False,
              "correlations": correlations,
              "caveats": ["Proxy-cover means are descriptive; mixed years and landcover definitions cannot confirm codes",
                          "Correlation may indicate redundant pressure but cannot establish shared ecological effect",
                          "No spatial-independence p-values or habitat parameters were estimated"]}
    (args.out / "manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "classes": len(rows), "threat_correlations": correlations}))


if __name__ == "__main__":
    main()
