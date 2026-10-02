"""Inventory full seasonal predictor pool directly inside an existing ZIP.

Keep the archive as the source; no large extracted copies, response-based
selection or model fitting. Numeric equality is distinct from source period,
units and independent upstream lineage.
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
import zipfile

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

from scripts.zhengzhou.export_maxent_oof import verify_split_inventory
from wetland_coupling.parameter_diagnostics import record_input


def parse_ascii(payload):
    lines = payload.decode('ascii').splitlines()
    header = {}
    index = 0
    while len(header) < 6 and index < len(lines):
        line = lines[index].strip(); index += 1
        if not line:
            continue
        key, value = line.split()
        header[key.lower()] = float(value)
    if set(header) != {'ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'}:
        raise ValueError('Unexpected ASCII header')
    width, height = int(header['ncols']), int(header['nrows'])
    data = np.fromstring(' '.join(lines[index:]), sep=' ', dtype=np.float64)
    if data.size != width * height:
        raise ValueError('ASCII count differs from grid')
    data = data.reshape(height, width)
    transform = Affine(header['cellsize'], 0, header['xllcorner'], 0, -header['cellsize'],
                       header['yllcorner'] + height * header['cellsize'])
    valid = np.isfinite(data) & (data != header['nodata_value'])
    return data, valid, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pilot = root / 'local_work/zhengzhou_pilot'
    out = args.out.resolve()
    out.relative_to(root / 'local_work')
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    inventory = {}
    record_input(args.archive, inventory)
    split = pilot / 'maxent_formal_v1_001/splits'
    verify_split_inventory(split)
    record_input(split / 'manifest.json', inventory)
    record_input(split / 'split_masks.npz', inventory)
    with np.load(split / 'split_masks.npz', allow_pickle=False) as z:
        common = z['common_valid'].astype(bool)
    reference_path = pilot / 'archival_baselines_003/lulc_cur_utm49_100m.tif'
    record_input(reference_path, inventory)
    with rasterio.open(reference_path) as ds:
        grid = (ds.shape, ds.transform, ds.crs)
    if grid[0] != common.shape or grid[2].to_epsg() != 32649:
        raise ValueError('Unexpected native reference grid')
    rows, common_full, selected_comparisons = [], common.copy(), []
    common_no_lst = common.copy()
    per_season = {}
    folders = {'spring':'spr_select','summer':'sum_select','autumn':'aut_select','winter':'win_select'}
    members = {}
    with zipfile.ZipFile(args.archive) as archive:
        member_lookup = {}
        for name in archive.namelist():
            member_lookup.setdefault(name.casefold(), []).append(name)
        for season, folder in folders.items():
            files = [i for i in archive.infolist() if f'/data/{season}/all/' in i.filename
                     and i.filename.lower().endswith('.asc')]
            if len(files) != 25 or len({Path(i.filename).stem.casefold() for i in files}) != 25:
                raise ValueError('Full seasonal 25-variable branch not uniquely identified')
            season_common = common.copy()
            season_no_lst = common.copy()
            for entry in sorted(files, key=lambda i:i.filename.casefold()):
                payload = archive.read(entry)  # zipfile validates member CRC
                members[entry.filename] = {'sha256':hashlib.sha256(payload).hexdigest(), 'bytes':len(payload), 'crc32':entry.CRC}
                data, valid, affine = parse_ascii(payload)
                if data.shape != grid[0] or affine != grid[1]:
                    raise ValueError('Archive numeric grid differs from frozen native grid')
                prj_name = str(Path(entry.filename).with_suffix('.prj')).replace('\\','/')
                matches = member_lookup.get(prj_name.casefold(), [])
                if len(matches) != 1:
                    raise ValueError('Projection sidecar not uniquely found: ' + prj_name)
                prj_name = matches[0]
                projection = archive.read(prj_name)
                members[prj_name] = {'sha256':hashlib.sha256(projection).hexdigest(), 'bytes':len(projection)}
                text = projection.decode('ascii').upper()
                if not all(value in text for value in ('UTM','49','WGS84','METERS')):
                    raise ValueError('Archive legacy PRJ differs from expected declared system')
                seasonal_name = Path(entry.filename).stem.casefold()
                is_lst = seasonal_name.startswith('lst')
                values = data[common & valid]
                row = {'season':season, 'variable':seasonal_name, 'archive_member':entry.filename,
                       'valid_pixels_in_frozen_common':int(len(values)),
                       'missing_pixels_in_frozen_common':int((common & ~valid).sum()),
                       'minimum_in_frozen_common':float(values.min()),
                       'maximum_in_frozen_common':float(values.max()),
                       'mean_in_frozen_common':float(values.mean()),
                       'constant_in_frozen_common':bool(np.ptp(values) == 0),
                       'is_static_lst_candidate':is_lst, 'source_period_units_qc_verified':False,
                       'declared_projection':'Legacy UTM zone 49 WGS84 meters; hemisphere not explicit in PRJ'}
                season_common &= valid; common_full &= valid
                if not is_lst:
                    season_no_lst &= valid; common_no_lst &= valid
                target = pilot / f'supermap/environment_exports/{folder}/{seasonal_name}.tif'
                if target.exists():
                    record_input(target, inventory)
                    with rasterio.open(target) as ds:
                        if (ds.shape, ds.transform, ds.crs) != grid:
                            raise ValueError('Current exported layer grid differs')
                        target_data = ds.read(1); target_valid = (ds.read_masks(1) != 0) & np.isfinite(target_data)
                    paired = common & valid & target_valid
                    # ASC float text and UDBX raster precision may differ; preserve both exact and tolerance checks.
                    delta = np.abs(data[paired] - target_data[paired].astype(float))
                    comparison = {'season':season, 'variable':seasonal_name, 'paired_pixels':int(paired.sum()),
                                  'valid_mask_equal_in_frozen_common':bool(np.array_equal(valid[common],target_valid[common])),
                                  'float32_values_equal_in_frozen_common':bool(np.array_equal(data[paired].astype(np.float32),target_data[paired].astype(np.float32))),
                                  'maximum_absolute_difference':float(delta.max()),
                                  'within_rtol_1e_6_atol_1e_6':bool(np.allclose(data[paired],target_data[paired],rtol=1e-6,atol=1e-6))}
                    selected_comparisons.append(comparison)
                rows.append(row)
            per_season[season] = {'variables':25,'including_lst_common_pixels':int(season_common.sum()),
                                  'excluding_lst_common_pixels':int(season_no_lst.sum())}
    if len(selected_comparisons) != 60:
        raise ValueError('Current 60-layer comparison inventory incomplete')
    pd.DataFrame(rows).to_csv(out/'full_pool_inventory.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(selected_comparisons).to_csv(out/'current_layer_numeric_comparison.csv',index=False,encoding='utf-8-sig')
    summary = {'status':'FULL_SEASONAL_CANDIDATE_POOL_LOCATED_NOT_NEW_MODEL_ADMISSION',
               'asc_members':len(rows),'current_layers_compared':len(selected_comparisons),
               'comparisons_with_equal_float32_values':sum(c['float32_values_equal_in_frozen_common'] for c in selected_comparisons),
               'comparisons_with_equal_valid_masks':sum(c['valid_mask_equal_in_frozen_common'] for c in selected_comparisons),
               'comparisons_with_tolerance_agreement':sum(c['within_rtol_1e_6_atol_1e_6'] for c in selected_comparisons),
               'per_season':per_season, 'all_100_layers_common_pixels':int(common_full.sum()),
               'all_non_lst_layers_common_pixels':int(common_no_lst.sum()),
               'existing_frozen_common_pixels':int(common.sum()),
               'extracted_large_files':0, 'new_fitted_models':0, 'locked_test_metrics_read':False,
               'upstream_pool_completely_response_independent':None,
               'source_period_units_and_qc_verified':False,
               'existing_oof_reclassified_as_strict':False, 'gate_eligible':False,
               'next_protocol':'Assess provenance and covariate support; preregister new development-only complete-pool folds without revisiting consumed test',
               'not_sufficient_for':['Current LULC code crosswalk','Independent management supervision','Automatic InVEST H_j calibration']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    sources, outputs = {}, {}
    record_input(__file__,sources)
    record_input(root/'wetland_coupling/parameter_diagnostics.py',sources)
    record_input(root/'scripts/zhengzhou/export_maxent_oof.py',sources)
    for p in sorted(out.iterdir()):
        if p.is_file(): record_input(p,outputs)
    manifest = {'status':summary['status'],'started_at_utc':started,'ended_at_utc':datetime.now(timezone.utc).isoformat(),
                'argv':sys.argv,'python':sys.version,'platform':platform.platform(),
                'git_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                'inputs':inventory,'archive_members':members,'source_code':sources,'outputs':outputs,
                'models_fitted':0,'locked_test_metrics_read':False,'large_files_extracted':0}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
