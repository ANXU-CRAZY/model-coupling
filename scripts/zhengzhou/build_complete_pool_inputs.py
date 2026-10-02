"""Build full-25 inputs on frozen development membership from an audited ZIP.

New arrays, no raster copies; parent inputs and partitions are never changed.
Bird/reference/visit tables exclude locked and buffer groups at construction.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
from affine import Affine

from scripts.zhengzhou.audit_predictor_archive import parse_ascii
from scripts.zhengzhou.export_maxent_oof import verify_split_inventory
from wetland_coupling.maxent_inputs import dataframe_cells, CELL_COLUMNS
from wetland_coupling.maxent_protocol import sha256, write_json
from wetland_coupling.maxent_splits import load_split_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','archive','archive-audit','parent-inputs','splits','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    out=args.out.resolve();out.relative_to(root/'local_work')
    if out.exists():raise FileExistsError(out)
    config=json.loads(args.config.read_text(encoding='utf-8'))
    audit=json.loads((args.archive_audit/'manifest.json').read_text(encoding='utf-8'))
    audit_summary=json.loads((args.archive_audit/'summary.json').read_text(encoding='utf-8'))
    if audit_summary['current_layers_compared']!=60 or audit_summary['comparisons_with_equal_float32_values']!=60 or audit_summary['comparisons_with_equal_valid_masks']!=60:
        raise ValueError('Full pool equivalence was not established')
    if sha256(args.archive)!=config['complete_pool']['archive_sha256']:
        raise ValueError('Archive hash changed')
    parent=json.loads((args.parent_inputs/'manifest.json').read_text(encoding='utf-8'))
    if sha256(args.parent_inputs/'manifest.json')!=config['complete_pool']['parent_input_manifest_sha256']:
        raise ValueError('Parent inputs changed')
    for name,meta in parent['outputs'].items():
        if sha256(args.parent_inputs/name)!=meta['sha256']:raise ValueError('Parent output changed: '+name)
    split=verify_split_inventory(args.splits)
    if sha256(args.splits/'manifest.json')!=config['complete_pool']['parent_split_manifest_sha256']:
        raise ValueError('Frozen split changed')
    plan=load_split_plan(args.splits)
    development=plan['development_fit_mask']
    grid=json.loads((args.parent_inputs/'grid.json').read_text(encoding='utf-8'))
    common=np.load(args.parent_inputs/'common_valid_mask.npy',allow_pickle=False)
    if int(common.sum())!=766386 or np.any(development & ~common):raise ValueError('Unexpected domain')
    out.mkdir(parents=True,exist_ok=False)
    started=datetime.now(timezone.utc).isoformat()
    inventory={str(args.archive.resolve()):{'sha256':config['complete_pool']['archive_sha256'],'bytes':args.archive.stat().st_size}}
    for p in (args.config,args.archive_audit/'manifest.json',args.archive_audit/'summary.json',args.parent_inputs/'manifest.json',args.splits/'manifest.json'):
        inventory[str(p.resolve())]={'sha256':sha256(p),'bytes':p.stat().st_size}
    write_json(out/'config_snapshot.json',config)
    shutil.copyfile(args.parent_inputs/'grid.json',out/'grid.json')
    shutil.copyfile(args.parent_inputs/'common_valid_mask.npy',out/'common_valid_mask.npy')
    seasonal={}
    with zipfile.ZipFile(args.archive) as archive:
        for season in ('spring','summer','autumn','winter'):
            directory=out/season;directory.mkdir()
            entries=sorted([i for i in archive.infolist() if f'/data/{season}/all/' in i.filename and i.filename.lower().endswith('.asc')],key=lambda i:Path(i.filename).stem.casefold())
            names=[Path(i.filename).stem.casefold() for i in entries]
            if len(names)!=25 or len(set(names))!=25 or sum(n.startswith('lst_') for n in names)!=2:
                raise ValueError('Need exactly 25 supplied families including two static LST layers')
            env=np.lib.format.open_memmap(directory/'env.npy',mode='w+',dtype='float32',shape=(grid['height'],grid['width'],25))
            sources={}
            for index,entry in enumerate(entries):
                payload=archive.read(entry)
                digest=hashlib.sha256(payload).hexdigest()
                if digest!=audit['archive_members'][entry.filename]['sha256']:raise ValueError('Archive member changed')
                data,valid,affine=parse_ascii(payload)
                if data.shape!=common.shape or affine!=Affine(*grid['transform']) or np.any(common & ~valid):
                    raise ValueError('Full-pool alignment or support differs')
                env[:,:,index]=np.where(common,data,np.nan).astype(np.float32)
                sources[names[index]]={'archive_member':entry.filename,'sha256':digest,'bytes':len(payload),
                                       'crs_assignment':'EPSG:32649 through archived UTM49 declaration and verified native reference grid'}
                del data,valid,payload
            env.flush();write_json(directory/'names.json',names)
            write_json(directory/'source_layers.json',sources)
            row_counts={}
            for name in ('presence.csv','B0.csv','B1.csv'):
                old=pd.read_csv(args.parent_inputs/season/name)
                row=old.raster_row.to_numpy(int);col=old.raster_col.to_numpy(int)
                retained=old.loc[development[row,col]].copy()
                indices=retained.raster_row.to_numpy(int)*grid['width']+retained.raster_col.to_numpy(int)
                fresh=dataframe_cells(indices,env,names,grid)
                if fresh.native_cell_id.tolist()!=retained.native_cell_id.tolist():raise ValueError('Source cell identity changed')
                # Preserve parent coordinates and all non-predictor metadata exactly.
                previous_names=json.loads((args.parent_inputs/season/'names.json').read_text(encoding='utf-8'))
                table=retained.drop(columns=previous_names).reset_index(drop=True)
                for variable in names:table[variable]=fresh[variable]
                table.to_csv(directory/name,index=False)
                row_counts[name]={'parent_rows':len(old),'development_rows':len(table),'excluded_non_development':len(old)-len(table)}
            visits=pd.read_csv(args.parent_inputs/season/'visit_cells.csv')
            visits=visits.loc[development[visits.raster_row.to_numpy(int),visits.raster_col.to_numpy(int)]].reset_index(drop=True)
            visits.to_csv(directory/'visit_cells.csv',index=False)
            row_counts['visit_cells.csv']={'development_rows':len(visits)}
            seasonal[season]={'predictor_count':25,'rows':row_counts,'no_locked_or_buffer_table_rows':True}
            del env;gc.collect()
            print(json.dumps({'phase':'full_pool_inputs','season':season,'predictors':25,'rows':row_counts}),flush=True)
    outputs={str(p.relative_to(out)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}
    manifest={'status':'COMPLETE_POOL_DEVELOPMENT_INPUTS_PREPARED','started_at_utc':started,'ended_at_utc':datetime.now(timezone.utc).isoformat(),
              'argv':sys.argv,'config_sha256':sha256(args.config),'parent_input_manifest_sha256':sha256(args.parent_inputs/'manifest.json'),
              'parent_split_manifest_sha256':sha256(args.splits/'manifest.json'),'canonical_split_sha256':split['split_hash'],
              'common_valid_cells':int(common.sum()),'development_cells':int(development.sum()),'grid':grid,'seasons':seasonal,
              'inputs':inventory,'outputs':outputs,'models_fitted':0,'locked_test_responses_in_output_tables':False,
              'source_code_sha256':{str(p.relative_to(root)):sha256(p) for p in [Path(__file__).resolve(),root/'scripts/zhengzhou/audit_predictor_archive.py',root/'wetland_coupling/maxent_inputs.py']}}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'development_cells':manifest['development_cells']}),flush=True)


if __name__=='__main__':main()
