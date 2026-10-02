"""Unlabelled development-domain land-cover compatibility, not class admission.

Archived component names are descriptors, not confirmed class semantics.
Matching a categorical raster to component maxima is not field validation.
No bird responses, suitability outputs or locked-test metrics are read.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from scipy.optimize import linear_sum_assignment
from scripts.zhengzhou.run_full_pool_development import development_split_plan, load
from wetland_coupling.maxent_protocol import sha256, write_json
from scripts.zhengzhou.run_maxent_nested_cv import stamp


FAMILIES=('water','trees','grass','floodedvegetation','crops','shrubscrub','builtarea','bareground','largebuildings')


def compatibility(labels,components):
    labels=np.asarray(labels);values=np.asarray(components,float)
    if not len(labels) or labels.ndim!=1 or values.shape!=(len(labels),9) or not np.isfinite(values).all() or not np.isin(labels,np.arange(1,10)).all():raise ValueError('Invalid compatibility inputs')
    labels=labels.astype(int)
    totals=values.sum(axis=1)
    probability_like=bool(np.all((values>=0)&(values<=1)) and np.mean(np.abs(totals-1)<=2e-5)>=.99)
    confusion=np.zeros((9,9),dtype=np.int64)
    if probability_like:
        maximum=np.argmax(values,axis=1)
        np.add.at(confusion,(labels-1,maximum),1)
    match=[]
    if probability_like:
        rows,cols=linear_sum_assignment(-confusion)
        for r,c in zip(rows,cols):
            n=int(confusion[r].sum())
            match.append({'numeric_code':int(r+1),'hypothesized_component_descriptor':FAMILIES[c],
                'class_pixels':n,'matched_component_maximum_pixels':int(confusion[r,c]),
                'within_class_component_maximum_fraction':float(confusion[r,c]/n) if n else None,
                'not_ground_truth_accuracy':True,'class_semantics_confirmed':False})
    return {'component_values_probability_like':probability_like,'sum_min':float(totals.min()),'sum_max':float(totals.max()),
        'sum_median':float(np.median(totals)),'sum_within_2e_minus5_of_one_fraction':float(np.mean(np.abs(totals-1)<=2e-5)),
        'maximum_ties':int(np.sum(np.sum(values==values.max(axis=1,keepdims=True),axis=1)>1)),
        'assignment_tie_rule':'Global count maximum via scipy linear_sum_assignment; component argmax uses listed family order',
        'max_assignment_agreement_fraction':float(sum(r['matched_component_maximum_pixels'] for r in match)/len(labels)) if match else None},confusion,match


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','splits','lulc','lulc-evidence','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2];out=args.out.resolve();out.relative_to(root/'local_work')
    if out.exists():raise FileExistsError(out)
    source=load(args.inputs/'manifest.json');evidence=load(args.lulc_evidence)
    if sha256(args.lulc)!=evidence['inputs'][str(args.lulc.resolve())]['sha256']:raise ValueError('LULC evidence changed')
    grid=load(args.inputs/'grid.json');plan=development_split_plan(args.splits)
    if sha256(args.splits/'manifest.json')!=source['parent_split_manifest_sha256']:raise ValueError('Unexpected partition')
    with rasterio.open(args.lulc) as ds:
        if ds.crs.to_epsg()!=32649 or ds.transform!=Affine(*grid['transform']) or ds.shape!=(grid['height'],grid['width']):raise ValueError('LULC alignment differs')
        categorical=ds.read(1);valid=ds.read_masks(1)>0
    valid &= plan['development_fit_mask'] & np.isin(categorical,np.arange(1,10))
    indices=np.flatnonzero(valid);labels=categorical.ravel()[indices].astype(int)
    if not len(indices):raise ValueError('No compatible development cells')
    out.mkdir(parents=True,exist_ok=False);started=stamp();seasonal={};rows=[];matches=[];profiles=[]
    consumed=[args.inputs/'manifest.json',args.splits/'manifest.json',args.lulc,args.lulc_evidence,args.inputs/'grid.json']
    for season in ('spring','summer','autumn','winter'):
        path=args.inputs/season/'env.npy';name_path=args.inputs/season/'names.json'
        for item in (path,name_path):
            if sha256(item)!=source['outputs'][str(item.relative_to(args.inputs))]['sha256']:raise ValueError('Component source changed')
        consumed.extend([path,name_path]);names=load(name_path);columns=[names.index(f'{family}_{season}') for family in FAMILIES]
        env=np.load(path,mmap_mode='r',allow_pickle=False).reshape(-1,len(names))
        values=np.concatenate([env[indices[s:s+50000]][:,columns] for s in range(0,len(indices),50000)])
        summary,confusion,assignment=compatibility(labels,values);seasonal[season]=summary
        for code in range(1,10):
            group=values[labels==code];n=len(group)
            for index,family in enumerate(FAMILIES):
                rows.append({'season':season,'numeric_code':code,'component_descriptor':family,'class_pixels':n,
                    'component_maximum_evaluated':summary['component_values_probability_like'],
                    'component_maximum_pixels':int(confusion[code-1,index]) if summary['component_values_probability_like'] else None,
                    'component_maximum_fraction':float(confusion[code-1,index]/n) if n and summary['component_values_probability_like'] else None})
                profiles.append({'season':season,'numeric_code':code,'component_descriptor':family,'class_pixels':n,
                    'mean':float(group[:,index].mean()) if n else None,'median':float(np.median(group[:,index])) if n else None})
        matches.extend([{'season':season,**row} for row in assignment]);del env,values
    consistent=[]
    for code in range(1,10):
        found=[r for r in matches if r['numeric_code']==code];descriptors=sorted({r['hypothesized_component_descriptor'] for r in found})
        consistent.append({'numeric_code':code,'hypothesized_descriptors':descriptors,'same_assignment_all_four_seasons':len(found)==4 and len(descriptors)==1,'class_semantics_confirmed':False})
    pd.DataFrame(rows).to_csv(out/'component_maximum_cross_table.csv',index=False)
    pd.DataFrame(matches,columns=['season','numeric_code','hypothesized_component_descriptor','class_pixels','matched_component_maximum_pixels',
        'within_class_component_maximum_fraction','not_ground_truth_accuracy','class_semantics_confirmed']).to_csv(out/'hypothesized_matching.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'component_class_profiles.csv',index=False)
    result={'status':'EXPLORATORY_UNLABELLED_COMPATIBILITY_NOT_A_CLASS_CROSSWALK','development_cells':len(indices),
        'families_are_archived_descriptors':list(FAMILIES),'seasonal':seasonal,'assignment_consistency':consistent,
        'class_semantics_confirmed':False,'source_period_and_product_confirmed':False,'independent_accuracy_validation':False,
        'bird_responses_read':False,'locked_metrics_read':False,'official_hq_parameters_changed':False,'independent_supervision_created':0,
        'interpretation':'Numerical association and globally optimized correspondence only; shared imagery or classifier may make agreement circular. Seasonal components and current categorical raster may differ in period or aggregation.'}
    write_json(out/'summary.json',result)
    write_json(out/'manifest.json',{'status':result['status'],'started_at_utc':started,'ended_at_utc':stamp(),'argv':sys.argv,
        'source_code_sha256':sha256(Path(__file__)),'inputs':{str(p.resolve()):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in consumed},
        'outputs':{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}})
    print(result)


if __name__=='__main__':main()
