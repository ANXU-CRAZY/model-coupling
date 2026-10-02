"""Audit real complete-pool models and descriptive paired development outcomes.

--partial inspects only completed checkpoints and cannot approve a full run.
Never reads locked-test metrics or uses comparison outcomes for selection.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.zhengzhou.run_full_pool_development import load, development_split_plan, audit_engine
from scripts.zhengzhou.run_maxent_nested_cv import context, mask_rows, seeded, final_feature_plan, SEASONS, VARIANTS, SCHEMES, stamp
from scripts.zhengzhou.evaluate_maxent_cv import parse_convergence
from wetland_coupling.maxent_inputs import sample_background, dataframe_cells
from wetland_coupling.maxent_protocol import sha256, write_json, select_predictors, metrics, aggregate_candidates, choose_candidate, assert_oof_groups


def close(left,right,reason,atol=1e-9):
    a=np.asarray(left,float);b=np.asarray(right,float)
    if a.shape!=b.shape or not np.allclose(a,b,rtol=1e-10,atol=atol,equal_nan=True):raise ValueError(reason)


def metric_match(rebuilt,recorded):
    for key,value in rebuilt.items():
        other=recorded[key]
        if value is None:
            if other is not None and not (isinstance(other,float) and np.isnan(other)):raise ValueError('Metric mismatch: '+key)
        elif isinstance(value,(int,float,bool)):close([value],[other],'Metric mismatch: '+key,atol=1e-12)


def prediction_values(engine,expected_coordinates=None):
    pieces=[];offset=0
    for path in engine['prediction_files']:
        frame=pd.read_csv(path)
        if frame.shape[1]!=3 or 'cloglog' not in frame.columns[-1].lower():raise ValueError('Prediction schema')
        if expected_coordinates is not None:
            expected=expected_coordinates[offset:offset+len(frame)]
            if expected.shape!=frame.iloc[:,:2].shape or not np.allclose(frame.iloc[:,:2],expected,rtol=0,atol=5.1e-10):raise ValueError('Prediction order/coordinates changed')
        values=frame.iloc[:,-1].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid cloglog values')
        pieces.append(values);offset+=len(frame)
    values=np.concatenate(pieces)
    if len(values)!=engine['prediction_rows']:raise ValueError('Prediction row count')
    return values


def scope_audit(checkpoint,run,ctx,plan,config,consumed):
    done=load(checkpoint);parts=checkpoint.relative_to(run).parts
    variant,season,label,stage,background=parts[:5]
    if label=='full_development':
        if not stage.startswith('inner_'):raise ValueError('Unexpected deployment model')
        fold=plan['final_inner_folds'][int(stage.split('_')[1])]
    else:
        outer=next(f for f in plan['outer_folds'] if label==f"outer_{f['fold']}")
        fold=outer['inner_folds'][int(stage.split('_')[1])] if stage.startswith('inner_') else outer
    assert_oof_groups(fold['validation_groups'],fold['fit_groups'],fold['fit_groups'],plan['locked_groups'])
    seed=seeded(config['seed'],f'{season}/{label}/{int(stage.split("_")[1])}/{background}' if stage.startswith('inner_') else f'{season}/{label}/refit/{background}')
    bg=sample_background(background,fold['fit_mask'],ctx,ctx['visits'],ctx['B0'],config,seed)
    candidates=[n for n in ctx['names'] if variant=='static_lst' or not n.startswith('lst_')]
    selected,dropped=select_predictors(bg,candidates,config['predictors']['absolute_spearman_cutoff'])
    scope=Path(done['scope']);selection=load(scope/'selection.json')
    if selected!=selection['predictors'] or dropped!=selection['dropped_predictors']:raise ValueError('Fold-only selection mismatch')
    if set(selected)|set(dropped)!=set(candidates):raise ValueError('Candidate pool incomplete')
    train=mask_rows(ctx['presence'],fold['fit_mask'])
    for filename,source in (('train.csv',train),('background.csv',bg)):
        saved=pd.read_csv(scope/filename)
        if list(saved.columns)!=['species','longitude','latitude',*selected]:raise ValueError('Actual fit columns differ')
        close(saved.iloc[:,1:],source[['longitude','latitude',*selected]],'Actual fit values differ',atol=5.1e-10)
        consumed.add(scope/filename)
    if selection['background_audit']!=bg.attrs['background_audit']:raise ValueError('Fit-only background differs')
    pieces=[train,mask_rows(ctx['reference'],fold['fit_mask']),mask_rows(ctx['presence'],fold['validation_mask']),mask_rows(ctx['reference'],fold['validation_mask'])]
    offsets=np.cumsum([0,*[len(f) for f in pieces]]).tolist()
    if offsets!=selection['offsets'] or selection['projection_start']!=offsets[-1]:raise ValueError('Evaluation offsets differ')
    coordinates=pd.concat(pieces,ignore_index=True)[['longitude','latitude']].to_numpy(float)
    is_inner=stage.startswith('inner_')
    if not is_inner:
        indices=np.flatnonzero(fold['validation_mask'])
        grids=[]
        for start in range(0,len(indices),50000):
            frame=dataframe_cells(indices[start:start+50000],ctx['env'],ctx['names'],ctx['grid'])
            grids.append(frame[['longitude','latitude']].to_numpy(float))
        coordinates=np.concatenate([coordinates,*grids]);del grids
    records=done['rows'] if is_inner else [dict(done['provenance'],**done['metrics'],artifact=done['provenance']['engine_manifest'])]
    if is_inner and len(records)!=12:raise ValueError('Incomplete candidate batch')
    numerical=[];refit_prediction=None
    for row in records:
        manifest=Path(row['artifact']);engine=audit_engine(manifest)
        if engine['jar_sha256']!=config['maxent']['jar_sha256'] or engine['maxent_version']!=config['maxent']['version']:raise ValueError('Unexpected official runtime')
        if engine['rm']!=row['rm'] or engine['fc']!=row['fc'] or engine['inputs']['train']['columns'][3:]!=selected:raise ValueError('Actual engine design mismatch')
        if engine['inputs']['projection']['rows']!=len(coordinates):raise ValueError('Engine projection membership changed')
        values=prediction_values(engine,coordinates)
        metric_match(metrics(values[offsets[0]:offsets[1]],values[offsets[1]:offsets[2]],values[offsets[2]:offsets[3]],values[offsets[3]:offsets[4]],engine['nonzero_feature_count']),row)
        if row.get('candidate_predictor_count')!=(23 if variant=='no_lst' else 25):raise ValueError('Wrong candidate count')
        if row['fit_groups']!=fold['fit_groups'] or row['validation_groups']!=fold['validation_groups']:raise ValueError('Model group provenance differs')
        html=list(manifest.parent.glob('*.html'))
        if len(html)!=1:raise ValueError('Official termination evidence missing')
        termination=parse_convergence(html[0].read_text(encoding='utf-8',errors='replace'),engine['java_metrics'].get('Iterations'),config['maxent']['max_iterations'])
        numerical.append({'season':season,'variant':variant,'scope':label,'stage':stage,'background':background,'rm':row['rm'],'fc':row['fc'],
            'engine_manifest':str(manifest.resolve()),'candidate':is_inner,**termination})
        consumed.update([manifest,*html,*[Path(p) for p in engine['prediction_files']]])
        if not is_inner:refit_prediction=values[offsets[-1]:]
    if not is_inner:
        provenance=done['provenance']
        if sha256(provenance['engine_manifest'])!=provenance['engine_manifest_sha256'] or provenance['tune_groups']!=fold['fit_groups'] or provenance['calibrate_groups']:
            raise ValueError('Refit engine/tuning provenance differs')
    consumed.update([checkpoint,scope/'selection.json',scope/'projection_cleanup.json'])
    receipt=load(scope/'projection_cleanup.json')
    if receipt['status']!='REMOVED' or any(Path(r['path']).exists() for r in receipt['files']):raise ValueError('Projection cleanup incomplete')
    return {'season':season,'variant':variant,'scope':label,'stage':stage,'background':background,'candidate_count':len(candidates),
        'selected_count':len(selected),'train_presence_n':len(train),'train_background_n':len(bg),'models':len(records),
        'cleaned_bytes':receipt['bytes'],'candidate_schema_and_fit_only_selection_verified':True},numerical,refit_prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','inputs','splits','parent-run','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];run=args.run.resolve();out=args.out.resolve();out.relative_to(root/'local_work')
    if out.exists():raise FileExistsError(out)
    state=load(run/'manifests/run_manifest.json');config=load(run/'manifests/config_snapshot.json')
    if not args.partial and state['status']!='DEVELOPMENT_NESTED_CV_COMPLETE':raise ValueError('Training not complete')
    if state['locked_test_used'] or (run/'manifests/LOCKED_TEST_ATTEMPT.json').exists() or list(run.rglob('locked_test_metrics.csv')):raise ValueError('Forbidden test activity')
    for name,expected in state['source_code_sha256'].items():
        if sha256(root/name)!=expected:raise ValueError('Running source changed')
    if sha256(args.inputs/'manifest.json')!=state['input_manifest_sha256'] or sha256(args.splits/'manifest.json')!=state['split_manifest_sha256']:raise ValueError('Input/split identity changed')
    input_manifest=load(args.inputs/'manifest.json')
    for name,meta in input_manifest['outputs'].items():
        if sha256(args.inputs/name)!=meta['sha256']:raise ValueError('Prepared input changed')
    plan=development_split_plan(args.splits);consumed=set([args.inputs/'manifest.json',args.splits/'manifest.json',run/'manifests/config_snapshot.json'])
    protected=load(run/'manifests/parent_protection_snapshot.json')
    for name,meta in protected['files'].items():
        if sha256(name)!=meta['sha256']:raise ValueError('Parent protected file changed')
    checkpoints=sorted(run.rglob('completed.json'));scopes=[];numerical=[];refits={}
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'run_snapshot.json',state)
    for season in SEASONS:
        ctx=context(args.inputs,season)
        for checkpoint in checkpoints:
            if checkpoint.relative_to(run).parts[1]!=season:continue
            row,convergence,prediction=scope_audit(checkpoint,run,ctx,plan,config,consumed)
            scopes.append(row);numerical.extend(convergence)
            if prediction is not None:refits[(season,row['variant'],row['scope'],row['background'])]=prediction
        print(json.dumps({'audit_season':season,'completed_scopes_audited':len(scopes),'models_audited':len(numerical)}),flush=True)
        del ctx;gc.collect()
    paired=[];summary=[];oof=[];choices=[];stability=[];selected_inner_paths=set()
    if not args.partial:
        if len(numerical)!=3528 or len(checkpoints)!=360:raise ValueError('Completed model/scope counts differ')
        for name,meta in state['outputs'].items():
            if sha256(run/name)!=meta['sha256']:raise ValueError('Final output changed')
        outer=pd.read_csv(run/'reports/outer_metrics.csv')
        if len(outer)!=72 or outer.duplicated(['season','variant','fold','background']).any():raise ValueError('Outer table completeness')
        selections=load(run/'manifests/development_selection.json')['selections']
        parent_state=load(args.parent_run/'manifests/run_manifest.json')
        parent_records={k.replace('\\','/'):v for k,v in parent_state['outputs'].items()}
        for season in SEASONS:
            ctx=context(args.inputs,season)
            for label in [*[f"outer_{f['fold']}" for f in plan['outer_folds']],'full_development']:
                records=[r for p in checkpoints if p.relative_to(run).parts[1:3]==(season,label) for r in load(p).get('rows',[])]
                aggregate=aggregate_candidates(records);tuning=load(run/'reports'/f'tuning_{season}_{label}.json')
                if aggregate!=tuning['all_candidates']:raise ValueError('Inner aggregation changed')
                for variant in VARIANTS:
                    chosen=choose_candidate([r for r in aggregate if r['variant']==variant],config['selection']['mean_omission_eligibility_max'])
                    if chosen!=tuning['variant_winners'][variant]:raise ValueError('Selection rule changed')
                    if label=='full_development' and chosen!=selections[season]['variant_winners'][variant]:raise ValueError('Deployment candidate mismatch')
                    if label=='full_development':
                        actual=final_feature_plan(ctx,plan['development_fit_mask'],chosen['background'],variant,config,seeded(config['seed'],f"{season}/frozen_final/{chosen['background']}"))
                        if actual!=selections[season]['final_features'][variant]:raise ValueError('Deployment fit-only feature plan changed')
                    choices.append({'season':season,'scope':label,'variant':variant,**chosen})
                    selected_inner_paths.update(str(Path(r['artifact']).resolve()) for r in records if
                        r['variant']==variant and r['background']==chosen['background'] and r['rm']==chosen['rm'] and r['fc']==chosen['fc'])
                    for background in SCHEMES:
                        if choose_candidate([r for r in aggregate if r['variant']==variant and r['background']==background],config['selection']['mean_omission_eligibility_max'])!=tuning['scheme_winners'][variant][background]:raise ValueError('Background winner mismatch')
                if label=='full_development' and choose_candidate(aggregate,config['selection']['mean_omission_eligibility_max'])!=selections[season]['deployment_choice']:raise ValueError('Joint candidate mismatch')
            for variant in VARIANTS:
                path=run/'oof'/f'{season}_{variant}.npz';consumed.add(path)
                with np.load(path,allow_pickle=False) as data:
                    expected=plan['development_fit_mask'].ravel()
                    if not np.array_equal(np.isfinite(data['M_oof']),expected) or np.any(np.isfinite(data['M_oof'][plan['locked_mask'].ravel()])):raise ValueError('OOF domain mismatch')
                    for fold in plan['outer_folds']:
                        label=f"outer_{fold['fold']}";indices=np.flatnonzero(fold['validation_mask']);tuning=load(run/'reports'/f'tuning_{season}_{label}.json')
                        values=np.stack([refits[(season,variant,label,b)] for b in SCHEMES]);primary=SCHEMES.index(tuning['variant_winners'][variant]['background'])
                        close(data['M_oof'][indices],values[primary],'OOF primary mismatch',atol=6e-8)
                        close(data['std'][indices],values.std(axis=0),'OOF uncertainty mismatch',atol=6e-8)
                        for key,values_q in zip(('q05','median','q95'),np.quantile(values,[.05,.5,.95],axis=0)):close(data[key][indices],values_q,'OOF quantile mismatch',atol=6e-8)
                        if not np.all(data['fold'][indices]==fold['fold']) or not np.all(data['member_count'][indices]==3):raise ValueError('OOF fold/member mismatch')
                    oof.append({'season':season,'variant':variant,'finite_cells':int(expected.sum()),'locked_cells_predicted':0,'members':3})
                    old_path=args.parent_run/'oof'/f'{season}_{variant}.npz'
                    if sha256(old_path)!=parent_records[f'oof/{season}_{variant}.npz']['sha256']:raise ValueError('V1 OOF changed')
                    consumed.add(old_path)
                    with np.load(old_path,allow_pickle=False) as old_oof:
                        if not np.array_equal(np.isfinite(old_oof['M_oof']),expected) or not np.array_equal(old_oof['fold'],data['fold']):raise ValueError('V1 paired raster domain/fold mismatch')
                        a=old_oof['M_oof'][expected];b=data['M_oof'][expected]
                        ta=a>=np.quantile(a,.9);tb=b>=np.quantile(b,.9)
                        stability.append({'season':season,'variant':variant,'paired_cells':len(a),'spearman':float(spearmanr(a,b).statistic),
                            'top_decile_jaccard':float(np.sum(ta&tb)/np.sum(ta|tb)),'ties_retained_at_quantile':True,'descriptive_only':True})
            del ctx;gc.collect()
        parent_path=args.parent_run/'reports/outer_metrics.csv'
        if sha256(parent_path)!=parent_records['reports/outer_metrics.csv']['sha256']:raise ValueError('V1 outer metrics changed')
        consumed.add(parent_path);old=pd.read_csv(parent_path)
        keys=['season','variant','fold','background'];pair=outer.merge(old,on=keys,suffixes=('_v2','_v1'),validate='one_to_one')
        if len(pair)!=72 or not np.array_equal(pair.validation_presence_n_v2,pair.validation_presence_n_v1) or not np.array_equal(pair.validation_background_n_v2,pair.validation_background_n_v1):raise ValueError('Paired evaluation samples differ')
        paired=pair[keys+['validation_auc_v1','validation_auc_v2','omission_10_v1','omission_10_v2']].copy()
        paired['auc_difference']=paired.validation_auc_v2-paired.validation_auc_v1
        for (season,variant),f in paired.loc[paired.background.eq('B1_uniform')].groupby(['season','variant']):
            summary.append({'season':season,'variant':variant,'folds':len(f),'v1_mean_auc':float(f.validation_auc_v1.mean()),'v2_mean_auc':float(f.validation_auc_v2.mean()),
                'mean_auc_difference':float(f.auc_difference.mean()),'v1_mean_omission10':float(f.omission_10_v1.mean()),'v2_mean_omission10':float(f.omission_10_v2.mean()),
                'comparison_is_descriptive_development_only':True})
        paired.to_csv(out/'paired_outer_metrics.csv',index=False)
    pd.DataFrame(scopes).to_csv(out/'fit_scope_audit.csv',index=False)
    pd.DataFrame(numerical).to_csv(out/'model_convergence.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'paired_B1_summary.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'inner_selected_candidates.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'paired_raster_stability.csv',index=False)
    candidate_numerical=[r for r in numerical if r['candidate']];outer_numerical=[r for r in numerical if not r['candidate']]
    report={'status':'PARTIAL_COMPLETED_SCOPES_VALIDATED_NOT_FULL_RUN_APPROVAL' if args.partial else 'COMPLETE_SUPPLIED_POOL_DEVELOPMENT_AUDIT_PASSED_WITH_LIMITATIONS',
        'partial':args.partial,'checkpoints_audited':len(scopes),'models_audited':len(numerical),'inner_models':len(candidate_numerical),'outer_models':len(outer_numerical),
        'candidate_iteration_limit_reached':sum(bool(r['iteration_limit_reached']) for r in candidate_numerical),
        'outer_iteration_limit_reached':sum(bool(r['iteration_limit_reached']) for r in outer_numerical),
        'convergence_unverified':sum(not r['convergence_verified'] for r in numerical),
        'chosen_candidate_inner_models':sum(r['engine_manifest'] in selected_inner_paths for r in numerical),
        'chosen_candidate_inner_iteration_limit_reached':sum(bool(r['iteration_limit_reached']) for r in numerical if r['engine_manifest'] in selected_inner_paths),
        'cleaned_projection_bytes':sum(r['cleaned_bytes'] for r in scopes),'oof':oof,'paired_B1':summary,
        'paired_raster_stability':stability,
        'locked_test_used':False,'locked_metrics_read':False,'candidate_selection_modified':False,
        'strict_end_to_end_oof':False,'gate_eligible':False,'independent_management_labels':0,
        'limitations':['Supplied pool upstream curation, source period/units/QC unverified','Development folds previously used; comparison is descriptive','Pooled seasonal candidate community, no calibrated occurrence probability','No approved LULC crosswalk or independent HQ/management supervision'],
        'run_started_at_utc':state['started_at_utc'],'audit_ended_at_utc':stamp()}
    write_json(out/'summary.json',report)
    consumed.add(run/'manifests/parent_protection_snapshot.json')
    manifest={'status':report['status'],'argv':sys.argv,'source_code_sha256':sha256(Path(__file__)),
        'run_snapshot_sha256':sha256(out/'run_snapshot.json'),'inputs':{str(p.resolve()):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(consumed)},
        'outputs':{str(p.relative_to(out)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}}
    write_json(out/'manifest.json',manifest);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
