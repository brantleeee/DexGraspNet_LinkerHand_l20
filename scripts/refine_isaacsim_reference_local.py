"""Gated bounded reference search in fresh processes; never PPO/source overwrite."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.reference_variants import proposals,variant_targets
from isaacsim_grasp.isolated import compare_traces


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-results',type=Path,required=True)
    p.add_argument('--joints',nargs='+',help='Independent joints; default guided by lowest force fingers in failing directions')
    p.add_argument('--joint-deltas',nargs='+',help='One coupled candidate, e.g. index_pip=0.02 middle_pip=0.02; no coordinate sweep')
    p.add_argument('--delta-rad',type=float,default=.015)
    p.add_argument('--max-joint-proposals',type=int,default=6)
    p.add_argument('--allow-wrist',action='store_true',help='Only if joints fail to increase pass count: <=1mm/1degree wrist trials')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit=a.audit_results.resolve();out=a.output.resolve()
    if out.exists() and not a.plan_only:p.error('Output must be new')
    if a.max_joint_proposals<1:p.error('Use a positive proposal budget')
    verdict=json.loads((audit/'reproducibility.json').read_text())
    if not verdict.get('process_reproducible',verdict['reference_refinement_admitted']):
        p.error('Zero-zero process reproducibility failed; do not refine')
    spec=json.loads((audit/'worker_plan.json').read_text());base=json.loads((audit/'zero_a/result.json').read_text())
    if spec.get('variant'):p.error('Export/validate the derived source before starting another refinement stage')
    flags=np.array(base['info']['passed'],bool)
    if flags.all():p.error('All six pass in this fresh-process sample; select/recheck a genuinely failing sample first')
    ref=spec['reference'];joints=a.joints
    if not joints:
        with np.load(audit/'zero_a/trajectory.npz') as data:
            k=int(np.argmin(abs(data['time']-3.)));forces=data['finger_normal_force'][k,~flags,:5].mean(0)
        names=['thumb_mcp','index_pip','middle_pip','ring_pip','pinky_pip']
        joints=[names[i] for i in np.argsort(forces)]
    if a.joint_deltas:
        try:
            changes=dict((item.split('=',1)[0],float(item.split('=',1)[1])) for item in a.joint_deltas)
            variant=dict(parent_scene_sha256=ref['scene_sha256'],parent_targets=ref['targets'],name='coupled_joints',joint_deltas_rad=changes)
            variant_targets(ref,variant)
        except (ValueError,IndexError) as exc:p.error(str(exc))
        variants=[variant]
    else:variants=proposals(ref,joints,a.delta_rad)[:a.max_joint_proposals]
    wrist=proposals(ref,[],a.delta_rad,True) if a.allow_wrist else []
    plan=dict(audit_results=str(audit),source_reference=ref,config=spec['config'],samples=spec['samples'],
        diagnostic_selected_joints=joints,joint_proposals=variants,wrist_proposals_if_no_joint_pass_gain=wrist,
        baseline_pass_count=int(flags.sum()),workers_per_proposal=2,
        selection='must preserve all baseline successful directions and reproduce in fresh processes; single sample local candidate only')
    if a.plan_only:print(json.dumps(plan,indent=2));return
    out.mkdir(parents=True);(out/'plan.json').write_text(json.dumps(plan,indent=2))
    print(f'[refine] {len(variants)} joint candidate(s), two fresh processes each; output={out}',flush=True)
    (out/'status.json').write_text(json.dumps(dict(status='starting')))
    records=[];selected=None;best_count=int(flags.sum());best_margin=float('inf')
    def test_variant(v):
        nonlocal selected,best_count,best_margin
        folder=out/v['name'];folder.mkdir();worker=dict(spec,variant=v,controller='zero_residual')
        path=folder/'worker_plan.json';path.write_text(json.dumps(worker,indent=2))
        for repeat in ['a','b']:
            logpath=folder/f'{repeat}.log'
            print(f'[refine] {v["name"]}: repeat {repeat}/b starting; log={logpath}',flush=True)
            (out/'status.json').write_text(json.dumps(dict(status='running',candidate=v['name'],repeat=repeat,log=str(logpath))))
            command=['bash',str(ROOT/'scripts/isaacsim_isolated_case.sh'),'--plan',str(path),'--output',str(folder/repeat)]
            with logpath.open('w') as log:
                with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,bufsize=1,env=dict(os.environ,PYTHONUNBUFFERED='1')) as process:
                    for line in process.stdout:
                        log.write(line);log.flush()
                        if '[isolated]' in line or 'queued until' in line or line.startswith('Complete:'):
                            print(line.rstrip(),flush=True)
                    returncode=process.wait()
                if returncode:
                    (out/'status.json').write_text(json.dumps(dict(status='failed',candidate=v['name'],repeat=repeat,returncode=returncode,log=str(logpath))))
                    raise subprocess.CalledProcessError(returncode,command)
            if json.loads((folder/repeat/'status.json').read_text())['status']!='complete':raise RuntimeError('Worker incomplete')
        with np.load(folder/'a/trajectory.npz') as x,np.load(folder/'b/trajectory.npz') as y:
            repeatability=compare_traces(x,y,prefix_seconds=13.)
        r=json.loads((folder/'a/result.json').read_text());rb=json.loads((folder/'b/result.json').read_text())
        f=np.array(r['info']['passed'],bool);count=int(f.sum())
        reproducible=repeatability['passed'] and r['info']['passed']==rb['info']['passed']
        preserved=bool(np.all(f[flags]));margin=max([r['info']['max_slip_m'][i]/.005+r['info']['max_angle_deg'][i]/15 for i in np.flatnonzero(f)] or [float('inf')])
        accepted=reproducible and preserved and count>int(flags.sum())
        record=dict(name=v['name'],passed=count,preserved=preserved,reproducible=reproducible,repeatability=repeatability,eligible=accepted)
        records.append(record);(out/'results.json').write_text(json.dumps(records,indent=2))
        if accepted and (count>best_count or (count==best_count and margin<best_margin)):
            best_count=count;best_margin=margin;selected=folder/'a'
        print(v['name'],record['passed'],record['preserved'],record['reproducible'],flush=True)
    for v in variants:test_variant(v)
    if best_count==int(flags.sum()):
        for v in wrist:test_variant(v)
    result=dict(complete=True,baseline_pass_count=int(flags.sum()),selected_run=str(selected) if selected else None,
        best_pass_count=best_count,verified_scope='one saved translation, six directions, repeated fresh processes',
        rrl_training_allowed=False,note='Confirm across other translations before accepting a refined reference; never load old policy on changed targets')
    if selected:
        bundle=out/'best_reference';bundle.mkdir()
        snapshot=json.loads((selected/'reference_snapshot.json').read_text())
        (bundle/'reference.json').write_text(json.dumps(snapshot,indent=2));(bundle/'validation.json').write_text(json.dumps(result,indent=2))
        result['reference_bundle']=str(bundle)
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'status.json').write_text(json.dumps(dict(status='complete')))
    (out/'REPORT.md').write_text('# 局部参考细化\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
