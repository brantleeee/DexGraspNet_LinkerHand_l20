"""Admit a failed reference baseline and launch a new independent v4 policy."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.reference_workflow import admission,validate_baseline
from isaacsim_grasp.reference import load_reference,digest
from run_isaacsim_residual_v4 import implementation_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-results',type=Path,required=True);p.add_argument('--num-envs',type=int,default=24)
    p.add_argument('--iterations',type=int,default=200);p.add_argument('--rollout-steps',type=int,default=32)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();folder=a.baseline_results.resolve();config=json.loads((folder/'config.json').read_text())
    if a.num_envs<6 or a.num_envs%6 or min(a.iterations,a.rollout_steps)<1:p.error('Invalid dimensions or lengths')
    old=json.loads((folder/'reference_snapshot.json').read_text());ref=load_reference(old['source_results'],old['object'],old['candidate'])
    baseline=validate_baseline(folder,ref,config,json.loads((folder/'perturbations.json').read_text()))
    version=config['version'].removeprefix('isaacsim_sdf_residual_')
    if version=='v5':
        from run_isaacsim_residual_v5 import implementation_hashes as check_hashes
    elif version=='v4':check_hashes=implementation_hashes
    else:p.error('This workflow supports v4/v5 only')
    if config['implementation_sha256']!=check_hashes():p.error('Baseline/training implementation changed')
    gate=admission(baseline['trials'])
    if not gate['suitable']:p.error(f"Reference not admitted for gravity-only learning: {gate}")
    if a.output.exists() and not a.plan_only:p.error('Output must be new')
    command=['bash',str(ROOT/f'scripts/isaacsim_residual_{version}.sh'),'--mode','train','--object',ref['object'],'--candidate',str(ref['candidate']),
        '--reference-results',ref['source_results'],'--num-envs',str(a.num_envs),'--iterations',str(a.iterations),'--rollout-steps',str(a.rollout_steps),
        '--residual-scale',str(config['residual_scale']),'--residual-rate',str(config['residual_rate']),
        '--position-noise',str(config['position_noise']),'--close-seconds',str(config['close_seconds']),
        '--settle-seconds',str(config['settle_seconds']),'--gravity-seconds',str(config['gravity_seconds']),
        '--seed',str(config['seed']),'--output',str(a.output.resolve())]
    if version=='v5':command+=['--preload-seconds',str(config['preload_seconds'])]
    record=dict(baseline=str(folder),baseline_sha256=digest(folder/'baseline.json'),admission=gate,command=command,
        independent_policy=True,note='No checkpoint is loaded; this policy belongs only to the baseline reference')
    check=[sys.executable,str(ROOT/f'scripts/run_isaacsim_residual_{version}.py'),*command[2:]]
    check[check.index('--mode')+1]='plan'
    expected=json.loads(subprocess.run(check,cwd=ROOT,check=True,capture_output=True,text=True).stdout)['config']
    if {k:v for k,v in expected.items() if k!='num_envs'}!={k:v for k,v in config.items() if k!='num_envs'}:
        p.error('New training settings differ from baseline; rebuild a matched baseline')
    if a.plan_only:print(json.dumps(record,ensure_ascii=False,indent=2));return
    subprocess.run(command,cwd=ROOT,check=True)
    (a.output/'baseline_admission.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
