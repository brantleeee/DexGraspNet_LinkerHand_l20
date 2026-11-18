"""Validate an immutable ranked L20 subset in native Isaac Sim, with resumable outputs."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranking',type=Path,required=True);p.add_argument('--top-k',type=int,default=16)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--isolate-cases',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args();out=a.output.resolve();ranking=json.loads(a.ranking.read_text());source=Path(ranking['source']).resolve()
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if sha(source)!=ranking['source_sha256']:p.error('Ranked source changed')
    rows=ranking['rows'][:a.top_k]
    if a.top_k<1 or len(rows)!=a.top_k:p.error('Invalid top-k')
    raw=np.load(source,allow_pickle=True);indices=[r['candidate_index'] for r in rows]
    if len(set(indices))!=len(indices):p.error('Duplicate ranked index')
    scales={float(raw[i]['scale']) for i in indices}
    if len(scales)!=1:p.error('Use one exact scale per batch')
    scale=scales.pop();obj=ranking['object_code']
    from l20_object_catalog import OBJECTS,mesh_path
    aliases=[k for k,v in OBJECTS.items() if v['code']==obj]
    if len(aliases)!=1:p.error('Object must have a unique catalog entry')
    alias=aliases[0]
    for r in rows:
        if not np.isclose(raw[r['candidate_index']]['energy'],r['energy']) or abs(float(raw[r['candidate_index']]['scale'])-r['scale'])>1e-7:p.error('Ranking metadata differs from source')
    command=[sys.executable,str(ROOT/'scripts/batch_l20_isaacsim.py'),'--object',alias,'--scale',str(scale),
        '--result-file',str(source),'--indices',*map(str,indices),'--protocol','acquire',
        '--close-seconds','2','--settle-seconds','1','--open-fraction','0.6',
        '--steps','2400','--sdf-resolution','256','--output',str(out/alias)]
    if a.isolate_cases:command.append('--isolate-cases')
    plan=dict(expected_candidates=a.top_k,ranking=str(a.ranking.resolve()),ranking_sha256=sha(a.ranking),source=str(source),
        source_sha256=sha(source),mesh_sha256=sha(mesh_path(alias)),selected_indices=indices,ranked_rows=rows,scale=scale,
        protocol='acquire',density_kg_m3=500,close_seconds=2,settle_seconds=1,gravity_seconds=10,
        physical_mimic=True,sdf_resolution=256,case_isolation='fresh_process' if a.isolate_cases else 'shared_process',
        native_processes=a.top_k*(6 if a.isolate_cases else 1),command=command)
    if a.plan_only:print(json.dumps(plan,indent=2));return
    if a.resume:
        if not (out/'plan.json').exists() or json.loads((out/'plan.json').read_text())!=plan:p.error('Resume plan/input changed')
        command=command+['--resume']
    else:
        if out.exists():p.error('Output must be new; use --resume to reuse audited results')
        out.mkdir(parents=True);(out/'plan.json').write_text(json.dumps(plan,indent=2))
    status=out/'status.json';status.write_text(json.dumps(dict(status='running',expected_directions=6*a.top_k)))
    print(f'Validating {alias}: ranked indices={indices}; scale={scale}; {plan["native_processes"]} native processes',flush=True)
    try:
        subprocess.run(command,cwd=ROOT,check=True)
        status.write_text(json.dumps(dict(status='finished')))
    except BaseException as exc:
        status.write_text(json.dumps(dict(status='failed',error=str(exc))))
        raise
    finally:
        if (out/alias/'inputs/manifest.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/summarize_l20_isaacsim.py'),'--results',str(out)],cwd=ROOT,check=True)

if __name__=='__main__':main()
