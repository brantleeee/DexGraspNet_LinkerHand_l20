"""Generate scale-matched L20 poses, validate with SDF, and export index.html.

Runs commands sequentially. Use --plan-only to inspect the complete run without
launching generation, Isaac Sim, or any GPU task.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import trimesh

from l20_object_catalog import OBJECTS,ROOT,mesh_path


def resume_suite(output,isolate_cases=False):
    """Use the saved plan, never regenerate an existing pose file."""
    plan=json.loads((output/'plan.json').read_text())
    if Path(plan['output']).resolve()!=output:raise ValueError('Resume output differs from saved plan')
    logs=output/'logs';logs.mkdir(exist_ok=True)
    def status(**values):(output/'status.json').write_text(json.dumps(values,indent=2))
    try:
        for obj,record in plan['objects'].items():
            if hashlib.sha256(Path(record['mesh']).read_bytes()).hexdigest()!=record['mesh_sha256']:
                raise ValueError(f'Mesh changed: {obj}')
            result=Path(record['result_file'])
            if not result.is_file():
                raise ValueError(f'Generated poses missing; refusing to regenerate during resume: {result}')
            manifest_path=output/obj/'inputs/manifest.json'
            command=record['validate_command'].copy();command[0]=sys.executable
            if isolate_cases and '--isolate-cases' not in command:command.append('--isolate-cases')
            if manifest_path.exists():
                manifest=json.loads(manifest_path.read_text())
                if hashlib.sha256(result.read_bytes()).hexdigest()!=manifest['source_sha256']:
                    raise ValueError(f'Generated poses changed: {obj}')
                command.append('--resume')
            status(status='running',object=obj,phase='resume_validate')
            print(f'{obj}: resume validation (completed candidates are audited and reused)',flush=True)
            with (logs/f'{obj}_resume_validate.log').open('a') as f:
                subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        status(status='finished')
    except subprocess.CalledProcessError as exc:
        status(status='failed',error=str(exc))
        raise SystemExit(f'Validation still incomplete: {obj}. Inspect {logs}/{obj}_resume_validate.log and {output/obj}/candidate_*/error.json. Completed results are preserved.') from None
    except BaseException as exc:
        status(status='failed',error=str(exc));raise
    finally:
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_l20_isaacsim.py'),
                        '--results',str(output)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects',nargs='+',choices=OBJECTS,default=['banana','bottle','camera'])
    p.add_argument('--scale',type=float,default=.06)
    p.add_argument('--batch-size-per-scale',type=int,default=16)
    p.add_argument('--n-iter',type=int,default=6000)
    p.add_argument('--gravity-seconds',type=float,default=10.)
    p.add_argument('--close-seconds',type=float,default=2.)
    p.add_argument('--settle-seconds',type=float,default=1.)
    p.add_argument('--open-fraction',type=float,default=.6)
    p.add_argument('--sdf-resolution',type=int,default=256)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true',help='Use saved plan; audit completed candidates and retry incomplete ones without generation')
    p.add_argument('--isolate-cases',action='store_true',help='Run each direction of pending candidates in a fresh process')
    a=p.parse_args()
    if a.resume:
        if a.plan_only:p.error('--resume and --plan-only cannot be combined')
        resume_suite(a.output.resolve(),a.isolate_cases);return
    if len(set(a.objects))!=len(a.objects):p.error('Duplicate objects')
    if not all(math.isfinite(x) for x in [a.scale,a.gravity_seconds,a.close_seconds,a.settle_seconds,a.open_fraction]):p.error('Nonfinite parameter')
    if a.scale<=0 or a.gravity_seconds<=0 or a.close_seconds<=0 or a.settle_seconds<0 or not 0<a.open_fraction<=1:p.error('Invalid scale or phase parameters')
    if a.batch_size_per_scale<1 or a.n_iter<1 or a.sdf_resolution<32:p.error('Invalid batch size, iterations or SDF resolution')
    steps=round(a.gravity_seconds*240)
    if steps<1:p.error('Gravity duration shorter than one physics step')
    output=a.output.resolve()
    if output.exists() and not a.plan_only:p.error('Output must be new')
    suffix=hashlib.sha256(str(output).encode()).hexdigest()[:8]
    prefix=re.sub('[^A-Za-z0-9_-]','_',output.name)[:40]
    records={}
    for obj in a.objects:
        mesh=mesh_path(obj)
        if not mesh.is_file():p.error(f'Mesh missing: {mesh}')
        geometry=trimesh.load(mesh,force='mesh',process=True)
        if geometry.volume<=0:p.error(f'Nonpositive mesh volume: {mesh}')
        name=f'l20_{prefix}_{obj}_{suffix}'
        gen=ROOT/'data/experiments'/name
        if gen.exists() and not a.plan_only:p.error(f'Generation directory already exists: {gen}')
        result=gen/'results'/f'{OBJECTS[obj]["code"]}.npy'
        generate=[sys.executable,str(ROOT/'scripts/generate_small_grasps.py'),
                  '--object-code',OBJECTS[obj]['code'],'--scales',str(a.scale),
                  '--batch-size-per-scale',str(a.batch_size_per_scale),'--n-iter',str(a.n_iter),
                  '--hull-inflation','scale-aware','--name',name]
        validate=[sys.executable,str(ROOT/'scripts/batch_l20_isaacsim.py'),'--object',obj,
                  '--scale',str(a.scale),'--result-file',str(result),'--protocol','acquire',
                  '--close-seconds',str(a.close_seconds),'--settle-seconds',str(a.settle_seconds),
                  '--open-fraction',str(a.open_fraction),'--steps',str(steps),
                  '--sdf-resolution',str(a.sdf_resolution),'--output',str(output/obj)]
        if a.isolate_cases:validate.append('--isolate-cases')
        records[obj]=dict(code=OBJECTS[obj]['code'],label=OBJECTS[obj]['label'],mesh=str(mesh),
                          mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
                          extents_mm=(geometry.extents*a.scale*1000).tolist(),
                          mass_kg=geometry.volume*a.scale**3*500,
                          mesh_kind='recovered_coacd_mesh_not_original',count=a.batch_size_per_scale,
                          generation_directory=str(gen),result_file=str(result),
                          generate_command=generate,validate_command=validate)
    plan=dict(objects=records,expected_candidates=len(a.objects)*a.batch_size_per_scale,
              expected_directions=len(a.objects)*a.batch_size_per_scale*6,scale=a.scale,
              protocol='acquire',gravity_steps=steps,gravity_seconds=steps/240,
              close_seconds=a.close_seconds,settle_seconds=a.settle_seconds,
              sdf_resolution=a.sdf_resolution,output=str(output))
    if a.plan_only:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    output.mkdir(parents=True);logs=output/'logs';logs.mkdir()
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    def status(**values):(output/'status.json').write_text(json.dumps(values,indent=2))
    try:
        for obj,record in records.items():
            for phase,key in [('generate','generate_command'),('validate','validate_command')]:
                if hashlib.sha256(Path(record['mesh']).read_bytes()).hexdigest()!=record['mesh_sha256']:
                    raise ValueError(f'Mesh changed during run: {obj}')
                status(status='running',object=obj,phase=phase)
                log=logs/f'{obj}_{phase}.log'
                print(f'{obj}: {phase}; log={log}',flush=True)
                with log.open('w') as f:
                    subprocess.run(record[key],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        status(status='finished')
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_l20_isaacsim.py'),'--results',str(output)],cwd=ROOT,check=True)
        print(f'Complete: {output}/index.html',flush=True)
    except BaseException as exc:
        status(status='failed',error=str(exc))
        print(f'Failed; inspect {logs}. Existing outputs are preserved.',file=sys.stderr,flush=True)
        # A single numerical failure must not hide all completed results.
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_l20_isaacsim.py'),
                        '--results',str(output)],cwd=ROOT,check=False)
        raise


if __name__=='__main__':main()
