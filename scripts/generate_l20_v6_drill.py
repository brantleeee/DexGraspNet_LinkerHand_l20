"""Run clean-drill L20 generation using the frozen v6 point and normal library."""
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='l20_v2_clean_drill_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--contact-library',type=Path,default=ROOT/'data/hand_contacts/l20_contact_v6_confirmed_20261117')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--n-iter',type=int,default=6000)
    p.add_argument('--top-k',type=int,default=16);p.add_argument('--w-normal',type=float,default=1.)
    p.add_argument('--root-contact-policy',choices=['all','angle-only','all-roots-angle-only'],default='all-roots-angle-only')
    p.add_argument('--penetration-geometry',choices=['legacy-convex','raw-mesh'],default='raw-mesh')
    p.add_argument('--contact-sampling',choices=['global','grouped4','grouped5'],default='grouped5')
    p.add_argument('--background',action='store_true');a=p.parse_args()
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name):p.error('Invalid name')
    if a.batch_size<1 or a.n_iter<1 or not 1<=a.top_k<=a.batch_size:p.error('Invalid batch/iteration/top-k')
    out=ROOT/'data/experiments'/a.name;job=out.with_suffix('.job.json');log=out.with_suffix('.log')
    if out.exists():p.error('Experiment already exists')
    if a.background:
        if job.exists() or log.exists():p.error('Job name already exists')
        command=[sys.executable,str(Path(__file__).resolve()),'--name',a.name,'--contact-library',str(a.contact_library.resolve()),'--batch-size',str(a.batch_size),'--n-iter',str(a.n_iter),'--top-k',str(a.top_k),'--w-normal',str(a.w_normal),'--root-contact-policy',a.root_contact_policy,'--penetration-geometry',a.penetration_geometry,'--contact-sampling',a.contact_sampling]
        initial=dict(phase='launching',output=str(out),log=str(log),command=command)
        job.write_text(json.dumps(initial,indent=2)+'\n')
        with log.open('x') as f:process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        print(json.dumps(dict(pid=process.pid,output=str(out),log=str(log),status=str(job)),indent=2));return
    def status(phase,**extra):job.write_text(json.dumps(dict(phase=phase,output=str(out),log=str(log),**extra),indent=2)+'\n')
    status('preflight',pid=os.getpid())
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required; do not silently run the TorchSDF generation on CPU')
        sys.path.insert(0,str(ROOT/'grasp_generation'))
        from utils.contact_library import load_library
        manifest,_=load_library(a.contact_library)
        if manifest['version']!='l20-user-review-v6-20261117' or manifest['point_count']!=403:raise ValueError('Expected frozen v6, 403 points')
        mesh=ROOT/'data/meshdata/power-drill-clean-l20/coacd/decomposed.obj'
        imported=json.loads((mesh.parent.parent/'import.json').read_text())
        digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest(mesh)!=imported['prepared_mesh_sha256'] or digest(imported['source'])!=imported['source_sha256']:raise ValueError('Clean drill source/prepared asset identity changed')
        environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        command=[sys.executable,str(ROOT/'scripts/generate_small_grasps.py'),'--object-code','power-drill-clean-l20','--scales','1','--batch-size-per-scale',str(a.batch_size),'--n-iter',str(a.n_iter),'--hull-inflation','scale-aware','--export-top-k',str(a.top_k),'--contact-library',str(a.contact_library.resolve()),'--w-normal',str(a.w_normal),'--name',a.name,'--root-contact-policy',a.root_contact_policy,'--penetration-geometry',a.penetration_geometry,'--contact-sampling',a.contact_sampling]
        status('generating',pid=os.getpid(),command=command,contact_library=str(a.contact_library.resolve()))
        subprocess.run(command,cwd=ROOT,env=environment,check=True)
        pose=out/'results/scales/scale_1/power-drill-clean-l20.npy'
        status('exporting_canvas',pid=os.getpid())
        subprocess.run([sys.executable,str(ROOT/'scripts/export_grasps_canvas.py'),'--object-code','power-drill-clean-l20','--result-file',str(pose),'--contact-library',str(a.contact_library.resolve()),'--top-k',str(a.top_k),'--output',str(pose.parent/f'top_{a.top_k}_canvas')],cwd=ROOT,env=environment,check=True)
        status('complete',html=str(pose.parent/f'top_{a.top_k}_canvas/index.html'),result=str(pose))
        print('Complete:',out,flush=True)
    except BaseException as exc:
        status('failed',error=str(exc));raise


if __name__=='__main__':main()
