"""L20 asset/joint adaptation of the pinned official DexGraspNet IsaacValidator."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external/DexGraspNet_official/grasp_generation'
sys.path.insert(0,str(SOURCE))
# Gym must be imported before numpy/torch in its Python 3.8 runtime.
from utils.isaac_validator import IsaacValidator,gym
from isaacgym import gymapi
import numpy as np
from l20_object_catalog import OBJECTS


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def contacts(sim):
    return [sum((c[2] in h and c[3] in o) or (c[3] in h and c[2] in o)
                for c in gym.get_env_rigid_contacts(env))
            for env,h,o in zip(sim.envs,sim.hand_rigid_body_sets,sim.obj_rigid_body_sets)]


def write_report(summary,output):
    note='官方 Gym 协议的 L20 适配：100 步为原始时长；末帧六方向接触 + 源文件 E_pen < 0.001。'
    limits='手/物体使用 Gym 凸碰撞；21 关节均位置驱动，mimic 仅目标满足联动，无硬机械耦合。official 表示接触调整的 L20 适配，含梯度联动回传及限位投影，不是未修改的官方 ShadowHand 实验。'
    md=['# L20 · 官方 Gym 协议适配验证','',note,'',limits,'',
        f"仿真步数：{summary['sim_steps']}，时长：{summary['sim_steps']/60:.6f} 秒；执行完成：{summary['completed']}。",'',
        '|物体|处理|候选|数值有效|六方向接触通过|加 E_pen 后通过|','|---|---|---:|---:|---:|---:|']
    table=[]
    for run in summary['runs']:
        rows=run['candidates'];values=[run['object'],run['variant'],len(rows),sum(r['numerical_valid'] for r in rows),sum(r['simulated_pass'] for r in rows),sum(r['valid'] for r in rows)]
        md.append('|'+ '|'.join(map(str,values))+'|')
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    md+=['','逐候选、逐方向接触、实际 mimic 偏差和质量见 summary.json；初末状态见 *_states.npz。',
         '数值异常不算通过；普通无接触属于已执行但抓取未通过。没有额外施加滑移/转角/持续接触阈值。']
    (output/'REPORT.md').write_text('\n'.join(md)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>L20 Gym 验证</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}p{line-height:1.7}</style><h1>L20 官方 Gym 协议适配验证</h1><p>'+html.escape(note)+'</p><p>'+html.escape(limits)+'</p><p>执行完成：'+str(summary['completed'])+'</p><table><tr>'+''.join('<th>'+x+'</th>' for x in ['物体','处理','候选','数值有效','六方向接触通过','加 E_pen 后通过'])+'</tr>'+''.join(table)+'</table><p><a href="summary.json">逐候选 JSON</a> · <a href="REPORT.md">完整报告</a></p>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--objects',choices=OBJECTS,nargs='+',default=list(OBJECTS))
    p.add_argument('--variants',choices=['official','no-force'],nargs='+',default=['official','no-force'])
    p.add_argument('--indices',type=int,nargs='+',help='Original NPY indices, not packed row numbers')
    p.add_argument('--sim-steps',type=int,default=100)
    p.add_argument('--gui',action='store_true')
    p.add_argument('--autostart',action='store_true')
    p.add_argument('--viewer-seconds',type=float,default=0)
    p.add_argument('--display-speed',type=float,default=1)
    a=p.parse_args();inputs=a.inputs.resolve();out=a.output.resolve()
    if out.exists():p.error('Output must be new')
    if a.sim_steps<1 or len(set(a.objects))!=len(a.objects) or len(set(a.variants))!=len(a.variants):p.error('Invalid steps or duplicate selection')
    if not np.isfinite(a.display_speed) or a.display_speed<=0 or not np.isfinite(a.viewer_seconds) or a.viewer_seconds<0:p.error('Invalid viewer parameters')
    if a.gui and (len(a.objects)!=1 or len(a.variants)!=1 or len(a.indices or [])!=1):p.error('GUI requires one object, variant and original index')
    manifest=json.loads((inputs/'manifest.json').read_text())
    for filename,expected in manifest['files_sha256'].items():
        if hashlib.sha256((inputs/filename).read_bytes()).hexdigest()!=expected:raise ValueError(f'Prepared input changed: {filename}')
    if any(v not in manifest['variants'] for v in a.variants):p.error('Requested variant was not prepared')
    if any(obj not in manifest['objects'] for obj in a.objects):p.error('Requested object was not prepared')
    out.mkdir(parents=True)
    summary=dict(completed=False,engine='Isaac Gym Preview 4 GPU PhysX',hand='L20',
        protocol='official_DexGraspNet_adapted_L20_not_ShadowHand',sim_steps=a.sim_steps,
        parameters=dict(dt=1/60,substeps=2,gravity=[0,-9.8,0],stiffness=1000,damping=0,
            friction=3,density=500,solver_type=1,position_iterations=8,velocity_iterations=0,
            contact_offset=.01,rest_offset=0,hand_gravity=False,fixed_wrist=True,
            original_steps=100,seconds=a.sim_steps/60,zero_gravity_closure=False),
        source_validator_sha256=hashlib.sha256((SOURCE/'utils/isaac_validator.py').read_bytes()).hexdigest(),
        input_manifest=manifest,runs=[])
    save(out/'summary.json',summary)
    try:
        for obj in a.objects:
            with np.load(inputs/f'{obj}.npz') as data:arrays=dict(data)
            available=list(map(int,arrays['indices']));indices=a.indices if a.indices is not None else available
            if len(set(indices))!=len(indices) or any(i not in available for i in indices):p.error(f'Unknown/duplicate original index for {obj}')
            for variant in a.variants:
                print(f'{obj} {variant}: {len(indices)} candidates x 6 directions, {a.sim_steps} steps',flush=True)
                sim=IsaacValidator(gpu=0,sim_step=a.sim_steps,mode='gui' if a.gui else 'direct')
                try:
                    if sim.sim is None:raise RuntimeError('GPU simulator creation failed')
                    sim.joint_names=manifest['joint_names']
                    sim.set_asset(str(inputs/'assets/hand'),'l20.urdf',str(inputs/'assets'/obj),'object.urdf')
                    if sim.hand_asset is None or sim.obj_asset is None:raise RuntimeError('Asset loading failed')
                    actual=list(gym.get_asset_dof_names(sim.hand_asset))
                    if set(actual)!=set(sim.joint_names) or len(actual)!=21:raise RuntimeError(f'Unexpected L20 DOFs: {actual}')
                    q=arrays['force_qpos'] if variant=='official' else arrays['raw_qpos']
                    for index in indices:
                        k=available.index(index)
                        sim.add_env(arrays['rotations'][k],arrays['translations'][k],q[k],float(arrays['scales'][k]))
                    initial=[gym.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL).copy() for env,actor in zip(sim.envs,sim.obj_handles)]
                    start=gym.get_sim_time(sim.sim);viewer=None
                    if a.gui:
                        from shadowhand_gym_viewer import GymGraspViewer
                        viewer=GymGraspViewer(sim,gym,out,a.autostart,a.viewer_seconds,a.display_speed);viewer.run()
                    else:sim.run_sim()
                    gym.fetch_results(sim.sim,True)
                    elapsed=gym.get_sim_time(sim.sim)-start
                    if abs(elapsed-a.sim_steps/60)>.01:raise RuntimeError('Incomplete physics duration')
                    counts=contacts(sim);results=[];states={}
                    for row,index in enumerate(indices):
                        k=available.index(index);directions=[]
                        for direction in range(6):
                            n=row*6+direction;env=sim.envs[n]
                            final=gym.get_actor_rigid_body_states(env,sim.obj_handles[n],gymapi.STATE_ALL).copy()
                            hand=gym.get_actor_dof_states(env,sim.hand_handles[n],gymapi.STATE_ALL).copy()
                            xyz=np.column_stack([final['pose']['p'][v] for v in 'xyz'])
                            quat=np.column_stack([final['pose']['r'][v] for v in 'xyzw'])
                            velocity=np.column_stack([final['vel'][kind][v] for kind in ['linear','angular'] for v in 'xyz'])
                            finite=all(np.isfinite(v).all() for v in [xyz,quat,velocity,hand['pos'],hand['vel']])
                            errors=[float(abs(hand['pos'][actual.index(m['child'])]-float(m.get('multiplier',1))*hand['pos'][actual.index(m['joint'])]-float(m.get('offset',0)))) for m in manifest['mimics']]
                            directions.append(dict(rotation_index=direction,final_contact_count=int(counts[n]),final_contact=bool(counts[n]>0),
                                numerical_valid=bool(finite),final_max_mimic_error_rad=max(errors) if finite else None,
                                object_mass_kg=sum(float(b.mass) for b in gym.get_actor_rigid_body_properties(env,sim.obj_handles[n]))))
                            states[f'candidate_{index}_direction_{direction}_initial']=initial[n]
                            states[f'candidate_{index}_direction_{direction}_final']=final
                            states[f'candidate_{index}_direction_{direction}_hand_final']=hand
                        finite=all(d['numerical_valid'] for d in directions);passed=finite and all(d['final_contact'] for d in directions)
                        results.append(dict(index=index,scale=float(arrays['scales'][k]),E_pen=float(arrays['E_pen'][k]),numerical_valid=finite,
                            simulated_pass=passed,estimated_pass=bool(arrays['E_pen'][k]<.001),valid=bool(passed and arrays['E_pen'][k]<.001),directions=directions))
                    run=dict(object=obj,variant=variant,candidates=results,simulated_seconds=elapsed,
                        hand_dofs=actual,hand_rigid_shapes=gym.get_asset_rigid_shape_count(sim.hand_asset),
                        object_rigid_shapes=gym.get_asset_rigid_shape_count(sim.obj_asset),
                        actual_hand_dof_properties={key:gym.get_actor_dof_properties(sim.envs[0],sim.hand_handles[0])[key].tolist() for key in ['stiffness','damping','effort','lower','upper']})
                    summary['runs'].append(run)
                    np.savez_compressed(out/f'{obj}_{variant}_states.npz',**states)
                    save(out/'summary.json',summary);write_report(summary,out)
                    print(f'{obj} {variant}: valid={sum(r["valid"] for r in results)}/{len(results)}',flush=True)
                    if viewer is not None:viewer.hold()
                finally:sim.destroy()
        summary['completed']=True;save(out/'summary.json',summary);write_report(summary,out)
        print(f'Complete: {out}/index.html',flush=True)
    except BaseException as exc:
        save(out/'error.json',dict(type=type(exc).__name__,message=str(exc)))
        write_report(summary,out);raise


if __name__=='__main__':main()
