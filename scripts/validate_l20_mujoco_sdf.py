"""Ranked L20 grasp validation: native MuJoCo mesh SDF, physical mimic, acquire protocol."""
import argparse,copy,hashlib,json,sys,time,traceback,html,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco,trimesh,transforms3d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from validate_grasp_dynamics import build_template,project_pose,vector
from validate_l20_isaacsim import pregrasp_targets
from isaacsim_grasp.history import EpisodeHistory
from isaacsim_grasp.isolated import finger_name
from utils.grasp_results import verify_mesh_identity
CASES={'gx+':[9.81,0,0],'gx-':[-9.81,0,0],'gy+':[0,9.81,0],'gy-':[0,-9.81,0],'gz+':[0,0,9.81],'gz-':[0,0,-9.81]}


def trial(job):
    folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
    try:
        model=mujoco.MjModel.from_binary_path(job['model']);data=mujoco.MjData(model)
        scale=job['scale'];q=job['qpos'];names=job['joint_names'];targets=job['targets'];opened=job['opened']
        qids=np.array([model.joint(n).qposadr[0] for n in names]);dids=np.array([model.joint(n).dofadr[0] for n in names])
        data.mocap_pos[0]=[q['WRJT'+a] for a in 'xyz'];data.mocap_quat[0]=transforms3d.euler.euler2quat(*[q['WRJR'+a] for a in 'xyz'])
        data.qpos[qids]=[opened[n] for n in names]
        drives=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in model.actuator_trnid[:,0]]
        data.ctrl[:]=[opened[n] for n in drives];mujoco.mj_forward(model,data)
        oid=model.body('object').id;ogids=set(np.flatnonzero(model.geom_bodyid==oid).tolist())
        h=EpisodeHistory(1,job['close_steps'],job['settle_steps'],job['gravity_steps'],job['dt'])
        first=None;trace=[];qtrace=[];veltrace=[];ctrltrace=[];sample_times=[];fingerforces=[];counttrace=[];max_mimic=0.
        initial_object=data.xpos[oid].copy();initial_quat=data.xquat[oid].copy()
        started=time.monotonic()
        for step in range(h.total):
            u=min(1.,(step+1)/h.close);blend=10*u**3-15*u**4+6*u**5
            data.ctrl[:]=[opened[n]+(targets[n]-opened[n])*blend for n in drives]
            if step==h.start:model.opt.gravity[:]=CASES[job['case']]
            mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):raise RuntimeError('Numerical instability/warning in MuJoCo')
            count=0;depth=0.;force=0.;fingers=np.zeros(6)
            # Contact index is required by mj_contactForce; do not use object equality lookup.
            for ci in range(data.ncon):
                c=data.contact[ci]
                if (c.geom1 in ogids)==(c.geom2 in ogids):continue
                count+=1;depth=max(depth,max(0.,-float(c.dist)))
                fi=np.zeros(6);mujoco.mj_contactForce(model,data,ci,fi);force+=abs(float(fi[0]))
                other=c.geom2 if c.geom1 in ogids else c.geom1
                bodyname=model.body(int(model.geom_bodyid[other])).name;label=finger_name(bodyname)
                fingers[['thumb','index','middle','ring','pinky','palm'].index(label)]+=abs(float(fi[0]))
            h.record(step+1,np.array([depth]),np.array([count]),np.array([force]),data.xpos[oid][None].copy(),data.xquat[oid][None].copy())
            if h.failed[0] and first is None:first=dict(time_seconds=float(data.time),gravity_seconds=float(data.time-h.start*h.dt),phase='gravity' if step+1>h.start else 'preparation',reasons=[n for n,on in zip(['slip','angle','penetration','acquisition','last_second_contact'],h.reasons[0]) if on])
            for child,parent,m,b in job['mimics']:
                err=abs(data.qpos[model.joint(child).qposadr[0]]-m*data.qpos[model.joint(parent).qposadr[0]]-b);max_mimic=max(max_mimic,float(err))
            if step%4==0 or step+1==h.total or h.failed[0]:
                trace.append([float(data.time),*data.xpos[oid],*data.xquat[oid],force,depth,float(h.max_slip[0]),float(h.max_angle[0])])
                qtrace.append(data.qpos.copy());veltrace.append(data.qvel.copy());ctrltrace.append(data.ctrl.copy());sample_times.append(float(data.time));fingerforces.append(fingers);counttrace.append(count)
            if h.failed[0]:break  # strict failure cannot recover; preserve prefix, do not claim full duration
        np.savez_compressed(folder/'trajectory.npz',time=sample_times,qpos=qtrace,qvel=veltrace,ctrl=ctrltrace,mocap_pos=np.tile(data.mocap_pos,(len(sample_times),1,1)),mocap_quat=data.mocap_quat.copy(),object_trace=trace,finger_normal_force=fingerforces,contact_count=counttrace,joint_names=names,gravity_reference_position=h.ref_pos[0],gravity_reference_quaternion=h.ref_quat[0])
        result=dict(completed=True,rank=job['rank'],candidate=job['candidate'],case=job['case'],scale=scale,passed=bool(h.passed()[0]),simulated_seconds=float(data.time),full_duration=bool(h.step>=h.total),early_termination='first irreversible strict failure',mimic_quality_pass=bool(max_mimic<=.02),first_failure=first,max_slip_mm=float(h.max_slip[0]*1000),max_angle_deg=float(h.max_angle[0]),max_penetration_mm=float(h.max_depth[0]*1000),max_mimic_error_rad=max_mimic,acquired=bool(h.acquired[0]),last_second_contact=bool(h.contact_steps[0]>=h.tail),mass_kg=float(model.body_mass[oid]),runtime_seconds=time.monotonic()-started,model=job['model'])
        (folder/'result.json').write_text(json.dumps(result,indent=2));return result
    except Exception as e:
        r=dict(completed=False,rank=job['rank'],candidate=job['candidate'],case=job['case'],error=str(e),traceback=traceback.format_exc());(folder/'error.json').write_text(json.dumps(r,indent=2));return r


def write_report(output):
    from types import SimpleNamespace
    output=Path(output);plan=json.loads((output/'plan.json').read_text())
    ranking_path=Path(plan['ranking']);ranking=json.loads(ranking_path.read_text())
    selected=ranking['rows'][:len(plan['selected_indices'])]
    results=json.loads((output/'results.json').read_text())['rows']
    directions=plan.get('cases',list(CASES));a=SimpleNamespace(output=output,ranking=ranking_path,top_k=len(selected))
    lines=['# L20电钻MuJoCo SDF动力学验证','',f'选能量前{a.top_k}个候选，逐方向测试，未过滤几何FAIL；密度500kg/m³推算质量，不是实测电钻质量。手腕固定，主动PD5/0.05、±1Nm，5个mimic等式，23个SDF碰撞。','', '力指标与Isaac Sim不完全相同；本次不是跨引擎等价对照。','', '|排名|候选|'+ '|'.join(directions)+'|通过方向数|','|---|---|'+'---|'*(len(directions)+1)]
    for row in selected:
        records={r['case']:r for r in results if r['candidate']==row['candidate_index']}
        cells=[('通过' if records[c].get('passed') else '失败') if records[c]['completed'] else '数值/执行异常' for c in directions]
        lines.append(f'|{row["rank"]}|{row["candidate_index"]}|'+ '|'.join(cells)+f'|{sum(bool(r.get("passed")) for r in records.values())}/{len(directions)}|')
    lines+=['','失败方向在首次严格失败后提前结束，simulated_seconds/full_duration逐项记录；不是所有方向都完成13秒。无物体焊接、平面支撑、外力或重力阶段重置。','', '阈值：获取接触、重力最大滑移≤5mm/转角≤15°、全程采样穿透≤1mm、末1秒连续法向接触力指标>0.001N；失败历史不可逆。']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    cells=[]
    for row in selected:
        idx=row['candidate_index'];rs={r['case']:r for r in results if r['candidate']==idx}
        original=os.path.relpath(a.ranking.parent/row['html'],a.output)
        tr=f'<tr><td>{row["rank"]}</td><td>{idx}</td><td><a href="{html.escape(original)}">生成姿态</a></td>'
        for case in directions:
            r=rs[case];label='通过' if r.get('passed') else '失败' if r['completed'] else '执行异常'
            link=f'rank_{row["rank"]:02d}_candidate_{idx:03d}/{case}/'+('result.json' if r['completed'] else 'error.json')
            failure=r.get('first_failure');extra=failure['reasons'] if failure else ''
            tr+=f'<td><a href="{link}">{label}</a><br><small>{html.escape(str(extra))}<br>{r.get("simulated_seconds",0):.4f}s</small></td>'
        tr+=f'<td>{sum(bool(r.get("passed")) for r in rs.values())}/{len(directions)}</td></tr>';cells.append(tr)
    page='<!doctype html><meta charset="utf-8"><title>电钻L20 MuJoCo SDF验证</title><style>body{font:15px/1.6 system-ui;margin:30px}td,th{padding:9px;border:1px solid #ddd}table{border-collapse:collapse}small{color:#666}</style><h1>电钻 · L20 · MuJoCo原生SDF动力学验证</h1><p>按生成能量前16候选测试，非RRL；23个SDF碰撞体，5个mimic等式。零重力闭合2秒/稳定1秒，重力10秒；首次严格失败提前结束，表中为实际模拟时长。</p><p>质量由封闭网格体积×500kg/m³推算，不是实测电钻质量。SDF及接触求解器与Isaac Sim不同，不作同条件引擎排名。</p><p><a href="REPORT.md">报告</a> · <a href="results.json">逐方向完整结果</a> · <a href="collision_audit.json">碰撞审计</a> · <a href="plan.json">配置/源候选</a></p><table><tr><th>排名</th><th>原候选下标</th><th>静态HTML</th>'+''.join('<th>'+c+'</th>' for c in directions)+'<th>方向通过数</th></tr>'+''.join(cells)+'</table>'
    (a.output/'index.html').write_text(page)
    print(a.output/'REPORT.md')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ranking',type=Path);p.add_argument('--top-k',type=int,default=16)
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));p.add_argument('--gravity-seconds',type=float,default=10.)
    p.add_argument('--timestep',type=float,default=.001);p.add_argument('--workers',type=int,default=2);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true');p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args()
    if a.summarize_only:write_report(a.output);return
    if a.ranking is None:p.error('--ranking is required')
    ranking=json.loads(a.ranking.read_text());source=Path(ranking['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=ranking['source_sha256']:p.error('Ranked source has changed')
    selected=ranking['rows'][:a.top_k]
    if len(selected)!=a.top_k or a.top_k<1 or a.workers<1 or a.gravity_seconds<1 or not 0<a.timestep<=1/240:p.error('Invalid count/workers/gravity duration')
    data=np.load(source,allow_pickle=True);code=ranking['object_code'];meshpath=ROOT/'data/meshdata'/code/'coacd/decomposed.obj';verify_mesh_identity(source,meshpath)
    if a.output.exists():p.error('Output must be new')
    plan=dict(engine=mujoco.__version__,ranking=str(a.ranking.resolve()),source=str(source),scale=selected[0]['scale'],selected_indices=[r['candidate_index'] for r in selected],trials=len(selected)*len(a.cases),cases=a.cases,workers=a.workers,mimic_constraint_solref=[.002,1.],mimic_constraint_solimp=[.999,.999,.001,.5,2.],protocol='zero gravity close2 settle1, gravity10 by default; stop failed trials at first irreversible strict failure',object_density_kg_m3=500,physics_dt=a.timestep,collision='23 native mesh SDF geoms; no convex decomposition',force_metric='sum absolute object-hand contact normal forces, differs from Isaac Sim per-link resultant norms')
    if a.plan_only:print(json.dumps(plan,indent=2));return
    a.output.mkdir(parents=True);(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
    template,robot,_=build_template(ROOT,a.output,code,mesh_sdf=True)
    option=template.find('option');option.set('timestep',str(a.timestep));option.set('sdf_initpoints','40');option.set('sdf_iterations','10')
    for g in template.findall('.//geom'):g.set('friction','3 0.005 0.0001')
    mesh=trimesh.load(meshpath,force='mesh',process=True);jobs=[];audits=[]
    joints=[dict(name=j.get('name'),type=j.get('type'),limit={k:float(v) for k,v in j.find('limit').attrib.items()} if j.find('limit') is not None else {},mimic=dict(j.find('mimic').attrib) if j.find('mimic') is not None else None) for j in robot.findall('joint')]
    names=[j['name'] for j in joints if j['type']!='fixed']
    for row in selected:
        idx=row['candidate_index'];g=data[idx];targets,feasible=project_pose(robot,g['qpos'])
        if not feasible['source_pose_feasible']:raise ValueError('Ranked pose violates URDF')
        if not np.isclose(float(g['energy']),row['energy']) or abs(float(g['scale'])-row['scale'])>1e-7:raise ValueError('Ranking mismatch')
        opened=pregrasp_targets(joints,targets,.6);xml=copy.deepcopy(template);scale=float(g['scale'])
        for m in xml.findall('asset/mesh'):
            if m.get('name').startswith('object_part_') or m.get('name')=='object_visual':m.set('scale',vector([scale]*3))
        obj=xml.find("worldbody/body[@name='object']");physical=mesh.copy();physical.vertices*=scale;mp=physical.mass_properties
        ET.SubElement(obj,'inertial',pos=vector(mp['center_mass']),mass=str(float(mp['volume']*500)),fullinertia=vector([mp['inertia'][0,0]*500,mp['inertia'][1,1]*500,mp['inertia'][2,2]*500,mp['inertia'][0,1]*500,mp['inertia'][0,2]*500,mp['inertia'][1,2]*500]))
        model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
        model.eq_solref[:]=[.002,1.]
        for eq in xml.findall('equality/joint'):eq.set('solref','0.002 1');eq.set('solimp','0.999 0.999 0.001 0.5 2')
        model.eq_solimp[:]=[.999,.999,.001,.5,2.]
        active=(model.geom_contype!=0)|(model.geom_conaffinity!=0)
        if int(active.sum())!=23 or not np.all(model.geom_type[active]==int(mujoco.mjtGeom.mjGEOM_SDF)) or model.neq!=5:raise ValueError(f'Expected 23 SDF and 5 equalities; actual active={int(active.sum())}, types={model.geom_type[active].tolist()}, neq={model.neq}')
        path=a.output/f'rank_{row["rank"]:02d}_candidate_{idx:03d}';path.mkdir();(path/'scene.xml').write_text(ET.tostring(xml,encoding='unicode'))
        mjb=path/'scene.mjb';mujoco.mj_saveModel(model,str(mjb),None)
        audits.append(dict(rank=row['rank'],candidate=idx,sdf_geoms=int(active.sum()),mimic_equalities=model.neq,mass_kg=float(model.body_mass[model.body('object').id]),scale=scale))
        for case in a.cases:
            jobs.append(dict(folder=str(path/case),model=str(mjb.resolve()),rank=row['rank'],candidate=idx,scale=scale,qpos=g['qpos'],joint_names=names,targets=targets,opened=opened,case=case,close_steps=round(2/a.timestep),settle_steps=round(1/a.timestep),gravity_steps=round(a.gravity_seconds/a.timestep),dt=a.timestep,mimics=[(j['name'],j['mimic']['joint'],float(j['mimic'].get('multiplier',1)),float(j['mimic'].get('offset',0))) for j in joints if j['mimic']]))
    (a.output/'collision_audit.json').write_text(json.dumps(audits,indent=2));results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(trial,j) for j in jobs]
        for future in as_completed(futures):
            r=future.result();results.append(r);print(f'{len(results)}/{len(jobs)} rank={r["rank"]} {r["case"]}: '+str(r.get('passed',r.get('error'))),flush=True)
            (a.output/'results.json').write_text(json.dumps(dict(complete=len(results)==len(jobs),rows=results),indent=2))
    write_report(a.output)

if __name__=='__main__':main()
