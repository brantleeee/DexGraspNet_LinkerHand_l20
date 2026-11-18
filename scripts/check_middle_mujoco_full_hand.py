"""Whole-hand native MuJoCo SDF self-collision sweep; all 22 links and 231 pairs.

Kinematic configuration checking, not a motor-driven dynamics or continuous CCD proof.
"""
import argparse,copy,csv,hashlib,html,itertools,json,os,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
if '--video' in sys.argv:os.environ.setdefault('MUJOCO_GL','egl')
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1]
PATHS=['mcp_only','pip_only','coupled','mcp_then_pip','pip_then_mcp']
CONTEXTS=['neutral','others_closed','candidate_21','candidate_7']


def fmt(v):return ' '.join(format(float(x),'.17g') for x in v)


def make_scene(output,root_contact_policy='all'):
    source=ROOT/'grasp_generation/mjcf/linkerhand_l20_right.urdf';robot=ET.parse(source).getroot();urdf=copy.deepcopy(robot)
    compiler=ET.SubElement(ET.SubElement(urdf,'mujoco'),'compiler',discardvisual='true',fusestatic='false')
    for mesh in urdf.findall('.//mesh'):mesh.set('filename',str((source.parent/mesh.get('filename')).resolve()))
    model=mujoco.MjModel.from_xml_string(ET.tostring(urdf,encoding='unicode'))
    imported=output/'hand_import.xml';mujoco.mj_saveLastXML(str(imported),model);scene=ET.parse(imported).getroot()
    scene.set('model','l20_full_hand_middle_sdf_sweep')
    option=scene.find('option')
    if option is None:option=ET.SubElement(scene,'option')
    option.set('gravity','0 0 0');option.set('sdf_iterations','20');option.set('sdf_initpoints','40')
    option.set('timestep','0.001')
    contact=scene.find('contact')
    if contact is not None:scene.remove(contact)
    contact=ET.SubElement(scene,'contact');geoms=[]
    for body in scene.findall('.//body'):
        for j,g in enumerate(body.findall('geom')):
            if g.get('mesh') is None:raise ValueError('Expected original collision mesh on every link')
            g.set('name','collision_'+body.get('name')+(f'_{j}' if j else ''));g.set('type','sdf')
            g.set('contype','0');g.set('conaffinity','0');g.set('group','3');g.set('margin','0')
            g.set('rgba','0.95 0.65 0.15 1' if body.get('name').startswith('middle_') else '0.45 0.7 0.85 1')
            geoms.append((body.get('name'),g.get('name')))
    if len(geoms)!=22 or len({x[0] for x in geoms})!=22:raise ValueError('Incomplete hand import: '+str(geoms))
    for (left,lg),(right,rg) in itertools.combinations(geoms,2):
        ET.SubElement(contact,'pair',name=left+'__'+right,geom1=lg,geom2=rg,condim='1',margin='0',gap='0')
    eq=scene.find('equality')
    if eq is None:eq=ET.SubElement(scene,'equality')
    for joint in robot.findall('joint'):
        mimic=joint.find('mimic')
        if mimic is None:continue
        if not any(x.get('joint1')==joint.get('name') for x in eq.findall('joint')):
            ET.SubElement(eq,'joint',name='mimic_'+joint.get('name'),joint1=joint.get('name'),joint2=mimic.get('joint'),polycoef=fmt([float(mimic.get('offset','0')),float(mimic.get('multiplier','1')),0,0,0]))
    ET.SubElement(scene.find('worldbody'),'light',pos='0 -0.3 .5',dir='0 0 -1')
    from l20_root_collision_policy import apply_root_policy
    disabled=apply_root_policy(scene,root_contact_policy)
    path=output/'scene.xml';path.write_text(ET.tostring(scene,encoding='unicode'))
    model=mujoco.MjModel.from_xml_path(str(path))
    if model.ngeom!=22 or model.npair!=231-len(disabled) or not np.all(model.geom_type==int(mujoco.mjtGeom.mjGEOM_SDF)):raise ValueError('Incomplete native SDF/pair setup')
    assets=[source]+[Path(m.get('filename')) for m in urdf.findall('link/collision/geometry/mesh')]
    audit=dict(mujoco_version=mujoco.__version__,links=[x[0] for x in geoms],collision_geoms=model.ngeom,
        explicit_pairs=model.npair,all_pairs_explicit=not disabled,automatic_masks_disabled=True,root_contact_policy=root_contact_policy,disabled_pairs=disabled,
        collision_representation='22 native mesh SDF geoms from original URDF collision STLs; no hull/decomposition/simplification',
        equality_count=model.neq,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets},
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='All remaining pairs explicit; disabled root pairs listed separately. SDF is a numerical approximation; open source meshes may have sign ambiguity.')
    (output/'collision_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return model,robot


def path_angles(kind,step,mcp_max=1.4,pip_max=1.57):
    if not np.isfinite(step) or step<=0:raise ValueError('Invalid angular step')
    corners={'mcp_only':[(0,0),(mcp_max,0)],'pip_only':[(0,0),(0,pip_max)],'coupled':[(0,0),(mcp_max,pip_max)],
        'mcp_then_pip':[(0,0),(mcp_max,0),(mcp_max,pip_max)],'pip_then_mcp':[(0,0),(0,pip_max),(mcp_max,pip_max)]}[kind]
    result=[np.array(corners[0],dtype=float)]
    for start,end in zip(corners,corners[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(np.array(end)-start))/step)))
        result.extend(np.linspace(start,end,n+1)[1:])
    return np.asarray(result)


def targets_for(robot,context,source):
    joints={j.get('name'):j for j in robot.findall('joint') if j.get('type')!='fixed'}
    target={n:float(np.clip(0,float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))) for n,j in joints.items()}
    if context.startswith('candidate_'):
        row=np.load(source,allow_pickle=True)[int(context.split('_')[1])];target={n:float(row['qpos'][n]) for n in joints}
    if context=='others_closed':
        for f in ['index','ring','pinky']:target[f+'_mcp_pitch']=1.4;target[f+'_pip']=1.57
    target['middle_mcp_roll']=0.
    if context.startswith('candidate_'):target['middle_mcp_roll']=float(row['qpos']['middle_mcp_roll'])
    return target


def project_mimic(robot,target):
    result=dict(target)
    for j in robot.findall('joint'):
        m=j.find('mimic')
        if m is not None:result[j.get('name')]=float(m.get('multiplier','1'))*result[m.get('joint')]+float(m.get('offset','0'))
    return result


def collect(model,data):
    pairs={}
    for c in data.contact:
        left=model.body(int(model.geom_bodyid[c.geom1])).name;right=model.body(int(model.geom_bodyid[c.geom2])).name
        key=' / '.join(sorted([left,right]));depth=max(0.,-float(c.dist))*1000
        if key not in pairs or depth>pairs[key]['depth_mm']:
            pairs[key]=dict(left=left,right=right,depth_mm=depth,position_m=c.pos.tolist(),normal=c.frame[:3].tolist())
    return pairs


def playback(output,case,video=None):
    record=json.loads((output/'results.json').read_text());case=case or record['cases'][0]['name']
    spec=next(x for x in record['cases'] if x['name']==case);trace=np.load(output/spec['trajectory'])
    model=mujoco.MjModel.from_xml_path(str(output/'scene.xml'));data=mujoco.MjData(model);poses=trace['qpos']
    def load(i):
        data.qpos[:]=poses[i];data.qvel[:]=0;mujoco.mj_forward(model,data)
        model.geom_rgba[:]=[.45,.7,.85,1]
        for gi in range(model.ngeom):
            if model.body(int(model.geom_bodyid[gi])).name.startswith(spec.get('finger','middle')+'_'):model.geom_rgba[gi]=[.95,.65,.15,1]
        for c in data.contact:
            if c.dist<-.00001:model.geom_rgba[[c.geom1,c.geom2]]=[.9,.1,.15,1]
    if video:
        import imageio.v2 as imageio
        camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.14];camera.distance=.42;camera.azimuth=135;camera.elevation=-20
        option=mujoco.MjvOption();option.geomgroup[:]=1;option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True
        with mujoco.Renderer(model,height=480,width=640) as renderer,imageio.get_writer(video,fps=30,codec='libx264',quality=8) as writer:
            for i in np.linspace(0,len(poses)-1,181).round().astype(int):
                load(i);renderer.update_scene(data,camera=camera,scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                writer.append_data(renderer.render())
        print('Video:',video);return
    from mujoco import viewer as mjviewer
    state=dict(paused=False,frame=0,start=time.monotonic())
    def key_callback(key):
        if key==32:
            state['paused']=not state['paused']
            if not state['paused']:state['start']=time.monotonic()-state['frame']/max(1,len(poses)-1)*8
        elif key in [262,263]:
            state['paused']=True;state['frame']=int(np.clip(state['frame']+(1 if key==262 else -1),0,len(poses)-1))
    print('GUI: space=pause/resume, left/right=single saved frame, mouse=rotate/zoom',flush=True)
    with mjviewer.launch_passive(model,data,key_callback=key_callback) as viewer:
        viewer.cam.lookat[:]=[0,0,.14];viewer.cam.distance=.42;viewer.cam.azimuth=135;viewer.cam.elevation=-20
        viewer.opt.geomgroup[:]=1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True
        while viewer.is_running():
            if not state['paused']:
                phase=((time.monotonic()-state['start'])%8)/8;state['frame']=min(len(poses)-1,int(phase*len(poses)))
            load(state['frame']);viewer.sync();time.sleep(1/60)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--contexts',choices=CONTEXTS,nargs='+',default=CONTEXTS);p.add_argument('--paths',choices=PATHS,nargs='+',default=PATHS)
    p.add_argument('--max-step-rad',type=float,default=.01);p.add_argument('--depth-threshold-mm',type=float,default=.01)
    p.add_argument('--root-contact-policy',choices=['all','angle-only','all-roots-angle-only'],default='all-roots-angle-only')
    p.add_argument('--source-results',type=Path,default=ROOT/'data/experiments/l20_v6_clean_drill_20261117_135743/results/scales/scale_1/power-drill-clean-l20.npy')
    p.add_argument('--replay',action='store_true');p.add_argument('--video',type=Path);p.add_argument('--case');a=p.parse_args()
    if a.replay or a.video:playback(a.output,a.case,a.video);return
    if a.output.exists():p.error('Use a new output directory')
    if not np.isfinite([a.max_step_rad,a.depth_threshold_mm]).all() or not 0<a.max_step_rad<=.1 or a.depth_threshold_mm<0:p.error('Invalid angular step/depth threshold')
    a.output.mkdir(parents=True);model,robot=make_scene(a.output,a.root_contact_policy);data=mujoco.MjData(model)
    joints=[j.get('name') for j in robot.findall('joint') if j.get('type')!='fixed'];qids={n:int(model.joint(n).qposadr[0]) for n in joints}
    all_cases=[];all_pair_keys=set();root_start=time.monotonic()
    for context in a.contexts:
        base=targets_for(robot,context,a.source_results)
        for path in a.paths:
            name=context+'__'+path;folder=a.output/name;folder.mkdir();angles=path_angles(path,a.max_step_rad)
            frames=[];poses=[];pair_stats={};baseline=set();peak_depth=0.;max_mimic_error=0.
            for frame,(mcp,pip) in enumerate(angles):
                target=dict(base,middle_mcp_pitch=float(mcp),middle_pip=float(pip));target=project_mimic(robot,target)
                mujoco.mj_resetData(model,data)
                for n,q in target.items():data.qpos[qids[n]]=q
                mujoco.mj_forward(model,data)
                if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):raise RuntimeError('MuJoCo warning/nonfinite state')
                observed=collect(model,data);active={k for k,v in observed.items() if v['depth_mm']>a.depth_threshold_mm}
                if frame==0:baseline=active.copy()
                for key,c in observed.items():
                    if key not in pair_stats:pair_stats[key]=dict(left=c['left'],right=c['right'],first_frame=frame,first_mcp=float(mcp),first_pip=float(pip),baseline=key in baseline,max_depth_mm=0.,frames_over_threshold=0,middle_involved=c['left'].startswith('middle_') or c['right'].startswith('middle_'))
                    s=pair_stats[key];s['max_depth_mm']=max(s['max_depth_mm'],c['depth_mm']);s['frames_over_threshold']+=int(c['depth_mm']>a.depth_threshold_mm)
                    if c['depth_mm']>a.depth_threshold_mm and 'first_over_threshold_frame' not in s:s.update(first_over_threshold_frame=frame,first_over_threshold_mcp=float(mcp),first_over_threshold_pip=float(pip))
                    peak_depth=max(peak_depth,c['depth_mm'])
                for j in robot.findall('joint'):
                    mimic=j.find('mimic')
                    if mimic is not None:
                        err=abs(data.qpos[qids[j.get('name')]]-float(mimic.get('multiplier','1'))*data.qpos[qids[mimic.get('joint')]]-float(mimic.get('offset','0')));max_mimic_error=max(max_mimic_error,float(err))
                poses.append(data.qpos.copy());frames.append(dict(frame=frame,mcp=float(mcp),pip=float(pip),dip=float(target['middle_dip']),contact_count=data.ncon,pairs=observed))
            np.savez_compressed(folder/'trajectory.npz',qpos=np.asarray(poses),mcp=angles[:,0],pip=angles[:,1],joint_names=joints)
            (folder/'contacts.json').write_text(json.dumps(frames,indent=2)+'\n')
            stats=list(pair_stats.values());new=[s for s in stats if not s['baseline'] and s['frames_over_threshold']]
            entry=dict(name=name,context=context,path=path,frames=len(frames),trajectory=str(Path(name)/'trajectory.npz'),baseline_pairs=sorted(baseline),new_collision_pairs=new,pair_summary=stats,peak_depth_mm=peak_depth,max_mimic_error_rad=max_mimic_error,complete=True)
            all_cases.append(entry);all_pair_keys.update(pair_stats)
            result=dict(complete=False,engine='MuJoCo '+mujoco.__version__,mode='kinematic qpos sweep + mj_forward; no object, gravity or motor-driven dynamics',
                max_step_rad=a.max_step_rad,depth_threshold_mm=a.depth_threshold_mm,cases=all_cases,runtime_seconds=time.monotonic()-root_start,
                source_results=str(a.source_results.resolve()),root_contact_policy=a.root_contact_policy,pairs=model.npair,note='Finite angular samples do not certify continuous collision-free motion or the entire 2-DOF configuration space. Pair policy is recorded in collision_audit.json.')
            (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(name,'frames',len(frames),'baseline',len(baseline),'new_pairs',len(new),'peak_mm',round(peak_depth,5),flush=True)
    result['complete']=True;(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# MuJoCo中指完整弯曲：全手自碰撞检查','',f'22个原始碰撞link，22个原生SDF，显式{model.npair}个link配对；根部策略{a.root_contact_policy}；角步长≤{a.max_step_rad}rad，报告深度阈值{a.depth_threshold_mm}mm。',
        '','运动学逐帧指定qpos并调用mj_forward；不作动力学可达或连续无碰撞认证。没有物体或支撑平面。基线接触不自动忽略，新增按同背景同路径起点区分。','', '|背景/路径|帧数|起点碰撞对|新碰撞对|最大深度mm|','|---|---:|---:|---:|---:|']
    for c in all_cases:lines.append(f'|{c["name"]}|{c["frames"]}|{len(c["baseline_pairs"])}|{len(c["new_collision_pairs"])}|{c["peak_depth_mm"]:.5f}|')
    lines+=['','## 中指相关新碰撞','', '|背景/路径|连杆对|首次MCP/PIP rad|最大深度mm|','|---|---|---|---:|']
    for c in all_cases:
        for s in c['new_collision_pairs']:
            if s['middle_involved']:lines.append(f'|{c["name"]}|{s["left"]}/{s["right"]}|{s["first_over_threshold_mcp"]:.5f}/{s["first_over_threshold_pip"]:.5f}|{s["max_depth_mm"]:.5f}|')
    lines+=['','原STL可能开放/包含装配交叠，原生SDF是数值碰撞近似；不能将接触深度直接当原始三角面精确穿透。全体基线/新增/最大深度和逐帧接触见JSON及NPZ。']
    text='\n'.join(lines)+'\n';(a.output/'REPORT.md').write_text(text)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><pre>'+html.escape(text)+'</pre>')
    print('Complete:',a.output.resolve(),flush=True)


if __name__=='__main__':main()
