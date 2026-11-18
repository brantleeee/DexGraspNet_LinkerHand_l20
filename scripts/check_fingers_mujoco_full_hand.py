"""Per-finger full-hand native SDF sweeps, including the thumb's four independent DOFs."""
import argparse,json,html,hashlib,os,sys,time
from pathlib import Path
if '--video' in sys.argv:os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco
from check_middle_mujoco_full_hand import make_scene,collect,project_mimic,playback,ROOT


def motion_paths(robot,finger,step):
    joints={j.get('name'):j for j in robot.findall('joint') if j.get('type')!='fixed'}
    names=([finger+'_mcp_pitch',finger+'_pip'] if finger!='thumb' else ['thumb_cmc_yaw','thumb_cmc_roll','thumb_cmc_pitch','thumb_mcp'])
    lo=np.array([float(joints[n].find('limit').get('lower')) for n in names]);hi=np.array([float(joints[n].find('limit').get('upper')) for n in names])
    corners={'coupled':[lo,hi]}
    for i,n in enumerate(names):
        end=lo.copy();end[i]=hi[i];corners[n+'_only']=[lo,end]
    for label,order in [('proximal_first',range(len(names))),('distal_first',reversed(range(len(names))))]:
        chain=[lo.copy()];current=lo.copy()
        for i in order:current=current.copy();current[i]=hi[i];chain.append(current)
        corners[label]=chain
    paths={}
    for label,chain in corners.items():
        rows=[chain[0]]
        for start,end in zip(chain,chain[1:]):
            count=max(1,int(np.ceil(np.abs(end-start).max()/step)));rows.extend(np.linspace(start,end,count+1)[1:])
        paths[label]=np.array(rows)
    return names,paths


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--fingers',nargs='+',choices=['index','middle','ring','pinky','thumb'],default=['index','ring','pinky','thumb'])
    p.add_argument('--contexts',nargs='+',choices=['neutral','others_closed'],default=['neutral','others_closed'])
    p.add_argument('--root-contact-policy',choices=['all','angle-only','all-roots-angle-only'],default='all-roots-angle-only')
    p.add_argument('--max-step-rad',type=float,default=.01);p.add_argument('--replay',action='store_true');p.add_argument('--video',type=Path);p.add_argument('--case');a=p.parse_args()
    if a.replay or a.video:playback(a.output,a.case,a.video);return
    if a.output.exists():p.error('Use a new output directory')
    if not np.isfinite(a.max_step_rad) or not 0<a.max_step_rad<=.1:p.error('Invalid step')
    a.output.mkdir(parents=True);model,robot=make_scene(a.output,a.root_contact_policy);data=mujoco.MjData(model)
    joints={j.get('name'):j for j in robot.findall('joint') if j.get('type')!='fixed'};qids={n:int(model.joint(n).qposadr[0]) for n in joints}
    base={n:float(np.clip(0,float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))) for n,j in joints.items()}
    cases=[];started=time.monotonic()
    (a.output/'scanner_source.py').write_bytes(Path(__file__).read_bytes())
    for finger in a.fingers:
        names,paths=motion_paths(robot,finger,a.max_step_rad)
        for context in a.contexts:
            background=dict(base)
            if context=='others_closed':
                for f in ['index','middle','ring','pinky']:
                    if f!=finger:background[f+'_mcp_pitch']=1.4;background[f+'_pip']=1.57
            for path,angles in paths.items():
                name=finger+'__'+context+'__'+path;folder=a.output/name;folder.mkdir();poses=[];frames=[];stats={};baseline=set();mimic_error=0.
                for frame,row in enumerate(angles):
                    target=project_mimic(robot,dict(background,**dict(zip(names,map(float,row)))));mujoco.mj_resetData(model,data)
                    for n,value in target.items():data.qpos[qids[n]]=value
                    mujoco.mj_forward(model,data)
                    if any(w.number for w in data.warning):raise RuntimeError('MuJoCo warning')
                    pairs=collect(model,data)
                    if frame==0:baseline={k for k,v in pairs.items() if v['depth_mm']>.01}
                    for key,c in pairs.items():
                        if key not in stats:stats[key]=dict(left=c['left'],right=c['right'],baseline=key in baseline,max_depth_mm=0.,moving_finger_involved=any(x.startswith(finger+'_') for x in [c['left'],c['right']]))
                        s=stats[key];s['max_depth_mm']=max(s['max_depth_mm'],c['depth_mm'])
                        if c['depth_mm']>.01 and 'first_frame' not in s:s.update(first_frame=frame,first_angles=dict(zip(names,map(float,row))))
                    for j in joints.values():
                        m=j.find('mimic')
                        if m is not None:mimic_error=max(mimic_error,abs(data.qpos[qids[j.get('name')]]-float(m.get('multiplier','1'))*data.qpos[qids[m.get('joint')]]-float(m.get('offset','0'))))
                    poses.append(data.qpos.copy());frames.append(dict(frame=frame,angles=dict(zip(names,map(float,row))),pairs=pairs))
                np.savez_compressed(folder/'trajectory.npz',qpos=poses,angles=angles,joint_names=names)
                (folder/'contacts.json').write_text(json.dumps(frames,indent=2)+'\n')
                summary=list(stats.values());new=[s for s in summary if not s['baseline'] and 'first_frame' in s]
                entry=dict(name=name,finger=finger,context=context,path=path,frames=len(poses),trajectory=str(Path(name)/'trajectory.npz'),
                    baseline_pairs=sorted(baseline),new_collision_pairs=new,pair_summary=summary,max_mimic_error_rad=mimic_error,
                    moving_finger_peak_depth_mm=max([s['max_depth_mm'] for s in summary if s['moving_finger_involved']]+[0]),complete=True)
                cases.append(entry)
                result=dict(complete=False,cases=cases,engine='MuJoCo '+mujoco.__version__,links=22,pairs=model.npair,root_contact_policy=a.root_contact_policy,max_step_rad=a.max_step_rad,depth_threshold_mm=.01,
                    mode='kinematic sampled qpos + mj_forward, no gravity/object/motor dynamics',runtime_seconds=time.monotonic()-started)
                (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(name,'frames',len(poses),'new',len(new),flush=True)
    result['complete']=True;(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 各手指完整弯曲：MuJoCo整手自碰撞','',f'22link/{model.npair}个显式配对，根部策略{a.root_contact_policy}、原生SDF；深度阈值0.01mm，角步长≤'+str(a.max_step_rad)+'rad。有限角采样不是CCD或动力学认证。',
        '','others_closed把其他普通四指的MCP/PIP全闭，拇指保持零位；所有link仍参与。拇指分别扫描CMC yaw/roll/pitch和MCP，DIP按1.1619×MCP。','', '|手指/背景/路径|帧数|基线对|新增对|活动手指最大深度mm|','|---|---:|---:|---:|---:|']
    for c in cases:lines.append(f'|{c["name"]}|{c["frames"]}|{len(c["baseline_pairs"])}|{len(c["new_collision_pairs"])}|{c["moving_finger_peak_depth_mm"]:.5f}|')
    lines+=['','## 活动手指新增接触','', '|路径|连杆对|最大深度mm|首次独立关节角rad|','|---|---|---:|---|']
    for c in cases:
        for s in c['new_collision_pairs']:
            if s['moving_finger_involved']:lines.append(f'|{c["name"]}|{s["left"]}/{s["right"]}|{s["max_depth_mm"]:.5f}|{s["first_angles"]}|')
    lines+=['','基线装配接触未删除。SDF深度为数值估计，开放STL存在符号不确定性，不能宣称原始三角面精确穿透。']
    text='\n'.join(lines)+'\n';(a.output/'REPORT.md').write_text(text);(a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><pre>'+html.escape(text)+'</pre>')
    print('Complete:',a.output.resolve())


if __name__=='__main__':main()
