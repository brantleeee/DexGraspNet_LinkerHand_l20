"""Render saved Isaac Sim states with MuJoCo graphics only; never mj_step.
Meshes are the recorded scene's SDF input triangles, not SDF isosurfaces.
"""
import argparse,json,os,sys,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def vec(a):return ' '.join(map(str,a))
def quat(rpy):
    q=Rotation.from_euler('xyz',rpy).as_quat();return vec(q[[3,0,1,2]])


def model_xml(manifest,reference,meshes,folder):
    import trimesh
    root=ET.Element('mujoco',model='recorded_states_visualization_only')
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='480')
    ET.SubElement(visual,'headlight',ambient='.5 .5 .5',diffuse='.8 .8 .8')
    asset=ET.SubElement(root,'asset')
    folder.mkdir(exist_ok=True)
    for item in manifest['meshes']:
        key=item['key'];path=folder/f'{key}.stl'
        trimesh.Trimesh(vertices=meshes[key+'_vertices'],faces=meshes[key+'_faces'],process=False).export(path)
        ET.SubElement(asset,'mesh',name=key,file=str(path.resolve()))
    world=ET.SubElement(root,'worldbody');ET.SubElement(world,'light',pos='0 -1 1',dir='0 1 -1',directional='true')
    wrist=reference['qpos'];base=ET.SubElement(world,'body',name='base_link',pos=vec([wrist['WRJT'+k] for k in 'xyz']),quat=quat([wrist['WRJR'+k] for k in 'xyz']))
    bodies={'base_link':base};pending=list(manifest['joints'])
    while pending:
        ready=[j for j in pending if j['parent'] in bodies]
        if not ready:raise ValueError('Invalid tree')
        for j in ready:
            body=ET.SubElement(bodies[j['parent']],'body',name=j['child'],pos=vec(j['xyz']),quat=quat(j['rpy']))
            bodies[j['child']]=body
            if j['type']!='fixed':
                ET.SubElement(body,'inertial',pos='0 0 0',mass='1',diaginertia='1 1 1')
                ET.SubElement(body,'joint',name=j['name'],type='hinge',axis=vec(j['axis']),limited='false')
            pending.remove(j)
    for link in manifest['links']:
        for c in link['colliders']:
            ET.SubElement(bodies[link['name']],'geom',type='mesh',mesh=c['key'],pos=vec(c['xyz']),quat=quat(c['rpy']),rgba='.25 .58 .78 1',contype='0',conaffinity='0')
    obj=ET.SubElement(world,'body',name='object');ET.SubElement(obj,'freejoint',name='object_pose')
    ET.SubElement(obj,'inertial',pos='0 0 0',mass='1',diaginertia='1 1 1')
    ET.SubElement(obj,'geom',type='mesh',mesh='object',rgba='.95 .57 .12 1',contype='0',conaffinity='0')
    return ET.tostring(root,encoding='unicode')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--left-label',default='Zero residual');p.add_argument('--right-label',default='RRL')
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--case',default='gy+')
    p.add_argument('--fps',type=int,default=20);p.add_argument('--end-seconds',type=float,default=13.)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco,imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    if a.output.exists():p.error('Output must be new')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.inputs/'manifest.json').read_text())
    reference=json.loads((a.left/'reference_snapshot.json').read_text())
    right_ref=json.loads((a.right/'reference_snapshot.json').read_text())
    if reference['qpos']!=right_ref['qpos']:raise ValueError('Wrist/reference mismatch')
    with np.load(a.inputs/'meshes.npz') as m:meshdata={k:m[k] for k in m.files}
    xml=model_xml(manifest,reference,meshdata,a.output.parent/'render_meshes')
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    records=[]
    for run in [a.left,a.right]:
        with np.load(run/'trajectory.npz') as z:trace={k:z[k] for k in z.files}
        result=json.loads((run/'result.json').read_text());idx=result['info']['cases'].index(a.case)
        name_index={str(n.decode() if isinstance(n,bytes) else n):i for i,n in enumerate(trace['dof_names'])}
        addresses=[(model.joint(n).qposadr[0],j) for n,j in name_index.items()]
        records.append((trace,result,idx,addresses))
    if not np.array_equal(records[0][0]['time'],records[1][0]['time']):raise ValueError('Time mismatch')
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=ImageFont.truetype(font_path,19);small=ImageFont.truetype(font_path,15)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[0,0,.045];camera.distance=.48;camera.azimuth=145;camera.elevation=-20
    object_adr=model.joint('object_pose').qposadr[0]
    labels=[a.left_label,a.right_label];end=min(a.end_seconds,float(records[0][0]['time'][-1]))
    frames=np.arange(0,end+1e-9,1/a.fps)
    initial_saved=False;gravity_saved=False
    with mujoco.Renderer(model,480,640) as renderer,imageio.get_writer(a.output,fps=a.fps,codec='libx264',quality=8) as writer:
        for frame_no,sec in enumerate(frames):
            canvas=Image.new('RGB',(1280,640),'#f3f6fa');draw=ImageDraw.Draw(canvas)
            for side,(t,r,idx,addresses) in enumerate(records):
                k=int(np.argmin(abs(t['time']-sec)))
                for adr,j in addresses:data.qpos[adr]=t['joint_position'][k,idx,j]
                data.qpos[object_adr:object_adr+3]=t['object_position'][k,idx]
                data.qpos[object_adr+3:object_adr+7]=t['object_quaternion'][k,idx]
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
                canvas.paste(Image.fromarray(renderer.render()),(side*640,80))
                first=r['first_failure'][idx];failed=first is not None and sec>=first['time_seconds']
                color='#b72e35' if failed else '#164c39'
                draw.text((side*640+18,12),labels[side],fill='#172334',font=font)
                draw.text((side*640+18,43),f'{a.case} | t={sec:.2f}s | '+('零重力闭合' if sec<2 else '零重力预加载' if sec<3 else '重力保持'),fill='#354760',font=small)
                draw.text((side*640+18,568),'已触发严格失败' if failed else '截至当前未触发失败',fill=color,font=font)
                draw.text((side*640+18,601),f"完整10秒验收：{'通过' if r['info']['passed'][idx] else '失败'}",fill='#354760',font=small)
            draw.text((15,621),'Isaac Sim 保存轨迹 · 离线三角网格回放（MuJoCo仅用于渲染）· 1倍速 · 蓝：手 / 橙：杯子',fill='#526275',font=small)
            writer.append_data(np.asarray(canvas))
            if not initial_saved:
                canvas.save(a.output.with_suffix('.png'));initial_saved=True
            if not gravity_saved and sec>=3:
                canvas.save(a.output.with_name(a.output.stem+'_gravity.png'));gravity_saved=True
            if frame_no%100==0:print(f'rendered {frame_no}/{len(frames)}',flush=True)
    provenance=dict(left=str(a.left.resolve()),right=str(a.right.resolve()),case=a.case,fps=a.fps,end_seconds=end,physics_rerun=False,renderer='MuJoCo graphics; mj_forward only',geometry='SDF input meshes, not isosurfaces',camera=dict(azimuth=145,elevation=-20,distance=.48),source_hashes={str(run/'trajectory.npz'):hashlib.sha256((run/'trajectory.npz').read_bytes()).hexdigest() for run in [a.left,a.right]})
    a.output.with_suffix('.json').write_text(json.dumps(provenance,indent=2));print(a.output)

if __name__=='__main__':main()
