"""Export offline, rotatable collision-mesh playback of saved Isaac Sim trials.

Playback uses the saved joint and object states. It does not simulate physics.
The displayed meshes are the SDF input triangles, not reconstructed SDF isosurfaces.
"""
import argparse
from html import escape
import base64
import json
from pathlib import Path
import zlib

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'assets/web_replay/three-r169'


def encoded(array,dtype):
    return base64.b64encode(np.asarray(array,dtype=dtype).tobytes()).decode('ascii')


def export(result,inputs,output):
    summary=json.loads((result/'summary.json').read_text())
    manifest=summary['manifest']
    index=summary['results'][0]['candidate']
    candidate=next(c for c in manifest['candidates'] if c['index']==index)
    meshdata=np.load(inputs/'meshes.npz')
    meshes={p['key']:dict(vertices=encoded(meshdata[p['key']+'_vertices'],'<f4'),
                           faces=encoded(meshdata[p['key']+'_faces'],'<u4')) for p in manifest['meshes']}
    trials={}
    for trial in summary['results']:
        name=trial['case'];data=np.load(result/f'trajectory_{name}.npz')
        names=data['dof_names'].tolist()
        initialq=data['initial_joint_positions'] if 'initial_joint_positions' in data else [candidate['targets'][n] for n in names]
        phase=np.r_[0,data['phase']] if 'phase' in data else np.full(len(data['time'])+1,2)
        trials[name]=dict(summary=trial,names=names,time=np.r_[0,data['time']].tolist(),phase=phase.tolist(),
                          reference_position=data['gravity_reference_position'].tolist() if 'gravity_reference_position' in data else [0,0,0],
                          q=encoded(np.vstack([initialq,data['joint_positions']]),'<f4'),
                          positions=encoded(np.vstack([np.zeros(3),data['object_position']]),'<f4'),
                          orientations=encoded(np.vstack([[1,0,0,0],data['object_quaternion']]),'<f4'),
                          force=np.r_[0,data['contact_force']].tolist(),depth=np.r_[0,data['penetration']].tolist())
    payload=dict(robot=manifest.get('robot','L20'),object=manifest['object'],scale=manifest['scale'],index=index,mass=manifest['object_mass_kg'],
                 mesh_kind=manifest['object_mesh_kind'],links=manifest['links'],joints=manifest['joints'],
                 wrist=candidate['qpos'],meshes=meshes,trials=trials)
    packed=base64.b64encode(zlib.compress(json.dumps(payload,separators=(',',':')).encode(),9)).decode()
    html=render_html(packed,str(payload['robot']))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(html)
    print(output)


def render_html(packed,robot_label):
    """Substitute explicit UI markers; never rename text inside encoded assets."""
    three='data:text/javascript;base64,'+base64.b64encode((VENDOR/'three.module.min.js').read_bytes()).decode()
    controls='data:text/javascript;base64,'+base64.b64encode((VENDOR/'OrbitControls.js').read_bytes()).decode()
    page=TEMPLATE.replace('__ROBOT_LABEL__',escape(robot_label)).replace('__IMPORT_MAP__',json.dumps({'imports':{'three':three}})).replace('__CONTROLS__',controls).replace('__PAYLOAD__',packed)
    return page+'\n<!-- Three.js license:\n'+(VENDOR/'LICENSE').read_text()+'\n-->\n'


TEMPLATE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__ROBOT_LABEL__ Isaac Sim 抓取回放</title>
<style>body{margin:0;height:100vh;display:flex;flex-direction:column;font:15px system-ui;background:#f2f5f8;color:#172334}header{padding:14px 20px;background:white;border-bottom:1px solid #d9e1ea}h1{font-size:20px;margin:0 0 5px}p{margin:5px 0;color:#536074}.toolbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:12px}button,select{font:inherit;padding:5px 10px}label{white-space:nowrap}#seek{width:min(400px,50vw)}#view{flex:1;min-height:320px}header,#metrics{flex:none}#metrics{padding:9px 20px;background:white;font-variant-numeric:tabular-nums}#error{color:#b02727;white-space:pre-wrap}small{font-size:13px}</style>
<header><h1 id="title">正在加载碰撞网格…</h1><p>Isaac Sim 保存轨迹回放 · 蓝色：手部碰撞网格 · 橙色：物体碰撞网格</p>
<small>显示 SDF 输入三角网格；本页不重新仿真，也不显示 SDF 等值面。左键旋转，滚轮缩放，右键平移。</small>
<div class="toolbar"><select id="direction"></select><button id="play">播放</button><button id="reset">恢复视角</button>
<label>速度 <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select></label>
<label><input id="hand" type="checkbox" checked>手</label><label><input id="object" type="checkbox" checked>物体</label>
<label><input id="wire" type="checkbox">网格线</label><label><input id="loop" type="checkbox" checked>循环</label>
<input id="seek" type="range" min="0" max="1000" value="0" step="1"></div><p id="status"></p><div id="error"></div></header>
<div id="view"></div><div id="metrics"></div>
<script type="importmap">__IMPORT_MAP__</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from '__CONTROLS__';
try {
const bytes=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const stream=new Blob([bytes('__PAYLOAD__')]).stream().pipeThrough(new DecompressionStream('deflate'));
const data=JSON.parse(await new Response(stream).text());
const f32=s=>new Float32Array(bytes(s).buffer),u32=s=>new Uint32Array(bytes(s).buffer);
const view=document.getElementById('view'),scene=new THREE.Scene();scene.background=new THREE.Color('#f2f5f8');
const camera=new THREE.PerspectiveCamera(40,1,.001,500);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));view.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x71829a,2.4));
const light=new THREE.DirectionalLight(0xffffff,2.1);light.position.set(1,2,3);scene.add(light);
const fill=new THREE.DirectionalLight(0xffffff,.9);fill.position.set(-2,-1,1);scene.add(fill);
camera.up.set(0,0,1);
const handMaterial=new THREE.MeshStandardMaterial({color:0x64a9d1,roughness:.6,metalness:.08,side:THREE.DoubleSide});
const objectMaterial=new THREE.MeshStandardMaterial({color:0xe9a53d,roughness:.7,side:THREE.DoubleSide});
const geometry={};for(const [key,mesh] of Object.entries(data.meshes)){
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(f32(mesh.vertices),3));g.setIndex(new THREE.BufferAttribute(u32(mesh.faces),1));g.computeVertexNormals();geometry[key]=g;
}
const rotation=r=>new THREE.Quaternion().setFromEuler(new THREE.Euler(...r,'ZYX'));
const links={};for(const link of data.links){
 const group=new THREE.Group();links[link.name]=group;
 for(const collider of link.colliders){const mesh=new THREE.Mesh(geometry[collider.key],handMaterial);mesh.position.fromArray(collider.xyz);mesh.quaternion.copy(rotation(collider.rpy));group.add(mesh);}
}
const wrist=data.wrist,base=links.base_link;base.position.set(...['x','y','z'].map(k=>wrist['WRJT'+k]));base.quaternion.copy(rotation(['x','y','z'].map(k=>wrist['WRJR'+k])));scene.add(base);
const joints=data.joints.map(j=>({...j,axis:new THREE.Vector3(...j.axis).normalize(),origin:rotation(j.rpy)}));
for(const j of joints){links[j.parent].add(links[j.child]);links[j.child].position.fromArray(j.xyz);}
const object=new THREE.Mesh(geometry.object,objectMaterial);scene.add(object);
const origin=new THREE.AxesHelper(.025);scene.add(origin);
const away=new THREE.Vector3(-base.position.x,-base.position.y,0);if(away.length()<1e-5)away.set(1,1,0);away.normalize().multiplyScalar(.4);away.z=.22;
function resetCamera(){base.updateMatrixWorld(true);object.updateMatrixWorld(true);const box=new THREE.Box3().setFromObject(base).expandByObject(object),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()).length();controls.target.copy(center);camera.position.copy(center).add(away.clone().normalize().multiplyScalar(size*1.7));controls.update();}
const resize=()=>{camera.aspect=view.clientWidth/view.clientHeight;camera.updateProjectionMatrix();renderer.setSize(view.clientWidth,view.clientHeight);};new ResizeObserver(resize).observe(view);resize();
let trial,frame=0,playing=false,clock=0,last=performance.now();
const direction=document.getElementById('direction'),seek=document.getElementById('seek'),button=document.getElementById('play');
for(const [name,t] of Object.entries(data.trials)){direction.add(new Option(name+' · '+(t.summary.passed?'通过':'未通过'),name));}
function select(){trial=data.trials[direction.value];if(!trial.decoded){trial.q=f32(trial.q);trial.positions=f32(trial.positions);trial.orientations=f32(trial.orientations);trial.nameToIndex=Object.fromEntries(trial.names.map((n,i)=>[n,i]));trial.decoded=true;}
 frame=0;clock=0;seek.max=trial.time.length-1;seek.value=0;const s=trial.summary;
 document.getElementById('status').textContent=`${direction.value}：重力阶段最大位移 ${(s.max_slip_m*1000).toFixed(2)} mm · 最大转角 ${s.max_angle_deg.toFixed(2)}° · 全程最大接触穿透 ${(s.max_penetration_m*1000).toFixed(2)} mm · ${s.passed?'通过':'未通过'}`;apply();}
function apply(){
 const count=trial.names.length;
 for(const j of joints){const angle=j.type==='fixed'?0:trial.q[frame*count+trial.nameToIndex[j.name]];links[j.child].quaternion.copy(j.origin).multiply(new THREE.Quaternion().setFromAxisAngle(j.axis,angle));}
 object.position.fromArray(trial.positions,frame*3);const k=frame*4;object.quaternion.set(trial.orientations[k+1],trial.orientations[k+2],trial.orientations[k+3],trial.orientations[k]);
 const phase=trial.phase[frame],phaseLabel=['零重力闭合','零重力保持','重力验证'][phase];
 const reference=phase===2?new THREE.Vector3(...trial.reference_position):new THREE.Vector3();const distanceLabel=phase===2?'重力阶段位移':'闭合阶段推移';
 seek.value=frame;document.getElementById('metrics').textContent=`${phaseLabel} · ${trial.time[frame].toFixed(3)} / ${trial.time.at(-1).toFixed(3)} 秒 · ${distanceLabel} ${(object.position.distanceTo(reference)*1000).toFixed(2)} mm · 接触力 ${frame?trial.force[frame].toFixed(3)+' N':'初始帧未采样'} · 接触穿透 ${frame?(trial.depth[frame]*1000).toFixed(3)+' mm':'初始帧未采样'}`;
}
direction.onchange=select;seek.oninput=()=>{frame=Number(seek.value);clock=trial.time[frame];apply();};button.onclick=()=>{playing=!playing;button.textContent=playing?'暂停':'播放';};
document.getElementById('reset').onclick=resetCamera;document.getElementById('hand').onchange=e=>base.visible=e.target.checked;document.getElementById('object').onchange=e=>object.visible=e.target.checked;
document.getElementById('wire').onchange=e=>{handMaterial.wireframe=objectMaterial.wireframe=e.target.checked;};
document.getElementById('title').textContent=`${data.robot||'L20'} · ${data.object} · 候选 ${data.index} · scale ${data.scale} · ${(data.mass*1000).toFixed(2)} g`;
if(data.mesh_kind==='recovered_coacd_mesh_not_original'){document.getElementById('title').insertAdjacentHTML('afterend','<small>物体源网格为恢复的旧 CoACD 网格；SDF 不会恢复原物体细节。</small>');}
const query=new URLSearchParams(location.search);if(data.trials[query.get('case')])direction.value=query.get('case');select();if(query.has('frame')){frame=Math.max(0,Math.min(trial.time.length-1,Number(query.get('frame'))||0));clock=trial.time[frame];apply();}resetCamera();document.body.dataset.ready='true';window.l20Replay={data,select,apply,get trial(){return trial;},setFrame(i){frame=i;clock=trial.time[i];apply();}};
function animate(now){requestAnimationFrame(animate);if(playing){clock+=(now-last)/1000*Number(document.getElementById('speed').value);const end=trial.time.at(-1);if(clock>end){if(document.getElementById('loop').checked)clock%=end;else{clock=end;playing=false;button.textContent='播放';}}frame=Math.min(trial.time.length-1,Math.round(clock/end*(trial.time.length-1)));apply();}last=now;controls.update();renderer.render(scene,camera);}requestAnimationFrame(animate);
} catch(error){document.getElementById('error').textContent=error.stack||String(error);document.body.dataset.ready='error';console.error(error);}
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True,help='Single-candidate result directory')
    p.add_argument('--inputs',type=Path,required=True,help='Prepared input directory containing meshes.npz')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.results,a.inputs,a.output)


if __name__=='__main__':main()
