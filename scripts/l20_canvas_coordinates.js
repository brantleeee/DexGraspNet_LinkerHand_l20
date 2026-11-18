/* Software orthographic renderer: Canvas 2D only, no WebGL or network queries. */
const D=JSON.parse(document.getElementById('scene-data').textContent);
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
let yaw=.65,pitch=-.45,zoom=1,mode='3D',drag=null,pending=false;
const select=document.getElementById('point'),labels=document.getElementById('labels'),mesh=document.getElementById('mesh');
const vertices=D.vertices_mm,points=D.points;
const all=vertices.concat(points.map(p=>p.xyz_mm)).concat([[0,0,0]]);
const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
for(const p of all)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k]);}
const center=lo.map((v,k)=>(v+hi[k])/2);
let radius=1;for(const p of all)radius=Math.max(radius,Math.hypot(...p.map((x,k)=>x-center[k])));
const axisLength=Math.max(3,Math.min(10,radius*.35));
function camera(p){
 const [x,y,z]=p.map((v,k)=>v-center[k]);
 if(mode==='XY')return [x,y,z];if(mode==='XZ')return [x,z,-y];if(mode==='YZ')return [y,z,x];
 const x1=Math.cos(yaw)*x+Math.sin(yaw)*z,z1=-Math.sin(yaw)*x+Math.cos(yaw)*z;
 return [x1,Math.cos(pitch)*y-Math.sin(pitch)*z1,Math.sin(pitch)*y+Math.cos(pitch)*z1];
}
function screen(p){const a=camera(p),s=Math.min(canvas.width,canvas.height)*.40*zoom/radius;return [canvas.width/2+a[0]*s,canvas.height/2-a[1]*s,a[2]];}
function line(a,b,color,width=2){ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function arrow(a,b,color,label){
 line(a,b,color,2);const dx=b[0]-a[0],dy=b[1]-a[1],n=Math.hypot(dx,dy);
 if(n>1){const ux=dx/n,uy=dy/n;line(b,[b[0]-6*ux+3*uy,b[1]-6*uy-3*ux],color);line(b,[b[0]-6*ux-3*uy,b[1]-6*uy+3*ux],color);}
 else{ctx.beginPath();ctx.arc(b[0],b[1],3,0,Math.PI*2);ctx.strokeStyle=color;ctx.stroke();}
 ctx.fillStyle=color;ctx.font='11px sans-serif';ctx.fillText(label,b[0]+3,b[1]+3);
}
function draw(){
 if(!ctx){document.getElementById('state').textContent='Canvas 2D不可用，请查看下方静态SVG投影和坐标表';return;}
 const bounds=canvas.getBoundingClientRect(),w=Math.max(400,Math.round(bounds.width)),h=650;
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 ctx.clearRect(0,0,w,h);ctx.fillStyle='#f6f8fb';ctx.fillRect(0,0,w,h);
 if(mesh.checked){
  const pv=vertices.map(screen);
  const order=D.faces.map((f,i)=>({f,i,z:(pv[f[0]][2]+pv[f[1]][2]+pv[f[2]][2])/3})).sort((a,b)=>a.z-b.z);
  for(const item of order){const [a,b,c]=item.f.map(i=>pv[i]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.closePath();
   const g=Math.round(170+45*(item.z+radius)/(2*radius));ctx.fillStyle=D.face_colors?D.face_colors[item.i]:`rgb(${g},${g+5},${Math.min(255,g+12)})`;ctx.fill();}
 }
 // Reveal the exact supporting triangle for a selected point, above the mesh.
 // Overlay is intentional: this is inspection, not a visibility/collision test.
 const chosen=points.find(p=>String(p.global_id)===select.value);
 if(chosen&&chosen.support_triangle_mm){
  const [a,b,c]=chosen.support_triangle_mm.map(screen);
  ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.closePath();
  ctx.fillStyle='rgba(255,145,0,0.65)';ctx.fill();ctx.strokeStyle='#d07000';ctx.lineWidth=3;ctx.stroke();
  const q=screen(chosen.anchor_mm);line(screen(chosen.xyz_mm),q,'#9c24b5',2);
  ctx.fillStyle='#9c24b5';ctx.fillText('Q / 支撑面',q[0]+5,q[1]+12);
 }
 const origin=screen([0,0,0]);ctx.font='13px sans-serif';ctx.fillStyle='black';ctx.fillText('O',origin[0]+3,origin[1]-5);
 ['X','Y','Z'].forEach((name,k)=>{const e=[0,0,0];e[k]=axisLength;const end=screen(e),color=['#b71919','#16803a','#225cce'][k];line(origin,end,color,3);ctx.fillStyle=color;ctx.fillText('+'+name,end[0]+3,end[1]);});
 for(const p of points){
  if(select.value!=='all'&&select.value!==String(p.global_id))continue;
  if(select.value==='all'&&document.getElementById('near-only')?.checked&&!p.near_object&&!p.selected)continue;
  if(p.object_anchor_mm&&document.getElementById('object-anchors')?.checked){
   const anchor=screen(p.object_anchor_mm);line(screen(p.xyz_mm),anchor,'#9c24b5',1);
   ctx.beginPath();ctx.arc(anchor[0],anchor[1],6,0,Math.PI*2);ctx.strokeStyle='#0a9040';ctx.lineWidth=2;ctx.stroke();
  }
  if(document.getElementById('frames').checked){
   const local=document.getElementById('frame-mode').value==='contact';
   const directions=local?p.frame.axes:[[1,0,0],[0,1,0],[0,0,1]];
   const input=Number(document.getElementById('frame-length').value),length=Number.isFinite(input)?Math.max(.1,Math.min(20,input)):2;
   directions.forEach((axis,k)=>{const end=screen(p.xyz_mm.map((v,i)=>v+length*axis[i]));arrow(screen(p.xyz_mm),end,['#d11d1d','#16803a','#225cce'][k],`P${p.global_id} ${local?['Xc/T1','Yc/T2','Zc/N'][k]:['X','Y','Z'][k]}`);});
   if(p.review_normal){const end=screen(p.xyz_mm.map((v,i)=>v+length*1.3*p.review_normal[i]));arrow(screen(p.xyz_mm),end,'#111111',`P${p.global_id} 候选N（预览）`);}
  }
  const q=screen(p.xyz_mm);ctx.beginPath();ctx.arc(q[0],q[1],p.selected?6:4.5,0,Math.PI*2);ctx.fillStyle=p.marker_color||'#d52331';ctx.fill();
  if(labels.checked){ctx.fillStyle='#81111a';ctx.font='12px sans-serif';ctx.fillText(`P${p.global_id} (${p.xyz_mm.map(x=>x.toFixed(3)).join(', ')}) mm`+(p.gap_mm===undefined?'':` d=${p.gap_mm.toFixed(3)}mm`),q[0]+7,q[1]-7);}
 }
 document.getElementById('state').textContent=`${D.link} 局部系 | ${mode}正交投影 | ${points.length}点 | 标签mm，表格m/mm | 软件绘制，无WebGL`+(chosen&&chosen.surface_warning?' | '+chosen.surface_warning:'');
}
function request(){if(!pending){pending=true;requestAnimationFrame(()=>{pending=false;draw();});}}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!drag)return;mode='3D';yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];request();});
canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));request();},{passive:false});
for(const id of ['point','labels','mesh','frames','frame-mode','frame-length'])document.getElementById(id).addEventListener('change',request);
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.view;request();}));
document.getElementById('reset').addEventListener('click',()=>{yaw=.65;pitch=-.45;zoom=1;mode='3D';request();});
window.addEventListener('resize',request);request();
for(const id of ['near-only','object-anchors'])document.getElementById(id)?.addEventListener('change',request);
