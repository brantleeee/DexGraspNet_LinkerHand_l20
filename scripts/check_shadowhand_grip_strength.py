"""Fresh-process drive stiffness/cap ablation with immutable 600g reference."""
import argparse,copy,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CASES=['gx+','gx-','gy+','gy-','gz+','gz-']
CONDITIONS=[('baseline',5.,.05,1.),('stiffer',20.,.1,1.),('stiffer_cap2',20.,.1,2.)]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=int,default=30);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    source=a.baseline.resolve();out=a.output.resolve()
    if out.exists():p.error('Output must be new')
    m=json.loads((source/'inputs/manifest.json').read_text());c=next(c for c in m['candidates'] if c['index']==a.candidate)
    if not np.isclose(m['object_mass_kg'],.6):p.error('Requires user-confirmed 600g object')
    out.mkdir(parents=True);(out/'inputs').mkdir();shutil.copy2(source/'inputs/meshes.npz',out/'inputs/meshes.npz')
    new=copy.deepcopy(m);new['candidates']=[c];new['mass_source']='user_confirmed_600g';new['robot']='ShadowHand'
    (out/'inputs/manifest.json').write_text(json.dumps(new,indent=2))
    plan=dict(source=str(source),candidate=a.candidate,mass_kg=.6,conditions=CONDITIONS,cases=CASES,gravity_seconds=10,dt=1/240,sdf_resolution=256,friction=3,actuator_cap_is_not_contact_force=True,base_input_sha256=sha(source/'inputs/manifest.json'),base_arrays_sha256=sha(source/'inputs/meshes.npz'),force_validator_sha256=sha(ROOT/'scripts/validate_shadowhand_force_case.py'),note='Same initial targets and physical mesh; independent OS process per direction. No grip target offset or changed success threshold. Solver effort is not isolated motor torque.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    if a.plan_only:return
    from replay_l20_isaacsim_html import export
    rows=[]
    def report():
        (out/'results.json').write_text(json.dumps(rows,indent=2));summary={}
        for name,*_ in CONDITIONS:
            rr=[r for r in rows if r['condition']==name];summary[name]=dict(attempted=len(rr),completed=sum(r['completed'] for r in rr),passed=sum(r['completed'] and r['result']['passed'] for r in rr),exceptions=sum(not r['completed'] for r in rr))
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        lines=['# 600g电钻抓取加力对照','',f'候选{a.candidate}；初态、质量、摩擦、时长、几何和严格判据固定；仅修改增益和力矩上限。','', '|组|Kp|Kd|上限Nm|完成|通过|异常|','|---|---:|---:|---:|---:|---:|---:|']
        for name,kp,kd,cap in CONDITIONS:
            r=summary[name];lines.append(f'|{name}|{kp}|{kd}|{cap}|{r["completed"]}/6|{r["passed"]}|{r["exceptions"]}|')
        base={r['case']:r for r in rows if r['condition']=='baseline' and r['completed']}
        for name,*_ in CONDITIONS[1:]:
            count=dict(both_pass=0,repaired=0,damaged=0,both_fail=0)
            for r in rows:
                if r['condition']!=name or not r['completed'] or r['case'] not in base:continue
                b=base[r['case']]['result']['passed'];v=r['result']['passed'];key='both_pass' if b and v else 'repaired' if v else 'damaged' if b else 'both_fail';count[key]+=1
            lines+=['',f'{name} vs本批fresh baseline：{count}；未完成方向不计入对照。']
        lines+=['','PD需求为后步状态估算；求解器轴向力矩含物理载荷，不等于电机输出；不得由大读数直接宣称触及驱动上限。完整轨迹/关节速度/诊断见各方向NPZ。']
        text='\n'.join(lines)+'\n';(out/'REPORT.md').write_text(text)
        cells=''.join(f'<tr><td>{r["condition"]}</td><td>{r["case"]}</td><td><a href="{r["condition"]}/{r["case"]}/replay.html">'+('异常' if not r['completed'] else '通过' if r['result']['passed'] else '失败')+'</a></td></tr>' for r in rows)
        import html
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>600g电钻 加力对照</h1><pre>'+html.escape(text)+'</pre><table border="1">'+cells+'</table>')
    report()
    for name,kp,kd,cap in CONDITIONS:
        for case in CASES:
            folder=out/name/case;folder.parent.mkdir(parents=True,exist_ok=True)
            cmd=['bash',str(ROOT/'scripts/run_shadowhand_force_case.sh'),'--inputs',str(out/'inputs'),'--output',str(folder),'--indices',str(a.candidate),'--cases',case,'--protocol','direct','--steps','2400','--kp',str(kp),'--kd',str(kd),'--effort-limit',str(cap)]
            print(f'{name} candidate {a.candidate} {case}: kp={kp} kd={kd} cap={cap}',flush=True)
            with folder.with_suffix('.log').open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            row=dict(condition=name,case=case,completed=False,returncode=r.returncode)
            if r.returncode==0 and (folder/'summary.json').exists():
                data=json.loads((folder/'summary.json').read_text());row.update(completed=True,result=data['results'][0]);export(folder,out/'inputs',folder/'replay.html')
            rows.append(row);report()
    print(out/'REPORT.md')

if __name__=='__main__':main()
