"""Offline per-finger forces and 3D contact normals from one isolated run."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def export(directory,start_seconds=2.0,end_seconds=3.1,cases=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.offline import get_plotlyjs
    directory=Path(directory);result=json.loads((directory/'result.json').read_text())
    z=np.load(directory/'trajectory.npz');points=np.load(directory/'contact_points.npz');rows=points['normal_contacts']
    folder=directory/'contacts';folder.mkdir(exist_ok=True);(directory/'plotly.min.js').write_text(get_plotlyjs())
    fingers=list(points['finger_names']);links=list(points['link_names']);dt=float(z['dt']);lines=['# 各指接触诊断','']
    displayed=[]
    if not 0 <= start_seconds < end_seconds:raise ValueError('Require 0 <= start < end')
    for i,case in enumerate(result['info']['cases']):
        if cases and case not in cases:continue
        displayed.append(case)
        fail=result['first_failure'][i];step=fail['physics_step'] if fail else round(3/dt)
        # Exact physics frames: never substitute an earlier contact snapshot.
        steps=np.rint(z['time']/dt).astype(int)
        steps=steps[(z['time']>=start_seconds-1e-9)&(z['time']<=end_seconds+1e-9)]
        if not len(steps):raise ValueError('Selected window has no trajectory frames')
        step=int(steps[np.argmin(abs(steps-step))])
        case_rows=rows[(rows[:,1]==i)&(rows[:,0]>=steps[0])&(rows[:,0]<=steps[-1])]
        groups={}
        for frame in np.unique(case_rows[:,0]).astype(int):groups[frame]=case_rows[case_rows[:,0]==frame]
        def contact_traces(frame):
            selected=groups.get(int(frame),np.empty((0,11)))
            hover=[f'{links[int(v[2])]}<br>normal force={v[9]:.6g} N<br>separation={v[10]*1000:.5g} mm' for v in selected]
            dots=go.Scatter3d(x=selected[:,3]*1000,y=selected[:,4]*1000,z=selected[:,5]*1000,
                mode='markers',text=hover,hoverinfo='text',name='接触记录（含零力）',
                marker=dict(size=4,color=selected[:,2],colorscale='Turbo',cmin=0,cmax=len(links)-1))
            line=[]
            for v in selected:line.extend([v[3:6]*1000,(v[3:6]+v[6:9]*.003)*1000,[None,None,None]])
            line=np.asarray(line,dtype=object).reshape(-1,3)
            normals=go.Scatter3d(x=line[:,0],y=line[:,1],z=line[:,2],mode='lines',name='SDK法向',showlegend=False)
            caption=f'{case}: t={frame*dt:.5f}s; records={len(selected)}; sum |normal force|={abs(selected[:,9]).sum():.6g} N; passed={result["info"]["passed"][i]}'
            return [dots,normals],caption
        fig=make_subplots(rows=2,cols=2,specs=[[{'colspan':2},None],[{}, {'type':'scene'}]],
            subplot_titles=['各指法向力标量和（N）','物体平移（mm）',f'{step*dt:.4f}s的接触点/传感器法向'])
        for f,name in enumerate(fingers):fig.add_trace(go.Scatter(x=z['time'],y=z['finger_normal_force'][:,i,f],name=name),row=1,col=1)
        pos=z['object_position'][:,i];ref=pos[np.argmin(abs(z['time']-3.))]
        fig.add_trace(go.Scatter(x=z['time'],y=np.linalg.norm(pos-ref,axis=1)*1000,name='相对开重力位置'),row=2,col=1)
        trace_ids=[len(fig.data),len(fig.data)+1]
        traces,caption=contact_traces(step)
        for trace in traces:fig.add_trace(trace,row=2,col=2)
        frames=[]
        for frame in steps:
            traces,title=contact_traces(frame)
            frames.append(go.Frame(name=str(frame),data=traces,traces=trace_ids,
                layout=go.Layout(title=dict(text=title))))
        fig.frames=frames
        fig.update_layout(sliders=[dict(active=int(np.argmin(abs(steps-step))),currentvalue=dict(prefix='接触帧时间 s：'),
            steps=[dict(label=f'{frame*dt:.5f}',method='animate',args=[[str(frame)],
                dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for frame in steps])],
            updatemenus=[dict(type='buttons',buttons=[
                dict(label='播放',method='animate',args=[None,dict(fromcurrent=True,frame=dict(duration=40,redraw=True),transition=dict(duration=0))]),
                dict(label='暂停',method='animate',args=[[None],dict(mode='immediate',frame=dict(duration=0,redraw=False))])])])
        for boundary in [2.,2.5,3.]:
            fig.add_shape(type='line',x0=boundary,x1=boundary,y0=0,y1=1,xref='x',yref='y domain',line=dict(dash='dot'))
        fig.update_layout(title=caption,height=900)
        fig.update_scenes(aspectmode='data');path=folder/f'{case.replace("+","plus").replace("-","minus")}.html'
        fig.write_html(path,include_plotlyjs='../plotly.min.js')
        lines.append(f'- [{case}]({path.relative_to(directory)}): {fail}')
    (directory/'CONTACT_REPORT.md').write_text('\n'.join(lines)+'\n')
    links=''.join(f'<li><a href="{case.replace("+","plus").replace("-","minus")}.html">{case}</a></li>' for case in displayed)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>逐指接触诊断</h1><p>力和位移为全程曲线；3D接触可用滑块逐物理帧查看，默认窗口2–3.1秒。空帧直接显示为空，不回退。颜色区分连杆，悬停查看连杆名、法向力与间距。base_link为掌根；零力接触记录不等于承载接触。坐标为旋转前原始物体轴方向（mm），并非随动物体局部坐标；法向符号沿用SDK。</p><ul>'+links+'</ul>')
    return directory/'CONTACT_REPORT.md'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--start-seconds',type=float,default=2.)
    p.add_argument('--end-seconds',type=float,default=3.1)
    p.add_argument('--cases',nargs='+')
    a=p.parse_args();print(export(a.run,a.start_seconds,a.end_seconds,a.cases))
