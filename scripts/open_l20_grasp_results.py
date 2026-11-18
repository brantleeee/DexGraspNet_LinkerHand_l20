"""One-shot job status; open HTML only after successful export. No polling."""
import argparse,json,re,subprocess
from pathlib import Path


def inspect(run):
    run=Path(run).resolve();job=run.with_suffix('.job.json')
    if not job.exists():raise FileNotFoundError('任务状态文件不存在：'+str(job))
    state=json.loads(job.read_text());phase=state.get('phase','unknown')
    target=Path(state.get('html',run/'results/scales/scale_1/top_16_canvas/index.html'))
    log=Path(state.get('log',run.with_suffix('.log')));progress=None
    if log.exists():
        with log.open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-16384));tail=f.read().decode('utf-8',errors='replace')
        matches=re.findall(r'(\d+)/([0-9]+)\s*\[',tail)
        if matches:progress=tuple(map(int,matches[-1]))
    return dict(phase=phase,target=target,log=log,progress=progress,error=state.get('error'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--no-open',action='store_true');a=p.parse_args()
    try:state=inspect(a.run)
    except (OSError,ValueError) as e:p.exit(1,str(e)+'\n')
    phase=state['phase'];print('任务阶段：',phase)
    if state['progress']:print('最近日志进度：%d/%d'%state['progress'])
    if phase=='failed':p.exit(1,'任务失败：'+str(state['error'])+'\n日志：'+str(state['log'])+'\n')
    if phase!='complete':
        print('结果尚未导出，暂不打开浏览器。生成完成后重新执行本命令即可。\n日志：'+str(state['log']));return
    if not state['target'].is_file():p.exit(1,'状态为complete，但HTML缺失，请检查导出产物：'+str(state['target'])+'\n')
    print('结果：',state['target'])
    if not a.no_open:subprocess.run(['xdg-open',str(state['target'])],check=True)


if __name__=='__main__':main()
