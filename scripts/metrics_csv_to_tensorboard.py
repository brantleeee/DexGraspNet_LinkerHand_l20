"""Export native Isaac Sim residual metrics.csv to TensorBoard without changing training.

Use --follow in a separate terminal to append new CSV metrics while training runs.
"""
import argparse
import csv
import fcntl
import io
import json
import math
from pathlib import Path
import time

from torch.utils.tensorboard import SummaryWriter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--output',type=Path,help='Default: <CSV directory>/tensorboard_csv')
    p.add_argument('--follow',action='store_true')
    p.add_argument('--interval',type=float,default=5.)
    a=p.parse_args();source=a.csv.resolve();output=(a.output or source.parent/'tensorboard_csv').resolve()
    if not source.is_file():p.error('CSV does not exist')
    if not math.isfinite(a.interval) or a.interval<=0:p.error('Interval must be positive and finite')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.export.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another exporter is already writing this directory')
        state_path=output/'csv_export_state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(source=str(source),last_steps={},rows_exported=0)
        if state['source']!=str(source):p.error('Output belongs to a different source CSV')
        writer=SummaryWriter(str(output));total=0
        try:
            while True:
                # Ignore a currently incomplete final line; it is read on the next pass.
                text=source.read_text();end=text.rfind('\n');complete=text[:end+1] if end>=0 else ''
                reader=csv.DictReader(io.StringIO(complete))
                if reader.fieldnames!=['iteration','tag','value']:p.error('Unexpected CSV header')
                added=0
                for row in reader:
                    step=int(row['iteration']);tag=row['tag'];value=float(row['value'])
                    if not math.isfinite(value):raise ValueError(f'Nonfinite metric: {tag}, step {step}')
                    if step<=state['last_steps'].get(tag,-1):continue
                    writer.add_scalar(tag,value,step);state['last_steps'][tag]=step;added+=1
                writer.flush();total+=added;state['rows_exported']+=added
                temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(state_path)
                if added:print(f'Exported {added} scalars; latest iteration={max(state["last_steps"].values())}; output={output}',flush=True)
                if not a.follow:break
                time.sleep(a.interval)
        except KeyboardInterrupt:
            print('Exporter stopped; training is unaffected.',flush=True)
        finally:writer.close()
        print(f'This invocation exported {total} scalars.',flush=True)


if __name__=='__main__':main()
