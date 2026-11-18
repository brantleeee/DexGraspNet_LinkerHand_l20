"""Re-export a snapshot of completed native trials from unchanged NPZs; no physics."""
import argparse
import json
from pathlib import Path
from replay_l20_isaacsim_html import export


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True,type=Path);a=p.parse_args()
    root=a.batch.resolve();inputs=root/'inputs'
    if not (inputs/'manifest.json').is_file():p.error('Missing prepared inputs')
    # Snapshot only: do not wait or monitor simulations still running.
    folders=sorted({q.parent for q in root.glob('candidate_*/*/summary.json')})
    done=0
    for folder in folders:
        status=folder/'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status')!='complete':continue
        tmp=folder/'replay.repaired.tmp.html'
        export(folder,inputs,tmp)
        tmp.replace(folder/'replay.html');done+=1
    print(f'Re-exported {done} completed replay(s); original JSON/NPZ and physics results unchanged.')

if __name__=='__main__':main()
