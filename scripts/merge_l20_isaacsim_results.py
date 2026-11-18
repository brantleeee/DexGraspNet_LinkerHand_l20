"""Combine object batches and cached replays into one audited local index."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from l20_object_catalog import OBJECTS,ROOT
from replay_l20_isaacsim_html import export


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();output=a.output.resolve()
    if output.exists():p.error('Output must be new')
    batches={}
    for source in a.sources:
        source=source.resolve()
        if not source.is_dir():p.error(f'Source missing: {source}')
        for obj in OBJECTS:
            folder=source/obj;manifest=folder/'inputs/manifest.json'
            if not manifest.is_file():continue
            if obj in batches:p.error(f'Duplicate object across sources: {obj}')
            if not (folder/'summary.json').is_file():p.error(f'Batch summary missing: {folder}')
            batches[obj]=(source,folder)
    if not batches:p.error('No supported batches found')
    output.mkdir(parents=True);replays=output/'visualizations';replays.mkdir()
    provenance={}
    for obj,(source,folder) in batches.items():
        (output/obj).symlink_to(folder,target_is_directory=True)
        provenance[obj]=dict(source=str(source),batch=str(folder),
            manifest_sha256=hashlib.sha256((folder/'inputs/manifest.json').read_bytes()).hexdigest(),
            batch_summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest())
        for row in json.loads((folder/'summary.json').read_text()):
            if not row['completed']:continue
            name=f"{obj}_{row['candidate']:03d}.html"
            cached=source/'visualizations'/name;destination=replays/name
            result=Path(row['output'])
            newest=max((result/'summary.json').stat().st_mtime,
                       (folder/'inputs/meshes.npz').stat().st_mtime,
                       Path(export.__code__.co_filename).stat().st_mtime)
            if cached.is_file() and cached.stat().st_mtime>=newest:
                destination.symlink_to(cached)
            else:
                export(result,folder/'inputs',destination)
    (output/'sources.json').write_text(json.dumps(dict(objects=provenance,
        note='Local links to original batches and cached replays; preserve source directories.'),indent=2))
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_l20_isaacsim.py'),
                    '--results',str(output),'--skip-replays'],cwd=ROOT,check=True)
    print(f'Combined index: {output}/index.html')


if __name__=='__main__':main()
