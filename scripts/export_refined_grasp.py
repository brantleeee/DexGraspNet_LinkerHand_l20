"""Export a new source NPY for an isolated-validated local reference; never overwrite."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaacsim_grasp.reference import digest
from isaacsim_grasp.reference_variants import variant_targets


def export(bundle,output):
    bundle=Path(bundle).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Output must be new')
    validation=json.loads((bundle/'validation.json').read_text());ref=json.loads((bundle/'reference.json').read_text())
    if validation['selected_run'] is None or validation['best_pass_count']<=validation['baseline_pass_count']:
        raise ValueError('No independently validated local improvement to export')
    index=ref['candidate'];source_root=Path(ref['source_results'])
    row=next(r for r in json.loads((source_root/ref['object']/'summary.json').read_text()) if r['candidate']==index)
    manifest=json.loads((Path(row['output'])/'summary.json').read_text())['manifest'];source=Path(manifest['source'])
    if digest(source)!=ref['source_sha256']:raise ValueError('Parent source changed')
    parent=json.loads((Path(validation['selected_run']).parent/'worker_plan.json').read_text())['reference']
    if variant_targets(parent,ref['derived_variant'])!=ref['targets']:raise ValueError('Selected target mismatch')
    array=np.load(source,allow_pickle=True);derived=copy.deepcopy(array);entry=derived[index]
    for name,value in ref['qpos'].items():entry['qpos'][name]=value
    entry['parent_static_metrics']={k:entry.pop(k) for k in ['energy','E_fc','E_dis','E_pen','E_spen','E_joints'] if k in entry}
    entry['refinement_provenance']=dict(parent_source=str(source),parent_source_sha256=digest(source),
        selected_validation=str(bundle/'validation.json'),variant=ref['derived_variant'],
        static_energy_values_removed=True,physical_validation_scope=validation['verified_scope'])
    output.mkdir(parents=True);path=output/source.name;np.save(path,derived,allow_pickle=True)
    record=dict(result_file=str(path),object=ref['object'],candidate=index,scale=ref['scale'],
        source_sha256=digest(path),parent_source=str(source),parent_source_sha256=digest(source),
        warning='Candidate qpos changed; old static energies moved to parent_static_metrics, not valid for derived pose. Requires fresh validation/new policy; no old checkpoint reuse.')
    (output/'EXPORT.json').write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.bundle,a.output),indent=2))
