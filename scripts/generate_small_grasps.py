"""Generate smaller-object grasps and export one pose file per exact scale."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-code', default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    parser.add_argument('--scales', nargs='+', type=float, default=[.04,.06])
    parser.add_argument('--batch-size-per-scale', type=int, default=16)
    parser.add_argument('--n-iter', type=int, default=6000)
    parser.add_argument('--export-top-k', type=int, default=0, help='Export K lowest-energy standalone grasp HTML files per scale; 0 disables')
    parser.add_argument('--hull-inflation', choices=['paper', 'scale-aware'], default='scale-aware')
    parser.add_argument('--name', default='l20_small_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
    parser.add_argument('--contact-library',type=Path)
    parser.add_argument('--penetration-geometry',choices=['legacy-convex','raw-mesh'],default='legacy-convex')
    parser.add_argument('--contact-sampling',choices=['global','grouped4','grouped5'],default='global')
    parser.add_argument('--root-contact-policy',choices=['all','angle-only','all-roots-angle-only'],default='all')
    parser.add_argument('--w-normal',type=float,default=0.)
    args = parser.parse_args()
    if not args.scales or not np.isfinite(args.scales).all() or min(args.scales) <= 0:
        parser.error('Scales must be finite and positive')
    if len(set(args.scales)) != len(args.scales) or args.batch_size_per_scale < 1 or args.n_iter < 1:
        parser.error('Scales must be distinct; batch size and iterations must be positive')
    if not 0 <= args.export_top_k <= args.batch_size_per_scale:
        parser.error('--export-top-k must be between 0 and --batch-size-per-scale')
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        parser.error('Invalid experiment name')
    if not np.isfinite(args.w_normal) or args.w_normal<0 or (args.w_normal and not args.contact_library):
        parser.error('Normal loss requires a library and finite nonnegative weight')
    root = Path(__file__).resolve().parents[1]
    output = root/'data/experiments'/args.name
    if output.exists():
        parser.error(f'Experiment already exists: {output}')
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    extra=['--contact-library',str(args.contact_library.resolve()),'--w-normal',str(args.w_normal)] if args.contact_library else []
    subprocess.run([sys.executable, str(root/'grasp_generation/main.py'), '--name', args.name,
                    '--object_code_list', args.object_code, '--object-scales', *map(str,args.scales),
                    '--batch_size', str(len(args.scales)*args.batch_size_per_scale),
                    '--n_iter', str(args.n_iter), '--hull-inflation', args.hull_inflation,'--root-contact-policy',args.root_contact_policy,'--penetration-geometry',args.penetration_geometry,'--contact-sampling',args.contact_sampling,*extra], cwd=root, env=environment, check=True)
    from validate_grasp_dynamics import project_pose
    robot = ET.parse(root/'grasp_generation/mjcf/linkerhand_l20_right.urdf').getroot()
    manifest = json.loads((output/'results'/f'{args.object_code}_scales.json').read_text())
    checks = []
    for group in manifest['groups']:
        path = output/'results'/group['result_file']
        data = np.load(path, allow_pickle=True)
        assert len(data) == args.batch_size_per_scale
        assert np.isclose(group['scale'], args.scales).any()
        assert all(np.isclose(g['scale'],group['scale'],rtol=1e-7) for g in data)
        assert all(project_pose(robot,g['qpos'])[1]['source_pose_feasible'] for g in data)
        assert all(np.isfinite(list(g['qpos'].values())).all() for g in data)
        valid = [j for j,g in enumerate(data) if g['E_fc']<.3 and g['E_dis']<.005 and g['E_pen']<.001]
        checks.append({**group, 'urdf_feasible_count':len(data),'geometry_pass_count':len(valid),
                       'geometry_pass_indices':valid})
        subprocess.run([sys.executable, str(root/'grasp_generation/tests/visualize_result.py'),
                        '--object_code',args.object_code,'--result_path',str(path.parent)],
                       cwd=root,env=environment,check=True)
        print(f'Scale {group["scale"]:.4g}: extents (mm)={np.array(group["object_extents_m"])*1000}; result={path}',flush=True)
        if args.export_top_k:
            subprocess.run([sys.executable,str(root/'scripts/export_top_grasps.py'),
                            '--object-code',args.object_code,'--result-file',str(path),
                            '--top-k',str(args.export_top_k),'--output',str(path.parent/f'top_{args.export_top_k}')],
                           cwd=root,env=environment,check=True)
    (output/'scale_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(f'Scale-matched results: {output}',flush=True)


if __name__ == '__main__':
    main()
