"""Keep optimized world poses and the exact mesh scale together on export."""

import hashlib
import json
from pathlib import Path

import numpy as np


def verify_mesh_identity(result_file, mesh_file):
    result_file = Path(result_file).resolve()
    filename = f'{result_file.stem}_scales.json'
    candidates = [result_file.parent/filename, result_file.parent.parent.parent/filename]
    for path in candidates:
        if path.exists():
            manifest = json.loads(path.read_text())
            actual = hashlib.sha256(Path(mesh_file).read_bytes()).hexdigest()
            if actual != manifest['normalized_mesh_sha256']:
                raise ValueError('Normalized object mesh changed since grasp generation; do not scale the OBJ and saved scale twice')
            return manifest
    return None  # Legacy result files did not record a mesh identity.


def save_scale_groups(result_path, object_code, data, mesh_vertices, mesh_file):
    result_path = Path(result_path)
    scales = np.array([g['scale'] for g in data])
    extent = np.ptp(np.asarray(mesh_vertices), axis=0)
    groups = []
    tags = set()
    for scale in np.unique(scales):
        tag = f'{scale:.6f}'.rstrip('0').rstrip('.').replace('.', 'p')
        if tag in tags:
            raise ValueError('Scales are too close for distinct export directory names')
        tags.add(tag)
        indices = np.flatnonzero(scales == scale).tolist()
        path = Path('scales')/f'scale_{tag}'/f'{object_code}.npy'
        (result_path/path).parent.mkdir(parents=True, exist_ok=True)
        np.save(result_path/path, [data[i] for i in indices], allow_pickle=True)
        groups.append({'scale': float(scale), 'candidate_count': len(indices),
                       'source_indices': indices, 'result_file': str(path),
                       'object_extents_m': (extent*scale).tolist()})
    manifest = {'object_code': object_code,
                'normalized_mesh_sha256': hashlib.sha256(Path(mesh_file).read_bytes()).hexdigest(),
                'normalized_mesh_extents': extent.tolist(),
                'object_pose': 'identity in optimization world frame',
                'scale_semantics': 'uniform multiplier of ORIGINAL normalized mesh vertices',
                'pose_units': 'WRJT*: meters in optimization world; WRJR* and joints: radians',
                'pose_policy': 'Reoptimized for each stored scale. Never multiply qpos or hand geometry by object scale.',
                'groups': groups}
    (result_path/f'{object_code}_scales.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
