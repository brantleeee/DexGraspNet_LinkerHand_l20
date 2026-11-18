"""Local object identifiers shared by L20 generation, validation and replay."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OBJECTS={
    'mug':dict(code='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03',label='杯子',
               default_result='data/experiments/l20_small_20261030/results/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03.npy'),
    'jenga':dict(code='2_of_Jenga_Classic_Game',label='积木',
                 default_result='data/experiments/l20_jenga_006_screen_20261030/results/2_of_Jenga_Classic_Game.npy'),
    # Bundled poses for these meshes violate the current URDF. Require a newly
    # generated result file rather than silently projecting the legacy dataset.
    'banana':dict(code='ddg-gd_banana_poisson_002',label='香蕉',default_result=None),
    'bottle':dict(code='sem-Bottle-437678d4bc6be981c8724d5673a063a6',label='瓶子',default_result=None),
    'camera':dict(code='sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5',label='相机',default_result=None),
    'power-drill-clean-l20':dict(code='power-drill-clean-l20',label='电钻clean L20',default_result=None,
                                mesh_kind='open_clean_surface_unmodified'),
    'power-drill':dict(code='power-drill',label='电钻',default_result=None,
                       mesh_kind='repaired_nonconvex_solid_union'),
}


def mesh_path(name):
    return ROOT/'data/meshdata'/OBJECTS[name]['code']/'coacd/decomposed.obj'
