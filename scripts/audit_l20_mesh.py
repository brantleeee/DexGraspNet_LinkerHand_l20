#!/usr/bin/env python3
"""Read-only mesh/convex-hull audit for a DexGraspNet L20 mesh.

Python dependencies: numpy, scipy, trimesh, plotly.
Use --scale for the conversion from mesh units to meters, including the
URDF mesh scale when applicable. Geometry coordinates must already be in
one consistent link frame (the analysis only compares a mesh to its hull).
"""
import argparse
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', required=True, help='One STL/OBJ mesh; usually one L20 contact link')
    p.add_argument('--scale', type=float, default=1.0, help='Mesh-unit to meter scale (default 1)')
    p.add_argument('--samples', type=int, default=12000)
    p.add_argument('--output-dir', default='audits/geometry')
    args = p.parse_args()
    if args.scale <= 0 or args.samples < 1000:
        p.error('--scale must be positive and --samples must be >= 1000')
    path = Path(args.mesh).expanduser().resolve()
    mesh = trimesh.load(str(path), force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError(f'Unable to load valid triangle mesh: {path}')
    mesh = mesh.copy()
    # STL often duplicates vertices per face; weld coincident vertices for topology metrics.
    mesh.merge_vertices()
    mesh.apply_scale(args.scale)
    hull = mesh.convex_hull
    # Directed hull->source surface discrepancy. The KD tree approximates
    # nearest mesh surface using dense samples, not exact triangle distance.
    np.random.seed(0)
    original_pts, _ = trimesh.sample.sample_surface(mesh, args.samples * 2)
    hull_pts, _ = trimesh.sample.sample_surface(hull, args.samples)
    dist_m = cKDTree(original_pts).query(hull_pts, k=1)[0]
    volume_ratio = (abs(hull.volume) / abs(mesh.volume)
                    if mesh.is_watertight and abs(mesh.volume) > 1e-12 else None)
    info = dict(
        source=str(path), scale_to_m=args.scale,
        input_watertight=bool(mesh.is_watertight),
        input_faces=len(mesh.faces), hull_faces=len(hull.faces),
        source_extents_m=np.asarray(mesh.extents).tolist(),
        source_volume_m3=float(abs(mesh.volume)) if mesh.is_watertight else None,
        hull_volume_m3=float(abs(hull.volume)),
        hull_to_source_volume_ratio=volume_ratio,
        approximate_hull_surface_to_mesh_p50_mm=float(np.percentile(dist_m, 50) * 1000),
        approximate_hull_surface_to_mesh_p95_mm=float(np.percentile(dist_m, 95) * 1000),
        approximate_hull_surface_to_mesh_max_mm=float(np.max(dist_m) * 1000),
        important_note='Surface distance is sample-based approximation; not a collision or penetration test. Check URDF scale, orientation and MJCF collision geoms separately.'
    )
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = path.stem
    json_path = out / f'{base}_convex_audit.json'
    json_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(info, indent=2, ensure_ascii=False))
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for m, name, color, opacity in [
            (mesh, 'Original mesh', '#11a6a6', 0.65),
            (hull, 'One-link convex hull', '#db4c4c', 0.24),
        ]:
            fig.add_trace(go.Mesh3d(
                x=m.vertices[:, 0], y=m.vertices[:, 1], z=m.vertices[:, 2],
                i=m.faces[:, 0], j=m.faces[:, 1], k=m.faces[:, 2],
                name=name, color=color, opacity=opacity, flatshading=True,
                showlegend=True,
            ))
        fig.update_layout(title=f'{base}: original vs. convex hull',
                          scene_aspectmode='data',
                          scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'))
        html_path = out / f'{base}_convex_overlay.html'
        fig.write_html(str(html_path), include_plotlyjs=True)
        print('HTML:', html_path)
    except ImportError:
        print('Plotly unavailable: JSON saved; install plotly for interactive HTML.')
    print('JSON:', json_path)


if __name__ == '__main__':
    main()
