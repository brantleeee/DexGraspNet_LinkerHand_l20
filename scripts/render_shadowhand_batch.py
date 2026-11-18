"""Render every recorded ShadowHand candidate and create per-object image sheets."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("data/experiments/shadow_cup_jenga_006_all_mujoco_20261102"))
    parser.add_argument("--case", choices=("gx+", "gx-", "gy+", "gy-", "gz+", "gz-", "lift"),
                        default=None)
    parser.add_argument('--protocol', choices=('strict', 'gym'), default='strict')
    parser.add_argument('--variant', choices=('official', 'no-force'), default='official')
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show-contacts", action="store_true")
    args = parser.parse_args()
    gym_protocol = args.protocol == 'gym'
    args.case = args.case or ('gy-' if gym_protocol else 'lift')
    if gym_protocol and args.case == 'lift':
        parser.error('Gym protocol contains six gravity cases without lift')
    root = Path(__file__).resolve().parents[1]
    results = args.results.resolve()
    default_folder = args.case + ("_contacts" if args.show_contacts else "")
    base = results / 'visualizations'
    if gym_protocol:
        base = base / args.variant
    output = args.output.resolve() if args.output else base / default_folder
    summary = json.loads((results / "summary.json").read_text())
    if not summary["completed"] or (not gym_protocol and summary["backend"] != "cpu"):
        parser.error("completed native MuJoCo CPU results are required")
    output.mkdir(parents=True, exist_ok=True)
    by_object = {}
    entries = ([dict(entry, object=run['object']) for run in summary['runs'] if run['variant'] == args.variant
                for entry in run['candidates']] if gym_protocol else summary['candidates'])
    for entry in entries:
        obj = entry['object']
        index = entry['index'] if gym_protocol else entry['source_index']
        if gym_protocol:
            short = 'mug' if obj.startswith('core-mug-') else 'jenga'
            folder = results / short / args.variant / f'candidate_{index:04d}'
            scene, trajectory = folder / 'scene.xml', folder / 'trajectory.npz'
        else:
            folder = results / obj / f"candidate_{index:04d}"
            scene = folder / f"scene_{args.case}_cpu.xml"
            trajectory = folder / f"official_{args.case}_cpu.npz"
        if not scene.is_file() or not trajectory.is_file():
            parser.error(f"missing scene or trajectory: {folder}")
        video = output / f"{obj}_candidate_{index:02d}_{args.case}.mp4"
        command = [sys.executable, str(root / "scripts/render_grasp_dynamics.py"),
                   str(scene), "--trajectory", str(trajectory),
                   "--hand-mesh", "collision", "--object-mesh", "collision",
                   "--output", str(video)]
        if gym_protocol:
            direction = ('gy-', 'gy+', 'gx-', 'gx+', 'gz+', 'gz-').index(args.case)
            command += ['--object-body', f'd{direction}:object', '--speed', '1']
        if args.show_contacts:
            command.append("--show-contacts")
        if not video.is_file():
            subprocess.run(command, check=True, env={**os.environ, "MUJOCO_GL": "egl"})
        initial = video.with_name(video.stem + "_initial.png")
        if not initial.is_file():
            parser.error(f"initial frame missing: {initial}")
        by_object.setdefault(obj, []).append((index, initial))

    for obj, items in by_object.items():
        items.sort()
        thumb_w, thumb_h, columns = 320, 260, 4
        rows = (len(items) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "white")
        draw = ImageDraw.Draw(sheet)
        for cell, (index, path) in enumerate(items):
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((thumb_w, thumb_h - 24))
            x = cell % columns * thumb_w + (thumb_w - image.width) // 2
            y = cell // columns * thumb_h
            sheet.paste(image, (x, y))
            draw.text((cell % columns * thumb_w + 8, y + thumb_h - 21),
                      f"candidate {index:02d}", fill="black")
        sheet_path = output / f"{obj}_contact_sheet.png"
        sheet.save(sheet_path)
        print(f"Contact sheet: {sheet_path}", flush=True)
    print(f"Rendered {sum(map(len, by_object.values()))} candidates in {output}", flush=True)


if __name__ == "__main__":
    main()
