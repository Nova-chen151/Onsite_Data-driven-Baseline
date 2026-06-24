import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from inference import get_xodr_info


def natural_key(name: str):
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(digits) if digits else -1, name


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def interp_angle(a: float, b: float, ratio: float) -> float:
    return a + ratio * wrap_angle(b - a)


def proportional_range(
    range_x: Tuple[float, float],
    range_y: Tuple[float, float],
    aspect_ratio: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    len_x = range_x[1] - range_x[0]
    len_y = range_y[1] - range_y[0]
    center_x = (range_x[0] + range_x[1]) / 2.0
    center_y = (range_y[0] + range_y[1]) / 2.0
    if len_x > len_y * aspect_ratio:
        len_y = len_x / aspect_ratio
    else:
        len_x = len_y * aspect_ratio
    return (
        (center_x - len_x / 2.0, center_x + len_x / 2.0),
        (center_y - len_y / 2.0, center_y + len_y / 2.0),
    )


def world_position(vertex: ET.Element) -> Optional[ET.Element]:
    item = vertex.find("./Position/WorldPosition")
    if item is None:
        item = vertex.find(".//WorldPosition")
    return item


def read_xosc(xosc_path: Path):
    root = ET.parse(xosc_path).getroot()

    shapes: Dict[str, Tuple[float, float]] = {}
    for scenario_object in root.findall("./Entities/ScenarioObject"):
        name = scenario_object.get("name")
        dims = scenario_object.find(".//BoundingBox/Dimensions")
        if name is None:
            continue
        if dims is None:
            shapes[name] = (4.8, 2.0)
        else:
            shapes[name] = (
                float(dims.get("length", "4.8")),
                float(dims.get("width", "2.0")),
            )

    trajectories: Dict[str, np.ndarray] = {}
    for act in root.findall(".//Act"):
        refs = [
            item.get("entityRef")
            for item in act.findall(".//Actors//EntityRef")
            if item.get("entityRef")
        ]
        if not refs:
            continue
        entity_id = refs[0]
        points = []
        for vertex in act.findall(".//Trajectory//Vertex"):
            wp = world_position(vertex)
            if wp is None:
                continue
            points.append(
                (
                    float(vertex.get("time", "0")),
                    float(wp.get("x", "0")),
                    float(wp.get("y", "0")),
                    float(wp.get("h", "0")),
                )
            )
        if points:
            data = np.array(points, dtype=float)
            trajectories[entity_id] = data[np.argsort(data[:, 0])]

    stop_values = [
        float(item.get("value"))
        for item in root.findall(".//StopTrigger//SimulationTimeCondition")
        if item.get("value") is not None
    ]
    stop_time = max(stop_values) if stop_values else None

    delete_times: Dict[str, float] = {}
    for event in root.iter("Event"):
        delete_action = event.find(".//DeleteEntityAction")
        if delete_action is None:
            continue
        entity_action = event.find(".//EntityAction")
        entity_id = entity_action.get("entityRef") if entity_action is not None else None
        if entity_id is None:
            continue
        delays = [
            float(condition.get("delay"))
            for condition in event.findall("./StartTrigger//Condition")
            if condition.get("delay") is not None
        ]
        if not delays:
            continue
        delete_time = min(delays)
        if entity_id not in delete_times or delete_time < delete_times[entity_id]:
            delete_times[entity_id] = delete_time

    return trajectories, shapes, stop_time, delete_times


def plot_static_map(ax, map_info):
    line_style = [
        ["--", 2, "yellow"], ["--", 2, "grey"], ["--", 2, "grey"],
        ["--", 2, "yellow"], ["-", 2, "yellow"], ["-", 2, "grey"],
        ["--", 2, "yellow"], ["--", 2, "grey"], ["-", 2, "yellow"],
        ["-", 2, "grey"], ["--", 2, "grey"], ["--", 2, "yellow"],
        ["-", 3, "black"], [], [], [":", 2, "blue"], [],
    ]
    center_colors = ["lightcoral", "lightgreen", "lightyellow", "lightgray"]

    edge_index = map_info[("map_point", "to", "map_polygon")]["edge_index"]
    positions = map_info["map_point"]["position"][:, :2]
    point_types = map_info["map_point"]["type"]
    light_types = map_info["map_polygon"]["light_type"]
    num_polygons = int(map_info["map_polygon"]["num_nodes"])

    for idx in range(num_polygons):
        point_idx = edge_index[0, edge_index[1] == idx]
        if len(point_idx) == 0:
            continue
        data = positions[point_idx].detach().cpu().numpy()
        point_type = int(point_types[point_idx[0]].item())
        if point_type in (13, 14):
            continue
        if point_type == 16:
            light_type = int(light_types[idx].item())
            color = center_colors[light_type] if light_type < len(center_colors) else "lightgray"
            ax.plot(data[:, 0], data[:, 1], "-", linewidth=2, color=color, alpha=0.5)
        elif point_type < len(line_style) and line_style[point_type]:
            style = line_style[point_type]
            ax.plot(data[:, 0], data[:, 1], style[0], linewidth=style[1], color=style[2], alpha=0.8)


def vehicle_corners(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    c = math.cos(heading)
    s = math.sin(heading)
    local = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ]
    )
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def state_at(traj: np.ndarray, time_value: float, hold_after_last: bool):
    if time_value < traj[0, 0]:
        return None
    if time_value >= traj[-1, 0]:
        return traj[-1] if hold_after_last else None

    next_idx = int(np.searchsorted(traj[:, 0], time_value, side="right"))
    prev_idx = max(0, next_idx - 1)
    if next_idx >= len(traj):
        return traj[-1]
    p0 = traj[prev_idx]
    p1 = traj[next_idx]
    span = p1[0] - p0[0]
    ratio = 0.0 if span <= 1e-9 else (time_value - p0[0]) / span
    return np.array(
        [
            time_value,
            p0[1] + ratio * (p1[1] - p0[1]),
            p0[2] + ratio * (p1[2] - p0[2]),
            interp_angle(p0[3], p1[3], ratio),
        ],
        dtype=float,
    )


def get_view_range(map_info, trajectories, view: str, padding: float, aspect_ratio: float):
    xy_parts = []
    if view in ("map", "both"):
        xy_parts.append(map_info["map_point"]["position"][:, :2].detach().cpu().numpy())
    if view in ("traj", "both"):
        xy_parts.extend(data[:, 1:3] for data in trajectories.values())
    if not xy_parts:
        raise ValueError("no map or trajectory points available for view range")
    xy = np.vstack(xy_parts)
    xmin, ymin = xy.min(axis=0) - padding
    xmax, ymax = xy.max(axis=0) + padding
    return proportional_range((xmin, xmax), (ymin, ymax), aspect_ratio)


def get_follow_range(
    trajectories,
    entity_ids,
    time_value: float,
    trail_seconds: float,
    hold_after_last: bool,
    delete_times: Dict[str, float],
    padding: float,
    aspect_ratio: float,
    fallback_range,
):
    xy_parts = []
    for entity_id in entity_ids:
        delete_time = delete_times.get(entity_id)
        if delete_time is not None and time_value >= delete_time - 1e-9:
            continue

        traj = trajectories[entity_id]
        state = state_at(traj, time_value, hold_after_last=hold_after_last)
        if state is not None:
            xy_parts.append(state[1:3][None, :])

        t0 = max(float(traj[0, 0]), time_value - trail_seconds)
        mask = (traj[:, 0] >= t0 - 1e-9) & (traj[:, 0] <= time_value + 1e-9)
        if mask.any():
            xy_parts.append(traj[mask, 1:3])

    if not xy_parts:
        return fallback_range
    xy = np.vstack(xy_parts)
    xmin, ymin = xy.min(axis=0) - padding
    xmax, ymax = xy.max(axis=0) + padding
    return proportional_range((xmin, xmax), (ymin, ymax), aspect_ratio)


def style_axes(ax):
    ax.set_facecolor("#f8fafc")
    ax.grid(True, color="#d6dde5", linewidth=0.7, linestyle="-", alpha=0.45)
    ax.tick_params(colors="#344054", labelsize=9)
    ax.xaxis.label.set_color("#344054")
    ax.yaxis.label.set_color("#344054")
    for spine in ax.spines.values():
        spine.set_color("#667085")
        spine.set_linewidth(0.9)


def add_scene_legend(ax, color_by_agent: bool):
    bv_label = "Background vehicles (per-agent color)" if color_by_agent else "Background vehicles"
    handles = [
        Patch(facecolor="red", edgecolor="none", label="Ego vehicle"),
        Patch(facecolor="blue", edgecolor="none", label=bv_label),
        Line2D([0], [0], color="blue", linewidth=1.8, label="Recent trajectory"),
        Line2D([0], [0], color="black", linewidth=3, label="Road boundary"),
        Line2D([0], [0], color="grey", linestyle="--", linewidth=2, label="Lane marking"),
        Line2D([0], [0], color="lightgreen", linewidth=2, alpha=0.7, label="Lane centerline"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#d0d5dd",
        fontsize=8,
        borderpad=0.7,
        labelspacing=0.45,
    )
    legend.set_zorder(30)
    return legend


def make_gif(
    xodr_path: Path,
    xosc_path: Path,
    output_gif: Path,
    fps: int,
    frame_dt: float,
    duration: Optional[float],
    view: str,
    padding: float,
    trail_seconds: float,
    hold_after_last: bool,
    color_by_agent: bool,
    camera: str,
    follow_padding: float,
):
    trajectories, shapes, stop_time, delete_times = read_xosc(xosc_path)
    if not trajectories:
        raise ValueError(f"no trajectories found in {xosc_path}")

    map_info = get_xodr_info(str(xodr_path))
    max_traj_time = max(float(data[-1, 0]) for data in trajectories.values())
    duration = max_traj_time if duration is None else min(float(duration), max_traj_time)
    times = np.round(np.arange(0.0, duration + 1e-9, frame_dt), 10)

    fig_width = 8.0
    fig_height = 6.0
    fig, ax_map = plt.subplots(figsize=(fig_width, fig_height), dpi=110)
    plot_static_map(ax_map, map_info)
    aspect_ratio = fig_width / fig_height
    range_x, range_y = get_view_range(map_info, trajectories, view, padding, aspect_ratio)
    ax_map.set_xlim(*range_x)
    ax_map.set_ylim(*range_y)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.set_title(f"Scene <{xosc_path.stem}>")

    ax_agent = ax_map.twinx()
    ax_agent.set_xlim(*range_x)
    ax_agent.set_ylim(*range_y)
    ax_agent.set_aspect("equal")
    ax_agent.axis("off")

    entity_ids = sorted(trajectories, key=natural_key)
    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(entity_ids))))
    colors = {}
    for idx, entity_id in enumerate(entity_ids):
        if entity_id == "Ego":
            colors[entity_id] = "red"
        elif color_by_agent:
            colors[entity_id] = palette[idx % len(palette)]
        else:
            colors[entity_id] = "blue"

    trail_lines = {}
    patches = {}
    labels = {}
    for entity_id in entity_ids:
        color = colors[entity_id]
        line_width = 2.4 if entity_id == "Ego" else 1.8
        trail_lines[entity_id] = ax_agent.plot([], [], color=color, linewidth=line_width, alpha=0.9)[0]
        patch = Polygon([[0, 0]], closed=True, fill=True, facecolor=color, edgecolor=None, alpha=0.9)
        ax_agent.add_patch(patch)
        patches[entity_id] = patch
        labels[entity_id] = ax_agent.text(0, 0, entity_id, color=color, fontsize=8, weight="bold")

    time_text = ax_agent.text(
        0.02,
        0.96,
        "",
        transform=ax_agent.transAxes,
        fontsize=12,
        weight="bold",
        color="black",
    )
    stop_text = f"StopTrigger={stop_time:.3f}s" if stop_time is not None else "StopTrigger=N/A"
    info_text = ax_agent.text(
        0.02,
        0.925,
        f"{stop_text}, xosc_end={max_traj_time:.3f}s",
        transform=ax_agent.transAxes,
        fontsize=9,
        color="black",
    )
    legend = add_scene_legend(ax_agent, color_by_agent=color_by_agent)

    def update(frame_idx):
        t = float(times[frame_idx])
        time_text.set_text(f"t = {t:.1f}s / {duration:.1f}s")
        artists = [time_text, info_text]

        if camera == "follow":
            next_range_x, next_range_y = get_follow_range(
                trajectories=trajectories,
                entity_ids=entity_ids,
                time_value=t,
                trail_seconds=trail_seconds,
                hold_after_last=hold_after_last,
                delete_times=delete_times,
                padding=follow_padding,
                aspect_ratio=aspect_ratio,
                fallback_range=(range_x, range_y),
            )
            ax_map.set_xlim(*next_range_x)
            ax_map.set_ylim(*next_range_y)
            ax_agent.set_xlim(*next_range_x)
            ax_agent.set_ylim(*next_range_y)

        for entity_id in entity_ids:
            traj = trajectories[entity_id]
            delete_time = delete_times.get(entity_id)
            deleted = delete_time is not None and t >= delete_time - 1e-9
            state = None if deleted else state_at(traj, t, hold_after_last=hold_after_last)
            if state is None:
                trail_lines[entity_id].set_data([], [])
                patches[entity_id].set_visible(False)
                labels[entity_id].set_visible(False)
                artists.extend([trail_lines[entity_id], patches[entity_id], labels[entity_id]])
                continue

            t0 = max(float(traj[0, 0]), t - trail_seconds)
            mask = (traj[:, 0] >= t0 - 1e-9) & (traj[:, 0] <= t + 1e-9)
            if mask.any():
                trail_lines[entity_id].set_data(traj[mask, 1], traj[mask, 2])
            else:
                trail_lines[entity_id].set_data([state[1]], [state[2]])

            length, width = shapes.get(entity_id, (4.8, 2.0))
            patches[entity_id].set_xy(vehicle_corners(state[1], state[2], state[3], length, width))
            patches[entity_id].set_visible(True)
            labels[entity_id].set_position((state[1] + 0.8, state[2] + 0.8))
            labels[entity_id].set_visible(True)
            artists.extend([trail_lines[entity_id], patches[entity_id], labels[entity_id]])
        artists.append(legend)
        return artists

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=1000.0 / fps,
        blit=(camera == "static"),
    )
    animation.save(output_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF: {output_gif.resolve()}")
    print(f"Frames: {len(times)}, fps: {fps}, duration: {len(times) / fps:.2f}s")
    print(f"XOSC trajectory end: {max_traj_time:.3f}s")


def default_output_path(xosc_path: Path, output_dir: Optional[Path] = None) -> Path:
    base_dir = output_dir if output_dir is not None else xosc_path.parent / "gifs"
    return base_dir / f"{xosc_path.stem}.gif"


def scenario_name_from_output(xosc_path: Path) -> str:
    name = xosc_path.stem
    if name.endswith("_output"):
        return name[: -len("_output")]
    return name


def find_xodr_for_output(xosc_path: Path, xodr_root: Path) -> Path:
    scenario_name = scenario_name_from_output(xosc_path)
    preferred = xodr_root / scenario_name / f"{scenario_name}.xodr"
    if preferred.exists():
        return preferred
    candidates = sorted(xodr_root.glob(f"**/{scenario_name}.xodr"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"cannot find xodr for {xosc_path} under {xodr_root}")
    raise FileNotFoundError(f"multiple xodr files found for {xosc_path}: {candidates}")


def collect_xosc_files(scene_dir: Path) -> List[Path]:
    return sorted(scene_dir.glob("*_output.xosc"))


def select_xosc_files(
    xosc_files: List[Path],
    scene_num: Optional[int],
    selection: str,
    seed: int,
) -> List[Path]:
    if not xosc_files:
        raise ValueError("no *_output.xosc files found")
    if scene_num is None or scene_num <= 0 or scene_num >= len(xosc_files):
        selected = list(xosc_files)
    elif selection == "random":
        rng = random.Random(seed)
        selected = rng.sample(xosc_files, scene_num)
        selected.sort(key=lambda p: p.name)
    else:
        selected = xosc_files[:scene_num]
    return selected


def run_gif_job(args, xosc_path: Path, xodr_path: Path):
    output_gif = args.output_gif or default_output_path(xosc_path, args.output_dir)
    make_gif(
        xodr_path=xodr_path,
        xosc_path=xosc_path,
        output_gif=output_gif,
        fps=args.fps,
        frame_dt=args.frame_dt,
        duration=args.duration,
        view=args.view,
        padding=args.padding,
        trail_seconds=args.trail_seconds,
        hold_after_last=not args.hide_after_last,
        color_by_agent=args.color_by_agent,
        camera=args.camera,
        follow_padding=args.follow_padding,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize OpenSCENARIO xosc files with their xodr maps as GIFs. "
            "Use --scene_dir for batch mode, or --xosc_path + --xodr_path for a single file."
        )
    )
    parser.add_argument("--scene_dir", default=None, type=Path, help="Directory containing *_output.xosc files.")
    parser.add_argument("--xodr_root", default=Path("data/B"), type=Path, help="Root directory to locate xodr maps.")
    parser.add_argument("--scene_num", default=None, type=int, help="Number of scenes to visualize in batch mode.")
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="Pick the first N scenes or a random subset when --scene_num is set.",
    )
    parser.add_argument("--seed", default=2026, type=int, help="Random seed used when --selection random.")
    parser.add_argument("--output_dir", default=None, type=Path, help="Output directory for batch GIFs.")
    parser.add_argument("--xodr_path", default=None, type=Path)
    parser.add_argument("--xosc_path", default=None, type=Path)
    parser.add_argument("--output_gif", default=None, type=Path)
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--frame_dt", default=0.1, type=float)
    parser.add_argument("--duration", default=None, type=float)
    parser.add_argument("--view", choices=["traj", "map", "both"], default="traj")
    parser.add_argument("--padding", default=4.0, type=float)
    parser.add_argument("--trail_seconds", default=3.0, type=float)
    parser.add_argument(
        "--camera",
        choices=["follow", "static"],
        default="static",
        help=(
            "Camera mode. Use static for a stable fixed view; follow keeps agents centered "
            "but can make the GIF feel shaky."
        ),
    )
    parser.add_argument("--follow_padding", default=28.0, type=float)
    parser.add_argument("--hide_after_last", action="store_true")
    parser.add_argument("--color_by_agent", action="store_true")
    args = parser.parse_args()

    if args.scene_dir is not None:
        xosc_files = collect_xosc_files(args.scene_dir)
        selected = select_xosc_files(xosc_files, args.scene_num, args.selection, args.seed)
        print(
            f"Batch mode: {len(selected)}/{len(xosc_files)} scenes "
            f"(selection={args.selection}, scene_num={args.scene_num}, seed={args.seed})"
        )
        failures = []
        for idx, xosc_path in enumerate(selected, start=1):
            print(f"[{idx}/{len(selected)}] {xosc_path.name}")
            try:
                xodr_path = find_xodr_for_output(xosc_path, args.xodr_root)
                run_gif_job(args, xosc_path, xodr_path)
            except Exception as exc:
                failures.append((xosc_path, exc))
                print(f"Failed: {xosc_path.name} -> {exc}")
        if failures:
            raise RuntimeError(f"{len(failures)} scene(s) failed")
        return

    if args.xosc_path is None or args.xodr_path is None:
        parser.error("single-file mode requires --xosc_path and --xodr_path, or use --scene_dir for batch mode")

    run_gif_job(args, args.xosc_path, args.xodr_path)


if __name__ == "__main__":
    main()
