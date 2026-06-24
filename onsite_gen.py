import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import torch

from inference import (
    DEFAULT_MAX_TOKEN_CONTEXT,
    DEFAULT_ONSITE_CKPT_PATH,
    DEFAULT_SAMPLING_MODE,
    OnSiteSmartAgent,
    compute_velocities,
    get_xodr_info,
    load_onsite_smart_model,
)


DEFAULT_DT = 0.1
DEFAULT_HISTORY_FRAMES = 31
EPS = 1e-6
DEFAULT_DYNAMICS_MAX_ACCEL = 6.0
DEFAULT_DYNAMICS_MAX_DECEL = 6.0
DEFAULT_DYNAMICS_MAX_HEADING_DELTA = 0.20
DEFAULT_DYNAMICS_MAX_SPEED = 60.0
DEFAULT_DYNAMICS_HEADING_LOOKAHEAD_STEPS = 3
DEFAULT_DYNAMICS_MIN_LOOKAHEAD_DISTANCE = 0.5
DEFAULT_DELETE_OFF_ROAD = True
DEFAULT_ROAD_EXIT_CENTERLINE_DISTANCE = 5.0
MAP_POLYGON_TYPE_VEHICLE = 0
MAP_POINT_TYPE_CENTERLINE = 16

SCORING_DYNAMIC_START_INDEX = DEFAULT_HISTORY_FRAMES
SCORING_MAX_ACCEL = 9.8
SCORING_MAX_HEADING_DELTA = 0.7
SCORING_REVERSE_SPEED_THRESHOLD = 0.5
SCORING_REVERSE_FRAMES = 10


@dataclass
class TrajectoryRecord:
    entity_id: str
    act: ET.Element
    polyline: ET.Element
    vertices: List[ET.Element]


@dataclass
class DrivableArea:
    starts: np.ndarray
    ends: np.ndarray
    max_centerline_distance: float

    def contains(self, x: float, y: float) -> bool:
        if self.starts.size == 0:
            return True

        point = np.array([x, y], dtype=np.float32)
        lower = np.minimum(self.starts, self.ends).min(axis=0) - self.max_centerline_distance
        upper = np.maximum(self.starts, self.ends).max(axis=0) + self.max_centerline_distance
        if np.any(point < lower) or np.any(point > upper):
            return False

        segments = self.ends - self.starts
        seg_len_sq = np.sum(segments * segments, axis=1)
        valid = seg_len_sq > EPS
        if not np.any(valid):
            return True

        starts = self.starts[valid]
        segments = segments[valid]
        seg_len_sq = seg_len_sq[valid]
        projection = np.sum((point - starts) * segments, axis=1) / seg_len_sq
        projection = np.clip(projection, 0.0, 1.0)
        closest = starts + projection[:, None] * segments
        distances = np.linalg.norm(closest - point, axis=1)
        return bool(np.min(distances) <= self.max_centerline_distance)


def parse_xml(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def format_float(value: float) -> str:
    return f"{float(value):.16e}"


def format_time(value: float) -> str:
    rounded = round(float(value), 10)
    text = f"{rounded:.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def ceil_to_step(value: float, step: Optional[float]) -> float:
    if step is None or step <= 0:
        return float(value)
    return round(math.ceil((float(value) - EPS) / step) * step, 10)


def scenario_name_from_exam(exam_path: Path) -> str:
    name = exam_path.name
    if not name.endswith("_exam.xosc"):
        raise ValueError(f"not an exam xosc file: {exam_path}")
    return name[: -len("_exam.xosc")]


def output_name_from_exam(exam_path: Path) -> str:
    return exam_path.name.replace("_exam.xosc", "_output.xosc")


def find_exam_files(test_dir: Path) -> List[Path]:
    return sorted(p for p in test_dir.glob("**/*_exam.xosc") if p.is_file())


def find_xodr_for_exam(exam_path: Path) -> Path:
    scenario_name = scenario_name_from_exam(exam_path)
    preferred = exam_path.with_name(f"{scenario_name}.xodr")
    if preferred.exists():
        return preferred
    candidates = sorted(exam_path.parent.glob("*.xodr"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"cannot find unique xodr for {exam_path}")


def get_world_position(vertex: ET.Element) -> ET.Element:
    world_position = vertex.find("./Position/WorldPosition")
    if world_position is None:
        world_position = vertex.find(".//WorldPosition")
    if world_position is None:
        raise ValueError("Vertex has no WorldPosition")
    return world_position


def get_vertex_time(vertex: ET.Element) -> float:
    value = vertex.get("time")
    if value is None:
        raise ValueError("Vertex has no time")
    return float(value)


def get_vertex_xyh(vertex: ET.Element) -> Tuple[float, float, float]:
    world_position = get_world_position(vertex)
    return (
        float(world_position.get("x", "0")),
        float(world_position.get("y", "0")),
        float(world_position.get("h", "0")),
    )


def get_stop_time(root: ET.Element) -> float:
    values = []
    for item in root.findall(".//StopTrigger//SimulationTimeCondition"):
        value = item.get("value")
        if value is not None:
            values.append(float(value))
    if not values:
        raise ValueError("StopTrigger SimulationTimeCondition not found")
    return max(values)


def get_entity_refs(act: ET.Element) -> List[str]:
    refs = []
    for entity_ref in act.findall(".//Actors//EntityRef"):
        ref = entity_ref.get("entityRef")
        if ref:
            refs.append(ref)
    return refs


def get_polyline(act: ET.Element) -> ET.Element:
    polyline = act.find(".//Trajectory/Shape/Polyline")
    if polyline is None:
        polyline = act.find(".//Trajectory//Polyline")
    if polyline is None:
        raise ValueError(f"{act.get('name')} has no Trajectory/Polyline")
    return polyline


def collect_storyboard_trajectories(root: ET.Element) -> Dict[str, TrajectoryRecord]:
    records: Dict[str, TrajectoryRecord] = {}
    storyboard = root.find("Storyboard")
    if storyboard is None:
        raise ValueError("Storyboard not found")

    for act in storyboard.findall(".//Act"):
        act_name = act.get("name") or ""
        if not act_name.startswith("Act_A"):
            continue
        refs = [ref for ref in get_entity_refs(act) if ref != "Ego"]
        if not refs:
            continue
        polyline = get_polyline(act)
        vertices = list(polyline.findall("Vertex"))
        if not vertices:
            raise ValueError(f"{act_name} has no Vertex")
        for ref in refs:
            records[ref] = TrajectoryRecord(
                entity_id=ref,
                act=act,
                polyline=polyline,
                vertices=vertices,
            )
    return records


def collect_ego_record(root: ET.Element) -> Optional[TrajectoryRecord]:
    storyboard = root.find("Storyboard")
    if storyboard is None:
        return None
    for act in storyboard.findall(".//Act"):
        if "Ego" not in get_entity_refs(act):
            continue
        try:
            polyline = get_polyline(act)
        except ValueError:
            continue
        vertices = list(polyline.findall("Vertex"))
        if vertices:
            return TrajectoryRecord("Ego", act, polyline, vertices)
    return None


def natural_agent_key(agent_id: str) -> Tuple[str, int, str]:
    prefix = "".join(ch for ch in agent_id if not ch.isdigit())
    digits = "".join(ch for ch in agent_id if ch.isdigit())
    number = int(digits) if digits else -1
    return prefix, number, agent_id


def get_entity_shapes(root: ET.Element) -> Dict[str, Tuple[float, float, float, int]]:
    shapes: Dict[str, Tuple[float, float, float, int]] = {}
    entities = root.find("Entities")
    if entities is None:
        raise ValueError("Entities not found")

    for scenario_object in entities.findall("ScenarioObject"):
        name = scenario_object.get("name")
        if not name:
            continue
        dims = scenario_object.find(".//BoundingBox/Dimensions")
        length = float(dims.get("length", "4.8")) if dims is not None else 4.8
        width = float(dims.get("width", "2.0")) if dims is not None else 2.0
        height = float(dims.get("height", "1.5")) if dims is not None else 1.5

        agent_type = 0
        if scenario_object.find("Pedestrian") is not None:
            agent_type = 1
        elif scenario_object.find("Vehicle") is not None:
            vehicle = scenario_object.find("Vehicle")
            category = (vehicle.get("vehicleCategory") or "").lower() if vehicle is not None else ""
            agent_type = 2 if category in {"bicycle", "motorbike"} else 0
        shapes[name] = (length, width, height, agent_type)
    return shapes


def tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_drivable_area(
    map_info: dict,
    max_centerline_distance: float = DEFAULT_ROAD_EXIT_CENTERLINE_DISTANCE,
) -> Optional[DrivableArea]:
    try:
        positions = tensor_to_numpy(map_info["map_point"]["position"])[:, :2].astype(np.float32)
        orientations = tensor_to_numpy(map_info["map_point"]["orientation"]).astype(np.float32)
        magnitudes = tensor_to_numpy(map_info["map_point"]["magnitude"]).astype(np.float32)
        point_types = tensor_to_numpy(map_info["map_point"]["type"]).astype(np.int64)
        polygon_types = tensor_to_numpy(map_info["map_polygon"]["type"]).astype(np.int64)
        edge_index = tensor_to_numpy(map_info[("map_point", "to", "map_polygon")]["edge_index"]).astype(np.int64)
    except Exception:
        return None

    if positions.size == 0 or edge_index.size == 0:
        return None

    point_polygon_ids = edge_index[1]
    valid_poly = (point_polygon_ids >= 0) & (point_polygon_ids < polygon_types.shape[0])
    polygon_type_for_point = np.full(point_polygon_ids.shape, -1, dtype=np.int64)
    polygon_type_for_point[valid_poly] = polygon_types[point_polygon_ids[valid_poly]]
    vehicle_centerline = (
        valid_poly
        & (polygon_type_for_point == MAP_POLYGON_TYPE_VEHICLE)
        & (point_types == MAP_POINT_TYPE_CENTERLINE)
        & np.isfinite(positions).all(axis=1)
        & np.isfinite(orientations)
        & np.isfinite(magnitudes)
        & (magnitudes > EPS)
    )
    if not np.any(vehicle_centerline):
        return None

    starts = positions[vehicle_centerline]
    theta = orientations[vehicle_centerline]
    length = magnitudes[vehicle_centerline]
    offsets = np.stack([np.cos(theta) * length, np.sin(theta) * length], axis=1).astype(np.float32)
    ends = starts + offsets
    return DrivableArea(
        starts=starts.astype(np.float32),
        ends=ends.astype(np.float32),
        max_centerline_distance=float(max_centerline_distance),
    )


def build_agent_info(
    root: ET.Element,
    active_records: Dict[str, TrajectoryRecord],
    history_frames: int,
) -> Tuple[dict, Dict[str, int]]:
    ego_record = collect_ego_record(root)
    ids: List[str] = []
    records: Dict[str, TrajectoryRecord] = {}
    if ego_record is not None:
        ids.append("Ego")
        records["Ego"] = ego_record

    for entity_id in sorted(active_records, key=natural_agent_key):
        ids.append(entity_id)
        records[entity_id] = active_records[entity_id]

    if not ids:
        raise ValueError("no Storyboard trajectory agents found")

    entity_shapes = get_entity_shapes(root)
    num_agents = len(ids)
    positions = torch.zeros((num_agents, history_frames, 3), dtype=torch.float32)
    headings = torch.zeros((num_agents, history_frames), dtype=torch.float32)
    valid_mask = torch.zeros((num_agents, history_frames), dtype=torch.bool)
    shapes = torch.zeros((num_agents, history_frames, 3), dtype=torch.float32)
    agent_types = torch.zeros(num_agents, dtype=torch.uint8)

    for agent_idx, entity_id in enumerate(ids):
        shape = entity_shapes.get(entity_id, (4.8, 2.0, 1.5, 0))
        shapes[agent_idx, :, :] = torch.tensor(shape[:3], dtype=torch.float32)
        agent_types[agent_idx] = shape[3]

        vertices = records[entity_id].vertices
        obs = vertices[-history_frames:]
        start = history_frames - len(obs)
        for offset, vertex in enumerate(obs):
            frame_idx = start + offset
            world_position = get_world_position(vertex)
            positions[agent_idx, frame_idx, 0] = float(world_position.get("x", "0"))
            positions[agent_idx, frame_idx, 1] = float(world_position.get("y", "0"))
            positions[agent_idx, frame_idx, 2] = float(world_position.get("z", "0"))
            headings[agent_idx, frame_idx] = float(world_position.get("h", "0"))
            valid_mask[agent_idx, frame_idx] = True

    velocities = compute_velocities(positions, valid_mask, dt=DEFAULT_DT)
    index_by_id = {entity_id: idx for idx, entity_id in enumerate(ids)}
    agent_info = {
        "num_nodes": num_agents,
        "av_index": [index_by_id["Ego"]] if "Ego" in index_by_id else [],
        "valid_mask": valid_mask,
        "predict_mask": torch.ones(num_agents, dtype=torch.bool),
        "id": ids,
        "type": agent_types,
        "category": torch.ones(num_agents, dtype=torch.uint8) * 3,
        "position": positions,
        "heading": headings,
        "velocity": velocities,
        "shape": shapes,
    }
    return agent_info, index_by_id


def append_vertex(
    polyline: ET.Element,
    template_vertex: ET.Element,
    time_value: float,
    x: float,
    y: float,
    heading: float,
) -> ET.Element:
    template_wp = get_world_position(template_vertex)
    attrs = dict(template_wp.attrib)
    attrs["x"] = format_float(x)
    attrs["y"] = format_float(y)
    attrs["z"] = attrs.get("z", "0.0000000000000000e+00")
    attrs["h"] = format_float(heading)
    if "p" in template_wp.attrib:
        attrs["p"] = template_wp.attrib["p"]
    if "r" in template_wp.attrib:
        attrs["r"] = template_wp.attrib["r"]

    vertex = ET.Element("Vertex", {"time": format_time(time_value)})
    position = ET.SubElement(vertex, "Position")
    ET.SubElement(position, "WorldPosition", attrs)
    polyline.append(vertex)
    return vertex


def create_delete_event(entity_id: str, delay_text: str) -> ET.Element:
    event = ET.Element(
        "Event",
        {
            "maximumExecutionCount": "1",
            "name": f"DeleteEntityEvent_{entity_id}",
            "priority": "parallel",
        },
    )
    action = ET.SubElement(event, "Action", {"name": f"DeleteEntityAction_{entity_id}"})
    global_action = ET.SubElement(action, "GlobalAction")
    entity_action = ET.SubElement(global_action, "EntityAction", {"entityRef": entity_id})
    ET.SubElement(entity_action, "DeleteEntityAction")
    start_trigger = ET.SubElement(event, "StartTrigger")
    condition_group = ET.SubElement(start_trigger, "ConditionGroup")
    condition = ET.SubElement(
        condition_group,
        "Condition",
        {
            "conditionEdge": "none",
            "delay": delay_text,
            "name": "DelEntityCondition",
        },
    )
    by_value = ET.SubElement(condition, "ByValueCondition")
    ET.SubElement(
        by_value,
        "SimulationTimeCondition",
        {
            "rule": "greaterThan",
            "value": "0.0",
        },
    )
    return event


def ensure_delete_delay(act: ET.Element, entity_id: str, delay_text: str) -> bool:
    updated = False
    for event in act.iter("Event"):
        has_delete_action = any(
            action.find(".//DeleteEntityAction") is not None
            for action in event.findall("Action")
        )
        if not has_delete_action:
            continue
        for condition in event.findall("./StartTrigger//Condition"):
            condition.set("delay", delay_text)
            updated = True
    if updated:
        return True

    maneuver = act.find(".//Maneuver")
    if maneuver is None:
        return False
    maneuver.append(create_delete_event(entity_id, delay_text))
    updated = True
    return updated


def interpolate_angle(a: float, b: float, ratio: float) -> float:
    diff = math.atan2(math.sin(b - a), math.cos(b - a))
    return a + ratio * diff


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def clamp_angle_change(previous: float, target: float, max_delta: float) -> float:
    delta = normalize_angle(target - previous)
    return normalize_angle(previous + clamp(delta, -max_delta, max_delta))


def estimate_last_speed(vertices: List[ET.Element]) -> float:
    if len(vertices) < 2:
        return 0.0
    x0, y0, _ = get_vertex_xyh(vertices[-2])
    x1, y1, _ = get_vertex_xyh(vertices[-1])
    t0 = get_vertex_time(vertices[-2])
    t1 = get_vertex_time(vertices[-1])
    return math.hypot(x1 - x0, y1 - y0) / max(t1 - t0, EPS)


def choose_tracking_heading(
    raw_future: List[Tuple[float, float, float, float]],
    step_index: int,
    base_x: float,
    base_y: float,
    prev_heading: float,
    lookahead_steps: int,
    min_distance: float,
) -> float:
    """Estimate the heading the agent should track at ``step_index``.

    The direction is measured along the *raw* model trajectory (from the raw
    point just before ``step_index`` toward the upcoming raw points) instead of
    relative to the already-smoothed position. Anchoring on the raw path keeps
    the target tangent stable even when the smoothed trajectory has drifted away
    from the raw one. When the agent is essentially stationary (no raw point in
    the lookahead window is farther than ``min_distance``), the previous heading
    is held so a near-stopped vehicle never spins in place.
    """
    end = min(len(raw_future), step_index + max(1, lookahead_steps))
    for idx in range(step_index, end):
        _, target_x, target_y, _ = raw_future[idx]
        dx = target_x - base_x
        dy = target_y - base_y
        if math.hypot(dx, dy) >= min_distance:
            return math.atan2(dy, dx)
    return prev_heading


def constrain_next_state(
    raw_x: float,
    raw_y: float,
    raw_heading: float,
    prev_x: float,
    prev_y: float,
    prev_heading: float,
    prev_speed: float,
    dt: float,
    max_accel: float,
    max_decel: float,
    max_heading_delta: float,
    max_speed: float,
    target_heading: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    raw_dx = raw_x - prev_x
    raw_dy = raw_y - prev_y
    raw_dist = math.hypot(raw_dx, raw_dy)
    if target_heading is None:
        if raw_dist > EPS:
            target_heading = math.atan2(raw_dy, raw_dx)
        else:
            target_heading = raw_heading

    heading = clamp_angle_change(prev_heading, target_heading, max_heading_delta)
    if raw_dist > EPS:
        forward_progress = raw_dx * math.cos(heading) + raw_dy * math.sin(heading)
        target_speed = max(0.0, forward_progress / dt)
    else:
        target_speed = 0.0

    min_speed = max(0.0, prev_speed - max_decel * dt)
    max_allowed_speed = min(max_speed, prev_speed + max_accel * dt)
    if min_speed > max_allowed_speed:
        min_speed = max_allowed_speed
    speed = clamp(target_speed, min_speed, max_allowed_speed)

    next_x = prev_x + math.cos(heading) * speed * dt
    next_y = prev_y + math.sin(heading) * speed * dt
    return next_x, next_y, heading, speed


def append_predictions_to_xml(
    active_records: Dict[str, TrajectoryRecord],
    index_by_id: Dict[str, int],
    inference_info: dict,
    stop_time: float,
    history_frames: int,
    dt: float,
    enforce_dynamics: bool = True,
    max_accel: float = DEFAULT_DYNAMICS_MAX_ACCEL,
    max_decel: float = DEFAULT_DYNAMICS_MAX_DECEL,
    max_heading_delta: float = DEFAULT_DYNAMICS_MAX_HEADING_DELTA,
    max_speed: float = DEFAULT_DYNAMICS_MAX_SPEED,
    heading_lookahead_steps: int = DEFAULT_DYNAMICS_HEADING_LOOKAHEAD_STEPS,
    min_lookahead_distance: float = DEFAULT_DYNAMICS_MIN_LOOKAHEAD_DISTANCE,
    road_checker: Optional[DrivableArea] = None,
    delete_off_road: bool = DEFAULT_DELETE_OFF_ROAD,
) -> None:
    positions = inference_info["position"].detach().cpu().numpy()
    headings = inference_info["heading"].detach().cpu().numpy()
    available_future = positions.shape[1] - history_frames

    for entity_id, record in active_records.items():
        agent_idx = index_by_id[entity_id]
        last_vertex = record.vertices[-1]
        last_time = get_vertex_time(last_vertex)
        future_steps = int(math.ceil((stop_time - last_time - EPS) / dt))
        if future_steps <= 0:
            raise ValueError(f"{entity_id} has no room for future vertices before stop_time={stop_time}")
        future_steps = min(future_steps, available_future)
        last_x, last_y, last_heading = get_vertex_xyh(last_vertex)
        smooth_x = last_x
        smooth_y = last_y
        smooth_heading = last_heading
        smooth_speed = estimate_last_speed(record.vertices)
        if delete_off_road and road_checker is not None and not road_checker.contains(smooth_x, smooth_y):
            ensure_delete_delay(record.act, entity_id, last_vertex.get("time") or format_time(last_time))
            continue

        raw_future: List[Tuple[float, float, float, float]] = []
        for step in range(1, future_steps + 1):
            pred_idx = history_frames + step - 1
            x = float(positions[agent_idx, pred_idx, 0])
            y = float(positions[agent_idx, pred_idx, 1])
            heading = float(headings[agent_idx, pred_idx])
            time_value = last_time + step * dt
            if time_value > stop_time + EPS:
                prev_x, prev_y, prev_heading = last_x, last_y, last_heading
                if step > 1:
                    prev_idx = history_frames + step - 2
                    prev_x = float(positions[agent_idx, prev_idx, 0])
                    prev_y = float(positions[agent_idx, prev_idx, 1])
                    prev_heading = float(headings[agent_idx, prev_idx])
                ratio = (stop_time - (last_time + (step - 1) * dt)) / dt
                ratio = min(max(ratio, 0.0), 1.0)
                x = prev_x + ratio * (x - prev_x)
                y = prev_y + ratio * (y - prev_y)
                heading = interpolate_angle(prev_heading, heading, ratio)
                time_value = stop_time
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(heading)):
                break
            raw_future.append((time_value, x, y, heading))
            if abs(time_value - stop_time) <= EPS:
                break

        appended = 0
        for raw_idx, (time_value, x, y, heading) in enumerate(raw_future):
            if enforce_dynamics:
                if raw_idx == 0:
                    base_x, base_y = last_x, last_y
                else:
                    base_x, base_y = raw_future[raw_idx - 1][1], raw_future[raw_idx - 1][2]
                target_heading = choose_tracking_heading(
                    raw_future=raw_future,
                    step_index=raw_idx,
                    base_x=base_x,
                    base_y=base_y,
                    prev_heading=smooth_heading,
                    lookahead_steps=heading_lookahead_steps,
                    min_distance=min_lookahead_distance,
                )
                x, y, heading, smooth_speed = constrain_next_state(
                    x,
                    y,
                    heading,
                    smooth_x,
                    smooth_y,
                    smooth_heading,
                    smooth_speed,
                    dt,
                    max_accel,
                    max_decel,
                    max_heading_delta,
                    max_speed,
                    target_heading=target_heading,
                )
                smooth_x = x
                smooth_y = y
                smooth_heading = heading
            if delete_off_road and road_checker is not None and not road_checker.contains(x, y):
                break
            append_vertex(
                record.polyline,
                last_vertex,
                time_value,
                x,
                y,
                heading,
            )
            appended += 1
            if abs(time_value - stop_time) <= EPS:
                break

        record.vertices = list(record.polyline.findall("Vertex"))
        delay_text = record.vertices[-1].get("time")
        ensure_delete_delay(record.act, entity_id, delay_text)


def infer_one_scene(
    xodr_path: Path,
    exam_path: Path,
    output_path: Path,
    model=None,
    history_frames: int = DEFAULT_HISTORY_FRAMES,
    dt: float = DEFAULT_DT,
    end_time_align: Optional[float] = 0.5,
    max_token_context: Optional[int] = DEFAULT_MAX_TOKEN_CONTEXT,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    enforce_dynamics: bool = True,
    max_accel: float = DEFAULT_DYNAMICS_MAX_ACCEL,
    max_decel: float = DEFAULT_DYNAMICS_MAX_DECEL,
    max_heading_delta: float = DEFAULT_DYNAMICS_MAX_HEADING_DELTA,
    max_speed: float = DEFAULT_DYNAMICS_MAX_SPEED,
    heading_lookahead_steps: int = DEFAULT_DYNAMICS_HEADING_LOOKAHEAD_STEPS,
    min_lookahead_distance: float = DEFAULT_DYNAMICS_MIN_LOOKAHEAD_DISTANCE,
    delete_off_road: bool = DEFAULT_DELETE_OFF_ROAD,
    road_exit_distance: float = DEFAULT_ROAD_EXIT_CENTERLINE_DISTANCE,
) -> dict:
    tree = parse_xml(exam_path)
    root = tree.getroot()
    active_records = collect_storyboard_trajectories(root)
    if not active_records:
        raise ValueError("no Act_A* background vehicle found in Storyboard")

    stop_time = get_stop_time(root)
    generation_end_time = ceil_to_step(stop_time, end_time_align)
    max_future_steps = 0
    for record in active_records.values():
        last_time = get_vertex_time(record.vertices[-1])
        max_future_steps = max(max_future_steps, int(math.ceil((generation_end_time - last_time - EPS) / dt)))
    if max_future_steps <= 0:
        raise ValueError("all active agents already reach StopTrigger time")

    map_info = get_xodr_info(str(xodr_path))
    road_checker = build_drivable_area(map_info, max_centerline_distance=road_exit_distance) if delete_off_road else None
    agent_info, index_by_id = build_agent_info(root, active_records, history_frames=history_frames)
    scenario_info = {
        "scenario_id": scenario_name_from_exam(exam_path),
        **map_info,
        "agent": agent_info,
    }

    onsite_model = OnSiteSmartAgent(
        scenario_info,
        model=model,
        max_token_context=max_token_context,
        sampling_mode=sampling_mode,
    )
    with torch.no_grad():
        last_progress_frame = -1

        def progress_callback(current_frame: int, total_frame: int) -> None:
            nonlocal last_progress_frame
            if current_frame == last_progress_frame:
                return
            remaining = max(total_frame - current_frame, 0)
            if current_frame == history_frames or remaining == 0 or current_frame - last_progress_frame >= 50:
                last_progress_frame = current_frame
                print(f"  infer: frame {current_frame}/{total_frame}", flush=True)

        inference_info = onsite_model.inference(
            total_frame=history_frames + max_future_steps,
            progress_callback=progress_callback,
        )

    append_predictions_to_xml(
        active_records=active_records,
        index_by_id=index_by_id,
        inference_info=inference_info,
        stop_time=generation_end_time,
        history_frames=history_frames,
        dt=dt,
        enforce_dynamics=enforce_dynamics,
        max_accel=max_accel,
        max_decel=max_decel,
        max_heading_delta=max_heading_delta,
        max_speed=max_speed,
        heading_lookahead_steps=heading_lookahead_steps,
        min_lookahead_distance=min_lookahead_distance,
        road_checker=road_checker,
        delete_off_road=delete_off_road,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="\t")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    validate_one_output(exam_path, output_path)
    return {
        "active_agents": len(active_records),
        "stop_time": stop_time,
        "generation_end_time": generation_end_time,
        "output_path": str(output_path),
    }


def trajectory_times(record: TrajectoryRecord) -> List[float]:
    return [get_vertex_time(vertex) for vertex in record.polyline.findall("Vertex")]


def validate_strictly_increasing(times: Iterable[float], label: str) -> None:
    prev = None
    for time_value in times:
        if prev is not None and time_value <= prev + 1e-9:
            raise ValueError(f"{label} Vertex time is not strictly increasing: {prev} -> {time_value}")
        prev = time_value


def resample_vertices_for_scoring(
    vertices: List[ET.Element],
    num_frames: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = np.zeros(num_frames, dtype=bool)
    positions = np.zeros((num_frames, 2), dtype=np.float32)
    headings = np.zeros(num_frames, dtype=np.float32)
    if not vertices:
        return valid_mask, positions, headings

    points = []
    for vertex in vertices:
        x, y, heading = get_vertex_xyh(vertex)
        points.append((round(get_vertex_time(vertex), 6), x, y, heading))

    dedup = {time_value: (x, y, heading) for time_value, x, y, heading in points}
    times = np.array(sorted(dedup), dtype=np.float64)
    xs = np.array([dedup[time_value][0] for time_value in times], dtype=np.float64)
    ys = np.array([dedup[time_value][1] for time_value in times], dtype=np.float64)
    hs = np.array([dedup[time_value][2] for time_value in times], dtype=np.float64)
    frame_times = np.arange(num_frames, dtype=np.float64) * dt

    if len(times) == 1:
        idx = int(round(float(times[0]) / dt))
        if 0 <= idx < num_frames:
            valid_mask[idx] = True
            positions[idx] = [xs[0], ys[0]]
            headings[idx] = np.mod(hs[0], 2 * np.pi)
        return valid_mask, positions, headings

    inside = (frame_times >= times[0] - 1e-6) & (frame_times <= times[-1] + 1e-6)
    valid_mask[inside] = True
    if np.any(inside):
        t_valid = frame_times[inside]
        positions[inside, 0] = np.interp(t_valid, times, xs).astype(np.float32)
        positions[inside, 1] = np.interp(t_valid, times, ys).astype(np.float32)
        headings[inside] = np.mod(np.interp(t_valid, times, np.unwrap(hs)), 2 * np.pi).astype(np.float32)
    return valid_mask, positions, headings


def validate_scoring_dynamics(root: ET.Element, dt: float = DEFAULT_DT) -> None:
    records = collect_storyboard_trajectories(root)
    entity_shapes = get_entity_shapes(root)
    if not records:
        return

    max_time = get_stop_time(root)
    for record in records.values():
        if record.vertices:
            max_time = max(max_time, get_vertex_time(record.vertices[-1]))
    num_frames = int(max_time / dt + 1e-6) + 1
    start_idx = SCORING_DYNAMIC_START_INDEX
    threshold_eps = 1e-4

    for entity_id, record in records.items():
        shape = entity_shapes.get(entity_id, (4.8, 2.0, 1.5, 0))
        if shape[3] != 0:
            continue

        valid_mask, positions, headings = resample_vertices_for_scoring(record.vertices, num_frames, dt)

        triplet_mask = valid_mask[2:] & valid_mask[1:-1] & valid_mask[:-2]
        t_triplets = np.where(triplet_mask)[0] + 2
        t_triplets = t_triplets[t_triplets >= max(start_idx, 2)]
        if t_triplets.size > 0:
            speed_now = np.linalg.norm(positions[t_triplets] - positions[t_triplets - 1], axis=1) / dt
            speed_prev = np.linalg.norm(positions[t_triplets - 1] - positions[t_triplets - 2], axis=1) / dt
            accel_vals = (speed_now - speed_prev) / dt
            min_accel = float(np.min(accel_vals))
            max_accel = float(np.max(accel_vals))
            if min_accel < -SCORING_MAX_ACCEL - threshold_eps or max_accel > SCORING_MAX_ACCEL + threshold_eps:
                raise ValueError(
                    f"{entity_id} violates scoring dynamics: acc_out_of_range "
                    f"min={min_accel:.6f} max={max_accel:.6f}"
                )

        pair_mask = valid_mask[1:] & valid_mask[:-1]
        t_pairs = np.where(pair_mask)[0] + 1
        t_pairs = t_pairs[t_pairs >= max(start_idx, 1)]
        if t_pairs.size > 0:
            heading_changes = headings[t_pairs] - headings[t_pairs - 1]
            heading_changes = np.arctan2(np.sin(heading_changes), np.cos(heading_changes))
            min_heading = float(np.min(heading_changes))
            max_heading = float(np.max(heading_changes))
            if min_heading < -SCORING_MAX_HEADING_DELTA - threshold_eps or max_heading > SCORING_MAX_HEADING_DELTA + threshold_eps:
                raise ValueError(
                    f"{entity_id} violates scoring dynamics: heading_change_out_of_range "
                    f"min={min_heading:.6f} max={max_heading:.6f}"
                )

        consecutive_reverse_count = 0
        for t_idx in range(max(start_idx, 1), num_frames):
            if not (valid_mask[t_idx] and valid_mask[t_idx - 1]):
                consecutive_reverse_count = 0
                continue

            vx, vy = (positions[t_idx] - positions[t_idx - 1]) / dt
            speed = math.hypot(float(vx), float(vy))
            if speed < SCORING_REVERSE_SPEED_THRESHOLD:
                consecutive_reverse_count = 0
                continue

            heading = float(headings[t_idx])
            forward_speed = vx * math.cos(heading) + vy * math.sin(heading)
            if forward_speed < 0:
                consecutive_reverse_count += 1
                if consecutive_reverse_count >= SCORING_REVERSE_FRAMES:
                    raise ValueError(f"{entity_id} violates scoring dynamics: continuous_reverse")
            else:
                consecutive_reverse_count = 0


def validate_one_output(exam_path: Path, output_path: Path) -> None:
    if not output_path.exists():
        raise FileNotFoundError(f"missing output: {output_path}")

    exam_tree = parse_xml(exam_path)
    output_tree = parse_xml(output_path)
    exam_root = exam_tree.getroot()
    output_root = output_tree.getroot()

    for tag in ["RoadNetwork", "Entities", "Storyboard"]:
        if output_root.find(tag) is None:
            raise ValueError(f"{output_path.name} missing {tag}")

    exam_entities = {
        item.get("name")
        for item in exam_root.findall("./Entities/ScenarioObject")
        if item.get("name") is not None
    }
    output_entities = {
        item.get("name")
        for item in output_root.findall("./Entities/ScenarioObject")
        if item.get("name") is not None
    }
    missing_entities = sorted(exam_entities - output_entities, key=natural_agent_key)
    if missing_entities:
        raise ValueError(f"{output_path.name} missing ScenarioObjects: {missing_entities}")

    exam_records = collect_storyboard_trajectories(exam_root)
    output_records = collect_storyboard_trajectories(output_root)
    output_stop_time = get_stop_time(output_root)
    missing_agents = sorted(set(exam_records) - set(output_records), key=natural_agent_key)
    if missing_agents:
        raise ValueError(f"{output_path.name} missing active agents: {missing_agents}")

    for entity_id, exam_record in exam_records.items():
        output_record = output_records[entity_id]
        exam_count = len(exam_record.polyline.findall("Vertex"))
        output_vertices = output_record.polyline.findall("Vertex")
        times = trajectory_times(output_record)
        validate_strictly_increasing(times, f"{output_path.name} {entity_id}")
        last_time = times[-1]

        delete_delays = []
        for event in output_record.act.iter("Event"):
            has_delete_action = any(
                action.find(".//DeleteEntityAction") is not None
                for action in event.findall("Action")
            )
            if not has_delete_action:
                continue
            for condition in event.findall("./StartTrigger//Condition"):
                delay = condition.get("delay")
                if delay is not None:
                    delete_delays.append(float(delay))
        has_delete_at_last_time = any(abs(delay - last_time) <= 1e-6 for delay in delete_delays)
        if len(output_vertices) <= exam_count and not has_delete_at_last_time:
            raise ValueError(
                f"{output_path.name} {entity_id} vertex count not increased: "
                f"{exam_count} -> {len(output_vertices)}"
            )
        if last_time + 1e-6 < output_stop_time and not has_delete_at_last_time:
            raise ValueError(
                f"{output_path.name} {entity_id} last Vertex time {last_time} "
                f"is earlier than StopTrigger {output_stop_time} without DeleteEntityAction"
            )
        if delete_delays and not has_delete_at_last_time:
            raise ValueError(
                f"{output_path.name} {entity_id} DeleteEntityAction delay "
                f"{delete_delays} != last Vertex time {last_time}"
            )

    validate_scoring_dynamics(output_root, dt=DEFAULT_DT)


def validate_all_outputs(test_dir: Path, output_dir: Path) -> List[str]:
    failures = []
    for exam_path in find_exam_files(test_dir):
        output_path = output_dir / output_name_from_exam(exam_path)
        try:
            validate_one_output(exam_path, output_path)
        except Exception as exc:
            failures.append(f"{exam_path}: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch SMART inference for OnSite Track 5 test scenarios.")
    parser.add_argument("--test_dir", default="data/test", type=Path)
    parser.add_argument("--output_dir", default="scene_sub", type=Path)
    parser.add_argument("--history_frames", default=DEFAULT_HISTORY_FRAMES, type=int)
    parser.add_argument("--dt", default=DEFAULT_DT, type=float)
    parser.add_argument(
        "--end_time_align",
        default=0.5,
        type=float,
        help="Round each scenario's StopTrigger up to this boundary for generated trajectories; use 0 to disable.",
    )
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Optional debug limit for the number of scenes.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip output files that already exist and pass validation.",
    )
    parser.add_argument(
        "--max_token_context",
        default=DEFAULT_MAX_TOKEN_CONTEXT,
        type=int,
        help="Use only the most recent N trajectory tokens during autoregressive inference; use 0 for full history.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=Path(DEFAULT_ONSITE_CKPT_PATH),
        type=Path,
        help="SMART checkpoint used for inference.",
    )
    parser.add_argument(
        "--sampling_mode",
        default=DEFAULT_SAMPLING_MODE,
        choices=("greedy", "topk_sample"),
        help="Token selection strategy during autoregressive rollout.",
    )
    parser.add_argument(
        "--disable_dynamics_filter",
        action="store_true",
        help="Write raw SMART trajectories without acceleration/heading projection.",
    )
    parser.add_argument("--max_accel", default=DEFAULT_DYNAMICS_MAX_ACCEL, type=float)
    parser.add_argument("--max_decel", default=DEFAULT_DYNAMICS_MAX_DECEL, type=float)
    parser.add_argument("--max_heading_delta", default=DEFAULT_DYNAMICS_MAX_HEADING_DELTA, type=float)
    parser.add_argument("--max_speed", default=DEFAULT_DYNAMICS_MAX_SPEED, type=float)
    parser.add_argument("--heading_lookahead_steps", default=DEFAULT_DYNAMICS_HEADING_LOOKAHEAD_STEPS, type=int)
    parser.add_argument("--min_lookahead_distance", default=DEFAULT_DYNAMICS_MIN_LOOKAHEAD_DISTANCE, type=float)
    parser.add_argument(
        "--disable_delete_off_road",
        action="store_true",
        help="Keep writing generated vertices even after an agent leaves the drivable road buffer.",
    )
    parser.add_argument(
        "--road_exit_distance",
        default=DEFAULT_ROAD_EXIT_CENTERLINE_DISTANCE,
        type=float,
        help="Delete a vehicle when its center is farther than this many meters from any vehicle lane centerline.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_dir = args.test_dir.resolve()
    output_dir = args.output_dir.resolve()
    ckpt_path = args.ckpt_path.expanduser().resolve()
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr, flush=True)
        return 1
    exam_files = find_exam_files(test_dir)
    if args.limit is not None:
        exam_files = exam_files[: args.limit]
    if not exam_files:
        print(f"No *_exam.xosc files found under {test_dir}", file=sys.stderr, flush=True)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(exam_files)} exam scenarios under {test_dir}", flush=True)
    print(f"Loading SMART model once for batch inference: {ckpt_path}", flush=True)
    model = load_onsite_smart_model(ckpt_path)

    failures: List[Tuple[Path, str]] = []
    success_count = 0
    total = len(exam_files)
    for idx, exam_path in enumerate(exam_files, start=1):
        scenario_name = scenario_name_from_exam(exam_path)
        output_path = output_dir / output_name_from_exam(exam_path)
        print(f"[{idx}/{total}] {scenario_name}", flush=True)
        try:
            xodr_path = find_xodr_for_exam(exam_path)
            if args.skip_existing and output_path.exists():
                try:
                    validate_one_output(exam_path, output_path)
                    success_count += 1
                    print(f"  skip existing valid output -> {output_path}", flush=True)
                    continue
                except Exception as exc:
                    print(f"  existing output invalid, regenerating: {exc}", flush=True)
            summary = infer_one_scene(
                xodr_path=xodr_path,
                exam_path=exam_path,
                output_path=output_path,
                model=model,
                history_frames=args.history_frames,
                dt=args.dt,
                end_time_align=args.end_time_align,
                max_token_context=args.max_token_context,
                sampling_mode=args.sampling_mode,
                enforce_dynamics=not args.disable_dynamics_filter,
                max_accel=args.max_accel,
                max_decel=args.max_decel,
                max_heading_delta=args.max_heading_delta,
                max_speed=args.max_speed,
                heading_lookahead_steps=args.heading_lookahead_steps,
                min_lookahead_distance=args.min_lookahead_distance,
                delete_off_road=not args.disable_delete_off_road,
                road_exit_distance=args.road_exit_distance,
            )
            success_count += 1
            print(
                f"  ok: agents={summary['active_agents']} "
                f"stop={summary['stop_time']:.3f} "
                f"gen_end={summary['generation_end_time']:.3f} -> {output_path}",
                flush=True,
            )
        except Exception as exc:
            reason = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            failures.append((exam_path, reason))
            print(f"  FAILED: {reason}", file=sys.stderr, flush=True)
            traceback.print_exc()

    validation_failures = validate_all_outputs(test_dir if args.limit is None else test_dir, output_dir)
    if args.limit is not None:
        expected_limited = {output_name_from_exam(path) for path in exam_files}
        validation_failures = [
            item for item in validation_failures
            if Path(str(item).split(":", 1)[0]).name.replace("_exam.xosc", "_output.xosc") in expected_limited
        ]

    print("=" * 80, flush=True)
    print(f"Success: {success_count}", flush=True)
    print(f"Failed: {len(failures)}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    if failures:
        print("Failed scenes:", flush=True)
        for exam_path, reason in failures:
            print(f"  - {exam_path}: {reason}", flush=True)
    if validation_failures:
        print("Validation failures:", flush=True)
        for item in validation_failures:
            print(f"  - {item}", flush=True)

    return 1 if failures or validation_failures or success_count != total else 0


if __name__ == "__main__":
    raise SystemExit(main())
