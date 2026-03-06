"""
Inspect metric_cache.pkl files to understand their data structures.

Usage:
    # Inspect a random sample from the cache directory (prints + saves JSON)
    python tools/inspect_metric_cache.py

    # Inspect a specific pkl file
    python tools/inspect_metric_cache.py --pkl /path/to/metric_cache.pkl

    # Inspect by scene token
    python tools/inspect_metric_cache.py --token 7af34f0692605ad1

    # Custom output JSON path
    python tools/inspect_metric_cache.py --output /tmp/my_cache.json
"""

import argparse
import json
import lzma
import pickle
import numpy as np
from pathlib import Path
from glob import glob


CACHE_ROOT = "/export/scratch_large/ding/navsim_workspace/navmini_metric_cache"


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ─── JSON serialization helpers ───────────────────────────────────────

def ego_state_to_dict(ego_state):
    rear = ego_state.rear_axle
    vel = ego_state.dynamic_car_state.rear_axle_velocity_2d
    acc = ego_state.dynamic_car_state.rear_axle_acceleration_2d
    vp = ego_state.car_footprint.vehicle_parameters
    return {
        "position": {"x": rear.x, "y": rear.y},
        "heading_rad": rear.heading,
        "time_us": ego_state.time_point.time_us,
        "time_s": ego_state.time_point.time_s,
        "velocity": {"vx": vel.x, "vy": vel.y},
        "acceleration": {"ax": acc.x, "ay": acc.y},
        "vehicle": {
            "length": vp.length,
            "width": vp.width,
            "wheelbase": vp.wheel_base,
        },
    }


def trajectory_to_dict(trajectory):
    states = trajectory._trajectory
    waypoints = []
    for s in states:
        r = s.rear_axle
        v = s.dynamic_car_state.rear_axle_velocity_2d
        waypoints.append({
            "time_s": s.time_point.time_s,
            "x": r.x, "y": r.y, "heading": r.heading,
            "vx": v.x, "vy": v.y,
        })
    return {
        "num_states": len(states),
        "start_time_s": trajectory.start_time.time_s,
        "end_time_s": trajectory.end_time.time_s,
        "duration_s": trajectory.end_time.time_s - trajectory.start_time.time_s,
        "waypoints": waypoints,
    }


def agent_to_dict(agent):
    """Convert a nuplan Agent/StaticObject to dict."""
    d = {"type": type(agent).__name__}
    if hasattr(agent, 'center'):
        c = agent.center
        d["center"] = {"x": c.x, "y": c.y, "heading": c.heading}
    if hasattr(agent, 'velocity'):
        v = agent.velocity
        d["velocity"] = {"vx": v.x, "vy": v.y}
    if hasattr(agent, 'tracked_object_type'):
        d["object_type"] = str(agent.tracked_object_type)
    if hasattr(agent, 'box'):
        box = agent.box
        d["box"] = {"length": box.length, "width": box.width}
    return d


def observation_to_dict(observation):
    unique = observation.unique_objects
    objects = {}
    for token, agent in unique.items():
        objects[token] = agent_to_dict(agent)

    # occupancy maps: list of lists of polygons per timestep
    occ_maps = []
    if hasattr(observation, '_occupancy_maps'):
        for step_map in observation._occupancy_maps:
            if step_map is None:
                occ_maps.append(None)
            elif isinstance(step_map, np.ndarray):
                occ_maps.append({"shape": list(step_map.shape), "dtype": str(step_map.dtype)})
            else:
                occ_maps.append(str(type(step_map).__name__))

    return {
        "num_unique_objects": len(unique),
        "collided_track_ids": list(observation._collided_track_ids) if hasattr(observation, '_collided_track_ids') else [],
        "map_radius": getattr(observation, '_map_radius', None),
        "observation_samples": getattr(observation, '_observation_samples', None),
        "sample_interval": getattr(observation, '_sample_interval', None),
        "red_light_token": getattr(observation, '_red_light_token', None),
        "num_occupancy_map_steps": len(occ_maps),
        "unique_objects": objects,
    }


def centerline_to_dict(centerline):
    waypoints = []
    if hasattr(centerline, '_states'):
        for s in centerline._states:
            if hasattr(s, 'x'):
                waypoints.append({"x": s.x, "y": s.y, "heading": s.heading})

    progress = None
    if hasattr(centerline, '_progress') and isinstance(centerline._progress, np.ndarray):
        p = centerline._progress
        progress = {
            "num_points": len(p),
            "range": [float(p[0]), float(p[-1])],
            "values": p.tolist(),
        }

    return {
        "num_waypoints": len(waypoints),
        "total_length": float(centerline._progress[-1]) if progress else None,
        "progress": progress,
        "waypoints": waypoints,
    }


def drivable_area_map_to_dict(drivable_map):
    d = {"type": type(drivable_map).__name__}

    if hasattr(drivable_map, 'tokens'):
        tokens = drivable_map.tokens
        d["num_polygons"] = len(tokens)
        d["tokens"] = list(tokens) if not isinstance(tokens, list) else tokens

    if hasattr(drivable_map, 'map_types'):
        d["map_types"] = [str(t) for t in drivable_map.map_types]

    # Try to extract polygon coordinates
    if hasattr(drivable_map, '_polygons'):
        polys = drivable_map._polygons
        d["num_raw_polygons"] = len(polys) if hasattr(polys, '__len__') else str(type(polys))
        if hasattr(polys, '__len__') and len(polys) > 0:
            first = polys[0]
            if hasattr(first, 'exterior'):
                coords = list(first.exterior.coords)
                d["sample_polygon_0"] = {
                    "num_vertices": len(coords),
                    "vertices": [{"x": c[0], "y": c[1]} for c in coords],
                }

    return d


def metric_cache_to_dict(mc):
    return {
        "file_path": str(mc.file_path),
        "ego_state": ego_state_to_dict(mc.ego_state),
        "trajectory": trajectory_to_dict(mc.trajectory),
        "observation": observation_to_dict(mc.observation),
        "centerline": centerline_to_dict(mc.centerline),
        "route_lane_ids": mc.route_lane_ids,
        "drivable_area_map": drivable_area_map_to_dict(mc.drivable_area_map),
    }


# ─── Console inspection (unchanged) ──────────────────────────────────

def inspect_ego_state(ego_state):
    separator("ego_state (EgoState)")
    print(f"  Type:         {type(ego_state).__name__}")

    rear = ego_state.rear_axle
    print(f"  Position:     x={rear.x:.4f}, y={rear.y:.4f}")
    print(f"  Heading:      {rear.heading:.4f} rad")

    print(f"  Time point:   {ego_state.time_point.time_us} us "
          f"({ego_state.time_point.time_s:.3f} s)")

    vel = ego_state.dynamic_car_state.rear_axle_velocity_2d
    print(f"  Velocity:     vx={vel.x:.4f}, vy={vel.y:.4f} m/s")

    acc = ego_state.dynamic_car_state.rear_axle_acceleration_2d
    print(f"  Acceleration: ax={acc.x:.4f}, ay={acc.y:.4f} m/s²")

    vp = ego_state.car_footprint.vehicle_parameters
    print(f"  Vehicle:      length={vp.length:.2f}, width={vp.width:.2f}, "
          f"wheelbase={vp.wheel_base:.2f}")


def inspect_trajectory(trajectory):
    separator("trajectory (InterpolatedTrajectory)")
    print(f"  Type:         {type(trajectory).__name__}")

    states = trajectory._trajectory
    print(f"  Num states:   {len(states)}")
    print(f"  Time range:   {trajectory.start_time.time_s:.3f}s "
          f"-> {trajectory.end_time.time_s:.3f}s")

    duration = trajectory.end_time.time_s - trajectory.start_time.time_s
    print(f"  Duration:     {duration:.3f}s")

    print(f"\n  Sample waypoints (first 5):")
    print(f"  {'idx':>4s}  {'time_s':>10s}  {'x':>10s}  {'y':>10s}  {'heading':>10s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for i, state in enumerate(states[:5]):
        r = state.rear_axle
        print(f"  {i:4d}  {state.time_point.time_s:10.3f}  "
              f"{r.x:10.4f}  {r.y:10.4f}  {r.heading:10.4f}")

    if len(states) > 5:
        print(f"  ... ({len(states) - 5} more)")
        last = states[-1]
        r = last.rear_axle
        print(f"  {len(states)-1:4d}  {last.time_point.time_s:10.3f}  "
              f"{r.x:10.4f}  {r.y:10.4f}  {r.heading:10.4f}")


def inspect_observation(observation):
    separator("observation (PDMObservation)")
    print(f"  Type:         {type(observation).__name__}")

    attrs = [a for a in dir(observation) if not a.startswith('_')]
    print(f"  Attributes:   {attrs}")

    if hasattr(observation, '_collided_track_ids'):
        print(f"  Collided IDs: {observation._collided_track_ids}")

    if hasattr(observation, 'unique_objects'):
        unique = observation.unique_objects
        print(f"\n  unique_objects:")
        print(f"    Type:       {type(unique).__name__}")
        if hasattr(unique, '__len__'):
            print(f"    Count:      {len(unique)}")
            for i, obj in enumerate(list(unique)[:3]):
                print(f"    [{i}] {type(obj).__name__}: {obj}")

    # Inspect all private attributes that hold data
    private_attrs = [a for a in dir(observation) if a.startswith('_') and not a.startswith('__')]
    for attr_name in sorted(private_attrs):
        try:
            val = getattr(observation, attr_name)
        except Exception:
            continue
        if isinstance(val, np.ndarray):
            print(f"\n  {attr_name}:")
            print(f"    Shape: {val.shape}, dtype: {val.dtype}")
            if val.size <= 20:
                print(f"    Values: {val}")
            else:
                print(f"    Sample (first 5 flat): {val.flat[:5]}...")
        elif isinstance(val, dict):
            print(f"\n  {attr_name}:")
            print(f"    Type: dict, len={len(val)}")
            for i, (k, v) in enumerate(list(val.items())[:3]):
                if isinstance(v, np.ndarray):
                    print(f"    [{k}] ndarray shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"    [{k}] {type(v).__name__}: {v}")
            if len(val) > 3:
                print(f"    ... ({len(val) - 3} more entries)")
        elif isinstance(val, (list, set)):
            print(f"\n  {attr_name}:")
            print(f"    Type: {type(val).__name__}, len={len(val)}")
        elif not callable(val):
            print(f"\n  {attr_name}: {val}")


def inspect_centerline(centerline):
    separator("centerline (PDMPath)")
    print(f"  Type:         {type(centerline).__name__}")

    attrs = [a for a in dir(centerline) if not a.startswith('_')]
    print(f"  Attributes:   {attrs}")

    if hasattr(centerline, '_states'):
        states = centerline._states
        print(f"\n  _states (discrete waypoints):")
        print(f"    Type:  {type(states).__name__}")
        if isinstance(states, (list, np.ndarray)):
            length = len(states)
            print(f"    Count: {length}")
            print(f"\n    Sample waypoints (first 3):")
            for i, s in enumerate(states[:3]):
                if hasattr(s, 'x'):
                    print(f"    [{i}] x={s.x:.4f}, y={s.y:.4f}, heading={s.heading:.4f}")
                else:
                    print(f"    [{i}] {s}")
            if length > 3:
                print(f"    ... ({length - 3} more)")

    if hasattr(centerline, '_progress'):
        progress = centerline._progress
        print(f"\n  _progress (cumulative distance along path):")
        if isinstance(progress, np.ndarray):
            print(f"    Shape: {progress.shape}")
            print(f"    Range: [{progress[0]:.4f}, {progress[-1]:.4f}]")
        else:
            print(f"    Type: {type(progress).__name__}, value: {progress}")


def inspect_route_lane_ids(route_lane_ids):
    separator("route_lane_ids (List[str])")
    print(f"  Type:         {type(route_lane_ids).__name__}")
    print(f"  Count:        {len(route_lane_ids)}")
    if len(route_lane_ids) <= 20:
        for i, lid in enumerate(route_lane_ids):
            print(f"    [{i}] {lid}")
    else:
        for i, lid in enumerate(route_lane_ids[:5]):
            print(f"    [{i}] {lid}")
        print(f"    ... ({len(route_lane_ids) - 5} more)")


def inspect_drivable_area_map(drivable_map):
    separator("drivable_area_map (PDMDrivableMap)")
    print(f"  Type:         {type(drivable_map).__name__}")

    attrs = [a for a in dir(drivable_map) if not a.startswith('_')]
    print(f"  Attributes:   {attrs}")

    for attr_name in ['_drivable_polygon', '_drivable_area_map', '_map_radius',
                       '_drivable_area', 'drivable_polygon']:
        if hasattr(drivable_map, attr_name):
            val = getattr(drivable_map, attr_name)
            print(f"\n  {attr_name}:")
            print(f"    Type: {type(val).__name__}")
            if isinstance(val, np.ndarray):
                print(f"    Shape: {val.shape}, dtype: {val.dtype}")
            elif hasattr(val, 'area'):
                print(f"    Area: {val.area:.2f}")
            elif isinstance(val, (int, float)):
                print(f"    Value: {val}")


# ─── Log file generation ──────────────────────────────────────────────

def generate_log(mc, pkl_path):
    """Generate a detailed log string with structure + explanations."""
    lines = []
    w = lines.append  # shorthand

    file_size_kb = Path(pkl_path).stat().st_size / 1024

    w(f"MetricCache Inspection Report")
    w(f"{'='*74}")
    w(f"Source:    {pkl_path}")
    w(f"File size: {file_size_kb:.1f} KB (LZMA-compressed pickle)")
    w(f"")
    w(f"MetricCache is a dataclass that stores pre-computed scene context for")
    w(f"PDM reward evaluation during GRPO training. Each pkl corresponds to one")
    w(f"driving scenario. It is generated by run_metric_caching.py using the")
    w(f"PDM-Closed planner and nuPlan map API.")
    w(f"")
    w(f"Fields: {[f.name for f in mc.__dataclass_fields__.values()]}")

    # ── file_path ──
    w(f"")
    w(f"{'='*74}")
    w(f"1. file_path (Path)")
    w(f"{'='*74}")
    w(f"   Description: Path to this pkl file on disk.")
    w(f"   Value: {mc.file_path}")

    # ── ego_state ──
    w(f"")
    w(f"{'='*74}")
    w(f"2. ego_state (EgoState)")
    w(f"{'='*74}")
    w(f"   Description: The ego vehicle's state at the current timestep (t=0),")
    w(f"   in the GLOBAL coordinate frame. This is the starting point for")
    w(f"   closed-loop simulation. Contains pose, velocity, acceleration,")
    w(f"   and vehicle geometry (for collision box).")
    w(f"")
    rear = mc.ego_state.rear_axle
    vel = mc.ego_state.dynamic_car_state.rear_axle_velocity_2d
    acc = mc.ego_state.dynamic_car_state.rear_axle_acceleration_2d
    vp = mc.ego_state.car_footprint.vehicle_parameters
    w(f"   Position:     x={rear.x:.4f}, y={rear.y:.4f}  (UTM meters)")
    w(f"   Heading:      {rear.heading:.4f} rad")
    w(f"   Time:         {mc.ego_state.time_point.time_us} us ({mc.ego_state.time_point.time_s:.3f} s)")
    w(f"   Velocity:     vx={vel.x:.4f}, vy={vel.y:.4f} m/s")
    w(f"   Acceleration: ax={acc.x:.4f}, ay={acc.y:.4f} m/s²")
    w(f"   Vehicle:      length={vp.length:.2f}m, width={vp.width:.2f}m, wheelbase={vp.wheel_base:.2f}m")

    # ── trajectory ──
    w(f"")
    w(f"{'='*74}")
    w(f"3. trajectory (InterpolatedTrajectory)")
    w(f"{'='*74}")
    w(f"   Description: The PDM-Closed reference trajectory — an expert-like")
    w(f"   trajectory generated by the IDM (Intelligent Driver Model) planner.")
    w(f"   Used as the baseline (index 0) during PDM scoring. The model's")
    w(f"   predicted trajectory is compared against this.")
    w(f"   Coordinate frame: GLOBAL (UTM).")
    w(f"")
    states = mc.trajectory._trajectory
    duration = mc.trajectory.end_time.time_s - mc.trajectory.start_time.time_s
    w(f"   Num states:   {len(states)}")
    w(f"   Time range:   {mc.trajectory.start_time.time_s:.3f}s -> {mc.trajectory.end_time.time_s:.3f}s")
    w(f"   Duration:     {duration:.3f}s")
    w(f"   Sampling:     {len(states)-1} intervals = {(len(states)-1)/duration:.0f} Hz")
    w(f"")
    w(f"   Waypoints (first 5, last 1):")
    w(f"   {'idx':>4s}  {'time_s':>16s}  {'x':>12s}  {'y':>12s}  {'heading':>10s}")
    w(f"   {'-'*4}  {'-'*16}  {'-'*12}  {'-'*12}  {'-'*10}")
    for i, s in enumerate(states[:5]):
        r = s.rear_axle
        w(f"   {i:4d}  {s.time_point.time_s:16.3f}  {r.x:12.4f}  {r.y:12.4f}  {r.heading:10.4f}")
    if len(states) > 5:
        w(f"   ... ({len(states) - 5} more)")
        last = states[-1]
        r = last.rear_axle
        w(f"   {len(states)-1:4d}  {last.time_point.time_s:16.3f}  {r.x:12.4f}  {r.y:12.4f}  {r.heading:10.4f}")

    # ── observation ──
    w(f"")
    w(f"{'='*74}")
    w(f"4. observation (PDMObservation)")
    w(f"{'='*74}")
    w(f"   Description: Ground-truth observations of other traffic participants")
    w(f"   (vehicles, pedestrians, cyclists, static objects) over the simulation")
    w(f"   horizon. Detections are interpolated from 2Hz to 10Hz. Used for:")
    w(f"     - Collision detection (no_at_fault_collisions metric)")
    w(f"     - TTC (time-to-collision) computation")
    w(f"   Also stores per-timestep occupancy maps for spatial queries.")
    w(f"")
    unique = mc.observation.unique_objects
    w(f"   Num unique objects:    {len(unique)}")
    w(f"   Collided track IDs:   {mc.observation._collided_track_ids}")
    w(f"   Map radius:           {getattr(mc.observation, '_map_radius', 'N/A')}m")
    w(f"   Observation samples:  {getattr(mc.observation, '_observation_samples', 'N/A')}")
    w(f"   Sample interval:      {getattr(mc.observation, '_sample_interval', 'N/A')}s")
    w(f"   Occupancy map steps:  {len(mc.observation._occupancy_maps)}")
    w(f"   Red light token:      {getattr(mc.observation, '_red_light_token', 'N/A')}")
    w(f"")
    w(f"   Sample objects (first 5):")
    for i, (token, agent) in enumerate(list(unique.items())[:5]):
        desc = type(agent).__name__
        if hasattr(agent, 'center'):
            c = agent.center
            desc += f"  pos=({c.x:.1f}, {c.y:.1f}, h={c.heading:.2f})"
        if hasattr(agent, 'tracked_object_type'):
            desc += f"  type={agent.tracked_object_type}"
        w(f"     [{token}] {desc}")
    if len(unique) > 5:
        w(f"     ... ({len(unique) - 5} more)")

    # ── centerline ──
    w(f"")
    w(f"{'='*74}")
    w(f"5. centerline (PDMPath)")
    w(f"{'='*74}")
    w(f"   Description: The planned route's centerline, represented as a dense")
    w(f"   polyline with progress-based interpolation. Used for:")
    w(f"     - Driving direction compliance (is the car going the right way?)")
    w(f"     - Ego progress metric (how far along the route did the car travel?)")
    w(f"   Stored as a list of StateSE2 waypoints with cumulative arc-length (_progress).")
    w(f"")
    if hasattr(mc.centerline, '_states'):
        cl_states = mc.centerline._states
        w(f"   Num waypoints:  {len(cl_states)}")
    if hasattr(mc.centerline, '_progress') and isinstance(mc.centerline._progress, np.ndarray):
        p = mc.centerline._progress
        w(f"   Total length:   {p[-1]:.2f}m")
        w(f"   Progress array: shape={p.shape}, range=[{p[0]:.4f}, {p[-1]:.4f}]")
    w(f"")
    if hasattr(mc.centerline, '_states'):
        w(f"   Sample waypoints (first 3):")
        for i, s in enumerate(mc.centerline._states[:3]):
            if hasattr(s, 'x'):
                w(f"     [{i}] x={s.x:.4f}, y={s.y:.4f}, heading={s.heading:.4f}")

    # ── route_lane_ids ──
    w(f"")
    w(f"{'='*74}")
    w(f"6. route_lane_ids (List[str])")
    w(f"{'='*74}")
    w(f"   Description: IDs of all lane segments that constitute the planned")
    w(f"   route. Used together with the drivable_area_map to verify the ego")
    w(f"   vehicle stays on the correct route lanes.")
    w(f"")
    w(f"   Count: {len(mc.route_lane_ids)}")
    w(f"   First 10: {mc.route_lane_ids[:10]}")
    if len(mc.route_lane_ids) > 10:
        w(f"   ... ({len(mc.route_lane_ids) - 10} more)")

    # ── drivable_area_map ──
    w(f"")
    w(f"{'='*74}")
    w(f"7. drivable_area_map (PDMDrivableMap)")
    w(f"{'='*74}")
    w(f"   Description: A polygon-based representation of the drivable area")
    w(f"   within ~100m of the ego vehicle. Built from the nuPlan HD map.")
    w(f"   Used for the drivable_area_compliance metric — checking whether")
    w(f"   the ego trajectory stays within road boundaries at every timestep.")
    w(f"")
    if hasattr(mc.drivable_area_map, 'tokens'):
        w(f"   Num polygons:  {len(mc.drivable_area_map.tokens)}")
    if hasattr(mc.drivable_area_map, 'map_types'):
        w(f"   Map types:     {[str(t) for t in mc.drivable_area_map.map_types]}")
    attrs = [a for a in dir(mc.drivable_area_map) if not a.startswith('_')]
    w(f"   Public methods: {attrs}")

    # ── Summary ──
    w(f"")
    w(f"{'='*74}")
    w(f"PDM Score Usage Summary")
    w(f"{'='*74}")
    w(f"   During GRPO training, for each generated trajectory:")
    w(f"   1. ego_state      -> simulation starting point (global frame)")
    w(f"   2. trajectory      -> reference trajectory (index 0 in scoring)")
    w(f"   3. Model output    -> transform from ego frame to global frame")
    w(f"   4. Both trajectories interpolated to 40 poses @ 10Hz (4s horizon)")
    w(f"   5. PDMSimulator    -> closed-loop sim with bicycle model kinematics")
    w(f"   6. PDMScorer evaluates against:")
    w(f"      - observation       -> collision + TTC")
    w(f"      - centerline        -> driving direction + ego progress")
    w(f"      - route_lane_ids    -> route adherence")
    w(f"      - drivable_area_map -> road boundary compliance")
    w(f"   7. Final score (0-10) = binary_gates * weighted_sum(continuous_metrics)")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────

def inspect_metric_cache(pkl_path, output_json=None):
    print(f"Loading: {pkl_path}")
    print(f"File size: {Path(pkl_path).stat().st_size / 1024:.1f} KB")

    with lzma.open(pkl_path, "rb") as f:
        mc = pickle.load(f)

    # Console output
    separator(f"MetricCache Overview")
    print(f"  Type:       {type(mc).__name__}")
    print(f"  file_path:  {mc.file_path}")
    print(f"  Fields:     {[f.name for f in mc.__dataclass_fields__.values()]}")

    inspect_ego_state(mc.ego_state)
    inspect_trajectory(mc.trajectory)
    inspect_observation(mc.observation)
    inspect_centerline(mc.centerline)
    inspect_route_lane_ids(mc.route_lane_ids)
    inspect_drivable_area_map(mc.drivable_area_map)

    print(f"\n{'='*70}")
    print(f"  Done!")
    print(f"{'='*70}")

    # JSON export
    if output_json is None:
        output_json = str(pkl_path).replace('.pkl', '.json')

    print(f"\nExporting to JSON: {output_json}")
    data = metric_cache_to_dict(mc)
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"JSON saved ({Path(output_json).stat().st_size / 1024:.1f} KB)")

    # Log/txt export
    output_log = str(pkl_path).replace('.pkl', '_structure.txt')
    log_content = generate_log(mc, pkl_path)
    with open(output_log, 'w') as f:
        f.write(log_content)
    print(f"Log saved: {output_log} ({Path(output_log).stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Inspect metric_cache.pkl files")
    parser.add_argument("--pkl", type=str, default=None,
                        help="Path to a specific metric_cache.pkl file")
    parser.add_argument("--token", type=str, default=None,
                        help="Scene token to search for in cache directory")
    parser.add_argument("--cache-root", type=str, default=CACHE_ROOT,
                        help="Root directory of metric cache")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: alongside pkl)")
    args = parser.parse_args()

    if args.pkl:
        pkl_path = args.pkl
    elif args.token:
        matches = glob(f"{args.cache_root}/**/{args.token}/metric_cache.pkl", recursive=True)
        if not matches:
            print(f"No metric_cache.pkl found for token: {args.token}")
            return
        pkl_path = matches[0]
    else:
        # Pick the first available pkl
        matches = glob(f"{args.cache_root}/**/metric_cache.pkl", recursive=True)
        if not matches:
            print(f"No metric_cache.pkl found in: {args.cache_root}")
            return
        pkl_path = matches[0]
        print(f"(No --pkl or --token specified, using first available file)\n")

    inspect_metric_cache(pkl_path, args.output)


if __name__ == "__main__":
    main()
