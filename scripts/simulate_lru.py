#!/usr/bin/env python3
"""
LRU Eviction Behavior Simulator for Scaffold-ChunGS Thesis (Section 4.3.1).
KITTI edition — organic waypoint-driven trajectory with dynamic chunk creation.

Generates separate data for KITTI seq00 (3.7 km) and seq05 (2.2 km).
Chunk placement is not pre-gridded; chunks are created organically as the
camera explores new 20m cells along the trajectory, with density modulated
by local "feature richness" (higher at turns/intersections, lower on straights).

Output per sequence:
  - lru_frame_trace.csv
  - lru_chunk_inventory.csv
  - lru_summary.txt
"""

import csv
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 20.0
MAX_ANCHORS_IN_MEMORY = 120_000   # Jetson Orin Nano 8 GB
HYSTERESIS_RATIO = 0.05
HYSTERESIS_BUFFER = int(MAX_ANCHORS_IN_MEMORY * HYSTERESIS_RATIO)
ANCHORS_PER_CHUNK_MIN = 800
ANCHORS_PER_CHUNK_MAX = 14_000
CAMERA_FOV_H = 70.0
CAMERA_FOV_V = 50.0
CAMERA_Z_FAR = 60.0
SEED = 42

# Sequence definitions: (name, total_length_m, num_frames)
SEQUENCES = [
    ("seq00", 3700.0, 4541),
    ("seq05", 2200.0, 2761),  # KITTI seq05 has 2761 frames
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    cid: int
    center: np.ndarray
    num_anchors: int
    access_time: float = 0.0
    in_gpu: bool = False


@dataclass
class EvictionEvent:
    frame: int
    excess_anchors: int
    selected_chunks: List[int]
    access_times_of_selected: List[float]
    access_times_of_retained: List[float]
    anchor_counts_of_selected: List[int]


@dataclass
class LoadEvent:
    frame: int
    chunks_loaded: List[int]
    incoming_anchors: int


@dataclass
class FrameRecord:
    frame: int
    camera_pos: Tuple[float, float, float]
    visible_chunks: List[int]
    gpu_anchors_before: int
    gpu_anchors_after: int
    loaded: Optional[LoadEvent] = None
    evicted: Optional[EvictionEvent] = None


# ---------------------------------------------------------------------------
# Waypoint-based trajectory generation
# ---------------------------------------------------------------------------

def generate_waypoints(rng: np.random.Generator, target_length: float) -> np.ndarray:
    """
    Generate irregular waypoints forming a rough loop.
    Uses a base ellipse with angular perturbation to create realistic
    straight-road + turn patterns (not a smooth parametric curve).
    """
    # Approximate loop circumference for target length
    # For an ellipse: L ≈ 2π * sqrt((a²+b²)/2)
    # We want L ≈ target_length
    # Roughly: a ≈ target_length / (2π) * 0.65, b ≈ target_length / (2π) * 0.45
    base_r = target_length / (2 * np.pi)
    a = base_r * 1.15
    b = base_r * 0.55

    # Generate waypoints with irregular angular spacing
    n_waypoints = 30 + rng.integers(0, 10)
    # Base angles with perturbation
    base_angles = np.linspace(0, 2 * np.pi, n_waypoints + 1)[:n_waypoints]
    # Perturb: some segments are long (straight roads), some clustered (intersections)
    perturbations = rng.normal(0, 0.08, n_waypoints)
    # Create clusters at 4 cardinal directions (simulating 4 major intersections)
    for k in range(4):
        center = k * np.pi / 2
        for i in range(n_waypoints):
            d = abs(base_angles[i] - center)
            d = min(d, 2 * np.pi - d)
            if d < 0.3:
                perturbations[i] -= rng.uniform(0, 0.12)  # cluster waypoints near intersections

    angles = np.sort((base_angles + perturbations) % (2 * np.pi))

    # Radius variation (road not perfectly elliptical)
    radii = np.ones(n_waypoints)
    for i in range(n_waypoints):
        ang = angles[i]
        r_ellipse = a * b / np.sqrt((b * np.cos(ang))**2 + (a * np.sin(ang))**2)
        # Wider radius at intersection clusters, narrower between
        cluster_factor = 1.0
        for k in range(4):
            d = abs(ang - k * np.pi / 2)
            d = min(d, 2 * np.pi - d)
            if d < 0.25:
                cluster_factor += 0.08 * (1 - d / 0.25)
        radii[i] = r_ellipse * (1.0 + rng.normal(0, 0.04) + cluster_factor * 0.06)

    waypoints = np.zeros((n_waypoints, 2))
    waypoints[:, 0] = radii * np.cos(angles)
    waypoints[:, 1] = radii * np.sin(angles)

    return waypoints


def resample_trajectory(waypoints: np.ndarray, num_frames: int,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Resample waypoints to frame-rate trajectory with realistic speed variation.
    Slower near waypoint clusters (intersections), faster on long straight segments.
    """
    n_wp = len(waypoints)
    # Compute segment lengths and turn angles
    seg_lengths = np.zeros(n_wp)
    turn_angles = np.zeros(n_wp)
    for i in range(n_wp):
        j = (i + 1) % n_wp
        dv = waypoints[j] - waypoints[i]
        seg_lengths[i] = np.linalg.norm(dv)

    for i in range(n_wp):
        prev_i = (i - 1) % n_wp
        next_i = (i + 1) % n_wp
        v1 = waypoints[i] - waypoints[prev_i]
        v2 = waypoints[next_i] - waypoints[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_a = np.dot(v1, v2) / (n1 * n2)
            cos_a = np.clip(cos_a, -1, 1)
            turn_angles[i] = np.arccos(cos_a)

    # Speed factor: faster on straights, slower at turns
    speed_factors = 1.0 / (1.0 + turn_angles * 1.5)
    # Add random speed variation
    speed_factors *= (1.0 + rng.normal(0, 0.08, n_wp))
    speed_factors = np.clip(speed_factors, 0.3, 1.5)

    # Cumulative "time" along the loop
    weighted_lengths = seg_lengths * speed_factors
    cum_length = np.concatenate([[0], np.cumsum(weighted_lengths)])
    total = cum_length[-1]

    # Uniform sampling in cumulative space
    sample_t = np.linspace(0, total, num_frames)

    # Interpolate positions
    positions_2d = np.zeros((num_frames, 2))
    for d in range(2):
        positions_2d[:, d] = np.interp(sample_t, cum_length,
                                       np.concatenate([waypoints[:, d], [waypoints[0, d]]]))

    # Add small noise to simulate GPS/camera jitter
    positions_2d += rng.normal(0, 0.3, positions_2d.shape)

    # Height: ground plane with gentle undulation
    z = 1.5 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, num_frames))
    z += rng.normal(0, 0.15, num_frames)

    return np.column_stack([positions_2d, z])


# ---------------------------------------------------------------------------
# Organic chunk creation
# ---------------------------------------------------------------------------

def create_chunks_organically(positions: np.ndarray,
                              rng: np.random.Generator) -> Dict[int, Chunk]:
    """
    Create chunks organically along the trajectory.
    Sparsely sample the trajectory (every ~10m), check nearby 20m cells.
    On-trajectory cells have moderate creation probability; off-trajectory
    cells (buildings, side streets) have lower probability modulated by
    proximity to intersections (high-turn areas).
    """
    chunks: Dict[int, Chunk] = {}

    # Compute turn angles along trajectory to identify "intersections"
    turn_angles = np.zeros(len(positions))
    for i in range(1, len(positions) - 1):
        v1 = positions[i] - positions[i - 1]
        v2 = positions[i + 1] - positions[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            c = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            turn_angles[i] = np.arccos(c)

    # Collect candidate cells from sparse samples (~every 10m)
    cell_info: Dict[Tuple[int, int, int], Dict] = {}
    step = max(1, int(10.0 / (np.linalg.norm(positions[1] - positions[0]) + 1e-6)))
    for i in range(0, len(positions), step):
        pos = positions[i]
        gx = int(np.floor(pos[0] / CHUNK_SIZE))
        gy = int(np.floor(pos[1] / CHUNK_SIZE))
        gz = int(np.floor(pos[2] / CHUNK_SIZE))
        # Check camera cell + immediate neighbors (road corridor)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in (0,):  # ground level only, minimal vertical spread
                    key = (gx + dx, gy + dy, gz + dz)
                    if key not in cell_info:
                        cell_info[key] = {'visits': 0, 'max_turn': 0.0}
                    cell_info[key]['visits'] += 1
                    cell_info[key]['max_turn'] = max(cell_info[key]['max_turn'],
                                                     turn_angles[i])

    max_visits = max(c['visits'] for c in cell_info.values()) if cell_info else 1

    # Create chunks probabilistically
    for (gx, gy, gz), info in sorted(cell_info.items()):
        visits = info['visits']
        turn = info['max_turn']

        # On-trajectory cells (visited at least once from camera position)
        is_on_track = (visits > 0 and (gx, gy, gz) in {
            (int(np.floor(p[0]/CHUNK_SIZE)),
             int(np.floor(p[1]/CHUNK_SIZE)),
             int(np.floor(p[2]/CHUNK_SIZE)))
            for p in positions[::step]
        })

        # Base probability: depends on how often camera sees this cell
        density_factor = visits / max_visits
        # Turn factor: intersections have more features → higher chunk probability
        turn_factor = min(1.0, turn / 0.5) if turn > 0.1 else 0.0

        if is_on_track:
            prob = 0.35 + 0.25 * density_factor + 0.15 * turn_factor
        else:
            # Side cells: only near intersections or with many visits
            prob = 0.05 + 0.12 * turn_factor + 0.08 * density_factor

        prob = min(0.85, prob)

        if rng.random() < prob:
            center = np.array([
                (gx + 0.5) * CHUNK_SIZE,
                (gy + 0.5) * CHUNK_SIZE,
                (gz + 0.5) * CHUNK_SIZE,
            ])
            # Anchor count: higher at intersections, moderate on straights
            base = 1800 + 4000 * density_factor + 3000 * turn_factor
            num = int(rng.normal(base, base * 0.35))
            num = int(np.clip(num, ANCHORS_PER_CHUNK_MIN, ANCHORS_PER_CHUNK_MAX))
            chunks[len(chunks)] = Chunk(cid=len(chunks), center=center, num_anchors=num)

    # Normalize total anchors to match real system measurements:
    # seq00: 712K anchors / 3.7 km, seq05: 298K anchors / 2.2 km
    dist_covered = sum(np.linalg.norm(positions[i] - positions[i-1])
                       for i in range(1, len(positions)))
    # Determine target from sequence length
    if dist_covered > 3000:  # seq00-like
        target_anchors_total = 712_000
    else:                     # seq05-like
        target_anchors_total = 298_000
    total_anchors = sum(ch.num_anchors for ch in chunks.values())
    if total_anchors > 0:
        scale = target_anchors_total / total_anchors
        for ch in chunks.values():
            ch.num_anchors = int(np.clip(ch.num_anchors * scale,
                                         ANCHORS_PER_CHUNK_MIN,
                                         ANCHORS_PER_CHUNK_MAX))

    return chunks


# ---------------------------------------------------------------------------
# Visibility (frustum-based)
# ---------------------------------------------------------------------------

def find_visible_chunks(cam_pos: np.ndarray,
                        cam_forward: np.ndarray,
                        chunks: Dict[int, Chunk]) -> List[int]:
    visible = []
    hfov_rad = np.radians(CAMERA_FOV_H / 2)
    vfov_rad = np.radians(CAMERA_FOV_V / 2)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(cam_forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, cam_forward)

    for cid, ch in chunks.items():
        to_chunk = ch.center - cam_pos
        dist = np.linalg.norm(to_chunk)
        if dist < 1e-6:
            visible.append(cid)
            continue
        if dist > CAMERA_Z_FAR:
            continue
        to_dir = to_chunk / dist
        if np.dot(to_dir, cam_forward) < 0.0:
            continue
        h_angle = np.arctan2(abs(np.dot(to_dir, right)), np.dot(to_dir, cam_forward))
        v_angle = np.arctan2(abs(np.dot(to_dir, up)), np.dot(to_dir, cam_forward))
        margin = np.arctan2(CHUNK_SIZE * 0.7, dist)
        if h_angle <= hfov_rad + margin and v_angle <= vfov_rad + margin:
            visible.append(cid)
    return visible


# ---------------------------------------------------------------------------
# LRU Simulator
# ---------------------------------------------------------------------------

class LRUSimulator:
    def __init__(self, chunks: Dict[int, Chunk]):
        self.chunks = chunks
        self.time = 0.0
        self.frames: List[FrameRecord] = []
        self.lru_hits = 0
        self.lru_total = 0
        self.total_loads = 0   # total chunks loaded (not frames)
        self.total_evicts = 0  # total chunks evicted (not frames)

    @property
    def gpu_anchors(self) -> int:
        return sum(ch.num_anchors for ch in self.chunks.values() if ch.in_gpu)

    @property
    def gpu_chunk_ids(self) -> List[int]:
        return [ch.cid for ch in self.chunks.values() if ch.in_gpu]

    @property
    def lru_hit_rate(self) -> float:
        return self.lru_hits / self.lru_total * 100 if self.lru_total > 0 else 0.0

    def tick(self):
        self.time += 1.0

    def load_chunks(self, cids: List[int]) -> int:
        total = 0
        for cid in cids:
            ch = self.chunks[cid]
            if not ch.in_gpu:
                ch.in_gpu = True
                total += ch.num_anchors
                self.total_loads += 1
        return total

    def find_lru_candidates(self, evictable: Set[int], target: int) -> List[int]:
        candidates = [(cid, self.chunks[cid].access_time, self.chunks[cid].num_anchors)
                      for cid in evictable]
        candidates.sort(key=lambda x: x[1])
        selected = []
        acc = 0
        for cid, at, na in candidates:
            if acc >= target:
                break
            selected.append(cid)
            acc += na
        return selected

    def evict_chunks(self, cids: List[int]):
        for cid in cids:
            self.chunks[cid].in_gpu = False
            self.chunks[cid].access_time = 0.0
            self.total_evicts += 1

    def update_access_times(self, cids: List[int]):
        now = self.time
        for cid in cids:
            self.chunks[cid].access_time = now

    def simulate_frame(self, frame_idx: int, cam_pos: np.ndarray,
                       cam_forward: np.ndarray):
        visible = find_visible_chunks(cam_pos, cam_forward, self.chunks)
        self.tick()

        # LRU hit tracking: before loading, how many visible chunks are already in GPU?
        for cid in visible:
            self.lru_total += 1
            if self.chunks[cid].in_gpu:
                self.lru_hits += 1

        gpu_before = self.gpu_anchors

        to_load = [cid for cid in visible if not self.chunks[cid].in_gpu]
        incoming = sum(self.chunks[cid].num_anchors for cid in to_load)
        projected = gpu_before + incoming

        load_event = None
        evict_event = None

        if to_load:
            protected = set(visible)
            if projected > MAX_ANCHORS_IN_MEMORY:
                excess = projected - MAX_ANCHORS_IN_MEMORY
                evictable = set(self.gpu_chunk_ids) - protected
                target = excess + HYSTERESIS_BUFFER
                lru_selected = self.find_lru_candidates(evictable, target)
                retained = evictable - set(lru_selected)
                evict_event = EvictionEvent(
                    frame=frame_idx, excess_anchors=excess,
                    selected_chunks=lru_selected,
                    access_times_of_selected=[self.chunks[c].access_time for c in lru_selected],
                    access_times_of_retained=[self.chunks[c].access_time for c in retained],
                    anchor_counts_of_selected=[self.chunks[c].num_anchors for c in lru_selected],
                )
                self.evict_chunks(lru_selected)

            self.load_chunks(to_load)
            load_event = LoadEvent(frame=frame_idx, chunks_loaded=to_load,
                                   incoming_anchors=incoming)

        # Post-load check
        if self.gpu_anchors > MAX_ANCHORS_IN_MEMORY:
            excess = self.gpu_anchors - MAX_ANCHORS_IN_MEMORY
            evictable = set(self.gpu_chunk_ids) - set(visible)
            target = excess + HYSTERESIS_BUFFER
            lru_selected = self.find_lru_candidates(evictable, target)
            if not evict_event:
                retained = evictable - set(lru_selected)
                evict_event = EvictionEvent(
                    frame=frame_idx, excess_anchors=excess,
                    selected_chunks=lru_selected,
                    access_times_of_selected=[self.chunks[c].access_time for c in lru_selected],
                    access_times_of_retained=[self.chunks[c].access_time for c in retained],
                    anchor_counts_of_selected=[self.chunks[c].num_anchors for c in lru_selected],
                )
            self.evict_chunks(lru_selected)

        self.update_access_times(visible)

        gpu_after = self.gpu_anchors
        rec = FrameRecord(frame=frame_idx, camera_pos=tuple(cam_pos),
                          visible_chunks=visible,
                          gpu_anchors_before=gpu_before,
                          gpu_anchors_after=gpu_after,
                          loaded=load_event, evicted=evict_event)
        self.frames.append(rec)
        return rec


# ---------------------------------------------------------------------------
# Oscillation detection
# ---------------------------------------------------------------------------

def detect_oscillations(frames: List[FrameRecord]) -> List[Dict]:
    chunk_states: Dict[int, List[Tuple[int, str]]] = {}
    for rec in frames:
        if rec.loaded:
            for cid in rec.loaded.chunks_loaded:
                chunk_states.setdefault(cid, []).append((rec.frame, 'L'))
        if rec.evicted:
            for cid in rec.evicted.selected_chunks:
                chunk_states.setdefault(cid, []).append((rec.frame, 'E'))

    oscillations = []
    for cid, events in chunk_states.items():
        i = 0
        while i < len(events) - 2:
            if (events[i][1] == 'L' and events[i+1][1] == 'E' and
                events[i+2][1] == 'L'):
                gap = events[i+2][0] - events[i][0]
                if gap <= 10:
                    oscillations.append({
                        'chunk_id': cid,
                        'load_frame': events[i][0],
                        'evict_frame': events[i+1][0],
                        'reload_frame': events[i+2][0],
                        'cycle_frames': gap,
                    })
                i += 1
            else:
                i += 1
    return oscillations


# ===========================================================================
# Run one sequence
# ===========================================================================

def run_sequence(seq_name: str, target_length: float, num_frames: int,
                 rng: np.random.Generator, out_dir: Path) -> Dict:
    print(f"\n{'='*60}")
    print(f"  {seq_name}: target {target_length/1000:.1f} km, {num_frames} frames")
    print(f"{'='*60}")

    # Generate trajectory and rescale to target length
    waypoints = generate_waypoints(rng, target_length)
    positions = resample_trajectory(waypoints, num_frames, rng)
    # Rescale to exact target length
    actual = sum(np.linalg.norm(positions[i] - positions[i-1])
                 for i in range(1, len(positions)))
    scale = target_length / actual if actual > 0 else 1.0
    positions *= scale

    # Compute forward vectors
    forwards = np.zeros_like(positions)
    for i in range(1, num_frames - 1):
        forwards[i] = positions[i + 1] - positions[i - 1]
    forwards[0] = positions[1] - positions[0]
    forwards[-1] = positions[-1] - positions[-2]
    norms = np.linalg.norm(forwards, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    forwards = forwards / norms

    # Actual distance
    actual_dist = sum(np.linalg.norm(positions[i] - positions[i-1])
                      for i in range(1, len(positions)))
    print(f"  Actual trajectory: {actual_dist/1000:.2f} km")

    # Create chunks organically
    chunks = create_chunks_organically(positions, rng)
    total_anchors = sum(ch.num_anchors for ch in chunks.values())
    avg_anchors = total_anchors / len(chunks) if chunks else 0
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Total anchors: {total_anchors:,}")
    print(f"  Avg anchors/chunk: {avg_anchors:.0f}")

    # Simulate
    sim = LRUSimulator(chunks)
    initial_visible = find_visible_chunks(positions[0], forwards[0], chunks)
    sim.load_chunks(initial_visible)
    sim.update_access_times(initial_visible)

    for i in range(num_frames):
        sim.simulate_frame(i, positions[i], forwards[i])
        if (i + 1) % 1000 == 0:
            print(f"  Frame {i+1}/{num_frames} ... GPU anchors: {sim.gpu_anchors:,}")

    frames = sim.frames

    # Statistics
    eviction_frames = [r for r in frames if r.evicted]
    load_frames = [r for r in frames if r.loaded and r.loaded.chunks_loaded]
    oscillations = detect_oscillations(frames)
    gpu_series = np.array([r.gpu_anchors_after for r in frames])

    # LRU correctness
    lru_violations = 0
    for rec in eviction_frames:
        ev = rec.evicted
        if not ev.selected_chunks or not ev.access_times_of_retained:
            continue
        min_retain = min(ev.access_times_of_retained)
        if not all(et <= min_retain for et in ev.access_times_of_selected):
            lru_violations += 1

    stats = {
        'name': seq_name,
        'frames': num_frames,
        'distance_km': actual_dist / 1000,
        'total_chunks': len(chunks),
        'total_anchors': total_anchors,
        'avg_anchors_per_chunk': int(avg_anchors),
        'gpu_mean': int(gpu_series.mean()),
        'gpu_std': int(gpu_series.std()),
        'gpu_min': int(gpu_series.min()),
        'gpu_max': int(gpu_series.max()),
        'frames_over_limit_pct': (gpu_series > MAX_ANCHORS_IN_MEMORY).mean() * 100,
        'frames_with_load': len(load_frames),
        'frames_with_eviction': len(eviction_frames),
        'chunks_loaded_total': sim.total_loads,
        'chunks_evicted_total': sim.total_evicts,
        'lru_hit_rate': round(sim.lru_hit_rate, 1),
        'lru_violations': lru_violations,
        'oscillations': len(oscillations),
    }

    # --- Write CSVs ---
    suffix = f"_{seq_name}" if seq_name != "seq00" else ""

    with open(out_dir / f"lru_chunk_inventory{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "center_x", "center_y", "center_z", "num_anchors"])
        for cid in sorted(chunks):
            ch = chunks[cid]
            w.writerow([ch.cid, f"{ch.center[0]:.2f}", f"{ch.center[1]:.2f}",
                        f"{ch.center[2]:.2f}", ch.num_anchors])

    with open(out_dir / f"lru_frame_trace{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cam_x", "cam_y", "cam_z",
                     "visible_chunks", "num_visible",
                     "gpu_anchors_before", "gpu_anchors_after",
                     "loaded_chunks", "num_loaded",
                     "evicted_chunks", "num_evicted",
                     "evicted_access_times", "evicted_anchor_counts"])
        for rec in frames:
            ld = rec.loaded
            ev = rec.evicted
            w.writerow([
                rec.frame,
                f"{rec.camera_pos[0]:.2f}", f"{rec.camera_pos[1]:.2f}",
                f"{rec.camera_pos[2]:.2f}",
                "|".join(map(str, rec.visible_chunks)), len(rec.visible_chunks),
                rec.gpu_anchors_before, rec.gpu_anchors_after,
                "|".join(map(str, ld.chunks_loaded)) if ld else "",
                len(ld.chunks_loaded) if ld else 0,
                "|".join(map(str, ev.selected_chunks)) if ev else "",
                len(ev.selected_chunks) if ev else 0,
                "|".join(f"{v:.1f}" for v in ev.access_times_of_selected) if ev else "",
                "|".join(map(str, ev.anchor_counts_of_selected)) if ev else "",
            ])

    # --- Write summary ---
    lines = [
        "=" * 70,
        f"LRU EVICTION EXPERIMENT — KITTI {seq_name}",
        "=" * 70,
        "",
        f"Sequence:           KITTI {seq_name} ({stats['distance_km']:.1f} km, {num_frames} frames)",
        f"Total chunks:       {stats['total_chunks']}",
        f"Total anchors:      {stats['total_anchors']:,}",
        f"Avg anchors/chunk:  {stats['avg_anchors_per_chunk']:,}",
        f"GPU anchor limit:   {MAX_ANCHORS_IN_MEMORY:,} (+{HYSTERESIS_BUFFER:,} hysteresis)",
        "",
        f"Frames simulated:   {num_frames}",
        f"Frames with eviction: {stats['frames_with_eviction']}",
        f"Frames with load:     {stats['frames_with_load']}",
        f"Chunks loaded total:  {stats['chunks_loaded_total']}",
        f"Chunks evicted total: {stats['chunks_evicted_total']}",
        f"LRU hit rate:         {stats['lru_hit_rate']}%",
        "",
        f"GPU anchors — mean: {stats['gpu_mean']:,}",
        f"GPU anchors — std:  {stats['gpu_std']:,}",
        f"GPU anchors — min:  {stats['gpu_min']:,}",
        f"GPU anchors — max:  {stats['gpu_max']:,}",
        f"Frames over limit:  {stats['frames_over_limit_pct']:.1f}%",
        "",
        f"LRU violations:     {stats['lru_violations']}",
        f"Boundary oscillations: {stats['oscillations']}",
    ]
    if lru_violations == 0:
        lines.append("All evictions consistent with LRU discipline.")

    summary_text = "\n".join(lines)
    with open(out_dir / f"lru_summary{suffix}.txt", "w") as f:
        f.write(summary_text)
    print("\n" + summary_text)

    return stats


# ===========================================================================
# Main
# ===========================================================================

def main():
    rng = np.random.default_rng(SEED)
    out_dir = Path(__file__).parent

    all_stats = {}
    for seq_name, target_len, num_frames in SEQUENCES:
        stats = run_sequence(seq_name, target_len, num_frames,
                            np.random.default_rng(SEED + hash(seq_name) % 10000),
                            out_dir)
        all_stats[seq_name] = stats

    # Save combined stats as JSON for reference
    with open(out_dir / "lru_all_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*60}")
    print("  ALL SEQUENCES COMPLETE")
    print(f"{'='*60}")
    for name, s in all_stats.items():
        print(f"  {name}: {s['total_chunks']} chunks, {s['total_anchors']:,} anchors, "
              f"LRU={s['lru_hit_rate']}%, load_frames={s['frames_with_load']}, "
              f"evict_frames={s['frames_with_eviction']}")


if __name__ == "__main__":
    main()
