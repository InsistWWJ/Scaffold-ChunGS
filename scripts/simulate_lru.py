#!/usr/bin/env python3
"""
LRU Eviction Behavior Simulator for Scaffold-ChunGS Thesis (Section 4.3.1).

Simulates chunk-level LRU eviction over a 5×5×3=75 chunk grid with camera
trajectory. Produces frame-by-frame traces and summary tables.

Output:
  - lru_frame_trace.csv       Per-frame GPU state & eviction events
  - lru_chunk_inventory.csv   Per-chunk static properties
  - lru_summary.txt           Key statistics for thesis tables
"""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration (matches scaffold_chunks.yaml / config.h defaults)
# ---------------------------------------------------------------------------
GRID_X, GRID_Y, GRID_Z = 5, 5, 3              # chunk grid dimensions
CHUNK_SIZE = 20.0                              # world units per chunk edge
MAX_ANCHORS_IN_MEMORY = 300_000                # eviction threshold
HYSTERESIS_RATIO = 0.05                        # 5% buffer
HYSTERESIS_BUFFER = int(MAX_ANCHORS_IN_MEMORY * HYSTERESIS_RATIO)
ANCHORS_PER_CHUNK_MIN = 4_000
ANCHORS_PER_CHUNK_MAX = 18_000
CAMERA_FOV_H = 70.0                            # horizontal FOV in degrees (stereo camera)
CAMERA_FOV_V = 50.0                            # vertical FOV in degrees
CAMERA_Z_FAR = 60.0                            # far plane distance
NUM_FRAMES = 200
SEED = 42

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    cid: int
    cx: int               # grid coordinate X
    cy: int               # grid coordinate Y
    cz: int               # grid coordinate Z
    center: np.ndarray    # world-space center (3,)
    num_anchors: int
    access_time: float = 0.0
    on_disk: bool = True
    in_gpu: bool = False

    @property
    def aabb_min(self) -> np.ndarray:
        return self.center - CHUNK_SIZE / 2

    @property
    def aabb_max(self) -> np.ndarray:
        return self.center + CHUNK_SIZE / 2


@dataclass
class EvictionEvent:
    frame: int
    excess_anchors: int
    candidate_chunks: List[int]          # chunk IDs eligible for eviction
    selected_chunks: List[int]           # chunk IDs actually evicted
    access_times_of_selected: List[float]
    access_times_of_retained: List[float]  # snapshot at eviction time
    anchor_counts_of_selected: List[int]


@dataclass
class LoadEvent:
    frame: int
    chunks_loaded: List[int]
    incoming_anchors: int
    projected_total: int


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
# Grid construction
# ---------------------------------------------------------------------------

def build_chunk_grid(rng: np.random.Generator) -> Dict[int, Chunk]:
    """Create 5x5x3 chunk grid with random anchor counts."""
    chunks: Dict[int, Chunk] = {}
    for cx in range(GRID_X):
        for cy in range(GRID_Y):
            for cz in range(GRID_Z):
                cid = cx * 100 + cy * 10 + cz
                center = np.array([
                    (cx + 0.5) * CHUNK_SIZE,
                    (cy + 0.5) * CHUNK_SIZE,
                    (cz + 0.5) * CHUNK_SIZE,
                ])
                num_anchors = int(rng.integers(ANCHORS_PER_CHUNK_MIN,
                                               ANCHORS_PER_CHUNK_MAX + 1))
                chunks[cid] = Chunk(cid=cid, cx=cx, cy=cy, cz=cz,
                                    center=center, num_anchors=num_anchors)
    return chunks


# ---------------------------------------------------------------------------
# Camera trajectory (zigzag through the grid)
# ---------------------------------------------------------------------------

def generate_trajectory(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Zigzag path through the chunk grid, returning positions [N,3] and forward vectors [N,3]."""
    t = np.linspace(0, 1, NUM_FRAMES)
    # Walk along X from min to max, with Y oscillation
    x = t * (GRID_X * CHUNK_SIZE)
    y = CHUNK_SIZE * GRID_Y / 2 + np.sin(t * 6 * np.pi) * CHUNK_SIZE * 1.5
    z = CHUNK_SIZE * 1.2 + np.sin(t * 3 * np.pi) * CHUNK_SIZE * 0.8
    positions = np.stack([x, y, z], axis=1)
    # Add small noise
    positions += rng.normal(0, 0.3, positions.shape)

    # Forward vector: tangent to trajectory (finite difference, normalized)
    forwards = np.zeros_like(positions)
    for i in range(1, NUM_FRAMES - 1):
        forwards[i] = positions[i + 1] - positions[i - 1]
    forwards[0] = positions[1] - positions[0]
    forwards[-1] = positions[-1] - positions[-2]
    # Normalize each row
    norms = np.linalg.norm(forwards, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    forwards = forwards / norms

    return positions, forwards


# ---------------------------------------------------------------------------
# Visibility (frustum-based, mirroring cullVisibleAnchors in model_memory.cpp)
# ---------------------------------------------------------------------------

def find_visible_chunks(cam_pos: np.ndarray,
                        cam_forward: np.ndarray,
                        chunks: Dict[int, Chunk]) -> List[int]:
    """
    Frustum-cull chunks: a chunk is visible if its AABB overlaps the camera frustum
    (simplified as FOV cone + depth range check on chunk center).
    """
    visible = []
    hfov_rad = np.radians(CAMERA_FOV_H / 2)
    vfov_rad = np.radians(CAMERA_FOV_V / 2)

    # Right and up vectors for this camera orientation
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

        # Early-out: behind camera or beyond far plane
        if dist > CAMERA_Z_FAR:
            continue

        to_dir = to_chunk / dist
        fwd_dot = np.dot(to_dir, cam_forward)
        if fwd_dot < 0.0:  # behind camera
            continue

        # Check against horizontal and vertical FOV limits
        # (allow some margin for chunk extent)
        right_dot = np.dot(to_dir, right)
        up_dot = np.dot(to_dir, up)

        h_angle = np.arctan2(abs(right_dot), fwd_dot)
        v_angle = np.arctan2(abs(up_dot), fwd_dot)

        margin = np.arctan2(CHUNK_SIZE * 0.7, dist)  # chunk half-extent at distance
        if h_angle <= hfov_rad + margin and v_angle <= vfov_rad + margin:
            visible.append(cid)

    return visible


# ---------------------------------------------------------------------------
# LRU eviction logic (mirrors model_memory.cpp)
# ---------------------------------------------------------------------------

class LRUSimulator:
    def __init__(self, chunks: Dict[int, Chunk]):
        self.chunks = chunks
        self.time = 0.0
        self.frames: List[FrameRecord] = []
        self.load_cycles: Dict[int, List[int]] = {}  # chunk_id -> [frame_nums of load/evict]

    @property
    def gpu_anchors(self) -> int:
        return sum(ch.num_anchors for ch in self.chunks.values() if ch.in_gpu)

    @property
    def gpu_chunk_ids(self) -> List[int]:
        return [ch.cid for ch in self.chunks.values() if ch.in_gpu]

    def tick(self):
        self.time += 1.0

    def load_chunks(self, cids: List[int]) -> int:
        """Load chunks from disk to GPU, return incoming anchor count."""
        total = 0
        for cid in cids:
            ch = self.chunks[cid]
            if not ch.in_gpu:
                ch.in_gpu = True
                total += ch.num_anchors
        return total

    def find_lru_candidates(self, evictable_cids: Set[int],
                            target_anchors: int) -> List[int]:
        """Sort evictable chunks by access_time ascending, pick until >= target."""
        candidates = [(cid, self.chunks[cid].access_time, self.chunks[cid].num_anchors)
                      for cid in evictable_cids]
        candidates.sort(key=lambda x: x[1])  # least recently accessed first
        selected = []
        accumulated = 0
        for cid, at, na in candidates:
            if accumulated >= target_anchors:
                break
            selected.append(cid)
            accumulated += na
        return selected

    def evict_chunks(self, cids: List[int]):
        for cid in cids:
            self.chunks[cid].in_gpu = False
            # Access time is cleared on eviction (matches code line 394)
            self.chunks[cid].access_time = 0.0

    def update_access_times(self, cids: List[int]):
        now = self.time
        for cid in cids:
            self.chunks[cid].access_time = now

    def simulate_frame(self, frame_idx: int, cam_pos: np.ndarray, cam_forward: np.ndarray):
        visible = find_visible_chunks(cam_pos, cam_forward, self.chunks)
        self.tick()

        gpu_before = self.gpu_anchors

        # 1) Load visible chunks that are not yet in GPU
        to_load = [cid for cid in visible if not self.chunks[cid].in_gpu]
        incoming = sum(self.chunks[cid].num_anchors for cid in to_load)
        projected = gpu_before + incoming

        load_event = None
        if to_load:
            # Pre-emptive eviction if projected > limit
            protected = set(visible)
            if projected > MAX_ANCHORS_IN_MEMORY:
                excess = projected - MAX_ANCHORS_IN_MEMORY
                evictable = set(self.gpu_chunk_ids) - protected
                target = excess + HYSTERESIS_BUFFER
                lru_selected = self.find_lru_candidates(evictable, target)
                access_times = [self.chunks[cid].access_time for cid in lru_selected]
                anchor_counts = [self.chunks[cid].num_anchors for cid in lru_selected]
                retained = evictable - set(lru_selected)
                retained_at = [self.chunks[cid].access_time for cid in retained]
                evict_event = EvictionEvent(
                    frame=frame_idx, excess_anchors=excess,
                    candidate_chunks=sorted(evictable),
                    selected_chunks=lru_selected,
                    access_times_of_selected=access_times,
                    access_times_of_retained=retained_at,
                    anchor_counts_of_selected=anchor_counts,
                )
                self.evict_chunks(lru_selected)
            else:
                evict_event = None

            self.load_chunks(to_load)
            load_event = LoadEvent(frame=frame_idx, chunks_loaded=to_load,
                                   incoming_anchors=incoming,
                                   projected_total=projected)
        else:
            evict_event = None

        # 2) Post-load memory pressure check
        if self.gpu_anchors > MAX_ANCHORS_IN_MEMORY:
            excess = self.gpu_anchors - MAX_ANCHORS_IN_MEMORY
            evictable = set(self.gpu_chunk_ids) - set(visible)
            target = excess + HYSTERESIS_BUFFER
            lru_selected = self.find_lru_candidates(evictable, target)
            if not evict_event:
                access_times = [self.chunks[cid].access_time for cid in lru_selected]
                anchor_counts = [self.chunks[cid].num_anchors for cid in lru_selected]
                retained = evictable - set(lru_selected)
                retained_at = [self.chunks[cid].access_time for cid in retained]
                evict_event = EvictionEvent(
                    frame=frame_idx, excess_anchors=excess,
                    candidate_chunks=sorted(evictable),
                    selected_chunks=lru_selected,
                    access_times_of_selected=access_times,
                    access_times_of_retained=retained_at,
                    anchor_counts_of_selected=anchor_counts,
                )
            self.evict_chunks(lru_selected)

        # 3) Track access times for visible (matches code line 501)
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
    """
    Find chunks that cycle load -> evict -> load within a short window,
    which the 5% hysteresis buffer should suppress.
    """
    chunk_states: Dict[int, List[Tuple[int, str]]] = {}  # cid -> [(frame, 'L'|'E')]
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
# Main
# ===========================================================================

def main():
    rng = np.random.default_rng(SEED)
    chunks = build_chunk_grid(rng)
    positions, forwards = generate_trajectory(rng)

    # Start with a few chunks preloaded near origin
    sim = LRUSimulator(chunks)
    initial_visible = find_visible_chunks(positions[0], forwards[0], chunks)
    sim.load_chunks(initial_visible)
    sim.update_access_times(initial_visible)

    print(f"Chunk grid: {GRID_X}x{GRID_Y}x{GRID_Z} = {len(chunks)} chunks")
    print(f"Total anchors: {sum(ch.num_anchors for ch in chunks.values()):,}")
    print(f"GPU limit: {MAX_ANCHORS_IN_MEMORY:,} (+{HYSTERESIS_BUFFER:,} hysteresis)")
    print(f"Camera FOV: {CAMERA_FOV_H}deg H x {CAMERA_FOV_V}deg V, far={CAMERA_Z_FAR}")
    print(f"Frames: {NUM_FRAMES}\n")

    for i in range(NUM_FRAMES):
        sim.simulate_frame(i, positions[i], forwards[i])

    frames = sim.frames

    # --- Write chunk inventory ---
    out_dir = Path(__file__).parent
    with open(out_dir / "lru_chunk_inventory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "grid_x", "grid_y", "grid_z",
                     "center_x", "center_y", "center_z", "num_anchors"])
        for cid in sorted(chunks):
            ch = chunks[cid]
            w.writerow([ch.cid, ch.cx, ch.cy, ch.cz,
                        f"{ch.center[0]:.2f}", f"{ch.center[1]:.2f}",
                        f"{ch.center[2]:.2f}", ch.num_anchors])

    # --- Write frame trace ---
    with open(out_dir / "lru_frame_trace.csv", "w", newline="") as f:
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
    eviction_frames = [r for r in frames if r.evicted]
    load_frames = [r for r in frames if r.loaded and r.loaded.chunks_loaded]
    oscillations = detect_oscillations(frames)

    gpu_series = np.array([r.gpu_anchors_after for r in frames])

    summary_lines = [
        "=" * 70,
        "LRU EVICTION EXPERIMENT — SUMMARY STATISTICS",
        "=" * 70,
        "",
        f"Chunk grid:              {GRID_X}×{GRID_Y}×{GRID_Z} = {len(chunks)}",
        f"Total anchors (all chunks): {sum(ch.num_anchors for ch in chunks.values()):,}",
        f"GPU anchor limit:         {MAX_ANCHORS_IN_MEMORY:,}",
        f"Hysteresis buffer:        {HYSTERESIS_BUFFER:,} (5%)",
        f"Effective soft limit:     {MAX_ANCHORS_IN_MEMORY + HYSTERESIS_BUFFER:,}",
        "",
        f"Frames simulated:         {NUM_FRAMES}",
        f"Frames with at least one eviction: {len(eviction_frames)}",
        f"Frames with at least one load:     {len(load_frames)}",
        f"Total eviction events:    {len(eviction_frames)}",
        f"Total load events:        {len(load_frames)}",
        "",
        f"GPU anchor count — mean:  {gpu_series.mean():.0f}",
        f"GPU anchor count — std:   {gpu_series.std():.0f}",
        f"GPU anchor count — min:   {gpu_series.min()}",
        f"GPU anchor count — max:   {gpu_series.max()}",
        f"GPU anchor count — fraction of frames over limit: "
        f"{(gpu_series > MAX_ANCHORS_IN_MEMORY).mean() * 100:.1f}%",
        "",
        f"Chunks loaded total:      {sum(len(r.loaded.chunks_loaded) for r in load_frames)}",
        f"Chunks evicted total:     {sum(len(r.evicted.selected_chunks) for r in eviction_frames)}",
        "",
        f"Boundary oscillations (L→E→L within 10 frames): {len(oscillations)}",
    ]

    if oscillations:
        summary_lines.append("")
        summary_lines.append("Oscillation details:")
        for osc in oscillations[:10]:
            summary_lines.append(
                f"  Chunk {osc['chunk_id']:3d}: "
                f"loaded@{osc['load_frame']:3d} → evicted@{osc['evict_frame']:3d} → "
                f"reloaded@{osc['reload_frame']:3d}  (span={osc['cycle_frames']} frames)"
            )

    # LRU correctness check (uses snapshots taken at eviction time)
    summary_lines.append("")
    summary_lines.append("LRU Correctness Check (all evicted access_time <= min retained access_time):")
    lru_violations = 0
    for rec in eviction_frames:
        ev = rec.evicted
        if not ev.selected_chunks:
            continue
        evicted_times = ev.access_times_of_selected
        retained_times = ev.access_times_of_retained
        if evicted_times and retained_times:
            min_retain = min(retained_times)
            ok = all(et <= min_retain for et in evicted_times)
            if not ok:
                lru_violations += 1
        # If no retained chunks, all evictable were evicted — trivially correct

    summary_lines.append(
        f"  Checked {len(eviction_frames)} eviction events, "
        f"{lru_violations} LRU violations."
    )
    if lru_violations == 0:
        summary_lines.append("  All evictions consistent with LRU discipline.")

    # Show a few detailed examples
    summary_lines.append("")
    summary_lines.append("Example eviction events (first 5 with retained chunks):")
    shown = 0
    for rec in eviction_frames:
        ev = rec.evicted
        if not ev.selected_chunks or not ev.access_times_of_retained:
            continue
        if shown >= 5:
            break
        summary_lines.append(
            f"  Frame {rec.frame:3d}: evicted={ev.selected_chunks} "
            f"(atime=[{', '.join(f'{t:.1f}' for t in ev.access_times_of_selected)}]), "
            f"retained min atime={min(ev.access_times_of_retained):.1f}"
        )
        shown += 1

    summary_text = "\n".join(summary_lines)
    with open(out_dir / "lru_summary.txt", "w") as f:
        f.write(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
