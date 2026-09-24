"""Run complete fixed-seed UAV simulation episodes with a frozen Actor policy.

Usage:
    python run_inference.py --episodes 1 --seed-start 260001
    python run_inference.py --episodes 5 --output results.json

Place actor_deterministic.pt2 beside this script. Requires Python 3.12, NumPy
and PyTorch 2.8 or a compatible PyTorch version supporting torch.export.load.
This file contains the simulation runtime, observation encoding and inference
loop for complete episodes.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


# The following modules are plain, readable Python source included so this
# single file runs without importing files from the private research project.
EMBEDDED_RUNTIME_MODULES = [
    ('env.state_types', r'''
"""Typed state containers shared by the Phase 1 environment modules.

The classes in this module deliberately separate simulator truth from the
knowledge snapshots that are allowed to reach the future graph builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np


Array = np.ndarray
MeasurementId = Tuple[int, int, int]


class EntityKind(str, Enum):
    UAV = "uav"
    TARGET = "target"
    OBSTACLE = "obstacle"
    BOUNDARY = "boundary"


class CollisionKind(str, Enum):
    UAV_OBSTACLE = "uav_obstacle"
    UAV_UAV = "uav_uav"
    UAV_TARGET = "uav_target"
    TARGET_OBSTACLE = "target_obstacle"
    UAV_BOUNDARY = "uav_boundary"
    TARGET_BOUNDARY = "target_boundary"


class MeasurementKind(IntEnum):
    NONE = 0
    COARSE = 1
    FINE = 2

    @property
    def confidence(self) -> float:
        if self is MeasurementKind.FINE:
            return 1.0
        if self is MeasurementKind.COARSE:
            return 0.6
        return 0.0


class TargetTrackState(str, Enum):
    UNSEEN = "UNSEEN"
    ACQUIRED = "ACQUIRED"
    TRACKED = "TRACKED"
    LOST = "LOST"
    REACQUIRED = "REACQUIRED"


@dataclass(frozen=True)
class EnvConfig:
    map_size: float = 500.0
    dt: float = 1.0
    max_steps: int = 1000
    n_uavs: int = 4
    n_obstacles: int = 8
    num_targets: int = 1
    max_targets: int = 4
    curriculum_stage: int = 1

    # The second continuous action maps linearly onto this speed interval.
    # Matching the 5 m/s target at the lower bound makes sustained tracking
    # physically possible while retaining 10 m/s for search.
    uav_speed_min: float = 5.0
    uav_speed: float = 10.0
    uav_max_turn_deg: float = 30.0
    uav_radius: float = 1.0
    target_speed: float = 5.0
    target_max_turn_deg: float = 20.0
    target_radius: float = 1.0

    comm_radius: float = 83.0
    target_coarse_radius: float = 30.0
    target_fine_radius: float = 10.0
    coverage_radius: float = 10.0
    obstacle_sense_surface: float = 10.0
    obstacle_activation_surface: float = 30.0
    boundary_sense_distance: float = 10.0
    target_uav_sense_radius: float = 5.0
    target_obstacle_sense_surface: float = 12.0

    obstacle_radius_min: float = 4.0
    obstacle_radius_max: float = 8.0
    coverage_cell_size: float = 5.0
    coverage_context_radius: float = 100.0

    target_patrol_radius_min: float = 25.0
    target_patrol_radius_max: float = 50.0
    target_evasion_hold_steps: int = 4
    target_candidate_turn_step_deg: float = 5.0

    process_accel_sigma: float = 2.0
    coarse_position_sigma: float = 3.0
    fine_position_sigma: float = 0.5
    belief_decay: float = 0.8
    belief_lost_age: int = 4

    # Each UAV retains its local target belief across control steps.
    target_belief_temporal_memory_enabled: bool = True

    # Communication-constrained task allocation.  The allocator only uses
    # beliefs and UAV states that already exist inside a communication
    # component; it never reads an undiscovered target truth state.
    task_allocation_enabled: bool = False
    task_allocation_hold_steps: int = 8
    task_allocation_switch_margin: float = 0.10
    task_allocation_distance_weight: float = 0.70
    task_allocation_heading_weight: float = 0.30
    task_allocation_observation_bonus: float = 0.15
    # Optional opportunistic primary handoff.  A challenger must already have
    # a legal direct observation and must not be the incumbent for another
    # target, so enabling this never creates a permanent backup tracker or
    # relies on the unknown total target count.
    task_allocation_emergency_handoff_enabled: bool = False
    task_allocation_emergency_handoff_horizon_steps: float = 2.0
    task_allocation_emergency_handoff_margin: float = 2.0
    task_allocation_emergency_handoff_cooldown_steps: int = 0

    init_uav_uav_distance: float = 10.0
    init_uav_target_distance: float = 40.0
    init_target_target_distance: float = 40.0
    init_obstacle_surface_clearance: float = 10.0
    init_entity_obstacle_clearance: float = 15.0
    init_entity_boundary_clearance: float = 15.0
    patrol_clearance: float = 12.0
    init_max_attempts: int = 10_000

    include_truth_in_info: bool = True
    # Two simulator distances retained from the resolved episode config.
    preferred_track_distance: float = 10.0
    target_soft_distance: float = 7.0

    def validate(self) -> None:
        if self.n_uavs != 4:
            raise ValueError("Phase 1 requires exactly four UAVs")
        if not 1 <= self.num_targets <= self.max_targets <= 4:
            raise ValueError("num_targets must be in [1, 4]")
        if not 1 <= self.n_obstacles <= 8:
            raise ValueError("n_obstacles must be in [1, 8]")
        if self.map_size <= 0 or self.dt <= 0 or self.max_steps <= 0:
            raise ValueError("map size, dt and max_steps must be positive")
        if not 0.0 < self.uav_speed_min <= self.uav_speed:
            raise ValueError("UAV speed bounds must satisfy 0 < min <= max")
        if not (
            0 < self.target_fine_radius <= self.target_coarse_radius
            and self.obstacle_sense_surface < self.obstacle_activation_surface
        ):
            raise ValueError("sensor and activation radii are inconsistent")
        if not np.isclose(self.obstacle_activation_surface, 30.0):
            raise ValueError("the accepted obstacle activation surface range is 30 m")
        if self.task_allocation_hold_steps < 1:
            raise ValueError("task allocation hold steps must be positive")
        if self.task_allocation_switch_margin < 0.0:
            raise ValueError("task allocation switch margin must be nonnegative")
        if self.task_allocation_emergency_handoff_horizon_steps <= 0.0:
            raise ValueError("emergency handoff horizon must be positive")
        if self.task_allocation_emergency_handoff_margin < 0.0:
            raise ValueError("emergency handoff margin must be nonnegative")
        if self.task_allocation_emergency_handoff_cooldown_steps < 0:
            raise ValueError("emergency handoff cooldown cannot be negative")
        if (
            self.task_allocation_distance_weight < 0.0
            or self.task_allocation_heading_weight < 0.0
            or self.task_allocation_observation_bonus < 0.0
        ):
            raise ValueError("task allocation weights must be nonnegative")
        if (
            self.task_allocation_distance_weight
            + self.task_allocation_heading_weight
            <= 0.0
        ):
            raise ValueError("task allocation needs a nonzero geometric weight")
        cells = self.map_size / self.coverage_cell_size
        if not np.isclose(cells, round(cells)):
            raise ValueError("coverage grid must divide the map exactly")
        if int(round(cells)) % 10 != 0:
            raise ValueError(
                "coverage grid side must be divisible by 10"
            )

    @property
    def coverage_grid_size(self) -> int:
        return int(round(self.map_size / self.coverage_cell_size))

    @property
    def uav_max_turn_rad(self) -> float:
        return float(np.deg2rad(self.uav_max_turn_deg))

    @property
    def target_max_turn_rad(self) -> float:
        return float(np.deg2rad(self.target_max_turn_deg))


@dataclass
class UAVState:
    uav_id: int
    position: Array
    heading: float
    previous_action: float = 0.0

    def clone(self) -> "UAVState":
        return UAVState(
            self.uav_id,
            np.asarray(self.position, dtype=np.float64).copy(),
            float(self.heading),
            float(self.previous_action),
        )


@dataclass
class TargetState:
    target_id: int
    position: Array
    heading: float
    patrol_center: Array
    patrol_radius: float
    patrol_direction: int
    evasion_hold_remaining: int = 0
    last_evasion_heading: Optional[float] = None

    def clone(self) -> "TargetState":
        return TargetState(
            self.target_id,
            np.asarray(self.position, dtype=np.float64).copy(),
            float(self.heading),
            np.asarray(self.patrol_center, dtype=np.float64).copy(),
            float(self.patrol_radius),
            int(self.patrol_direction),
            int(self.evasion_hold_remaining),
            None if self.last_evasion_heading is None else float(self.last_evasion_heading),
        )


@dataclass(frozen=True)
class ObstacleState:
    obstacle_id: int
    center: Array
    radius: float

    def clone(self) -> "ObstacleState":
        return ObstacleState(
            int(self.obstacle_id),
            np.asarray(self.center, dtype=np.float64).copy(),
            float(self.radius),
        )


@dataclass
class WorldState:
    uavs: List[UAVState]
    targets: List[TargetState]
    obstacles: List[ObstacleState]
    step_count: int = 0

    def clone(self) -> "WorldState":
        return WorldState(
            [u.clone() for u in self.uavs],
            [t.clone() for t in self.targets],
            [o.clone() for o in self.obstacles],
            int(self.step_count),
        )


@dataclass(frozen=True)
class MotionProposal:
    kind: EntityKind
    entity_id: int
    start: Array
    end: Array
    start_heading: float
    proposed_heading: float

    @property
    def displacement(self) -> Array:
        return np.asarray(self.end, dtype=np.float64) - np.asarray(self.start, dtype=np.float64)


@dataclass(frozen=True)
class SweepPath:
    """A proposed linear path that may stop at a collision fraction.

    The simulator rolls a collided entity back for its next physical state, but
    sensing covers the physically traversed prefix up to the first contact.
    """

    start: Array
    proposed_end: Array
    stop_fraction: float = 1.0

    @property
    def displacement(self) -> Array:
        return np.asarray(self.proposed_end, dtype=np.float64) - np.asarray(self.start, dtype=np.float64)

    @property
    def sweep_end(self) -> Array:
        return np.asarray(self.start, dtype=np.float64) + float(self.stop_fraction) * self.displacement

    def position_at(self, tau: float) -> Array:
        clipped = min(max(float(tau), 0.0), float(self.stop_fraction))
        return np.asarray(self.start, dtype=np.float64) + clipped * self.displacement


@dataclass(frozen=True)
class CollisionEvent:
    kind: CollisionKind
    tau: float
    entity_a_kind: EntityKind
    entity_a_id: int
    entity_b_kind: EntityKind
    entity_b_id: Optional[int]
    normal_a: Array
    obstacle_id: Optional[int] = None
    boundary_name: Optional[str] = None

    @property
    def entity_keys(self) -> Tuple[Tuple[EntityKind, int], ...]:
        first = (self.entity_a_kind, self.entity_a_id)
        if self.entity_b_kind in (EntityKind.UAV, EntityKind.TARGET) and self.entity_b_id is not None:
            return first, (self.entity_b_kind, self.entity_b_id)
        return (first,)


@dataclass(frozen=True)
class TargetMeasurement:
    target_id: int
    source_uav_id: int
    control_step: int
    tau: float
    kind: MeasurementKind
    position: Array
    covariance: Array

    @property
    def measurement_id(self) -> MeasurementId:
        return self.target_id, self.source_uav_id, self.control_step

    @property
    def observation_time(self) -> float:
        return float(self.control_step) + float(self.tau)


@dataclass(frozen=True)
class TargetObservationMetadata:
    self_level: float = 0.0
    component_best_level: float = 0.0
    component_observer_ratio: float = 0.0


@dataclass(frozen=True)
class TargetAssignmentMetadata:
    """Legal task-allocation message received by one UAV for one target."""

    self_primary: float = 0.0
    component_primary_present: float = 0.0
    self_bid: float = 1.0
    winning_bid: float = 1.0
    assignment_age_fraction: float = 0.0


@dataclass
class TargetTeamRecord:
    target_id: int
    state: TargetTrackState = TargetTrackState.UNSEEN
    ever_acquired: bool = False
    ever_confirmed: bool = False
    consecutive_unobserved: int = 0
    lost_cycle: int = 0
    reacquisition_reported_cycle: int = 0

    def clone(self) -> "TargetTeamRecord":
        return TargetTeamRecord(
            target_id=self.target_id,
            state=self.state,
            ever_acquired=self.ever_acquired,
            ever_confirmed=self.ever_confirmed,
            consecutive_unobserved=self.consecutive_unobserved,
            lost_cycle=self.lost_cycle,
            reacquisition_reported_cycle=self.reacquisition_reported_cycle,
        )


@dataclass(frozen=True)
class TargetLifecycleEvents:
    first_acquired: FrozenSet[int] = frozenset()
    first_confirmed: FrozenSet[int] = frozenset()
    lost: FrozenSet[int] = frozenset()
    reacquired: FrozenSet[int] = frozenset()
    currently_tracked: FrozenSet[int] = frozenset()


@dataclass
class LocalKnowledgeSnapshot:
    """Legal per-UAV knowledge before Phase 2 graph encoding."""

    ego: UAVState
    direct_neighbors: List[UAVState]
    known_obstacles: List[ObstacleState]
    active_obstacles: List[ObstacleState]
    target_beliefs: Dict[int, object]
    target_metadata: Dict[int, TargetObservationMetadata]
    coverage_knowledge: Array
    component_members: Tuple[int, ...]
    observed_obstacle_ids_this_step: FrozenSet[int]
    target_assignments: Dict[int, TargetAssignmentMetadata] = field(
        default_factory=dict
    )


'''),
    ('env.geometry_utils', r'''
"""Continuous 2-D geometry for swept sensing and collision handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .state_types import (
    CollisionEvent,
    CollisionKind,
    EntityKind,
    MotionProposal,
    ObstacleState,
    SweepPath,
)


EPS = 1e-12


def wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def angle_difference(target: float, source: float) -> float:
    return wrap_angle(float(target) - float(source))


def heading_vector(heading: float) -> np.ndarray:
    return np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)


def heading_from_vector(vector: np.ndarray, fallback: float = 0.0) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    if float(np.dot(vector, vector)) <= EPS:
        return wrap_angle(fallback)
    return wrap_angle(float(np.arctan2(vector[1], vector[0])))


def closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> Tuple[np.ndarray, float]:
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= EPS:
        return start.copy(), 0.0
    tau = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    return start + tau * delta, tau


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> Tuple[float, float]:
    closest, tau = closest_point_on_segment(point, start, end)
    return float(np.linalg.norm(np.asarray(point, dtype=np.float64) - closest)), tau


def relative_closest_approach(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> Tuple[float, float]:
    r0 = np.asarray(start_a, dtype=np.float64) - np.asarray(start_b, dtype=np.float64)
    relative_delta = (
        np.asarray(end_a, dtype=np.float64)
        - np.asarray(start_a, dtype=np.float64)
        - np.asarray(end_b, dtype=np.float64)
        + np.asarray(start_b, dtype=np.float64)
    )
    denominator = float(np.dot(relative_delta, relative_delta))
    if denominator <= EPS:
        tau = 0.0
    else:
        tau = float(np.clip(-np.dot(r0, relative_delta) / denominator, 0.0, 1.0))
    distance = float(np.linalg.norm(r0 + tau * relative_delta))
    return tau, distance


def _quadratic_interval(r0: np.ndarray, velocity: np.ndarray, radius: float) -> Optional[Tuple[float, float]]:
    """Return the interval in [0, 1] where |r0 + t velocity| <= radius."""

    r0 = np.asarray(r0, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    a = float(np.dot(velocity, velocity))
    c = float(np.dot(r0, r0) - radius * radius)
    if a <= EPS:
        return (0.0, 1.0) if c <= 0.0 else None
    b = 2.0 * float(np.dot(r0, velocity))
    discriminant = b * b - 4.0 * a * c
    if discriminant < -EPS:
        return None
    root = float(np.sqrt(max(discriminant, 0.0)))
    first = (-b - root) / (2.0 * a)
    second = (-b + root) / (2.0 * a)
    low = max(0.0, min(first, second))
    high = min(1.0, max(first, second))
    if low > high + EPS:
        return None
    return float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0))


def interval_within_radius(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
    radius: float,
) -> Optional[Tuple[float, float]]:
    r0 = np.asarray(start_a, dtype=np.float64) - np.asarray(start_b, dtype=np.float64)
    velocity = (
        np.asarray(end_a, dtype=np.float64)
        - np.asarray(start_a, dtype=np.float64)
        - np.asarray(end_b, dtype=np.float64)
        + np.asarray(start_b, dtype=np.float64)
    )
    return _quadratic_interval(r0, velocity, float(radius))


def swept_paths_interval_within_radius(
    path_a: SweepPath,
    path_b: SweepPath,
    radius: float,
) -> Optional[Tuple[float, float]]:
    """Solve proximity for paths that stop at their first collision fractions."""

    breaks = sorted(
        set(
            [
                0.0,
                float(np.clip(path_a.stop_fraction, 0.0, 1.0)),
                float(np.clip(path_b.stop_fraction, 0.0, 1.0)),
                1.0,
            ]
        )
    )
    intervals: List[Tuple[float, float]] = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        if right - left <= EPS:
            continue
        pa0 = path_a.position_at(left)
        pb0 = path_b.position_at(left)
        pa1 = path_a.position_at(right)
        pb1 = path_b.position_at(right)
        local = interval_within_radius(pa0, pa1, pb0, pb1, radius)
        if local is None:
            continue
        scale = right - left
        intervals.append((left + local[0] * scale, left + local[1] * scale))
    if not intervals:
        # Include the degenerate all-stationary case and exact break contacts.
        for tau in breaks:
            if np.linalg.norm(path_a.position_at(tau) - path_b.position_at(tau)) <= radius + EPS:
                intervals.append((tau, tau))
    if not intervals:
        return None
    return min(item[0] for item in intervals), max(item[1] for item in intervals)


def swept_paths_min_distance(path_a: SweepPath, path_b: SweepPath) -> Tuple[float, float]:
    breaks = sorted(
        set([0.0, float(path_a.stop_fraction), float(path_b.stop_fraction), 1.0])
    )
    best_tau = 0.0
    best_distance = float("inf")
    for left, right in zip(breaks[:-1], breaks[1:]):
        if right - left <= EPS:
            continue
        pa0, pa1 = path_a.position_at(left), path_a.position_at(right)
        pb0, pb1 = path_b.position_at(left), path_b.position_at(right)
        local_tau, distance = relative_closest_approach(pa0, pa1, pb0, pb1)
        global_tau = left + local_tau * (right - left)
        if distance < best_distance:
            best_tau, best_distance = global_tau, distance
    for tau in breaks:
        distance = float(np.linalg.norm(path_a.position_at(tau) - path_b.position_at(tau)))
        if distance < best_distance:
            best_tau, best_distance = tau, distance
    return float(best_tau), float(best_distance)


def segment_circle_first_contact(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    combined_radius: float,
) -> Optional[float]:
    r0 = np.asarray(start, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    velocity = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    if float(np.dot(r0, r0)) <= combined_radius * combined_radius + EPS:
        return 0.0
    interval = _quadratic_interval(r0, velocity, float(combined_radius))
    return None if interval is None else float(interval[0])


def moving_circles_first_contact(
    start_a: np.ndarray,
    end_a: np.ndarray,
    radius_a: float,
    start_b: np.ndarray,
    end_b: np.ndarray,
    radius_b: float,
) -> Optional[float]:
    interval = interval_within_radius(start_a, end_a, start_b, end_b, radius_a + radius_b)
    return None if interval is None else float(interval[0])


def boundary_first_contact(
    start: np.ndarray,
    end: np.ndarray,
    entity_radius: float,
    map_size: float,
) -> Optional[Tuple[float, str, np.ndarray]]:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    lower = float(entity_radius)
    upper = float(map_size - entity_radius)
    candidates: List[Tuple[float, str, np.ndarray]] = []

    if start[0] < lower - EPS:
        candidates.append((0.0, "left", np.array([1.0, 0.0])))
    elif delta[0] < -EPS and end[0] < lower - EPS:
        candidates.append(((lower - start[0]) / delta[0], "left", np.array([1.0, 0.0])))
    if start[0] > upper + EPS:
        candidates.append((0.0, "right", np.array([-1.0, 0.0])))
    elif delta[0] > EPS and end[0] > upper + EPS:
        candidates.append(((upper - start[0]) / delta[0], "right", np.array([-1.0, 0.0])))
    if start[1] < lower - EPS:
        candidates.append((0.0, "bottom", np.array([0.0, 1.0])))
    elif delta[1] < -EPS and end[1] < lower - EPS:
        candidates.append(((lower - start[1]) / delta[1], "bottom", np.array([0.0, 1.0])))
    if start[1] > upper + EPS:
        candidates.append((0.0, "top", np.array([0.0, -1.0])))
    elif delta[1] > EPS and end[1] > upper + EPS:
        candidates.append(((upper - start[1]) / delta[1], "top", np.array([0.0, -1.0])))

    valid = [item for item in candidates if -EPS <= item[0] <= 1.0 + EPS]
    if not valid:
        return None
    tau, name, normal = min(valid, key=lambda item: (item[0], item[1]))
    return float(np.clip(tau, 0.0, 1.0)), name, normal


def surface_distance_to_obstacle(position: np.ndarray, obstacle: ObstacleState) -> float:
    return float(np.linalg.norm(np.asarray(position) - obstacle.center) - obstacle.radius)


def segment_surface_distance(start: np.ndarray, end: np.ndarray, obstacle: ObstacleState) -> float:
    center_distance, _ = point_segment_distance(obstacle.center, start, end)
    return float(center_distance - obstacle.radius)


def reflect_vector(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= EPS:
        return -vector
    unit = normal / norm
    return vector - 2.0 * float(np.dot(vector, unit)) * unit


def _safe_normal(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        vector = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vector / norm


def enumerate_collision_events(
    proposals: Sequence[MotionProposal],
    obstacles: Sequence[ObstacleState],
    entity_radii: Dict[EntityKind, float],
    map_size: float,
) -> List[CollisionEvent]:
    events: List[CollisionEvent] = []
    by_kind: Dict[EntityKind, List[MotionProposal]] = {
        EntityKind.UAV: [],
        EntityKind.TARGET: [],
    }
    for proposal in proposals:
        by_kind[proposal.kind].append(proposal)
        radius = entity_radii[proposal.kind]
        boundary = boundary_first_contact(proposal.start, proposal.end, radius, map_size)
        if boundary is not None:
            tau, name, normal = boundary
            kind = (
                CollisionKind.UAV_BOUNDARY
                if proposal.kind is EntityKind.UAV
                else CollisionKind.TARGET_BOUNDARY
            )
            events.append(
                CollisionEvent(
                    kind,
                    tau,
                    proposal.kind,
                    proposal.entity_id,
                    EntityKind.BOUNDARY,
                    None,
                    normal,
                    boundary_name=name,
                )
            )
        for obstacle in obstacles:
            tau = segment_circle_first_contact(
                proposal.start,
                proposal.end,
                obstacle.center,
                radius + obstacle.radius,
            )
            if tau is None:
                continue
            contact = proposal.start + tau * proposal.displacement
            normal = _safe_normal(contact - obstacle.center, -proposal.displacement)
            kind = (
                CollisionKind.UAV_OBSTACLE
                if proposal.kind is EntityKind.UAV
                else CollisionKind.TARGET_OBSTACLE
            )
            events.append(
                CollisionEvent(
                    kind,
                    tau,
                    proposal.kind,
                    proposal.entity_id,
                    EntityKind.OBSTACLE,
                    obstacle.obstacle_id,
                    normal,
                    obstacle_id=obstacle.obstacle_id,
                )
            )

    uavs = sorted(by_kind[EntityKind.UAV], key=lambda p: p.entity_id)
    targets = sorted(by_kind[EntityKind.TARGET], key=lambda p: p.entity_id)
    for index, first in enumerate(uavs):
        for second in uavs[index + 1 :]:
            tau = moving_circles_first_contact(
                first.start,
                first.end,
                entity_radii[EntityKind.UAV],
                second.start,
                second.end,
                entity_radii[EntityKind.UAV],
            )
            if tau is None:
                continue
            pa = first.start + tau * first.displacement
            pb = second.start + tau * second.displacement
            normal = _safe_normal(pa - pb, first.displacement - second.displacement)
            events.append(
                CollisionEvent(
                    CollisionKind.UAV_UAV,
                    tau,
                    EntityKind.UAV,
                    first.entity_id,
                    EntityKind.UAV,
                    second.entity_id,
                    normal,
                )
            )
    for uav in uavs:
        for target in targets:
            tau = moving_circles_first_contact(
                uav.start,
                uav.end,
                entity_radii[EntityKind.UAV],
                target.start,
                target.end,
                entity_radii[EntityKind.TARGET],
            )
            if tau is None:
                continue
            pa = uav.start + tau * uav.displacement
            pb = target.start + tau * target.displacement
            normal = _safe_normal(pa - pb, uav.displacement - target.displacement)
            events.append(
                CollisionEvent(
                    CollisionKind.UAV_TARGET,
                    tau,
                    EntityKind.UAV,
                    uav.entity_id,
                    EntityKind.TARGET,
                    target.entity_id,
                    normal,
                )
            )
    return events


def resolve_first_collisions(
    proposals: Sequence[MotionProposal],
    obstacles: Sequence[ObstacleState],
    entity_radii: Dict[EntityKind, float],
    map_size: float,
) -> Tuple[Dict[Tuple[EntityKind, int], np.ndarray], Dict[Tuple[EntityKind, int], float], Dict[Tuple[EntityKind, int], SweepPath], List[CollisionEvent], List[CollisionEvent]]:
    """Resolve each entity's earliest deterministic event with rollback+reflect."""

    proposal_map = {(p.kind, p.entity_id): p for p in proposals}
    all_events = enumerate_collision_events(proposals, obstacles, entity_radii, map_size)
    kind_order = {kind: index for index, kind in enumerate(CollisionKind)}
    all_events.sort(
        key=lambda e: (
            round(float(e.tau), 12),
            kind_order[e.kind],
            e.entity_a_kind.value,
            e.entity_a_id,
            e.entity_b_kind.value,
            -1 if e.entity_b_id is None else e.entity_b_id,
        )
    )
    resolved: Set[Tuple[EntityKind, int]] = set()
    accepted: List[CollisionEvent] = []
    for event in all_events:
        keys = event.entity_keys
        if any(key in resolved for key in keys):
            continue
        accepted.append(event)
        resolved.update(keys)

    accepted_by_entity: Dict[Tuple[EntityKind, int], CollisionEvent] = {}
    for event in accepted:
        for key in event.entity_keys:
            accepted_by_entity[key] = event

    positions: Dict[Tuple[EntityKind, int], np.ndarray] = {}
    headings: Dict[Tuple[EntityKind, int], float] = {}
    sweep_paths: Dict[Tuple[EntityKind, int], SweepPath] = {}
    for key, proposal in proposal_map.items():
        event = accepted_by_entity.get(key)
        if event is None:
            positions[key] = np.asarray(proposal.end, dtype=np.float64).copy()
            headings[key] = wrap_angle(proposal.proposed_heading)
            sweep_paths[key] = SweepPath(proposal.start.copy(), proposal.end.copy(), 1.0)
            continue
        positions[key] = np.asarray(proposal.start, dtype=np.float64).copy()
        normal = np.asarray(event.normal_a, dtype=np.float64)
        if key != (event.entity_a_kind, event.entity_a_id):
            normal = -normal
        reflected = reflect_vector(proposal.displacement, normal)
        headings[key] = heading_from_vector(reflected, proposal.proposed_heading + np.pi)
        sweep_paths[key] = SweepPath(
            proposal.start.copy(),
            proposal.end.copy(),
            float(np.clip(event.tau, 0.0, 1.0)),
        )

    # Rollback is deliberately non-local in time: an entity that collided at
    # tau>0 is returned to its step-start position.  That start position may
    # have been vacated in the original joint proposal, allowing another
    # entity to finish there without an event in ``all_events``.  The old
    # resolver consequently produced overlapping end states; the next step
    # then detected tau=0 forever and generated long collision streaks.
    #
    # Close those rollback-induced contacts deterministically.  An entity
    # whose earlier event was already resolved keeps that first response.  Any
    # still-moving participant is rolled back and reflected against the now
    # stationary participant.  At most one new dynamic entity is resolved per
    # pass, so the loop is bounded by the number of proposals.
    uav_keys = sorted(
        (key for key in proposal_map if key[0] is EntityKind.UAV),
        key=lambda key: key[1],
    )
    target_keys = sorted(
        (key for key in proposal_map if key[0] is EntityKind.TARGET),
        key=lambda key: key[1],
    )
    dynamic_pairs = [
        (first, second)
        for index, first in enumerate(uav_keys)
        for second in uav_keys[index + 1 :]
    ] + [(uav, target) for uav in uav_keys for target in target_keys]

    for _pass in range(len(proposal_map)):
        cascade = None
        for first_key, second_key in dynamic_pairs:
            combined_radius = (
                entity_radii[first_key[0]] + entity_radii[second_key[0]]
            )
            final_distance = float(
                np.linalg.norm(positions[first_key] - positions[second_key])
            )
            if final_distance > combined_radius + 1e-9:
                continue
            unresolved = [
                key for key in (first_key, second_key) if key not in resolved
            ]
            if not unresolved:
                raise AssertionError(
                    "collision rollback produced overlapping resolved starts"
                )

            first = proposal_map[first_key]
            second = proposal_map[second_key]
            first_end = first.start if first_key in resolved else first.end
            second_end = second.start if second_key in resolved else second.end
            tau = moving_circles_first_contact(
                first.start,
                first_end,
                entity_radii[first_key[0]],
                second.start,
                second_end,
                entity_radii[second_key[0]],
            )
            if tau is None:
                # Endpoint overlap guarantees contact; this fallback only
                # guards floating-point degeneracy in the quadratic solver.
                tau = 1.0
            first_displacement = first_end - first.start
            second_displacement = second_end - second.start
            first_contact = first.start + tau * first_displacement
            second_contact = second.start + tau * second_displacement
            normal = _safe_normal(
                first_contact - second_contact,
                first_displacement - second_displacement,
            )
            kind = (
                CollisionKind.UAV_UAV
                if second_key[0] is EntityKind.UAV
                else CollisionKind.UAV_TARGET
            )
            cascade = (
                CollisionEvent(
                    kind,
                    float(tau),
                    first_key[0],
                    first_key[1],
                    second_key[0],
                    second_key[1],
                    normal,
                ),
                unresolved,
            )
            break
        if cascade is None:
            break

        event, unresolved = cascade
        accepted.append(event)
        for key in unresolved:
            proposal = proposal_map[key]
            resolved.add(key)
            positions[key] = np.asarray(proposal.start, dtype=np.float64).copy()
            normal = np.asarray(event.normal_a, dtype=np.float64)
            if key != (event.entity_a_kind, event.entity_a_id):
                normal = -normal
            reflected = reflect_vector(proposal.displacement, normal)
            headings[key] = heading_from_vector(
                reflected, proposal.proposed_heading + np.pi
            )
            sweep_paths[key] = SweepPath(
                proposal.start.copy(),
                proposal.end.copy(),
                float(np.clip(event.tau, 0.0, 1.0)),
            )
    else:
        raise AssertionError("collision rollback closure did not converge")

    return positions, headings, sweep_paths, accepted, all_events
'''),
    ('graph_system.communication_manager', r'''
"""Direct communication graph, components and relay diagnostics."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np


class CommunicationManager:
    def __init__(self, n_uavs: int, communication_radius: float):
        if n_uavs <= 0 or communication_radius <= 0:
            raise ValueError("n_uavs and communication_radius must be positive")
        self.n_uavs = int(n_uavs)
        self.communication_radius = float(communication_radius)

    def build_adjacency(self, positions: Sequence[np.ndarray]) -> np.ndarray:
        array = np.asarray(positions, dtype=np.float64)
        if array.shape != (self.n_uavs, 2):
            raise ValueError(
                "positions must have shape ({}, 2), got {}".format(self.n_uavs, array.shape)
            )
        delta = array[:, None, :] - array[None, :, :]
        distances_squared = np.sum(delta * delta, axis=-1)
        adjacency = distances_squared <= self.communication_radius ** 2 + 1e-12
        np.fill_diagonal(adjacency, False)
        return adjacency.astype(bool, copy=False)

    def connected_components(self, adjacency: np.ndarray) -> List[Tuple[int, ...]]:
        adjacency = np.asarray(adjacency, dtype=bool)
        if adjacency.shape != (self.n_uavs, self.n_uavs):
            raise ValueError("invalid adjacency shape")
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError("communication graph must be symmetric")
        visited = np.zeros(self.n_uavs, dtype=bool)
        components: List[Tuple[int, ...]] = []
        for root in range(self.n_uavs):
            if visited[root]:
                continue
            stack = [root]
            visited[root] = True
            members: List[int] = []
            while stack:
                node = stack.pop()
                members.append(node)
                neighbors = np.flatnonzero(adjacency[node])
                for neighbor in reversed(neighbors.tolist()):
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(int(neighbor))
            components.append(tuple(sorted(members)))
        return components

    def articulation_points(self, adjacency: np.ndarray) -> Set[int]:
        """Tarjan DFS articulation points; used only for contribution logging."""

        adjacency = np.asarray(adjacency, dtype=bool)
        if adjacency.shape != (self.n_uavs, self.n_uavs):
            raise ValueError("invalid adjacency shape")
        discovery = [-1] * self.n_uavs
        low = [-1] * self.n_uavs
        parent = [-1] * self.n_uavs
        points: Set[int] = set()
        clock = 0

        def dfs(node: int) -> None:
            nonlocal clock
            discovery[node] = low[node] = clock
            clock += 1
            children = 0
            for neighbor in np.flatnonzero(adjacency[node]).tolist():
                neighbor = int(neighbor)
                if discovery[neighbor] == -1:
                    parent[neighbor] = node
                    children += 1
                    dfs(neighbor)
                    low[node] = min(low[node], low[neighbor])
                    if parent[node] == -1 and children > 1:
                        points.add(node)
                    if parent[node] != -1 and low[neighbor] >= discovery[node]:
                        points.add(node)
                elif neighbor != parent[node]:
                    low[node] = min(low[node], discovery[neighbor])

        for root in range(self.n_uavs):
            if discovery[root] == -1:
                dfs(root)
        return points

    @staticmethod
    def component_lookup(components: Sequence[Sequence[int]], n_uavs: int) -> np.ndarray:
        lookup = np.full(int(n_uavs), -1, dtype=np.int16)
        for component_index, component in enumerate(components):
            for member in component:
                if lookup[int(member)] != -1:
                    raise ValueError("UAV appears in more than one component")
                lookup[int(member)] = component_index
        if np.any(lookup < 0):
            raise ValueError("components do not cover all UAVs")
        return lookup

'''),
    ('graph_system.target_belief', r'''
"""Fractional-time CV Kalman beliefs, component synchronization and lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from env.state_types import (
    EnvConfig,
    MeasurementId,
    MeasurementKind,
    TargetLifecycleEvents,
    TargetMeasurement,
    TargetObservationMetadata,
    TargetTeamRecord,
    TargetTrackState,
)


def transition_matrix(delta: float) -> np.ndarray:
    delta = float(delta)
    return np.array(
        [
            [1.0, 0.0, delta, 0.0],
            [0.0, 1.0, 0.0, delta],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def process_covariance(delta: float, acceleration_sigma: float) -> np.ndarray:
    delta = float(delta)
    scale = float(acceleration_sigma) ** 2
    return scale * np.array(
        [
            [delta ** 4 / 4.0, 0.0, delta ** 3 / 2.0, 0.0],
            [0.0, delta ** 4 / 4.0, 0.0, delta ** 3 / 2.0],
            [delta ** 3 / 2.0, 0.0, delta ** 2, 0.0],
            [0.0, delta ** 3 / 2.0, 0.0, delta ** 2],
        ],
        dtype=np.float64,
    )


@dataclass
class TargetBelief:
    target_id: int
    mean: np.ndarray
    covariance: np.ndarray
    confidence: float
    age: int
    last_real_observation_time: float
    last_measurement_type: MeasurementKind
    source_uav_id: int
    filter_time: float
    measurement_ids: Set[MeasurementId] = field(default_factory=set)

    def clone(self) -> "TargetBelief":
        return TargetBelief(
            target_id=int(self.target_id),
            mean=np.asarray(self.mean, dtype=np.float64).copy(),
            covariance=np.asarray(self.covariance, dtype=np.float64).copy(),
            confidence=float(self.confidence),
            age=int(self.age),
            last_real_observation_time=float(self.last_real_observation_time),
            last_measurement_type=self.last_measurement_type,
            source_uav_id=int(self.source_uav_id),
            filter_time=float(self.filter_time),
            measurement_ids=set(self.measurement_ids),
        )

    @classmethod
    def from_measurement(cls, measurement: TargetMeasurement, config: EnvConfig) -> "TargetBelief":
        position_variance = (
            config.fine_position_sigma ** 2
            if measurement.kind is MeasurementKind.FINE
            else config.coarse_position_sigma ** 2
        )
        mean = np.array(
            [measurement.position[0], measurement.position[1], 0.0, 0.0],
            dtype=np.float64,
        )
        covariance = np.diag([position_variance, position_variance, 25.0, 25.0]).astype(
            np.float64
        )
        return cls(
            target_id=measurement.target_id,
            mean=mean,
            covariance=covariance,
            confidence=measurement.kind.confidence,
            age=0,
            last_real_observation_time=measurement.observation_time,
            last_measurement_type=measurement.kind,
            source_uav_id=measurement.source_uav_id,
            filter_time=measurement.observation_time,
            measurement_ids={measurement.measurement_id},
        )

    def predict_to(self, target_time: float, acceleration_sigma: float) -> None:
        target_time = float(target_time)
        delta = target_time - self.filter_time
        if delta < -1e-10:
            raise ValueError("out-of-sequence prediction is not supported")
        if delta <= 1e-12:
            self.filter_time = target_time
            return
        transition = transition_matrix(delta)
        self.mean = transition @ self.mean
        self.covariance = (
            transition @ self.covariance @ transition.T
            + process_covariance(delta, acceleration_sigma)
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.filter_time = target_time

    def apply_measurement(self, measurement: TargetMeasurement, config: EnvConfig) -> bool:
        if measurement.target_id != self.target_id:
            raise ValueError("measurement target ID does not match belief")
        if measurement.measurement_id in self.measurement_ids:
            return False
        self.predict_to(measurement.observation_time, config.process_accel_sigma)
        observation = np.asarray(measurement.position, dtype=np.float64)
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        innovation = observation - h @ self.mean
        innovation_covariance = h @ self.covariance @ h.T + measurement.covariance
        gain = self.covariance @ h.T @ np.linalg.inv(innovation_covariance)
        identity = np.eye(4, dtype=np.float64)
        self.mean = self.mean + gain @ innovation
        # Joseph form preserves positive semidefiniteness under roundoff.
        residual = identity - gain @ h
        self.covariance = (
            residual @ self.covariance @ residual.T
            + gain @ measurement.covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.confidence = measurement.kind.confidence
        self.age = 0
        self.last_real_observation_time = measurement.observation_time
        self.last_measurement_type = measurement.kind
        self.source_uav_id = measurement.source_uav_id
        self.measurement_ids.add(measurement.measurement_id)
        return True


def belief_priority_key(belief: TargetBelief) -> Tuple[float, int, float, int]:
    return (
        -float(belief.last_real_observation_time),
        -int(belief.last_measurement_type),
        float(np.trace(belief.covariance)),
        int(belief.source_uav_id),
    )


def select_best_belief(candidates: Sequence[TargetBelief]) -> TargetBelief:
    if not candidates:
        raise ValueError("at least one belief candidate is required")
    return min(candidates, key=belief_priority_key).clone()


class TargetBeliefManager:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.beliefs: List[Dict[int, TargetBelief]] = [
            dict() for _ in range(config.n_uavs)
        ]
        self.metadata: List[Dict[int, TargetObservationMetadata]] = [
            dict() for _ in range(config.n_uavs)
        ]

    def reset(self) -> None:
        self.beliefs = [dict() for _ in range(self.config.n_uavs)]
        self.metadata = [dict() for _ in range(self.config.n_uavs)]

    def advance_and_synchronize(
        self,
        measurements: Sequence[TargetMeasurement],
        components: Sequence[Sequence[int]],
        end_time: float,
    ) -> None:
        unique_measurements: Dict[MeasurementId, TargetMeasurement] = {}
        for measurement in measurements:
            current = unique_measurements.get(measurement.measurement_id)
            if current is None or (
                measurement.observation_time,
                int(measurement.kind),
            ) > (
                current.observation_time,
                int(current.kind),
            ):
                unique_measurements[measurement.measurement_id] = measurement
        direct_by_pair: Dict[Tuple[int, int], TargetMeasurement] = {}
        for measurement in unique_measurements.values():
            key = (measurement.source_uav_id, measurement.target_id)
            existing = direct_by_pair.get(key)
            if existing is None or (
                measurement.observation_time,
                int(measurement.kind),
            ) > (
                existing.observation_time,
                int(existing.kind),
            ):
                direct_by_pair[key] = measurement

        candidates: List[Dict[int, TargetBelief]] = [dict() for _ in range(self.config.n_uavs)]
        all_target_ids = set()
        for local in self.beliefs:
            all_target_ids.update(local)
        all_target_ids.update(measurement.target_id for measurement in direct_by_pair.values())

        for uav_id in range(self.config.n_uavs):
            for target_id in sorted(all_target_ids):
                old = self.beliefs[uav_id].get(target_id)
                measurement = direct_by_pair.get((uav_id, target_id))
                if measurement is not None:
                    if old is None:
                        belief = TargetBelief.from_measurement(measurement, self.config)
                    else:
                        belief = old.clone()
                        belief.apply_measurement(measurement, self.config)
                    belief.predict_to(end_time, self.config.process_accel_sigma)
                    candidates[uav_id][target_id] = belief
                elif old is not None:
                    belief = old.clone()
                    belief.predict_to(end_time, self.config.process_accel_sigma)
                    candidates[uav_id][target_id] = belief

        next_beliefs: List[Dict[int, TargetBelief]] = [
            dict() for _ in range(self.config.n_uavs)
        ]
        next_metadata: List[Dict[int, TargetObservationMetadata]] = [
            dict() for _ in range(self.config.n_uavs)
        ]
        for component in components:
            members = tuple(sorted(int(item) for item in component))
            propagation_groups = tuple(
                ((uav_id,), (uav_id,)) for uav_id in members
            )

            for source_members, recipient_members in propagation_groups:
                component_targets = set()
                for uav_id in source_members:
                    component_targets.update(candidates[uav_id])
                for target_id in sorted(component_targets):
                    direct_members = [
                        uav_id
                        for uav_id in source_members
                        if (uav_id, target_id) in direct_by_pair
                    ]
                    if direct_members:
                        pool = [
                            candidates[uav_id][target_id]
                            for uav_id in direct_members
                        ]
                    else:
                        pool = [
                            candidates[uav_id][target_id]
                            for uav_id in source_members
                            if target_id in candidates[uav_id]
                        ]
                    if not pool:
                        continue
                    selected = select_best_belief(pool)
                    if direct_members:
                        best_level = max(
                            direct_by_pair[(uav_id, target_id)].kind.confidence
                            for uav_id in direct_members
                        )
                        selected.age = 0
                        # Preserve the real component denominator so disabling
                        # posterior copying does not rescale this feature.
                        observer_ratio = len(direct_members) / max(len(members), 1)
                    else:
                        best_level = 0.0
                        observer_ratio = 0.0
                        selected.age += 1
                        selected.confidence *= self.config.belief_decay
                    if selected.age >= self.config.belief_lost_age:
                        continue
                    for uav_id in recipient_members:
                        next_beliefs[uav_id][target_id] = selected.clone()
                        own_measurement = direct_by_pair.get((uav_id, target_id))
                        self_level = (
                            own_measurement.kind.confidence
                            if own_measurement is not None
                            else 0.0
                        )
                        next_metadata[uav_id][target_id] = (
                            TargetObservationMetadata(
                                self_level=float(self_level),
                                component_best_level=float(best_level),
                                component_observer_ratio=float(observer_ratio),
                            )
                        )
        self.beliefs = next_beliefs
        self.metadata = next_metadata

class TargetLifecycleManager:
    def __init__(self, target_ids: Sequence[int], lost_after_steps: int = 4):
        self.lost_after_steps = int(lost_after_steps)
        self.records: Dict[int, TargetTeamRecord] = {
            int(target_id): TargetTeamRecord(int(target_id)) for target_id in target_ids
        }

    def update(self, measurements: Sequence[TargetMeasurement]) -> TargetLifecycleEvents:
        level_by_target: Dict[int, MeasurementKind] = {}
        for measurement in measurements:
            previous = level_by_target.get(measurement.target_id, MeasurementKind.NONE)
            if measurement.kind > previous:
                level_by_target[measurement.target_id] = measurement.kind

        first_acquired: Set[int] = set()
        first_confirmed: Set[int] = set()
        lost: Set[int] = set()
        reacquired: Set[int] = set()
        tracked: Set[int] = set()
        for target_id, record in self.records.items():
            level = level_by_target.get(target_id, MeasurementKind.NONE)
            if level is not MeasurementKind.NONE:
                was_lost = record.state is TargetTrackState.LOST
                if not record.ever_acquired:
                    record.ever_acquired = True
                    first_acquired.add(target_id)
                if level is MeasurementKind.FINE and not record.ever_confirmed:
                    record.ever_confirmed = True
                    first_confirmed.add(target_id)
                record.consecutive_unobserved = 0
                if was_lost:
                    if record.reacquisition_reported_cycle < record.lost_cycle:
                        reacquired.add(target_id)
                        record.reacquisition_reported_cycle = record.lost_cycle
                    record.state = TargetTrackState.REACQUIRED
                elif level is MeasurementKind.FINE:
                    record.state = TargetTrackState.TRACKED
                else:
                    record.state = TargetTrackState.ACQUIRED
                if level is MeasurementKind.FINE:
                    tracked.add(target_id)
                continue

            if not record.ever_acquired:
                record.state = TargetTrackState.UNSEEN
                record.consecutive_unobserved = 0
                continue
            record.consecutive_unobserved += 1
            if record.consecutive_unobserved >= self.lost_after_steps:
                if record.state is not TargetTrackState.LOST:
                    record.state = TargetTrackState.LOST
                    record.lost_cycle += 1
                    lost.add(target_id)
            # Before four missed steps, keep the previous task state while the
            # component belief is still alive.

        return TargetLifecycleEvents(
            first_acquired=frozenset(first_acquired),
            first_confirmed=frozenset(first_confirmed),
            lost=frozenset(lost),
            reacquired=frozenset(reacquired),
            currently_tracked=frozenset(tracked),
        )
'''),
    ('graph_system.graph_types', r'''
"""Fixed-size local observation snapshots for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


ACTOR_SPECS = {
    "self_features": ((4, 17), np.float32),
    "neighbor_features": ((4, 3, 6), np.float32),
    "neighbor_mask": ((4, 3), np.bool_),
    "obstacle_features": ((4, 8, 5), np.float32),
    "obstacle_mask": ((4, 8), np.bool_),
    "target_features": ((4, 4, 17), np.float32),
    "target_mask": ((4, 4), np.bool_),
}


def _validate_fields(instance: object, specs: Mapping[str, Tuple[Tuple[int, ...], object]]) -> None:
    for name, (shape, dtype) in specs.items():
        value = getattr(instance, name)
        if not isinstance(value, np.ndarray):
            raise TypeError("{} must be a NumPy array".format(name))
        if value.shape != shape:
            raise ValueError("{} has shape {}, expected {}".format(name, value.shape, shape))
        if value.dtype != np.dtype(dtype):
            raise TypeError("{} has dtype {}, expected {}".format(name, value.dtype, np.dtype(dtype)))
        if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
            raise ValueError("{} contains NaN or Inf".format(name))


@dataclass
class ActorGraphSnapshot:
    self_features: np.ndarray
    neighbor_features: np.ndarray
    neighbor_mask: np.ndarray
    obstacle_features: np.ndarray
    obstacle_mask: np.ndarray
    target_features: np.ndarray
    target_mask: np.ndarray

    def validate(self) -> None:
        _validate_fields(self, ACTOR_SPECS)

    def as_dict(self, copy: bool = False) -> Dict[str, np.ndarray]:
        return {
            name: getattr(self, name).copy() if copy else getattr(self, name)
            for name in ACTOR_SPECS
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, np.ndarray], copy: bool = False) -> "ActorGraphSnapshot":
        instance = cls(
            **{
                name: np.asarray(values[name]).copy() if copy else np.asarray(values[name])
                for name in ACTOR_SPECS
            }
        )
        instance.validate()
        return instance

    def clone(self) -> "ActorGraphSnapshot":
        return ActorGraphSnapshot.from_dict(self.as_dict(), copy=True)


'''),
    ('graph_system.coverage_knowledge', r'''
"""Physical and legally shared coverage maps with swept-capsule updates."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from env.geometry_utils import point_segment_distance
from env.state_types import EnvConfig, ObstacleState, SweepPath


class CoverageKnowledge:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.grid_size = config.coverage_grid_size
        self.cell_size = float(config.coverage_cell_size)
        centers = (np.arange(self.grid_size, dtype=np.float64) + 0.5) * self.cell_size
        self.grid_x, self.grid_y = np.meshgrid(centers, centers)
        self.physical = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.local = np.zeros(
            (config.n_uavs, self.grid_size, self.grid_size), dtype=bool
        )
        self.free_mask = np.ones_like(self.physical)

    def reset(self, obstacles: Sequence[ObstacleState]) -> None:
        self.physical.fill(False)
        self.local.fill(False)
        self.free_mask.fill(True)
        for obstacle in obstacles:
            inside = (
                (self.grid_x - obstacle.center[0]) ** 2
                + (self.grid_y - obstacle.center[1]) ** 2
                <= obstacle.radius ** 2
            )
            self.free_mask[inside] = False

    def capsule_mask(self, start: np.ndarray, end: np.ndarray, radius: float) -> np.ndarray:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        radius = float(radius)
        minimum = np.minimum(start, end) - radius
        maximum = np.maximum(start, end) + radius
        col_start = max(0, int(np.floor(minimum[0] / self.cell_size)))
        col_end = min(self.grid_size, int(np.ceil(maximum[0] / self.cell_size)))
        row_start = max(0, int(np.floor(minimum[1] / self.cell_size)))
        row_end = min(self.grid_size, int(np.ceil(maximum[1] / self.cell_size)))
        mask = np.zeros_like(self.physical)
        if col_start >= col_end or row_start >= row_end:
            return mask
        x = self.grid_x[row_start:row_end, col_start:col_end]
        y = self.grid_y[row_start:row_end, col_start:col_end]
        delta = end - start
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-12:
            distance_squared = (x - start[0]) ** 2 + (y - start[1]) ** 2
        else:
            tau = np.clip(
                ((x - start[0]) * delta[0] + (y - start[1]) * delta[1])
                / denominator,
                0.0,
                1.0,
            )
            closest_x = start[0] + tau * delta[0]
            closest_y = start[1] + tau * delta[1]
            distance_squared = (x - closest_x) ** 2 + (y - closest_y) ** 2
        mask[row_start:row_end, col_start:col_end] = distance_squared <= radius ** 2 + 1e-12
        return mask

    def update_sweeps(self, paths: Dict[int, SweepPath]) -> Tuple[int, np.ndarray, List[np.ndarray]]:
        masks: List[np.ndarray] = []
        for uav_id in range(self.config.n_uavs):
            path = paths[uav_id]
            mask = self.capsule_mask(path.start, path.sweep_end, self.config.coverage_radius)
            masks.append(mask)
            self.local[uav_id] |= mask
        stacked = np.stack(masks, axis=0)
        team_sweep = np.any(stacked, axis=0) & self.free_mask
        newly_covered = team_sweep & ~self.physical
        self.physical |= team_sweep
        cover_counts = np.sum(stacked & newly_covered[None, :, :], axis=0)
        contributions = np.zeros(self.config.n_uavs, dtype=np.float64)
        for uav_id in range(self.config.n_uavs):
            shared_cells = stacked[uav_id] & newly_covered
            if np.any(shared_cells):
                contributions[uav_id] = float(
                    np.sum(1.0 / cover_counts[shared_cells].astype(np.float64))
                )
        return int(np.count_nonzero(newly_covered)), contributions, masks

    def synchronize(self, components: Sequence[Sequence[int]]) -> None:
        for component in components:
            members = np.asarray(tuple(int(item) for item in component), dtype=np.int64)
            merged = np.any(self.local[members], axis=0)
            self.local[members] = merged

    def team_knowledge(self) -> np.ndarray:
        return np.any(self.local, axis=0)

    @property
    def physical_ratio(self) -> float:
        denominator = int(np.count_nonzero(self.free_mask))
        if denominator == 0:
            return 0.0
        return float(np.count_nonzero(self.physical & self.free_mask) / denominator)

    def directional_unsearched_ratios(
        self,
        uav_id: int,
        position: np.ndarray,
        heading: float,
        known_obstacles: Sequence[ObstacleState],
    ) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        dx = self.grid_x - position[0]
        dy = self.grid_y - position[1]
        radius_mask = dx * dx + dy * dy <= self.config.coverage_context_radius ** 2
        eligible = radius_mask.copy()
        for obstacle in known_obstacles:
            inside = (
                (self.grid_x - obstacle.center[0]) ** 2
                + (self.grid_y - obstacle.center[1]) ** 2
                <= obstacle.radius ** 2
            )
            eligible &= ~inside
        relative_angle = (np.arctan2(dy, dx) - float(heading)) % (2.0 * np.pi)
        sector = np.floor((relative_angle + np.pi / 8.0) / (np.pi / 4.0)).astype(np.int16) % 8
        ratios = np.zeros(8, dtype=np.float64)
        local = self.local[int(uav_id)]
        for index in range(8):
            candidate = eligible & (sector == index)
            denominator = int(np.count_nonzero(candidate))
            if denominator:
                ratios[index] = float(np.count_nonzero(candidate & ~local) / denominator)
        return ratios

'''),
    ('graph_system.obstacle_knowledge', r'''
"""Permanent per-UAV static-obstacle knowledge and component synchronization."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

from env.geometry_utils import surface_distance_to_obstacle
from env.state_types import ObstacleState


class ObstacleKnowledge:
    def __init__(self, n_uavs: int, activation_surface_distance: float = 30.0):
        if n_uavs <= 0:
            raise ValueError("n_uavs must be positive")
        if activation_surface_distance <= 0:
            raise ValueError("activation distance must be positive")
        self.n_uavs = int(n_uavs)
        self.activation_surface_distance = float(activation_surface_distance)
        self.knowledge: List[Dict[int, ObstacleState]] = [dict() for _ in range(self.n_uavs)]
        self.observed_this_step: List[Set[int]] = [set() for _ in range(self.n_uavs)]

    def reset(self) -> None:
        self.knowledge = [dict() for _ in range(self.n_uavs)]
        self.observed_this_step = [set() for _ in range(self.n_uavs)]

    def begin_step(self) -> None:
        self.observed_this_step = [set() for _ in range(self.n_uavs)]

    def observe(self, uav_id: int, obstacle: ObstacleState) -> None:
        uav_id = int(uav_id)
        if not 0 <= uav_id < self.n_uavs:
            raise IndexError("invalid UAV id")
        self.knowledge[uav_id][obstacle.obstacle_id] = obstacle.clone()
        self.observed_this_step[uav_id].add(obstacle.obstacle_id)

    def synchronize(self, components: Sequence[Sequence[int]]) -> None:
        for component in components:
            merged: Dict[int, ObstacleState] = {}
            for uav_id in sorted(int(item) for item in component):
                for obstacle_id, obstacle in self.knowledge[uav_id].items():
                    merged[obstacle_id] = obstacle.clone()
            for uav_id in component:
                self.knowledge[int(uav_id)].update(
                    {obstacle_id: obstacle.clone() for obstacle_id, obstacle in merged.items()}
                )

    def active_obstacles(self, uav_id: int, uav_position: np.ndarray) -> List[ObstacleState]:
        active = [
            obstacle.clone()
            for obstacle in self.knowledge[int(uav_id)].values()
            if surface_distance_to_obstacle(uav_position, obstacle)
            <= self.activation_surface_distance + 1e-12
        ]
        return sorted(active, key=lambda obstacle: obstacle.obstacle_id)

    def team_union(self) -> List[ObstacleState]:
        merged: Dict[int, ObstacleState] = {}
        for local in self.knowledge:
            for obstacle_id, obstacle in local.items():
                merged[obstacle_id] = obstacle.clone()
        return [merged[key] for key in sorted(merged)]

    def local_copy(self, uav_id: int) -> Dict[int, ObstacleState]:
        return {
            obstacle_id: obstacle.clone()
            for obstacle_id, obstacle in self.knowledge[int(uav_id)].items()
        }

'''),
    ('graph_system.task_allocation', r'''
"""Legal communication-component target allocation with switch hysteresis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from env.geometry_utils import angle_difference
from env.state_types import (
    EnvConfig,
    TargetAssignmentMetadata,
    TargetObservationMetadata,
    UAVState,
)

from .target_belief import TargetBelief


@dataclass(frozen=True)
class TargetAssignment:
    component: Tuple[int, ...]
    target_id: int
    primary_uav_id: int
    winning_bid: float
    age: int


@dataclass(frozen=True)
class TaskHandoffEvent:
    """One primary-role transfer computed only from legal component state."""

    component: Tuple[int, ...]
    target_id: int
    previous_primary_uav_id: int
    new_primary_uav_id: int
    emergency: bool
    reason: str
    previous_observation_level: float
    new_observation_level: float
    previous_predicted_distance: float
    new_predicted_distance: float


class TaskAllocationManager:
    """Assign at most one primary tracker per target inside each component.

    The manager is deliberately a communication protocol, not an oracle task
    scheduler: every bid uses only the component's already synchronized target
    belief, the participating UAV's own state and its own direct observation
    metadata.  Disconnected components independently maintain assignments.
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.assignments: Dict[Tuple[Tuple[int, ...], int], TargetAssignment] = {}
        self.metadata: list[Dict[int, TargetAssignmentMetadata]] = [
            {} for _ in range(config.n_uavs)
        ]
        self.last_handoff_events: Tuple[TaskHandoffEvent, ...] = ()

    def reset(self) -> None:
        self.assignments = {}
        self.metadata = [{} for _ in range(self.config.n_uavs)]
        self.last_handoff_events = ()

    def _bid(
        self,
        uav: UAVState,
        belief: TargetBelief,
        metadata: TargetObservationMetadata,
    ) -> float:
        displacement = np.asarray(belief.mean[:2], dtype=np.float64) - uav.position
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-8:
            heading_error = 0.0
        else:
            desired_heading = float(np.arctan2(displacement[1], displacement[0]))
            heading_error = abs(angle_difference(desired_heading, uav.heading)) / np.pi
        distance_term = float(np.clip(distance / 250.0, 0.0, 1.0))
        geometric_weight = (
            self.config.task_allocation_distance_weight
            + self.config.task_allocation_heading_weight
        )
        raw = (
            self.config.task_allocation_distance_weight * distance_term
            + self.config.task_allocation_heading_weight * heading_error
        ) / geometric_weight
        # A direct observer has fresher information than a recipient of the
        # same multi-hop belief, but this is only a bounded tie-breaker.
        raw -= self.config.task_allocation_observation_bonus * float(
            np.clip(metadata.self_level, 0.0, 1.0)
        )
        return float(np.clip(raw, 0.0, 1.0))

    def _predicted_distance(
        self,
        uav: UAVState,
        belief: TargetBelief,
    ) -> float:
        """Conservative straight-line separation over the handoff horizon.

        The prediction uses the synchronized CV belief and the UAV's current
        heading at minimum tracking speed.  It never reads target truth.  It is
        deliberately used only as an emergency tie-breaker; the learned Actor
        remains responsible for actual steering and speed control.
        """

        horizon = (
            self.config.task_allocation_emergency_handoff_horizon_steps
            * self.config.dt
        )
        target_future = np.asarray(belief.mean[:2], dtype=np.float64) + horizon * np.asarray(
            belief.mean[2:4], dtype=np.float64
        )
        uav_velocity = self.config.uav_speed_min * np.array(
            [np.cos(uav.heading), np.sin(uav.heading)], dtype=np.float64
        )
        uav_future = np.asarray(uav.position, dtype=np.float64) + horizon * uav_velocity
        return float(np.linalg.norm(target_future - uav_future))

    @staticmethod
    def _current_distance(uav: UAVState, belief: TargetBelief) -> float:
        return float(
            np.linalg.norm(
                np.asarray(belief.mean[:2], dtype=np.float64)
                - np.asarray(uav.position, dtype=np.float64)
            )
        )

    def _emergency_handoff(
        self,
        *,
        component: Tuple[int, ...],
        target_id: int,
        incumbent: TargetAssignment,
        candidate_uav_ids: Sequence[int],
        uavs: Sequence[UAVState],
        beliefs: Sequence[Mapping[int, TargetBelief]],
        observation_metadata: Sequence[Mapping[int, TargetObservationMetadata]],
        bids: Mapping[int, float],
    ) -> Optional[Tuple[int, str, float, float, float, float]]:
        """Return an opportunistic searcher-to-primary role swap if warranted.

        A challenger must already have a current legal direct observation of
        the target and cannot be the incumbent for another target.  Therefore
        this mechanism creates no permanent backup role and does not consume
        search capacity unless an actual handoff is made.
        """

        if not self.config.task_allocation_emergency_handoff_enabled:
            return None
        old_id = int(incumbent.primary_uav_id)
        old_belief = beliefs[old_id].get(target_id)
        if old_belief is None or old_belief.confidence <= 0.0:
            return None
        old_metadata = observation_metadata[old_id].get(
            target_id, TargetObservationMetadata()
        )
        old_level = float(np.clip(old_metadata.self_level, 0.0, 1.0))
        old_predicted = self._predicted_distance(uavs[old_id], old_belief)

        candidates = []
        for candidate_id in sorted(int(value) for value in candidate_uav_ids):
            if candidate_id == old_id:
                continue
            candidate_belief = beliefs[candidate_id].get(target_id)
            if candidate_belief is None or candidate_belief.confidence <= 0.0:
                continue
            candidate_metadata = observation_metadata[candidate_id].get(
                target_id, TargetObservationMetadata()
            )
            candidate_level = float(np.clip(candidate_metadata.self_level, 0.0, 1.0))
            if candidate_level <= 0.0:
                continue
            current_distance = self._current_distance(
                uavs[candidate_id], candidate_belief
            )
            # Swept sensing can observe a target before the end of the step.
            # Requiring the end-state belief distance to remain in the coarse
            # neighbourhood prevents handing off to a UAV that already flew by.
            if current_distance > (
                self.config.target_coarse_radius
                + self.config.task_allocation_emergency_handoff_margin
            ):
                continue
            predicted_distance = self._predicted_distance(
                uavs[candidate_id], candidate_belief
            )
            candidates.append(
                (
                    -candidate_level,
                    predicted_distance,
                    current_distance,
                    float(bids.get(candidate_id, 1.0)),
                    candidate_id,
                    candidate_level,
                )
            )
        if not candidates:
            return None

        (
            _negative_level,
            new_predicted,
            _new_current,
            _new_bid,
            new_id,
            new_level,
        ) = min(candidates)
        margin = self.config.task_allocation_emergency_handoff_margin
        reason = None
        if old_level <= 0.0 and new_level >= 1.0:
            reason = "observer_takeover"
        elif (
            incumbent.age
            < self.config.task_allocation_emergency_handoff_cooldown_steps
        ):
            return None
        elif (
            new_level > old_level
            and new_predicted + margin < old_predicted
        ):
            reason = "observation_upgrade"
        elif (
            old_level >= 1.0
            and new_level >= 1.0
            and old_predicted >= self.config.target_fine_radius
            and new_predicted >= self.config.target_soft_distance
            and new_predicted + margin < old_predicted
        ):
            reason = "predicted_fine_exit"
        elif (
            old_level > 0.0
            and new_level >= old_level
            and old_predicted >= self.config.target_coarse_radius
            and new_predicted >= self.config.target_soft_distance
            and new_predicted + margin < old_predicted
        ):
            reason = "predicted_coarse_exit"
        if reason is None:
            return None
        return (
            int(new_id),
            reason,
            old_level,
            float(new_level),
            float(old_predicted),
            float(new_predicted),
        )

    @staticmethod
    def _minimum_cost_matching(
        target_ids: Sequence[int],
        available_uavs: Sequence[int],
        bids: Mapping[int, Mapping[int, float]],
    ) -> Tuple[Tuple[int, int, float], ...]:
        """Return a deterministic maximum-cardinality, minimum-cost matching.

        A greedy target order can allocate the same best UAV to an easy target
        first and strand a harder target with a very poor tracker.  Teams and
        target sets are both bounded by four, so exhaustive matching is tiny
        (at most a few hundred branches) and gives an exact assignment without
        adding a solver dependency.
        """

        ordered_targets = tuple(sorted(int(value) for value in target_ids))
        ordered_uavs = tuple(sorted(int(value) for value in available_uavs))
        best_key = None
        best_pairs: Tuple[Tuple[int, int, float], ...] = ()

        def search(
            index: int,
            remaining_uavs: Tuple[int, ...],
            pairs: Tuple[Tuple[int, int, float], ...],
            total_cost: float,
        ) -> None:
            nonlocal best_key, best_pairs
            if index >= len(ordered_targets):
                assigned = len(pairs)
                worst_cost = max((item[2] for item in pairs), default=0.0)
                deterministic_pairs = tuple((item[0], item[1]) for item in pairs)
                key = (-assigned, total_cost, worst_cost, deterministic_pairs)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pairs = pairs
                return

            target_id = ordered_targets[index]
            # Skipping is required when a communication component contains
            # fewer UAVs than valid target beliefs.  Cardinality dominates the
            # key, so a target is skipped only when no larger matching exists.
            search(index + 1, remaining_uavs, pairs, total_cost)
            for uav_id in remaining_uavs:
                bid = bids.get(target_id, {}).get(uav_id)
                if bid is None:
                    continue
                search(
                    index + 1,
                    tuple(value for value in remaining_uavs if value != uav_id),
                    pairs + ((target_id, uav_id, float(bid)),),
                    total_cost + float(bid),
                )

        search(0, ordered_uavs, (), 0.0)
        return best_pairs

    def update(
        self,
        uavs: Sequence[UAVState],
        beliefs: Sequence[Mapping[int, TargetBelief]],
        observation_metadata: Sequence[Mapping[int, TargetObservationMetadata]],
        components: Sequence[Sequence[int]],
    ) -> None:
        if len(uavs) != self.config.n_uavs:
            raise ValueError("unexpected UAV count")
        if len(beliefs) != self.config.n_uavs or len(observation_metadata) != self.config.n_uavs:
            raise ValueError("task allocation belief dimensions are inconsistent")
        self.metadata = [{} for _ in range(self.config.n_uavs)]
        self.last_handoff_events = ()
        if not self.config.task_allocation_enabled:
            self.assignments = {}
            return

        previous = self.assignments
        next_assignments: Dict[Tuple[Tuple[int, ...], int], TargetAssignment] = {}
        handoff_events = []
        for raw_component in components:
            component = tuple(sorted(int(uav_id) for uav_id in raw_component))
            members = tuple(component)
            target_ids = sorted(
                {
                    int(target_id)
                    for uav_id in members
                    for target_id in beliefs[uav_id]
                }
            )
            if not target_ids:
                continue

            bids: Dict[int, Dict[int, float]] = {}
            for target_id in target_ids:
                per_uav: Dict[int, float] = {}
                for uav_id in members:
                    belief = beliefs[uav_id].get(target_id)
                    if belief is None or belief.confidence <= 0.0:
                        continue
                    per_uav[uav_id] = self._bid(
                        uavs[uav_id],
                        belief,
                        observation_metadata[uav_id].get(
                            target_id, TargetObservationMetadata()
                        ),
                    )
                if per_uav:
                    bids[target_id] = per_uav

            available = set(members)
            chosen: Dict[int, TargetAssignment] = {}
            incumbent_by_target = {
                target_id: previous.get((component, target_id))
                for target_id in target_ids
            }
            # Preserve a valid incumbent while it is inside its minimum hold
            # period, or until another UAV is materially better.  This avoids
            # deterministic role flipping when two bids are nearly equal.
            for target_id in target_ids:
                old = previous.get((component, target_id))
                per_uav = bids.get(target_id, {})
                if old is None or old.primary_uav_id not in available:
                    continue
                if old.primary_uav_id not in per_uav:
                    continue
                old_bid = per_uav[old.primary_uav_id]
                best_bid = min(per_uav.values())
                protected_other_primaries = {
                    int(other.primary_uav_id)
                    for other_target, other in incumbent_by_target.items()
                    if other_target != target_id and other is not None
                }
                emergency = self._emergency_handoff(
                    component=component,
                    target_id=target_id,
                    incumbent=old,
                    candidate_uav_ids=tuple(
                        sorted(available - protected_other_primaries)
                    ),
                    uavs=uavs,
                    beliefs=beliefs,
                    observation_metadata=observation_metadata,
                    bids=per_uav,
                )
                if emergency is not None:
                    (
                        new_primary,
                        reason,
                        old_level,
                        new_level,
                        old_predicted,
                        new_predicted,
                    ) = emergency
                    chosen[target_id] = TargetAssignment(
                        component=component,
                        target_id=target_id,
                        primary_uav_id=new_primary,
                        winning_bid=float(per_uav[new_primary]),
                        age=0,
                    )
                    available.remove(new_primary)
                    handoff_events.append(
                        TaskHandoffEvent(
                            component=component,
                            target_id=target_id,
                            previous_primary_uav_id=old.primary_uav_id,
                            new_primary_uav_id=new_primary,
                            emergency=True,
                            reason=reason,
                            previous_observation_level=old_level,
                            new_observation_level=new_level,
                            previous_predicted_distance=old_predicted,
                            new_predicted_distance=new_predicted,
                        )
                    )
                    continue
                old_observation_level = float(
                    np.clip(
                        observation_metadata[old.primary_uav_id]
                        .get(target_id, TargetObservationMetadata())
                        .self_level,
                        0.0,
                        1.0,
                    )
                )
                # A normal bid change must never downgrade a live direct
                # observer to a UAV with weaker/no observation.  While the
                # incumbent still sees the target, only the explicit
                # risk-aware emergency protocol may transfer the role.  This
                # avoids assignment-induced LOST events without reserving a
                # permanent backup or recalling a distant search UAV.
                keep = old_observation_level > 0.0 or (
                    old.age < self.config.task_allocation_hold_steps
                    or old_bid
                    <= best_bid + self.config.task_allocation_switch_margin
                )
                if keep:
                    chosen[target_id] = TargetAssignment(
                        component=component,
                        target_id=target_id,
                        primary_uav_id=old.primary_uav_id,
                        winning_bid=old_bid,
                        age=old.age + 1,
                    )
                    available.remove(old.primary_uav_id)

            unassigned = [
                target_id for target_id in target_ids if target_id not in chosen
            ]
            for target_id, primary, winning_bid in self._minimum_cost_matching(
                unassigned, tuple(available), bids
            ):
                chosen[target_id] = TargetAssignment(
                    component=component,
                    target_id=target_id,
                    primary_uav_id=int(primary),
                    winning_bid=float(winning_bid),
                    age=0,
                )
                available.remove(primary)

            for target_id, assignment in chosen.items():
                next_assignments[(component, target_id)] = assignment
                per_uav = bids[target_id]
                age_fraction = float(
                    np.clip(
                        assignment.age
                        / max(self.config.task_allocation_hold_steps, 1),
                        0.0,
                        1.0,
                    )
                )
                for uav_id in members:
                    self.metadata[uav_id][target_id] = TargetAssignmentMetadata(
                        self_primary=float(uav_id == assignment.primary_uav_id),
                        component_primary_present=1.0,
                        self_bid=float(per_uav.get(uav_id, 1.0)),
                        winning_bid=float(assignment.winning_bid),
                        assignment_age_fraction=age_fraction,
                    )
                old = previous.get((component, target_id))
                already_recorded = any(
                    event.component == component
                    and event.target_id == target_id
                    for event in handoff_events
                )
                if (
                    old is not None
                    and old.primary_uav_id != assignment.primary_uav_id
                    and not already_recorded
                ):
                    old_belief = beliefs[old.primary_uav_id].get(target_id)
                    new_belief = beliefs[assignment.primary_uav_id].get(target_id)
                    if old_belief is not None and new_belief is not None:
                        handoff_events.append(
                            TaskHandoffEvent(
                                component=component,
                                target_id=target_id,
                                previous_primary_uav_id=old.primary_uav_id,
                                new_primary_uav_id=assignment.primary_uav_id,
                                emergency=False,
                                reason="normal_bid_handoff",
                                previous_observation_level=float(
                                    observation_metadata[old.primary_uav_id]
                                    .get(target_id, TargetObservationMetadata())
                                    .self_level
                                ),
                                new_observation_level=float(
                                    observation_metadata[assignment.primary_uav_id]
                                    .get(target_id, TargetObservationMetadata())
                                    .self_level
                                ),
                                previous_predicted_distance=self._predicted_distance(
                                    uavs[old.primary_uav_id], old_belief
                                ),
                                new_predicted_distance=self._predicted_distance(
                                    uavs[assignment.primary_uav_id], new_belief
                                ),
                            )
                        )
        self.assignments = next_assignments
        self.last_handoff_events = tuple(handoff_events)
'''),
    ('env.dynamics', r'''
"""Variable-speed UAV dynamics and deterministic rule-based target control."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .geometry_utils import (
    angle_difference,
    heading_vector,
    segment_circle_first_contact,
    segment_surface_distance,
    surface_distance_to_obstacle,
    wrap_angle,
)
from .state_types import EntityKind, EnvConfig, MotionProposal, ObstacleState, TargetState, UAVState


def propose_uav_motion(
    uav: UAVState, action: np.ndarray, config: EnvConfig
) -> MotionProposal:
    commands = np.asarray(action, dtype=np.float64).reshape(-1)
    if commands.shape != (2,):
        raise ValueError("UAV action must contain [turn, speed]")
    turn_command, speed_command = np.clip(commands, -1.0, 1.0)
    heading = wrap_angle(uav.heading + float(turn_command) * config.uav_max_turn_rad)
    speed = config.uav_speed_min + 0.5 * (float(speed_command) + 1.0) * (
        config.uav_speed - config.uav_speed_min
    )
    end = uav.position + speed * config.dt * heading_vector(heading)
    return MotionProposal(
        EntityKind.UAV,
        uav.uav_id,
        np.asarray(uav.position, dtype=np.float64).copy(),
        np.asarray(end, dtype=np.float64),
        float(uav.heading),
        heading,
    )


def _patrol_nominal_heading(target: TargetState, config: EnvConfig) -> float:
    relative = target.position - target.patrol_center
    theta = float(np.arctan2(relative[1], relative[0]))
    delta = (
        target.patrol_direction
        * config.target_speed
        * config.dt
        / max(target.patrol_radius, 1e-9)
    )
    desired = target.patrol_center + target.patrol_radius * np.array(
        [np.cos(theta + delta), np.sin(theta + delta)], dtype=np.float64
    )
    direction = desired - target.position
    return wrap_angle(float(np.arctan2(direction[1], direction[0])))


def target_nominal_heading(
    target: TargetState,
    uavs: Sequence[UAVState],
    config: EnvConfig,
) -> Tuple[float, int, float]:
    nearby = [
        uav.position
        for uav in uavs
        if np.linalg.norm(uav.position - target.position) <= config.target_uav_sense_radius
    ]
    if nearby:
        centroid = np.mean(np.stack(nearby, axis=0), axis=0)
        away = target.position - centroid
        if float(np.dot(away, away)) <= 1e-12:
            away = -heading_vector(target.heading)
        heading = wrap_angle(float(np.arctan2(away[1], away[0])))
        return heading, config.target_evasion_hold_steps, heading
    if target.evasion_hold_remaining > 0 and target.last_evasion_heading is not None:
        return (
            wrap_angle(target.last_evasion_heading),
            target.evasion_hold_remaining - 1,
            wrap_angle(target.last_evasion_heading),
        )
    return _patrol_nominal_heading(target, config), 0, target.last_evasion_heading or target.heading


def _candidate_clearance(
    start: np.ndarray,
    end: np.ndarray,
    visible_obstacles: Sequence[ObstacleState],
    config: EnvConfig,
) -> Tuple[bool, float]:
    lower = config.target_radius
    upper = config.map_size - config.target_radius
    boundary_clearance = float(
        min(end[0] - lower, upper - end[0], end[1] - lower, upper - end[1])
    )
    safe = boundary_clearance >= 0.0
    min_clearance = boundary_clearance
    for obstacle in visible_obstacles:
        tau = segment_circle_first_contact(
            start,
            end,
            obstacle.center,
            obstacle.radius + config.target_radius,
        )
        swept_clearance = (
            segment_surface_distance(start, end, obstacle) - config.target_radius
        )
        min_clearance = min(min_clearance, swept_clearance)
        if tau is not None:
            safe = False
    return safe, min_clearance


def propose_target_motion(
    target: TargetState,
    uavs: Sequence[UAVState],
    obstacles: Sequence[ObstacleState],
    config: EnvConfig,
) -> Tuple[MotionProposal, int, float]:
    nominal, hold_remaining, last_evasion = target_nominal_heading(target, uavs, config)
    visible = [
        obstacle
        for obstacle in obstacles
        if surface_distance_to_obstacle(target.position, obstacle)
        <= config.target_obstacle_sense_surface
    ]
    increments = np.deg2rad(
        np.arange(
            -config.target_max_turn_deg,
            config.target_max_turn_deg + 0.5 * config.target_candidate_turn_step_deg,
            config.target_candidate_turn_step_deg,
        )
    )
    candidates = []
    for increment in increments:
        heading = wrap_angle(target.heading + float(increment))
        end = target.position + config.target_speed * config.dt * heading_vector(heading)
        safe, clearance = _candidate_clearance(target.position, end, visible, config)
        candidates.append(
            {
                "increment": float(increment),
                "heading": heading,
                "end": end,
                "safe": safe,
                "clearance": float(clearance),
                "error": abs(angle_difference(nominal, heading)),
            }
        )
    safe_candidates = [item for item in candidates if item["safe"]]
    if safe_candidates:
        selected = min(
            safe_candidates,
            key=lambda item: (
                round(item["error"], 12),
                -round(item["clearance"], 12),
                round(abs(item["increment"]), 12),
                0 if item["increment"] < 0 else 1,
            ),
        )
    else:
        selected = min(
            candidates,
            key=lambda item: (
                -round(item["clearance"], 12),
                round(item["error"], 12),
                round(abs(item["increment"]), 12),
                0 if item["increment"] < 0 else 1,
            ),
        )
    proposal = MotionProposal(
        EntityKind.TARGET,
        target.target_id,
        np.asarray(target.position, dtype=np.float64).copy(),
        np.asarray(selected["end"], dtype=np.float64),
        float(target.heading),
        float(selected["heading"]),
    )
    return proposal, hold_remaining, float(last_evasion)
'''),
    ('env.initialization', r'''
"""Deterministic seeded rejection sampling for legal episode initial states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from graph_system.communication_manager import CommunicationManager

from .geometry_utils import wrap_angle
from .state_types import EnvConfig, ObstacleState, TargetState, UAVState, WorldState


class InitializationError(RuntimeError):
    pass


@dataclass(frozen=True)
class InitializationResult:
    world: WorldState
    topology_mode: str


class EpisodeInitializer:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.communication = CommunicationManager(config.n_uavs, config.comm_radius)

    def _random_position(self, rng: np.random.Generator, margin: float) -> np.ndarray:
        return rng.uniform(margin, self.config.map_size - margin, size=2).astype(np.float64)

    def _clear_of_obstacles(
        self,
        position: np.ndarray,
        obstacles: Sequence[ObstacleState],
        surface_clearance: float,
    ) -> bool:
        return all(
            np.linalg.norm(position - obstacle.center) - obstacle.radius
            >= surface_clearance - 1e-12
            for obstacle in obstacles
        )

    def _sample_obstacles(self, rng: np.random.Generator) -> List[ObstacleState]:
        obstacles: List[ObstacleState] = []
        for obstacle_id in range(self.config.n_obstacles):
            accepted = False
            for _ in range(self.config.init_max_attempts):
                radius = float(
                    rng.uniform(
                        self.config.obstacle_radius_min,
                        self.config.obstacle_radius_max,
                    )
                )
                boundary_margin = radius + self.config.init_obstacle_surface_clearance
                center = self._random_position(rng, boundary_margin)
                if all(
                    np.linalg.norm(center - existing.center)
                    - radius
                    - existing.radius
                    >= self.config.init_obstacle_surface_clearance - 1e-12
                    for existing in obstacles
                ):
                    obstacles.append(ObstacleState(obstacle_id, center, radius))
                    accepted = True
                    break
            if not accepted:
                raise InitializationError(
                    "could not place obstacle {} after {} attempts".format(
                        obstacle_id, self.config.init_max_attempts
                    )
                )
        return obstacles

    def _valid_uav_position(
        self,
        position: np.ndarray,
        existing: Sequence[np.ndarray],
        obstacles: Sequence[ObstacleState],
    ) -> bool:
        margin = self.config.init_entity_boundary_clearance
        if np.any(position < margin) or np.any(position > self.config.map_size - margin):
            return False
        if not self._clear_of_obstacles(
            position, obstacles, self.config.init_entity_obstacle_clearance
        ):
            return False
        return all(
            np.linalg.norm(position - other) >= self.config.init_uav_uav_distance - 1e-12
            for other in existing
        )

    def _sample_free_uav_position(
        self,
        rng: np.random.Generator,
        existing: Sequence[np.ndarray],
        obstacles: Sequence[ObstacleState],
    ) -> np.ndarray:
        for _ in range(self.config.init_max_attempts):
            position = self._random_position(rng, self.config.init_entity_boundary_clearance)
            if self._valid_uav_position(position, existing, obstacles):
                return position
        raise InitializationError("could not place a legal UAV")

    def _sample_near_position(
        self,
        rng: np.random.Generator,
        anchor: np.ndarray,
        existing: Sequence[np.ndarray],
        obstacles: Sequence[ObstacleState],
        minimum: float,
        maximum: float,
        forbidden: Sequence[np.ndarray] = (),
    ) -> np.ndarray:
        for _ in range(self.config.init_max_attempts):
            distance = float(rng.uniform(minimum, maximum))
            angle = float(rng.uniform(-np.pi, np.pi))
            position = anchor + distance * np.array([np.cos(angle), np.sin(angle)])
            if not self._valid_uav_position(position, existing, obstacles):
                continue
            if any(
                np.linalg.norm(position - other) <= self.config.comm_radius + 1e-9
                for other in forbidden
            ):
                continue
            return position.astype(np.float64)
        raise InitializationError("could not place a UAV near its topology anchor")

    def _sample_uavs(
        self,
        rng: np.random.Generator,
        obstacles: Sequence[ObstacleState],
        topology_mode: str,
    ) -> List[UAVState]:
        positions: List[np.ndarray] = []
        near_min = max(self.config.init_uav_uav_distance + 1.0, 20.0)
        near_max = min(self.config.comm_radius * 0.85, 70.0)
        if topology_mode == "connected":
            positions.append(self._sample_free_uav_position(rng, positions, obstacles))
            while len(positions) < self.config.n_uavs:
                parent = positions[-1]
                positions.append(
                    self._sample_near_position(
                        rng, parent, positions, obstacles, near_min, near_max
                    )
                )
        elif topology_mode == "two_components":
            first = self._sample_free_uav_position(rng, positions, obstacles)
            positions.append(first)
            positions.append(
                self._sample_near_position(
                    rng, first, positions, obstacles, near_min, near_max
                )
            )
            for _ in range(self.config.init_max_attempts):
                candidate = self._sample_free_uav_position(rng, positions, obstacles)
                if all(
                    np.linalg.norm(candidate - member) > self.config.comm_radius + 1e-9
                    for member in positions[:2]
                ):
                    positions.append(candidate)
                    break
            if len(positions) != 3:
                raise InitializationError("could not separate the second UAV component")
            positions.append(
                self._sample_near_position(
                    rng,
                    positions[2],
                    positions,
                    obstacles,
                    near_min,
                    near_max,
                    forbidden=positions[:2],
                )
            )
        elif topology_mode == "random":
            for _ in range(self.config.n_uavs):
                positions.append(self._sample_free_uav_position(rng, positions, obstacles))
        else:
            raise ValueError("unknown topology mode: {}".format(topology_mode))

        adjacency = self.communication.build_adjacency(positions)
        component_count = len(self.communication.connected_components(adjacency))
        if topology_mode == "connected" and component_count != 1:
            raise InitializationError("constructed connected topology is not connected")
        if topology_mode == "two_components" and component_count != 2:
            raise InitializationError("constructed topology does not contain two components")
        return [
            UAVState(
                uav_id=index,
                position=position,
                heading=float(rng.uniform(-np.pi, np.pi)),
                previous_action=0.0,
            )
            for index, position in enumerate(positions)
        ]

    def _patrol_circle_is_safe(
        self,
        center: np.ndarray,
        radius: float,
        obstacles: Sequence[ObstacleState],
    ) -> bool:
        margin = radius + self.config.patrol_clearance
        if np.any(center < margin) or np.any(center > self.config.map_size - margin):
            return False
        return all(
            np.linalg.norm(center - obstacle.center)
            - radius
            - obstacle.radius
            >= self.config.patrol_clearance - 1e-12
            for obstacle in obstacles
        )

    def _sample_targets(
        self,
        rng: np.random.Generator,
        obstacles: Sequence[ObstacleState],
        uavs: Sequence[UAVState],
    ) -> List[TargetState]:
        targets: List[TargetState] = []
        for target_id in range(self.config.num_targets):
            accepted = False
            for _ in range(self.config.init_max_attempts):
                patrol_radius = float(
                    rng.uniform(
                        self.config.target_patrol_radius_min,
                        self.config.target_patrol_radius_max,
                    )
                )
                margin = patrol_radius + self.config.patrol_clearance
                center = self._random_position(rng, margin)
                if not self._patrol_circle_is_safe(center, patrol_radius, obstacles):
                    continue
                theta = float(rng.uniform(-np.pi, np.pi))
                position = center + patrol_radius * np.array(
                    [np.cos(theta), np.sin(theta)], dtype=np.float64
                )
                boundary_margin = self.config.init_entity_boundary_clearance
                if np.any(position < boundary_margin) or np.any(
                    position > self.config.map_size - boundary_margin
                ):
                    continue
                if not self._clear_of_obstacles(
                    position, obstacles, self.config.init_entity_obstacle_clearance
                ):
                    continue
                if any(
                    np.linalg.norm(position - uav.position)
                    < self.config.init_uav_target_distance - 1e-12
                    for uav in uavs
                ):
                    continue
                if any(
                    np.linalg.norm(position - target.position)
                    < self.config.init_target_target_distance - 1e-12
                    for target in targets
                ):
                    continue
                direction = -1 if int(rng.integers(0, 2)) == 0 else 1
                heading = wrap_angle(theta + direction * np.pi / 2.0)
                targets.append(
                    TargetState(
                        target_id=target_id,
                        position=position.astype(np.float64),
                        heading=heading,
                        patrol_center=center.astype(np.float64),
                        patrol_radius=patrol_radius,
                        patrol_direction=direction,
                    )
                )
                accepted = True
                break
            if not accepted:
                raise InitializationError(
                    "could not place target {} with a safe patrol circle".format(target_id)
                )
        return targets

    def choose_topology_mode(self, rng: np.random.Generator) -> str:
        stage = self.config.curriculum_stage
        draw = float(rng.random())
        if stage == 1:
            return "connected" if draw < 0.70 else "two_components"
        if stage == 2:
            if draw < 0.50:
                return "connected"
            if draw < 0.80:
                return "two_components"
            return "random"
        if draw < 1.0 / 3.0:
            return "connected"
        if draw < 2.0 / 3.0:
            return "two_components"
        return "random"

    def sample(
        self,
        rng: np.random.Generator,
        topology_mode: Optional[str] = None,
    ) -> InitializationResult:
        self.config.validate()
        mode = topology_mode or self.choose_topology_mode(rng)
        last_error: Optional[Exception] = None
        # A failed topology construction restarts the complete scene. This is
        # bounded and deterministic for a fixed RNG seed.
        for _ in range(self.config.init_max_attempts):
            try:
                obstacles = self._sample_obstacles(rng)
                uavs = self._sample_uavs(rng, obstacles, mode)
                targets = self._sample_targets(rng, obstacles, uavs)
                return InitializationResult(
                    WorldState(uavs=uavs, targets=targets, obstacles=obstacles, step_count=0),
                    mode,
                )
            except InitializationError as error:
                last_error = error
        raise InitializationError(
            "episode initialization failed for mode {!r}: {}".format(mode, last_error)
        )
'''),
    ('graph_system.graph_builder', r'''
"""Encode legal local knowledge into Actor observations."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from env.geometry_utils import angle_difference, surface_distance_to_obstacle
from env.state_types import (
    EnvConfig,
    LocalKnowledgeSnapshot,
    ObstacleState,
    TargetAssignmentMetadata,
    TargetObservationMetadata,
)
from graph_system.target_belief import TargetBelief

from .graph_types import ActorGraphSnapshot


def _ego_rotation(heading: float) -> np.ndarray:
    cosine = float(np.cos(heading))
    sine = float(np.sin(heading))
    return np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)


def _normalize_covariance(position_covariance: np.ndarray) -> Tuple[float, float, float]:
    covariance = np.asarray(position_covariance, dtype=np.float64)
    pxx = max(float(covariance[0, 0]), 0.0)
    pyy = max(float(covariance[1, 1]), 0.0)
    pxy = float(covariance[0, 1])
    denominator = np.log1p(2500.0)
    pxx_norm = float(np.clip(np.log1p(pxx) / denominator, 0.0, 1.0))
    pyy_norm = float(np.clip(np.log1p(pyy) / denominator, 0.0, 1.0))
    correlation = float(np.clip(pxy / (np.sqrt(pxx * pyy) + 1e-8), -1.0, 1.0))
    return pxx_norm, pyy_norm, correlation


class GraphBuilder:
    def __init__(self, config: EnvConfig):
        self.config = config
        centers = (
            np.arange(config.coverage_grid_size, dtype=np.float64) + 0.5
        ) * config.coverage_cell_size
        self.grid_x, self.grid_y = np.meshgrid(centers, centers)

    def _boundary_features(self, position: np.ndarray) -> np.ndarray:
        x, y = float(position[0]), float(position[1])
        distance = self.config.boundary_sense_distance
        return np.array(
            [
                np.clip((distance - x) / distance, 0.0, 1.0),
                np.clip((distance - (self.config.map_size - x)) / distance, 0.0, 1.0),
                np.clip((distance - y) / distance, 0.0, 1.0),
                np.clip((distance - (self.config.map_size - y)) / distance, 0.0, 1.0),
            ],
            dtype=np.float64,
        )

    def _coverage_sectors(self, local: LocalKnowledgeSnapshot) -> np.ndarray:
        position = local.ego.position
        dx = self.grid_x - position[0]
        dy = self.grid_y - position[1]
        eligible = dx * dx + dy * dy <= self.config.coverage_context_radius ** 2
        for obstacle in local.known_obstacles:
            eligible &= (
                (self.grid_x - obstacle.center[0]) ** 2
                + (self.grid_y - obstacle.center[1]) ** 2
                > obstacle.radius ** 2
            )
        angle = (np.arctan2(dy, dx) - local.ego.heading) % (2.0 * np.pi)
        sector = (
            np.floor((angle + np.pi / 8.0) / (np.pi / 4.0)).astype(np.int16) % 8
        )
        result = np.zeros(8, dtype=np.float64)
        for index in range(8):
            cells = eligible & (sector == index)
            count = int(np.count_nonzero(cells))
            if count:
                result[index] = float(
                    np.count_nonzero(cells & ~local.coverage_knowledge) / count
                )
        return result

    def _self_feature(self, local: LocalKnowledgeSnapshot) -> np.ndarray:
        feature = np.concatenate(
            [
                np.array(
                    [
                        np.sin(local.ego.heading),
                        np.cos(local.ego.heading),
                        local.ego.previous_action,
                    ],
                    dtype=np.float64,
                ),
                self._boundary_features(local.ego.position),
                self._coverage_sectors(local),
                np.array(
                    [
                        len(local.direct_neighbors) / 3.0,
                        len(local.component_members) / 4.0,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        if feature.shape != (17,):
            raise AssertionError("ego feature dimension drifted")
        return feature.astype(np.float32)

    def _neighbor_feature(self, local: LocalKnowledgeSnapshot, neighbor: object) -> np.ndarray:
        rotation = _ego_rotation(local.ego.heading)
        relative = rotation @ (neighbor.position - local.ego.position)
        distance = float(np.linalg.norm(relative))
        feature = np.array(
            [
                np.clip(relative[0] / self.config.comm_radius, -1.0, 1.0),
                np.clip(relative[1] / self.config.comm_radius, -1.0, 1.0),
                np.clip(distance / self.config.comm_radius, 0.0, 1.0),
                np.sin(neighbor.heading - local.ego.heading),
                np.cos(neighbor.heading - local.ego.heading),
                np.clip(neighbor.previous_action, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return feature

    def _obstacle_feature(
        self, local: LocalKnowledgeSnapshot, obstacle: ObstacleState
    ) -> np.ndarray:
        rotation = _ego_rotation(local.ego.heading)
        relative = rotation @ (obstacle.center - local.ego.position)
        surface = surface_distance_to_obstacle(local.ego.position, obstacle)
        scale = self.config.obstacle_radius_max + self.config.obstacle_activation_surface
        return np.array(
            [
                np.clip(relative[0] / scale, -1.0, 1.0),
                np.clip(relative[1] / scale, -1.0, 1.0),
                np.clip(surface / self.config.obstacle_activation_surface, 0.0, 1.0),
                np.clip(obstacle.radius / self.config.obstacle_radius_max, 0.0, 1.0),
                1.0
                if obstacle.obstacle_id in local.observed_obstacle_ids_this_step
                else 0.0,
            ],
            dtype=np.float32,
        )

    def _target_feature(
        self,
        local: LocalKnowledgeSnapshot,
        belief: TargetBelief,
        metadata: TargetObservationMetadata,
    ) -> np.ndarray:
        rotation = _ego_rotation(local.ego.heading)
        relative = rotation @ (belief.mean[:2] - local.ego.position)
        velocity = rotation @ belief.mean[2:4]
        rotated_covariance = rotation @ belief.covariance[:2, :2] @ rotation.T
        pxx, pyy, pxy = _normalize_covariance(rotated_covariance)
        assignment = local.target_assignments.get(
            belief.target_id, TargetAssignmentMetadata()
        )
        claimed_by_other = (
            assignment.component_primary_present * (1.0 - assignment.self_primary)
        )
        return np.array(
            [
                np.clip(relative[0] / 250.0, -1.0, 1.0),
                np.clip(relative[1] / 250.0, -1.0, 1.0),
                np.clip(np.linalg.norm(relative) / 250.0, 0.0, 1.0),
                np.clip(velocity[0] / self.config.target_speed, -1.0, 1.0),
                np.clip(velocity[1] / self.config.target_speed, -1.0, 1.0),
                np.clip(belief.confidence, 0.0, 1.0),
                np.clip(belief.age / self.config.belief_lost_age, 0.0, 1.0),
                pxx,
                pyy,
                pxy,
                np.clip(metadata.self_level, 0.0, 1.0),
                np.clip(metadata.component_best_level, 0.0, 1.0),
                np.clip(metadata.component_observer_ratio, 0.0, 1.0),
                np.clip(assignment.self_primary, 0.0, 1.0),
                np.clip(claimed_by_other, 0.0, 1.0),
                np.clip(assignment.self_bid - assignment.winning_bid, 0.0, 1.0),
                np.clip(assignment.assignment_age_fraction, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def build_actor(self, locals_: Sequence[LocalKnowledgeSnapshot]) -> ActorGraphSnapshot:
        if len(locals_) != self.config.n_uavs:
            raise ValueError("exactly four local knowledge snapshots are required")
        self_features = np.zeros((4, 17), dtype=np.float32)
        neighbor_features = np.zeros((4, 3, 6), dtype=np.float32)
        neighbor_mask = np.zeros((4, 3), dtype=bool)
        obstacle_features = np.zeros((4, 8, 5), dtype=np.float32)
        obstacle_mask = np.zeros((4, 8), dtype=bool)
        target_features = np.zeros((4, 4, 17), dtype=np.float32)
        target_mask = np.zeros((4, 4), dtype=bool)

        for local in sorted(locals_, key=lambda item: item.ego.uav_id):
            uav_id = local.ego.uav_id
            self_features[uav_id] = self._self_feature(local)
            neighbors = sorted(local.direct_neighbors, key=lambda item: item.uav_id)
            if len(neighbors) > 3:
                raise ValueError("a four-UAV team cannot have more than three neighbors")
            for index, neighbor in enumerate(neighbors):
                neighbor_features[uav_id, index] = self._neighbor_feature(local, neighbor)
                neighbor_mask[uav_id, index] = True
            obstacles = sorted(local.active_obstacles, key=lambda item: item.obstacle_id)
            if len(obstacles) > 8:
                raise ValueError("active obstacle count exceeds fixed limit")
            for index, obstacle in enumerate(obstacles):
                obstacle_features[uav_id, index] = self._obstacle_feature(local, obstacle)
                obstacle_mask[uav_id, index] = True
            target_ids = sorted(local.target_beliefs)
            if len(target_ids) > 4:
                raise ValueError("target belief count exceeds fixed limit")
            for index, target_id in enumerate(target_ids):
                target_features[uav_id, index] = self._target_feature(
                    local,
                    local.target_beliefs[target_id],
                    local.target_metadata.get(target_id, TargetObservationMetadata()),
                )
                target_mask[uav_id, index] = True
        snapshot = ActorGraphSnapshot(
            self_features,
            neighbor_features,
            neighbor_mask,
            obstacle_features,
            obstacle_mask,
            target_features,
            target_mask,
        )
        snapshot.validate()
        return snapshot

'''),
    ('graph_system.action_assist', r'''
"""Optional, observation-legal action assists for deterministic execution.

The helpers in this module only consume the same ego-centric Actor snapshot
available to the decentralized policy.  They therefore do not introduce
truth-state or centralized-information leakage.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .graph_types import ActorGraphSnapshot


# Target feature indices match the local Actor observation encoder.
TARGET_RELATIVE_X = 0
TARGET_RELATIVE_Y = 1
TARGET_VELOCITY_X = 3
TARGET_VELOCITY_Y = 4
TARGET_CONFIDENCE = 5
TARGET_AGE_FRACTION = 6
TARGET_SELF_OBSERVATION = 10
TARGET_COMPONENT_OBSERVATION = 11
TARGET_SELF_PRIMARY = 13


def _segment_origin_distance(start: np.ndarray, end: np.ndarray) -> float:
    """Minimum distance from the origin to a two-dimensional segment."""

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return float(np.linalg.norm(start))
    tau = float(np.clip(-np.dot(start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(start + tau * delta))


def apply_predictive_safety_assist(
    snapshot: ActorGraphSnapshot,
    actions: np.ndarray,
    *,
    blend: float,
    prediction_horizon_steps: float = 1.0,
    comm_radius: float = 83.0,
    obstacle_activation_surface: float = 30.0,
    obstacle_radius_max: float = 4.0,
    boundary_sense_distance: float = 30.0,
    uav_radius: float = 1.0,
    target_radius: float = 1.0,
    target_speed: float = 5.0,
    uav_speed_min: float = 5.0,
    uav_speed_max: float = 10.0,
    max_turn_degrees: float = 32.5,
    obstacle_clearance: float = 4.0,
    boundary_clearance: float = 5.0,
    uav_separation: float = 6.0,
    target_separation: float = 6.0,
    activation_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, int]:
    """Select a safer nearby turn using only the decentralized Actor input.

    The filter predicts one short motion segment for a small set of admissible
    turn commands while preserving the policy speed command.  It reads only
    ego boundary risks, directly communicated neighbours, active known
    obstacles and current legal target beliefs from ``ActorGraphSnapshot``.
    A command is changed only when the policy proposal crosses a hard contact
    or ends inside a configured warning margin. Future entity actions
    are unavailable to this helper.
    """

    blend = float(blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    horizon = float(prediction_horizon_steps)
    if horizon <= 0.0:
        raise ValueError("prediction_horizon_steps must be positive")
    positive = {
        "comm_radius": comm_radius,
        "obstacle_activation_surface": obstacle_activation_surface,
        "obstacle_radius_max": obstacle_radius_max,
        "boundary_sense_distance": boundary_sense_distance,
        "uav_radius": uav_radius,
        "target_radius": target_radius,
        "target_speed": target_speed,
        "uav_speed_min": uav_speed_min,
        "uav_speed_max": uav_speed_max,
        "max_turn_degrees": max_turn_degrees,
        "obstacle_clearance": obstacle_clearance,
        "boundary_clearance": boundary_clearance,
        "uav_separation": uav_separation,
        "target_separation": target_separation,
    }
    if any(float(value) <= 0.0 for value in positive.values()):
        raise ValueError("predictive safety distances and dynamics must be positive")
    if uav_speed_min > uav_speed_max:
        raise ValueError("UAV speeds must satisfy minimum <= maximum")
    if uav_separation <= 2.0 * uav_radius:
        raise ValueError("UAV separation must exceed hard contact distance")
    if target_separation <= uav_radius + target_radius:
        raise ValueError("target separation must exceed hard contact distance")

    if activation_mask is not None:
        if np.asarray(activation_mask).shape != (4,):
            raise ValueError("activation_mask must have shape (4,)")
        activation_mask[...] = False
    original = np.asarray(actions)
    reshaped = np.asarray(actions, dtype=np.float64).reshape(4, 2).copy()
    if blend == 0.0:
        return reshaped.astype(original.dtype, copy=False).reshape(original.shape), 0

    self_features = np.asarray(snapshot.self_features, dtype=np.float64)
    neighbor_features = np.asarray(snapshot.neighbor_features, dtype=np.float64)
    neighbor_mask = np.asarray(snapshot.neighbor_mask, dtype=bool)
    obstacle_features = np.asarray(snapshot.obstacle_features, dtype=np.float64)
    obstacle_mask = np.asarray(snapshot.obstacle_mask, dtype=bool)
    target_features = np.asarray(snapshot.target_features, dtype=np.float64)
    target_mask = np.asarray(snapshot.target_mask, dtype=bool)
    if self_features.shape != (4, 17):
        raise ValueError("unexpected Actor self snapshot shape")
    if neighbor_features.shape != (4, 3, 6) or neighbor_mask.shape != (4, 3):
        raise ValueError("unexpected Actor neighbour snapshot shape")
    if obstacle_features.shape != (4, 8, 5) or obstacle_mask.shape != (4, 8):
        raise ValueError("unexpected Actor obstacle snapshot shape")
    if target_features.shape != (4, 4, 17) or target_mask.shape != (4, 4):
        raise ValueError("unexpected Actor target snapshot shape")

    max_turn_radians = np.deg2rad(float(max_turn_degrees))
    obstacle_scale = float(obstacle_radius_max + obstacle_activation_surface)
    uav_contact = 2.0 * float(uav_radius)
    target_contact = float(uav_radius + target_radius)
    uav_warning_margin = float(uav_separation - uav_contact)
    target_warning_margin = float(target_separation - target_contact)
    candidate_grid = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    modified = 0
    epsilon = 1e-9

    for uav_id in range(4):
        policy_turn = float(np.clip(reshaped[uav_id, 0], -1.0, 1.0))
        speed_action = float(np.clip(reshaped[uav_id, 1], -1.0, 1.0))
        speed = float(uav_speed_min) + 0.5 * (speed_action + 1.0) * (
            float(uav_speed_max) - float(uav_speed_min)
        )
        heading = float(
            np.arctan2(self_features[uav_id, 0], self_features[uav_id, 1])
        )

        def score(turn: float) -> tuple[float, float, float]:
            local_delta = float(turn) * max_turn_radians
            ego_end = horizon * speed * np.array(
                [np.cos(local_delta), np.sin(local_delta)], dtype=np.float64
            )
            world_angle = heading + local_delta
            world_end = horizon * speed * np.array(
                [np.cos(world_angle), np.sin(world_angle)], dtype=np.float64
            )
            swept_clearances = []
            endpoint_scores = []

            boundary_risks = self_features[uav_id, 3:7]
            centre_distances = np.where(
                boundary_risks > 0.0,
                boundary_sense_distance * (1.0 - boundary_risks),
                boundary_sense_distance,
            )
            boundary_end = centre_distances + np.array(
                [world_end[0], -world_end[0], world_end[1], -world_end[1]],
                dtype=np.float64,
            )
            boundary_start_clearance = centre_distances - uav_radius
            boundary_end_clearance = boundary_end - uav_radius
            swept_clearances.append(
                float(np.min(np.minimum(boundary_start_clearance, boundary_end_clearance)))
            )
            endpoint_scores.append(
                float(np.min(boundary_end_clearance / boundary_clearance))
            )

            for index in np.flatnonzero(obstacle_mask[uav_id]):
                feature = obstacle_features[uav_id, int(index)]
                relative = feature[:2] * obstacle_scale
                radius = float(feature[3] * obstacle_radius_max)
                relative_end = relative - ego_end
                swept = _segment_origin_distance(relative, relative_end) - radius - uav_radius
                endpoint = float(np.linalg.norm(relative_end)) - radius - uav_radius
                swept_clearances.append(float(swept))
                endpoint_scores.append(float(endpoint / obstacle_clearance))

            for index in np.flatnonzero(neighbor_mask[uav_id]):
                relative = neighbor_features[uav_id, int(index), :2] * comm_radius
                relative_end = relative - ego_end
                swept = _segment_origin_distance(relative, relative_end) - uav_contact
                endpoint = float(np.linalg.norm(relative_end)) - uav_contact
                swept_clearances.append(float(swept))
                endpoint_scores.append(float(endpoint / uav_warning_margin))

            eligible_targets = (
                target_mask[uav_id]
                & (target_features[uav_id, :, TARGET_CONFIDENCE] > 0.0)
                & (target_features[uav_id, :, TARGET_AGE_FRACTION] <= 1e-6)
                & (target_features[uav_id, :, TARGET_SELF_OBSERVATION] > 0.0)
            )
            for index in np.flatnonzero(eligible_targets):
                feature = target_features[uav_id, int(index)]
                relative = feature[[TARGET_RELATIVE_X, TARGET_RELATIVE_Y]] * 250.0
                velocity = feature[[TARGET_VELOCITY_X, TARGET_VELOCITY_Y]] * target_speed
                relative_end = relative + horizon * velocity - ego_end
                swept = _segment_origin_distance(relative, relative_end) - target_contact
                endpoint = float(np.linalg.norm(relative_end)) - target_contact
                swept_clearances.append(float(swept))
                endpoint_scores.append(float(endpoint / target_warning_margin))

            minimum_swept = min(swept_clearances, default=float("inf"))
            minimum_endpoint = min(endpoint_scores, default=float("inf"))
            return (
                float(minimum_swept >= 0.0),
                float(minimum_endpoint),
                float(minimum_swept),
            )

        policy_score = score(policy_turn)
        if policy_score[0] > 0.5 and policy_score[1] >= 1.0:
            continue
        candidates = np.unique(np.append(candidate_grid, policy_turn))
        best_turn = policy_turn
        best_key = (*policy_score, 0.0)
        for candidate in candidates:
            candidate_score = score(float(candidate))
            key = (
                *candidate_score,
                -abs(float(candidate) - policy_turn),
            )
            if key > best_key:
                best_key = key
                best_turn = float(candidate)
        if abs(best_turn - policy_turn) <= epsilon:
            continue
        reshaped[uav_id, 0] = np.clip(
            (1.0 - blend) * policy_turn + blend * best_turn,
            -1.0,
            1.0,
        )
        if activation_mask is not None:
            activation_mask[uav_id] = True
        modified += 1

    return reshaped.astype(original.dtype, copy=False).reshape(original.shape), modified

def apply_primary_continuity_assist(
    snapshot: ActorGraphSnapshot,
    actions: np.ndarray,
    *,
    blend: float,
    prediction_horizon_steps: float = 2.0,
    fine_radius: float = 13.0,
    coarse_radius: float = 30.0,
    target_speed: float = 5.0,
    uav_speed_min: float = 5.0,
    uav_speed_max: float = 10.0,
    max_turn_degrees: float = 30.0,
    minimum_risk_speed: float | None = None,
    speed_boost_heading_limit_degrees: float = 60.0,
    activation_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, int]:
    """Steer an observed primary tracker before it exits a sensing radius.

    This is deliberately not a backup-tracker mechanism.  It only modifies
    the current primary UAV and leaves every search UAV untouched.  The risk
    estimate is computed from the ego Actor snapshot, the legal target belief,
    the current component observation level, and the current policy speed
    command. Only local observations are used.

    Fine observations are guarded against a predicted exit from ``fine_radius``;
    coarse observations are guarded against a predicted exit from
    ``coarse_radius``.  A correction is applied only while the relative radial
    motion is outward and the straight-line prediction crosses the relevant
    boundary.  ``minimum_risk_speed`` optionally raises only the at-risk
    primary's speed floor when the predicted target remains inside a bounded
    forward cone.  Search UAVs and stable tracking are never speed-modified.
    """

    blend = float(blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    horizon = float(prediction_horizon_steps)
    if horizon <= 0.0:
        raise ValueError("prediction_horizon_steps must be positive")
    fine_radius = float(fine_radius)
    coarse_radius = float(coarse_radius)
    target_speed = float(target_speed)
    uav_speed_min = float(uav_speed_min)
    uav_speed_max = float(uav_speed_max)
    if not 0.0 < fine_radius < coarse_radius:
        raise ValueError("radii must satisfy 0 < fine_radius < coarse_radius")
    if target_speed <= 0.0:
        raise ValueError("target_speed must be positive")
    if not 0.0 < uav_speed_min <= uav_speed_max:
        raise ValueError("UAV speeds must satisfy 0 < minimum <= maximum")
    if minimum_risk_speed is not None:
        minimum_risk_speed = float(minimum_risk_speed)
        if not uav_speed_min <= minimum_risk_speed <= uav_speed_max:
            raise ValueError("minimum_risk_speed must lie within UAV speed bounds")
    speed_heading_limit = np.deg2rad(float(speed_boost_heading_limit_degrees))
    if not 0.0 <= speed_heading_limit <= np.pi:
        raise ValueError("speed boost heading limit must be in [0, 180] degrees")
    max_turn_radians = np.deg2rad(float(max_turn_degrees))
    if max_turn_radians <= 0.0:
        raise ValueError("max_turn_degrees must be positive")

    if activation_mask is not None:
        if np.asarray(activation_mask).shape != (4,):
            raise ValueError("activation_mask must have shape (4,)")
        activation_mask[...] = False
    original = np.asarray(actions)
    reshaped = np.asarray(actions, dtype=np.float64).reshape(4, 2).copy()
    if blend == 0.0:
        return reshaped.astype(original.dtype, copy=False).reshape(original.shape), 0

    self_features = np.asarray(snapshot.self_features, dtype=np.float64)
    target_features = np.asarray(snapshot.target_features, dtype=np.float64)
    target_mask = np.asarray(snapshot.target_mask, dtype=bool)
    if self_features.shape != (4, 17):
        raise ValueError("unexpected Actor self snapshot shape")
    if target_features.shape != (4, 4, 17) or target_mask.shape != (4, 4):
        raise ValueError("unexpected Actor target snapshot shape")

    modified = 0
    epsilon = 1e-6
    for uav_id in range(4):
        eligible = (
            target_mask[uav_id]
            & (target_features[uav_id, :, TARGET_CONFIDENCE] > 0.0)
            & (target_features[uav_id, :, TARGET_AGE_FRACTION] <= epsilon)
            & (target_features[uav_id, :, TARGET_COMPONENT_OBSERVATION] > epsilon)
            & (target_features[uav_id, :, TARGET_SELF_PRIMARY] > 0.5)
        )
        candidate_indices = np.flatnonzero(eligible)
        if candidate_indices.size == 0:
            continue

        # The current Actor schema stores boundary risks at self-feature
        # indices 3:7; it does not store a previous speed command.  The
        # executable speed command is already available in the action being
        # assisted and is the only value consistent with this control step.
        speed_action = np.clip(reshaped[uav_id, 1], -1.0, 1.0)
        ego_speed = uav_speed_min + 0.5 * (speed_action + 1.0) * (
            uav_speed_max - uav_speed_min
        )

        best_index = None
        best_excess = -np.inf
        best_future_target = None
        for target_index in candidate_indices:
            feature = target_features[uav_id, int(target_index)]
            relative_position = feature[
                [TARGET_RELATIVE_X, TARGET_RELATIVE_Y]
            ] * 250.0
            distance = float(np.linalg.norm(relative_position))
            if distance <= epsilon:
                continue
            target_velocity = feature[
                [TARGET_VELOCITY_X, TARGET_VELOCITY_Y]
            ] * target_speed
            relative_velocity = target_velocity - np.array(
                [ego_speed, 0.0], dtype=np.float64
            )
            radial_velocity = float(
                np.dot(relative_position, relative_velocity) / distance
            )
            if radial_velocity <= 0.0:
                continue
            predicted_relative = relative_position + horizon * relative_velocity
            predicted_distance = float(np.linalg.norm(predicted_relative))
            observation_level = feature[TARGET_COMPONENT_OBSERVATION]
            boundary = fine_radius if observation_level >= 1.0 - epsilon else coarse_radius
            excess = predicted_distance - boundary
            if excess <= 0.0 or excess <= best_excess:
                continue
            best_index = int(target_index)
            best_excess = excess
            best_future_target = relative_position + horizon * target_velocity

        if best_index is None or best_future_target is None:
            continue
        desired_turn = np.clip(
            np.arctan2(best_future_target[1], best_future_target[0])
            / max_turn_radians,
            -1.0,
            1.0,
        )
        reshaped[uav_id, 0] = np.clip(
            (1.0 - blend) * reshaped[uav_id, 0] + blend * desired_turn,
            -1.0,
            1.0,
        )
        desired_heading_error = abs(
            float(np.arctan2(best_future_target[1], best_future_target[0]))
        )
        if (
            minimum_risk_speed is not None
            and desired_heading_error <= speed_heading_limit + epsilon
        ):
            speed_floor_action = (
                2.0
                * (minimum_risk_speed - uav_speed_min)
                / max(uav_speed_max - uav_speed_min, epsilon)
                - 1.0
            )
            reshaped[uav_id, 1] = max(
                reshaped[uav_id, 1], speed_floor_action
            )
        if activation_mask is not None:
            activation_mask[uav_id] = True
        modified += 1

    return reshaped.astype(original.dtype, copy=False).reshape(original.shape), modified


'''),
    ('env.graph_search_env', r'''
"""Phase 1 multi-UAV search/tracking environment with strict step causality."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from graph_system.communication_manager import CommunicationManager
from graph_system.coverage_knowledge import CoverageKnowledge
from graph_system.graph_builder import GraphBuilder
from graph_system.graph_types import ActorGraphSnapshot
from graph_system.obstacle_knowledge import ObstacleKnowledge
from graph_system.task_allocation import TaskAllocationManager
from graph_system.target_belief import TargetBeliefManager, TargetLifecycleManager

from .dynamics import propose_target_motion, propose_uav_motion
from .geometry_utils import (
    resolve_first_collisions,
    segment_surface_distance,
    swept_paths_interval_within_radius,
    swept_paths_min_distance,
)
from .initialization import EpisodeInitializer

from .state_types import (
    EntityKind,
    EnvConfig,
    LocalKnowledgeSnapshot,
    MeasurementKind,
    ObstacleState,
    SweepPath,
    TargetMeasurement,
    TargetObservationMetadata,
    UAVState,
    WorldState,
)


class GraphSearchEnv:
    """CPU environment; graph encoding is intentionally deferred to Phase 2."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng()
        self._install_config(self.config)
        self.lifecycle: Optional[TargetLifecycleManager] = None
        self.world: Optional[WorldState] = None
        self.communication_adjacency = np.zeros(
            (self.config.n_uavs, self.config.n_uavs), dtype=bool
        )
        self.components: List[Tuple[int, ...]] = []
        self.topology_mode: Optional[str] = None
        self.scenario_mode: str = "normal"
        self.last_local_knowledge_snapshots: Optional[List[LocalKnowledgeSnapshot]] = None
        self.last_actor_snapshots: Optional[ActorGraphSnapshot] = None
        self.last_info: Dict[str, Any] = {}

    def _install_config(self, config: EnvConfig) -> None:
        config.validate()
        self.config = config
        self.initializer = EpisodeInitializer(self.config)
        self.communication = CommunicationManager(
            self.config.n_uavs, self.config.comm_radius
        )
        self.obstacle_knowledge = ObstacleKnowledge(
            self.config.n_uavs, self.config.obstacle_activation_surface
        )
        self.coverage = CoverageKnowledge(self.config)
        self.target_beliefs = TargetBeliefManager(self.config)
        self.task_allocation = TaskAllocationManager(self.config)
        self.graph_builder = GraphBuilder(self.config)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ActorGraphSnapshot, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        if options is not None:
            requested_targets = int(options.get("num_targets", self.config.num_targets))
            requested_stage = int(
                options.get("curriculum_stage", self.config.curriculum_stage)
            )
            if (
                requested_targets != self.config.num_targets
                or requested_stage != self.config.curriculum_stage
            ):
                self._install_config(
                    replace(
                        self.config,
                        num_targets=requested_targets,
                        curriculum_stage=requested_stage,
                    )
                )
        topology_mode = None if options is None else options.get("topology_mode")
        result = self.initializer.sample(
            self.rng,
            topology_mode=topology_mode,
        )
        self.world = result.world
        self.topology_mode = result.topology_mode
        self.scenario_mode = "normal"
        self.obstacle_knowledge.reset()
        self.coverage.reset(self.world.obstacles)
        self.target_beliefs.reset()
        self.task_allocation.reset()
        self.lifecycle = TargetLifecycleManager(
            [target.target_id for target in self.world.targets],
            lost_after_steps=self.config.belief_lost_age,
        )
        self.communication_adjacency = self.communication.build_adjacency(
            [uav.position for uav in self.world.uavs]
        )
        self.components = self.communication.connected_components(
            self.communication_adjacency
        )
        initial_measurements: List[TargetMeasurement] = []
        self.task_allocation.update(
            self.world.uavs,
            self.target_beliefs.beliefs,
            self.target_beliefs.metadata,
            self.components,
        )
        local_knowledge = self._build_knowledge_snapshots()
        actors = self.graph_builder.build_actor(local_knowledge)
        info = self._build_info(
            measurements=initial_measurements,
            collision_events=(),
            lifecycle_events=None,
            new_coverage_cells=0,
            coverage_contributions=np.zeros(self.config.n_uavs),
        )
        self.last_actor_snapshots = actors
        self.last_local_knowledge_snapshots = local_knowledge
        self.last_info = info
        return actors, info

    def _require_world(self) -> WorldState:
        if self.world is None or self.lifecycle is None:
            raise RuntimeError("reset() must be called before step()")
        return self.world

    def _validate_action(self, joint_action: np.ndarray) -> np.ndarray:
        action = np.asarray(joint_action, dtype=np.float64)
        if action.shape != (self.config.n_uavs, 2):
            raise ValueError(
                "joint_action must have shape ({}, 2), got {}".format(
                    self.config.n_uavs, action.shape
                )
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("joint_action contains NaN or Inf")
        return np.clip(action, -1.0, 1.0)

    def _detect_obstacles(self, uav_paths: Mapping[int, SweepPath]) -> None:
        world = self._require_world()
        for uav_id, path in uav_paths.items():
            for obstacle in world.obstacles:
                if (
                    segment_surface_distance(path.start, path.sweep_end, obstacle)
                    <= self.config.obstacle_sense_surface + 1e-12
                ):
                    self.obstacle_knowledge.observe(uav_id, obstacle)

    def _detect_targets(
        self,
        uav_paths: Mapping[int, SweepPath],
        target_paths: Mapping[int, SweepPath],
        control_step: int,
    ) -> Tuple[List[TargetMeasurement], Dict[Tuple[int, int], float]]:
        measurements: List[TargetMeasurement] = []
        minimum_distances: Dict[Tuple[int, int], float] = {}
        for uav_id, uav_path in sorted(uav_paths.items()):
            for target_id, target_path in sorted(target_paths.items()):
                _tau_min, minimum_distance = swept_paths_min_distance(
                    uav_path, target_path
                )
                minimum_distances[(uav_id, target_id)] = minimum_distance
                fine_interval = swept_paths_interval_within_radius(
                    uav_path, target_path, self.config.target_fine_radius
                )
                if fine_interval is not None:
                    kind = MeasurementKind.FINE
                    tau = fine_interval[1]
                    sigma = self.config.fine_position_sigma
                else:
                    coarse_interval = swept_paths_interval_within_radius(
                        uav_path, target_path, self.config.target_coarse_radius
                    )
                    if coarse_interval is None:
                        continue
                    kind = MeasurementKind.COARSE
                    tau = coarse_interval[1]
                    sigma = self.config.coarse_position_sigma
                true_position = target_path.position_at(tau)
                measured_position = true_position + self.rng.normal(0.0, sigma, size=2)
                covariance = np.eye(2, dtype=np.float64) * sigma ** 2
                measurements.append(
                    TargetMeasurement(
                        target_id=target_id,
                        source_uav_id=uav_id,
                        control_step=control_step,
                        tau=float(tau),
                        kind=kind,
                        position=np.asarray(measured_position, dtype=np.float64),
                        covariance=covariance,
                    )
                )
        return measurements, minimum_distances

    def step(
        self, joint_action: np.ndarray
    ) -> Tuple[ActorGraphSnapshot, bool, bool, Dict[str, Any]]:
        world = self._require_world()
        action = self._validate_action(joint_action)
        previous_world = world.clone()
        self.obstacle_knowledge.begin_step()

        proposals = [
            propose_uav_motion(uav, action[uav.uav_id], self.config)
            for uav in previous_world.uavs
        ]
        target_decisions: Dict[int, Tuple[int, float]] = {}
        for target in previous_world.targets:
            proposal, hold_remaining, last_evasion = propose_target_motion(
                target, previous_world.uavs, previous_world.obstacles, self.config
            )
            proposals.append(proposal)
            target_decisions[target.target_id] = (hold_remaining, last_evasion)

        positions, headings, sweep_paths, accepted_collisions, _all_events = (
            resolve_first_collisions(
                proposals=proposals,
                obstacles=previous_world.obstacles,
                entity_radii={
                    EntityKind.UAV: self.config.uav_radius,
                    EntityKind.TARGET: self.config.target_radius,
                },
                map_size=self.config.map_size,
            )
        )

        for uav in world.uavs:
            key = (EntityKind.UAV, uav.uav_id)
            uav.position = positions[key].copy()
            uav.heading = float(headings[key])
            uav.previous_action = float(action[uav.uav_id, 0])
        for target in world.targets:
            key = (EntityKind.TARGET, target.target_id)
            target.position = positions[key].copy()
            target.heading = float(headings[key])
            hold_remaining, last_evasion = target_decisions[target.target_id]
            target.evasion_hold_remaining = int(hold_remaining)
            if hold_remaining > 0:
                # If collision reflected the target, preserve the physically
                # feasible reflected escape direction for the hold phase.
                target.last_evasion_heading = float(headings[key])
            else:
                target.last_evasion_heading = float(last_evasion)
        world.step_count = previous_world.step_count + 1

        self.communication_adjacency = self.communication.build_adjacency(
            [uav.position for uav in world.uavs]
        )
        self.components = self.communication.connected_components(
            self.communication_adjacency
        )

        uav_paths = {
            uav.uav_id: sweep_paths[(EntityKind.UAV, uav.uav_id)]
            for uav in world.uavs
        }
        target_paths = {
            target.target_id: sweep_paths[(EntityKind.TARGET, target.target_id)]
            for target in world.targets
        }
        self._detect_obstacles(uav_paths)
        measurements, minimum_distances = self._detect_targets(
            uav_paths, target_paths, previous_world.step_count
        )
        new_coverage_cells, coverage_contributions, _coverage_masks = (
            self.coverage.update_sweeps(uav_paths)
        )

        self.obstacle_knowledge.synchronize(self.components)
        self.coverage.synchronize(self.components)
        self.target_beliefs.advance_and_synchronize(
            measurements=measurements,
            components=self.components,
            end_time=float(world.step_count),
        )
        self.task_allocation.update(
            world.uavs,
            self.target_beliefs.beliefs,
            self.target_beliefs.metadata,
            self.components,
        )
        lifecycle_events = self.lifecycle.update(measurements)

        terminated = not self._world_is_finite_and_legal()
        truncated = bool(world.step_count >= self.config.max_steps and not terminated)
        local_knowledge = self._build_knowledge_snapshots()
        actors = self.graph_builder.build_actor(local_knowledge)
        info = self._build_info(
            measurements=measurements,
            collision_events=accepted_collisions,
            lifecycle_events=lifecycle_events,
            new_coverage_cells=new_coverage_cells,
            coverage_contributions=coverage_contributions,
        )
        self.last_actor_snapshots = actors
        self.last_local_knowledge_snapshots = local_knowledge
        self.last_info = info
        return actors, terminated, truncated, info

    def _world_is_finite_and_legal(self) -> bool:
        world = self._require_world()
        for entity in list(world.uavs) + list(world.targets):
            if not np.all(np.isfinite(entity.position)) or not np.isfinite(entity.heading):
                return False
            radius = (
                self.config.uav_radius
                if isinstance(entity, UAVState)
                else self.config.target_radius
            )
            if np.any(entity.position < radius - 1e-8) or np.any(
                entity.position > self.config.map_size - radius + 1e-8
            ):
                return False
        return True

    def _build_knowledge_snapshots(
        self,
    ) -> List[LocalKnowledgeSnapshot]:
        world = self._require_world()
        component_lookup = self.communication.component_lookup(
            self.components, self.config.n_uavs
        )
        actors: List[LocalKnowledgeSnapshot] = []
        for uav in world.uavs:
            neighbors = [
                world.uavs[index].clone()
                for index in np.flatnonzero(
                    self.communication_adjacency[uav.uav_id]
                ).tolist()
            ]
            component = self.components[int(component_lookup[uav.uav_id])]
            actors.append(
                LocalKnowledgeSnapshot(
                    ego=uav.clone(),
                    direct_neighbors=neighbors,
                    known_obstacles=[
                        obstacle.clone()
                        for obstacle in sorted(
                            self.obstacle_knowledge.knowledge[uav.uav_id].values(),
                            key=lambda item: item.obstacle_id,
                        )
                    ],
                    active_obstacles=self.obstacle_knowledge.active_obstacles(
                        uav.uav_id, uav.position
                    ),
                    target_beliefs={
                        target_id: belief.clone()
                        for target_id, belief in self.target_beliefs.beliefs[
                            uav.uav_id
                        ].items()
                    },
                    target_metadata=dict(
                        self.target_beliefs.metadata[uav.uav_id]
                    ),
                    coverage_knowledge=self.coverage.local[uav.uav_id].copy(),
                    component_members=tuple(component),
                    observed_obstacle_ids_this_step=frozenset(
                        self.obstacle_knowledge.observed_this_step[uav.uav_id]
                    ),
                    target_assignments=dict(
                        self.task_allocation.metadata[uav.uav_id]
                    ),
                )
            )
        self._assert_no_oracle_leakage(actors)
        return actors

    def _assert_no_oracle_leakage(
        self, actors: Sequence[LocalKnowledgeSnapshot]
    ) -> None:
        for actor in actors:
            legal_ids = set(self.obstacle_knowledge.knowledge[actor.ego.uav_id])
            if any(obstacle.obstacle_id not in legal_ids for obstacle in actor.active_obstacles):
                raise AssertionError("unknown obstacle leaked into Actor knowledge")
            if any(target_id not in self.target_beliefs.beliefs[actor.ego.uav_id] for target_id in actor.target_beliefs):
                raise AssertionError("unknown target leaked into Actor knowledge")

    def _build_info(
        self,
        measurements: Sequence[TargetMeasurement],
        collision_events: Sequence[Any],
        lifecycle_events: Optional[Any],
        new_coverage_cells: int,
        coverage_contributions: np.ndarray,
    ) -> Dict[str, Any]:
        world = self._require_world()
        info: Dict[str, Any] = {
            "step": int(world.step_count),
            "topology_mode": self.topology_mode,
            "scenario_mode": self.scenario_mode,
            "communication_components": [tuple(item) for item in self.components],
            "communication_adjacency": self.communication_adjacency.copy(),
            "coverage_ratio": self.coverage.physical_ratio,
            "new_coverage_cells": int(new_coverage_cells),
            "measurements": [
                {
                    "target_id": measurement.target_id,
                    "source_uav_id": measurement.source_uav_id,
                    "kind": measurement.kind.name,
                    "tau": measurement.tau,
                    "observation_time": measurement.observation_time,
                }
                for measurement in measurements
            ],
            "collision_events": [event.kind.value for event in collision_events],
            "target_states": {
                target_id: record.state.value
                for target_id, record in self.lifecycle.records.items()
            },
            "target_unobserved_counts": {
                target_id: int(record.consecutive_unobserved)
                for target_id, record in self.lifecycle.records.items()
            },
            "target_assignments": [
                {
                    "component": list(assignment.component),
                    "target_id": assignment.target_id,
                    "primary_uav_id": assignment.primary_uav_id,
                    "winning_bid": assignment.winning_bid,
                    "age": assignment.age,
                }
                for assignment in self.task_allocation.assignments.values()
            ],
            "task_handoff_events": [
                {
                    "component": list(event.component),
                    "target_id": event.target_id,
                    "previous_primary_uav_id": event.previous_primary_uav_id,
                    "new_primary_uav_id": event.new_primary_uav_id,
                    "emergency": event.emergency,
                    "reason": event.reason,
                    "previous_observation_level": event.previous_observation_level,
                    "new_observation_level": event.new_observation_level,
                    "previous_predicted_distance": event.previous_predicted_distance,
                    "new_predicted_distance": event.new_predicted_distance,
                }
                for event in self.task_allocation.last_handoff_events
            ],
        }
        if lifecycle_events is not None:
            info["lifecycle_events"] = {
                "first_acquired": sorted(lifecycle_events.first_acquired),
                "first_confirmed": sorted(lifecycle_events.first_confirmed),
                "lost": sorted(lifecycle_events.lost),
                "reacquired": sorted(lifecycle_events.reacquired),
                "currently_tracked": sorted(lifecycle_events.currently_tracked),
            }
        if self.config.include_truth_in_info:
            info["truth"] = {
                "uav_positions": np.stack(
                    [uav.position.copy() for uav in world.uavs], axis=0
                ),
                "target_positions": np.stack(
                    [target.position.copy() for target in world.targets], axis=0
                ),
                "obstacles": [
                    {
                        "obstacle_id": obstacle.obstacle_id,
                        "center": obstacle.center.copy(),
                        "radius": obstacle.radius,
                    }
                    for obstacle in world.obstacles
                ],
            }
        return info

    def close(self) -> None:
        return None
'''),
]


def _install_runtime_modules() -> None:
    for package_name in ("env", "graph_system"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        package.__package__ = package_name
        sys.modules[package_name] = package
    for name, source in EMBEDDED_RUNTIME_MODULES:
        module = types.ModuleType(name)
        module.__file__ = f"<embedded:{name}>"
        module.__package__ = name.rpartition(".")[0]
        sys.modules[name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)


_install_runtime_modules()

from env.graph_search_env import GraphSearchEnv
from env.state_types import EnvConfig
from graph_system.action_assist import (
    apply_predictive_safety_assist,
    apply_primary_continuity_assist,
)
from graph_system.graph_types import ACTOR_SPECS, ActorGraphSnapshot


def _episode_config() -> EnvConfig:
    """Frozen physical and sensing setup of the mainline stage-3 test."""
    config = EnvConfig(
        map_size=250.0,
        dt=1.0,
        max_steps=500,
        n_uavs=4,
        n_obstacles=4,
        num_targets=2,
        max_targets=4,
        curriculum_stage=3,
        uav_speed_min=5.0,
        uav_speed=10.0,
        uav_max_turn_deg=32.5,
        uav_radius=1.0,
        target_speed=5.0,
        target_max_turn_deg=20.0,
        target_radius=1.0,
        comm_radius=83.0,
        target_coarse_radius=30.0,
        target_fine_radius=13.0,
        coverage_radius=10.0,
        obstacle_sense_surface=10.0,
        obstacle_activation_surface=30.0,
        boundary_sense_distance=30.0,
        target_uav_sense_radius=5.0,
        target_obstacle_sense_surface=12.0,
        obstacle_radius_min=2.0,
        obstacle_radius_max=4.0,
        coverage_cell_size=5.0,
        coverage_context_radius=50.0,
        target_patrol_radius_min=25.0,
        target_patrol_radius_max=50.0,
        target_evasion_hold_steps=4,
        target_candidate_turn_step_deg=5.0,
        process_accel_sigma=2.0,
        coarse_position_sigma=3.0,
        fine_position_sigma=0.5,
        belief_decay=0.8,
        belief_lost_age=4,
        target_belief_temporal_memory_enabled=True,
        task_allocation_enabled=True,
        task_allocation_hold_steps=8,
        task_allocation_switch_margin=0.05,
        task_allocation_distance_weight=0.7,
        task_allocation_heading_weight=0.3,
        task_allocation_observation_bonus=0.15,
        task_allocation_emergency_handoff_enabled=True,
        task_allocation_emergency_handoff_horizon_steps=2.0,
        task_allocation_emergency_handoff_margin=4.0,
        task_allocation_emergency_handoff_cooldown_steps=0,
        init_uav_uav_distance=10.0,
        init_uav_target_distance=40.0,
        init_target_target_distance=40.0,
        init_obstacle_surface_clearance=10.0,
        init_entity_obstacle_clearance=15.0,
        init_entity_boundary_clearance=15.0,
        patrol_clearance=12.0,
        init_max_attempts=10_000,
        include_truth_in_info=False,
        preferred_track_distance=10.0,
        target_soft_distance=7.0,
    )
    config.validate()
    return config


def _load_actor(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Actor model not found: {path}")
    # Loading from bytes also supports non-ASCII paths on Windows.
    return torch.export.load(io.BytesIO(path.read_bytes())).module()


@torch.inference_mode()
def _select_action(model, snapshot: ActorGraphSnapshot) -> np.ndarray:
    pieces = [
        getattr(snapshot, name).astype(np.float32, copy=False).reshape(4, -1)
        for name in ACTOR_SPECS
    ]
    features = np.concatenate(pieces, axis=1)
    if features.shape != (4, 158):
        raise ValueError(f"Unexpected Actor observation shape: {features.shape}")
    output = model(torch.from_numpy(features[None].copy()))
    action = np.asarray(output.detach().cpu(), dtype=np.float32)
    if action.shape != (4, 2) or not np.all(np.isfinite(action)):
        raise ValueError(f"Unexpected Actor action: shape={action.shape}")
    return action


def _run_episode(model, seed: int, config: EnvConfig) -> dict[str, float | int]:
    environment = GraphSearchEnv(config)
    actor, info = environment.reset(seed=seed)
    target_count = config.num_targets
    acquired: set[int] = set()
    confirmed: set[int] = set()
    first_acquire: dict[int, int] = {}
    first_confirm: dict[int, int] = {}
    post_acquire_tracked = {target_id: 0 for target_id in range(target_count)}
    current_streak = {target_id: 0 for target_id in range(target_count)}
    longest_streak = {target_id: 0 for target_id in range(target_count)}
    tracked_target_steps = 0
    post_confirm_tracked_steps = 0
    post_confirm_opportunities = 0
    all_tracked_steps = 0
    collision_counts = {
        "uav_obstacle": 0,
        "uav_uav": 0,
        "uav_target": 0,
        "uav_boundary": 0,
    }
    lost_count = 0
    reacquired_count = 0
    steps = 0
    while True:
        action = _select_action(model, actor)
        action, _ = apply_primary_continuity_assist(
            actor,
            action,
            blend=0.30,
            prediction_horizon_steps=1.0,
            fine_radius=config.target_fine_radius,
            coarse_radius=config.target_coarse_radius,
            target_speed=config.target_speed,
            uav_speed_min=config.uav_speed_min,
            uav_speed_max=config.uav_speed,
            max_turn_degrees=config.uav_max_turn_deg,
        )
        action, _ = apply_predictive_safety_assist(
            actor,
            action,
            blend=0.25,
            prediction_horizon_steps=1.0,
            comm_radius=config.comm_radius,
            obstacle_activation_surface=config.obstacle_activation_surface,
            obstacle_radius_max=config.obstacle_radius_max,
            boundary_sense_distance=config.boundary_sense_distance,
            uav_radius=config.uav_radius,
            target_radius=config.target_radius,
            target_speed=config.target_speed,
            uav_speed_min=config.uav_speed_min,
            uav_speed_max=config.uav_speed,
            max_turn_degrees=config.uav_max_turn_deg,
        )
        actor, terminated, truncated, info = environment.step(action)
        steps += 1
        events = info["lifecycle_events"]
        newly_acquired = set(events["first_acquired"])
        newly_confirmed = set(events["first_confirmed"])
        tracked = set(events["currently_tracked"])
        acquired.update(newly_acquired)
        confirmed.update(newly_confirmed)
        for target_id in newly_acquired:
            first_acquire.setdefault(target_id, steps)
        for target_id in newly_confirmed:
            first_confirm.setdefault(target_id, steps)
        tracked_target_steps += len(tracked)
        all_tracked_steps += int(len(tracked) == target_count)
        for target_id, confirmation_step in first_confirm.items():
            if steps >= confirmation_step:
                post_confirm_opportunities += 1
                post_confirm_tracked_steps += int(target_id in tracked)
        for target_id, acquisition_step in first_acquire.items():
            if steps <= acquisition_step:
                continue
            if target_id in tracked:
                post_acquire_tracked[target_id] += 1
                current_streak[target_id] += 1
                longest_streak[target_id] = max(
                    longest_streak[target_id], current_streak[target_id]
                )
            else:
                current_streak[target_id] = 0
        lost_count += len(events["lost"])
        reacquired_count += len(events["reacquired"])
        for event in info["collision_events"]:
            if event in collision_counts:
                collision_counts[event] += 1
        if terminated or truncated:
            break
    environment.close()

    min_opportunity = max(1, int(np.ceil(config.max_steps * 0.20)))
    min_contiguous = max(1, int(np.ceil(config.max_steps * 0.10)))
    adjusted_ratios: list[float] = []
    passed: list[bool] = []
    for target_id in range(target_count):
        acquire_step = first_acquire.get(target_id)
        available = max(steps - acquire_step, 0) if acquire_step is not None else 0
        denominator = max(available, min_opportunity)
        tracked = min(post_acquire_tracked[target_id], available)
        streak = min(longest_streak[target_id], available)
        adjusted_ratios.append(tracked / denominator)
        passed.append(
            available >= min_opportunity
            and tracked / denominator >= 0.30
            and streak >= min_contiguous
        )

    return {
        "seed": seed,
        "targets": target_count,
        "steps": steps,
        "acquired": len(acquired),
        "confirmed": len(confirmed),
        "AcquisitionRate": len(acquired) / target_count,
        "ConfirmationRate": len(confirmed) / target_count,
        "TrackingRatio_full": tracked_target_steps / max(target_count * steps, 1),
        "TrackingRatio_post": post_confirm_tracked_steps / max(post_confirm_opportunities, 1),
        "TrackingRatio_post_acquire_adjusted": float(np.mean(adjusted_ratios)),
        "MultiTargetCoverageRatio": all_tracked_steps / max(steps, 1),
        "coverage_ratio": float(info["coverage_ratio"]),
        "collision_count": sum(collision_counts.values()),
        "obstacle_collision_count": collision_counts["uav_obstacle"],
        "uav_collision_count": collision_counts["uav_uav"],
        "target_contact_count": collision_counts["uav_target"],
        "boundary_event_count": collision_counts["uav_boundary"],
        "lost_count": lost_count,
        "reacquired_count": reacquired_count,
        "first_acquire_time": float(np.mean(list(first_acquire.values()))) if first_acquire else float(steps),
        "first_confirm_time": float(np.mean(list(first_confirm.values()))) if first_confirm else float(steps),
        "mission_success": int(len(confirmed) == target_count and all(passed)),
        "terminated": int(terminated),
        "truncated": int(truncated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().with_name("actor_deterministic.pt2"),
        help="path to the frozen Actor model",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=260001)
    parser.add_argument("--output", type=Path, help="optional output JSON path")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    actor_model = _load_actor(args.model)
    config = _episode_config()
    rows = [
        _run_episode(actor_model, args.seed_start + index, config)
        for index in range(args.episodes)
    ]
    means = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
        if key not in ("seed", "targets", "terminated", "truncated")
    }
    report = {
        "protocol": "deterministic_actor_stage3_2targets_500steps",
        "model": args.model.name,
        "episodes": rows,
        "means": means,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Saved {len(rows)} complete episode(s) to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
