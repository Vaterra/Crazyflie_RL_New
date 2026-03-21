# config/eval_scenarios.py
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

@dataclass
class EvalScenario:
    name: str
    evader_pos: np.ndarray
    chaser_pos: np.ndarray
    goal_pos: np.ndarray
    obstacles: List[Dict] = field(default_factory=list)

EVAL_SCENARIOS = [
    EvalScenario(
        name="open_field_easy",
        evader_pos=np.array([0.0, -1.6, 1.0], dtype=np.float32),
        chaser_pos=np.array([-1.8, 0.0, 1.0], dtype=np.float32),
        goal_pos=np.array([0.0, 1.6, 1.0], dtype=np.float32),
        obstacles=[],
    ),

    EvalScenario(
        name="single_wall_middle",
        evader_pos=np.array([0.0, -1.6, 1.0], dtype=np.float32),
        chaser_pos=np.array([-1.8, 0.0, 1.0], dtype=np.float32),
        goal_pos=np.array([0.0, 1.6, 1.0], dtype=np.float32),
        obstacles=[
            {
                "shape": "box",
                "half_extents": [0.15, 0.7, 1],
                "position": [0.0, 0.0, 0.5],
                "rgba": [0.8, 0.2, 0.2, 1.0],
            }
        ],
    ),

    EvalScenario(
        name="three_pillars_corridor",
        evader_pos=np.array([0.0, -1.8, 1.0], dtype=np.float32),
        chaser_pos=np.array([-1.7, -0.3, 1.0], dtype=np.float32),
        goal_pos=np.array([0.0, 1.8, 1.0], dtype=np.float32),
        obstacles=[
            {
                "shape": "cylinder",
                "radius": 0.25,
                "height": 3,
                "position": [-0.6, 0.0, 0.8],
                "rgba": [0.2, 0.2, 0.8, 1.0],
            },
            {
                "shape": "cylinder",
                "radius": 0.25,
                "height": 3,
                "position": [0.6, 0.0, 0.8],
                "rgba": [0.2, 0.2, 0.8, 1.0],
            },
            {
                "shape": "cylinder",
                "radius": 0.25,
                "height": 3,
                "position": [0, 1, 0.8],
                "rgba": [0.2, 0.2, 0.8, 1.0],
            },
        ],
    ),
    EvalScenario(
        name="Double_diamond",
        evader_pos=np.array([0.0, -1.8, 1.0], dtype=np.float32),
        chaser_pos=np.array([-1.7, -0.3, 1.0], dtype=np.float32),
        goal_pos=np.array([0.0, 1.8, 1.0], dtype=np.float32),
        obstacles=[
            {
                "shape": "cylinder",
                "radius": 0.25,
                "height": 3,
                "position": [-0.6, 0.0, 0.8],
                "rgba": [0.2, 0.2, 0.8, 1.0],
            },
            {
                "shape": "cylinder",
                "radius": 0.25,
                "height": 3,
                "position": [0.6, 0.0, 0.8],
                "rgba": [0.2, 0.2, 0.8, 1.0],
            },
            {
                "shape": "cylinder",
                "radius": 0.25,
                "height": 3,
                "position": [0, 1, 0.8],
                "rgba": [0.2, 0.2, 0.8, 1.0],
            },
            {
                "shape": "cylinder",
                "radius": 0.25,
                "height": 3,
                "position": [0, -1, 0.8],
                "rgba": [0.2, 0.2, 0.8, 1.0],
            },
        ],
    ),

    EvalScenario(
        name="maze_like",
        evader_pos=np.array([0.0, -1.7, 1.0], dtype=np.float32),
        chaser_pos=np.array([1.7, -0.2, 1.0], dtype=np.float32),
        goal_pos=np.array([0.0, 1.7, 1.0], dtype=np.float32),
        obstacles=[
            {
                "shape": "box",
                "half_extents": [0.8, 0.10, 1.5],
                "position": [0.0, -0.7, 0.5],
                "rgba": [0.7, 0.7, 0.2, 1.0],
            },
            {
                "shape": "box",
                "half_extents": [2, 0.10, 1.5],
                "position": [-1, 0.7, 0.5],
                "rgba": [0.7, 0.7, 0.2, 1.0],
            },
            {
                "shape": "box",
                "half_extents": [0.12, 0.8, 0.5],
                "position": [-0.9, 0.0, 0.5],
                "rgba": [0.7, 0.7, 0.2, 1.0],
            },
        ],
    ),
]