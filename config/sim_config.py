from dataclasses import dataclass, field

from gym_pybullet_drones.utils.enums import (
    DroneModel,
    Physics,
    ActionType,
    ObservationType,
)


@dataclass
class EnvConfig:
    # Base simulator settings
    drone_model: DroneModel = DroneModel.CF2X
    physics: Physics = Physics.PYB
    pyb_freq: int = 240
    ctrl_freq: int = 240
    gui: bool = False
    record: bool = False
    obs: ObservationType = ObservationType.KIN
    act: ActionType = ActionType.VEL

    # Environment parameters
    episode_len_sec: float = 30.0
    capture_radius: float = 0.25
    goal_radius: float = 0.35
    arena_xy: float = 3.0
    arena_z_min: float = 0.1
    arena_z_max: float = 3.0
    opponent_speed: float = 1.0
    speed_limit: float = 1.0

    # Ray sensor parameters
    use_raycast: bool = True
    if use_raycast:
        max_range: float = 5.0
        vizualise: bool = False

    fixed_spawns: bool = False
    
    # Scenario to train
    #scenario = "single_wall_middle"
    scenario: str = "three_pillars_corridor"
    #scenario = "Double_diamond"
    #scenario = "maze_like"
    #scenario = "open_field_easy"
