import numpy as np

def compute_evader_reward(
    goal_dist: float,
    prev_goal_dist: float | None,
    Evader_pos: float,
    Chaser_pos: float,
    info: dict,
) -> float:
    reward = 0.0
    E_2_C_pos = np.linalg.norm(Evader_pos - Chaser_pos)

    if prev_goal_dist is not None:
        reward += 10 * (prev_goal_dist - goal_dist)

    if E_2_C_pos < 1.0: # What distance is "unsafe"
        reward += -0.1 * (E_2_C_pos)

    if info["evader_reached_goal"]:
        reward += 100

    if info["captured"]:
        reward += -100

    if info["evader_out"]:
        reward += -100

    return float(reward)

def compute_chaser_reward(
    E_2_C_distance: float,
    prev_E_2_C_distance: float | None,
    info: dict,
    cfg: RewardConfig,
) -> float:
    reward = 0.0

    if prev_E_2_C_distance is not None:
        reward += 5.0 * (prev_E_2_C_distance - E_2_C_distance)

    if info["captured"]:
        reward += 100

    if info["evader_reached_goal"]:
        reward += -100

    if info["chaser_out"]:
        reward += -100

    return float(reward)