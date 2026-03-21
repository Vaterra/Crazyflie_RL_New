# evaluate_policies.py
import os
import json
import numpy as np
from stable_baselines3 import PPO

from envs.environment import base_aviary
from config.eval_scenarios import EVAL_SCENARIOS
from policies.scripted_policies import ScriptedChaserPolicy, ScriptedEvaderPolicy


class SB3PolicyWrapper:
    def __init__(self, model_path, device="cpu"):
        self.model = PPO.load(model_path, device=device)

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)


def run_episode(env, learner_policy, max_steps=2000):
    obs, info = env.reset(seed=12345)
    total_reward = 0.0

    for _ in range(max_steps):
        action, _ = learner_policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return {
        "return": float(total_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        **info,
    }


def evaluate_matchup(
    controlled_agent,
    learner_policy,
    opponent_policy,
    scenarios,
    gui=False,
):

    results = []

    env = base_aviary(
        controlled_agent=controlled_agent,
        gui=gui,
        scenario=scenarios[3],
    )
    env.set_opponent_policy(opponent_policy)

    try:
        for scenario in scenarios:
            env.set_scenario(scenario)
            ep_result = run_episode(env, learner_policy)
            ep_result["scenario"] = scenario.name
            results.append(ep_result)
    finally:
        env.close()

    return results


def summarize(results):
    summary = {
        "n": len(results),
        "goal_rate": np.mean([r.get("evader_reached_goal", False) for r in results]),
        "capture_rate": np.mean([r.get("captured", False) for r in results]),
        "timeout_rate": np.mean([r.get("timeout", False) for r in results]),
        "evader_collision_rate": np.mean([r.get("evader_collision", False) for r in results]),
        "chaser_collision_rate": np.mean([r.get("chaser_collision", False) for r in results]),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_goal_distance": float(np.mean([r.get("goal_distance", 0.0) for r in results])),
        "mean_capture_distance": float(np.mean([r.get("distance", 0.0) for r in results])),
    }
    return summary


if __name__ == "__main__":
    # Example 1: evaluate an evader policy against scripted chaser
    # evader_model_path = "models/version_1_three_pillars/Evader.zip"
    # evader_policy = SB3PolicyWrapper(evader_model_path, device="cpu")
    # chaser_opponent = ScriptedChaserPolicy(speed=1.0)

    # evader_results = evaluate_matchup(
    #     controlled_agent=base_aviary.AGENT_EVADER,
    #     learner_policy=evader_policy,
    #     opponent_policy=chaser_opponent,
    #     scenarios=EVAL_SCENARIOS,
    #     gui=True,
    # )

    # print("\nEvader per-scenario results:")
    # for r in evader_results:
    #     print(r)

    # print("\nEvader summary:")
    # print(summarize(evader_results))

    # Example 2: evaluate a chaser policy against scripted evader
    chaser_model_path = "models/version_1_three_pillars/chaser_seed_43_2026-03-21_19-10.zip"
    chaser_policy = SB3PolicyWrapper(chaser_model_path, device="cpu")
    evader_opponent = ScriptedEvaderPolicy(speed=1.0, w_goal=0.7, w_away=0.3)

    chaser_results = evaluate_matchup(
        controlled_agent=base_aviary.AGENT_CHASER,
        learner_policy=chaser_policy,
        opponent_policy=evader_opponent,
        scenarios=EVAL_SCENARIOS,
        gui=True,
    )

    print("\nChaser per-scenario results:")
    for r in chaser_results:
        print(r)

    print("\nChaser summary:")
    print(summarize(chaser_results))