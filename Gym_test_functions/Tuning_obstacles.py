import time
import numpy as np
import pybullet as p

from envs.environment import base_aviary
from config.eval_scenarios import EVAL_SCENARIOS
from config.sim_config import EnvConfig


class ZeroPolicy:
    """Keeps the required opponent policy interface satisfied."""
    def predict(self, obs, deterministic=True):
        return np.zeros(4, dtype=np.float32), None


class ScenarioCompat:
    """
    Wrap your EvalScenario dataclass so the current environment.reset()
    can use both attribute access and dict-like access.
    """
    def __init__(self, scenario):
        self.name = scenario.name
        self.evader_pos = np.array(scenario.evader_pos, dtype=np.float32)
        self.chaser_pos = np.array(scenario.chaser_pos, dtype=np.float32)
        self.goal_pos = np.array(scenario.goal_pos, dtype=np.float32)
        self.obstacles = list(scenario.obstacles)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


class ScenarioViewer:
    def __init__(self):
        self.env = base_aviary(
            controlled_agent=base_aviary.AGENT_EVADER,
            gui=True,
            scenario=None,
        )
        self.env.set_opponent_policy(ZeroPolicy())
        self.index = 0
        self.esc_key = getattr(p, "B3G_ESCAPE", None)

    def _key_triggered(self, keys, ch: str) -> bool:
        lower = ord(ch.lower())
        upper = ord(ch.upper())
        return (
            (lower in keys and keys[lower] & p.KEY_WAS_TRIGGERED)
            or (upper in keys and keys[upper] & p.KEY_WAS_TRIGGERED)
        )

    def _esc_triggered(self, keys) -> bool:
        return (
            self.esc_key is not None
            and self.esc_key in keys
            and keys[self.esc_key] & p.KEY_WAS_TRIGGERED
        )

    def _print_scene(self, scenario):
        print("\n" + "=" * 80)
        print(f"Scenario: {scenario.name}")
        print(f"Evader spawn: {scenario.evader_pos.tolist()}")
        print(f"Chaser spawn: {scenario.chaser_pos.tolist()}")
        print(f"Goal:         {scenario.goal_pos.tolist()}")
        print(f"Obstacle count: {len(scenario.obstacles)}")
        for i, spec in enumerate(scenario.obstacles):
            print(f"  [{i}] {spec}")
        print(f"Spawned obstacle ids: {getattr(self.env, 'obstacle_ids', [])}")
        for oid in getattr(self.env, "obstacle_ids", []):
            try:
                pos, orn = p.getBasePositionAndOrientation(oid, physicsClientId=self.env.CLIENT)
                pos = tuple(round(v, 3) for v in pos)
                orn = tuple(round(v, 3) for v in orn)
                print(f"    body {oid}: pos={pos} orn={orn}")
            except Exception as e:
                print(f"    body {oid}: unavailable ({e})")
        print("Controls: N=next, P=previous, R=reload, Q=quit")
        if self.esc_key is not None:
            print("ESC is also supported on this PyBullet build")
        print("=" * 80)

    def load_current(self):
        raw = EVAL_SCENARIOS[self.index]
        compat = ScenarioCompat(raw)
        self.env.set_scenario(compat)
        self.env.reset()

        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0.0, 0.0, 0.8],
            physicsClientId=self.env.CLIENT,
        )

        self._print_scene(raw)

    def run(self):
        self.load_current()
        last_key_time = 0.0

        try:
            while True:
                keys = p.getKeyboardEvents(physicsClientId=self.env.CLIENT)
                now = time.time()

                if now - last_key_time > 0.2:
                    if self._key_triggered(keys, 'n'):
                        self.index = (self.index + 1) % len(EVAL_SCENARIOS)
                        self.load_current()
                        last_key_time = now
                    elif self._key_triggered(keys, 'p'):
                        self.index = (self.index - 1) % len(EVAL_SCENARIOS)
                        self.load_current()
                        last_key_time = now
                    elif self._key_triggered(keys, 'r'):
                        self.load_current()
                        last_key_time = now
                    elif self._key_triggered(keys, 'q') or self._esc_triggered(keys):
                        break

                time.sleep(1.0 / 60.0)
        finally:
            self.env.close()


if __name__ == "__main__":
    ScenarioViewer().run()
