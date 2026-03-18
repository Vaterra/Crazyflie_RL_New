import numpy as np
import pybullet as p

from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType

from config.Carrot_Stick  import compute_evader_reward, compute_chaser_reward
from config.sim_config import EnvConfig

####################################
from utils.RayCast import RaySensor
####################################


## Debug RAM
import psutil 


class base_aviary(BaseRLAviary):
    """
    Two-drone pursuit-evasion game.

    Drone indexing:
    - drone 0 = evader
    - drone 1 = chaser

    Single-agent RL interface:
    - controlled_agent = "evader": learner controls evader
    - controlled_agent = "chaser": learner controls chaser

    The other drone is controlled by an injected opponent policy.
    """

    AGENT_EVADER = "evader"
    AGENT_CHASER = "chaser"
    config: EnvConfig

    def __init__(
        self,
        controlled_agent: str,
        gui: bool = False,
    ):

        #############
        # Init
        #############
        self.controlled_agent = controlled_agent
        self.step_info = None
        self.opponent_policy = None
        self.opponent_pool = None

        self.termination_stats = {
            "goal": 0,
            "captured": 0,
            "evader_out": 0,
            "chaser_out": 0,
            "timeout": 0,
        }    
        super().__init__(
            drone_model=self.config.drone_model,
            num_drones=2,
            physics=self.config.physics,
            pyb_freq=self.config.pyb_freq,
            ctrl_freq=self.config.ctrl_freq,
            gui=gui,
            record=self.config.record,
            obs=self.config.obs,
            act=self.config.act,
        )



    # ---------------------------------------------------------------------
    # Reset / step
    # ---------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        if self.opponent_pool is not None:
            self._sample_opponent()
        else:
            raise ValueError("No opponent policy or opponent pool has been set")

        self.goal_pos = self._sample_goal()
        self.Drones_init = self._sample_initial_positions()

        obs, info = super().reset(seed=seed, options=options)

        #self._create_arena_walls()
        #self._draw_goal_marker()

        evader_pos = self._pos(0)
        chaser_pos = self._pos(1)

        self.step_info = self._compute_step_info()

        #self.prev_goal_dist = float(np.linalg.norm(self.goal_pos - evader_pos))
        #self.prev_capture_dist = float(np.linalg.norm(chaser_pos - evader_pos))

        return obs, info

    def step(self, action):
        joint_action = self._build_single_agent_action(action)
        obs, reward, terminated, truncated, info = super().step(joint_action)
        #Debug
        #print(f"Available memory: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total:.2f}%")
        self.step_info = self._compute_step_info()

        if terminated or truncated:
            if info["evader_reached_goal"]:
                self.termination_stats["goal"] += 1
            elif info["captured"]:
                self.termination_stats["captured"] += 1
            elif info["evader_out"]:
                self.termination_stats["evader_out"] += 1
            elif info["chaser_out"]:
                self.termination_stats["chaser_out"] += 1
            elif info["timeout"]:
                self.termination_stats["timeout"] += 1

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Something
    # ---------------------------------------------------------------------

    def _get_agent_obs(self, agent: str) -> np.ndarray:
        evader_pos = self._pos(0)*2/self.arena_xy
        evader_vel = self._vel(0)
        chaser_pos = self._pos(1)*2/self.arena_xy
        chaser_vel = self._vel(1)
        rel = chaser_pos - evader_pos

        if agent == self.AGENT_EVADER:
            ray_obs = self._get_ray_obs(0)
            return np.concatenate([
                evader_pos,
                evader_vel/self.SPEED_LIMIT,
                rel,
                self.goal_pos*2/self.arena_xy,
                ray_obs,
            ]).astype(np.float32)

        if agent == self.AGENT_CHASER:
            ray_obs = self._get_ray_obs(1)
            return np.concatenate([
                chaser_pos,
                chaser_vel/self.SPEED_LIMIT,
                evader_pos,
                evader_vel/self.opponent_speed,
                rel,
                self.goal_pos*2/self.arena_xy,
                ray_obs,
            ]).astype(np.float32)

        raise ValueError(f"Unknown agent={agent}")


    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _pos(self, drone_id: int) -> np.ndarray:
        return self._getDroneStateVector(drone_id)[0:3].copy()

    def _vel(self, drone_id: int) -> np.ndarray:
        return self._getDroneStateVector(drone_id)[10:13].copy()

    def _sample_initial_positions(self) -> np.ndarray:
        rng = self.rng

        evader_pos = np.array([
            rng.uniform(-self.goal_radius, self.goal_radius),
            rng.uniform(-self.arena_xy + 2 * self.goal_radius, -self.arena_xy + 4 * self.goal_radius),
            rng.uniform(self.goal_radius * 2, self.arena_z_max - self.goal_radius * 2),
        ], dtype=np.float32)

        chaser_pos = np.array([
            rng.choice([
                rng.uniform(-self.arena_xy, -self.arena_xy + self.goal_radius),
                rng.uniform(self.arena_xy - self.goal_radius, self.arena_xy),
            ]),
            rng.uniform(-self.goal_radius, self.goal_radius),
            rng.uniform(self.arena_z_min, self.arena_z_max),
        ], dtype=np.float32)

        return np.vstack([evader_pos, chaser_pos])

    def _sample_goal(self) -> np.ndarray:
        rng = self.rng

        return np.array([
            rng.uniform(-self.goal_radius, self.goal_radius),
            rng.uniform(self.arena_xy - 3 * self.goal_radius, self.arena_xy - self.goal_radius),
            rng.uniform(self.arena_z_min, self.arena_z_max),
        ], dtype=np.float32)

    def _is_out(self, pos: np.ndarray) -> bool:
        return (
            abs(pos[0]) > self.arena_xy
            or abs(pos[1]) > self.arena_xy
            or pos[2] < self.arena_z_min
            or pos[2] > self.arena_z_max
        )
    # ---------------------------------------------------------------------
    # RL API hooks
    # ---------------------------------------------------------------------
    def _computeObs(self):
        return self._get_agent_obs(self.controlled_agent)

    def _computeInfo(self):
        evader_pos = self._pos(0)
        chaser_pos = self._pos(1)

        capture_dist = float(np.linalg.norm(chaser_pos - evader_pos))
        goal_dist = float(np.linalg.norm(self.goal_pos - evader_pos))

        captured = capture_dist <= self.capture_radius
        evader_reached_goal = goal_dist <= self.goal_radius
        evader_out = self._is_out(evader_pos)
        chaser_out = self._is_out(chaser_pos)
        timeout = (self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC

        return {
            "captured": captured,
            "evader_reached_goal": evader_reached_goal,
            "evader_out": evader_out,
            "chaser_out": chaser_out,
            "timeout": timeout,
            "distance": capture_dist,
            "goal_distance": goal_dist,
        }

    def _computeTerminated(self):
        info = self._computeInfo()
        return (
            info["captured"]
            or info["evader_reached_goal"]
            or info["evader_out"]
            or info["chaser_out"]
        )  ## Maybe change the termination depending on which agent we are training

    def _computeTruncated(self):
        info = self.step_info
        return info["timeout"]

    def _computeReward(self):
        info = self.step_info

        evader_pos = self._pos(0)
        chaser_pos = self._pos(1)

        goal_dist = float(np.linalg.norm(self.goal_pos - evader_pos))
        capture_dist = info["distance"]

        prev_goal_dist = self.prev_goal_dist
        prev_capture_dist = self.prev_capture_dist

        if self.controlled_agent == self.AGENT_EVADER:
            reward = compute_evader_reward(
                goal_dist=goal_dist,
                prev_goal_dist=prev_goal_dist,
                Evader_pos=evader_pos,
                Chaser_pos=chaser_pos,
                info=info,
            )
        elif self.controlled_agent == self.AGENT_CHASER:
            reward = compute_chaser_reward(
                E_2_C_distance=capture_dist,
                prev_E_2_C_distance=prev_capture_dist,
                info=info,
            )
        else:
            raise ValueError(f"Unknown controlled_agent={self.controlled_agent}")

        self.prev_goal_dist = goal_dist
        self.prev_capture_dist = capture_dist
        return reward

    # ---------------------------------------------------------------------
    # Box
    # ---------------------------------------------------------------------
    def _create_box(self, half_extents, position, rgba):
        col_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.CLIENT
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
            physicsClientId=self.CLIENT
        )
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            physicsClientId=self.CLIENT
        )
        return body_id

    def _create_arena_walls(self):
        # Remove previous walls if they exist
        if self.ray_visualize:
            for wid in getattr(self, "wall_ids", []):
                try:
                    p.removeBody(wid, physicsClientId=self.CLIENT)
                except Exception:
                    pass
        self.wall_ids = []

        wall_thickness = 0.02
        z_center = 0.5 * (self.arena_z_min + self.arena_z_max)
        wall_height = 0.5 * (self.arena_z_max - self.arena_z_min)

        rgba = [0.7, 0.7, 0.7, 0.15]

        # +X wall
        self.wall_ids.append(self._create_box(
            half_extents=[wall_thickness, self.arena_xy, wall_height],
            position=[ self.arena_xy + wall_thickness, 0.0, z_center],
            rgba=rgba
        ))

        # -X wall
        self.wall_ids.append(self._create_box(
            half_extents=[wall_thickness, self.arena_xy, wall_height],
            position=[-self.arena_xy - wall_thickness, 0.0, z_center],
            rgba=rgba
        ))

        # +Y wall
        self.wall_ids.append(self._create_box(
            half_extents=[self.arena_xy, wall_thickness, wall_height],
            position=[0.0,  self.arena_xy + wall_thickness, z_center],
            rgba=rgba
        ))

        # -Y wall
        self.wall_ids.append(self._create_box(
            half_extents=[self.arena_xy, wall_thickness, wall_height],
            position=[0.0, -self.arena_xy - wall_thickness, z_center],
            rgba=rgba
        ))

        # Floor
        self.wall_ids.append(self._create_box(
            half_extents=[self.arena_xy, self.arena_xy, wall_thickness],
            position=[0.0, 0.0, self.arena_z_min - wall_thickness],
            rgba=[0.5, 0.5, 0.5, 0.1]
        ))

        # Ceiling (optional)
        self.wall_ids.append(self._create_box(
            half_extents=[self.arena_xy, self.arena_xy, wall_thickness],
            position=[0.0, 0.0, self.arena_z_max + wall_thickness],
            rgba=[0.5, 0.5, 0.5, 0.1]
        ))