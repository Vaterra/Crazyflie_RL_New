import numpy as np
import pybullet as p


def RaySensor(drone_id, client_id=0, max_range=5.0, visualize=False):
    """
    Cast 4 rays 90 degrees apart from the drone in its local XY plane.

    Returns:
        np.ndarray of shape (4,) with distances in meters.
        If a ray hits nothing, its distance is max_range.
    """

    # Get drone world position and orientation
    pos, orn = p.getBasePositionAndOrientation(drone_id, physicsClientId=client_id)
    pos = np.array(pos, dtype=np.float32)

    # Rotation matrix from drone local frame -> world frame
    rot = np.array(
        p.getMatrixFromQuaternion(orn, physicsClientId=client_id),
        dtype=np.float32
    ).reshape(3, 3)

    # 4 local ray directions: forward, left, back, right
    local_dirs = np.array([
        [ 1.0,  0.0, 0.0],   # forward
        [ 0.0,  1.0, 0.0],   # left
        [-1.0,  0.0, 0.0],   # back
        [ 0.0, -1.0, 0.0],   # right
    ], dtype=np.float32)

    # Rotate local directions into world frame
    world_dirs = (rot @ local_dirs.T).T

    # Start rays slightly outside the drone body to avoid self-hit
    ray_start_offset = 0.1
    ray_length = max_range - ray_start_offset

    ray_from = pos[None, :] + ray_start_offset * world_dirs
    ray_to = ray_from + ray_length * world_dirs

    # Batch raycast
    results = p.rayTestBatch(
        ray_from.tolist(),
        ray_to.tolist(),
        physicsClientId=client_id
    )

    distances = np.empty(4, dtype=np.float32)

    for i, r in enumerate(results):
        hit_body_uid = r[0]
        hit_fraction = r[2]
        hit_position = np.array(r[3], dtype=np.float32)

        if hit_body_uid == -1:
            # No hit within range
            distances[i] = max_range
            end_pt = ray_to[i]
            color = [0, 1, 0]
        else:
            # Hit something
            dist = ray_start_offset + hit_fraction * ray_length
            distances[i] = min(dist, max_range)
            end_pt = hit_position
            color = [1, 0, 0]

        if visualize:
            p.addUserDebugLine(
                ray_from[i].tolist(),
                end_pt.tolist(),
                color,
                lifeTime=0.05,
                physicsClientId=client_id
            )

    return distances