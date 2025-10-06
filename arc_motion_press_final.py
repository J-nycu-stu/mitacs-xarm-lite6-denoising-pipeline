#!/usr/bin/env python3
"""
Lite-6 circular arc in the XZ-plane, planned with MoveIt.

WHAT THIS DOES
• The TCP follows a circular arc in the XZ plane (Y held constant).
• The circle is defined EXPLICITLY by its CENTER (cx,cz) and RADIUS (r).
• Start angle = 0° at the geometric TOP (x=cx, z=cz+r); you can offset it by START_ANGLE_OFFSET_DEG.
• Move along a configurable sweep angle (MOVE_ANGLE_DEG) in CW/CCW (DIRECTION_SIGN).
• At EVERY forward stop: arrive (up) → press-in (TOWARD THE CENTER) → dwell → press-out (lift) → move on.
• After the last forward stop, return along the same arc (no pressing).

CONVENTIONS
• Angles: 0° = top; +CCW when looking from +Y toward the center.
• Units: mm for positions; degrees for orientations.
• Orientation: roll=-180°, yaw=0°, pitch always faces the circle center.
"""

import math, rclpy, time
from rclpy.node import Node
from geometry_msgs.msg import Pose
from xarm_msgs.srv import PlanPose, PlanExec

# tqdm progress bar (falls back to no-op if not installed)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs): return iterable

# ======================= USER-DEFINED PARAMETERS =======================
# --- Circle geometry (MANDATORY) ---
PATH_CENTER_XZ_MM = (200.0, 376.5)   # (cx, cz) in mm
PATH_RADIUS_MM    = 71.5             # TCP path radius in mm (set to your measured value)
PATH_Y_MM         = 0.0              # keep Y constant along the path

# --- Start-angle offset around the circle ---
# Positive = CCW (left), Negative = CW (right)
START_ANGLE_OFFSET_DEG = 10.0
MAX_START_OFFSET_DEG   = 45.0   # safety clamp

# --- Sweep (moving) angle ---
MOVE_ANGLE_DEG = 60.0           # e.g., 60, 90, 120
MAX_SWEEP_DEG  = 170.0
MIN_SWEEP_DEG  = 5.0

# --- Direction sign ---
# +1 = CCW, -1 = CW (viewed from +Y toward center)
DIRECTION_SIGN = 1

# --- Base orientation (kept fixed; pitch is auto to face center) ---
ROLL_DEG = -180.0
YAW_DEG  = 0.0

# --- Waypoints/timing ---
# N_SAMPLES = 61                    # forward press stops (>=3)
N_SAMPLES = 5 
# DWELL_SEC = 25                  # press dwell seconds at each stop
DWELL_SEC = 0 

# --- Return path stops (no pressing) ---
N_RETURN_STOPS = 3               # set 1 to go straight back; up to N_SAMPLES

# --- Press parameters ---
PRESS_IN_MM     = 9.0
PRESS_MODE      = 'toward_point'         # ensure pressing toward center
PRESS_POINT_MM  = [0.0, 0.0, 0.0]        # will be overwritten to [cx, y, cz] at runtime
PRESS_WORLD_VEC = [0.0, 0.0, -1.0]       # unused in 'toward_point'
TOOL_Z_SIGN     = 1.0                    # unused in 'toward_point'
SETTLE_AFTER_PRESS_SEC = 0.0

# First-move fallback: tiny approach step along the arc(prevent from getting "planning failed" because of zero move for initial position)
FIRST_MOVE_APPROACH_FRAC = 0.01  # 1% of the sweep
# ==========================================================================

# q = qz * qy * qx (intrinsic ZYX RPY → quaternion: x,y,z,w)
def rpy_to_quat(r, p, y):
    r, p, y = map(math.radians, (r, p, y))
    cy, sy = math.cos(y/2), math.sin(y/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cr, sr = math.cos(r/2), math.sin(r/2)
    return (
        sr*cp*cy - cr*sp*sy,  # x
        cr*sp*cy + sr*cp*sy,  # y
        cr*cp*sy - sr*sp*cy,  # z
        cr*cp*cy + sr*cp*sy,  # w
    )

def wrap_pi(a):
    """Wrap angle to (-pi, pi]."""
    a = (a + math.pi) % (2*math.pi) - math.pi
    return a if a != -math.pi else math.pi

def pitch_facing_center(x, z, cx, cz):
    """With roll=-180, yaw=0, make pitch face the center (cx,cz) in XZ."""
    ang = math.degrees(math.atan2(x - cx, z - cz))
    if ang > 180: ang -= 360
    if ang == 180: ang = -180
    return ang

def unit(v):
    x,y,z = v
    n = math.sqrt(x*x + y*y + z*z)
    return (0.0,0.0,0.0) if n == 0 else (x/n, y/n, z/n)

def tool_z_axis_world(qx,qy,qz,qw):
    """Tool +Z in world frame from quaternion (x,y,z,w)."""
    tx = 2*(qx*qz + qw*qy)
    ty = 2*(qy*qz - qw*qx)
    tz = 1 - 2*(qx*qx + qy*qy)
    return (tx, ty, tz)


class ArcDemo(Node):
    ERR = {
        1:   'success',
        -31: 'no IK solution (joint limits / reach)',
        -10: 'goal in collision',
        -5:  'invalid motion plan',
        -2:  'planning failed',
    }

    def __init__(self):
        super().__init__('lite6_arc_demo')
        self.plan = self.create_client(PlanPose,  '/xarm_pose_plan')
        self.exec = self.create_client(PlanExec, '/xarm_exec_plan')

    def mm_pose(self, xyz_mm, rpy_deg):
        """Helper: pose builder with mm positions and RPY(deg)."""
        pose = Pose()
        pose.position.x = xyz_mm[0] / 1000.0
        pose.position.y = xyz_mm[1] / 1000.0
        pose.position.z = xyz_mm[2] / 1000.0
        pose.orientation.x, pose.orientation.y, \
        pose.orientation.z, pose.orientation.w = rpy_to_quat(*rpy_deg)
        return pose

    # ---- helpers ----
    def plan_exec(self, pose: Pose, label: str) -> bool:
        """Plan to pose then execute, with consistent logging."""
        if not self.call(PlanPose.Request(target=pose), f'plan {label}'):
            return False
        if not self.call(PlanExec.Request(wait=True),   f'exec {label}'):
            return False
        return True

    def make_arc_pose(self, th, y_const, cx0, cz0, r, roll, yaw) -> Pose:
        """Build a pose at angle th on the circle (center cx0,cz0; radius r)."""
        x = cx0 + r*math.sin(th)
        z = cz0 + r*math.cos(th)
        pitch = pitch_facing_center(x, z, cx0, cz0)
        return self.mm_pose([x, y_const, z], [roll, pitch, yaw])

    # -------- plan+exec low-level (blocking) --------
    def call(self, request, label):
        cli = self.plan if isinstance(request, PlanPose.Request) else self.exec
        fut = cli.call_async(request)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        ok   = getattr(res, 'success', False)
        code = getattr(res, 'ret', None)
        if code is None and isinstance(getattr(res, 'success', None), bool):
            ok = res.success
            code = 1 if ok else -2
        if not ok:
            msg = self.ERR.get(code, f'unknown error {code}')
            self.get_logger().error(f'{label} failed [{code}]: {msg}')
        return ok

    # -------- pressing logic: press-in → dwell → press-out (lift) --------
    def press_here(self, pose: Pose, label: str):
        if PRESS_IN_MM <= 0.0 or DWELL_SEC < 0.0:
            self.get_logger().warn('Pressing is disabled (PRESS_IN_MM <= 0 or DWELL_SEC < 0).')
            return True

        # Current waypoint in mm + quat
        x = pose.position.x * 1000.0
        y = pose.position.y * 1000.0
        z = pose.position.z * 1000.0
        qx,qy,qz,qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w

        # Direction for pressing
        if PRESS_MODE == 'toward_point':
            dx = PRESS_POINT_MM[0] - x
            dy = PRESS_POINT_MM[1] - y
            dz = PRESS_POINT_MM[2] - z
            ux,uy,uz = unit((dx,dy,dz))
        elif PRESS_MODE == 'world_vec':
            ux,uy,uz = unit(PRESS_WORLD_VEC)
        elif PRESS_MODE == 'tool_z':
            tz = tool_z_axis_world(qx,qy,qz,qw)
            ux,uy,uz = unit((TOOL_Z_SIGN*tz[0], TOOL_Z_SIGN*tz[1], TOOL_Z_SIGN*tz[2]))
        else:
            self.get_logger().error(f'Unknown PRESS_MODE={PRESS_MODE}; cannot press.')
            return False

        if ux == uy == uz == 0.0:
            self.get_logger().error('Zero press direction; cannot press.')
            return False

        # Build press-in pose (same orientation)
        press_pose = Pose()
        press_pose.position.x = (x + PRESS_IN_MM * ux) / 1000.0
        press_pose.position.y = (y + PRESS_IN_MM * uy) / 1000.0
        press_pose.position.z = (z + PRESS_IN_MM * uz) / 1000.0
        press_pose.orientation.x = qx
        press_pose.orientation.y = qy
        press_pose.orientation.z = qz
        press_pose.orientation.w = qw

        # press-in → dwell → lift(out)
        if not self.plan_exec(press_pose, f'{label} press-in'):
            self.get_logger().error(f'{label}: press-in failed.')
            return False
        if DWELL_SEC > 0:
            time.sleep(DWELL_SEC)
        if not self.plan_exec(pose, f'{label} press-out'):
            self.get_logger().error(f'{label}: press-out failed.')
            return False
        if SETTLE_AFTER_PRESS_SEC > 0:
            time.sleep(SETTLE_AFTER_PRESS_SEC)
        return True

    def run(self):
        self.plan.wait_for_service()
        self.exec.wait_for_service()
        self.get_logger().info('Services ready; planning arc with explicit center+radius, adjustable start & sweep.')

        # ----- Fixed plane and base orientation -----
        y_const = float(PATH_Y_MM)
        roll, yaw = float(ROLL_DEG), float(YAW_DEG)

        # ----- Circle geometry -----
        cx0, cz0 = PATH_CENTER_XZ_MM
        r        = float(PATH_RADIUS_MM)

        # Angles: 0° at top (x=cx0, z=cz0+r). Apply user start offset.
        off_deg_clamped = max(-MAX_START_OFFSET_DEG, min(MAX_START_OFFSET_DEG, float(START_ANGLE_OFFSET_DEG)))
        thStart         = math.radians(off_deg_clamped)

        # Sweep (clamped) and direction
        sweep_deg = max(MIN_SWEEP_DEG, min(MAX_SWEEP_DEG, float(MOVE_ANGLE_DEG)))
        sweep_rad = math.radians(sweep_deg)
        direction = 1 if DIRECTION_SIGN >= 0 else -1
        thStop    = thStart - direction * sweep_rad

        self.get_logger().info(
            f'Arc: center=({cx0:.1f},{cz0:.1f}) mm  radius={r:.1f} mm (Ø {2*r:.1f})  '
            f'start={math.degrees(thStart):.1f}°  stop={math.degrees(thStop):.1f}°  '
            f'dir={"CCW" if direction>0 else "CW"}'
        )

        # Ensure pressing aims at the center (update the module-level PRESS_POINT_MM)
        global PRESS_POINT_MM
        PRESS_POINT_MM = [cx0, y_const, cz0]

        # -------- Forward pass (start → along sweep) --------
        n_fwd = max(3, int(N_SAMPLES))
        thetas_fwd = [thStart + i*(thStop - thStart)/(n_fwd-1) for i in range(n_fwd)]
        poses_fwd  = [self.make_arc_pose(th, y_const, cx0, cz0, r, roll, yaw) for th in thetas_fwd]

        # Try first step; if rejected (already at start), do tiny approach then retry
        for i, pose in enumerate(tqdm(poses_fwd, desc='Forward stops', unit='pt'), start=1):
            if not self.plan_exec(pose, f'fwd {i}/{n_fwd}'):
                if i == 1:
                    self.get_logger().warn(
                        'First move failed — maybe already at start or controller rejected a zero-length path. '
                        'Trying a tiny approach shim and retrying...'
                    )
                    th_appr = thStart + FIRST_MOVE_APPROACH_FRAC * (thStop - thStart)
                    pose_appr = self.make_arc_pose(th_appr, y_const, cx0, cz0, r, roll, yaw)
                    if not self.plan_exec(pose_appr, 'approach start'):
                        self.get_logger().error('Approach shim failed; check reach/collision or robot state. Aborting.')
                        return
                    if not self.plan_exec(pose, f'fwd {i}/{n_fwd} (retry)'):
                        self.get_logger().error('Retry to first stop failed; aborting.')
                        return
                else:
                    return

            # Log and press at this stop
            xmm = pose.position.x * 1000.0
            ymm = pose.position.y * 1000.0
            zmm = pose.position.z * 1000.0
            pitch_deg = pitch_facing_center(xmm, zmm, cx0, cz0)
            self.get_logger().info(
                f'Stop {i}/{n_fwd}: pos = [{xmm:.1f}, {ymm:.1f}, {zmm:.1f}] mm | '
                f'rpy(deg) = [{roll:.1f}, {pitch_deg:.1f}, {yaw:.1f}]'
            )

            if not self.press_here(pose, f'fwd {i}/{n_fwd}'):
                self.get_logger().error(f'Press FAILED at forward stop {i}/{n_fwd}. Aborting.')
                return

        # -------- Return along the same arc (no pressing) --------
        self.get_logger().info('Returning along the same arc (no pressing).')
        poses_return_full = list(reversed(poses_fwd[:-1]))  # from next-up stop back to start
        M = len(poses_return_full)
        if M <= 0:
            self.get_logger().warn('No poses available for return path (nothing to do).')
            return
        K = max(1, min(int(N_RETURN_STOPS), M))
        if K == 1:
            indices = [M - 1]  # go straight to start
        else:
            indices = [round(i * (M - 1) / (K - 1)) for i in range(K)]
        poses_return_sel = [poses_return_full[idx] for idx in indices]
        self.get_logger().info(f'Returning with {K} stops (including start).')

        for i, pose in enumerate(tqdm(poses_return_sel, desc='Return stops', unit='pt'), start=1):
            if not self.plan_exec(pose, f'return {i}/{K}'):
                return

        self.get_logger().info('Arc task complete (back at start).')


def main():
    rclpy.init()
    demo = ArcDemo()
    demo.run()
    demo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
