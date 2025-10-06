#!/usr/bin/env python3
"""
Lite-6 straight-line Cartesian motion with a small 'press' at each waypoint.

OVERVIEW
--------
This ROS 2 node plans and executes a straight-line sequence of Cartesian waypoints
for a UFactory xArm Lite 6 (via MoveIt 2), and at each visited waypoint performs a
short "press" motion along a chosen direction before returning to that waypoint.

The press cycle at each waypoint is:
    arrive at waypoint → press-in (nudge) → optional dwell (hold) → press-out (return)

KEY IDEAS
---------
• Positions are specified in millimeters (mm). MoveIt/ROS Pose messages use meters,
  so this script converts mm → m internally when building Pose messages.
• Orientations are given either as fixed RPY angles (degrees) or interpolated via SLERP.
• "Press" direction can be:
    - toward a specific point in space (XZ-plane only in this setup),
    - along a fixed world vector, or
    - along the tool's +Z / −Z axis (derived from the current orientation).

PREREQUISITES / ASSUMPTIONS
---------------------------
• ROS 2 + MoveIt 2 stack is running with xarm_msgs services exposed:
      /xarm_pose_plan  (PlanPose)
      /xarm_exec_plan  (PlanExec)
• START/END waypoints are reachable and collision-free in your scene.
• Units: mm for positions in the parameter section, degrees for RPY; meters/quaternions in Pose.
• This script interpolates *linearly* in position from START to END across N_STEPS points.

SAFETY / TUNING NOTES
---------------------
• PRESS_IN_MM too large may cause contact/collision or planning failures. Start small.
• DWELL_SEC controls how long the end effector holds the pressed-in pose before returning.
• If planning or execution fails, the script logs a human-readable reason (see ERR map)
  and stops at that point.
• Progress bar (tqdm) is optional; the script works even if tqdm isn't installed.

"""

import math, time, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from xarm_msgs.srv import PlanPose, PlanExec

# --- tqdm (optional pretty progress bar) ---
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def make_pbar(total, initial=0, desc="Waypoints"):
    """Create a tqdm progress bar if tqdm is installed; otherwise return a no-op.

    This allows the same code path to run with or without tqdm present.
    The bar advances once a waypoint's full cycle completes (arrive + press).
    """
    if tqdm is None:
        class _NoBar:
            def update(self, n=1): pass
            def set_postfix(self, **k): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _NoBar()
    return tqdm(total=total, initial=initial, desc=desc, unit="pt", dynamic_ncols=True)

# ========================= USER PARAMETERS =========================
# Historical presets kept for convenience (do not delete):
# START = [100, -30, 255, -180, 0, 0]
# END   = [100,  30, 255, -180, 0, 0]

# START = [250, -250, 265, -180, 0, 0]
# END   = [250,  250, 265, -180, 0, 0]

# final 8/26 for pipe at a low position 
# START = [250, -250, 182, -180, 0, 0]
# END   = [250,  250, 182, -180, 0, 0]

# final 8/28 for pipe at a high position
# START/END format: [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
START  = [200, -150, 455, -180,    0,   0]   # "Top reference" (geometric TOP) in world frame
END    = [200,  150, 455, -180,    0,   0]   # "Top reference" (geometric TOP) in world frame

PRESS_IN_MM    = 9            # Press-in distance (mm). Note: 9 sometimes triggers press errors; try 8 if unstable.
# DWELL_SEC      = 25           # Hold time (seconds) at the pressed-in pose before returning.
DWELL_SEC      = 1

# N_STEPS = 61                  # Total waypoints along the line (including start and end). Must be ≥ 2.
N_STEPS = 2
PAUSE_BETWEEN_STEPS_SEC = 0   # Optional pause after each *travel* step (seconds). Press dwell is separate.
ORIENTATION_MODE = 'hold'     # 'hold'  → keep START orientation everywhere
                             # 'slerp' → slerp between START and END orientations per waypoint

# ---- Press/nudge parameters ----
PRESS_ON_LAST  = True         # If True, perform a press at the final waypoint as well.
PRESS_MODE     = 'toward_point'      # 'toward_point' | 'world_vec' | 'tool_z'
PRESS_POINT_MM = [250.0, 0.0, 100.0] # Used when PRESS_MODE == 'toward_point' (see press_here for Y handling).
PRESS_WORLD_VEC = [0.0, 0.0, -1.0]   # Used when PRESS_MODE == 'world_vec' (direction in world coords).
TOOL_Z_SIGN    = -1.0                # Used when PRESS_MODE == 'tool_z': +1 → along tool +Z, -1 → along tool −Z
# ==================================================================


def rpy_to_quat(r, p, y):
    """Convert intrinsic ZYX Euler angles (deg) to quaternion (x, y, z, w).

    Intrinsic ZYX means yaw (Z) → pitch (Y) → roll (X) applied in that order.
    """
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

def quat_normalize(q):
    """Return q normalized to unit length; default to identity if zero-length."""
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        return (0, 0, 0, 1)
    return (x/n, y/n, z/n, w/n)

def quat_slerp(q0, q1, t):
    """Spherical linear interpolation between quaternions q0 and q1 at fraction t ∈ [0,1]."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    dot = x0*x1 + y0*y1 + z0*z1 + w0*w1

    # Ensure shortest path by flipping sign of q1 if needed
    if dot < 0.0:
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1
        dot = -dot

    # If nearly aligned, fall back to normalized lerp
    if dot > 0.9995:
        x = x0 + t*(x1 - x0)
        y = y0 + t*(y1 - y0)
        z = z0 + t*(z1 - z0)
        w = w0 + t*(w1 - w0)
        return quat_normalize((x, y, z, w))

    # Standard SLERP
    theta0 = math.acos(dot)
    sin0 = math.sin(theta0)
    s0 = math.sin((1.0 - t)*theta0) / sin0
    s1 = math.sin(t*theta0) / sin0
    return (s0*x0 + s1*x1, s0*y0 + s1*y1, s0*z0 + s1*z1, s0*w0 + s1*w1)

def unit(v):
    """Normalize a 3D vector; return (0,0,0) for zero-length input."""
    x, y, z = v
    n = math.sqrt(x*x + y*y + z*z)
    if n == 0:
        return (0, 0, 0)
    return (x/n, y/n, z/n)

def tool_z_axis_world(qx, qy, qz, qw):
    """Return the tool's +Z axis (0,0,1 in tool frame) expressed in world coordinates.

    Uses the quaternion → rotation-matrix mapping and extracts the third column.
    """
    tx = 2*(qx*qz + qw*qy)
    ty = 2*(qy*qz - qw*qx)
    tz = 1 - 2*(qx*qx + qy*qy)
    return (tx, ty, tz)


class StraightLineDemo(Node):
    """ROS 2 node to:
       1) Build N_STEPS evenly spaced Cartesian waypoints from START → END,
       2) Move to each waypoint in order,
       3) At each waypoint, perform a press-in → dwell → press-out sequence.
    """

    # Map common MoveIt/xarm return codes to human-readable messages.
    ERR = {
        1:   'success',
        -31: 'no IK solution (joint limits / reach)',
        -10: 'goal in collision',
        -5:  'invalid motion plan',
        -2:  'planning failed',
    }

    def __init__(self):
        super().__init__('lite6_straight_line_demo')
        # Service clients exposed by the xArm MoveIt pipeline
        self.plan = self.create_client(PlanPose,  '/xarm_pose_plan')
        self.exec = self.create_client(PlanExec, '/xarm_exec_plan')

    # -------- Pose builders (convert mm + RPY/quat → ROS Pose in meters) --------
    def pose_from_mm_rpy(self, xyz_mm, rpy_deg):
        """Build a Pose from mm + RPY(deg)."""
        pose = Pose()
        pose.position.x = xyz_mm[0] / 1000.0
        pose.position.y = xyz_mm[1] / 1000.0
        pose.position.z = xyz_mm[2] / 1000.0
        qx, qy, qz, qw = rpy_to_quat(*rpy_deg)
        pose.orientation.x = qx; pose.orientation.y = qy
        pose.orientation.z = qz; pose.orientation.w = qw
        return pose

    def pose_from_mm_quat(self, xyz_mm, quat_xyzw):
        """Build a Pose from mm + quaternion (x,y,z,w)."""
        pose = Pose()
        pose.position.x = xyz_mm[0] / 1000.0
        pose.position.y = xyz_mm[1] / 1000.0
        pose.position.z = xyz_mm[2] / 1000.0
        x, y, z, w = quat_xyzw
        pose.orientation.x = x; pose.orientation.y = y
        pose.orientation.z = z; pose.orientation.w = w
        return pose

    # -------- Main routine --------
    def run(self):
        """Wait for services, build the path, execute travel + press cycles with progress feedback."""
        self.plan.wait_for_service()
        self.exec.wait_for_service()
        self.get_logger().info('Services ready; executing straight line with presses.')

        # ---- 1) Build waypoints along the straight segment ----
        assert N_STEPS >= 2, "N_STEPS must be at least 2"

        p0 = START[:3];   RPY0 = START[3:]
        p1 = END[:3];     RPY1 = END[3:]

        q0 = quat_normalize(rpy_to_quat(*RPY0))
        q1 = quat_normalize(rpy_to_quat(*RPY1))

        waypoints = []
        for i in range(N_STEPS):
            t = i / (N_STEPS - 1)  # normalized position along the line: 0 → start, 1 → end

            # Linear interpolation of position (mm)
            x = p0[0] + t*(p1[0] - p0[0])
            y = p0[1] + t*(p1[1] - p0[1])
            z = p0[2] + t*(p1[2] - p0[2])

            # Orientation choice:
            #   'slerp' → interpolate between START and END orientations
            #   'hold'  → keep the START orientation everywhere (simpler, often sufficient)
            if ORIENTATION_MODE.lower() == 'slerp':
                q = quat_slerp(q0, q1, t)
                pose = self.pose_from_mm_quat([x, y, z], q)
            else:
                pose = self.pose_from_mm_rpy([x, y, z], RPY0)

            waypoints.append(pose)

        # ---- 2) Execute: move to the first point, then step along the line ----
        # Progress bar shows how many waypoints have completed their arrive+press cycle.
        with make_pbar(total=N_STEPS, initial=0, desc="Waypoints visited") as pbar:

            # Move to start
            if not self.call_plan_exec(waypoints[0], 'goto start'):
                return
            if PAUSE_BETWEEN_STEPS_SEC > 0:
                time.sleep(PAUSE_BETWEEN_STEPS_SEC)

            # Optional: press at the start (uses current PRESS_* settings)
            if not self.press_here(waypoints[0], 'press start'):
                self.get_logger().warn('Press at start skipped/failed; continuing.')

            # Count the start point as visited
            pbar.update(1)

            # March along the line
            for i, pose in enumerate(waypoints[1:], start=2):
                # Travel to the next waypoint
                if not self.call_plan_exec(pose, f'step {i}/{N_STEPS}'):
                    return

                # Perform press at this waypoint
                if (i < N_STEPS or PRESS_ON_LAST) and not self.press_here(pose, f'press {i}/{N_STEPS}'):
                    self.get_logger().warn(f'Press at step {i} skipped/failed; continuing.')

                # Optional pause between travel steps (separate from dwell time during the press)
                if i < N_STEPS and PAUSE_BETWEEN_STEPS_SEC > 0:
                    time.sleep(PAUSE_BETWEEN_STEPS_SEC)

                # One more waypoint fully processed
                pbar.update(1)

        self.get_logger().info('Straight line finished.')

    # -------- utilities: plan/exec + error messages --------
    def call_plan_exec(self, pose: Pose, label: str):
        """Plan to a target pose and execute the resulting plan (blocking)."""
        preq = PlanPose.Request()
        preq.target = pose
        if not self._call(preq, f'plan {label}'):
            return False
        if not self._call(PlanExec.Request(wait=True), f'exec {label}'):
            return False
        return True

    def _call(self, request, label):
        """Generic service-call helper with result checking and readable error logs."""
        cli = self.plan if isinstance(request, PlanPose.Request) else self.exec
        fut = cli.call_async(request)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()

        # xarm_msgs may provide (success, ret) or just success; normalize to (ok, code)
        ok   = getattr(res, 'success', False)
        code = getattr(res, 'ret', None)
        if code is None and isinstance(getattr(res, 'success', None), bool):
            ok = res.success
            code = 1 if ok else -2

        if not ok:
            msg = self.ERR.get(code, f'unknown error {code}')
            self.get_logger().error(f'{label} failed: {msg}')
        return ok

    # -------- pressing logic --------
    def press_here(self, pose: Pose, label: str):
        """Perform a press at 'pose':
           1) Compute a unit press direction (based on PRESS_MODE and parameters),
           2) Move PRESS_IN_MM along that direction (press-in),
           3) Dwell (DWELL_SEC),
           4) Return to the exact starting pose (press-out).

           Returns True if the full press cycle completes; False if a plan/exec step fails.
        """
        if PRESS_IN_MM <= 0.0 or DWELL_SEC < 0.0:
            # Nothing to do or invalid dwell; treat as success to keep the main loop flowing.
            return True

        # Extract current waypoint (mm) and orientation (quaternion)
        x = pose.position.x * 1000.0
        y = pose.position.y * 1000.0
        z = pose.position.z * 1000.0
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w

        # Choose press direction (expressed in world coordinates), then normalize to unit length
        if PRESS_MODE == 'toward_point':
            # Vector from current point to PRESS_POINT_MM.
            # IMPORTANT: for this specific use-case we keep the press in the XZ-plane,
            # so we intentionally set dy = 0 (i.e., no Y component). The original code
            # achieves this by computing y - y (which is 0) to maintain the current Y.
            dx = PRESS_POINT_MM[0] - x
            # dy = PRESS_POINT_MM[1] - y  # (disabled to stay in XZ-plane)
            dy = y - y                   # → 0.0; press direction has no Y-component
            dz = PRESS_POINT_MM[2] - z
            ux, uy, uz = unit((dx, dy, dz))

        elif PRESS_MODE == 'world_vec':
            # Use a fixed direction in world coordinates (will be normalized).
            ux, uy, uz = unit(PRESS_WORLD_VEC)

        elif PRESS_MODE == 'tool_z':
            # Project the tool's Z-axis into world coordinates and select +Z or −Z.
            tz = tool_z_axis_world(qx, qy, qz, qw)
            ux, uy, uz = (TOOL_Z_SIGN*tz[0], TOOL_Z_SIGN*tz[1], TOOL_Z_SIGN*tz[2])
            ux, uy, uz = unit((ux, uy, uz))

        else:
            self.get_logger().warn(f'Unknown PRESS_MODE={PRESS_MODE}, skipping press.')
            return True

        if ux == uy == uz == 0.0:
            self.get_logger().warn('Zero press direction; skipping press.')
            return True

        # Build the press-in pose (same orientation as the waypoint)
        px = x + PRESS_IN_MM * ux
        py = y + PRESS_IN_MM * uy
        pz = z + PRESS_IN_MM * uz

        press_pose = Pose()
        press_pose.position.x = px / 1000.0
        press_pose.position.y = py / 1000.0
        press_pose.position.z = pz / 1000.0
        press_pose.orientation.x = qx
        press_pose.orientation.y = qy
        press_pose.orientation.z = qz
        press_pose.orientation.w = qw

        # Plan/execute press-in
        if not self.call_plan_exec(press_pose, f'{label} (in)'):
            return False

        # Hold at pressed-in pose, if requested
        if DWELL_SEC > 0:
            time.sleep(DWELL_SEC)

        # Plan/execute press-out (return to the exact waypoint pose)
        if not self.call_plan_exec(pose, f'{label} (out)'):
            return False

        return True


def main():
    rclpy.init()
    demo = StraightLineDemo()
    demo.run()
    demo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
