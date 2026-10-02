"""
Classical baseline used in industry-style pipelines:
  * lane change  : quintic-polynomial path from the current lateral position to the target lane
                   over a fixed manoeuvre time (T_LC)
  * steering     : Stanley controller (heading error + cross-track term) with curvature
                   feed-forward and yaw-rate damping
  * longitudinal : adaptive cruise control - PI speed control towards
                   min(v_des, v_lead + K_GAP*(gap - (D0 + T_H*v))) for the lead car in the same lane
No vehicle model, no preview of the friction, no constraints except the actuator limits.
The gains are tuned on a DRY road (see tune_dry() in compare_classical.py).
"""
import time
import numpy as np
import vehicle_model as vm

TS = 0.1
GAINS = dict(k_stanley=0.6, k_soft=1.0, k_yaw=0.12, kp_v=1.2, ki_v=0.15, T_LC=3.0,
             D0=8.0, T_H=1.2, K_GAP=0.4, A_MIN=-3.5, A_MAX=2.0, TTC_AEB=1.6)


class Classical:
    name = 'PID + Stanley'

    def __init__(self, surface='map', gains=None):
        self.g = dict(GAINS, **(gains or {}))
        self.reset()

    def reset(self):
        self.path = None          # (s0, ey0, s1, ey1)
        self.target = None
        self.i_v = 0.0

    def _path(self, s):
        s0, e0, s1, e1 = self.path
        if s >= s1:
            return e1, 0.0, 0.0
        if s <= s0:
            return e0, 0.0, 0.0
        L = s1 - s0; u = (s - s0) / L
        p = 10 * u**3 - 15 * u**4 + 6 * u**5
        dp = (30 * u**2 - 60 * u**3 + 30 * u**4) / L
        ddp = (60 * u - 180 * u**2 + 120 * u**3) / L**2
        return e0 + (e1 - e0) * p, (e1 - e0) * dp, (e1 - e0) * ddp

    def step(self, x_abs, obs, ey_t, v_des):
        t0 = time.perf_counter()
        g = self.g
        s, ey, epsi, vx, vy, r, d, F = x_abs
        if self.target is None or abs(ey_t - self.target) > 1e-6:
            self.target = ey_t
            self.path = (s, ey, s + max(vx, 5.0) * g['T_LC'], ey_t)
        e_ref, de_ref, dde_ref = self._path(s)
        # ---- Stanley lateral control (front axle cross-track error) ------
        psi_ref = np.arctan(de_ref)
        k_road = vm.kappa(s)
        e_front = (ey + vm.LF * np.sin(epsi)) - e_ref
        head_err = epsi - psi_ref
        kappa_path = k_road + dde_ref
        delta_ff = vm.L * kappa_path
        r_ref = vx * kappa_path
        delta_cmd = (delta_ff - head_err - np.arctan2(g['k_stanley'] * e_front, g['k_soft'] + vx)
                     - g['k_yaw'] * (r - r_ref))
        delta_cmd = np.clip(delta_cmd, -vm.DELTA_MAX, vm.DELTA_MAX)
        # ---- ACC: lead vehicle in the current lane -------------------------
        v_tgt = v_des; aeb = False
        for so, eo, vo, act, *_ in obs:
            if act < 0.5 or so <= 0:
                continue
            gap = so - vm.CAR_LEN / 2 - vm.EGO_FRONT
            if abs(eo - self.target) < 2.0:      # ACC follows the lead car in the TARGET lane (during a lane change too)
                v_tgt = min(v_tgt, vo + g['K_GAP'] * (gap - (g['D0'] + g['T_H'] * vx)))
            if abs(eo - ey) < 1.8 and (vx > vo and gap / (vx - vo) < g['TTC_AEB'] or gap < 3.0):
                aeb = True                       # automatic emergency braking: anything in my current path
        v_tgt = max(v_tgt, 0.0)
        ev = v_tgt - vx
        self.i_v = np.clip(self.i_v + ev * TS, -10, 10)
        a_cmd = np.clip(g['kp_v'] * ev + g['ki_v'] * self.i_v, g['A_MIN'], g['A_MAX'])   # ACC comfort limits
        F_cmd = np.clip((vm.M * a_cmd + vm.C_DRAG * vx**2 + vm.C_ROLL * vm.M * vm.G) / 1000, vm.FX_MIN, vm.FX_MAX)
        if aeb:
            F_cmd = vm.FX_MIN; self.i_v = 0.0
        u = np.array([np.clip((delta_cmd - d) / TS, -vm.DDELTA_MAX, vm.DDELTA_MAX),
                      np.clip((F_cmd - F) / TS, -vm.DFX_MAX, vm.DFX_MAX)])
        # planned path for display (3 s ahead)
        ss = s + np.linspace(0, 3.0 * max(vx, 5), 31)
        pred = np.zeros((8, 31)); pred[0] = ss; pred[1] = [self._path(q)[0] for q in ss]
        return u, pred, dict(t=time.perf_counter() - t0, ok=True, slack_col=0.0)
