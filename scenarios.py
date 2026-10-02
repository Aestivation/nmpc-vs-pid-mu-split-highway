"""
Rainy-highway scenarios, the shared behaviour layer and the closed-loop runner.

Road: two lanes (right = slow lane, left = fast lane), gentle S-bend.
Weather: rain. The left lane is dry where it runs under the upper deck of a
stacked (two-level) highway between s = 40 m and s = 400 m; everything else is wet.
Traffic is scripted (does not react to the ego car) and drives at constant speed.
"""
import numpy as np
import vehicle_model as vm
import controllers as C


class Scenario:
    def __init__(self, name, title, x0, v_des, traffic, sensor, s_end, t_max=30.0, believed_surface='map'):
        self.name, self.title = name, title
        self.x0, self.v_des, self.traffic = np.array(x0, float), v_des, traffic
        self.sensor, self.s_end, self.t_max = sensor, s_end, t_max
        self.believed_surface = believed_surface
        self.ego_track = []               # (t, s, v) of the ego car, filled while running

    def ego_sv(self, t):
        if not self.ego_track:
            return self.x0[0] + self.x0[3] * t, self.x0[3]
        tr = np.array(self.ego_track)
        if t >= tr[-1, 0]:                # extrapolate a little beyond the last sample
            return tr[-1, 1] + tr[-1, 2] * (t - tr[-1, 0]), tr[-1, 2]
        return float(np.interp(t, tr[:, 0], tr[:, 1])), float(np.interp(t, tr[:, 0], tr[:, 2]))

    def traffic_at(self, t):
        out = []
        for v in self.traffic:
            ey = v['ey']
            if 'sync' in v:
                # Synchronised target car (as in Euro NCAP cut-in tests): it adapts its speed so that it is
                # exactly d metres ahead of the ego car when the ego reaches s_trig; then it drives at constant
                # speed and changes into the left lane within T seconds.
                sy = v['sync']
                if 't_trig' not in sy or t < sy['t_trig']:
                    se, ve = self.ego_sv(t)
                    s_c = se + sy['d'] + sy['k'] * (sy['s_trig'] - se)
                    out.append(dict(s=s_c, ey=ey, v=(1 - sy['k']) * ve, kind=v.get('kind', 'car')))
                else:
                    u = np.clip((t - sy['t_trig']) / sy['T'], 0, 1)
                    ey = v['ey'] + (vm.LANE_L - v['ey']) * (10 * u**3 - 15 * u**4 + 6 * u**5)
                    out.append(dict(s=sy['s_c'] + sy['v_c'] * (t - sy['t_trig']), ey=ey, v=sy['v_c'],
                                    kind=v.get('kind', 'car')))
                continue
            if 'cut' in v:                      # scripted lane change (cut-in): quintic lateral motion
                t0, T, ey1 = v['cut']
                u = np.clip((t - t0) / T, 0, 1)
                ey = v['ey'] + (ey1 - v['ey']) * (10 * u**3 - 15 * u**4 + 6 * u**5)
            out.append(dict(s=v['s0'] + v['v'] * t, ey=ey, v=v['v'], kind=v.get('kind', 'car')))
        return out


def car(lane, s0, v, kind='car'):
    return dict(ey=vm.LANE_R if lane == 'R' else vm.LANE_L, s0=s0, v=v, kind=kind)


def scenario_overtake():
    right = [car('R', s, 20.0) for s in (30, 60, 90, 120, 150, 180, 210, 240, 270)]   # dense, slow
    left = [car('L', -35, 27.0), car('L', 75, 27.5), car('L', 230, 27.5)]            # sparse, faster
    return Scenario('S1_overtake', 'S1 – overtaking in rain (sensor 100 m)',
                    x0=[0, vm.LANE_R, 0, 22.0, 0, 0, 0, 0.3], v_des=27.0, traffic=right + left,
                    sensor=100.0, s_end=420.0)


def scenario_emergency(believed='map'):
    stalled = [car('R', 140.0, 0.0, kind='stalled')]                                   # broken-down car
    left = [car('L', -2.0, 26.0), car('L', -50.0, 26.0), car('L', 180.0, 26.0)]
    tag = {'map': '', 'all_dry': ' – controllers think the road is DRY'}[believed]
    return Scenario('S2_emergency' + ('' if believed == 'map' else '_' + believed),
                    'S2 – stalled car, spray limits sensor to 50 m' + tag,
                    x0=[0, vm.LANE_R, 0, 25.0, 0, 0, 0, 0.35], v_des=25.0, traffic=stalled + left,
                    sensor=50.0, s_end=300.0, believed_surface=believed)


# ---------------------------------------------------------------------------
def detect(scn, x, t):
    """Vehicles inside the sensor range (front and rear), nearest first, as controller slots."""
    tr = [v for v in scn.traffic_at(t) if abs(v['s'] - x[0]) <= scn.sensor]
    # vehicles ahead matter more than those already passed: rank by an effective distance
    tr.sort(key=lambda v: (v['s'] - x[0]) if v['s'] >= x[0] else 3.0 * (x[0] - v['s']))
    obs = np.zeros((C.K, 5)); obs[:, 0] = -1e4
    ahead = {id(v): v for v in tr}
    nxt = scn.traffic_at(t + 0.05); prv = scn.traffic_at(max(t - 0.05, 0.0))
    allv = scn.traffic_at(t)
    for j, v in enumerate(tr[:C.K]):
        i = next(i for i, w in enumerate(allv) if w['s'] == v['s'] and w['ey'] == v['ey'])
        vey = (nxt[i]['ey'] - prv[i]['ey']) / (0.05 + min(t, 0.05))     # measured lateral velocity
        obs[j] = [v['s'] - x[0], v['ey'], v['v'], 1.0, vey]
    return obs, tr


def behaviour(scn, x, detected, mem):
    """Shared high-level decision: which lane to aim for (identical for both controllers).
    Change to the left lane when the vehicle ahead in the right lane is slower than desired
    (or stopped) and within 70 m. Gap acceptance / timing is left to the MPC (collision
    constraints with the left-lane cars)."""
    if mem.get('target') == 'L':
        return vm.LANE_L
    ahead = [v for v in detected if v['ey'] < 0 and 0 < v['s'] - x[0] < 70]
    if ahead:
        lead = min(ahead, key=lambda v: v['s'])
        if lead['v'] < scn.v_des - 2:
            mem['target'] = 'L'; mem['t_decide'] = mem['t']
            return vm.LANE_L
    return vm.LANE_R


def run(scn, ctrl, behaviour_fn=None, stop_fn=None):
    behaviour_fn = behaviour_fn or behaviour
    ctrl.reset()
    x = scn.x0.copy(); t = 0.0; mem = {}
    log = dict(t=[t], x=[x.copy()], u=[], pred=[], solve=[], ok=[], slack=[], target=[], t_decide=None, lost=False)
    stopped = 0.0
    while t < scn.t_max - 1e-9 and x[0] < scn.s_end:
        mem['t'] = t
        scn.ego_track.append((t, x[0], x[3]))
        for v in scn.traffic:                   # synchronised cut-in: triggered at a fixed ROAD position
            if 'sync' in v and 't_trig' not in v['sync'] and x[0] >= v['sync']['s_trig']:
                sy = v['sync']; se, ve = scn.ego_sv(t)
                sy['t_trig'] = t; sy['s_c'] = se + sy['d']; sy['v_c'] = (1 - sy['k']) * ve
                mem.setdefault('cuts', []).append((t, x[0]))
        for v in scn.traffic:                   # distance-triggered cut-in (same situation for every controller)
            if 'cut_trigger' in v and 'cut' not in v:
                gap_trig, T_cut = v['cut_trigger']
                d = v['s0'] + v['v'] * t - x[0]
                if x[1] > 0.5 and 0 < d < gap_trig:
                    v['cut'] = (t, T_cut, vm.LANE_L); mem['cut_t'] = t
        obs, det = detect(scn, x, t)
        ey_t = behaviour_fn(scn, x, det, mem)
        u, pred, info = ctrl.step(x, obs, ey_t, scn.v_des)
        log['u'].append(u); log['pred'].append(pred); log['solve'].append(info['t'])
        log['ok'].append(info['ok']); log['slack'].append(info['slack_col']); log['target'].append(ey_t)
        x = vm.plant_step(x.copy(), u, C.TS)
        t += C.TS
        log['t'].append(t); log['x'].append(x.copy())
        stopped = stopped + C.TS if x[3] < 0.3 else 0.0
        if stopped > 1.0:
            break
        if stop_fn is not None and stop_fn(x, t, mem):
            break
        if abs(x[1]) > vm.ROAD_HALF + 1.0 or abs(x[2]) > 1.0:     # left the road / spun
            log['lost'] = True
            break
    log['t_decide'] = mem.get('t_decide')
    log['mem'] = dict(mem)
    for k in ('t', 'x', 'u', 'solve', 'ok', 'slack', 'target'):
        log[k] = np.array(log[k])
    return log


# ------------------------------------------------------------- evaluation --
def _ego_box(s, ey, epsi):
    c, sn = np.cos(epsi), np.sin(epsi)
    pts = np.array([[vm.EGO_FRONT, vm.CAR_WID / 2], [vm.EGO_FRONT, -vm.CAR_WID / 2],
                    [-vm.EGO_REAR, -vm.CAR_WID / 2], [-vm.EGO_REAR, vm.CAR_WID / 2]])
    return np.c_[s + pts[:, 0] * c - pts[:, 1] * sn, ey + pts[:, 0] * sn + pts[:, 1] * c]


def _box(s, ey):
    hl, hw = vm.CAR_LEN / 2, vm.CAR_WID / 2
    return np.array([[s - hl, ey - hw], [s + hl, ey - hw], [s + hl, ey + hw], [s - hl, ey + hw]])


def clearance(P, Q):
    """Separating-axis distance between two convex quadrilaterals (<0 means overlap)."""
    best = -np.inf
    for R in (P, Q):
        for i in range(4):
            e = R[(i + 1) % 4] - R[i]; n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            a, b = P @ n, Q @ n
            best = max(best, b.min() - a.max(), a.min() - b.max())
    return best


def evaluate(scn, log):
    t, X = log['t'], log['x']
    # collision check at 10 sub-samples per control step (linear interpolation)
    min_clear, worst = np.inf, None
    for k in range(len(t) - 1):
        for a in np.linspace(0, 1, 10, endpoint=False):
            xi = (1 - a) * X[k] + a * X[k + 1]; ti = (1 - a) * t[k] + a * t[k + 1]
            E = _ego_box(xi[0], xi[1], xi[2])
            for v in scn.traffic_at(ti):
                if abs(v['s'] - xi[0]) > 15:
                    continue
                c = clearance(E, _box(v['s'], v['ey']))
                if c < min_clear:
                    min_clear, worst = c, (ti, v['kind'], round(v['ey'], 2))
    util = np.array([vm.utilisation(x)[0] for x in X])
    ax = np.gradient(X[:, 3], t) - X[:, 4] * X[:, 5]
    ay = np.gradient(X[:, 4], t) + X[:, 3] * X[:, 5]
    jerk = np.gradient(ax, t)
    in_left = np.where(X[:, 1] > 0)[0]
    lc_time = t[in_left[0]] if len(in_left) else np.nan
    reached = X[-1, 0] >= scn.s_end
    off_road = bool(np.any(np.abs(X[:, 1]) + vm.CAR_WID / 2 > vm.ROAD_HALF + 0.05))
    spun = bool(np.any(np.abs(X[:, 2]) > 0.5))
    return dict(collision=bool(min_clear < 0), off_road=off_road, spun=spun,
                safe=bool(min_clear >= 0 and not off_road and not spun), min_clear=min_clear, worst=worst,
                max_util=float(util.max()), util=util,
                max_brake_g=float(-ax.min() / vm.G), max_ay_g=float(np.abs(ay).max() / vm.G),
                max_jerk=float(np.abs(jerk).max()), ax=ax, ay=ay,
                min_v=float(X[:, 3].min()), mean_v=float(np.mean(X[:, 3])),
                t_end=float(t[-1]), reached=bool(reached), s_final=float(X[-1, 0]),
                lc_time=lc_time,
                solve_mean_ms=1e3 * float(np.mean(log['solve'])), solve_max_ms=1e3 * float(np.max(log['solve'])),
                solve_p95_ms=1e3 * float(np.percentile(log['solve'], 95)),
                fails=int(np.sum(~log['ok'])), max_slack=float(np.nanmax(log['slack'])))
