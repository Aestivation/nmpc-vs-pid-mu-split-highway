"""Long rainy-highway drive: clusters of slow cars in the (wet) right lane, sparse fast
traffic in the left lane (dry under the overpass decks). The car overtakes each cluster
and returns to the right lane (keep-right rule) -> 5 lane changes."""
import numpy as np
import vehicle_model as vm
import scenarios as S

V_SLOW, V_FAST, V_DES = 20.0, 29.0, 27.0
FINISH_S = 2100.0          # finish line [m] - the run ends here for every controller
# three cut-ins at fixed road positions: (s_trig [m], gap d [m], lane-change time T [s])
CUTS = [(650.0, 20.0, 2.0), (1220.0, 19.0, 2.0), (1850.0, 19.0, 1.8)]


def make():
    right = [S.car('R', 30 + 25 * i, V_SLOW) for i in range(3)]      # slow cluster at the start
    right += [S.car('R', 396, V_SLOW)]                                # a lone slow car between cut-in 2 and 3
    for (s_trig, d, T) in CUTS:                                       # synchronised cut-in cars
        c = S.car('R', 0.0, V_SLOW, kind='cut-in')
        c['sync'] = dict(s_trig=s_trig, d=d, T=T, k=1 - V_SLOW / V_DES)
        right.append(c)
    left = [S.car('L', 60, V_FAST), S.car('L', 160, V_FAST)]
    return S.Scenario('YT_rain_highway', 'Autonomous driving in the rain – NMPC vs PID + Stanley',
                      x0=[0, vm.LANE_R, 0, 23.0, 0, 0, 0, 0.3], v_des=V_DES,
                      traffic=right + left, sensor=120.0, s_end=FINISH_S, t_max=110.0)


def behaviour(scn, x, detected, mem):
    """Keep right; overtake a slower car ahead; return right when there is room."""
    in_left = mem.get('target', 'R') == 'L'
    rel = [(v['s'] - x[0], v) for v in detected]
    if not in_left:
        ahead = [d for d, v in rel if v['ey'] < 0 and 0 < d < 55 and v['v'] < scn.v_des - 2]
        left_busy = any(v['ey'] > 0 and -15 < d < 20 for d, v in rel)
        if ahead and not left_busy:
            mem['target'] = 'L'; mem.setdefault('changes', []).append((mem['t'], 'R->L'))
    else:
        right_busy = any(v['ey'] < 0 and -12 < d < 45 for d, v in rel)
        if not right_busy and mem['changes'][-1][0] < mem['t'] - 3:
            mem['target'] = 'R'; mem['changes'].append((mem['t'], 'L->R'))
    return vm.LANE_L if mem.get('target', 'R') == 'L' else vm.LANE_R


def stop(x, t, mem):
    """No extra stop rule: every run ends at the same finish line (s = FINISH_S)."""
    return False
