"""
Two model-predictive controllers with IDENTICAL cost, horizon, constraints and
inputs, differing only in how they use the vehicle model:

  NMPC     full nonlinear model (Pacejka combined slip, load transfer, surface map,
           road curvature) inside the optimisation; solved with IPOPT
           (multiple shooting, RK4).  Obstacle constraints are nonconvex
           (ellipses), the optimiser can choose to steer around or brake.

  LTV-MPC  the same model is linearised once per sample at the current state and
           the previous input (as in Falcone et al., IEEE TCST 2007), discretised
           exactly (matrix exponential) and kept constant over the horizon.
           Tyre/friction constraints are linearised at the same point; obstacle
           ellipses are linearised along the previous predicted trajectory
           (tangent half-planes).  Solved as a QP with OSQP.

Both:   Ts = 0.1 s, N = 30 (3 s look-ahead), inputs = steering rate and
        longitudinal force rate, same weights, same soft-constraint penalties.
"""
import time
import numpy as np
import casadi as ca
import scipy.sparse as sp
from scipy.linalg import expm
import osqp
import vehicle_model as vm

TS, N, K = 0.1, 30, 6          # sample time, horizon, max. number of tracked vehicles
NSUB = 2                       # RK4 sub-steps per interval inside the NMPC
NX, NU = 8, 2
W = dict(ey=0.1, epsi=2.0, v=0.3, r=20.0, dd=100.0, dF=0.01, F=0.02, slack=1e3, slack2=1e4, col=1e4, col2=1e5)

CIRC_OFF = np.array([-1.533, 0.0, 1.533])      # ego body covered by 3 circles
CIRC_R = 1.2
A_ELL = vm.CAR_LEN / 2 * np.sqrt(2) + CIRC_R + 4.0    # ellipse around a vehicle: longitudinal
B_ELL = vm.CAR_WID / 2 * np.sqrt(2) + CIRC_R + 0.15   # lateral
EY_LIM = vm.ROAD_HALF - vm.CAR_WID / 2 - 0.25
RHO_MAX = 0.95


# per wheel: slip angle within the stable side of the Pacejka peak, friction ratio below RHO_MAX
# slack groups: 0 front slip, 1 rear slip, 2 front friction, 3 rear friction
SLACK_MAP = []
for _i in range(4):
    SLACK_MAP += [0 if _i < 2 else 1] * 2 + [2 if _i < 2 else 3] * 2
NG = len(SLACK_MAP)


def _tyre_constraints(x, surface, s_off, lib, w_fixed=None):
    _, a = vm.dynamics(x, [0, 0], surface, lib, False, s_off, w_fixed)
    g = []
    for i in range(4):
        g += [a['alpha'][i] - a['apk'][i], -a['alpha'][i] - a['apk'][i], a['rho'][i] - RHO_MAX, -a['rho'][i] - RHO_MAX]
    return g


# =========================================================================
class NMPC:
    name = 'NMPC'

    def __init__(self, surface='map'):
        self.surface = surface
        opti = ca.Opti()
        X = opti.variable(NX, N + 1); U = opti.variable(NU, N)
        Sc = opti.variable(K, N); Se = opti.variable(1, N); Sa = opti.variable(4, N)
        x0 = opti.parameter(NX); s_off = opti.parameter(); obs = opti.parameter(K, 5)
        ey_t = opti.parameter(); v_des = opti.parameter()
        f = lambda x, u: vm.dynamics(x, u, surface, ca, False, s_off)[0]
        J = 0
        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            xn = X[:, k]
            for _ in range(NSUB):
                xn = vm.rk4(f, xn, U[:, k], TS / NSUB)
            opti.subject_to(X[:, k + 1] == xn)
            xk = X[:, k + 1]
            J += (W['ey'] * (xk[1] - ey_t)**2 + W['epsi'] * xk[2]**2 + W['v'] * (xk[3] - v_des)**2
                  + W['r'] * xk[5]**2 + W['F'] * xk[7]**2 + W['dd'] * U[0, k]**2 + W['dF'] * U[1, k]**2)
            sl = ca.vertcat(Se[k], Sa[:, k])
            J += W['slack'] * ca.sum1(sl) + W['slack2'] * ca.sumsqr(sl)
            J += W['col'] * ca.sum1(Sc[:, k]) + W['col2'] * ca.sumsqr(Sc[:, k])     # collision: much stiffer
            g = _tyre_constraints(xk, surface, s_off, ca)
            for gi in range(NG):
                opti.subject_to(g[gi] <= Sa[SLACK_MAP[gi], k])
            opti.subject_to(opti.bounded(-EY_LIM - Se[k], xk[1], EY_LIM + Se[k]))
            opti.subject_to(opti.bounded(-vm.DELTA_MAX, xk[6], vm.DELTA_MAX))
            opti.subject_to(opti.bounded(vm.FX_MIN, xk[7], vm.FX_MAX))
            opti.subject_to(xk[3] >= 0)
            opti.subject_to(opti.bounded(-vm.DDELTA_MAX, U[0, k], vm.DDELTA_MAX))
            opti.subject_to(opti.bounded(-vm.DFX_MAX, U[1, k], vm.DFX_MAX))
            tk = (k + 1) * TS
            for j in range(K):
                so = obs[j, 0] + obs[j, 2] * tk
                # lateral motion of the other car (measured lateral speed), it stays within the road lanes
                eo = ca.fmin(ca.fmax(obs[j, 1] + obs[j, 4] * tk, vm.LANE_R), vm.LANE_L)
                for o in CIRC_OFF:
                    cs = xk[0] + o * ca.cos(xk[2]); cy = xk[1] + o * ca.sin(xk[2])
                    h = ((cs - so) / A_ELL)**2 + ((cy - eo) / B_ELL)**2
                    opti.subject_to(obs[j, 3] * (1 - h) <= Sc[j, k])
        opti.subject_to(ca.vec(Sc) >= 0); opti.subject_to(ca.vec(Se) >= 0); opti.subject_to(ca.vec(Sa) >= 0)
        opti.minimize(J)
        opti.solver('ipopt', {'expand': True, 'print_time': False, 'error_on_fail': False},
                    {'print_level': 0, 'max_iter': 200, 'tol': 1e-5, 'acceptable_tol': 1e-3,
                     'acceptable_iter': 5, 'mu_init': 1e-2})
        self.fn = opti.to_function('nmpc', [x0, s_off, obs, ey_t, v_des, X, U, Sc, Se, Sa],
                                   [X, U, Sc, Se, Sa, J])
        xs, us, so_ = ca.SX.sym('x', NX), ca.SX.sym('u', NU), ca.SX.sym('so')
        xn = xs
        for _ in range(NSUB):
            xn = vm.rk4(lambda a, b: vm.dynamics(a, b, surface, ca, False, so_)[0], xn, us, TS / NSUB)
        self.Fd = ca.Function('Fd', [xs, us, so_], [xn])
        self.guess = None

    def reset(self):
        self.guess = None

    def _solve(self, x0, s_off, obs, ey_t, v_des, guess):
        out = self.fn(x0, s_off, obs, ey_t, v_des, *guess)
        Xs, Us, Scs, Ses, Sas, Jv = [np.array(o) for o in out]
        defect = max(np.max(np.abs(np.array(self.Fd(Xs[:, k], Us[:, k], s_off)).ravel() - Xs[:, k + 1]))
                     for k in range(N))
        ok = bool(defect < 1e-3 and np.all(np.isfinite(Us)))
        return Xs, Us, Scs, Ses, Sas, float(np.ravel(Jv)[0]), ok

    def step(self, x_abs, obs, ey_t, v_des):
        s_off = x_abs[0]
        x0 = x_abs.copy(); x0[0] = 0.0
        if self.guess is None:
            Xg = np.zeros((NX, N + 1)); Ug = np.zeros((NU, N))
            Xg[:, 0] = x0
            for k in range(N):
                Xg[:, k + 1] = np.array(self.Fd(Xg[:, k], Ug[:, k], s_off)).ravel()
            self.guess = [Xg, Ug, np.zeros((K, N)), np.zeros((1, N)), np.zeros((4, N))]
        else:                                           # shift the previous solution one step
            Xg = np.c_[self.guess[0][:, 1:], self.guess[0][:, -1:]].copy()
            Xg[0] -= s_off - self.s_off_prev            # re-centre s on the new position
            Xg[:, 0] = x0
            self.guess = [Xg] + [np.c_[g[:, 1:], g[:, -1:]] for g in self.guess[1:]]
        self.s_off_prev = s_off
        t0 = time.perf_counter()
        cands = [self._solve(x0, s_off, obs, ey_t, v_des, self.guess)]
        # multi-start: when a vehicle is close ahead, also start from a "brake hard, hold the lane" guess
        # and keep the better local optimum (avoids getting stuck in the "squeeze past" solution)
        near = [j for j in range(K) if obs[j, 3] > 0.5 and 0 < obs[j, 0] < 50 and abs(obs[j, 1] - x0[1]) < 3.5]
        if near:
            Xb = np.zeros((NX, N + 1)); Ub = np.zeros((NU, N)); Xb[:, 0] = x0
            for k in range(N):
                Ub[1, k] = -vm.DFX_MAX if Xb[7, k] > 0.55 * vm.FX_MIN else 0.0
                Xb[:, k + 1] = np.array(self.Fd(Xb[:, k], Ub[:, k], s_off)).ravel()
            cands.append(self._solve(x0, s_off, obs, ey_t, v_des,
                                     [Xb, Ub, np.zeros((K, N)), np.zeros((1, N)), np.zeros((4, N))]))
            # third start: "evade into the other lane" (lateral quintic over 2 s at constant speed)
            e1 = vm.LANE_R if x0[1] > 0 else vm.LANE_L
            Xe = np.tile(x0[:, None], (1, N + 1)); Ue = np.zeros((NU, N))
            tt = np.arange(N + 1) * TS; u_ = np.clip(tt / 2.0, 0, 1)
            Xe[0] = x0[3] * tt
            Xe[1] = x0[1] + (e1 - x0[1]) * (10 * u_**3 - 15 * u_**4 + 6 * u_**5)
            de = (e1 - x0[1]) * (30 * u_**2 - 60 * u_**3 + 30 * u_**4) / 2.0      # d(ey)/dt
            Xe[2] = np.clip(np.arctan2(de, max(x0[3], 1.0)), -0.2, 0.2); Xe[4] = 0.0; Xe[5] = 0.0
            cands.append(self._solve(x0, s_off, obs, ey_t, v_des,
                                     [Xe, Ue, np.zeros((K, N)), np.zeros((1, N)), np.zeros((4, N))]))
        dt = time.perf_counter() - t0
        good = [c_ for c_ in cands if c_[-1]]
        if good:
            Xs, Us, Scs, Ses, Sas, Jv, ok = min(good, key=lambda c_: c_[5])
        else:
            Xs, Us, Scs, Ses, Sas, Jv, ok = cands[0]
            if self.guess is not None:                 # fall back to the shifted previous plan
                Us = self.guess[1]; Xs = self.guess[0]
        self.guess = [Xs, Us, Scs, Ses, Sas]
        pred = Xs.copy(); pred[0] += s_off
        return Us[:, 0].copy(), pred, dict(t=dt, ok=ok, slack_col=float(np.max(Scs)) if ok else np.nan)


# =========================================================================
class LTVMPC:
    name = 'LTV-MPC'

    def __init__(self, surface='map'):
        self.surface = surface
        # The friction level is measured/estimated at the CURRENT position and held
        # constant over the horizon (no look-ahead on the surface map).
        x, u, so, w = ca.SX.sym('x', NX), ca.SX.sym('u', NU), ca.SX.sym('so'), ca.SX.sym('w', 4)
        fx = vm.dynamics(x, u, surface, ca, False, so, [w[i] for i in range(4)])[0]
        self.fjac = ca.Function('fjac', [x, u, so, w], [fx, ca.jacobian(fx, x), ca.jacobian(fx, u)])
        g = ca.vertcat(*_tyre_constraints(x, surface, so, ca, [w[i] for i in range(4)]))
        self.gjac = ca.Function('gjac', [x, so, w], [g, ca.jacobian(g, x)])
        self.reset()

    def reset(self):
        self.u_prev = np.zeros(NU)
        self.Xbar = None
        self.Uprev = None

    # variable layout --------------------------------------------------------
    @staticmethod
    def ix(k, i): return k * NX + i
    @staticmethod
    def iu(k, i): return (N + 1) * NX + k * NU + i
    @staticmethod
    def isc(k, j): return (N + 1) * NX + N * NU + k * K + j
    @staticmethod
    def ise(k): return (N + 1) * NX + N * NU + N * K + k
    @staticmethod
    def isa(k, i): return (N + 1) * NX + N * NU + N * K + N + 4 * k + i
    NZ = (N + 1) * NX + N * NU + N * K + N + 4 * N

    def step(self, x_abs, obs, ey_t, v_des):
        t0 = time.perf_counter()
        s_off = x_abs[0]
        x0 = x_abs.copy(); x0[0] = 0.0
        # ---- linearise and discretise at (x0, u_prev) -----------------------
        _, a_now = vm.dynamics(x0, self.u_prev, self.surface, np, False, s_off)
        w_now = np.array([float(v) for v in a_now['w']])
        f0, A, B = [np.array(v) for v in self.fjac(x0, self.u_prev, s_off, w_now)]
        f0 = f0.ravel()
        c = f0 - A @ x0 - B @ self.u_prev
        Mx = np.zeros((NX + NU + 1, NX + NU + 1))
        Mx[:NX, :NX] = A; Mx[:NX, NX:NX + NU] = B; Mx[:NX, -1] = c
        E = expm(Mx * TS)
        Ad, Bd, cd = E[:NX, :NX], E[:NX, NX:NX + NU], E[:NX, -1]
        g0, G = self.gjac(x0, s_off, w_now); g0 = np.array(g0).ravel(); G = np.array(G)
        # ---- reference trajectory for the obstacle linearisation ------------
        if self.Xbar is None:
            Xb = np.zeros((NX, N + 1)); Xb[:, 0] = x0
            for k in range(N):
                Xb[:, k + 1] = Ad @ Xb[:, k] + Bd @ self.u_prev + cd
        else:
            Xb = np.c_[self.Xbar[:, 1:], self.Xbar[:, -1:]].copy()
            Xb[0] -= s_off - self.s_off_prev
            Xb[:, -1] = Ad @ Xb[:, -2] + Bd @ self.u_prev + cd
        # ---- cost ------------------------------------------------------------
        P = np.zeros(self.NZ); q = np.zeros(self.NZ)
        for k in range(1, N + 1):
            for i, w, ref in [(1, W['ey'], ey_t), (2, W['epsi'], 0), (3, W['v'], v_des), (5, W['r'], 0), (7, W['F'], 0)]:
                P[self.ix(k, i)] += 2 * w; q[self.ix(k, i)] += -2 * w * ref
        for k in range(N):
            P[self.iu(k, 0)] += 2 * W['dd']; P[self.iu(k, 1)] += 2 * W['dF']
            for idx in [self.ise(k)] + [self.isa(k, i) for i in range(4)] + [self.isc(k, j) for j in range(K)]:
                P[idx] += 2 * W['slack2']; q[idx] += W['slack']
        Pm = sp.diags(P).tocsc()
        # ---- constraints -----------------------------------------------------
        rows, cols, vals, lo, up = [], [], [], [], []
        r = [0]

        def add(entries, l, u_):
            for cc, v in entries:
                rows.append(r[0]); cols.append(cc); vals.append(v)
            lo.append(l); up.append(u_); r[0] += 1
        for i in range(NX):
            add([(self.ix(0, i), 1.0)], x0[i], x0[i])
        for k in range(N):
            for i in range(NX):
                e = [(self.ix(k + 1, i), 1.0)]
                e += [(self.ix(k, j), -Ad[i, j]) for j in range(NX) if Ad[i, j] != 0]
                e += [(self.iu(k, j), -Bd[i, j]) for j in range(NU) if Bd[i, j] != 0]
                add(e, cd[i], cd[i])
        for k in range(N):
            add([(self.iu(k, 0), 1.0)], -vm.DDELTA_MAX, vm.DDELTA_MAX)
            add([(self.iu(k, 1), 1.0)], -vm.DFX_MAX, vm.DFX_MAX)
            xk = k + 1
            add([(self.ix(xk, 6), 1.0)], -vm.DELTA_MAX, vm.DELTA_MAX)
            add([(self.ix(xk, 7), 1.0)], vm.FX_MIN, vm.FX_MAX)
            add([(self.ix(xk, 3), 1.0)], 0.0, np.inf)
            add([(self.ix(xk, 1), 1.0), (self.ise(k), -1.0)], -np.inf, EY_LIM)
            add([(self.ix(xk, 1), -1.0), (self.ise(k), -1.0)], -np.inf, EY_LIM)
            # tyre constraints  g0 + G (x - x0) <= slack (slip) / 0 (friction)
            for gi in range(NG):
                e = [(self.ix(xk, j), G[gi, j]) for j in range(NX) if G[gi, j] != 0]
                e.append((self.isa(k, SLACK_MAP[gi]), -1.0))
                add(e, -np.inf, -g0[gi] + G[gi] @ x0)
            # obstacles: tangent half-planes of the ellipses at the reference trajectory
            tk = xk * TS
            sb, eb, pb = Xb[0, xk], Xb[1, xk], Xb[2, xk]
            for j in range(K):
                if obs[j, 3] < 0.5:
                    continue
                so = obs[j, 0] + obs[j, 2] * tk
                for o in CIRC_OFF:
                    cs = sb + o * np.cos(pb); cy = eb + o * np.sin(pb)
                    h = ((cs - so) / A_ELL)**2 + ((cy - obs[j, 1]) / B_ELL)**2
                    dhs = 2 * (cs - so) / A_ELL**2; dhy = 2 * (cy - obs[j, 1]) / B_ELL**2
                    dhp = dhs * (-o * np.sin(pb)) + dhy * (o * np.cos(pb))
                    # h + dh.(x - xb) >= 1 - sc   ->  -dh.x - sc <= h - dh.xb - 1
                    add([(self.ix(xk, 0), -dhs), (self.ix(xk, 1), -dhy), (self.ix(xk, 2), -dhp),
                         (self.isc(k, j), -1.0)], -np.inf, h - dhs * sb - dhy * eb - dhp * pb - 1)
        for k in range(N):
            for idx in [self.ise(k)] + [self.isa(k, i) for i in range(4)] + [self.isc(k, j) for j in range(K)]:
                add([(idx, 1.0)], 0.0, np.inf)
        Am = sp.csc_matrix((vals, (rows, cols)), shape=(r[0], self.NZ))
        solver = osqp.OSQP()
        solver.setup(Pm, q, Am, np.array(lo), np.array(up), verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                     max_iter=20000, polish=True)
        res = solver.solve()
        dt = time.perf_counter() - t0
        ok = res.info.status_val in (1, 2)          # solved / solved inaccurate
        if ok:
            z = res.x
            Xs = np.array([[z[self.ix(k, i)] for k in range(N + 1)] for i in range(NX)])
            Us = np.array([[z[self.iu(k, i)] for k in range(N)] for i in range(NU)])
            sc = max(z[self.isc(k, j)] for k in range(N) for j in range(K))
        else:                                   # fall back to the shifted previous plan
            Xs = Xb
            Us = np.c_[self.Uprev[:, 1:], self.Uprev[:, -1:]] if self.Uprev is not None else np.zeros((NU, N))
            sc = np.nan
        self.Uprev = Us
        u = Us[:, 0].copy()
        self.u_prev = u
        self.Xbar = Xs
        self.s_off_prev = s_off
        pred = Xs.copy(); pred[0] += s_off
        return u, pred, dict(t=dt, ok=ok, slack_col=sc)
