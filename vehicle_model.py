"""
Vehicle, tyre, road and surface models for the rainy-highway overtaking study.

Vehicle: nonlinear FOUR-WHEEL (double-track) model written in the road-aligned
(Frenet) frame. Every wheel has its own position, its own surface (dry / wet),
its own normal load (longitudinal AND lateral load transfer), its own slip angle
and its own Pacejka "Magic Formula" force with combined slip (friction ellipse).
Braking/drive force is shared 60/40 front/rear and equally left/right, and each
wheel is limited by its own friction (ABS) - so with one side on dry and the
other on wet asphalt ("mu-split") braking produces a yaw moment, as in reality.
Parameters m, Iz, lf, lr are those of support_files_car.py (Mark Misin).

  state  x = [s, ey, epsi, vx, vy, r, delta, Fx]      (Fx in kN)
  input  u = [delta_dot (rad/s), Fx_dot (kN/s)]

  s     distance along the road centre line          [m]
  ey    lateral offset from the centre line (+ left) [m]
  epsi  heading error w.r.t. the road                [rad]
  vx,vy body-frame velocities                        [m/s]
  r     yaw rate                                     [rad/s]
  delta front steering angle                         [rad]
  Fx    total longitudinal tyre force (+drive/-brake) [kN]

Surface: everything is wet (rain) except the LEFT lane on the stretch that is
covered by the upper deck of the overpass, which is dry.  The surface map is
a smooth (sigmoid) function so the NMPC can differentiate it; the simulated
"real" car uses a much sharper version of the same map.
"""
import numpy as np
import casadi as ca

# ------------------------------------------------------------- vehicle ----
M, IZ, LF, LR = 1500.0, 3000.0, 2.0, 3.0      # from support_files_car.py
L = LF + LR
H_CG = 0.55            # centre-of-gravity height [m]
TW = 1.60              # track width [m]
# wheel order: FL, FR, RL, RR   (x forward, y left, body frame)
WX = [LF, LF, -LR, -LR]
WY = [TW / 2, -TW / 2, TW / 2, -TW / 2]
WHEELS = ['FL', 'FR', 'RL', 'RR']
G = 9.81
SPLIT_F = 0.6          # share of the longitudinal force on the front axle
C_DRAG = 0.5 * 1.2 * 0.70   # 0.5*rho*Cd*A  [N/(m/s)^2]
C_ROLL = 0.015
CAR_LEN, CAR_WID = 4.6, 1.8       # body used for collision checks (traffic and ego)
EGO_FRONT, EGO_REAR = LF + 0.3, LR - 0.7   # ego body: 2.3 m ahead / 2.3 m behind the CG

# actuator limits
DELTA_MAX = 0.5                    # rad
DDELTA_MAX = 0.5                   # rad/s
FX_MIN, FX_MAX = -M * G / 1000, 0.35 * M * G / 1000    # kN (brake / drive)
DFX_MAX = M * G / 1000 / 0.25      # kN/s  (full braking in 0.25 s)

# ---------------------------------------------------------------- tyres ---
TYRE_DRY = dict(B=10.0, C=1.9, D=1.00, E=0.97)   # dry asphalt (typical Magic Formula set)
TYRE_WET = dict(B=12.0, C=2.3, D=0.60, E=1.00)   # wet asphalt, heavy rain (peak mu 0.6)


def mf(alpha, t, lib=np):
    """Normalised lateral force Fy/Fz (pure slip)."""
    Ba = t['B'] * alpha
    return t['D'] * lib.sin(t['C'] * lib.arctan(Ba - t['E'] * (Ba - lib.arctan(Ba))))


def alpha_peak(t):
    a = np.linspace(0, 0.6, 60001)
    return float(a[np.argmax(mf(a, t))])


A_PK_DRY, A_PK_WET = alpha_peak(TYRE_DRY), alpha_peak(TYRE_WET)

# ----------------------------------------------------------------- road ---
LANE_W = 3.5
LANE_R, LANE_L = -LANE_W / 2, LANE_W / 2          # lane centres (two lanes, + is left)
ROAD_HALF = LANE_W                                  # road edges at +-3.5 m
S_MAX = 2800.0


def sig(x, lib=np):
    return 0.5 * (1 + lib.tanh(0.5 * x))


def kappa(s, lib=np):
    """Gentle highway: left bend R=900 m, right bend R=1200 m, left bend R=1000 m."""
    return (1 / 900 * (sig((s - 150) / 20, lib) - sig((s - 550) / 20, lib))
            - 1 / 1200 * (sig((s - 750) / 20, lib) - sig((s - 1250) / 20, lib))
            + 1 / 1000 * (sig((s - 1500) / 20, lib) - sig((s - 2000) / 20, lib)))


def _centreline():
    s = np.arange(0, S_MAX + 0.25, 0.25)
    psi = np.concatenate([[0], np.cumsum(0.5 * (kappa(s[1:]) + kappa(s[:-1])) * 0.25)])
    X = np.concatenate([[0], np.cumsum(0.5 * (np.cos(psi[1:]) + np.cos(psi[:-1])) * 0.25)])
    Y = np.concatenate([[0], np.cumsum(0.5 * (np.sin(psi[1:]) + np.sin(psi[:-1])) * 0.25)])
    return s, X, Y, psi


_S, _X, _Y, _PSI = _centreline()


def to_cartesian(s, ey, epsi=0.0):
    s = np.asarray(s, float)
    Xc, Yc, pc = np.interp(s, _S, _X), np.interp(s, _S, _Y), np.interp(s, _S, _PSI)
    return Xc - ey * np.sin(pc), Yc + ey * np.cos(pc), pc + epsi


# -------------------------------------------------------------- surface ---
# surface descriptions:
#   'map'     : rain everywhere, left lane dry for S_DRY[0] < s < S_DRY[1] (under the overpass)
#   'all_wet' : controller assumes wet everywhere
#   'all_dry' : controller assumes dry everywhere (rain not detected)
S_DRY_SECTIONS = [(-60.0, 600.0), (740.0, 1150.0), (1300.0, 2300.0)]   # overpass decks over the left lane
S_DRY = S_DRY_SECTIONS[0]


def dry_weight(s, ey, surface, lib=np, sharp=False):
    if surface == 'all_wet':
        return 0.0 * s
    if surface == 'all_dry':
        return 0.0 * s + 1.0
    kl, ks = (0.02, 0.2) if sharp else (0.25, 2.0)
    under = 0
    for a, b in S_DRY_SECTIONS:
        under = under + sig((s - a) / ks, lib) * sig((b - s) / ks, lib)
    return sig(ey / kl, lib) * under


# ------------------------------------------------------------ dynamics ----
def dynamics(x, u, surface='map', lib=np, sharp=False, s_off=0.0, w_fixed=None):
    """Continuous-time four-wheel model. Returns (xdot, aux). Works with numpy or casadi.
    w_fixed: optional list of 4 dry-weights (surface frozen, e.g. estimated now)."""
    s, ey, epsi, vx, vy, r, d, Fk = [x[i] for i in range(8)]
    fmin_ = lib.fmin if lib is ca else np.minimum
    fmax_ = lib.fmax if lib is ca else np.maximum
    sa = s + s_off
    Fx = 1000.0 * Fk
    Fres = C_DRAG * vx**2 + C_ROLL * M * G
    ax_est = (Fx - Fres) / M
    ay_est = vx * r
    dz_long = M * ax_est * H_CG / L / 2                       # per wheel, front -> rear
    dz_lat_f = M * ay_est * H_CG / TW * (LR / L)              # per wheel, inner -> outer
    dz_lat_r = M * ay_est * H_CG / TW * (LF / L)
    Fz = [M * G * LR / L / 2 - dz_long - dz_lat_f, M * G * LR / L / 2 - dz_long + dz_lat_f,
          M * G * LF / L / 2 + dz_long - dz_lat_r, M * G * LF / L / 2 + dz_long + dz_lat_r]
    Fz = [fmax_(f, 150.0) for f in Fz]
    req = [SPLIT_F * Fx / 2, SPLIT_F * Fx / 2, (1 - SPLIT_F) * Fx / 2, (1 - SPLIT_F) * Fx / 2]
    alpha, rho, Fxw, Fyw, mu, w, apk = [], [], [], [], [], [], []
    FXb, FYb, MZ = 0, 0, 0
    cd, sd = lib.cos(d), lib.sin(d)
    for i in range(4):
        xw, yw = WX[i], WY[i]
        if w_fixed is None:
            sw = sa + xw * lib.cos(epsi) - yw * lib.sin(epsi)
            eyw = ey + xw * lib.sin(epsi) + yw * lib.cos(epsi)
            wi = dry_weight(sw, eyw, surface, lib, sharp)
        else:
            wi = w_fixed[i]
        mui = wi * TYRE_DRY['D'] + (1 - wi) * TYRE_WET['D']
        vxw = fmax_(vx - r * yw, 1.0)
        vyw = vy + r * xw
        steer = d if i < 2 else 0.0
        a = steer - lib.arctan(vyw / vxw)
        cap = 0.99 * mui * Fz[i]
        fx = fmin_(fmax_(req[i], -cap), cap)                  # ABS / traction control per wheel
        rh = fx / (mui * Fz[i])
        fy = Fz[i] * (wi * mf(a, TYRE_DRY, lib) + (1 - wi) * mf(a, TYRE_WET, lib)) * lib.sqrt(fmax_(1 - rh**2, 1e-3))
        if i < 2:
            fxb, fyb = fx * cd - fy * sd, fx * sd + fy * cd
        else:
            fxb, fyb = fx, fy
        FXb = FXb + fxb; FYb = FYb + fyb; MZ = MZ + xw * fyb - yw * fxb
        alpha.append(a); rho.append(rh); Fxw.append(fx); Fyw.append(fy); mu.append(mui); w.append(wi)
        apk.append(wi * A_PK_DRY + (1 - wi) * A_PK_WET)
    k = kappa(sa, lib)
    sdot = (vx * lib.cos(epsi) - vy * lib.sin(epsi)) / (1 - k * ey)
    xdot = [sdot,
            vx * lib.sin(epsi) + vy * lib.cos(epsi),
            r - k * sdot,
            (FXb - Fres) / M + vy * r,
            FYb / M - vx * r,
            MZ / IZ,
            u[0],
            u[1]]
    aux = dict(alpha=alpha, rho=rho, Fx=Fxw, Fy=Fyw, Fz=Fz, mu=mu, w=w, apk=apk, Mz=MZ)
    return (ca.vertcat(*xdot) if lib is ca else np.array(xdot, dtype=float)), aux


def rk4(f, x, u, h):
    k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def plant_step(x, u, Ts, sub=20):
    """The simulated real car: sharp surface map, fine integration, physical saturations."""
    u = np.array([np.clip(u[0], -DDELTA_MAX, DDELTA_MAX), np.clip(u[1], -DFX_MAX, DFX_MAX)])

    def f(xx, uu):
        xx = xx.copy()
        if xx[3] < 0.3 and xx[7] < 0:
            xx[7] = 0.0
        dx, _ = dynamics(xx, uu, 'map', np, sharp=True)
        if xx[3] < 0.3 and dx[3] < 0:
            dx[3] = 0.0; dx[0] = max(dx[0], 0.0)
        return dx
    for _ in range(sub):
        x = rk4(f, x, u, Ts / sub)
        x[6] = np.clip(x[6], -DELTA_MAX, DELTA_MAX)
        x[7] = np.clip(x[7], FX_MIN, FX_MAX)
        x[3] = max(x[3], 0.0)
    return x


def utilisation(x):
    """Friction utilisation sqrt(Fx^2+Fy^2)/(mu*Fz) of each wheel [FL, FR, RL, RR] in the real car."""
    _, a = dynamics(x, [0, 0], 'map', np, sharp=True)
    u = np.array([np.hypot(a['Fx'][i], a['Fy'][i]) / (a['mu'][i] * a['Fz'][i]) for i in range(4)])
    return u, a
