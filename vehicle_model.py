"""
Vehicle Model
-------

The road and its wet/dry map are predefined, but the car's path is not: the controller plans it online every 0.1 s (local path planning).
At each step we find where all 4 wheels are on the road and check wet or dry under each one, so during a lane change the wheels on one side can be on dry while the other side is still on wet.
The tyre force comes from the Pacejka "Magic Formula". tyre has a limited amount of traction to share between braking and turning; the more it uses for braking, the less is left for steering.

Brakes and drive are split 60/40 front/rear and evenly left/right. Every wheel has
its own ABS, so when the left wheels are on dry asphalt and the right ones on wet
("mu-split"), braking twists the car toward the grippy side; just like in real life.

  state  x = [s, ey, epsi, vx, vy, r, delta, Fx]      (Fx in kN)
  input  u = [delta_dot (rad/s), Fx_dot (kN/s)]

  s     how far along the road we are               [m]
  ey    sideways offset from the centre line (+left) [m]
  epsi  heading relative to the road                 [rad]
  vx,vy speed forward / sideways (car frame)         [m/s]
  r     yaw rate (how fast the car turns)            [rad/s]
  delta front wheel steering angle                   [rad]
  Fx    total tyre force along the car (+gas/-brake) [kN]

The road
--------
It's raining, so everything is wet – except the LEFT lane wherever an overpass
sits on top of it; that part stays dry. The NMPC gets a smooth version of this
wet/dry map (so it can take derivatives), the "real" simulated car gets a sharp one.
"""
import numpy as np
import casadi as ca

# ------------------------------------------------------------- vehicle ----
M, IZ, LF, LR = 1500.0, 3000.0, 2.0, 3.0      # mass [kg], yaw inertia [kg m^2], CG to front / rear axle [m]
L = LF + LR
H_CG = 0.55            # how high the centre of gravity sits [m]
TW = 1.60              # distance between left and right wheels [m]
# wheels: front-left, front-right, rear-left, rear-right   (x forward, y left)
WX = [LF, LF, -LR, -LR]
WY = [TW / 2, -TW / 2, TW / 2, -TW / 2]
WHEELS = ['FL', 'FR', 'RL', 'RR']
G = 9.81
SPLIT_F = 0.6          # 60 % of braking/drive goes to the front wheels
C_DRAG = 0.5 * 1.2 * 0.70   # air drag: 0.5*rho*Cd*A  [N/(m/s)^2]
C_ROLL = 0.015              # rolling resistance
CAR_LEN, CAR_WID = 4.6, 1.8       # car size used for crash checks (all cars)
EGO_FRONT, EGO_REAR = LF + 0.3, LR - 0.7   # our car sticks out 2.3 m in front of and behind its CG

# what the steering and brakes can physically do
DELTA_MAX = 0.5                    # max steering angle [rad]
DDELTA_MAX = 0.5                   # max steering speed [rad/s]
FX_MIN, FX_MAX = -M * G / 1000, 0.35 * M * G / 1000    # full brake / full gas [kN]
DFX_MAX = M * G / 1000 / 0.25      # brakes need 0.25 s to go from zero to full [kN/s]

# ---------------------------------------------------------------- tyres ---
TYRE_DRY = dict(B=10.0, C=1.9, D=1.00, E=0.97)   # dry asphalt (typical Magic Formula values)
TYRE_WET = dict(B=12.0, C=2.3, D=0.60, E=1.00)   # wet asphalt in heavy rain (max grip 0.6)


def mf(alpha, t, lib=np):
    """Sideways tyre force per unit of load (Fy/Fz) for a given slip angle."""
    Ba = t['B'] * alpha
    return t['D'] * lib.sin(t['C'] * lib.arctan(Ba - t['E'] * (Ba - lib.arctan(Ba))))


def alpha_peak(t):
    """Slip angle where the tyre gives its maximum grip – beyond this it starts sliding."""
    a = np.linspace(0, 0.6, 60001)
    return float(a[np.argmax(mf(a, t))])


A_PK_DRY, A_PK_WET = alpha_peak(TYRE_DRY), alpha_peak(TYRE_WET)

# ----------------------------------------------------------------- road ---
LANE_W = 3.5
LANE_R, LANE_L = -LANE_W / 2, LANE_W / 2          # middle of the right / left lane
ROAD_HALF = LANE_W                                  # road edges are 3.5 m from the centre line
S_MAX = 2800.0


def sig(x, lib=np):
    """Smooth step from 0 to 1 (a soft on/off switch)."""
    return 0.5 * (1 + lib.tanh(0.5 * x))


def kappa(s, lib=np):
    """Road curvature: a gentle highway – left bend (R 900 m), right bend (R 1200 m), left bend (R 1000 m)."""
    return (1 / 900 * (sig((s - 150) / 20, lib) - sig((s - 550) / 20, lib))
            - 1 / 1200 * (sig((s - 750) / 20, lib) - sig((s - 1250) / 20, lib))
            + 1 / 1000 * (sig((s - 1500) / 20, lib) - sig((s - 2000) / 20, lib)))


def _centreline():
 """Compute the x/y shape of the road from its curvature. This is only used to draw the road in the video and doesn't affect the simulation."""
    s = np.arange(0, S_MAX + 0.25, 0.25)
    psi = np.concatenate([[0], np.cumsum(0.5 * (kappa(s[1:]) + kappa(s[:-1])) * 0.25)])
    X = np.concatenate([[0], np.cumsum(0.5 * (np.cos(psi[1:]) + np.cos(psi[:-1])) * 0.25)])
    Y = np.concatenate([[0], np.cumsum(0.5 * (np.sin(psi[1:]) + np.sin(psi[:-1])) * 0.25)])
    return s, X, Y, psi


_S, _X, _Y, _PSI = _centreline()


def to_cartesian(s, ey, epsi=0.0):
    """Road coordinates (s, ey) -> map coordinates (X, Y) and heading."""
    s = np.asarray(s, float)
    Xc, Yc, pc = np.interp(s, _S, _X), np.interp(s, _S, _Y), np.interp(s, _S, _PSI)
    return Xc - ey * np.sin(pc), Yc + ey * np.cos(pc), pc + epsi


# -------------------------------------------------------------- surface ---
# Which road does a controller "believe" in?
#   'map'     : the real thing – wet everywhere, but the left lane is dry under the overpasses
#   'all_wet' : assume it's wet everywhere
#   'all_dry' : assume it's dry everywhere (as if the rain wasn't noticed)
S_DRY_SECTIONS = [(-60.0, 600.0), (740.0, 1150.0), (1300.0, 2300.0)]   # where the overpasses cover the left lane [m]
S_DRY = S_DRY_SECTIONS[0]


def dry_weight(s, ey, surface, lib=np, sharp=False):
    """1 = dry, 0 = wet, at road position (s, ey). sharp=True gives a crisp edge (used by the real car)."""
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
    """The heart of it: how the car moves right now. Returns (xdot, extra info).
    Works with plain numbers (numpy) and with CasADi symbols (for the NMPC).
    w_fixed: optionally force the wet/dry value of the 4 wheels instead of looking it up."""
    s, ey, epsi, vx, vy, r, d, Fk = [x[i] for i in range(8)]
    fmin_ = lib.fmin if lib is ca else np.minimum
    fmax_ = lib.fmax if lib is ca else np.maximum
    sa = s + s_off
    Fx = 1000.0 * Fk
    Fres = C_DRAG * vx**2 + C_ROLL * M * G
    ax_est = (Fx - Fres) / M
    ay_est = vx * r
    dz_long = M * ax_est * H_CG / L / 2                       # braking shifts weight to the front wheels
    dz_lat_f = M * ay_est * H_CG / TW * (LR / L)              # turning shifts weight to the outer wheels
    dz_lat_r = M * ay_est * H_CG / TW * (LF / L)
    Fz = [M * G * LR / L / 2 - dz_long - dz_lat_f, M * G * LR / L / 2 - dz_long + dz_lat_f,
          M * G * LF / L / 2 + dz_long - dz_lat_r, M * G * LF / L / 2 + dz_long + dz_lat_r]
    Fz = [fmax_(f, 150.0) for f in Fz]                        # a wheel never fully lifts off
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
        fx = fmin_(fmax_(req[i], -cap), cap)                  # ABS: a wheel can't brake harder than its grip allows
        rh = fx / (mui * Fz[i])
        # sideways force, reduced when the same tyre is also braking (friction ellipse)
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
    """Move the "real" car forward by one control step (fine time steps, crisp wet/dry edges)."""
    u = np.array([np.clip(u[0], -DDELTA_MAX, DDELTA_MAX), np.clip(u[1], -DFX_MAX, DFX_MAX)])

    def f(xx, uu):
        xx = xx.copy()
        if xx[3] < 0.3 and xx[7] < 0:             # standing still: brakes hold, no rolling backwards
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
    """How much of the available traction each wheel is using right now (0 = none, 1 = at the limit)."""
    _, a = dynamics(x, [0, 0], 'map', np, sharp=True)
    u = np.array([np.hypot(a['Fx'][i], a['Fy'][i]) / (a['mu'][i] * a['Fz'][i]) for i in range(4)])
    return u, a
