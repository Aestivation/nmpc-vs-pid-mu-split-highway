"""
YouTube video renderer (1920x1080, 30 fps) for the rainy-highway NMPC vs LTV-MPC run.
  python render_video.py test            -> a few still frames (png)
  python render_video.py chunk i n       -> render chunk i of n into chunk_i.mp4
  python render_video.py concat n        -> join chunks into final video
"""
import sys, pickle, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Ellipse, FancyBboxPatch, Circle
from matplotlib import cm
import vehicle_model as vm

FPS = 30
W_PX, H_PX = 1920, 1080
BG = '#0b1120'
GRASS = '#18251c'
ASPH_WET = '#262b33'
ASPH_DRY = '#4b4f57'
SHOULDER = '#353a42'
C_N, C_L = '#3b82f6', '#f59e0b'
TXT = '#e5e7eb'
SUB = '#94a3b8'
UTIL_CMAP = cm.get_cmap('RdYlGn_r')

INTRO_S, OUTRO_S = 7.0, 22.0

# ------------------------------------------------------------------ data ---
# 'ltv' slot now holds the classical PID + Stanley run
D = {'nmpc': pickle.load(open('cut3_nmpc.pkl', 'rb')), 'ltv': pickle.load(open('cut3_classic.pkl', 'rb'))}
SCN = D['nmpc']['scn']
NAME = {'nmpc': 'NMPC', 'ltv': 'PID + Stanley'}
import scenario_youtube as _Y
FINISH_S = _Y.FINISH_S
for _k in D:                      # cut every log exactly at the finish line (same end point for both)
    _L = D[_k]['log']; _n = int(np.searchsorted(_L['x'][:, 0], FINISH_S)) + 1
    for _f in ('t', 'x'):
        _L[_f] = _L[_f][:_n + 1]
    for _f in ('pred', 'solve', 'ok', 'slack', 'target'):
        _L[_f] = _L[_f][:_n]
    for _f in ('ax', 'ay', 'util'):
        D[_k]['ev'][_f] = D[_k]['ev'][_f][:_n + 1]
T_END = {k: float(np.interp(FINISH_S, D[k]['log']['x'][:, 0], D[k]['log']['t'])) for k in D}
CUTS = {k: [c[0] for c in D[k]['log']['mem']['cuts']] for k in D}     # cut-in times (same road positions)
CUT_S = [c[1] for c in D['nmpc']['log']['mem']['cuts']]
T_SIM = max(T_END.values()) + 1.5
N_SIM = int(T_SIM * FPS)
N_INTRO, N_OUTRO = int(INTRO_S * FPS), int(OUTRO_S * FPS)
REPLAY_S, REPLAY_SPEED = 14.0, 0.25
N_REPLAY = int(REPLAY_S * FPS)
N_TOTAL = N_INTRO + N_SIM + N_REPLAY + N_OUTRO


def state_at(key, t):
    L = D[key]['log']; tt = L['t']
    k = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2))
    a = np.clip((t - tt[k]) / (tt[k + 1] - tt[k]), 0, 1)
    return (1 - a) * L['x'][k] + a * L['x'][k + 1], k


def lane_change_events(key):
    X = D[key]['log']['x']; t = D[key]['log']['t']
    ev = []
    for k in np.where(np.diff(np.sign(X[:, 1])) != 0)[0]:
        s = X[k, 0]
        to_left = X[k + 1, 1] > 0
        dry_left = vm.dry_weight(s, 1.75, 'map', np, True) > 0.5
        if to_left:
            txt = 'wet → dry (under the overpass)' if dry_left else 'wet → wet (open sky)'
        else:
            txt = 'dry → wet' if dry_left else 'wet → wet'
        ev.append((t[k], 'right → left' if to_left else 'left → right', txt))
    return ev


EVENTS = lane_change_events('nmpc')


def _nmpc_reactions():
    L = D['nmpc']['log']; X = L['x']; t = L['t']; out = []
    for ct in CUTS['nmpc']:
        m = (t > ct) & (t < ct + 4.0)
        evaded = np.any(X[m, 1] < 0)
        out.append((ct, 'predicts the cut-in  →  steers into the free gap, keeps its speed' if evaded
                    else 'predicts the cut-in  →  brakes early and gently, no emergency'))
    return out


NM_REACT = _nmpc_reactions()


def nmpc_msg(t):
    for ct, msg in NM_REACT:
        if ct + 0.6 <= t <= ct + 3.4:
            return msg
    return None
EVENTS_L = lane_change_events('ltv')

# precomputed decoration: puddles in the wet area
rng = np.random.default_rng(3)
PUDDLES = []
for s in np.arange(0, 2400, 6.0):
    ey = rng.uniform(-3.2, 3.2)
    if vm.dry_weight(s, ey, 'map', np, True) < 0.5:
        PUDDLES.append((s + rng.uniform(-2, 2), ey, rng.uniform(1.0, 2.6), rng.uniform(0.4, 0.9)))
RAIN = np.c_[rng.uniform(0, 1, 420), rng.uniform(0, 1, 420), rng.uniform(0.6, 1.0, 420)]
TRAFFIC_COL = ['#d1d5db', '#9ca3af', '#e5e7eb', '#b91c1c', '#6b7280', '#f3f4f6', '#64748b', '#991b1b',
               '#cbd5e1', '#a1a1aa', '#e2e8f0', '#78716c', '#d6d3d1']

# ------------------------------------------------------------ figure ------
fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100, facecolor=BG)
ax_views = {'nmpc': fig.add_axes([0.0, 0.635, 0.70, 0.365]), 'ltv': fig.add_axes([0.0, 0.27, 0.70, 0.365])}
ax_rain = fig.add_axes([0.0, 0.27, 0.70, 0.73]); ax_rain.set_zorder(5)
ax_hud = fig.add_axes([0.70, 0.0, 0.30, 1.0])
ax_map = fig.add_axes([0.01, 0.02, 0.27, 0.235])
ax_plt = fig.add_axes([0.33, 0.045, 0.355, 0.20])
ax_card = fig.add_axes([0, 0, 1, 1]); ax_card.set_zorder(10)
VIEW_X = (-18.0, 32.0)
_hy = (VIEW_X[1] - VIEW_X[0]) * 0.365 * H_PX / (0.70 * W_PX) / 2
VIEW_Y = (-_hy, _hy)
CAM_EY = 2.0


def body_shape(L=4.6, W=1.8):
    c = 0.45
    return np.array([[L/2 - c, W/2], [L/2, W/2 - c], [L/2, -W/2 + c], [L/2 - c, -W/2],
                     [-L/2 + 0.2, -W/2], [-L/2, -W/2 + 0.2], [-L/2, W/2 - 0.2], [-L/2 + 0.2, W/2]])


def local_to_world(pts, X, Y, h):
    c, s = np.cos(h), np.sin(h)
    return np.c_[X + pts[:, 0] * c - pts[:, 1] * s, Y + pts[:, 0] * s + pts[:, 1] * c]


class Cam:
    def __init__(self, s_c):
        X, Y, h = vm.to_cartesian(s_c, CAM_EY)
        self.X, self.Y, self.h = float(X), float(Y), float(h)
        self.c, self.s = np.cos(-self.h), np.sin(-self.h)

    def __call__(self, P):
        P = np.atleast_2d(P); dx, dy = P[:, 0] - self.X, P[:, 1] - self.Y
        return np.c_[dx * self.c - dy * self.s, dx * self.s + dy * self.c]


def frenet_poly(cam, s0, s1, ey0, ey1, n=30):
    ss = np.linspace(s0, s1, n)
    Xa, Ya, _ = vm.to_cartesian(ss, ey0); Xb, Yb, _ = vm.to_cartesian(ss, ey1)
    return cam(np.r_[np.c_[Xa, Ya], np.c_[Xb, Yb][::-1]])


def draw_car(ax, cam, s, ey, epsi, color, alpha=1.0, z=6, L=4.6, W=1.8, ego=None, braking=False):
    X, Y, h = vm.to_cartesian(s, ey, epsi)
    if ego is None:      # traffic: body is centred
        off = 0.0
    else:                # ego: CG is 2.0 m behind the front axle; body centred 0 m from CG (EGO_FRONT=EGO_REAR)
        off = 0.0
    body = cam(local_to_world(body_shape(L, W) + [off, 0], X, Y, h))
    ax.add_patch(Polygon(body, closed=True, fc=color, ec='#0f172a', lw=1.2, alpha=alpha, zorder=z))
    wind = np.array([[0.9, 0.72], [1.45, 0.62], [1.45, -0.62], [0.9, -0.72]]) + [off, 0]
    rear = np.array([[-1.25, 0.7], [-1.6, 0.6], [-1.6, -0.6], [-1.25, -0.7]]) + [off, 0]
    for p in (wind, rear):
        ax.add_patch(Polygon(cam(local_to_world(p, X, Y, h)), closed=True, fc='#0f172a', alpha=0.75 * alpha, zorder=z + 0.1))
    for yy in (0.62, -0.62):   # tail lights (rain -> lights on)
        tl = np.array([[-2.3, yy + 0.18], [-2.18, yy + 0.18], [-2.18, yy - 0.18], [-2.3, yy - 0.18]]) + [off, 0]
        ax.add_patch(Polygon(cam(local_to_world(tl, X, Y, h)), closed=True, fc='#ff1a1a' if braking else '#b91c1c',
                             alpha=alpha, zorder=z + 0.2))
        if braking:      # brake-light glow
            P = cam(local_to_world(np.array([[-2.4, yy]]), X, Y, h))[0]
            ax.add_patch(Circle(P, 1.1, fc='#ff2a2a', alpha=0.35, lw=0, zorder=z + 0.15))
    return X, Y, h


def draw_wheels(ax, cam, x, util, z=8):
    X, Y, h = vm.to_cartesian(x[0], x[1], x[2])
    for i in range(4):
        wl, ww = 0.75, 0.32
        p = np.array([[wl/2, ww/2], [wl/2, -ww/2], [-wl/2, -ww/2], [-wl/2, ww/2]])
        if i < 2:
            cd, sd = np.cos(x[6] * 3), np.sin(x[6] * 3)    # steering drawn 3x for visibility
            p = p @ np.array([[cd, sd], [-sd, cd]])
        p = p + [vm.WX[i] * 0.86, vm.WY[i] * 1.12]
        ax.add_patch(Polygon(cam(local_to_world(p, X, Y, h)), closed=True, fc=UTIL_CMAP(min(util[i], 1.0)),
                             ec='white', lw=0.8, zorder=z))


def spray(ax, cam, s, ey, frame, z=4):
    if vm.dry_weight(s, ey, 'map', np, True) > 0.5:
        return
    r = np.random.default_rng(frame * 7 + int(s * 10) % 1000)
    for _ in range(9):
        ds = -2.6 - r.uniform(0, 5.5); de = r.uniform(-1.3, 1.3)
        X, Y, _ = vm.to_cartesian(s + ds, ey + de)
        P = cam(np.array([[X, Y]]))[0]
        ax.add_patch(Circle(P, r.uniform(0.25, 0.7), fc='white', alpha=r.uniform(0.05, 0.14), lw=0, zorder=z))


def draw_view(key, t, frame):
    ax = ax_views[key]; ax.cla(); ax.set_facecolor(GRASS)
    xe, ke = state_at(key, t)
    name, ecol = NAME[key], (C_N if key == 'nmpc' else C_L)
    t_raw = t; t = min(t, T_END[key])
    events = EVENTS if key == 'nmpc' else EVENTS_L
    s_c = xe[0] + 7.0
    cam = Cam(s_c)
    s0, s1 = s_c + VIEW_X[0] - 25, s_c + VIEW_X[1] + 25
    # shoulders, road, lanes
    ax.add_patch(Polygon(frenet_poly(cam, s0, s1, -5.0, 5.0), fc=SHOULDER, lw=0, zorder=1))
    ax.add_patch(Polygon(frenet_poly(cam, s0, s1, -3.5, 3.5), fc=ASPH_WET, lw=0, zorder=1.1))
    for a, b in vm.S_DRY_SECTIONS:
        if b > s0 and a < s1:
            ax.add_patch(Polygon(frenet_poly(cam, max(a, s0), min(b, s1), 0.0, 3.5), fc=ASPH_DRY, lw=0, zorder=1.2))
    for (ps, pe, pw, ph) in PUDDLES:
        if s0 < ps < s1:
            X, Y, hh = vm.to_cartesian(ps, pe)
            P = cam(np.array([[X, Y]]))[0]
            ax.add_patch(Ellipse(P, pw, ph, angle=np.degrees(hh - cam.h), fc='#64748b', alpha=0.28, lw=0, zorder=1.3))
    for ey in (3.5, -3.5):
        ss = np.linspace(s0, s1, 40); Xe, Ye, _ = vm.to_cartesian(ss, ey); P = cam(np.c_[Xe, Ye])
        ax.plot(P[:, 0], P[:, 1], color='#e5e7eb', lw=2.2, zorder=2)
    d0 = np.floor(s0 / 12) * 12
    for sd in np.arange(d0, s1, 12.0):
        ss = np.linspace(sd, sd + 4, 4); Xe, Ye, _ = vm.to_cartesian(ss, 0.0); P = cam(np.c_[Xe, Ye])
        ax.plot(P[:, 0], P[:, 1], color='#f8fafc', lw=2.2, zorder=2)
    if s0 < FINISH_S < s1:        # chequered finish line
        for j in range(14):
            for q in range(2):
                e0 = -3.5 + j * 0.5
                P = frenet_poly(cam, FINISH_S + q * 0.5, FINISH_S + q * 0.5 + 0.5, e0, e0 + 0.5, n=2)
                ax.add_patch(Polygon(P, fc='white' if (j + q) % 2 == 0 else '#111827', lw=0, zorder=2.5))
    # predicted trajectory
    for col, xs, kk in ((ecol, xe, ke),):
        pr = D[key]['log']['pred'][min(kk, len(D[key]['log']['pred']) - 1)]
        X, Y, _ = vm.to_cartesian(pr[0], pr[1]); P = cam(np.c_[X, Y])
        ax.plot(P[:, 0], P[:, 1], color=col, lw=7, alpha=0.18, zorder=3, solid_capstyle='round')
        ax.plot(P[:, 0], P[:, 1], color=col, lw=2, alpha=0.9, zorder=3.1)
    # traffic
    tr_now, tr_prev = D[key]['scn'].traffic_at(t), D[key]['scn'].traffic_at(max(t - 0.2, 0))
    for i, (v, vp) in enumerate(zip(tr_now, tr_prev)):
        if s0 - 5 < v['s'] < s1 + 5:
            spray(ax, cam, v['s'], v['ey'], frame)
            yaw = np.arctan2(v['ey'] - vp['ey'], max(v['s'] - vp['s'], 1e-3))
            col = '#ef4444' if v['kind'] == 'cut-in' else TRAFFIC_COL[i % len(TRAFFIC_COL)]
            draw_car(ax, cam, v['s'], v['ey'], yaw, col)
    # ego
    spray(ax, cam, xe[0], xe[1], frame + 3)
    axl = D[key]['ev']['ax']; ka = min(int(round(t / 0.1)), len(axl) - 1)
    draw_car(ax, cam, xe[0], xe[1], xe[2], ecol, z=6.5, ego=True, braking=axl[ka] < -1.5)
    draw_wheels(ax, cam, xe, vm.utilisation(xe)[0], z=8)
    # overpass deck (semi transparent) with pillars and shadow edges
    for a, b in vm.S_DRY_SECTIONS:
        if b > s0 and a < s1:
            P = frenet_poly(cam, max(a, s0), min(b, s1), 0.0, 13.0)
            ax.add_patch(Polygon(P, fc='#94a3b8', alpha=0.16, lw=0, zorder=9))
            for edge in (a, b):
                if s0 < edge < s1:
                    ss = np.full(2, edge); Xe, Ye, _ = vm.to_cartesian(ss, np.array([0.0, 13.0])); Pe = cam(np.c_[Xe, Ye])
                    ax.plot(Pe[:, 0], Pe[:, 1], color='#cbd5e1', lw=5, alpha=0.6, zorder=9.1)
            for sp in np.arange(np.ceil(max(a, s0) / 30) * 30, min(b, s1), 30.0):
                for eyp in (4.4, 11.5):
                    Xp, Yp, hp = vm.to_cartesian(sp, eyp); Pp = cam(np.array([[Xp, Yp]]))[0]
                    ax.add_patch(Rectangle((Pp[0] - 0.8, Pp[1] - 0.8), 1.6, 1.6, fc='#475569', ec='#94a3b8', lw=1, zorder=9.2))
            mid = 0.5 * (max(a, s0) + min(b, s1))
            Xm, Ym, _ = vm.to_cartesian(mid, 8.0); Pm = cam(np.array([[Xm, Ym]]))[0]
            if VIEW_X[0] + 14 < Pm[0] < VIEW_X[1] - 22:
                ax.text(Pm[0], Pm[1], 'OVERPASS  ·  left lane DRY underneath', color='#e2e8f0', fontsize=12,
                        ha='center', va='center', alpha=0.85, zorder=9.3, fontweight='bold')
    ax.set_xlim(*VIEW_X); ax.set_ylim(*VIEW_Y); ax.set_aspect('equal'); ax.axis('off')
    # overlay texts
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=ecol, lw=3, zorder=30))
    ax.text(0.012, 0.95, name, transform=ax.transAxes, color='white', fontsize=20, fontweight='bold', va='top', zorder=20,
            bbox=dict(boxstyle='round,pad=0.35', fc=ecol, ec='none'))
    under = vm.dry_weight(xe[0], 1.75, 'map', np, True) > 0.5
    ax.text(0.012, 0.80, 'right lane: WET μ=0.6\nleft lane: ' + ('DRY μ=1.0 (overpass)' if under else 'WET μ=0.6 (open sky)'),
            transform=ax.transAxes, color='#e2e8f0', fontsize=12, fontweight='bold', va='top', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3', fc='#0f172a', ec='none', alpha=0.7))
    n_done = sum(1 for e in events if e[0] <= t)
    n_cut = sum(1 for c in CUTS[key] if c <= t)
    ax.text(0.988, 0.95, f'LANE CHANGES {n_done}  ·  CUT-INS {n_cut}/3', transform=ax.transAxes, color='white', fontsize=16,
            fontweight='bold', ha='right', va='top', zorder=20)
    ax.text(0.988, 0.80, f'{xe[3] * 3.6:5.1f} km/h', transform=ax.transAxes, color=SUB, fontsize=14,
            ha='right', va='top', zorder=20)
    axl = D[key]['ev']['ax']; ka = min(int(round(t / 0.1)), len(axl) - 1)
    nm_msg = nmpc_msg(t) if key == 'nmpc' else None
    action = axl[ka] < -5.0 or nm_msg is not None
    for idx, (te, direction, txt) in enumerate(events):
        if te - 1.8 <= t <= te + 1.6 and not action:
            a_ = min(1, (t - te + 1.8) / 0.4, (te + 1.6 - t) / 0.4)
            ax.text(0.5, 0.06, f'LANE CHANGE {idx + 1}   {direction}   ·   {txt}', transform=ax.transAxes, color='white',
                    fontsize=15, fontweight='bold', ha='center', va='bottom', alpha=a_, zorder=21,
                    bbox=dict(boxstyle='round,pad=0.45', fc='#111827', ec='#f8fafc', alpha=0.85 * a_))
    for ci, ct in enumerate(CUTS[key]):
      if ct - 0.1 <= t <= ct + 2.6:
        blink = 1.0 if int(frame / 6) % 2 == 0 else 0.55
        ax.text(0.5, 0.80, f'⚠  CUT-IN {ci + 1}/3!  the red car changes into my lane', transform=ax.transAxes, color='white',
                fontsize=19, fontweight='bold', ha='center', va='top', zorder=22, alpha=blink,
                bbox=dict(boxstyle='round,pad=0.45', fc='#dc2626', ec='none', alpha=0.9 * blink))
    if axl[ka] < -5.0:
        ax.text(0.5, 0.05, f'EMERGENCY BRAKING (AEB)   {axl[ka] / vm.G:.2f} g', transform=ax.transAxes, color='white',
                fontsize=21, fontweight='bold', ha='center', va='bottom', zorder=23,
                bbox=dict(boxstyle='round,pad=0.5', fc='#7f1d1d', ec='#fecaca', lw=2, alpha=0.92))
    if nm_msg is not None:
        ax.text(0.5, 0.05, nm_msg, transform=ax.transAxes,
                color='white', fontsize=17, fontweight='bold', ha='center', va='bottom', zorder=23,
                bbox=dict(boxstyle='round,pad=0.45', fc='#1e3a8a', ec='#bfdbfe', lw=2, alpha=0.9))
    if t_raw > T_END[key]:
        ax.text(0.5, 0.5, f'FINISH LINE ({FINISH_S:.0f} m)  ✓   {T_END[key]:.1f} s', transform=ax.transAxes, color='white', fontsize=30,
                fontweight='bold', ha='center', va='center', zorder=25,
                bbox=dict(boxstyle='round,pad=0.6', fc=ecol, ec='white', lw=2, alpha=0.95))
    w4 = vm.utilisation(xe)[1]['w']
    if min(w4) < 0.5 < max(w4):
        side = 'left' if w4[0] > 0.5 else 'right'
        other = 'right' if side == 'left' else 'left'
        ax.text(0.5, 0.95, f'μ-SPLIT:  {side} wheels DRY  ·  {other} wheels WET', transform=ax.transAxes,
                color='#111827', fontsize=15, fontweight='bold', ha='center', va='top', zorder=21,
                bbox=dict(boxstyle='round,pad=0.4', fc='#fbbf24', ec='none', alpha=0.95))
    return xe, ke


def draw_scene(t, frame, rain_clock=None, t2=None):
    rain_clock = frame if rain_clock is None else rain_clock
    t2 = t if t2 is None else t2
    xn, kn = draw_view('nmpc', t, frame)
    xl, kl = draw_view('ltv', t2, frame)
    # rain overlay (screen space)
    axr = ax_rain; axr.cla(); axr.set_xlim(0, 1); axr.set_ylim(0, 1); axr.axis('off')
    sp = (RAIN[:, 1] - (rain_clock * 0.035 * RAIN[:, 2])) % 1.0
    xs_ = (RAIN[:, 0] + rain_clock * 0.006) % 1.0
    for x_, y_, v_ in zip(xs_, sp, RAIN[:, 2]):
        axr.plot([x_, x_ + 0.006 * v_], [y_, y_ + 0.035 * v_], color='#cbd5e1', lw=0.9, alpha=0.22 * v_)
    return xn, xl, kn, kl


# ------------------------------------------------------------- HUD --------
def wheel_card(ax, x0, y0, util, w4, col):
    # small car outline with 4 wheels coloured by utilisation, letter D/W
    ax.add_patch(FancyBboxPatch((x0 + 0.03, y0), 0.06, 0.16, boxstyle='round,pad=0.005', fc='#1f2937', ec=col, lw=1.5))
    pos = [(x0 + 0.015, y0 + 0.125), (x0 + 0.105, y0 + 0.125), (x0 + 0.015, y0 + 0.02), (x0 + 0.105, y0 + 0.02)]
    for i, (px, py) in enumerate(pos):
        ax.add_patch(Rectangle((px - 0.012, py - 0.005), 0.024, 0.035, fc=UTIL_CMAP(min(util[i], 1)), ec='white', lw=0.8))
        surf = 'DRY' if w4[i] > 0.5 else 'WET'
        tx = px - 0.03 if i % 2 == 0 else px + 0.03
        ax.text(tx, py + 0.012, f'{surf}\n{100 * util[i]:3.0f}%', color='#fde68a' if surf == 'DRY' else '#93c5fd',
                fontsize=10.5, ha='right' if i % 2 == 0 else 'left', va='center', fontweight='bold')


def draw_hud(t, xn, xl, kn, kl):
    ax = ax_hud; ax.cla(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 1, 1, fc='#0f172a', lw=0))
    ax.text(0.5, 0.975, 'NMPC  vs  PID + Stanley', color='white', fontsize=22, fontweight='bold', ha='center', va='top')
    ax.text(0.5, 0.94, 'same car · same traffic · same lane decisions', color=SUB, fontsize=11.5, ha='center', va='top')
    for i, (key, name, col, xs, kk) in enumerate((('nmpc', 'NMPC', C_N, xn, kn), ('ltv', 'PID + Stanley', C_L, xl, kl))):
        y = 0.505 - i * 0.465
        ax.add_patch(FancyBboxPatch((0.04, y), 0.92, 0.405, boxstyle='round,pad=0.01', fc='#111827', ec=col, lw=2))
        ax.text(0.08, y + 0.37, name, color=col, fontsize=20, fontweight='bold', va='center')
        sub = '4-wheel model + friction map, 3 s look-ahead' if key == 'nmpc' else 'no model · ACC + emergency brake (AEB)'
        ax.text(0.08, y + 0.335, sub, color=SUB, fontsize=10.5, va='center')
        ax.text(0.92, y + 0.355, f'{xs[3] * 3.6:5.1f}', color='white', fontsize=34, fontweight='bold', ha='right', va='center')
        ax.text(0.92, y + 0.305, 'km/h', color=SUB, fontsize=11, ha='right', va='center')
        u4, a4 = vm.utilisation(xs)
        ay = (xs[3] * xs[5]) / vm.G
        solve = 1e3 * D[key]['log']['solve'][min(kk, len(D[key]['log']['solve']) - 1)]
        axl = D[key]['ev']['ax']; along = axl[min(kk + 1, len(axl) - 1)] / vm.G
        rows = [('steering', f'{np.degrees(xs[6]):+5.1f}°'),
                ('braking / accel.', f'{along:+.2f} g'),
                ('lateral accel.', f'{ay:+.2f} g'),
                ('compute time / step', f'{solve:4.0f} ms' if solve >= 1 else '< 1 ms')]
        for j, (lab, val) in enumerate(rows):
            yy = y + 0.255 - j * 0.052
            ax.text(0.08, yy, lab, color=SUB, fontsize=12, va='center')
            vc = 'white'
            if lab.startswith('compute'):
                vc = '#4ade80' if solve < 100 else '#f87171'
            if lab.startswith('braking') and along < -0.5:
                vc = '#f87171'
            ax.text(0.55, yy, val, color=vc, fontsize=14, fontweight='bold', ha='right', va='center')
        ax.text(0.08, y + 0.035, 'real-time limit 100 ms', color='#64748b', fontsize=9.5, va='center')
        wheel_card(ax, 0.70, y + 0.06, u4, a4['w'], col)
    # utilisation colour legend
    for j, v in enumerate(np.linspace(0, 1, 40)):
        ax.add_patch(Rectangle((0.12 + j * 0.019, 0.012), 0.019, 0.012, fc=UTIL_CMAP(v), lw=0))
    ax.text(0.12, 0.03, 'wheel colour = share of available grip used (0 → 100 %)', color=SUB, fontsize=9.5, va='bottom')


def draw_bottom(t, xn, xl, t2=None):
    t2 = t if t2 is None else t2
    ax = ax_map; ax.cla(); ax.set_facecolor(BG); ax.axis('off')
    ss = np.linspace(0, FINISH_S, 500)
    X, Y, _ = vm.to_cartesian(ss, 0.0); ax.plot(X, Y, color='#475569', lw=5, solid_capstyle='round')
    for a, b in vm.S_DRY_SECTIONS:
        sd = np.linspace(a, min(b, SCN.s_end), 100); Xd, Yd, _ = vm.to_cartesian(sd, 0.0)
        ax.plot(Xd, Yd, color='#cbd5e1', lw=2.2)
    for key, col, xs in (('ltv', C_L, xl), ('nmpc', C_N, xn)):
        Xp, Yp, _ = vm.to_cartesian(xs[0], xs[1]); ax.plot([Xp], [Yp], 'o', color=col, ms=9, mec='white')
    ax.set_aspect('equal')
    ax.text(0.0, 1.0, 'ROUTE  (light = overpass sections)', transform=ax.transAxes, color=SUB, fontsize=10, va='top')
    ax2 = ax_plt; ax2.cla(); ax2.set_facecolor('#0f172a')
    for key, col, tk in (('ltv', C_L, t2), ('nmpc', C_N, t)):
        L = D[key]['log']; m = L['t'] <= tk
        ax2.plot(L['t'][m], 3.6 * L['x'][m, 3], color=col, lw=2.2, label=NAME[key])
    tm = max(t, t2)
    ax2.set_xlim(max(0, tm - 30), max(30, tm)); ax2.set_ylim(50, 105)
    ax2.legend(loc='lower left', fontsize=9, facecolor='#0f172a', edgecolor='#334155', labelcolor=TXT)
    ax2.set_title('speed [km/h]', color=SUB, fontsize=11, loc='left')
    ax2.tick_params(colors=SUB, labelsize=9)
    for sp_ in ax2.spines.values(): sp_.set_color('#334155')
    ax2.grid(alpha=0.15)
    ax2.set_xlabel('time [s]', color=SUB, fontsize=9)


# ---------------------------------------------------------- cards ---------
def results():
    out = {}
    for key in ('nmpc', 'ltv'):
        e = D[key]['ev']; L = D[key]['log']
        sm = 1e3 * np.mean(L['solve'])
        aeb = 0
        for ct in CUTS[key]:
            m = (L['t'] >= ct) & (L['t'] <= ct + 5)
            aeb += int(np.any(e['ax'][:len(L['t'])][m] < -5.0))
        out[key] = [('collisions', '0' if not e['collision'] else 'YES'),
                    ('cut-ins needing EMERGENCY braking', f'{aeb} / 3'),
                    ('hardest braking', f"{e['max_brake_g']:.2f} g"),
                    ('lowest speed', f"{3.6 * e['min_v']:.0f} km/h"),
                    ('max jerk (comfort)', f"{e['max_jerk']:.0f} m/s³"),
                    (f'time to the finish line ({FINISH_S:.0f} m)', f"{T_END[key]:.1f} s"),
                    ('mean compute time / step', f"{sm:.0f} ms" if sm >= 1 else '< 1 ms')]
    return out


RES = results()


def draw_card(kind, a):
    ax = ax_card; ax.cla(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 1, 1, fc=BG, alpha=a, lw=0))
    if kind == 'intro':
        ax.text(0.5, 0.66, 'Autonomous Overtaking in the Rain', color='white', fontsize=52, fontweight='bold', ha='center', alpha=a)
        ax.text(0.455, 0.56, 'NMPC', color=C_N, fontsize=40, fontweight='bold', ha='right', alpha=a)
        ax.text(0.5, 0.56, 'vs', color='white', fontsize=34, ha='center', alpha=a)
        ax.text(0.545, 0.56, 'PID + Stanley', color=C_L, fontsize=40, fontweight='bold', ha='left', alpha=a)
        lines = ['four-wheel vehicle model · Pacejka tyres · braking & load transfer',
                 'rain: right lane WET (μ = 0.6) · left lane DRY under the overpass (μ = 1.0)',
                 'every wheel feels its own surface  →  μ-split during each lane change',
                 'overtaking at 97 km/h  +  3 sudden CUT-INs  (same road positions for both controllers)',
                 'NMPC: nonlinear model predictive control   ·   classic: Stanley steering + PID cruise control + AEB']
        for i, l in enumerate(lines):
            ax.text(0.5, 0.43 - i * 0.055, l, color=TXT, fontsize=21, ha='center', alpha=a)
        ax.text(0.5, 0.08, 'blue = NMPC     orange = PID + Stanley', color=SUB, fontsize=18, ha='center', alpha=a)
    elif kind == 'why':
        ax.text(0.5, 0.91, 'Why is the classical controller weaker?', color='white', fontsize=42, fontweight='bold',
                ha='center', alpha=a)
        items = [('1. It cannot see the future',
                  'Stanley + PID react only to the error right now. NMPC predicts the next 3 s –\n'
                  'its own car and the other cars – and acts before the danger arrives.'),
                 ('2. Steering and braking do not talk to each other',
                  'Two separate loops (Stanley steers, PID/ACC sets the speed), so its only escape is to brake (AEB).\n'
                  'NMPC optimises steering AND braking together, so it can swerve into the free gap.'),
                 ('3. It knows nothing about the road and the tyres',
                  'Gains are tuned once, on dry asphalt; it cannot tell wet from dry or feel μ-split.\n'
                  'NMPC uses the 4-wheel Pacejka model and the grip under every single wheel.'),
                 ('4. Limits are hand-made rules',
                  'Safe distance, grip limit and lane edges are bolted on (ACC rules, emergency brake).\n'
                  'In NMPC they are constraints inside the optimisation – respected by design.')]
        for i, (h, b) in enumerate(items):
            y = 0.78 - i * 0.165
            ax.text(0.10, y, h, color=C_N, fontsize=25, fontweight='bold', va='center', alpha=a)
            ax.text(0.10, y - 0.065, b, color=TXT, fontsize=17.5, va='center', alpha=a, linespacing=1.4)
        ax.text(0.5, 0.06, 'The price: the classical controller is ~1000× cheaper to compute.  '
                           'NMPC needs a fast solver (e.g. acados, C code) to run in real time on a car.',
                color='#fbbf24', fontsize=17.5, ha='center', va='center', alpha=a)
    else:
        ax.text(0.5, 0.88, 'Results', color='white', fontsize=48, fontweight='bold', ha='center', alpha=a)
        ax.text(0.62, 0.78, 'NMPC', color=C_N, fontsize=28, fontweight='bold', ha='center', alpha=a)
        ax.text(0.80, 0.78, 'PID + Stanley', color=C_L, fontsize=28, fontweight='bold', ha='center', alpha=a)
        for i, ((lab, vn), (_, vl)) in enumerate(zip(RES['nmpc'], RES['ltv'])):
            y = 0.70 - i * 0.075
            ax.plot([0.12, 0.90], [y - 0.035, y - 0.035], color='#334155', lw=1, alpha=a)
            ax.text(0.14, y, lab, color=TXT, fontsize=24, va='center', alpha=a)
            ax.text(0.62, y, vn, color='white', fontsize=26, fontweight='bold', ha='center', va='center', alpha=a)
            ax.text(0.80, y, vl, color='white', fontsize=26, fontweight='bold', ha='center', va='center', alpha=a)
        ax.text(0.5, 0.08, 'Both reach the finish without a collision – but very differently.',
                color=SUB, fontsize=19, ha='center', va='center', alpha=a)


REPLAY_PRE = 0.9


def render_frame(f):
    rain_clock = f
    replay = False; t2 = None
    if f < N_INTRO:
        t = 0.0
        a = 1.0 if f < N_INTRO - 15 else (N_INTRO - f) / 15
    elif f < N_INTRO + N_SIM:
        t = (f - N_INTRO) / FPS; a = 0.0
    elif f < N_INTRO + N_SIM + N_REPLAY:
        i = f - N_INTRO - N_SIM
        tau = i / FPS * REPLAY_SPEED
        t = CUTS['nmpc'][2] - REPLAY_PRE + tau; t2 = CUTS['ltv'][2] - REPLAY_PRE + tau
        rain_clock = N_INTRO + N_SIM + i * REPLAY_SPEED
        replay = True; a = 0.0
    else:
        t = T_SIM; a = min(1.0, (f - N_INTRO - N_SIM - N_REPLAY) / 15)
    xn, xl, kn, kl = draw_scene(t, f, rain_clock, t2)
    draw_hud(t, xn, xl, kn, kl)
    draw_bottom(t, xn, xl, t2)
    if f < N_INTRO:
        draw_card('intro', a)
    elif f >= N_INTRO + N_SIM + N_REPLAY:
        i = f - N_INTRO - N_SIM - N_REPLAY
        if i < 9 * FPS:
            draw_card('outro', a)
        else:
            draw_card('why', 1.0)
    else:
        ax_card.cla(); ax_card.set_xlim(0, 1); ax_card.set_ylim(0, 1); ax_card.axis('off')
        if replay:
            i = f - N_INTRO - N_SIM
            fade = min(1.0, i / 10, (N_REPLAY - i) / 10)
            ax_card.text(0.35, 0.635, f'▶ SLOW-MOTION REPLAY  ×{REPLAY_SPEED:g}   ·   cut-in 3 (under the overpass), side by side',
                         color='#111827', fontsize=17, fontweight='bold', ha='center', va='center', alpha=fade,
                         bbox=dict(boxstyle='round,pad=0.45', fc='#f8fafc', ec='none', alpha=0.92 * fade))


def write_chunk(frames, out):
    from matplotlib.animation import FFMpegWriter
    w = FFMpegWriter(fps=FPS, codec='libx264', bitrate=-1,
                     extra_args=['-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium'])
    with w.saving(fig, out, dpi=100):
        for f in frames:
            render_frame(f)
            w.grab_frame(facecolor=BG)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'test':
        for f in [int(v) for v in sys.argv[2:]]:
            render_frame(f); fig.savefig(f'frame_{f}.png', dpi=100, facecolor=BG)
    elif cmd == 'chunk':
        i, n = int(sys.argv[2]), int(sys.argv[3])
        fr = np.array_split(np.arange(N_TOTAL), n)[i]
        write_chunk(fr, f'chunk_{i}.mp4')
    elif cmd == 'concat':
        n = int(sys.argv[2])
        open('chunks.txt', 'w').write(''.join(f"file 'chunk_{i}.mp4'\n" for i in range(n)))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'chunks.txt',
                        '-c', 'copy', 'NMPC_vs_PID_rain_3_cut_ins.mp4'], check=True)
    print('frames total', N_TOTAL, 'sim', T_SIM)
