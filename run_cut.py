import sys, pickle, time
import scenarios as S, scenario_youtube as Y, classical as CL, controllers as C
key = sys.argv[1]
scn = Y.make()
ctrl = {'nmpc': lambda: C.NMPC(), 'classic': lambda: CL.Classical(), 'ltv': lambda: C.LTVMPC()}[key]()
t0 = time.time()
log = S.run(scn, ctrl, Y.behaviour, Y.stop); ev = S.evaluate(scn, log)
pickle.dump(dict(scn=scn, log=log, ev=ev, ctrl=ctrl.name), open(f'cut3_{key}.pkl', 'wb'))
print(key, 'safe', ev['safe'], 'clear %.2f' % ev['min_clear'], ev['worst'], 'brake %.2f' % ev['max_brake_g'],
      'util %.2f' % ev['max_util'], 'ay %.2f' % ev['max_ay_g'], 'jerk %.1f' % ev['max_jerk'], 'minv %.1f' % ev['min_v'],
      'changes', len(log['mem'].get('changes', [])), 'cuts', [(round(a, 1), round(b)) for a, b in log['mem'].get('cuts', [])], 't_end %.1f' % log['t'][-1],
      'wall %.0f' % (time.time() - t0), flush=True)
