# NMPC vs PID + Stanley – rainy highway, overpass, 3 synchronised cut-ins

Install:  pip install numpy scipy matplotlib casadi osqp   (+ ffmpeg)

1) simulate (each command writes a .pkl file)
   python run_cut.py nmpc          # NMPC  (slow: ~20-30 min)
   python run_cut.py classic       # PID + Stanley (~10 s)
2) render the video (frames are split into chunks so you can use several CPU cores)
   python render_v3.py chunk 0 2   # can run in parallel with the next line
   python render_v3.py chunk 1 2
   python render_v3.py concat 2    # -> NMPC_vs_PID_rain_3_cut_ins.mp4
   (python render_v3.py test 1700  renders one still frame for checking)

All on-screen texts (title card, banners, HUD, results table) are in render_v3.py:
  draw_card()  -> intro / results / "why is the classical controller weaker" cards
  draw_view()  -> CUT-IN / EMERGENCY BRAKING / lane-change / mu-split banners
  draw_hud()   -> right-hand panel
The "compute time" numbers are measured on the computer that runs the simulation,
so they will differ on yours; the driving itself is deterministic.
