# NMPC vs PID + Stanley on a μ split highway

What happens when a smart controller and a classic one drive the same car, on the same road, in the same rain?
This project puts a **Nonlinear Model Predictive Controller (NMPC)** and a classical **PID + Stanley** controller side by side and lets them overtake traffic and deal with three sudden cut ins on a highway where the grip changes from lane to lane.

## Importance

Most controller demos use a perfect, dry road. Real roads are not like that. In the rain, grip can drop to about half, and it is often **not the same everywhere**: one lane can be wet while the other stays dry, for example under an overpass, in a tunnel, or next to a puddle. Engineers call this a **μ split** road.

μ split situations are tricky:

* during a lane change, the wheels on one side of the car are on dry asphalt while the other side is still on the wet lane;
* if the car brakes there, the dry side brakes harder than the wet side and the car is pulled sideways;
* a controller that does not know about the grip can easily ask the tyres for more than they can give.

The goal of this project is to show, in a clear and reproducible way, how a controller that **understands the car and the road** (NMPC) compares with a classical controller that only reacts to errors (PID + Stanley) in exactly this kind of situation.

## The scenario

* A two lane highway, about 2 km long, with gentle bends.
* It is raining: the right lane is **wet (μ = 0.6)**.
* The left lane runs under a stacked overpass and stays **dry (μ = 1.0)** wherever the overpass covers it.
* The right lane has slow traffic; the left lane is the faster lane.
* The car drives at about 97 km/h, overtakes slower cars and returns to the right lane when there is room.
* **Three cars suddenly cut in** from the right lane while our car is overtaking. Each cut in happens at **exactly the same place on the road** for both controllers, so the comparison is fair.

The road and its wet/dry map are predefined, but the car's path is not: the controller plans it online every 0.1 s (local path planning).

## Why a four wheel model?

A simple "bicycle" model squeezes the left and right wheels into one. That is fine on a uniform road, but it cannot show what happens when the left wheels are on dry asphalt and the right wheels are on wet asphalt.

So this project uses a **four wheel (double track) vehicle model**:

* at every step we find where all 4 wheels are on the road and check wet or dry under each one, so during a lane change the wheels on one side can be on dry while the other side is still on wet;
* each wheel carries its own share of the weight: more on the front wheels when braking, more on the outer wheels in a turn;
* the tyre forces come from the **Pacejka "Magic Formula"**, the standard tyre model in vehicle dynamics;
* each tyre has only so much grip. The more it uses for braking, the less is left for steering;
* every wheel has its own ABS, so braking on a μ split road really does pull the car toward the dry side.

## The two controllers

**NMPC (Nonlinear Model Predictive Control)**
* Uses the full four wheel model and the wet/dry map inside its optimisation.
* Looks 3 seconds ahead and predicts where the other cars will be.
* Plans steering and braking **together**, so it can choose to steer around a problem instead of only braking.
* Safety distance, grip limits and road edges are built in as constraints.
* Solved with CasADi + IPOPT.

**PID + Stanley (classical)**
* Stanley controller for steering, PID cruise control (ACC) for speed, and automatic emergency braking (AEB).
* Lane changes follow a smooth preshaped curve.
* No vehicle model, no knowledge of the road surface, no look ahead. It reacts to what is in front of it right now.
* Tuned on a dry road, as is common in practice.

Both controllers drive the same car, use the same lane change decisions and have the same steering and braking limits.

## Results

| | NMPC | PID + Stanley |
|---|---|---|
| Collisions | 0 | 0 |
| Cut ins that needed emergency braking | 0 / 3 | 3 / 3 |
| Hardest braking | 0.21 g | 0.92 g |
| Lowest speed during the cut ins | 94 km/h | 63 km/h |
| Time to the finish line (2100 m) | 78.0 s | 80.7 s |
| Compute time per step | about 1.6 s | under 1 ms |

Both controllers reach the finish safely, but in very different ways. The classical controller survives each cut in only by slamming on the brakes. NMPC sees the cut in coming, steers into the free gap and keeps its speed.

The price is computation. In Python, NMPC is far from real time. On a real car it would need a fast solver such as acados with generated C code.

## How to run it

Install the packages (you also need ffmpeg to make the video):

```
pip install -r requirements.txt
```

**1. Run the simulations.** Each command saves its results to a `.pkl` file.

```
python run_cut.py nmpc       # NMPC, slow: about 20 to 30 minutes
python run_cut.py classic    # PID + Stanley, about 10 seconds
```

**2. Make the video.** The frames are split into chunks so you can use several CPU cores; the two chunk commands can run at the same time.

```
python render_v3.py chunk 0 2
python render_v3.py chunk 1 2
python render_v3.py concat 2   # creates NMPC_vs_PID_rain_3_cut_ins.mp4
```

To check a single frame without rendering everything:

```
python render_v3.py test 1700
```

## Files

| File | What it does |
|---|---|
| `vehicle_model.py` | Four wheel car model, Pacejka tyres, road shape and the wet/dry map |
| `controllers.py` | The NMPC controller |
| `classical.py` | The PID + Stanley controller with ACC and emergency braking |
| `scenarios.py` | Traffic, cut ins, simulation loop and safety checks |
| `scenario_youtube.py` | The highway scenario used in the video |
| `run_cut.py` | Runs one controller through the scenario |
| `render_v3.py` | Draws the video |

All texts shown in the video (title, banners, side panel, results and the final explanation) are in `render_v3.py`. The compute time numbers are measured on the computer that runs the simulation, so yours will be different; the driving itself is exactly the same every time.

## Limitations

* The other cars follow a script and do not react to our car.
* Perception is perfect: the controllers know the exact position and speed of every nearby car.
* NMPC is given the wet/dry map. A real car would have to estimate the road grip.
* Wheel spin dynamics are not modelled; ABS is assumed to work perfectly.
* This is one scenario, not a statistical study.

## References

* J. A. E. Andersson et al., "CasADi: a software framework for nonlinear optimization and optimal control", *Mathematical Programming Computation*, 2019.
* A. Wächter and L. T. Biegler, "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming" (IPOPT), *Mathematical Programming*, 2006.
* H. B. Pacejka, *Tyre and Vehicle Dynamics*, Butterworth-Heinemann.
* R. Rajamani, *Vehicle Dynamics and Control*, Springer.
* S. Thrun et al., "Stanley: The robot that won the DARPA Grand Challenge", *Journal of Field Robotics*, 2006.
* P. Falcone et al., "Predictive active steering control for autonomous vehicle systems", *IEEE Transactions on Control Systems Technology*, 2007.
* Euro NCAP test protocols for cut in scenarios.


## Watch the result

The video of the full run is on YouTube. If you want to see both controllers in action, check it out here:
**https://youtu.be/hULIXalt-vA**
