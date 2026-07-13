# Silverstone racing-line optimization prototype

This repository is a runnable, simulation-first racing-line MVP. The default
demo now uses the modern Silverstone Grand Prix/Arena circuit and combines:

1. periodic track resampling and geometry;
2. a bounded minimum-curvature baseline trajectory;
3. a friction-, acceleration-, and braking-limited velocity profile;
4. a single-track vehicle model with Pacejka-style lateral forces and a
   low-speed kinematic fallback;
5. a residual-action safety supervisor;
6. a dependency-free training environment plus optional PPO support; and
7. an optional adapter for the current official F1TENTH Gym development API.

The default Silverstone demo is self-contained. It does not need Gymnasium,
PyTorch, Stable-Baselines3, or F1TENTH.

## Quick start

From PowerShell in this folder:

```powershell
python -m pip install -e .
racing-line demo
```

The bundled Silverstone profile uses 600 optimization samples and
circuit-specific controller gains. Its editable equivalent is
[`configs/silverstone.yaml`](configs/silverstone.yaml):

```powershell
racing-line demo --config configs/silverstone.yaml
```

Without installing the package, use:

```powershell
$env:PYTHONPATH = "src"
python -m racing_line demo
```

The demo writes these reproducible artifacts under `outputs/silverstone/`:

- `baseline_trajectory.csv` — optimized position, curvature, speed, and offset;
- `resampled_track.csv` — the aligned centerline and widths;
- `optimizer_diagnostics.json` — convergence and constraint diagnostics;
- `racing_line.png` and `speed_profile.png`;
- `simulation/telemetry.csv`, `summary.json`, and `driven_path.png`.

## Web dashboard

Run the Silverstone strategy simulator in a browser with:

```powershell
racing-line web
```

The page opens at `http://127.0.0.1:8765` with a ten-lap Silverstone plan. You
can choose 1–25 laps, edit the strategy, and rerun it without restarting the
server. To choose the initial lap count or avoid opening a browser
automatically:

```powershell
racing-line web --laps 10 --no-browser
```

Build the race plan before pressing **Run strategy**:

- Choose a starting tyre, then add one or more scheduled pit stops. A new
  stint starting on lap 6 means the stop occurs between laps 5 and 6 and the
  selected tyre is active from the start of lap 6.
- Add weather phases that begin on selected laps. Each phase has a condition,
  track temperature, and rain intensity, so grip and pace can change during
  the race.
- Use **Dry one-stop** for a quick slick-tyre baseline or **Rain arriving** for
  a changing-weather example, then adjust either preset as needed.
- Read the strategy timeline and executed-plan cards to confirm exactly which
  tyre and weather phase applied on each lap.

The circuit explorer uses most of the available browser width and supports
mouse-wheel or pinch zoom, drag-to-pan, dedicated zoom/reset buttons, keyboard
zoom and arrow-key panning, and full-screen mode. Toggle the track limits,
optimized line, driven path, and speed heatmap independently. Hover over the
driven path for lap, time, speed, clearance, tyre, and weather telemetry; click
to pin the readout while inspecting nearby points. The replay, run summary,
lap-time chart, and timing sheet remain available below the map.

This is a local web app: the server runs on your computer and is intentionally
available only to your computer. Publishing it at a public URL requires a
separate hosting step.

### Strategy-model scope

The dashboard is a calibrated comparative model, not an official Formula 1
strategy tool. Its Silverstone slick labels follow the public 2026 allocation:
C1 as Hard, C2 as Medium, and C3 as Soft, as listed in the
[Formula 1 tyre preview](https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2026-british-grand-prix.3qD9d5o8X4x3se0F7Zg5i1).
Intermediate and Full Wet usage follows the broad roles described in
[Pirelli's Formula 1 tyre guide](https://www.pirelli.com/tyres/en-ww/motorsport/car/formula-1).
All numerical grip, warm-up, wear, temperature, rain-crossover, and pace
effects are documented engineering estimates rather than team data.

Each scheduled tyre change adds a fixed `25.0 s` pit loss to total race time.
This makes one-stop and multi-stop plans directly comparable, but it does not
simulate the pit-lane path, entry and exit variation, traffic, or stop errors.
Weather evolves in user-defined lap phases rather than from a forecast or a
spatial rain model. The optimized reference line is a single-car baseline;
other cars, overtaking, defending, DRS, safety cars, and opponent-aware racing
lines are not simulated.

With the committed settings, the deterministic run converges in 218 optimizer
iterations. It estimates `132.00 s` for the optimized reference versus
`156.40 s` for the centerline (`15.6%` improvement), then completes the
closed-loop modeled lap in `139.70 s` with zero off-track samples. The minimum
modeled vehicle-edge clearance is `0.271 m`, retaining the configured `0.25 m`
runtime buffer. These are prototype model results, not real Formula 1 lap-time
predictions.

To retain the earlier generated circuit as a comparison:

```powershell
racing-line demo --circuit synthetic --output outputs/synthetic
```

## Bundled Silverstone data

The centerline and estimated widths are bundled from the
[TUMFTM racetrack database](https://github.com/TUMFTM/racetrack-database) at a
pinned revision. Its `5,886.8 m` reconstructed centerline is close to the
official `5.891 km` lap length shown on the
[FIA circuit map](https://www.fia.com/system/files/decision-document/2025_silverstone_event_-_circuit_map_-_silverstone_2025_0.pdf).
It is a circa-2020 research reconstruction derived from OpenStreetMap and
satellite imagery, not official survey data. Full provenance and license
details are in [`src/racing_line/data/README.md`](src/racing_line/data/README.md)
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Use your own track

Create a CSV with one row per centerline point. Do not repeat the first row at
the end (a repeated closure point is accepted and removed, though):

```csv
x_m,y_m,width_left_m,width_right_m,grip_mu
100.0,0.0,6.0,6.0,1.70
99.8,4.5,6.0,6.0,1.70
...
```

`grip_mu` is optional. Coordinates must follow the intended driving direction;
positive optimized offsets then mean left of that direction.

```powershell
racing-line optimize tracks/my_track.csv `
  --config configs/default.yaml `
  --output outputs/my_track
```

To get a complete CSV template:

```powershell
racing-line export-demo-track --output tracks/silverstone.csv
```

To replay an exported result:

```powershell
racing-line simulate `
  outputs/my_track/resampled_track.csv `
  outputs/my_track/baseline_trajectory.csv
```

## Residual PPO layer

The learning action is deliberately narrow: PPO asks for a lateral-reference
offset and a speed-reference adjustment. The safety supervisor then projects
that request into the track boundary, action-rate, speed, and friction limits.
The baseline controller—not PPO—produces steering and acceleration.

Install and train the optional stack with:

```powershell
python -m pip install -e ".[rl]"
racing-line train --timesteps 250000
```

Replay the trained residual policy through the same controller, safety
supervisor, and vehicle model used by the deterministic simulation:

```powershell
racing-line simulate `
  outputs/silverstone/resampled_track.csv `
  outputs/silverstone/baseline_trajectory.csv `
  --config configs/silverstone.yaml `
  --checkpoint models/ppo/residual_ppo.zip `
  --output outputs/ppo-replay
```

When a track contains `grip_mu`, the resampled grip profile is stored with the
trajectory and used by both the vehicle model and the residual policy during
replay. Older trajectory CSV files remain supported; the companion track CSV
provides the grip profile when available.

The documented PPO defaults use the review's sourced starting values: learning
rate `3e-4`, batch size `128`, discount `0.99`, and entropy coefficient `0.01`.
The remaining PPO values are editable engineering defaults, not claims from the
review.

## F1TENTH integration

F1TENTH is kept optional because its official `main` branch still exposes a
legacy OpenAI Gym API, while the actively maintained `dev-humble` line uses
Gymnasium. The `f1tenth` extra pins the reviewed official commit so an upstream
branch update cannot silently change this project:

```powershell
python -m pip install -e ".[f1tenth]"
```

Use `racing_line.f1tenth.create_environment(...)` and
`racing_line.f1tenth.clipped_control(...)` as the narrow integration boundary.
The pin and adapter follow the official
[F1TENTH repository](https://github.com/f1tenth/f1tenth_gym/tree/dev-humble),
[waypoint example](https://github.com/f1tenth/f1tenth_gym/blob/dev-humble/examples/waypoint_follow.py),
and [Gymnasium conformance test](https://github.com/f1tenth/f1tenth_gym/blob/dev-humble/tests/test_f110_env.py).
Native Windows is not covered by that branch's official CI, so WSL2/Ubuntu is a
reasonable fallback if its Numba or rendering dependencies fail to install.
For that optional simulator stack, Python 3.11 currently offers the broadest
third-party wheel compatibility; the self-contained core has been verified here
on Python 3.13.

## Configuration

General vehicle, optimizer, safety, simulation, and PPO defaults live in
[`configs/default.yaml`](configs/default.yaml); Silverstone-specific numerical
overrides live in [`configs/silverstone.yaml`](configs/silverstone.yaml). The
full-scale values are plausible research assumptions only. Real current
Formula 1 tyre, aerodynamic, and power-unit maps are proprietary and are not
implied by this prototype. The web strategy layer adds editable simplified
tyre and weather multipliers plus the fixed pit-loss assumption described
above; it does not turn the underlying model into official F1 telemetry.

The minimum-curvature stage is an offline bounded nonlinear optimization over
lateral offsets. The velocity stage applies a lateral friction envelope and
iterates cyclic forward acceleration and backward braking constraints. The
`speed_profile.minimum_speed_mps` value is a feasibility floor: generation
fails with a clear error if the physical constraints require a lower speed,
rather than silently violating the grip envelope. Per-point `grip_mu` affects
both cornering and longitudinal friction-circle limits. The
closed-loop simulator uses pure pursuit and longitudinal proportional control.
These are appropriate MVP baselines; minimum-time SCP, MPCC, detailed aero,
high-fidelity tyre-energy and degradation models, opponent-aware policies, and
a commercial-grade simulator remain later work.

## Tests

```powershell
python -m pip install -e ".[dev]"
pytest
```

The suite covers geometry, bounds, speed constraints, configuration, track CSV
round trips, the tyre model, the safety projection, and the dependency-free RL
environment. All stochastic entry points accept deterministic seeds.

## Important limitation

This is a research prototype, not vehicle-control software and not evidence of
Formula 1 lap-time performance. Its lap times are simplified model estimates:
the model omits detailed aerodynamics, elevation, banking, fuel burn,
high-fidelity tyre temperature/pressure/energy behaviour, power-unit
deployment, spatial weather, track evolution, race control, and opponents.
Its tyre wear and weather effects are coarse editable approximations, and its
pit stop is a fixed time addition rather than a pit-lane simulation. Validate
any learned policy in F1TENTH and then in a higher-fidelity simulator before
drawing full-scale conclusions.
