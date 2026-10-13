# FretLab — Virtual Luthier Fretboard Modeler

**FretLab** is a physics-based fretboard / fingerboard modeler for anyone who works on
fretted instruments: guitar techs, luthiers, violin-family restorers, bass setup
specialists, and curious players. It answers the questions that dominate a good setup:

> *How close can I get the strings to the board before it buzzes? Is this bow helping or
> hurting? Do I need shims, a new nut, a truss-rod adjustment, or a plane?*

Instead of guessing with a ruler and feeler gauge, you configure the instrument, and
FretLab renders the side-profile geometry from **nut → frets → saddle**, overlays the
**peak (maximum-displacement) vibration envelope** of each string model, and reports
quantified **clearance / buzz-risk** metrics at every fret so you can minimize action
safely.

---

## Why Python?

FretLab is implemented in **Python 3.10+** using `numpy` / `scipy` for the math,
`pydantic` for validated configuration models, `matplotlib` for rendering, `pandas`
for data-driven fret profiles, and `click` + `rich` for the CLI. This stack gives us:

* excellent numerical and plotting ergonomics for a geometry/physics tool,
* painless CSV import/export (data-driven fret heights, measured bow curves),
* an easy path to a later web UI (`FastAPI`) or interactive desktop mode (`tkinter`/Qt).

---

## The Physics Model (v0.1 scope)

### 1. Fretboard / neck profile along the string axis

The board surface height `z_board(x)` at distance `x` from the nut is composed of:

```
z_board(x) = z_plane(x) + θ · x            # machined plane + fingerboard angle
           + bow(x)                        # relief / back-bow (selectable model or data)
```

and the **fret crown line** adds:

```
z_crown(x) = z_board(x) + fret_height(x)   # uniform | taper | per-fret data
```

* **Neck relief (bow)** — selectable functions:
  * `flat`, `constant_curvature` (signed radius R), `cubic` (classic shape with max
    relief near the 8th fret), `circular_arc`, and `data` (measured feeler-gauge
    points interpolated with `scipy.interpolate.CubicSpline`).
  * A **truss-rod model** maps rod force/turns → change in curvature. v0.1 uses a
    linearized engineering-beam approximation:
    `Δ(1/R) = (M_rod − ΔT·e) / (E·I)` — rod moment plus the string-tension moment
    (tension × arm offset from the neutral axis) set net curvature. Two calibrated
    per-instrument constants (`truss_stiffness`, `string_load_coefficient`) keep this
    honest without a full FEM.
* **Fret heights** — selectable: `uniform`, `taper` (linear ramp, e.g. worn boards),
  `data` (per-fret micrometer measurements from CSV). Fret positions follow the
  **Rule of 17**: `d_n = S·(1 − 2^(−n/N))`; custom position tables allowed.
* **Neck fall / heel carve** — depth past the last fret, modeled as a slope change;
  critical for bolt-on acoustics and carved-top instruments.

### 2. String static geometry

Each string runs **nut → fretted contact point → saddle** as two straight segments
(fretting at fret *k* shortens the speaking length and moves the upper anchor to the
crown of fret *k*; open position contacts only nut & saddle). Parameters: nut height &
position, saddle height & position, scale length, string gauge, tension, and per-string
lateral offsets (reserved for the later fan/twist features).

### 3. Peak vibration envelope (no time domain needed)

For a plucked string we render the **maximum displacement over all time** at each
position `x` — the classic "envelope of peak excursion".  For a triangular pluck
of deflection `A` released from rest, the exact d'Alembert solution reaches a
global peak excursion of exactly `A` at the pluck point; off-center plucks have
an *asymmetric* profile with smaller swing at mid-string (≈0.61·A at p=0.18).

with user-selectable excitation models:

| Model | What it does |
|---|---|
| `fundamental` | single `A·sin(πx/L)` hump — simple, conservative |
| `pluck_spectrum` | simulation-calibrated Gibbs envelope: shape interpolated from a direct finite-difference wave solve (`fretlab/data/pluck_envelope_table.npz`, regenerate via `python -m fretlab.tools.calibrate_pluck`), normalized so peak = `A` |
| `hi_harmonic` | worst-case high-harmonic ring-out buzz test (harmonic node at fret k) |

String diameter & tension set the reported frequency and are carried through the API
now so stiffness/dispersion corrections can slot in later without breaking configs.

### 4. Interaction metrics (the payoff)

At every fret, FretLab computes:

* **Open clearance** — min gap between the open-string envelope and the fret crown.
* **Fretted clearance** — gap between the envelope of the string fretted *at that fret*
  and the **next** fret crown (the classic buzz test).
* **Buzz risk score** — clearance vs. configurable thresholds; flags `OK / MARGINAL / BUZZ`.
* **Action @ 12th fret & last fret**, nut-slot depths, and heuristic suggested
  corrections ("lower saddle ≈0.4 mm", "add ≈¼ turn relief") in v0.1; a real
  inverse-solving optimizer arrives in v0.2.

---

## Configuration

Everything is a declarative YAML/JSON document validated by `pydantic`:

```yaml
instrument:
  name: "Strat-style electric"
  family: guitar            # guitar | bass | violin | cello | mandolin | custom
  scale_length_mm: 648.0
  num_frets: 22
  board_angle_deg: 0.0      # fingerboard angle relative to string plane reference
  neck_fall_mm: 0.8         # heel carve past last fret

fretboard:
  length_mm: 695.0
  width_at_nut_mm: 42.8
  width_at_last_fret_mm: 56.0
  material: rosewood        # informational (E, density presets built in)
  fret_positions: rule_of_17
  bow:
    model: cubic            # flat | cubic | constant_curvature | circular_arc | data | custom
    max_relief_mm: 0.25
  fret_heights:
    model: uniform
    height_mm: 1.15

truss:
  state_turns: 0.0          # + = tighter (straightens), − = looser (more relief)
  truss_stiffness: 0.0018   # calibrated: curvature change per turn (1/mm/turn)
  string_load_coefficient: 0.00002  # calibrated: curvature added per N of tension

strings:
  - {name: E1, gauge_mm: 0.46, tension_N: 75.0, lateral_mm: -21.4}
  - {name: A2, gauge_mm: 0.58, tension_N: 74.0, lateral_mm: -12.8}
  # ...

contact_points:
  nut_height_mm: 1.30       # above board surface at the nut
  nut_position_mm: 0.0
  saddle_height_mm: 3.20    # string centerline height above board line at saddle
  saddle_position_mm: 658.0 # scale length + compensation

excitation:
  model: pluck_spectrum     # fundamental | pluck_spectrum | hi_harmonic
  pluck_amplitude_mm: 1.2
  pluck_position_ratio: 0.18
  harmonics: 12

buzz:
  warn_clearance_mm: 0.25
  buzz_clearance_mm: 0.12
```

Selectable functions are swappable everywhere a curve is needed, and any curve can be
overridden by a **user function** — a sandboxed expression in `x` (mm from nut), or a
Python callable when used as a library:

```yaml
fretboard:
  bow:
    model: custom
    expr: "0.35*math.sin(math.pi*x/648)**2"   # safe-eval: math, numpy available
```

Data-driven mode reads CSVs:

```csv
# fret_heights.csv                # bow_profile.csv
fret,dist_mm,height_mm            x_mm,deflection_mm
1,36.75,1.18                      0,0.0
2,69.55,1.19                      162,0.28
...                               ...,...
```

## Presets

`presets/` ships starting points so you can be productive immediately:
`strat_648.yaml`, `lespaul_628.yaml`, `precision_bass_864.yaml`,
`violin_328.yaml` (fretless — the "obstacle" becomes the board surface itself),
`cello_1065.yaml`, `dreadnought_645.yaml`.

---

## CLI

```bash
# Scaffold a config from a preset
fretlab new my-guitar --preset strat_648

# Validate a config
fretlab validate config.yaml

# Render side profile + envelope for one string (default: first string)
fretlab plot config.yaml --string E1 --out profile.png

# Full clearance/buzz report table for every string & fret
fretlab report config.yaml

# Truss what-if sweep (writes one PNG per step into sweeps/)
fretlab sweep-truss config.yaml --from -1 --to 2 --step 0.25 --outdir sweeps/

# Minimum-action optimizer (v0.2 solver; stub warns in v0.1)
fretlab optimize config.yaml --objective saddle_min
```

*(module form during development: `python -m fretlab ...`)*

## Library

```python
from fretlab import Instrument

inst = Instrument.from_yaml("config.yaml")
res = inst.analyze(string="E1")
res.buzz_table()          # pandas DataFrame
res.plot("profile.png")   # matplotlib figure saved
```

---

## Roadmap

| Version | Feature |
|---|---|
| **v0.1 (now)** | Project skeleton, validated config schema, fret/board/string geometry, bow models + truss linearization, peak-envelope plots, clearance/buzz tables, CLI, presets |
| v0.2 | Truss calibration from measured before/after bow pairs; min-action optimizer (bisection on saddle/nut); fret wear & crowning inputs; multi-string fan + neck-twist offsets |
| v0.3 | Interactive matplotlib rig (sliders for truss/saddle/nut); SVG/PDF export for shop wall |
| v0.4 | Optional switchable 3-D view (`plotly`) — same engine, different renderer |
| v0.5 | Web UI (FastAPI + htmx), instrument library persistence (SQLite), laser-micrometer import helpers |

## Repo layout

```
fretlab/
  __init__.py          public API surface (Instrument, analyze, …)
  __main__.py          `python -m fretlab`
  units.py             mm-first unit helpers (inch/mils conversions)
  config.py            pydantic v2 schemas; YAML/JSON load & save
  materials.py         board-material presets (E, density)
  geometry.py          fret positions, board plane + angle + neck fall, fret heights
  bow.py               relief models, data splines, truss-rod linearized solver
  strings.py           string spec, frequency, static nut→fret→saddle polyline
  envelope.py          peak-displacement envelope models
  analysis.py          clearance computation, buzz scoring, fret-by-fret tables
  functions/           user-function sandbox (safe eval of curve expressions)
  plotting.py          matplotlib side-profile renderer (2-D now, 3-D toggle later)
  cli.py               click entrypoint: new | validate | plot | report | sweep-truss | optimize
presets/*.yaml
examples/              sample configs + sample CSV data
tests/
pyproject.toml
```

## Install & develop

```bash
pip install -e ".[dev]"
pytest
```

## License / disclaimer

MIT. FretLab is a modeling aid — always verify critical dimensions physically before
cutting, planing, or cranking rods. The truss model is a calibrated linearization, not
a substitute for a good feeler gauge and judgment.
