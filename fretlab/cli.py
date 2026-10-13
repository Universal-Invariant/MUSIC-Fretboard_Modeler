"""FretLab command-line interface.

Commands
--------
new          scaffold a config from a preset
presets      list available presets
validate     load & validate a config, echo resolved summary
plot         render the nut→saddle side profile for one string
report       clearance/buzz tables for every string (rich)
sweep-truss  what-if: render a PNG per truss-rod turn step
optimize     minimum-action solver (v0.2 — stub in v0.1)
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from .config import FretLabConfig
from .instrument import Instrument
from . import presets as preset_registry

console = Console()


@click.group()
@click.version_option(package_name="fretlab", prog_name="fretlab")
def main() -> None:
    """FretLab — virtual luthier fretboard modeler."""


# --------------------------------------------------------------------------- #

@main.command("presets")
def cmd_presets() -> None:
    """List bundled/user presets."""
    t = Table(title="FretLab presets")
    t.add_column("preset", style="cyan")
    t.add_column("title")
    t.add_column("description")
    for name in preset_registry.list_presets():
        info = preset_registry.preset_info(name)
        t.add_row(name, info["title"], info["description"])
    console.print(t)


@main.command("new")
@click.argument("dest", type=click.Path())
@click.option("--preset", "-p", default=None, help="Preset name (see `fretlab presets`).")
@click.option("--force", is_flag=True, help="Overwrite existing file.")
def cmd_new(dest: str, preset: str | None, force: bool) -> None:
    """Create a new config at DEST (from --preset, or a blank template)."""
    p = Path(dest)
    if p.suffix not in (".yaml", ".yml"):
        p = p.with_suffix(".yaml")
    if preset:
        out = preset_registry.scaffold(preset, p, force=force)
        console.print(f"[green]created[/green] {out} [dim](from preset {preset})[/dim]")
        return
    if p.exists() and not force:
        raise click.ClickException(f"{p} exists (use --force)")
    template = _blank_template()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(template, encoding="utf-8")
    console.print(f"[green]created[/green] {p} [dim](blank template)[/dim]")


BLANK_MIN = {
    "instrument": {"name": "My instrument", "scale_length_mm": 648.0, "num_frets": 22},
    "strings": [
        {"name": "e1", "gauge_mm": 0.46, "tension_N": 75.0},
        {"name": "E6", "gauge_mm": 1.09, "tension_N": 68.0},
    ],
    "contact_points": {"nut_height_mm": 1.3, "saddle_height_mm": 3.2},
}


def _blank_template() -> str:
    header = (
        "# FretLab configuration — edit values, then run:\n"
        "#   fretlab validate this.yaml\n"
        "#   fretlab plot this.yaml --string e1\n"
        "# Full schema & options: see README.md\n\n"
    )
    return header + yaml.safe_dump(BLANK_MIN, sort_keys=False)


# --------------------------------------------------------------------------- #

def _load(path: str) -> Instrument:
    try:
        cfg = FretLabConfig.load(path)
    except FileNotFoundError:
        raise click.ClickException(f"no such file: {path}")
    except Exception as e:  # pydantic ValidationError etc.
        console.print(f"[red]config error:[/red] {e}")
        raise click.ClickException("validation failed")
    return Instrument(cfg)


@main.command("validate")
@click.argument("config", type=click.Path(exists=True))
def cmd_validate(config: str) -> None:
    """Validate CONFIG and print the resolved setup summary."""
    inst = _load(config)
    c = inst.cfg
    from .bow import resolve_truss

    state = resolve_truss(c, inst.total_tension_N)
    t = Table(title=f"{c.instrument.name} — resolved", show_lines=False)
    t.add_column("item", style="cyan"); t.add_column("value")
    t.add_row("family / scale", f"{c.instrument.family} / {c.instrument.scale_length_mm:.1f} mm")
    t.add_row("frets", str(c.instrument.num_frets))
    t.add_row("total tension", f"{inst.total_tension_N:.1f} N")
    t.add_row("effective relief", f"{state.effective_relief_mm:+.3f} mm ({state.note})")
    t.add_row("strings", ", ".join(
        f"{s.name} {s.gauge_mm:.2f}mm @{s.tension_N:.0f}N" for s in c.strings))
    console.print(t)
    console.print("[green]OK[/green] config is valid.")


@main.command("plot")
@click.argument("config", type=click.Path(exists=True))
@click.option("--string", "-s", default=None, help="String name (default: first).")
@click.option("--fretted-at", "-f", default=None, type=int, help="Overlay note fretted at fret k.")
@click.option("--out", "-o", default="profile.png", type=click.Path())
def cmd_plot(config: str, string: str | None, fretted_at: int | None, out: str) -> None:
    """Render side profile + vibration envelope for one string."""
    inst = _load(config)
    res = inst.analyze(string=string)
    from .plotting import plot_string_profile

    plot_string_profile(res, path=out, fretted_at=fretted_at, instrument=inst)
    console.print(f"[green]wrote[/green] {out}  "
                  f"[dim]({res.physics.name}: min clearance "
                  f"{res.min_clearance_mm:.3f} mm → {res.status})[/dim]")


@main.command("report")
@click.argument("config", type=click.Path(exists=True))
@click.option("--string", "-s", default=None,
              help="Limit report to one string (default: all).")
@click.option("--frets/--no-frets", default=True, help="Include per-fret table for worst string.")
def cmd_report(config: str, string: str | None, frets: bool) -> None:
    """Per-string action/clearance/buzz report."""
    inst = _load(config)
    df = inst.report()
    if string is not None:
        df = df[df["string"] == string]
        if df.empty:
            names = ", ".join(s.name for s in inst.cfg.strings)
            raise click.BadParameter(f"unknown string '{string}' (known: {names})")
    t = Table(title=f"{inst.cfg.instrument.name} — setup report")
    for col in df.columns:
        t.add_column(str(col))
    for _, row in df.iterrows():
        status = str(row["status"])
        style = {"BUZZ": "red", "MARGINAL": "yellow"}.get(status, "green")
        t.add_row(*[str(v) for v in row], )
    console.print(t)

    worst = df.loc[df["min_clearance_mm"].idxmin()]
    if frets and worst["status"] != "OK":
        res = inst.analyze(string=worst["string"])
        bt = res.buzz_table()
        t2 = Table(title=f"per-fret clearance — {worst['string']}")
        for col in bt.columns:
            t2.add_column(str(col))
        for _, r in bt.iterrows():
            st = str(r["status"])
            mark = {"BUZZ": "[red]", "MARGINAL": "[yellow]"}.get(st, "")
            end = "[/]" if mark else ""
            t2.add_row(*[f"{mark}{v}{end}" for v in r])
        console.print(t2)

    for a in df["string"]:
        res = inst.analyze(string=a)
        for note in res.model_notes:
            console.print(f"  [yellow]{a}[/yellow] (model): {note}")
        for adv in res.advisories:
            console.print(f"  [cyan]{a}[/cyan]: {adv}")


@main.command("sweep-truss")
@click.argument("config", type=click.Path(exists=True))
@click.option("--string", "-s", default=None)
@click.option("--from", "turns_from", default=-1.0, type=float)
@click.option("--to", "turns_to", default=2.0, type=float)
@click.option("--step", default=0.25, type=float)
@click.option("--outdir", "-o", default="sweeps", type=click.Path())
def cmd_sweep_truss(config: str, string: str | None, turns_from: float,
                    turns_to: float, step: float, outdir: str) -> None:
    """What-if sweep: one profile PNG per truss-turn step."""
    import numpy as np

    inst = _load(config)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    steps = np.arange(turns_from, turns_to + step / 2, step)
    rows = []
    for turns in steps:
        variant = inst.with_mods(**{"truss.state_turns": float(turns),
                                   "truss.solve_mode": "manual"})
        res = variant.analyze(string=string)
        png = out / f"truss_{turns:+.2f}.png"
        from .plotting import plot_string_profile

        plot_string_profile(res, path=str(png), instrument=variant)
        rows.append((round(float(turns), 2), round(res.min_clearance_mm, 3),
                     res.status, res.worst_fret))
    t = Table(title="truss sweep")
    for h in ("turns", "min_clear_mm", "status", "worst_fret"):
        t.add_column(h)
    for r in rows:
        t.add_row(*[str(v) for v in r])
    console.print(t)
    console.print(f"[green]wrote {len(steps)} images to[/green] {outdir}/")


@main.command("optimize")
@click.argument("config", type=click.Path(exists=True))
@click.option("--objective", "-O", type=click.Choice(["saddle_min", "nut_min", "relief_best"]),
              default="saddle_min")
def cmd_optimize(config: str, objective: str) -> None:
    """Minimum-action optimizer (arrives in v0.2)."""
    console.print(f"[yellow]not implemented yet[/yellow] — '{objective}' lands in v0.2. "
                  "Use `sweep-truss` and manual saddle tweaks for now.")
    sys.exit(2)


if __name__ == "__main__":
    main()
