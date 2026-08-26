"""
Surface3D.py -- interactive 3-D risk-geometry visualisation (Plotly).

Consumes the dict returned by Analysis_points.risk_geometry_3d(...):
    g["pass_grid"]  : ndarray[n_stop, n_risk, n_rr]  pass rate %
    g["days_grid"]  : ndarray[n_stop, n_risk, n_rr]  mean days-to-pass (may hold nan)
    g["wr_grid"]    : ndarray[n_stop, n_rr]          measured WR (risk-independent)
    g["stop_grid"], g["risk_pcts"], g["rr_grid"]     the three axes
    g["best"]       : dict(stop_points, risk_pct, rr, pass_pct, ...)

Two views, both with hover that reads off the exact (stop, risk%, R:R) and the
pass rate / days-to-pass / WR at that point:

  surface_by_stop(g)  -> one pass-rate SURFACE over (risk% x R:R), with a SLIDER
                         across stop-points. This is the "3-D surface you hover"
                         you asked for; the slider is the 4th axis (stop).
  scatter_4d(g)       -> every grid cell as a point at (stop, risk%, R:R), colour
                         = pass rate. The whole cube at once; best cell ringed.

write_html(fig, path) saves a standalone file that opens in any browser and
renders inside Jupyter via fig.show().
"""
from __future__ import annotations
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as e:                       # keep the import error actionable
    raise ImportError("Surface3D needs plotly: pip install plotly") from e


_COLORSCALE = [                                 # red -> amber -> green, matches your heatmaps
    [0.00, "#8a1f1f"], [0.25, "#c0392b"], [0.45, "#e67e22"],
    [0.62, "#c9b02a"], [0.80, "#4a9a3a"], [1.00, "#1e7a2e"]]


def _axes(g):
    stops = np.asarray(g["stop_grid"], float)
    risks = np.asarray(g["risk_pcts"], float)
    rrs = np.asarray(g["rr_grid"], float)
    return stops, risks, rrs


def surface_by_stop(g, firm_name="", start_balance=50_000.0, max_days_to_pass=None):
    """Pass-rate surface over (risk% x R:R), one frame per stop, stepped by a
    slider. Hover shows risk%, risk$, R:R, pass%, days, WR at that vertex."""
    stops, risks, rrs = _axes(g)
    pg = np.asarray(g["pass_grid"], float)               # [stop, risk, rr]
    dg = np.asarray(g.get("days_grid"), float) if g.get("days_grid") is not None else np.full_like(pg, np.nan)
    wr = np.asarray(g.get("wr_grid"), float) if g.get("wr_grid") is not None else None  # [stop, rr]
    RR, RISK = np.meshgrid(rrs, risks)                   # both [n_risk, n_rr]
    risk_dollars = RISK * start_balance / 100.0

    frames, steps = [], []
    for si, stop in enumerate(stops):
        Z = pg[si]                                       # [risk, rr]
        wr_layer = (np.repeat(wr[si][None, :], len(risks), axis=0) if wr is not None
                    else np.full_like(Z, np.nan))        # broadcast WR (risk-independent) across risk rows
        custom = np.stack([risk_dollars, dg[si], wr_layer * 100.0], axis=-1)
        hover = ("R:R %{x:.2f}<br>risk %{y:.2f}%  ($%{customdata[0]:,.0f})<br>"
                 "<b>pass %{z:.1f}%</b><br>days %{customdata[1]:.0f}<br>"
                 "WR %{customdata[2]:.0f}%<extra></extra>")
        surf = go.Surface(x=rrs, y=risks, z=Z, customdata=custom,
                          colorscale=_COLORSCALE, cmin=0, cmax=100,
                          hovertemplate=hover, colorbar=dict(title="pass %"),
                          name=f"{stop:g}pt")
        frames.append(go.Frame(data=[surf], name=f"{stop:g}"))
        steps.append(dict(method="animate", label=f"{stop:g}pt",
                          args=[[f"{stop:g}"], dict(mode="immediate",
                                frame=dict(duration=0, redraw=True), transition=dict(duration=0))]))

    fig = go.Figure(data=frames[0].data, frames=frames)
    b = g.get("best", {})
    title = f"3-D risk geometry — {firm_name}".strip(" —")
    if b:
        title += (f"<br><sup>best: {b.get('stop_points'):g}pt / {b.get('rr'):g}R "
                  f"@ {b.get('risk_pct'):.2f}% = {b.get('pass_pct'):.0f}% pass</sup>")
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="R:R", yaxis_title="risk per trade (%)", zaxis_title="pass rate (%)",
                   zaxis=dict(range=[0, 100])),
        sliders=[dict(active=0, currentvalue=dict(prefix="stop = "), pad=dict(t=40), steps=steps)],
        margin=dict(l=0, r=0, t=70, b=0), height=680)
    return fig


def scatter_4d(g, firm_name="", start_balance=50_000.0, max_days_to_pass=None):
    """Every cell as a point: x=stop, y=risk%, z=R:R, colour=pass%. Optionally
    dims cells that breach the days ceiling. Best cell ringed in cyan."""
    stops, risks, rrs = _axes(g)
    pg = np.asarray(g["pass_grid"], float)
    dg = np.asarray(g.get("days_grid"), float) if g.get("days_grid") is not None else np.full_like(pg, np.nan)
    S, Ri, Rr = np.meshgrid(stops, risks, rrs, indexing="ij")
    x, y, z, c, d = S.ravel(), Ri.ravel(), Rr.ravel(), pg.ravel(), dg.ravel()
    m = np.isfinite(c)
    if max_days_to_pass is not None:
        keep = m & (np.isfinite(d)) & (d <= max_days_to_pass)
        m = keep if keep.any() else m
    x, y, z, c, d = x[m], y[m], z[m], c[m], d[m]
    custom = np.stack([y * start_balance / 100.0, d], axis=-1)
    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=4, color=c, colorscale=_COLORSCALE, cmin=0, cmax=100,
                    colorbar=dict(title="pass %"), opacity=0.85),
        customdata=custom,
        hovertemplate=("stop %{x:g}pt<br>risk %{y:.2f}%  ($%{customdata[0]:,.0f})<br>"
                       "R:R %{z:.2f}<br><b>pass %{marker.color:.1f}%</b><br>"
                       "days %{customdata[1]:.0f}<extra></extra>")))
    b = g.get("best", {})
    if b:
        fig.add_trace(go.Scatter3d(
            x=[b["stop_points"]], y=[b["risk_pct"]], z=[b["rr"]], mode="markers",
            marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color="cyan", width=4)),
            name="best", hovertemplate=f"BEST {b['pass_pct']:.0f}% pass<extra></extra>"))
    fig.update_layout(
        title=f"3-D risk geometry (all cells) — {firm_name}".strip(" —"),
        scene=dict(xaxis_title="stop (points)", yaxis_title="risk per trade (%)", zaxis_title="R:R"),
        margin=dict(l=0, r=0, t=50, b=0), height=680)
    return fig


def write_html(fig, path):
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path