"""Plotly figure specifications as plain dicts (no plotly Python dependency).

Colours are written as role tokens that the page resolves for the active
light/dark theme:

* ``@s1`` … ``@s8``  categorical slots (fixed order)
* ``@seq:0.42``      position on the sequential (blue) ramp
* ``@div:0.10``      position on the diverging ramp (0 = blue, 1 = red)
* ``@seqscale`` / ``@divscale`` colourscales, ``@ink``, ``@muted`` …
"""

import math

import numpy as np
import pandas as pd

from . import geo

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]


def _r(values, digits=3):
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            out.append(None)
        elif isinstance(v, (float, np.floating)):
            out.append(round(float(v), digits))
        elif isinstance(v, (np.integer,)):
            out.append(int(v))
        else:
            out.append(v)
    return out


def figure(data, layout=None, height=360, kind="chart", **extra):
    spec = {"data": data, "layout": layout or {}, "height": height, "kind": kind}
    spec.update(extra)
    return spec


def norm(value, vmin, vmax):
    if vmax is None or vmin is None or vmax <= vmin:
        return 0.5
    return float(min(1.0, max(0.0, (value - vmin) / (vmax - vmin))))


def seq(value, vmin, vmax):
    return f"@seq:{norm(value, vmin, vmax):.3f}"


def div(value, vabs):
    if not vabs:
        return "@div:0.500"
    return f"@div:{0.5 + 0.5 * max(-1.0, min(1.0, value / vabs)):.3f}"


# ---------------------------------------------------------------------------
# Generic charts
# ---------------------------------------------------------------------------
def bars(categories, series, unit="", stacked=False, horizontal=False,
         height=340, digits=2, ytitle=None, legend=True, text=False):
    """series: list of dicts(name, values, color)."""
    data = []
    for s in series:
        vals = _r(s["values"], digits)
        trace = {
            "type": "bar",
            "name": s["name"],
            "marker": {"color": s["color"]},
            "hovertemplate": f"%{{{'y' if horizontal else 'x'}}}<br>{s['name']}: "
                             f"%{{{'x' if horizontal else 'y'}:,.{digits}f}} {unit}<extra></extra>",
        }
        if horizontal:
            trace.update({"y": list(categories), "x": vals, "orientation": "h"})
        else:
            trace.update({"x": list(categories), "y": vals})
        if text:
            trace.update({"text": [f"{v:,.{digits}f}" if v is not None else "" for v in vals],
                          "textposition": "outside", "cliponaxis": False,
                          "textfont": {"color": "@ink2", "size": 11}})
        data.append(trace)
    layout = {
        "barmode": "relative" if stacked else "group",
        "showlegend": legend and len(series) > 1,
    }
    axis_title = {"text": ytitle or unit}
    if horizontal:
        layout["xaxis"] = {"title": axis_title, "zeroline": True}
        layout["yaxis"] = {"automargin": True, "autorange": "reversed"}
    else:
        layout["yaxis"] = {"title": axis_title, "zeroline": True}
    return figure(data, layout, height)


def lines(x, series, unit="", height=340, xtitle=None, digits=2, fill=None,
          hovermode="x unified", step=False):
    data = []
    for s in series:
        tr = {
            "type": "scatter", "mode": "lines", "name": s["name"],
            "x": list(x), "y": _r(s["values"], digits),
            "line": {"color": s["color"], "width": s.get("width", 2),
                     "shape": "hv" if step else "linear"},
            "hovertemplate": f"{s['name']}: %{{y:,.{digits}f}} {unit}<extra></extra>",
        }
        if s.get("dash"):
            tr["line"]["dash"] = s["dash"]
        if s.get("fill") or fill:
            tr["fill"] = s.get("fill") or fill
            tr["fillcolor"] = s.get("fillcolor", s["color"])
        data.append(tr)
    layout = {
        "hovermode": hovermode,
        "yaxis": {"title": {"text": unit}},
        "xaxis": {"title": {"text": xtitle or ""}},
        "showlegend": len(series) > 1,
    }
    return figure(data, layout, height)


def heatmap(z, x, y, unit="", scale="@seqscale", height=320, zmid=None,
            digits=1, xtitle="", ytitle=""):
    trace = {
        "type": "heatmap", "z": [_r(row, digits) for row in z], "x": list(x),
        "y": list(y), "colorscale": scale, "xgap": 2, "ygap": 2,
        "hovertemplate": f"%{{y}} · %{{x}}: %{{z:,.{digits}f}} {unit}<extra></extra>",
        "colorbar": {"title": {"text": unit, "side": "right"}, "thickness": 10,
                     "outlinewidth": 0},
    }
    if zmid is not None:
        trace["zmid"] = zmid
    layout = {"xaxis": {"title": {"text": xtitle}, "showgrid": False},
              "yaxis": {"title": {"text": ytitle}, "showgrid": False,
                        "automargin": True, "autorange": "reversed"}}
    return figure([trace], layout, height)


def histogram(groups, unit="", height=320, nbins=30, xtitle=""):
    data = []
    for g in groups:
        data.append({
            "type": "histogram", "name": g["name"], "x": _r(g["values"], 4),
            "marker": {"color": g["color"]}, "nbinsx": nbins, "opacity": 0.85,
            "hovertemplate": f"{g['name']}<br>%{{x}}: %{{y}} lines<extra></extra>",
        })
    layout = {"barmode": "overlay", "xaxis": {"title": {"text": xtitle}},
              "yaxis": {"title": {"text": "count"}}, "showlegend": len(groups) > 1}
    return figure(data, layout, height)


def dots(categories, series, unit="", height=320, digits=2):
    """Dot plot (Cleveland) - one row per category, one dot per series."""
    data = []
    for s in series:
        data.append({
            "type": "scatter", "mode": "markers", "name": s["name"],
            "y": list(categories), "x": _r(s["values"], digits),
            "marker": {"color": s["color"], "size": 11, "symbol": s.get("symbol", "circle"),
                       "line": {"color": "@surface", "width": 2}},
            "hovertemplate": f"%{{y}}<br>{s['name']}: %{{x:,.{digits}f}} {unit}<extra></extra>",
        })
    layout = {"xaxis": {"title": {"text": unit}},
              "yaxis": {"automargin": True, "autorange": "reversed"},
              "showlegend": True, "hovermode": "closest"}
    return figure(data, layout, height)


# ---------------------------------------------------------------------------
# Maps (equirectangular in a cartesian frame - works offline, no tiles)
# ---------------------------------------------------------------------------
class MapFigure:
    def __init__(self, extent, height=560, title_unit=""):
        self.extent = extent
        self.data = []
        self.height = height
        self.annotations = []

    def background(self, shapes):
        for s in shapes:
            self.data.append({
                "type": "scatter", "mode": "lines", "x": s["x"], "y": s["y"],
                "fill": "toself", "fillcolor": "@land" if s["model"] else "@landout",
                "line": {"color": "@border", "width": 0.8},
                "hoverinfo": "skip", "showlegend": False,
            })

    def polygon(self, geom, fill, line="@surface", width=1.5, name=None,
                hover=None, legend=False, opacity=1.0, tolerance=0.02):
        xs, ys = geo.polygon_coords(geom, tolerance)
        tr = {
            "type": "scatter", "mode": "lines", "x": xs, "y": ys,
            "fill": "toself", "fillcolor": fill, "opacity": opacity,
            "line": {"color": line, "width": width},
            "name": name or "", "showlegend": legend,
            "hoveron": "fills", "hoverinfo": "text" if hover else "skip",
            "text": hover or "",
        }
        if legend:
            tr["legendgroup"] = name
        self.data.append(tr)

    def outline(self, geom, color="@ink", width=1.6, name="Focus region (SH)",
                dash=None, legend=True):
        xs, ys = geo.polygon_coords(geom, 0.005)
        tr = {"type": "scatter", "mode": "lines", "x": xs, "y": ys,
              "line": {"color": color, "width": width}, "name": name,
              "hoverinfo": "skip", "showlegend": legend}
        if dash:
            tr["line"]["dash"] = dash
        self.data.append(tr)

    def segments(self, df, color_col, width_col=None, hover_col=None,
                 wmin=1.0, wmax=7.0, name=None, wvmax=None):
        """One trace per segment so each line can carry its own colour.

        ``wvmax`` fixes the value mapped to ``wmax`` (shared scale across maps).
        """
        if width_col is not None and len(df):
            v = df[width_col].astype(float)
            vmax = wvmax or (v.max() if v.max() > 0 else 1.0)
            widths = wmin + (wmax - wmin) * np.sqrt(v.clip(lower=0) / vmax)
        else:
            widths = pd.Series(2.0, index=df.index)
        for idx, row in df.iterrows():
            self.data.append({
                "type": "scatter", "mode": "lines",
                "x": [round(row.x0, 4), round(row.x1, 4)],
                "y": [round(row.y0, 4), round(row.y1, 4)],
                "line": {"color": row[color_col], "width": round(float(widths[idx]), 2)},
                "hoverinfo": "skip", "showlegend": False,
            })
        if hover_col is not None and len(df):
            self.data.append({
                "type": "scatter", "mode": "markers",
                "x": _r((df.x0 + df.x1) / 2, 4), "y": _r((df.y0 + df.y1) / 2, 4),
                "marker": {"size": 10, "color": "rgba(0,0,0,0)"},
                "text": list(df[hover_col]), "hovertemplate": "%{text}<extra></extra>",
                "showlegend": False, "name": name or "",
            })

    def points(self, x, y, color, size=7, hover=None, name="", legend=False,
               line="@surface", symbol="circle", opacity=1.0):
        self.data.append({
            "type": "scatter", "mode": "markers", "x": _r(x, 4), "y": _r(y, 4),
            "marker": {"color": color, "size": size if np.isscalar(size) else _r(size, 2),
                       "line": {"color": line, "width": 1}, "symbol": symbol,
                       "opacity": opacity},
            "text": hover if hover is not None else "",
            "hovertemplate": "%{text}<extra></extra>" if hover is not None else None,
            "hoverinfo": None if hover is not None else "skip",
            "name": name, "showlegend": legend,
        })

    def labels(self, x, y, text, size=11):
        self.data.append({
            "type": "scatter", "mode": "text", "x": _r(x, 4), "y": _r(y, 4),
            "text": list(text), "textfont": {"color": "@ink", "size": size},
            "hoverinfo": "skip", "showlegend": False,
        })

    def colorbar(self, vmin, vmax, title, scale="@seqscale"):
        self.data.append({
            "type": "scatter", "mode": "markers", "x": [None], "y": [None],
            "marker": {"color": [vmin, vmax], "colorscale": scale, "cmin": vmin,
                       "cmax": vmax, "size": 0.1, "showscale": True,
                       "colorbar": {"title": {"text": title, "side": "right"},
                                    "thickness": 10, "len": 0.6, "outlinewidth": 0,
                                    "x": 1.0}},
            "hoverinfo": "skip", "showlegend": False,
        })

    def size_legend(self, values, sizes, unit, color="@muted"):
        for v, s in zip(values, sizes):
            self.data.append({
                "type": "scatter", "mode": "markers", "x": [None], "y": [None],
                "marker": {"size": s, "color": color, "line": {"color": "@surface", "width": 1}},
                "name": (f"{v:,.0f} {unit}" if v >= 100 else f"{v:,.2g} {unit}"), "showlegend": True,
                "legendgroup": "size",
            })

    def spec(self):
        x0, x1, y0, y1 = self.extent
        lat = (y0 + y1) / 2
        layout = {
            "xaxis": {"range": [x0, x1], "visible": False, "fixedrange": False},
            "yaxis": {"range": [y0, y1], "visible": False,
                      "scaleanchor": "x", "scaleratio": round(1 / math.cos(math.radians(lat)), 3)},
            "hovermode": "closest",
            "showlegend": True,
            "legend": {"orientation": "h", "y": -0.02, "yanchor": "top", "x": 0},
            "dragmode": "pan",
            "annotations": self.annotations,
        }
        return figure(self.data, layout, self.height, kind="map")


def bubble_sizes(values, smax=34, smin=4, vmax=None):
    v = np.abs(np.asarray(values, dtype=float))
    vmax = vmax or (v.max() if len(v) and v.max() > 0 else 1.0)
    return smin + (smax - smin) * np.sqrt(v / vmax), vmax
