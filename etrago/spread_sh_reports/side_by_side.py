"""Side-by-side page: all configurations next to each other.

* KPI matrix  - every indicator for every configuration in one view,
  coloured by the change against the status quo (better / worse).
* KPI dot charts - one small chart per indicator.
* Map matrix  - one row per map theme, one column per configuration, all
  maps of a row on the same colour and size scale, with linked pan/zoom and
  a page-wide Germany / Schleswig-Holstein switch.
"""

import json

import numpy as np
import pandas as pd

from . import charts as C
from . import geo
from . import html as H
from .config import SCENARIO_COLOR, SCENARIO_LABELS
from .report import zone_anchor

MAP_H = 330

# (key, label, unit, digits, better: "lower" | "higher" | None, getter)
KPIS = [
    ("de_avg_price", "Avg. German wholesale price", "€/MWh", 2, "lower", lambda m: m.de_avg_price),
    ("price_gap", "Max. zonal price gap", "€/MWh", 3, None, lambda m: m.kpis()["max_zone_price_gap"]),
    ("convergence", "Hours with one German price", "%", 1, None, lambda m: m.price_convergence * 100),
    ("consumer", "Wholesale cost of DE demand", "bn€", 2, "lower", lambda m: m.consumer_cost.sum() / 1e9),
    ("rd_total", "Redispatch volume DE", "TWh", 2, "lower",
     lambda m: m.rd_total["up_de_twh"] + m.rd_total["down_de_twh"]),
    ("rd_cost", "Redispatch cost", "M€", 0, "lower", lambda m: m.rd_total["cost_meur"]),
    ("curt_mkt", "Curtailment in the market DE", "TWh", 1, "lower", lambda m: m.curt_total["market_de_twh"]),
    ("curt_grid", "Curtailment after redispatch DE", "TWh", 1, "lower", lambda m: m.curt_total["grid_de_twh"]),
    ("exp_gw", "AC grid expansion DE", "GW", 1, "lower", lambda m: m.lines_total["ac_exp_gw_de"]),
    ("exp_gwkm", "AC grid expansion DE", "GW·km", 0, "lower", lambda m: m.lines_total["ac_exp_gwkm_de"]),
    ("grid_inv", "Grid investment (AC + DC)", "M€/a", 0, "lower",
     lambda m: m.lines_total["ac_invest_meur"] + m.lines_total["dc_invest_meur"]),
    ("loading", "Mean line loading DE", "%", 1, None, lambda m: m.lines_total["mean_loading_de"] * 100),
    ("rent", "Congestion rent, DE corridors", "M€", 0, None,
     lambda m: m.exchange[m.exchange.kind != "Abroad"].rent_meur.sum() if len(m.exchange) else 0.0),
    ("ely", "Electrolysers DE", "GW", 2, None, lambda m: m.ely_by_zone.gw.sum()),
    ("bat", "Batteries DE", "GW", 2, None, lambda m: m.bat_by_zone[["existing_gw", "new_gw"]].sum().sum()),
    ("cost_ex", "System cost excl. gas imports", "bn€/a", 2, "lower", lambda m: m.kpis()["cost_excl_gas_beur"]),
    ("sh_curt", "SH: curtailment after redispatch", "TWh", 2, "lower", lambda m: m.focus["curt_grid_twh"]),
    ("sh_rd", "SH: redispatch volume", "TWh", 2, "lower", lambda m: m.focus["rd_down_twh"] + m.focus["rd_up_twh"]),
    ("sh_price", "SH: mean nodal price", "€/MWh", 2, None, lambda m: m.focus["nodal_price"]),
    ("sh_ely", "SH: electrolysers", "GW", 2, None, lambda m: m.focus["ely_gw"]),
]


def _fmt(v, digits):
    return H.fmt(v, digits)


class SideBySideReport:
    def __init__(self, metrics, nav, links, data_dir, plotly_js=None):
        self.M = metrics
        self.nav = nav
        self.links = links
        self.data_dir = data_dir
        self.plotly_js = plotly_js
        self.figs = {}
        self.sections = []
        self.toc = []
        self.sq = metrics.get("status_quo")
        self.K = {k: {kp[0]: kp[5](m) for kp in KPIS} for k, m in metrics.items()}

    fig = lambda self, fid, spec: (self.figs.__setitem__(fid, spec), fid)[1]

    def add(self, sid, title, lead, body):
        self.toc.append((sid, title))
        self.sections.append(H.section(sid, title, lead, body))

    def labels(self):
        return [SCENARIO_LABELS.get(k, k) for k in self.M]

    # ------------------------------------------------------------------
    def build(self):
        self.kpi_matrix()
        self.kpi_dots()
        self.map_matrix()
        de, sh = geo.MAP_EXTENT["de"], geo.FOCUS_EXTENT
        ext = lambda e: H.esc(json.dumps({"group": "de", "x": [e[0], e[1]], "y": [e[2], e[3]]}))
        controls = (
            '<span class="seg" role="group" aria-label="Map extent">'
            f'<button type="button" data-extent="{ext(de)}" aria-pressed="true">Germany</button>'
            f'<button type="button" data-extent="{ext(sh)}" aria-pressed="false">Schleswig-Holstein</button>'
            "</span>"
        )
        shared = {"de": self._background(geo.MAP_EXTENT["de"]),
                  "eu": self._background(geo.MAP_EXTENT["europe"])}
        return H.page(
            title="SPREAD.SH side by side",
            heading="All configurations side by side",
            intro=("Every indicator and every map for the five bidding-zone configurations at once. "
                   "Maps in a row share one colour and size scale, so differences are directly "
                   "visible. Panning or zooming one map moves its whole row; the switch in the header "
                   "zooms all German maps to Schleswig-Holstein."),
            chips=[f"{len(self.M)} configurations", "shared scales per row", "linked zoom",
                   "Δ = change vs status quo"],
            nav=self.nav, toc=self.toc, sections_html="".join(self.sections), figs=self.figs,
            plotly_js=self.plotly_js, footer="Generated by spread_sh_reports.",
            shared=shared, controls=controls, body_class="wide",
        )

    def _background(self, extent):
        mp = C.MapFigure(extent)
        mp.background(geo.background(self.data_dir, extent))
        return mp.data

    # ------------------------------------------------------------------
    # KPI overview
    # ------------------------------------------------------------------
    def kpi_matrix(self):
        keys = list(self.M)
        rows, z, text, hover = [], [], [], []
        for key, label, unit, dg, better, _ in KPIS:
            rows.append(f"{label} [{unit}]")
            zr, tr, hr = [], [], []
            ref = self.K["status_quo"][key] if self.sq is not None else None
            for k in keys:
                v = self.K[k][key]
                if ref is None or k == "status_quo" or better is None or not np.isfinite(v):
                    score = 0.0
                else:
                    rel = (v - ref) / abs(ref) if abs(ref) > 1e-9 else 0.0
                    score = rel if better == "lower" else -rel  # + = worse
                    score = float(np.clip(score / 0.15, -1, 1))  # ±15 % saturates
                zr.append(round(score, 3))
                d = ""
                if ref is not None and k != "status_quo":
                    dv = v - ref
                    d = f" ({'+' if dv >= 0 else '−'}{_fmt(abs(dv), dg)})"
                tr.append(f"{_fmt(v, dg)}{d}")
                hr.append(f"{SCENARIO_LABELS.get(k, k)}<br>{label}: {_fmt(v, dg)} {unit}{d}"
                          + ("" if better else "<br>(no better/worse direction)"))
            z.append(zr)
            text.append(tr)
            hover.append(hr)
        trace = {
            "type": "heatmap", "z": z, "x": self.labels(), "y": rows,
            "text": text, "texttemplate": "%{text}", "textfont": {"size": 12, "color": "@ink"},
            "customdata": hover, "hovertemplate": "%{customdata}<extra></extra>",
            "colorscale": "@divsoftscale", "zmin": -1, "zmax": 1, "zmid": 0, "xgap": 3, "ygap": 3,
            "showscale": False,
        }
        layout = {"xaxis": {"side": "top", "showgrid": False, "tickfont": {"size": 13}},
                  "yaxis": {"autorange": "reversed", "showgrid": False, "automargin": True},
                  "margin": {"l": 10, "r": 10, "t": 40, "b": 10}}
        fid = self.fig("sbs-kpi-matrix", C.figure([trace], layout, height=70 + 34 * len(rows)))
        legend = ('<div class="note">Cell colour: <span style="color:var(--good)">blue = better</span>, '
                  '<span style="color:var(--bad)">red = worse</span> than the status quo, saturating at ±15 %. '
                  'Grey = reference or an indicator without a preferable direction (prices, flexible '
                  'assets, loading). Values in brackets: difference to the status quo.</div>')
        cards = []
        for k, m in self.M.items():
            cards.append(f'<a class="card" href="{H.esc(self.links[k])}"><div class="ft">'
                         f'{H.esc(SCENARIO_LABELS.get(k, k))}</div><div class="cs">'
                         f'{", ".join(H.esc(m.label(z)) for z in m.de_zones)}</div></a>')
        body = (H.grid(H.fig_block(fid, "KPI matrix", wide=True)) + legend
                + f"<div class='cards'>{''.join(cards)}</div>")
        self.add("kpis", "KPI matrix",
                 "All headline indicators for all configurations in one table.", body)

    def kpi_dots(self):
        blocks = []
        keys = list(self.M)
        for key, label, unit, dg, better, _ in KPIS:
            vals = [self.K[k][key] for k in keys]
            data = [{
                "type": "scatter", "mode": "markers+text", "y": self.labels(), "x": C._r(vals, 4),
                "marker": {"color": [SCENARIO_COLOR.get(k, "@muted") for k in keys], "size": 13,
                           "line": {"color": "@surface", "width": 2}},
                "text": [_fmt(v, dg) for v in vals], "textposition": "middle right",
                "textfont": {"size": 11, "color": "@ink2"}, "cliponaxis": False,
                "hovertemplate": "%{y}: %{x:,." + str(dg) + "f} " + unit + "<extra></extra>",
            }]
            lo, hi = min(vals), max(vals)
            pad = (hi - lo) * 0.35 if hi > lo else (abs(hi) * 0.05 or 1)
            layout = {"xaxis": {"range": [lo - pad * 0.4, hi + pad], "title": {"text": unit},
                                "zeroline": False},
                      "yaxis": {"autorange": "reversed", "automargin": True},
                      "showlegend": False, "hovermode": "closest",
                      "margin": {"l": 8, "r": 10, "t": 6, "b": 8}}
            spec = C.figure(data, layout, height=210)
            spec["compact"] = True
            fid = self.fig(f"sbs-dot-{key}", spec)
            hint = "" if better is None else (" · lower is better" if better == "lower" else " · higher is better")
            blocks.append(f'<figure class="card"><figcaption><div class="ft">{H.esc(label)}'
                          f'<span class="note" style="font-weight:400">{hint}</span></div></figcaption>'
                          f'<div class="chart" id="{fid}" role="img" aria-label="{H.esc(label)}"></div></figure>')
        self.add("kpi-charts", "KPI charts",
                 "One small chart per indicator. Each dot is a configuration (same colour on every chart). "
                 "Axes are zoomed to the data, so check the numbers before judging how large a gap is.",
                 f"<div class='mini'>{''.join(blocks)}</div>")

    # ------------------------------------------------------------------
    # Map matrix
    # ------------------------------------------------------------------
    def _base(self, m, extent, fills=None, height=MAP_H):
        mp = C.MapFigure(extent, height=height)
        for _, row in m.zone_gdf.iterrows():
            fill = (fills or {}).get(row.zone, "rgba(0,0,0,0)")
            mp.polygon(row.geometry, fill, line="@ink2", width=1.0)
        mp.outline(geo.focus_outline(m.data_dir), color="@ink", width=1.2, dash="dot", legend=False)
        return mp

    def _finish(self, mp, theme, k, extent_group="de", bg="de"):
        spec = mp.spec()
        spec["bg"] = bg
        spec["group"] = theme
        spec["extentGroup"] = extent_group
        spec["compact"] = True
        spec["layout"]["showlegend"] = False
        return self.fig(f"sbs-{theme}-{k}", spec)

    def _legend(self, theme, vmin, vmax, title, scale="@seqscale"):
        data = [{
            "type": "scatter", "mode": "markers", "x": [None], "y": [None],
            "marker": {"color": [vmin, vmax], "colorscale": scale, "cmin": vmin, "cmax": vmax,
                       "size": 0.1, "showscale": True,
                       "colorbar": {"orientation": "h", "x": 0, "xanchor": "left", "y": 0.5,
                                    "yanchor": "middle", "len": 1, "thickness": 10,
                                    "outlinewidth": 0, "title": {"text": title, "side": "top"}}},
            "hoverinfo": "skip", "showlegend": False,
        }]
        layout = {"xaxis": {"visible": False}, "yaxis": {"visible": False},
                  "margin": {"l": 6, "r": 6, "t": 0, "b": 0}, "plot_bgcolor": "rgba(0,0,0,0)"}
        spec = C.figure(data, layout, height=64)
        spec["compact"] = True
        return self.fig(f"sbs-legend-{theme}", spec)

    def _delta_html(self, k, value_fn, dg, unit, better):
        if self.sq is None or k == "status_quo":
            return ""
        dv = value_fn(self.M[k]) - value_fn(self.sq)
        if abs(dv) < 10 ** (-dg) / 2:
            return " <span>±0 vs SQ</span>"
        cls = "" if better is None else ("good" if (dv < 0) == (better == "lower") else "bad")
        return f' <span class="{cls}">{"+" if dv > 0 else "−"}{_fmt(abs(dv), dg)} {unit} vs SQ</span>'

    def _row(self, theme, title, note, cells, legend_id=None, size_note=""):
        """cells: list of (key, fig_id, kpi_html)."""
        parts = []
        for k, fid, kpi in cells:
            parts.append(
                f'<div class="mcell"><div class="mlabel"><span>{H.esc(SCENARIO_LABELS.get(k, k))}</span>'
                f'<a href="{H.esc(self.links[k])}">report →</a></div>'
                f'<div class="chart" id="{fid}" role="img" aria-label="{H.esc(title)} – {H.esc(SCENARIO_LABELS.get(k, k))}"></div>'
                f'<div class="mkpi">{kpi}</div></div>')
        leg = f'<div class="legendbar chart" id="{legend_id}"></div>' if legend_id else ""
        size = f'<p class="note">{size_note}</p>' if size_note else ""
        self.toc.append((f"row-{theme}", title))
        self.sections.append(
            f'<div class="maprow" id="row-{theme}"><h3>{H.esc(title)}</h3><p class="note">{note}</p>'
            f'{leg}{size}<div class="mrow-wrap"><div class="mrow" style="--n:{len(cells)}">{"".join(parts)}</div></div></div>')

    def map_matrix(self):
        self.sections.append(H.section(
            "maps", "Maps side by side",
            "Rows = topics, columns = configurations. Scroll inside a map to zoom; the whole row follows. "
            "Double-click resets.", ""))
        self.toc.append(("maps", "Maps side by side"))
        M = self.M
        ext = geo.MAP_EXTENT["de"]

        # 1. Zone layout -------------------------------------------------
        cells = []
        for k, m in M.items():
            fills = {z: f"@fade:s{i + 1}" for i, z in enumerate(m.de_zones)}
            mp = self._base(m, ext, fills)
            ac = m.ac[m.ac.zone.isin(m.de_zones)]
            mp.points(ac.x, ac.y, [f"@s{m.de_zones.index(z) + 1}" for z in ac.zone], size=5,
                      hover=[f"Bus {b} · {H.esc(m.label(z))}" for b, z in zip(ac.index, ac.zone)])
            lab = [zone_anchor(m, z) for z in m.de_zones]
            mp.labels([p[0] for p in lab], [p[1] for p in lab],
                      [f"<b>{m.label(z).split(' ')[0]}</b>" for z in m.de_zones], size=12)
            kpi = f"<b>{len(m.de_zones)}</b> German zone{'s' if len(m.de_zones) > 1 else ''} · SH in {H.esc(m.label(m.focus_zone))}"
            cells.append((k, self._finish(mp, "zones", k), kpi))
        self._row("zones", "Bidding-zone layout", "German bidding zones and the clustered AC nodes they contain. "
                  "Dotted outline = Schleswig-Holstein.", cells)

        # 2. Zonal prices -------------------------------------------------
        allp = pd.concat([m.price_stats.loc[m.de_zones, "mean"] for m in M.values()])
        vmin, vmax = float(allp.min()), float(allp.max())
        cells = []
        for k, m in M.items():
            vals = m.price_stats["mean"]
            mp = self._base(m, ext, {z: C.seq(vals[z], vmin, vmax) for z in m.de_zones})
            lab = [zone_anchor(m, z) for z in m.de_zones]
            mp.labels([p[0] for p in lab], [p[1] for p in lab],
                      [f"<b>{m.label(z).split(' ')[0]}</b><br>{vals[z]:.2f}" for z in m.de_zones], size=11)
            kpi = (f"<b>{H.fmt(m.de_avg_price, 2, '€/MWh')}</b> DE average"
                   + self._delta_html(k, lambda x: x.de_avg_price, 2, "€/MWh", "lower")
                   + f"<br>one price in {H.pct(m.price_convergence)} of hours")
            cells.append((k, self._finish(mp, "price", k), kpi))
        self._row("price", "Average zonal market price",
                  f"Time-weighted mean price per German zone. The shared scale spans only "
                  f"{vmax - vmin:.2f} €/MWh, so small differences look strong.", cells,
                  self._legend("price", vmin, vmax, "mean zonal price [€/MWh]"))

        # 3. Nodal shadow prices ------------------------------------------
        alln = pd.concat([m.nodal_price.reindex(m.ac.index[m.ac.zone.isin(m.de_zones)]) for m in M.values()]).dropna()
        lo, hi = float(alln.quantile(0.02)), float(alln.quantile(0.98))
        cells = []
        for k, m in M.items():
            mp = self._base(m, ext)
            de = m.ac[m.ac.zone.isin(m.de_zones)]
            v = m.nodal_price.reindex(de.index)
            mp.points(de.x, de.y, [C.seq(p, lo, hi) for p in v], size=8,
                      hover=[f"Bus {b}<br>{p:.2f} €/MWh" for b, p in zip(de.index, v)])
            kpi = (f"SH <b>{H.fmt(m.focus['nodal_price'], 2)}</b> · DE {H.fmt(m.focus['nodal_price_de'], 2)} €/MWh"
                   + self._delta_html(k, lambda x: x.focus["nodal_price"], 2, "€/MWh", None))
            cells.append((k, self._finish(mp, "nodal", k), kpi))
        self._row("nodal", "Nodal shadow prices (grid optimisation)",
                  "Marginal cost of supply at each node in the grid model. Gradients show where the grid binds.",
                  cells, self._legend("nodal", lo, hi, "nodal price [€/MWh]"))

        # 4. Redispatch ---------------------------------------------------
        vmax_rd = max(float(m.rd_nodes.volume.max()) for m in M.values())
        cells = []
        for k, m in M.items():
            mp = self._base(m, ext)
            n = m.rd_nodes[m.rd_nodes.volume > 1e-4]
            ac = m.ac.loc[n.index]
            sizes, _ = C.bubble_sizes(n.volume.values, smax=28, smin=3, vmax=vmax_rd)
            share = (n.net / n.volume).fillna(0)
            mp.points(ac.x, ac.y, [C.div(s, 1.0) for s in share], size=sizes, opacity=0.9,
                      hover=[f"Bus {b}<br>up {u:.2f} · down {-d:.2f} TWh" for b, u, d in zip(n.index, n.up, n.down)])
            t = m.rd_total
            kpi = (f"<b>{H.fmt(t['up_de_twh'] + t['down_de_twh'], 1, 'TWh')}</b> in DE"
                   + self._delta_html(k, lambda x: x.rd_total["up_de_twh"] + x.rd_total["down_de_twh"], 1, "TWh", "lower")
                   + f"<br>up {H.fmt(t['up_de_twh'], 1)} · down {H.fmt(t['down_de_twh'], 1)} · {H.fmt(t['cost_meur'], 0)} M€")
            cells.append((k, self._finish(mp, "rd", k), kpi))
        self._row("rd", "Redispatch", "Bubble area = redispatched energy per node (shared scale); colour = "
                  "mainly ramped down (blue) or up (red).", cells,
                  self._legend("rd", -100, 100, "net direction [%]  (− down … up +)", "@divscale"),
                  f"Largest bubble = {vmax_rd:.1f} TWh.")

        # 5. Curtailment --------------------------------------------------
        vmax_c = max(float(m.curt_nodes.curt_grid_twh.max()) for m in M.values())
        rates = pd.concat([(m.curt_nodes.curt_grid_twh / m.curt_nodes.avail_twh) for m in M.values()]).dropna()
        rmax = max(0.05, float(rates.quantile(0.98)))
        cells = []
        for k, m in M.items():
            mp = self._base(m, ext)
            n = m.curt_nodes[m.curt_nodes.curt_grid_twh > 1e-3]
            ac = m.ac.loc[n.index]
            rate = n.curt_grid_twh / n.avail_twh
            sizes, _ = C.bubble_sizes(n.curt_grid_twh.values, smax=28, smin=3, vmax=vmax_c)
            mp.points(ac.x, ac.y, [C.seq(r, 0, rmax) for r in rate], size=sizes,
                      hover=[f"Bus {b}<br>{c:.2f} TWh ({r * 100:.1f} %)" for b, c, r in zip(n.index, n.curt_grid_twh, rate)])
            t = m.curt_total
            kpi = (f"<b>{H.fmt(t['grid_de_twh'], 1, 'TWh')}</b> after redispatch"
                   + self._delta_html(k, lambda x: x.curt_total["grid_de_twh"], 1, "TWh", "lower")
                   + f"<br>market {H.fmt(t['market_de_twh'], 1)} TWh · SH {H.fmt(t['grid_focus_twh'], 1)} TWh")
            cells.append((k, self._finish(mp, "curt", k), kpi))
        self._row("curt", "Wind & solar curtailment after redispatch",
                  "Bubble area = curtailed energy per node (shared scale); colour = curtailment rate.", cells,
                  self._legend("curt", 0, rmax * 100, "curtailment rate [%]"),
                  f"Largest bubble = {vmax_c:.1f} TWh.")

        # 6. Electrolysers ------------------------------------------------
        vmax_e = max(float(m.ely_nodes.gw.max()) for m in M.values() if len(m.ely_nodes))
        cells = []
        for k, m in M.items():
            mp = self._base(m, ext)
            n = m.ely_nodes[m.ely_nodes.gw > 0.001]
            ac = m.ac.loc[n.index]
            flh = (n.twh * 1e6 / (n.gw * 1e3)).fillna(0)
            sizes, _ = C.bubble_sizes(n.gw.values, smax=28, smin=3, vmax=vmax_e)
            mp.points(ac.x, ac.y, [C.seq(f, 0, 8760) for f in flh], size=sizes,
                      hover=[f"Bus {b}<br>{g:.2f} GW · {f:,.0f} h" for b, g, f in zip(n.index, n.gw, flh)])
            kpi = (f"<b>{H.fmt(m.ely_by_zone.gw.sum(), 1, 'GW')}</b> in DE"
                   + self._delta_html(k, lambda x: x.ely_by_zone.gw.sum(), 1, "GW", None)
                   + f"<br>SH {H.fmt(m.focus['ely_gw'], 2)} GW")
            cells.append((k, self._finish(mp, "ely", k), kpi))
        self._row("ely", "Electrolysers", "Bubble area = electrolysis capacity (shared scale); colour = full-load "
                  "hours in the grid optimisation.", cells, self._legend("ely", 0, 8760, "full-load hours"),
                  f"Largest bubble = {vmax_e:.2f} GW.")

        # 7. Batteries ----------------------------------------------------
        tot = {k: (m.bat_nodes.existing_gw + m.bat_nodes.new_gw) for k, m in M.items()}
        vmax_b = max(float(t.max()) for t in tot.values())
        cells = []
        for k, m in M.items():
            mp = self._base(m, ext)
            t = tot[k][tot[k] > 0.005]
            n = m.bat_nodes.loc[t.index]
            ac = m.ac.loc[t.index]
            sizes, _ = C.bubble_sizes(t.values, smax=28, smin=3, vmax=vmax_b)
            mp.points(ac.x, ac.y, [C.seq(s, 0, 1) for s in (n.new_gw / t).fillna(0)], size=sizes,
                      hover=[f"Bus {b}<br>{v:.2f} GW ({a:.2f} added)" for b, v, a in zip(t.index, t, n.new_gw)])
            v = m.bat_by_zone[["existing_gw", "new_gw"]].sum()
            kpi = (f"<b>{H.fmt(v.sum(), 1, 'GW')}</b> in DE"
                   + self._delta_html(k, lambda x: x.bat_by_zone[["existing_gw", "new_gw"]].sum().sum(), 2, "GW", None)
                   + f"<br>added by grid optimisation {H.fmt(v['new_gw'], 2)} GW")
            cells.append((k, self._finish(mp, "bat", k), kpi))
        self._row("bat", "Battery storage", "Bubble area = battery power (shared scale); colour = share added by "
                  "the grid optimisation.", cells, self._legend("bat", 0, 100, "added capacity [%]"),
                  f"Largest bubble = {vmax_b:.2f} GW.")

        # 8-10. Lines ------------------------------------------------------
        de_lines = {k: m.lines[m.lines.category != "Abroad"].copy() for k, m in M.items()}
        wv = max(float(L.s_nom_opt.max()) for L in de_lines.values())

        def line_row(theme, title, note, col, vmax, legend_title, scale_factor, kpi_fn, hover_fn):
            cells = []
            for k, m in M.items():
                mp = self._base(m, ext)
                L = de_lines[k].copy()
                L["color"] = [C.seq(v, 0, vmax) for v in L[col]]
                L["hover"] = [hover_fn(r) for r in L.itertuples()]
                mp.segments(L, "color", "s_nom_opt", "hover", wmin=0.8, wmax=6, wvmax=wv)
                cells.append((k, self._finish(mp, theme, k), kpi_fn(k, m)))
            self._row(theme, title, note, cells, self._legend(theme, 0, vmax * scale_factor, legend_title))

        line_row("loading", "Average line loading", "Mean |flow| / limit after expansion; width = optimised capacity "
                 "(shared scale).", "mean_loading", 1.0, "mean loading [%]", 100,
                 lambda k, m: (f"<b>{H.pct(m.lines_total['mean_loading_de'])}</b> mean (length-weighted)"
                               f"<br>{m.lines_total['congested_lines_de']} lines at limit > 10 % of hours"),
                 lambda r: f"Line {r.Index}<br>mean loading {r.mean_loading * 100:.0f} %")
        cmax = max(0.2, max(float(L.congested_share.max()) for L in de_lines.values()))
        line_row("limit", "Hours at the thermal limit", "Share of hours a line runs at ≥ 98 % of its limit.",
                 "congested_share", cmax, "hours at limit [%]", 100,
                 lambda k, m: (f"<b>{m.lines_total['congested_lines_de']}</b> of {m.lines_total['n_lines_de']} lines "
                               "congested > 10 % of hours"),
                 lambda r: f"Line {r.Index}<br>at limit {r.congested_share * 100:.1f} % of hours")
        for L in de_lines.values():
            L["exp_capped"] = L.expansion_rel.fillna(0).clip(upper=3)
        line_row("exp", "Grid expansion", "Capacity added relative to the existing line (capped at 300 %).",
                 "exp_capped", 3.0, "expansion [% of existing]", 100,
                 lambda k, m: (f"<b>{H.fmt(m.lines_total['ac_exp_gwkm_de'], 0, 'GW·km')}</b>"
                               + self._delta_html(k, lambda x: x.lines_total["ac_exp_gwkm_de"], 0, "GW·km", "lower")
                               + f"<br>{H.fmt(m.lines_total['ac_exp_gw_de'], 1)} GW · "
                                 f"{H.fmt(m.lines_total['ac_invest_meur'] + m.lines_total['dc_invest_meur'], 0)} M€/a"),
                 lambda r: f"Line {r.Index}<br>{r.s_nom_base:,.0f} → {r.s_nom_opt:,.0f} MVA")

        # 11. Change in expansion vs status quo ---------------------------
        if self.sq is not None:
            base = de_lines["status_quo"].expansion_mw
            d_all = pd.concat([de_lines[k].expansion_mw - base for k in M if k != "status_quo"])
            vabs = max(200.0, float(d_all.abs().quantile(0.98)))
            cells = []
            for k, m in M.items():
                mp = self._base(m, ext)
                L = de_lines[k].copy()
                if k == "status_quo":
                    L["color"] = "@muted"
                    L["hover"] = [f"Line {i}<br>expansion {v:,.0f} MW (reference)" for i, v in zip(L.index, L.expansion_mw)]
                    mp.segments(L, "color", "expansion_mw", "hover", wmin=0.6, wmax=5)
                    kpi = "<b>Reference</b> · line width = expansion"
                else:
                    d = L.expansion_mw - base.reindex(L.index).fillna(0)
                    L["color"] = [C.div(v, vabs) for v in d]
                    L["absd"] = d.abs()
                    L["hover"] = [f"Line {i}<br>Δ expansion {v:+,.0f} MW" for i, v in zip(L.index, d)]
                    mp.segments(L, "color", "absd", "hover", wmin=0.6, wmax=6, wvmax=vabs)
                    kpi = (f"<b>{(d > 50).sum()}</b> lines more · <b>{(d < -50).sum()}</b> lines less "
                           f"(&gt; 50 MW)<br>net {d.sum() / 1e3:+.1f} GW")
                cells.append((k, self._finish(mp, "dexp", k), kpi))
            self._row("dexp", "Change in grid expansion vs status quo",
                      "Red = more expansion than in the status quo, blue = less; width = size of the change.",
                      cells, self._legend("dexp", -vabs, vabs, "Δ expansion [MW]", "@divscale"))

        # 12. Cross-zonal corridors (Europe) ------------------------------
        caps = pd.concat([m.exchange.capacity_mw for m in M.values() if len(m.exchange)])
        cong_max = max(0.5, max(float(m.exchange.congested_share.max()) for m in M.values() if len(m.exchange)))
        cells = []
        eu = geo.MAP_EXTENT["europe"]
        for k, m in M.items():
            mp = self._base(m, eu, {z: f"@fade:s{i + 1}" for i, z in enumerate(m.de_zones)})
            ex = m.exchange.copy()
            if len(ex):
                an = {z: zone_anchor(m, z) for z in set(ex.zone_a) | set(ex.zone_b)}
                ex["x0"] = ex.zone_a.map(lambda z: an[z][0])
                ex["y0"] = ex.zone_a.map(lambda z: an[z][1])
                ex["x1"] = ex.zone_b.map(lambda z: an[z][0])
                ex["y1"] = ex.zone_b.map(lambda z: an[z][1])
                ex["color"] = [C.seq(v, 0, cong_max) for v in ex.congested_share]
                ex["hover"] = [f"{H.esc(r.label)}<br>{r.capacity_mw:,.0f} MW · congested {r.congested_share * 100:.0f} %"
                               f"<br>rent {r.rent_meur:,.0f} M€" for r in ex.itertuples()]
                mp.segments(ex, "color", "capacity_mw", "hover", wmin=1.0, wmax=8, wvmax=float(caps.max()))
            internal = ex[ex.kind == "DE internal"] if len(ex) else ex
            kpi = (f"<b>{H.fmt(self.K[k]['rent'], 0, 'M€')}</b> congestion rent on DE corridors"
                   + (f"<br>internal corridors congested {H.pct(internal.congested_share.mean())} of hours (mean)"
                      if len(internal) else "<br>no internal German corridors"))
            cells.append((k, self._finish(mp, "corr", k, extent_group="eu", bg="eu"), kpi))
        self._row("corr", "Cross-zonal corridors (market model)",
                  "Width = transfer capacity (shared scale), colour = share of hours congested.", cells,
                  self._legend("corr", 0, cong_max * 100, "% of hours congested"))
