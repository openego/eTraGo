"""Local-prices page: the figures of Agora Energiewende & Fraunhofer IEE (2025)
"Local electricity prices in Germany" reproduced with the eTraGo runs.

Agora compares a single zone, three zones and "local prices" (22 hubs). Here:

* single zone   = status-quo market model (DE/LU),
* zones         = the DE2 … DE5 market models,
* local prices  = nodal shadow prices of the status-quo grid optimisation at
  the German AC nodes (benchmark for congestion-minimising prices, as the
  local-price simulation in the Agora study).

Figures A–E follow the numbering of the Agora summary.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import charts as C
from . import geo
from . import html as H
from .config import SCENARIO_COLOR, SCENARIO_LABELS, VRES_CARRIERS
from .loader import Stage
from .report import base_map, zone_anchor

PRICE_CLIP = (-200.0, 500.0)  # load-shedding spikes (0.01 % of values)
LOCAL = "Local prices"
LOCAL_COLOR = "@s7"
CHEAP = 20.0          # €/MWh threshold for "cheap" electrolysis hours
ELY_HOURS = 3000.0    # annual operation for the procurement price
CONGESTED_MW = 3000.0  # Agora: hours with more than 3 GW redispatch
REGIONS = {  # name -> (lon, lat) reference point, nearest German node
    "Neckar (Mündung)": (8.55, 49.45),
    "Elbsandstein": (14.0, 50.95),
}
# Agora Energiewende & Fraunhofer IEE (2025), summary, 2019-2023 retrospective
AGORA = {
    "conv_single": "36 % (2019) · 17 % (2023)",
    "conv_zones": "DE2: 20–54 % · DE3: 20–66 % · DE5: 20–98 %",
    "coast_dev": "−19 … −29 %",
    "south_dev": "+4 … +7 %",
    "rd_cost": "5.3 (single) · 4.7 (three zones) · 0 (local)",
    "price_rd": "102 (single) · 95 (local, load-weighted) · 69–104 range",
    "north_gap": "−23 (all hours) · −57 (> 3 GW redispatch)",
    "south_gap": "+7 (all hours) · +24 (> 3 GW redispatch)",
    "res_rev": "17.4 / 16.3 / 13.5 bn€ (single / three zones / local)",
    "ely": "coast up to 10× the full-load hours of the south; −8 … 67 €/MWh",
}


def _wmean(v, w):
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    return float((v * w).sum() / w.sum()) if w.sum() else np.nan


class LocalPricesReport:
    def __init__(self, metrics, nav, links, data_dir, plotly_js=None, nodal_dir=None):
        self.M = metrics
        self.nav = nav
        self.links = links
        self.data_dir = data_dir
        self.plotly_js = plotly_js
        self.nodal_dir = Path(nodal_dir) if nodal_dir else None
        self.figs = {}
        self.sections = []
        self.toc = []
        self.ref = metrics.get("status_quo") or next(iter(metrics.values()))
        self.zonal_keys = [k for k in metrics if k != "status_quo"]
        self._prepare()

    fig = lambda self, fid, spec: (self.figs.__setitem__(fid, spec), fid)[1]

    def add(self, sid, title, lead, body):
        self.toc.append((sid, title))
        self.sections.append(H.section(sid, title, lead, body))

    # ------------------------------------------------------------------
    def _prepare(self):
        """Local prices from a nodal run (market_optimization.active = false)
        if given, otherwise the redispatch shadow prices of the status quo."""
        ref = self.ref
        if self.nodal_dir is not None:
            d = self.nodal_dir
            stage_dir = d / "grid_optimization" if (d / "grid_optimization").exists() else d
            st = Stage(stage_dir)
            focus_file = d / "busmap_within_focus.csv"
            self.focus = (set(pd.read_csv(focus_file)["cluster"].astype(str))
                          if focus_file.exists() else set())
            self.source = "nodal"
        else:
            st = ref.grid
            self.focus = set(ref.focus_buses)
            self.source = "redispatch"
        self.lstage = st
        b = st.static("buses")
        de = b[(b.carrier == "AC") & (b.country == "DE")]
        gp = st.series("buses", "marginal_price")
        self.nodes = de.loc[[i for i in de.index if i in gp]]
        local = gp[self.nodes.index].clip(*PRICE_CLIP)
        # common snapshots of the local-price run and the market runs
        common = local.index.intersection(ref.prices.index)
        self.local = local.loc[common]
        self.w = st.weights.reindex(common)
        loads = st.static("loads")
        acl = loads[(loads.carrier == "AC") & loads.bus.isin(self.nodes.index)]
        lp = st.series("loads", "p").reindex(columns=acl.index, fill_value=0).loc[common]
        self.node_load = lp.T.groupby(acl.bus).sum().T.reindex(
            columns=self.nodes.index, fill_value=0.0)
        self.lw = self.node_load.mul(self.w, axis=0)
        # Zone of every node per configuration (point in polygon, as eTraGo)
        self.node_zone, self.zonal = {}, {}
        for k, m in self.M.items():
            zone = geo.assign_zones(self.nodes, m.zone_gdf, k)
            self.node_zone[k] = zone
            self.zonal[k] = pd.DataFrame(
                {n: m.prices[z].reindex(common).values for n, z in zone.items()}, index=common)
        # Load-weighted annual means (time-weighted where a node has no load)
        def node_mean(df):
            lw = self.lw.sum()
            out = (df * self.lw).sum() / lw.replace(0, np.nan)
            tw = df.mul(self.w, axis=0).sum() / self.w.sum()
            return out.fillna(tw)
        self.local_mean = node_mean(self.local)
        self.zonal_mean = {k: node_mean(z) for k, z in self.zonal.items()}
        # Redispatch cost per MWh of German demand (all redispatch units:
        # in the model most up-regulation happens abroad).
        de_mwh = self.lw.sum().sum() * len(ref.grid.snapshots) / max(len(common), 1)
        self.rd_per_mwh = {k: m.rd_total["cost_meur"] * 1e6 / de_mwh for k, m in self.M.items()}
        # Regions of figure A
        sh = self.nodes[self.nodes.index.isin(self.focus)]
        north = (self.local_mean.reindex(sh.index).idxmin() if len(sh)
                 else self.local_mean[self.nodes.y > 53.5].idxmin())
        self.regions = {"Nord-Ostsee (SH)": north}
        for name, (x, y) in REGIONS.items():
            dist = np.hypot(self.nodes.x - x, (self.nodes.y - y) * 1.6)
            self.regions[name] = dist.idxmin()
        self.rd_up = ref.rd_ts["up"].reindex(common).fillna(0)

    # ------------------------------------------------------------------
    def build(self):
        self.summary()
        self.fig_a()
        self.fig_b()
        self.fig_c()
        self.fig_d()
        self.fig_e()
        return H.page(
            title="SPREAD.SH local prices",
            heading="Local prices: Agora/Fraunhofer IEE figures with eTraGo",
            intro=("The figures of <em>Local electricity prices in Germany</em> (Agora Energiewende &amp; "
                   "Fraunhofer IEE, 2025) rebuilt from the eTraGo full-year runs. Agora looks back at "
                   "2019–2023 with real market data; these runs model 2035 (eGon2035). Absolute numbers "
                   "therefore differ, but the mechanisms can be compared one-to-one."),
            chips=["single zone = status quo", "zones = DE2 … DE5",
                   ("local prices = nodal eTraGo run" if self.source == "nodal" else "local prices ≈ redispatch shadow prices (proxy)"),
                   f"{len(self.nodes)} German nodes", "eGon2035 · 8760 h"],
            nav=self.nav, toc=self.toc, sections_html="".join(self.sections), figs=self.figs,
            plotly_js=self.plotly_js, footer="Generated by spread_sh_reports.",
        )

    # ------------------------------------------------------------------
    def _convergence(self, k):
        """Share of hours in which the local prices of all nodes of a zone lie
        within ±1 €/MWh (max − min ≤ 2), per zone of configuration k."""
        m = self.M[k]
        zone = self.node_zone[k]
        out = {}
        for z in m.de_zones:
            cols = zone.index[zone == z]
            if len(cols) < 2:
                continue
            p = self.local[cols]
            out[z] = _wmean((p.max(axis=1) - p.min(axis=1)) <= 2.0, self.w)
        return pd.Series(out)

    def _region_gap(self, k, node, mask=None):
        d = self.zonal[k][node] - self.local[node]
        w = self.w if mask is None else self.w * mask
        return _wmean(d, w)

    def summary(self):
        ref = self.ref
        conv = {k: self._convergence(k) for k in self.M}
        dev = (self.local_mean / self.zonal_mean.get("status_quo", self.local_mean) - 1) * 100
        sh = dev.reindex(list(self.focus)).dropna()
        south = dev[self.nodes.y < 49.8]
        mask = (self.rd_up > CONGESTED_MW).astype(float)
        sq = "status_quo" if "status_quo" in self.M else next(iter(self.M))
        north, neckar = self.regions["Nord-Ostsee (SH)"], self.regions["Neckar (Mündung)"]
        rows = [
            ("Single zone fits the local prices (hours within ±1 €/MWh)", AGORA["conv_single"],
             H.pct(conv[sq].iloc[0], 0) if len(conv[sq]) else "–"),
            ("Same per zone of the split configurations", AGORA["conv_zones"],
             " · ".join(f"{k}: {H.pct(c.min(), 0)}–{H.pct(c.max(), 0)}"
                        for k, c in conv.items() if k != sq and len(c))),
            ("Local price at coastal hubs vs single zone", AGORA["coast_dev"],
             f"{sh.min():+.0f} … {sh.max():+.0f} % (SH nodes)"),
            ("Local price in the south vs single zone", AGORA["south_dev"],
             f"{south.min():+.0f} … {south.max():+.0f} % (nodes south of 49.8° N)"),
            ("Redispatch cost per MWh German demand [€/MWh]", AGORA["rd_cost"],
             " · ".join(f"{'single' if k == 'status_quo' else k}: {v:.2f}" for k, v in self.rd_per_mwh.items())
             + " · local: 0"),
            ("Price + redispatch cost [€/MWh]", AGORA["price_rd"],
             f"{_wmean(self.zonal_mean[sq], self.lw.sum()) + self.rd_per_mwh[sq]:.1f} (single) · "
             f"{_wmean(self.local_mean, self.lw.sum()):.1f} (local, load-weighted) · "
             f"{self.local_mean.min():.1f}–{self.local_mean.max():.1f} range"),
            ("Nord-Ostsee: local minus single-zone price [€/MWh]", AGORA["north_gap"],
             f"{-self._region_gap(sq, north):+.1f} (all hours) · {-self._region_gap(sq, north, mask):+.1f} (> 3 GW)"),
            ("Neckar: local minus single-zone price [€/MWh]", AGORA["south_gap"],
             f"{-self._region_gap(sq, neckar):+.1f} (all hours) · {-self._region_gap(sq, neckar, mask):+.1f} (> 3 GW)"),
            ("Revenues of wind & solar in Germany", AGORA["res_rev"], self._res_summary()),
        ]
        head = "<tr><th>Indicator</th><th>Agora (2019–2023)</th><th>eTraGo 2035 (this study)</th></tr>"
        body = "".join(f"<tr><th scope='row'>{H.esc(a)}</th><td>{H.esc(b)}</td><td>{H.esc(c)}</td></tr>"
                       for a, b, c in rows)
        table = (f"<div class='card wide'><div class='ft'>Key numbers side by side</div>"
                 f"<div class='tscroll'><table class='tbl'><thead>{head}</thead><tbody>{body}"
                 f"</tbody></table></div></div>")
        note = H.callout(
            "<strong>How to read the comparison.</strong><ul>"
            "<li>Agora models real years 2019–2023 (gas-crisis prices, about 100 €/MWh); eTraGo models 2035 "
            "with far more wind and solar (prices about 20 €/MWh). Compare relative effects and patterns, "
            "not absolute levels.</li>"
            + (
                "<li>Local prices come from a <strong>nodal eTraGo run</strong> (market optimisation "
                "switched off, dispatch optimised directly on the grid), like Agora's local-price simulation "
                "with 22 hubs; here at every German node.</li>"
                if self.source == "nodal" else
                "<li><strong>Caution – no nodal run available.</strong> Local prices are approximated by the "
                "shadow prices of eTraGo's <em>redispatch</em> optimisation of the status-quo run. There, "
                "down-regulation is compensated at the zonal price, so northern node prices stay close to the "
                "zonal price and the overall level is about 2–3 €/MWh higher. The spatial pattern (south and "
                "west expensive, convergence per zone) is meaningful; the <em>northern discount</em> that Agora "
                "finds cannot appear with this proxy. Rebuild with "
                "<code>--nodal &lt;results of a run with market_optimization.active = false&gt;</code> for a "
                "true local-price benchmark.</li>")
            + "<li>Load-shedding price spikes beyond −200/+500 €/MWh (0.01 % of values) are clipped.</li></ul>")
        self.add("summary", "Agora vs eTraGo at a glance",
                 "The headline statements of the Agora summary next to the same indicator from the runs.",
                 note + H.grid(table))

    def _res_summary(self):
        rev = self._res_revenues()
        return " / ".join(f"{v.sum():.1f}" for v in rev.values()) + " bn€ (" + \
            " / ".join(n.split(" ")[0] for n in rev) + ")"

    # ------------------------------------------------------------------
    def fig_a(self):
        sq = "status_quo" if "status_quo" in self.M else next(iter(self.M))
        k3 = "DE3" if "DE3" in self.M else (self.zonal_keys[0] if self.zonal_keys else sq)
        daily = self.rd_up.groupby(self.rd_up.index.date).mean()
        day = pd.Timestamp(daily.idxmax())
        sel = (self.local.index >= day - pd.Timedelta(days=1)) & (self.local.index < day + pd.Timedelta(days=2))
        idx = self.local.index[sel]
        x = [t.strftime("%d.%m %H:%M") for t in idx]
        blocks = []
        for name, node in self.regions.items():
            series = [
                {"name": "Single zone", "values": (self.zonal[sq].loc[idx, node] - self.local.loc[idx, node]).values,
                 "color": SCENARIO_COLOR[sq], "width": 2.5},
                {"name": SCENARIO_LABELS.get(k3, k3).split(" ")[0] + " zones" if k3 != sq else "",
                 "values": (self.zonal[k3].loc[idx, node] - self.local.loc[idx, node]).values,
                 "color": SCENARIO_COLOR.get(k3, "@s3")},
            ]
            spec = C.lines(x, series, unit="€/MWh vs local price", height=300, digits=1)
            spec["layout"]["shapes"] = [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 0, "y1": 0,
                                         "line": {"color": LOCAL_COLOR, "width": 2}}]
            spec["layout"]["annotations"] = [
                {"xref": "paper", "x": 1, "y": 0, "text": "local price", "showarrow": False,
                 "xanchor": "right", "yanchor": "bottom", "font": {"color": LOCAL_COLOR, "size": 11}},
                {"xref": "paper", "yref": "paper", "x": 0, "y": 1, "text": "excessive consumption incentive ↑",
                 "showarrow": False, "xanchor": "left", "yanchor": "top", "font": {"color": "@muted", "size": 10}},
                {"xref": "paper", "yref": "paper", "x": 0, "y": 0, "text": "insufficient consumption incentive ↓",
                 "showarrow": False, "xanchor": "left", "yanchor": "bottom", "font": {"color": "@muted", "size": 10}},
            ]
            fid = self.fig(f"lp-a-{name[:5]}", spec)
            blocks.append(H.fig_block(fid, f"{name} (node {node})"))
        rd = {k: self.M[k].rd_ts["up"].reindex(idx).fillna(0).values / 1e3 for k in (sq, k3)}
        spec = C.lines(x, [{"name": SCENARIO_LABELS.get(k, k), "values": v, "color": SCENARIO_COLOR.get(k)}
                           for k, v in rd.items()], unit="GW up-regulation in DE", height=300, digits=2)
        spec["data"].append({"type": "scatter", "mode": "lines", "x": x, "y": [0] * len(x), "name": LOCAL,
                             "line": {"color": LOCAL_COLOR, "width": 2}, "hoverinfo": "skip"})
        blocks.append(H.fig_block(self.fig("lp-a-rd", spec), "Redispatch in the same days",
                                  "Local prices include grid constraints in the price, so (ideally) no redispatch is needed."))
        mask = (self.rd_up > CONGESTED_MW).astype(float)
        rows = []
        for name, node in self.regions.items():
            for k in (sq, k3):
                rows.append({"Region": name, "Market": SCENARIO_LABELS.get(k, k),
                             "Gap, all hours [€/MWh]": self._region_gap(k, node),
                             "Gap, > 3 GW redispatch [€/MWh]": self._region_gap(k, node, mask),
                             "Local price [€/MWh]": float(self.local_mean[node])})
        tbl = pd.DataFrame(rows).set_index("Region")
        blocks.append(H.table_block(tbl, "Annual price gap (zonal minus local price)",
                                    note=f"Positive = zonal price too high (insufficient incentive to consume). "
                                         f"Congested hours: status-quo up-regulation above 3 GW "
                                         f"({H.pct(_wmean(mask, self.w), 0)} of the year)."))
        self.add("fig-a", "Fig. A · Wrong incentives of large zones",
                 f"Hourly difference between zonal prices and the local price in three model regions, around the "
                 f"day with the most redispatch ({day:%d.%m.}). Above zero the zonal price overstates the value "
                 f"of consuming there, below zero it understates it (Agora: Nord-Ostsee −50 €/MWh on 1 Jan 2023).",
                 H.grid(*blocks))

    # ------------------------------------------------------------------
    def fig_b(self):
        keys = list(self.M)
        lw = self.lw.sum()
        data = []
        names = []
        for k in keys:
            v = (self.zonal_mean[k] + self.rd_per_mwh[k]).values
            names.append(SCENARIO_LABELS.get(k, k))
            data.append({"type": "box", "x": C._r(v, 2), "name": names[-1], "orientation": "h",
                         "boxpoints": "all", "jitter": 0.4, "pointpos": 0,
                         "marker": {"color": SCENARIO_COLOR.get(k, "@muted"), "size": 5},
                         "line": {"color": SCENARIO_COLOR.get(k, "@muted"), "width": 1.5},
                         "fillcolor": "rgba(0,0,0,0)",
                         "text": [f"node {b}" for b in self.nodes.index],
                         "hovertemplate": "%{text}: %{x:.2f} €/MWh<extra></extra>"})
        data.append({"type": "box", "x": C._r(self.local_mean.values, 2), "name": LOCAL, "orientation": "h",
                     "boxpoints": "all", "jitter": 0.4, "pointpos": 0,
                     "marker": {"color": LOCAL_COLOR, "size": 5}, "line": {"color": LOCAL_COLOR, "width": 1.5},
                     "fillcolor": "rgba(0,0,0,0)", "text": [f"node {b}" for b in self.nodes.index],
                     "hovertemplate": "%{text}: %{x:.2f} €/MWh<extra></extra>"})
        spec = C.figure(data, {"xaxis": {"title": {"text": "load-weighted price + redispatch cost [€/MWh]"}},
                               "yaxis": {"autorange": "reversed", "automargin": True},
                               "showlegend": False, "hovermode": "closest"}, height=380)
        f_box = self.fig("lp-b-box", spec)
        # Map: local price deviation from the single zone
        sq = "status_quo" if "status_quo" in self.M else keys[0]
        dev = (self.local_mean / self.zonal_mean[sq] - 1) * 100
        vabs = max(5.0, float(dev.abs().quantile(0.95)))
        sizes, _ = C.bubble_sizes(lw.reindex(self.nodes.index).values, smax=26, smin=6)
        mp = base_map(self.ref, geo.MAP_EXTENT["de"], height=560)
        mp.points(self.nodes.x, self.nodes.y, [C.div(v, vabs) for v in dev], size=sizes,
                  hover=[f"node {b}<br>local {self.local_mean[b]:.1f} €/MWh<br>{d:+.1f} % vs single zone"
                         for b, d in dev.items()])
        mp.colorbar(-vabs, vabs, "local vs single zone [%]", "@divscale")
        f_map = self.fig("lp-b-map", mp.spec())
        blocks = [H.fig_block(f_box, "Price plus redispatch cost per node",
                              "One dot per German node: load-weighted wholesale price plus the configuration's "
                              "German redispatch cost per MWh of demand. Local prices carry no redispatch cost."),
                  H.fig_block(f_map, "Local price: deviation from the single price zone",
                              "Bubble size = electricity demand at the node. Blue = cheaper, red = more expensive "
                              "than the single zone.", table=False)]
        for k in self.zonal_keys:
            m = self.M[k]
            single = _wmean(self.zonal_mean[sq], self.lw.sum())
            zd = {}
            for z in m.de_zones:
                cols = self.node_zone[k].index[self.node_zone[k] == z]
                if len(cols):
                    zd[z] = (_wmean(self.zonal_mean[k][cols], self.lw.sum()[cols]) / single - 1) * 100
            mp = base_map(m, geo.MAP_EXTENT["de"], zones_fill={z: C.div(v, 5.0) for z, v in zd.items()}, height=360)
            an = [zone_anchor(m, z) for z in zd]
            mp.labels([a[0] for a in an], [a[1] for a in an],
                      [f"<b>{m.label(z).split(' ')[0]}</b><br>{v:+.1f} %" for z, v in zd.items()], size=11)
            spec = mp.spec()
            spec["layout"]["showlegend"] = False
            blocks.append(H.fig_block(self.fig(f"lp-b-zone-{k}", spec), f"{SCENARIO_LABELS.get(k, k)}: zone price vs single zone",
                                      "Load-weighted zonal price compared with the single-zone price (colour scale ±5 %).",
                                      table=False))
        self.add("fig-b", "Fig. B · Price plus redispatch cost",
                 "Agora: the sum of electricity price and redispatch cost is lower with local prices for 18 of 22 hubs; "
                 "three zones narrow the gap only partly.", H.grid(*blocks))

    # ------------------------------------------------------------------
    def _res_revenues(self):
        if hasattr(self, "_rev"):
            return self._rev
        out = {}
        groups = {"wind_onshore": "Wind onshore", "wind_offshore": "Wind offshore",
                  "solar": "Solar", "solar_rooftop": "Solar"}
        for k, m in self.M.items():
            mk = m.mkt
            mg = mk.static("generators")
            gg = m.grid.static("generators")
            sel = mg.index[mg.carrier.isin(VRES_CARRIERS)].intersection(gg.index)
            german = gg.loc[sel, "bus"].map(m.bus_zone).map(geo.is_german_zone).fillna(False).astype(bool)
            sel = sel[german.values]
            p = mk.series("generators", "p").reindex(columns=sel, fill_value=0)
            price = pd.DataFrame({g: m.prices[m.mkt_zone[mg.at[g, "bus"]]].values for g in sel}, index=p.index)
            rev = (p * price).mul(mk.weights, axis=0).sum()
            s = rev.groupby(mg.loc[sel, "carrier"].map(groups)).sum() / 1e9
            u = m.rd_units
            comp = u[u.german & (u.direction == "down") & u.carrier.isin(VRES_CARRIERS)].cost.sum() / 1e9
            s["Redispatch compensation"] = max(comp, 0.0)
            out[SCENARIO_LABELS.get(k, k)] = s
        # Local prices: final infeed (incl. any redispatch) times local price
        st = self.lstage
        gg = st.static("generators")
        sel = gg.index[gg.carrier.isin(VRES_CARRIERS) & gg.bus.isin(self.nodes.index)]
        gp = st.series("generators", "p").loc[self.local.index]
        final = gp.reindex(columns=sel, fill_value=0)
        for suf in (" ramp_up", " ramp_down"):
            final = final + gp.reindex(columns=[c + suf for c in sel], fill_value=0).values
        price = pd.DataFrame({g: self.local[gg.at[g, "bus"]].values for g in sel}, index=final.index)
        rev = (final * price).mul(self.w, axis=0).sum() * len(st.snapshots) / len(self.local.index)
        s = rev.groupby(gg.loc[sel, "carrier"].map(groups)).sum() / 1e9
        s["Redispatch compensation"] = 0.0
        out[LOCAL] = s
        self._rev = out
        return out

    def fig_c(self):
        rev = self._res_revenues()
        cats = list(rev)
        order = [("Solar", "@s4"), ("Wind onshore", "@s1"), ("Wind offshore", "@s7"), ("Redispatch compensation", "@s5")]
        series = [{"name": n, "values": [rev[c].get(n, 0.0) for c in cats], "color": col} for n, col in order]
        spec = C.bars(cats, series, unit="bn€/a", stacked=True, height=380)
        tot = [rev[c].sum() for c in cats]
        spec["data"].append({"type": "scatter", "mode": "text", "x": cats, "y": C._r(tot, 2),
                             "text": [f"{t:.1f}" for t in tot], "textposition": "top center",
                             "textfont": {"color": "@ink", "size": 12}, "showlegend": False, "hoverinfo": "skip"})
        f = self.fig("lp-c", spec)
        self.add("fig-c", "Fig. C · Revenues of wind and solar",
                 "Market revenues of German wind and solar plus redispatch compensation for curtailed renewables. "
                 "Agora 2023: 17.4 (single zone) → 16.3 (three zones) → 13.5 bn€ (local prices), with offshore "
                 "wind losing most.",
                 H.grid(H.fig_block(f, "Wind and solar revenues per market design",
                                    "Zonal configurations: market dispatch × zonal price. Local prices: final infeed after "
                                    "redispatch × nodal price. Compensation = cost of renewable down-regulation.",
                                    wide=True)))

    # ------------------------------------------------------------------
    def fig_d(self):
        blocks = []
        rows = {}
        for k, m in self.M.items():
            conv = self._convergence(k)
            if conv.empty:
                continue
            rows[SCENARIO_LABELS.get(k, k)] = {m.label(z): v for z, v in conv.items()}
            mp = base_map(m, geo.MAP_EXTENT["de"], zones_fill={z: C.seq(v, 0, 1) for z, v in conv.items()},
                          height=360)
            an = [zone_anchor(m, z) for z in conv.index]
            mp.labels([a[0] for a in an], [a[1] for a in an],
                      [f"<b>{m.label(z).split(' ')[0]}</b><br>{v * 100:.0f} %" for z, v in conv.items()], size=12)
            spec = mp.spec()
            spec["layout"]["showlegend"] = False
            blocks.append(H.fig_block(self.fig(f"lp-d-{k}", spec), SCENARIO_LABELS.get(k, k),
                                      "Colour/label = share of hours in which all node prices of the zone agree "
                                      "within ±1 €/MWh.", table=False))
        tbl = pd.DataFrame(rows).T
        tbl.index.name = "Configuration"
        blocks.append(H.table_block(tbl, "Price convergence per zone",
                                    formats={c: (lambda v: H.pct(v, 0)) for c in tbl.columns}))
        self.add("fig-d", "Fig. D · Price convergence within zones",
                 "Agora: a zone is a good market area only if its local prices coincide; with one zone this was the "
                 "case in 36 % (2019) and 17 % (2023) of hours, the north-east never exceeded 35 %. Evaluated here "
                 "with the local (nodal) prices of the status-quo run for every zone layout.", H.grid(*blocks))

    # ------------------------------------------------------------------
    def fig_e(self):
        w = self.w
        flh = (self.local < CHEAP).mul(w, axis=0).sum()
        proc = {}
        for b in self.nodes.index:
            order = np.argsort(self.local[b].values)
            cw = np.cumsum(w.values[order])
            n = int(np.searchsorted(cw, ELY_HOURS)) + 1
            proc[b] = _wmean(self.local[b].values[order][:n], w.values[order][:n])
        proc = pd.Series(proc)
        sizes, vmax = C.bubble_sizes(flh.values, smax=32, smin=4)
        mp = base_map(self.ref, geo.MAP_EXTENT["de"], height=560)
        mp.points(self.nodes.x, self.nodes.y, "@s1", size=sizes, opacity=0.85,
                  hover=[f"node {b}<br>{h:,.0f} h below {CHEAP:.0f} €/MWh" for b, h in flh.items()])
        mp.size_legend([vmax, vmax / 4], [32, 4 + 28 * 0.5], "h")
        f1 = self.fig("lp-e-flh", mp.spec())
        lo, hi = float(proc.quantile(0.02)), float(proc.quantile(0.98))
        mp = base_map(self.ref, geo.MAP_EXTENT["de"], height=560)
        mp.points(self.nodes.x, self.nodes.y, [C.seq(v, lo, hi) for v in proc], size=14,
                  hover=[f"node {b}<br>{v:.1f} €/MWh for the cheapest {ELY_HOURS:,.0f} h" for b, v in proc.items()])
        mp.colorbar(lo, hi, "procurement price [€/MWh]")
        f2 = self.fig("lp-e-price", mp.spec())
        sh = flh.reindex(list(self.focus)).dropna()
        south = flh[self.nodes.y < 49.8]
        rows = []
        for k, m in self.M.items():
            for z in m.de_zones:
                p = m.prices[z]
                order = np.argsort(p.values)
                cw = np.cumsum(w.values[order])
                n = int(np.searchsorted(cw, ELY_HOURS)) + 1
                rows.append({"Configuration": SCENARIO_LABELS.get(k, k), "Zone": m.label(z),
                             f"Hours < {CHEAP:.0f} €/MWh": float(((p < CHEAP) * w).sum()),
                             f"Price, cheapest {ELY_HOURS:,.0f} h [€/MWh]": _wmean(p.values[order][:n], w.values[order][:n])})
        tbl = pd.DataFrame(rows).set_index("Configuration")
        callout = H.callout(
            f"With local prices, SH nodes reach {sh.mean():,.0f} h below {CHEAP:.0f} €/MWh on average, nodes in the "
            f"south {south.mean():,.0f} h (ratio {sh.mean() / south.mean():.1f}×; Agora 2023: up to 10×). "
            f"The cheapest {ELY_HOURS:,.0f} h cost {proc.min():.1f}–{proc.max():.1f} €/MWh across nodes.")
        self.add("fig-e", "Fig. E · Electrolysis: full-load hours and procurement price",
                 f"Agora: electrolysers running only below {CHEAP:.0f} €/MWh reach far more hours at the coast; for "
                 f"{ELY_HOURS:,.0f} h of operation the coastal procurement price was even negative.",
                 callout + H.grid(
                     H.fig_block(f1, f"Hours with a local price below {CHEAP:.0f} €/MWh", table=False),
                     H.fig_block(f2, f"Average price of the cheapest {ELY_HOURS:,.0f} hours", table=False),
                     H.table_block(tbl, "Same indicators with zonal prices", formats={
                         f"Hours < {CHEAP:.0f} €/MWh": lambda v: H.fmt(v, 0)})))
