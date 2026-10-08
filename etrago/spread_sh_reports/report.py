"""Turn ScenarioMetrics into report pages."""

import numpy as np
import pandas as pd

from . import charts as C
from . import geo
from . import html as H
from .config import (
    CURTAIL_GROUP_ORDER, FLEX_COLOR, FLEX_ORDER, GEN_GROUP_COLOR,
    GEN_GROUP_ORDER, SCENARIO_COLOR, SCENARIO_LABELS,
)

STAGE_COLOR = {"market": "@s1", "grid": "@s2"}
UP, DOWN = "@div:0.86", "@div:0.14"
MONTHS = C.MONTHS


def zone_color(m, zone):
    if zone in m.de_zones:
        return f"@s{m.de_zones.index(zone) + 1}"
    return "@muted"


def zone_anchor(m, zone):
    gdf = m.zone_gdf
    hit = gdf[gdf.zone == zone]
    if len(hit):
        p = hit.geometry.iloc[0].representative_point()
        return p.x, p.y
    mb = m.mkt.static("buses")
    b = m.mkt_zone.index[m.mkt_zone == zone]
    fb = (float(mb.loc[b[0], "x"]), float(mb.loc[b[0], "y"])) if len(b) else (np.nan, np.nan)
    return geo.country_anchor(m.data_dir, zone, fb)


def base_map(m, extent, zones_fill=None, outline_zones=True, focus=True, height=560,
             bg_extent=None):
    mp = C.MapFigure(extent, height=height)
    mp.background(geo.background(m.data_dir, bg_extent or extent))
    if zones_fill is not None or outline_zones:
        for _, row in m.zone_gdf.iterrows():
            fill = zones_fill.get(row.zone, "rgba(0,0,0,0)") if zones_fill else "rgba(0,0,0,0)"
            mp.polygon(row.geometry, fill, line="@ink2", width=1.0)
    if focus:
        mp.outline(geo.focus_outline(m.data_dir), color="@ink", width=1.4, dash="dot")
    return mp


def _delta(all_m, m, fn):
    sq = all_m.get("status_quo")
    if sq is None or sq is m:
        return None
    try:
        return fn(m) - fn(sq)
    except Exception:
        return None


# ===========================================================================
# Scenario report
# ===========================================================================
class ScenarioReport:
    def __init__(self, m, all_metrics, nav, plotly_js=None):
        self.m = m
        self.all = all_metrics
        self.nav = nav
        self.plotly_js = plotly_js
        self.figs = {}
        self.sections = []
        self.toc = []

    def fig(self, fid, spec):
        self.figs[fid] = spec
        return fid

    def add(self, sid, title, lead, body):
        self.toc.append((sid, title))
        self.sections.append(H.section(sid, title, lead, body))

    # ------------------------------------------------------------------
    def build(self):
        self.overview()
        self.zones()
        self.prices()
        self.exchange()
        self.redispatch()
        self.curtailment()
        self.flexibility()
        self.batteries()
        self.grid()
        self.costs()
        self.focus()
        self.notes()
        m = self.m
        k = m.kpis()
        chips = [
            f"{len(m.de_zones)} German bidding zone{'s' if len(m.de_zones) > 1 else ''}",
            f"{int(m.args['network_clustering']['electricity_grid']['n_clusters'])} AC nodes "
            f"({int(m.args['network_clustering']['electricity_grid'].get('n_clusters_focus') or 0)} in SH)",
            f"{len(m.grid.snapshots)} snapshots × {int(m.grid.weights.iloc[0])} h",
            f"scenario {m.args['scn_name']}",
            m.sc.run_name,
        ]
        intro = (
            "eTraGo results for the SPREAD.SH bidding-zone study: market dispatch on the "
            "bidding zones, then grid-constrained redispatch and grid expansion on the "
            "clustered transmission grid. "
            f"German average wholesale price {H.fmt(k['de_avg_price'], 2, '€/MWh')}, "
            f"redispatch {H.fmt(k['rd_up_de_twh'] + k['rd_down_de_twh'], 1, 'TWh')} in Germany."
        )
        return H.page(
            title=f"SPREAD.SH {m.sc.key} report",
            heading=m.sc.label,
            intro=intro, chips=chips, nav=self.nav, toc=self.toc,
            sections_html="".join(self.sections), figs=self.figs,
            plotly_js=self.plotly_js,
            footer=f"Generated from {H.esc(m.sc.path)} by spread_sh_reports.",
        )

    # ------------------------------------------------------------------
    def overview(self):
        m, A = self.m, self.all
        k = m.kpis()
        d = lambda key: _delta(A, m, lambda x: x.kpis()[key])
        tiles = [
            H.kpi("Average German wholesale price", H.fmt(k["de_avg_price"], 2, "€/MWh"),
                  "load-weighted over German zones", d("de_avg_price"), "lower", "€/MWh"),
            H.kpi("Hours with one German price", H.pct(k["price_convergence"], 1),
                  "spread between DE zones < 1 €/MWh", None),
            H.kpi("Redispatch in Germany", H.fmt(k["rd_up_de_twh"] + k["rd_down_de_twh"], 1, "TWh"),
                  f"up {H.fmt(k['rd_up_de_twh'], 1)} · down {H.fmt(k['rd_down_de_twh'], 1)} TWh",
                  _delta(A, m, lambda x: x.kpis()["rd_up_de_twh"] + x.kpis()["rd_down_de_twh"]),
                  "lower", "TWh"),
            H.kpi("Redispatch cost (all units)", H.fmt(k["rd_cost_meur"], 0, "M€"), None,
                  d("rd_cost_meur"), "lower", "M€"),
            H.kpi("RES curtailment after redispatch", H.fmt(k["curt_grid_de_twh"], 1, "TWh"),
                  f"{H.pct(k['curt_grid_de_rate'])} of available wind & solar in DE",
                  d("curt_grid_de_twh"), "lower", "TWh"),
            H.kpi("AC grid expansion in DE", H.fmt(k["ac_exp_gwkm_de"], 0, "GW·km"),
                  f"{H.fmt(k['ac_exp_gw_de'], 1, 'GW')} additional capacity",
                  d("ac_exp_gwkm_de"), "lower", "GW·km"),
            H.kpi("Electrolysers in DE", H.fmt(k["ely_gw_de"], 1, "GW"),
                  f"{H.fmt(m.focus['ely_gw'], 2, 'GW')} in Schleswig-Holstein",
                  d("ely_gw_de"), "neutral", "GW"),
            H.kpi("System cost excl. gas imports", H.fmt(k["cost_excl_gas_beur"], 2, "bn€/a"),
                  f"incl. natural-gas imports {H.fmt(k['total_cost_beur'], 2, 'bn€/a')}",
                  d("cost_excl_gas_beur"), "lower", "bn€"),
        ]
        ps = m.price_stats.loc[m.de_zones]
        cheapest = ps["mean"].idxmin()
        dearest = ps["mean"].idxmax()
        rdz = m.rd_by_zone.drop(index="Abroad", errors="ignore")
        rd_vol = (rdz.get("up", 0) - rdz.get("down", 0))
        top_rd = rd_vol.idxmax() if len(rd_vol) else None
        bullets = []
        if len(m.de_zones) > 1:
            bullets.append(
                f"Zonal prices are close: the cheapest German zone ({H.esc(m.label(cheapest))}, "
                f"{H.fmt(ps.loc[cheapest, 'mean'], 2, '€/MWh')}) and the most expensive "
                f"({H.esc(m.label(dearest))}, {H.fmt(ps.loc[dearest, 'mean'], 2, '€/MWh')}) differ by "
                f"{H.fmt(k['max_zone_price_gap'], 2, '€/MWh')} on average. The zones share one price in "
                f"{H.pct(k['price_convergence'])} of hours.")
        else:
            bullets.append("Germany and Luxembourg form a single bidding zone (reference case).")
        if top_rd is not None:
            bullets.append(
                f"Most German redispatch is needed in {H.esc(m.label(top_rd))} "
                f"({H.fmt(rd_vol[top_rd], 1, 'TWh')} up + down). Schleswig-Holstein accounts for "
                f"{H.fmt(m.focus['rd_down_twh'], 1, 'TWh')} of down-regulation.")
        bullets.append(
            f"Wind and solar curtailment in Germany rises from {H.fmt(k['curt_market_de_twh'], 1, 'TWh')} "
            f"in the market to {H.fmt(k['curt_grid_de_twh'], 1, 'TWh')} after redispatch.")
        z_ely = m.ely_by_zone.gw
        if len(z_ely):
            bullets.append(
                f"The pre-market investment places {H.fmt(z_ely.sum(), 1, 'GW')} of electrolysis in Germany, "
                f"most of it in {H.esc(m.label(z_ely.idxmax()))} ({H.fmt(z_ely.max(), 1, 'GW')}).")
        body = H.kpi_grid(tiles) + H.callout("<strong>Key observations</strong><ul>" +
                                             "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
        if m.notes:
            body += H.callout("<strong>Data notes</strong><ul>" +
                              "".join(f"<li>{H.esc(n)}</li>" for n in m.notes) + "</ul>", "warn")
        self.add("overview", "Overview",
                 "Headline indicators for Germany. The arrows show the change against the "
                 "status-quo DE/LU zone where that run is available. Green marks a change "
                 "in the preferable direction.", body)

    # ------------------------------------------------------------------
    def zones(self):
        m = self.m
        fills = {z: f"@fade:s{i + 1}" for i, z in enumerate(m.de_zones)}
        mp = base_map(m, geo.MAP_EXTENT["de"], zones_fill=fills)
        ac = m.ac
        for z in m.de_zones:
            sub = ac[ac.zone == z]
            hover = [f"Bus {b}<br>{H.esc(m.label(z))}{' · SH focus' if f else ''}"
                     for b, f in zip(sub.index, sub.focus)]
            mp.points(sub.x, sub.y, zone_color(m, z), size=7, hover=hover,
                      name=m.label(z), legend=True)
        foreign = ac[~ac.zone.isin(m.de_zones)]
        mp.points(foreign.x, foreign.y, "@muted", size=6,
                  hover=[f"Bus {b} · {z}" for b, z in zip(foreign.index, foreign.zone)],
                  name="Neighbouring zones", legend=True)
        ax, ay, at = [], [], []
        for z in m.de_zones:
            x, y = zone_anchor(m, z)
            ax.append(x)
            ay.append(y)
            at.append(f"<b>{m.label(z).split(' ')[0]}</b>")
        mp.labels(ax, ay, at, size=13)
        f1 = self.fig("zones-map", mp.spec())

        bal = m.balance.loc[m.de_zones]
        gen = m.generation.reindex(index=m.de_zones, columns=GEN_GROUP_ORDER, fill_value=0)
        cats = [m.label(z) for z in m.de_zones]
        series = [{"name": g, "values": gen[g].values, "color": GEN_GROUP_COLOR[g]}
                  for g in GEN_GROUP_ORDER if gen[g].sum() > 0.01]
        spec = C.bars(cats, series, unit="TWh", stacked=True, height=380)
        spec["data"].append({
            "type": "scatter", "mode": "markers", "name": "Electricity demand",
            "x": cats, "y": C._r(bal.demand_twh.values, 1),
            "marker": {"symbol": "line-ew-open", "size": 34, "color": "@ink",
                       "line": {"width": 3, "color": "@ink"}},
            "hovertemplate": "%{x}<br>Demand: %{y:,.1f} TWh<extra></extra>",
        })
        spec["layout"]["showlegend"] = True
        f2 = self.fig("zones-balance", spec)
        tbl = pd.DataFrame({
            "Zone": [m.label(z) for z in m.de_zones],
            "AC nodes": [int((ac.zone == z).sum()) for z in m.de_zones],
            "Demand [TWh]": bal.demand_twh.values,
            "Generation [TWh]": bal.generation_twh.values,
            "Wind+solar / demand": bal.vres_share.values,
            "Net export [TWh]": bal.net_export_twh.values,
            "Mean price [€/MWh]": m.price_stats.loc[m.de_zones, "mean"].values,
        }).set_index("Zone")
        body = H.grid(
            H.fig_block(f1, "Bidding-zone layout and clustered AC nodes",
                        "Fill = German bidding zone, dots = AC nodes of the grid model coloured by zone, "
                        "dotted outline = Schleswig-Holstein focus region. Scroll to zoom, drag to pan.",
                        table=False),
            H.fig_block(f2, "Zonal electricity balance (market)",
                        "Annual generation by technology in each German zone; the black bar marks the zone's "
                        "electricity demand. Zones above the bar export, zones below import."),
            H.table_block(tbl, "Zone summary", formats={
                "Wind+solar / demand": lambda v: H.pct(v, 0),
                "AC nodes": lambda v: f"{int(v)}",
                "Net export [TWh]": lambda v: H.fmt(v, 1, signed=True),
            }),
        )
        self.add("zones", "Bidding zones",
                 "How the configuration splits Germany, and how supply and demand are distributed "
                 "across the zones in the market model.", body)

    # ------------------------------------------------------------------
    def prices(self):
        m = self.m
        ps = m.price_stats
        w = m.mkt.weights
        # Map: mean zonal price for DE zones and neighbours
        vals = ps["mean"]
        vmin, vmax = float(vals.min()), float(vals.max())
        fills = {z: C.seq(vals[z], vmin, vmax) for z in m.de_zones}
        mp = base_map(m, geo.MAP_EXTENT["europe"], zones_fill=fills, height=600)
        countries = geo.countries(m.data_dir)
        for z in ps.index:
            if z in m.de_zones:
                continue
            hit = countries[countries.code == z]
            if len(hit):
                mp.polygon(hit.geometry.iloc[0], C.seq(vals[z], vmin, vmax), line="@surface",
                           width=1, hover=f"{z}: {vals[z]:.2f} €/MWh", tolerance=0.03)
        lx, ly, lt = [], [], []
        for z in ps.index:
            x, y = zone_anchor(m, z)
            lx.append(x)
            ly.append(y)
            lt.append(f"{m.label(z).split(' ')[0]}<br>{vals[z]:.1f}")
        mp.labels(lx, ly, lt, size=11)
        mp.colorbar(vmin, vmax, "€/MWh")
        f_map = self.fig("price-map", mp.spec())

        # Price duration curves (German zones)
        n = len(m.prices)
        idx = np.linspace(0, n - 1, min(n, 400)).astype(int)
        x = np.round(idx / (n - 1) * 100, 2)
        series = []
        for z in m.de_zones:
            srt = np.sort(m.prices[z].values)[::-1][idx]
            series.append({"name": m.label(z), "values": srt, "color": zone_color(m, z)})
        spec = C.lines(x, series, unit="€/MWh", xtitle="share of the year [%]", height=340)
        f_pdc = self.fig("price-duration", spec)

        # Monthly heatmap (all zones)
        mon = m.price_monthly
        order = m.de_zones + [z for z in mon.columns if z not in m.de_zones]
        z = [mon[c].values for c in order]
        f_heat = self.fig("price-monthly", C.heatmap(
            z, [MONTHS[i - 1] for i in mon.index], [m.label(c) for c in order],
            unit="€/MWh", height=60 + 24 * len(order), digits=1))

        blocks = [
            H.fig_block(f_map, "Average zonal price",
                        "Time-weighted mean market-clearing price per bidding zone. German zones are "
                        "outlined, neighbouring market zones are coloured on the same scale.", table=False),
            H.fig_block(f_pdc, "Price duration curves of the German zones",
                        "Prices sorted from highest to lowest. Curves that lie on top of each other mean "
                        "the zones are rarely separated by congestion."),
            H.fig_block(f_heat, "Monthly mean price per zone", wide=True),
        ]
        if len(m.de_zones) > 1:
            sp = np.sort(m.de_spread.values)[::-1][idx]
            spec = C.lines(
                x, [{"name": "Max − min across German zones", "values": sp, "color": "@s7"}],
                unit="€/MWh", xtitle="share of the year [%]", height=300)
            # Zoom on the hours in which the zones actually separate.
            cut = float((m.de_spread > 0.01).mean() * 100)
            spec["layout"]["xaxis"]["range"] = [0, min(100, max(2.0, cut * 1.5))]
            f_sp = self.fig("price-spread", spec)
            blocks.append(H.fig_block(
                f_sp, "Price spread between German zones",
                f"Largest price difference between any two German zones, sorted and zoomed on the "
                f"hours with a spread > 0.01 €/MWh ({H.pct((m.de_spread > 0.01).mean())} of the year). "
                f"The zones share one price (< 1 €/MWh apart) in {H.pct(m.price_convergence)} of hours."))
        hp = m.price_hourly
        f_hour = self.fig("price-hour", C.lines(
            hp.index, [{"name": m.label(z), "values": hp[z].values, "color": zone_color(m, z)}
                       for z in m.de_zones],
            unit="€/MWh", xtitle="hour of day", height=300))
        blocks.append(H.fig_block(f_hour, "Average daily price profile",
                                  "Mean price by hour of day. The 5-hour sampling cycles through "
                                  "every hour of the day over five days."))

        # Nodal shadow prices from the grid optimisation
        npz = m.nodal_price
        ac = m.ac.loc[npz.index]
        de = ac[ac.zone.isin(m.de_zones)]
        v = npz.loc[de.index]
        lo, hi = float(v.quantile(0.02)), float(v.quantile(0.98))
        mp2 = base_map(m, geo.MAP_EXTENT["de"])
        mp2.points(de.x, de.y, [C.seq(p, lo, hi) for p in v], size=10,
                   hover=[f"Bus {b} · {H.esc(m.label(zz))}<br>{p:.2f} €/MWh"
                          for b, zz, p in zip(de.index, de.zone, v)])
        mp2.colorbar(lo, hi, "€/MWh")
        f_nodal = self.fig("price-nodal", mp2.spec())
        blocks.append(H.fig_block(
            f_nodal, "Nodal shadow prices of the grid optimisation",
            "Marginal cost of supplying one more MWh at each node in the redispatch/expansion model. "
            "Differences between nodes indicate where the grid binds, i.e. the price signal a nodal "
            "or finer zonal split would reveal. These are not market prices.", table=False))
        tbl = ps.assign(label=[m.label(z) for z in ps.index]).set_index("label")[
            ["mean", "load_weighted", "p05", "p95", "max", "std", "demand_twh"]]
        tbl.columns = ["Mean", "Load-weighted", "P5", "P95", "Max", "Std. dev.", "Demand [TWh]"]
        tbl.index.name = "Zone"
        blocks.append(H.table_block(tbl, "Price statistics [€/MWh]"))
        self.add("prices", "Electricity prices",
                 "Market-clearing prices of the bidding zones (rolling-horizon unit-commitment market "
                 "model). Splitting Germany only creates price differences when the capacity between "
                 "the zones is congested.", H.grid(*blocks))

    # ------------------------------------------------------------------
    def exchange(self):
        m = self.m
        ex = m.exchange.copy()
        if ex.empty:
            return
        anchors = {z: zone_anchor(m, z) for z in set(ex.zone_a) | set(ex.zone_b)}
        ex["x0"] = ex.zone_a.map(lambda z: anchors[z][0])
        ex["y0"] = ex.zone_a.map(lambda z: anchors[z][1])
        ex["x1"] = ex.zone_b.map(lambda z: anchors[z][0])
        ex["y1"] = ex.zone_b.map(lambda z: anchors[z][1])
        ex["color"] = [C.seq(v, 0, max(0.5, ex.congested_share.max())) for v in ex.congested_share]
        ex["hover"] = [
            f"<b>{H.esc(r.label)}</b><br>capacity {r.capacity_mw:,.0f} MW<br>"
            f"{H.esc(m.label(r.zone_a))} → {H.esc(m.label(r.zone_b))}: {r.a_to_b_twh:.1f} TWh<br>"
            f"{H.esc(m.label(r.zone_b))} → {H.esc(m.label(r.zone_a))}: {r.b_to_a_twh:.1f} TWh<br>"
            f"mean utilisation {r.mean_util * 100:.0f} %, congested {r.congested_share * 100:.0f} % of hours<br>"
            f"congestion rent {r.rent_meur:,.0f} M€"
            for r in ex.itertuples()
        ]
        mp = base_map(m, geo.MAP_EXTENT["europe"], zones_fill={z: f"@fade:s{i + 1}" for i, z in enumerate(m.de_zones)},
                      height=600)
        mp.segments(ex, "color", "capacity_mw", "hover", wmin=1.5, wmax=11)
        lx, ly, lt = zip(*[(anchors[z][0], anchors[z][1], m.label(z).split(" ")[0]) for z in anchors])
        mp.points(lx, ly, "@ink", size=7)
        mp.labels(lx, [y + 0.45 for y in ly], lt, size=11)
        mp.colorbar(0, max(0.5, ex.congested_share.max()) * 100, "% hours congested")
        f_map = self.fig("exchange-map", mp.spec())

        net = m.net_position.reindex([z for z in m.zones if z in m.net_position.index])
        net = net[net.abs() > 0.01]
        f_net = self.fig("exchange-net", {
            **C.bars([m.label(z) for z in net.index],
                     [{"name": "Net export", "values": net.values,
                       "color": [UP if v > 0 else DOWN for v in net.values]}],
                     unit="TWh", horizontal=True, height=60 + 26 * len(net), ytitle="net export [TWh]"),
        })
        blocks = [
            H.fig_block(f_map, "Cross-zonal transmission in the market model",
                        "Lines connect zone centres; width = transfer capacity, colour = share of hours in "
                        "which the corridor is congested (≥ 98 % of its limit). Hover for flows and rent.",
                        wide=True, table=False),
            H.fig_block(f_net, "Net export position per zone",
                        "Positive (red) = net exporter, negative (blue) = net importer over the year."),
        ]
        de_int = ex[ex.kind == "DE internal"].sort_values("capacity_mw", ascending=False).head(8)
        if len(de_int):
            n = len(next(iter(m.exchange_flows.values())))
            idx = np.linspace(0, n - 1, min(n, 300)).astype(int)
            x = np.round(idx / (n - 1) * 100, 2)
            series = []
            for i, r in enumerate(de_int.itertuples()):
                f = m.exchange_flows[(r.zone_a, r.zone_b)]
                series.append({"name": f"{m.label(r.zone_a).split(' ')[0]} → {m.label(r.zone_b).split(' ')[0]}",
                               "values": np.sort(f.values)[::-1][idx] / 1e3, "color": f"@s{i + 1}"})
            f_fdc = self.fig("exchange-fdc", C.lines(x, series, unit="GW",
                                                     xtitle="share of the year [%]", height=320))
            blocks.append(H.fig_block(
                f_fdc, "Flow duration of the corridors between German zones",
                "Sorted hourly flow on each internal corridor (positive = in the direction named). Flat "
                "plateaus at the ends show hours in which the corridor is at its limit."))
        tbl = ex[ex.kind != "Abroad"].sort_values("congested_share", ascending=False)
        tbl = pd.DataFrame({
            "Corridor": tbl.label, "Type": tbl.kind,
            "Capacity [MW]": tbl.capacity_mw,
            "A→B [TWh]": tbl.a_to_b_twh, "B→A [TWh]": tbl.b_to_a_twh,
            "Mean utilisation": tbl.mean_util, "Congested hours": tbl.congested_share,
            "Mean |Δprice| [€/MWh]": tbl.mean_abs_price_diff,
            "Congestion rent [M€]": tbl.rent_meur,
        }).set_index("Corridor")
        blocks.append(H.table_block(tbl, "Corridors touching Germany", formats={
            "Capacity [MW]": lambda v: H.fmt(v, 0),
            "Mean utilisation": lambda v: H.pct(v, 0), "Congested hours": lambda v: H.pct(v, 1),
            "Congestion rent [M€]": lambda v: H.fmt(v, 1),
        }))
        self.add("exchange", "Cross-zonal exchange & congestion",
                 "Flows between bidding zones in the market model. Corridors between German zones "
                 "are aggregated AC lines (transport model); price differences arise only when "
                 "they are fully used. Congestion rent = |flow| × |price difference|.", H.grid(*blocks))

    # ------------------------------------------------------------------
    def redispatch(self):
        m = self.m
        t = m.rd_total
        A = self.all
        d = lambda fn: _delta(A, m, fn)
        tiles = [
            H.kpi("Up-regulation in DE", H.fmt(t["up_de_twh"], 2, "TWh"), None,
                  d(lambda x: x.rd_total["up_de_twh"]), "lower", "TWh"),
            H.kpi("Down-regulation in DE", H.fmt(t["down_de_twh"], 2, "TWh"), None,
                  d(lambda x: x.rd_total["down_de_twh"]), "lower", "TWh"),
            H.kpi("Redispatch cost", H.fmt(t["cost_meur"], 0, "M€"),
                  f"of which German units {H.fmt(t['cost_de_meur'], 0, 'M€')}",
                  d(lambda x: x.rd_total["cost_meur"]), "lower", "M€"),
            H.kpi("Redispatch abroad", H.fmt(t["up_abroad_twh"] + t["down_abroad_twh"], 1, "TWh"),
                  f"up {H.fmt(t['up_abroad_twh'], 1)} · down {H.fmt(t['down_abroad_twh'], 1)} TWh", None),
            H.kpi("Down-regulation in SH", H.fmt(t["down_focus_twh"], 2, "TWh"),
                  f"{H.pct(t['down_focus_twh'] / t['down_de_twh'] if t['down_de_twh'] else np.nan)} of German down-regulation",
                  d(lambda x: x.rd_total["down_focus_twh"]), "lower", "TWh"),
        ]
        g = m.rd_by_group.reindex(GEN_GROUP_ORDER).fillna(0)
        g = g[(g.abs().sum(axis=1)) > 0.001]
        f_grp = self.fig("rd-group", C.bars(
            list(g.index),
            [{"name": "Up (more generation)", "values": g.get("up", pd.Series(0, index=g.index)).values, "color": UP},
             {"name": "Down (less generation)", "values": g.get("down", pd.Series(0, index=g.index)).values, "color": DOWN}],
            unit="TWh", stacked=True, horizontal=True, height=60 + 34 * len(g)))
        zz = m.rd_by_zone.reindex([z for z in m.de_zones + ["Abroad"] if z in m.rd_by_zone.index]).fillna(0)
        f_zone = self.fig("rd-zone", C.bars(
            [m.label(z) for z in zz.index],
            [{"name": "Up", "values": zz.get("up", 0 * zz.iloc[:, 0]).values, "color": UP},
             {"name": "Down", "values": zz.get("down", 0 * zz.iloc[:, 0]).values, "color": DOWN}],
            unit="TWh", stacked=True, height=340))

        nodes = m.rd_nodes
        nodes = nodes[nodes.volume > 1e-4]
        ac = m.ac.loc[nodes.index]
        sizes, vmax = C.bubble_sizes(nodes.volume.values, smax=36)
        share = (nodes.net / nodes.volume).fillna(0)
        mp = base_map(m, geo.MAP_EXTENT["de"])
        mp.points(ac.x, ac.y, [C.div(s, 1.0) for s in share], size=sizes,
                  hover=[f"Bus {b} · {H.esc(m.label(z))}<br>up {u:.2f} TWh<br>down {-dn:.2f} TWh"
                         for b, z, u, dn in zip(nodes.index, ac.zone, nodes.up, nodes.down)],
                  opacity=0.9)
        mp.colorbar(-100, 100, "net direction [%]<br>(− down … up +)", "@divscale")
        mp.size_legend([vmax, vmax / 4], [36, 4 + 32 * 0.5], "TWh")
        f_map = self.fig("rd-map", mp.spec())

        wk = m.rd_weekly
        f_ts = self.fig("rd-weekly", C.lines(
            [d_.strftime("%Y-%m-%d") for d_ in wk.index],
            [{"name": "Up", "values": wk["up"].values, "color": UP, "fill": "tozeroy",
              "fillcolor": "@fade:div:0.86"},
             {"name": "Down", "values": wk["down"].values, "color": DOWN, "fill": "tozeroy",
              "fillcolor": "@fade:div:0.14"}],
            unit="TWh per week", height=300, digits=3))
        units = m.rd_units[m.rd_units.german]
        top = (units.groupby("bus").agg(up=("mwh", lambda s: s[s > 0].sum() / 1e6),
                                        down=("mwh", lambda s: -s[s < 0].sum() / 1e6),
                                        cost=("cost", lambda s: s.sum() / 1e6)))
        top["volume"] = top.up + top.down
        top = top.sort_values("volume", ascending=False).head(15)
        tbl = pd.DataFrame({
            "Node": top.index, "Zone": [m.label(m.bus_zone.get(b, "")) for b in top.index],
            "SH": ["yes" if b in m.focus_buses else "" for b in top.index],
            "Up [TWh]": top.up.values, "Down [TWh]": top.down.values, "Cost [M€]": top.cost.values,
        }).set_index("Node")
        body = H.kpi_grid(tiles) + H.grid(
            H.fig_block(f_map, "Where redispatch happens",
                        "Bubble area = redispatched energy at the node (up + down); colour shows whether the "
                        "node is mainly ramped down (blue) or up (red).", table=False),
            H.fig_block(f_grp, "Redispatch by technology (Germany)",
                        "Positive = additional generation (ramp-up), negative = reduced generation (ramp-down)."),
            H.fig_block(f_zone, "Redispatch by bidding zone",
                        "German zones and all neighbouring units together ('Abroad')."),
            H.fig_block(f_ts, "Weekly redispatch in Germany", wide=True),
            H.table_block(tbl, "Top 15 redispatch nodes in Germany"),
        )
        self.add("redispatch", "Redispatch",
                 "The grid optimisation starts from the market dispatch and adds ramp-up/ramp-down units "
                 "to relieve congestion (eTraGo add_redispatch_generators). Costs follow eTraGo: ramp-up at "
                 "max(marginal cost, zonal price), ramp-down compensated at the zonal price minus "
                 "saved fuel cost.", body)

    # ------------------------------------------------------------------
    def curtailment(self):
        m = self.m
        t = m.curt_total
        A = self.all
        d = lambda fn: _delta(A, m, fn)
        tiles = [
            H.kpi("Curtailment in the market", H.fmt(t["market_de_twh"], 1, "TWh"),
                  f"{H.pct(t['market_de_rate'])} of available energy",
                  d(lambda x: x.curt_total["market_de_twh"]), "lower", "TWh"),
            H.kpi("Curtailment after redispatch", H.fmt(t["grid_de_twh"], 1, "TWh"),
                  f"{H.pct(t['grid_de_rate'])} of available energy",
                  d(lambda x: x.curt_total["grid_de_twh"]), "lower", "TWh"),
            H.kpi("Available wind & solar (DE)", H.fmt(t["avail_de_twh"], 0, "TWh"), None),
            H.kpi("Curtailment in SH after redispatch", H.fmt(t["grid_focus_twh"], 1, "TWh"),
                  f"{H.pct(t['grid_focus_twh'] / t['avail_focus_twh'] if t['avail_focus_twh'] else np.nan)} of SH potential",
                  d(lambda x: x.curt_total["grid_focus_twh"]), "lower", "TWh"),
        ]
        g = m.curt_by_group.reindex(CURTAIL_GROUP_ORDER).fillna(0)
        f_grp = self.fig("curt-group", C.bars(
            list(g.index),
            [{"name": "Market", "values": g.curt_market_twh.values, "color": STAGE_COLOR["market"]},
             {"name": "After redispatch", "values": g.curt_grid_twh.values, "color": STAGE_COLOR["grid"]}],
            unit="TWh", height=320))
        z = m.curt_by_zone.reindex([x for x in m.de_zones + ["Abroad"] if x in m.curt_by_zone.index])
        f_zone = self.fig("curt-zone", C.bars(
            [m.label(x) for x in z.index],
            [{"name": "Market", "values": (z.curt_market_twh / z.avail_twh).values * 100, "color": STAGE_COLOR["market"]},
             {"name": "After redispatch", "values": (z.curt_grid_twh / z.avail_twh).values * 100, "color": STAGE_COLOR["grid"]}],
            unit="% of available", height=320, digits=1))
        nodes = m.curt_nodes[m.curt_nodes.curt_grid_twh > 1e-3]
        ac = m.ac.loc[nodes.index]
        rate = nodes.curt_grid_twh / nodes.avail_twh
        sizes, vmax = C.bubble_sizes(nodes.curt_grid_twh.values, smax=34)
        mp = base_map(m, geo.MAP_EXTENT["de"])
        rmax = max(0.05, float(rate.quantile(0.98)))
        mp.points(ac.x, ac.y, [C.seq(r, 0, rmax) for r in rate], size=sizes,
                  hover=[f"Bus {b} · {H.esc(m.label(zn))}<br>curtailed {c:.2f} TWh ({r * 100:.1f} %)<br>"
                         f"market only {cm:.2f} TWh"
                         for b, zn, c, r, cm in zip(nodes.index, ac.zone, nodes.curt_grid_twh, rate, nodes.curt_market_twh)])
        mp.colorbar(0, rmax * 100, "curtailment rate [%]")
        mp.size_legend([vmax, vmax / 4], [34, 4 + 30 * 0.5], "TWh")
        f_map = self.fig("curt-map", mp.spec())
        mon = m.curt_monthly
        f_mon = self.fig("curt-month", C.bars(
            [MONTHS[i - 1] for i in mon.index],
            [{"name": "Market", "values": mon.market.values, "color": STAGE_COLOR["market"]},
             {"name": "After redispatch", "values": mon.grid.values, "color": STAGE_COLOR["grid"]}],
            unit="TWh", height=300))
        n = len(m.curt_ts)
        idx = np.linspace(0, n - 1, min(n, 400)).astype(int)
        x = np.round(idx / (n - 1) * 100, 2)
        f_dur = self.fig("curt-duration", C.lines(x, [
            {"name": "Market", "values": np.sort(m.curt_ts.market.values)[::-1][idx] / 1e3, "color": STAGE_COLOR["market"]},
            {"name": "After redispatch", "values": np.sort(m.curt_ts.grid.values)[::-1][idx] / 1e3, "color": STAGE_COLOR["grid"]},
        ], unit="GW", xtitle="share of the year [%]", height=300))
        body = H.kpi_grid(tiles) + H.grid(
            H.fig_block(f_map, "Curtailment after redispatch by node",
                        "Bubble area = curtailed wind & solar energy; colour = share of the node's "
                        "available energy that is curtailed.", table=False),
            H.fig_block(f_grp, "Curtailment by technology (Germany)",
                        "Market = curtailment chosen by the market model (price-driven). After redispatch "
                        "additionally includes ramp-down of renewables for grid reasons."),
            H.fig_block(f_zone, "Curtailment rate per zone"),
            H.fig_block(f_mon, "Monthly curtailment in Germany"),
            H.fig_block(f_dur, "Duration curve of curtailed power in Germany"),
        )
        self.add("curtailment", "Curtailment of wind and solar",
                 "Available energy = installed capacity × hourly capacity factor. Curtailment is the "
                 "available energy that is not fed in — once after the market and once after "
                 "grid-related redispatch.", body)

    # ------------------------------------------------------------------
    def flexibility(self):
        m = self.m
        fs = m.flex_summary
        opts = [o for o in FLEX_ORDER if o in fs.index]
        f_sum = self.fig("flex-summary", C.bars(
            opts,
            [{"name": "Market", "values": fs.loc[opts, ("market", "twh")].values, "color": STAGE_COLOR["market"]},
             {"name": "Grid optimisation", "values": fs.loc[opts, ("grid", "twh")].values, "color": STAGE_COLOR["grid"]}],
            unit="TWh", horizontal=True, height=80 + 40 * len(opts)))
        cap = m.flex_cap_by_zone.reindex(index=m.de_zones, columns=opts).fillna(0)
        cap = cap[[c for c in cap.columns if c not in ("E-mobility charging",)]]
        f_cap = self.fig("flex-cap", C.bars(
            [m.label(z) for z in cap.index],
            [{"name": o, "values": cap[o].values, "color": FLEX_COLOR[o]} for o in cap.columns if cap[o].sum() > 0.01],
            unit="GW", stacked=True, height=360))
        hp = m.flex_hourly
        hp = hp[[c for c in opts if c in hp.columns]]
        f_hour = self.fig("flex-hour", C.lines(
            hp.index, [{"name": o, "values": hp[o].values, "color": FLEX_COLOR[o]} for o in hp.columns],
            unit="GW", xtitle="hour of day", height=340))
        pc = m.price_capture
        series = [{"name": "Zone average", "values": pc.avg_price.values, "color": "@ink", "symbol": "line-ns-open"}]
        for col, name, color, sym in [("ely_price", "Electrolysis (consumption-weighted)", "@s3", "circle"),
                                      ("bat_charge_price", "Battery charging", "@s1", "triangle-left"),
                                      ("bat_discharge_price", "Battery discharging", "@s2", "triangle-right"),
                                      ("hp_price", "Heat pumps", "@s4", "diamond")]:
            if col in pc:
                series.append({"name": name, "values": pc[col].values, "color": color, "symbol": sym})
        f_pc = self.fig("flex-price", C.dots(list(pc.label), series, unit="€/MWh", height=120 + 48 * len(pc)))

        ely = m.ely_nodes[m.ely_nodes.gw > 0.001]
        ac = m.ac.loc[ely.index]
        flh = (ely.twh * 1e6 / (ely.gw * 1e3)).fillna(0)
        sizes, vmax = C.bubble_sizes(ely.gw.values, smax=34)
        mp = base_map(m, geo.MAP_EXTENT["de"])
        mp.points(ac.x, ac.y, [C.seq(f, 0, 8760) for f in flh], size=sizes,
                  hover=[f"Bus {b} · {H.esc(m.label(z))}<br>{g:.2f} GW · {f:,.0f} full-load hours"
                         for b, z, g, f in zip(ely.index, ac.zone, ely.gw, flh)])
        mp.colorbar(0, 8760, "full-load hours")
        mp.size_legend([vmax, vmax / 4], [34, 4 + 30 * 0.5], "GW")
        f_ely = self.fig("ely-map", mp.spec())
        ez = m.ely_by_zone.reindex(m.de_zones).fillna(0)
        tbl = pd.DataFrame({
            "Zone": [m.label(z) for z in ez.index],
            "Electrolysis [GW]": ez.gw.values, "H2 input [TWh el]": ez.twh.values,
            "Full-load hours": (ez.twh * 1e6 / (ez.gw * 1e3)).replace([np.inf], np.nan).values,
            "Price paid [€/MWh]": pc.reindex(ez.index).get("ely_price", pd.Series(np.nan, index=ez.index)).values,
            "Zone avg price [€/MWh]": pc.reindex(ez.index).avg_price.values,
        }).set_index("Zone")
        body = H.grid(
            H.fig_block(f_sum, "Use of flexibility options in Germany",
                        "Annual energy: electricity drawn (electrolysis, heat, e-mobility), discharged "
                        "(storage), shifted (DSM) or produced (H2-to-power). The grid optimisation re-runs "
                        "storage and sector-coupling dispatch, so it can differ from the market."),
            H.fig_block(f_cap, "Flexible capacity per zone (grid model)",
                        "Installed power of flexible consumers and storage after investment. E-mobility "
                        "charging is left out to keep the scale readable."),
            H.fig_block(f_hour, "Average daily profile of flexibility use (market, DE)",
                        "Mean power by hour of day: electrolysers and batteries charge around solar noon "
                        "and back off in the evening peak."),
            H.fig_block(f_pc, "Prices captured by flexible demand (market)",
                        "Consumption-weighted average zonal price paid by each option vs. the zone's "
                        "average price. Larger gaps mean more price-responsive operation."),
            H.fig_block(f_ely, "Electrolyser locations and utilisation",
                        "Capacity from the pre-market investment optimisation, fixed in the grid model; "
                        "colour = full-load hours in the grid optimisation.", table=False),
            H.table_block(tbl, "Electrolysis by zone", wide=False, formats={
                "Full-load hours": lambda v: H.fmt(v, 0)}),
        )
        self.add("flexibility", "Flexibility & sector coupling",
                 "How electrolysers, heat pumps, e-mobility, demand-side management and storage respond "
                 "to zonal prices, and where electrolysis capacity is built.", body)

    # ------------------------------------------------------------------
    def batteries(self):
        m = self.m
        bz = m.bat_by_zone.reindex(m.de_zones).fillna(0)
        f_bar = self.fig("bat-zone", C.bars(
            [m.label(z) for z in bz.index],
            [{"name": "Scenario minimum (eGon2035)", "values": bz.existing_gw.values, "color": "@s1"},
             {"name": "Added by grid optimisation", "values": bz.new_gw.values, "color": "@s3"}],
            unit="GW", stacked=True, height=320))
        heat = m.bat_heat
        vabs = float(np.nanmax(np.abs(heat.values))) if heat.size else 1
        f_heat = self.fig("bat-heat", C.heatmap(
            heat.values, [f"{h:02d}h" for h in heat.columns], [MONTHS[i - 1] for i in heat.index],
            unit="GW (+ discharge / − charge)", scale="@divscale", zmid=0, height=380, digits=2))
        f_heat_spec = self.figs[f_heat]
        f_heat_spec["data"][0]["zmin"] = -vabs
        f_heat_spec["data"][0]["zmax"] = vabs
        nodes = m.bat_nodes
        nodes = nodes[(nodes.existing_gw + nodes.new_gw) > 0.005]
        ac = m.ac.loc[nodes.index]
        tot = nodes.existing_gw + nodes.new_gw
        newshare = (nodes.new_gw / tot).fillna(0)
        sizes, vmax = C.bubble_sizes(tot.values, smax=32)
        mp = base_map(m, geo.MAP_EXTENT["de"])
        mp.points(ac.x, ac.y, [C.seq(s, 0, 1) for s in newshare], size=sizes,
                  hover=[f"Bus {b} · {H.esc(m.label(z))}<br>{t_:.2f} GW ({e:.2f} scenario min., {n:.2f} added)<br>{g:.1f} GWh"
                         for b, z, t_, e, n, g in zip(nodes.index, ac.zone, tot, nodes.existing_gw, nodes.new_gw, nodes.gwh)])
        mp.colorbar(0, 100, "added capacity [%]")
        mp.size_legend([vmax, vmax / 4], [32, 4 + 28 * 0.5], "GW")
        f_map = self.fig("bat-map", mp.spec())
        pc = m.price_capture
        tbl = pd.DataFrame({
            "Zone": [m.label(z) for z in bz.index],
            "Scenario min. [GW]": bz.existing_gw.values, "Added [GW]": bz.new_gw.values,
            "Energy [GWh]": bz.gwh.values, "Discharge [TWh]": bz.discharge_twh.values,
            "Full cycles": (bz.discharge_twh * 1e3 / bz.gwh.replace(0, np.nan)).values,
            "Charge price (market)": pc.reindex(bz.index).get("bat_charge_price", pd.Series(np.nan, index=bz.index)).values,
            "Discharge price (market)": pc.reindex(bz.index).get("bat_discharge_price", pd.Series(np.nan, index=bz.index)).values,
        }).set_index("Zone")
        body = H.grid(
            H.fig_block(f_map, "Battery storage at the grid nodes",
                        "Bubble area = battery power; colour = share added by the grid optimisation on top "
                        "of the eGon2035 scenario minimum (p_nom_min).", table=False),
            H.fig_block(f_bar, "Battery power per zone"),
            H.fig_block(f_heat, "When batteries charge and discharge (grid model, DE)",
                        "Average net battery power by month and hour of day. Red = discharging, "
                        "blue = charging.", wide=True),
            H.table_block(tbl, "Battery indicators per zone", formats={"Full cycles": lambda v: H.fmt(v, 0)}),
        )
        self.add("batteries", "Batteries",
                 "Large-scale batteries are optimised in the grid model (extendable at every node). "
                 "Their siting shows where flexibility relieves the grid.", body)

    # ------------------------------------------------------------------
    def _line_map(self, df, value_col, vmax, title_unit, hover_fn, extent=None, height=600,
                  width_col="s_nom_opt", dc=None, scale_factor=1.0):
        m = self.m
        mp = base_map(m, extent or geo.MAP_EXTENT["de"], height=height)
        df = df.copy()
        df["color"] = [C.seq(v, 0, vmax) for v in df[value_col]]
        df["hover"] = [hover_fn(r) for r in df.itertuples()]
        mp.segments(df, "color", width_col, "hover", wmin=1.0, wmax=8)
        if dc is not None and len(dc):
            dc = dc.copy()
            dc["color"] = [C.seq(v, 0, vmax) for v in dc[value_col]]
            dc["hover"] = [hover_fn(r, dc=True) for r in dc.itertuples()]
            mp.segments(dc, "color", "p_nom_opt", "hover", wmin=1.0, wmax=8)
        ac = m.ac
        mp.points(ac.x, ac.y, "@ink2", size=4, line="@surface")
        mp.colorbar(0, vmax * scale_factor, title_unit)
        return mp.spec()

    def grid(self):
        m = self.m
        L = m.lines
        lt = m.lines_total
        A = self.all
        d = lambda fn: _delta(A, m, fn)
        tiles = [
            H.kpi("AC expansion in DE", H.fmt(lt["ac_exp_gw_de"], 1, "GW"),
                  f"+{H.pct(lt['rel_exp_de'], 0)} of existing capacity",
                  d(lambda x: x.lines_total["ac_exp_gw_de"]), "lower", "GW"),
            H.kpi("Expansion × length", H.fmt(lt["ac_exp_gwkm_de"], 0, "GW·km"), None,
                  d(lambda x: x.lines_total["ac_exp_gwkm_de"]), "lower", "GW·km"),
            H.kpi("Grid investment (AC + DC)", H.fmt(lt["ac_invest_meur"] + lt["dc_invest_meur"], 0, "M€/a"),
                  "annualised", d(lambda x: x.lines_total["ac_invest_meur"] + x.lines_total["dc_invest_meur"]),
                  "lower", "M€"),
            H.kpi("Expansion between DE zones", H.fmt(lt["between_zone_gw"], 2, "GW"),
                  "on AC lines crossing a zone border"),
            H.kpi("Mean line loading (DE)", H.pct(lt["mean_loading_de"], 1),
                  f"{lt['congested_lines_de']} of {lt['n_lines_de']} lines congested >10 % of hours",
                  None),
        ]
        dc = m.dc[m.dc.zone0.map(geo.is_german_zone).fillna(False) | m.dc.zone1.map(geo.is_german_zone).fillna(False)]

        def hov_exp(r, dc=False):
            if dc:
                return (f"<b>DC link {r.Index}</b><br>{r.p_nom_base:,.0f} → {r.p_nom_opt:,.0f} MW "
                        f"(+{r.expansion_mw:,.0f} MW)<br>{r.length:,.0f} km")
            return (f"<b>Line {r.Index}</b> · {r.v_nom:.0f} kV · {r.category}<br>{r.s_nom_base:,.0f} → "
                    f"{r.s_nom_opt:,.0f} MVA (+{(r.expansion_rel if r.expansion_rel == r.expansion_rel else 0) * 100:.0f} %)"
                    f"<br>{r.length:,.0f} km · {r.invest_meur:,.1f} M€/a")

        def hov_load(r, dc=False):
            return (f"<b>{'DC link' if dc else 'Line'} {r.Index}</b><br>mean loading {r.mean_loading * 100:.0f} %"
                    f"<br>congested {r.congested_share * 100:.1f} % of hours")

        Lx = L[L.category != "Abroad"]
        f_exp = self.fig("grid-exp-map", self._line_map(
            Lx.assign(v=Lx.expansion_rel.fillna(0).clip(upper=3)), "v", 3.0,
            "expansion [% of existing]", hov_exp, dc=dc.assign(v=(dc.expansion_mw / dc.p_nom_base.replace(0, np.nan)).fillna(0).clip(upper=3)),
            scale_factor=100))
        f_load = self.fig("grid-load-map", self._line_map(
            Lx, "mean_loading", 1.0, "mean loading [%]", hov_load, dc=dc, scale_factor=100))
        f_cong = self.fig("grid-cong-map", self._line_map(
            Lx, "congested_share", max(0.2, float(Lx.congested_share.max())), "hours at limit [%]",
            hov_load, dc=dc, scale_factor=100))

        cat = m.exp_by_category.reindex(["Within zone", "Between DE zones", "Cross-border", "Abroad"]).fillna(0)
        f_cat = self.fig("grid-cat", C.bars(
            list(cat.index), [{"name": "Expansion", "values": cat.expansion_gwkm.values, "color": "@s1"}],
            unit="GW·km", horizontal=True, height=220, text=True, digits=0))
        ez = m.exp_by_zone.reindex(m.de_zones).fillna(0)
        f_zone = self.fig("grid-zone", C.bars(
            [m.label(z) for z in ez.index], [{"name": "Expansion", "values": ez.expansion_gwkm.values, "color": "@s1"}],
            unit="GW·km", height=300, text=True, digits=0))
        f_hist = self.fig("grid-hist", C.histogram(
            [{"name": "German lines", "values": Lx.mean_loading.values * 100, "color": "@s1"}],
            xtitle="mean loading after expansion [%]", height=300, nbins=25))
        series = []
        for i, (lid, vals) in enumerate(m.loading_duration.items()):
            n = len(vals)
            idx = np.linspace(0, n - 1, min(n, 300)).astype(int)
            r = L.loc[lid]
            series.append({"name": f"Line {lid} ({m.label(r.zone0).split(' ')[0]}–{m.label(r.zone1).split(' ')[0]})",
                           "values": vals[idx] * 100, "color": f"@s{i + 1}"})
        x = np.round(np.linspace(0, 100, len(series[0]["values"])), 2) if series else []
        f_ldc = self.fig("grid-ldc", C.lines(x, series, unit="loading [%]", xtitle="share of the year [%]", height=320))
        top = Lx.sort_values(["congested_share", "mean_loading"], ascending=False).head(20)
        tbl = pd.DataFrame({
            "Line": top.index, "kV": top.v_nom.values,
            "From zone": [m.label(z) for z in top.zone0], "To zone": [m.label(z) for z in top.zone1],
            "Type": top.category.values, "SH": ["yes" if f else "" for f in top.focus],
            "Length [km]": top.length.values, "Existing [MVA]": top.s_nom_base.values,
            "Optimal [MVA]": top.s_nom_opt.values, "Mean loading": top.mean_loading.values,
            "Hours at limit": top.congested_share.values,
        }).set_index("Line")
        body = H.kpi_grid(tiles) + H.grid(
            H.fig_block(f_exp, "Grid expansion",
                        "Colour = capacity added relative to the existing line (capped at 300 %), width = "
                        "optimised capacity. DC links are drawn in the same way.", table=False),
            H.fig_block(f_load, "Average line loading after expansion",
                        "Time-weighted mean of |flow| / (s_nom_opt × s_max_pu).", table=False),
            H.fig_block(f_cong, "Hours at the thermal limit",
                        "Share of hours in which a line runs at ≥ 98 % of its (expanded) limit.", table=False),
            H.fig_block(f_cat, "Expansion by line type",
                        "Within zone = both ends in the same German zone; between DE zones = line crosses a "
                        "German bidding-zone border; cross-border = connects Germany with a neighbour."),
            H.fig_block(f_zone, "Expansion inside each German zone"),
            H.fig_block(f_hist, "Distribution of mean line loading"),
            H.fig_block(f_ldc, "Loading duration of the most congested lines"),
            H.table_block(tbl, "Most congested lines in and around Germany", formats={
                "kV": lambda v: H.fmt(v, 0), "Length [km]": lambda v: H.fmt(v, 0),
                "Existing [MVA]": lambda v: H.fmt(v, 0), "Optimal [MVA]": lambda v: H.fmt(v, 0),
                "Mean loading": lambda v: H.pct(v, 0), "Hours at limit": lambda v: H.pct(v, 1)}),
        )
        self.add("grid", "Grid expansion & line loading",
                 "Results of the grid optimisation on the clustered transmission grid "
                 f"({len(m.ac)} AC nodes, {len(L)} AC lines). Lines are extendable; investment uses "
                 "eTraGo's annualised capital cost ((s_nom_opt − s_nom_min) × capital_cost).", body)

    # ------------------------------------------------------------------
    def costs(self):
        m = self.m
        c = m.costs
        A = self.all
        comp = c.drop(["Market dispatch (fuel & variable)", "Natural-gas imports (CH4_NG)"])
        comp = comp[comp.abs() > 0.5]
        tiles = [
            H.kpi("System cost excl. gas imports", H.fmt((c.sum() - c["Natural-gas imports (CH4_NG)"]) / 1e3, 2, "bn€/a"),
                  f"incl. gas imports {H.fmt(c.sum() / 1e3, 2, 'bn€/a')}",
                  _delta(A, m, lambda x: (x.costs.sum() - x.costs["Natural-gas imports (CH4_NG)"]) / 1e3), "lower", "bn€"),
            H.kpi("Market dispatch cost", H.fmt(c["Market dispatch (fuel & variable)"] / 1e3, 2, "bn€/a"),
                  "whole model area, fuel + variable O&M, without gas imports",
                  _delta(A, m, lambda x: x.costs["Market dispatch (fuel & variable)"] / 1e3), "lower", "bn€"),
            H.kpi("Natural-gas imports", H.fmt(c["Natural-gas imports (CH4_NG)"] / 1e3, 2, "bn€/a"),
                  "see data note on gas storage",
                  _delta(A, m, lambda x: x.costs["Natural-gas imports (CH4_NG)"] / 1e3), "neutral", "bn€"),
            H.kpi("Wholesale cost of German demand", H.fmt(m.consumer_cost.sum() / 1e9, 2, "bn€"),
                  "AC demand × zonal price", _delta(A, m, lambda x: x.consumer_cost.sum() / 1e9), "lower", "bn€"),
            H.kpi("Congestion rent (DE corridors)",
                  H.fmt(m.exchange[m.exchange.kind != "Abroad"].rent_meur.sum(), 0, "M€"), None,
                  _delta(A, m, lambda x: x.exchange[x.exchange.kind != "Abroad"].rent_meur.sum()), "neutral", "M€"),
        ]
        f_comp = self.fig("cost-comp", C.bars(
            list(comp.index), [{"name": "Annual cost", "values": comp.values, "color": "@s1"}],
            unit="M€/a", horizontal=True, height=80 + 36 * len(comp), text=True, digits=0))
        blocks = [H.fig_block(f_comp, "Cost components besides market dispatch",
                              "Annual values. Investments are annualised capital costs as used in the objective.")]
        sq = A.get("status_quo")
        if sq is not None and sq is not m:
            dlt = (c - sq.costs.reindex(c.index).fillna(0)).drop("Natural-gas imports (CH4_NG)")
            dlt = dlt[dlt.abs() > 0.5]
            f_d = self.fig("cost-delta", C.bars(
                list(dlt.index), [{"name": "Δ vs status quo", "values": dlt.values,
                                   "color": [UP if v > 0 else DOWN for v in dlt.values]}],
                unit="M€/a", horizontal=True, height=80 + 36 * len(dlt), text=True, digits=0))
            blocks.append(H.fig_block(f_d, "Change against the status quo",
                                      "Red = more expensive than the single DE/LU zone, blue = cheaper. "
                                      "Natural-gas imports are left out (see the note above)."))
        cc = pd.DataFrame({
            "Zone": [m.label(z) for z in m.de_zones],
            "Demand [TWh]": m.price_stats.loc[m.de_zones, "demand_twh"].values,
            "Price [€/MWh]": m.price_stats.loc[m.de_zones, "load_weighted"].values,
            "Cost [M€]": (m.consumer_cost.reindex(m.de_zones) / 1e6).values,
        }).set_index("Zone")
        blocks.append(H.table_block(cc, "Wholesale electricity cost per zone", wide=False,
                                    note="Price = load-weighted zonal price.",
                                    formats={"Cost [M€]": lambda v: H.fmt(v, 0)}))
        tb = pd.DataFrame({"Cost [M€/a]": c}).rename_axis("Component")
        obj = m.objectives
        tb.loc["Solver objective: pre-market [reference]"] = (obj["pre_market"] or np.nan) / 1e6
        tb.loc["Solver objective: grid optimisation [reference]"] = (obj["grid"] or np.nan) / 1e6
        blocks.append(H.table_block(tb, "Cost breakdown", wide=False,
                                    formats={"Cost [M€/a]": lambda v: H.fmt(v, 0)}))
        body = H.kpi_grid(tiles) + H.grid(*blocks)
        if getattr(m, "gas_store_check", None):
            g = m.gas_store_check
            body = H.callout(
                f"<strong>Natural-gas imports are reported separately.</strong> In the market export the "
                f"CH4 stores discharge {g['net_discharge_twh']:,.0f} TWh net over the year while their stored "
                f"energy changes by {g['soc_drop_twh']:,.1f} TWh. Gas imports fill this gap, so their cost "
                "differs between runs for reasons unrelated to the bidding zones. Compare configurations on "
                "the cost <em>excluding</em> gas imports.", "warn") + body
        self.add("costs", "Costs",
                 "System cost = market dispatch + start-up + redispatch + annualised investments of the "
                 "grid stage (lines, DC links, batteries, stores) + electrolysers decided in the pre-market "
                 "model. Wholesale cost and congestion rent show how money is redistributed between "
                 "consumers, producers and grid operators.", body)

    # ------------------------------------------------------------------
    def focus(self):
        m = self.m
        f = m.focus
        A = self.all
        d = lambda k: _delta(A, m, lambda x: x.focus[k])
        tiles = [
            H.kpi("Zone containing SH", f["zone"], f"zonal mean price {H.fmt(f['zone_price'], 2, '€/MWh')}"),
            H.kpi("Mean nodal price in SH", H.fmt(f["nodal_price"], 2, "€/MWh"),
                  f"German average {H.fmt(f['nodal_price_de'], 2, '€/MWh')}", d("nodal_price"), "neutral", "€/MWh"),
            H.kpi("Wind & solar capacity in SH", H.fmt(f["vres_gw"], 1, "GW"),
                  f"vs. {H.fmt(f['demand_twh'], 1, 'TWh')} electricity demand"),
            H.kpi("Curtailment in SH", H.fmt(f["curt_grid_twh"], 1, "TWh"),
                  f"market {H.fmt(f['curt_market_twh'], 1)} TWh", d("curt_grid_twh"), "lower", "TWh"),
            H.kpi("Redispatch in SH", H.fmt(f["rd_down_twh"] + f["rd_up_twh"], 2, "TWh"),
                  f"down {H.fmt(f['rd_down_twh'], 2)} · up {H.fmt(f['rd_up_twh'], 2)} TWh",
                  _delta(A, m, lambda x: x.focus["rd_down_twh"] + x.focus["rd_up_twh"]), "lower", "TWh"),
            H.kpi("Electrolysers in SH", H.fmt(f["ely_gw"], 2, "GW"),
                  f"{H.fmt(f['ely_twh'], 2)} TWh consumed", d("ely_gw"), "neutral", "GW"),
            H.kpi("Batteries in SH", H.fmt(f["bat_gw"], 2, "GW"), None, d("bat_gw"), "neutral", "GW"),
            H.kpi("Line expansion touching SH", H.fmt(f["line_exp_gwkm"], 0, "GW·km"),
                  f"{H.fmt(f['line_exp_gw'], 1, 'GW')}", d("line_exp_gwkm"), "lower", "GW·km"),
        ]
        L = m.lines
        sel = L[L.focus | ((L.x0.between(7.5, 11.8) & L.y0.between(53.0, 55.4)))]
        mp = C.MapFigure(geo.FOCUS_EXTENT, height=620)
        mp.background(geo.background(m.data_dir, geo.FOCUS_EXTENT))
        for _, row in m.zone_gdf.iterrows():
            mp.polygon(row.geometry, "rgba(0,0,0,0)", line="@ink2", width=1.2, tolerance=0.005)
        mp.outline(geo.focus_outline(m.data_dir), color="@ink", width=1.4, dash="dot")
        s = sel.copy()
        s["color"] = [C.seq(v, 0, 1) for v in s.mean_loading]
        s["hover"] = [f"<b>Line {r.Index}</b> · {r.v_nom:.0f} kV<br>{r.s_nom_base:,.0f} → {r.s_nom_opt:,.0f} MVA"
                      f"<br>mean loading {r.mean_loading * 100:.0f} %, at limit {r.congested_share * 100:.1f} % of hours"
                      for r in s.itertuples()]
        mp.segments(s, "color", "s_nom_opt", "hover", wmin=1.5, wmax=9)
        fb = m.ac[m.ac.focus]
        cn = m.curt_nodes.reindex(fb.index).fillna(0)
        en = m.ely_nodes.reindex(fb.index).fillna(0)
        sizes, vmax = C.bubble_sizes(cn.curt_grid_twh.values, smax=30)
        mp.points(fb.x, fb.y, "@s2", size=sizes, line="@surface", opacity=0.75, name="Curtailment after redispatch",
                  legend=True,
                  hover=[f"Bus {b}<br>curtailed {c:.2f} TWh<br>electrolysis {e:.2f} GW<br>"
                         f"nodal price {m.nodal_price.get(b, np.nan):.2f} €/MWh"
                         for b, c, e in zip(fb.index, cn.curt_grid_twh, en.gw)])
        ely = en[en.gw > 0.001]
        if len(ely):
            es, _ = C.bubble_sizes(ely.gw.values, smax=18, smin=5)
            mp.points(fb.loc[ely.index].x, fb.loc[ely.index].y, "@s3", size=es, symbol="diamond",
                      name="Electrolysers", legend=True,
                      hover=[f"Bus {b}<br>electrolysis {g:.2f} GW" for b, g in zip(ely.index, ely.gw)])
        mp.colorbar(0, 100, "mean line loading [%]")
        f_map = self.fig("focus-map", mp.spec())
        rd = m.rd_nodes.reindex(fb.index).fillna(0)
        tbl = pd.DataFrame({
            "Node": fb.index,
            "Zone": [m.label(z) for z in fb.zone],
            "Nodal price [€/MWh]": m.nodal_price.reindex(fb.index).values,
            "Available RES [TWh]": cn.avail_twh.values,
            "Curtailed [TWh]": cn.curt_grid_twh.values,
            "Redispatch down [TWh]": (-rd.down).values,
            "Redispatch up [TWh]": rd.up.values,
            "Electrolysis [GW]": en.gw.values,
            "Batteries [GW]": (m.bat_nodes.reindex(fb.index).fillna(0)[["existing_gw", "new_gw"]].sum(axis=1)).values,
        }).set_index("Node").sort_values("Curtailed [TWh]", ascending=False)
        body = H.kpi_grid(tiles) + H.grid(
            H.fig_block(f_map, "Schleswig-Holstein: line loading, curtailment and electrolysers",
                        "Lines coloured by mean loading (width = capacity), orange bubbles = curtailed energy "
                        "per node, green diamonds = electrolysers. Thin grey lines = bidding-zone borders.",
                        wide=True, table=False),
            H.table_block(tbl, f"The {len(fb)} SH nodes", max_rows=60),
        )
        self.add("focus", "Focus region Schleswig-Holstein",
                 "Schleswig-Holstein is modelled with 50 AC nodes of its own (exact focus clustering). "
                 "These indicators show how the bidding-zone design affects the region.", body)

    # ------------------------------------------------------------------
    def notes(self):
        m = self.m
        sh = m.shedding
        items = [
            "Time resolution: 1752 snapshots in 5-hour steps, each weighted with 5 h, so annual "
            "sums cover 8760 h.",
            "Zones: grid nodes are assigned to bidding zones by point-in-polygon on the same shapefiles "
            "used by <code>etrago.tools.market_zones</code>, with the nearest zone for German nodes "
            "outside every polygon. Market buses are mapped to zones via the shared generators.",
            "Prices are the marginal prices of the market model (rolling-horizon unit commitment). "
            "Nodal prices come from the linear grid optimisation and include redispatch effects.",
            "Redispatch = dispatch of eTraGo's ramp_up/ramp_down generators and gas-turbine links. "
            "Costs = Σ p × marginal_cost × weighting (calc_etrago_results).",
            "Curtailment uses capacity × p_max_pu from the market model. Grid-stage infeed = market "
            "dispatch + ramp_up + ramp_down of the same unit.",
            "Line loading = |p0| / (s_nom_opt × s_max_pu(t)). Congested = loading ≥ 98 %.",
            f"Load shedding: market {sh['market_twh'] * 1e3:.3f} GWh, grid {sh['grid_twh'] * 1e3:.1f} GWh "
            f"(peak {sh['grid_peak_mw']:.1f} MW). Results are not distorted by unserved load.",
            "Costs outside Germany are included in the market dispatch cost, because the market "
            "model optimises all countries together.",
        ] + [H.esc(n) for n in m.notes]
        body = H.callout("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
        self.add("notes", "Method & data notes", "How the numbers in this report are derived.", body)


# ===========================================================================
# Comparison report
# ===========================================================================
class ComparisonReport:
    def __init__(self, metrics, nav, links, plotly_js=None):
        self.M = metrics  # ordered dict key -> ScenarioMetrics
        self.nav = nav
        self.links = links
        self.plotly_js = plotly_js
        self.figs = {}
        self.sections = []
        self.toc = []

    fig = ScenarioReport.fig
    add = ScenarioReport.add

    def keys(self):
        return list(self.M)

    def labels(self):
        return [SCENARIO_LABELS.get(k, k) for k in self.M]

    def colors(self):
        return [SCENARIO_COLOR.get(k, "@muted") for k in self.M]

    def single(self, fid, values, unit, digits=1, height=300):
        """One bar per configuration (colour = configuration identity)."""
        return self.fig(fid, C.bars(self.labels(), [{"name": unit, "values": values, "color": self.colors()}],
                                    unit=unit, height=height, legend=False, text=True, digits=digits))

    def build(self):
        self.overview()
        self.zones()
        self.prices()
        self.redispatch()
        self.grid()
        self.flex()
        self.costs()
        self.focus()
        return H.page(
            title="SPREAD.SH bidding-zone comparison",
            heading="Bidding-zone configurations compared",
            intro=("Five eTraGo runs that differ only in the bidding-zone layout of Germany "
                   "(status quo DE/LU and the DE2–DE5 splits), with the same grid clustering "
                   "(100 AC nodes, 50 in Schleswig-Holstein), eGon2035 data and a full year in 5-hour steps."),
            chips=[f"{len(self.M)} configurations", "eGon2035", "8760 h · 5-h steps",
                   "focus: Schleswig-Holstein"],
            nav=self.nav, toc=self.toc, sections_html="".join(self.sections), figs=self.figs,
            plotly_js=self.plotly_js, footer="Generated by spread_sh_reports.",
        )

    def overview(self):
        cards = []
        for k, m in self.M.items():
            kp = m.kpis()
            cards.append(
                f'<a class="card" href="{H.esc(self.links[k])}"><div class="ft">{H.esc(SCENARIO_LABELS.get(k, k))}</div>'
                f'<div class="cs">{len(m.de_zones)} German zone{"s" if len(m.de_zones) > 1 else ""} · '
                f'{H.fmt(kp["de_avg_price"], 2, "€/MWh")} · redispatch {H.fmt(kp["rd_up_de_twh"] + kp["rd_down_de_twh"], 1, "TWh")}</div>'
                f'<div class="cs">Open full report →</div></a>')
        rows = {
            "German zones": ("n_de_zones", lambda v: f"{int(v)}"),
            "Avg. German price [€/MWh]": ("de_avg_price", lambda v: H.fmt(v, 2)),
            "Hours with one German price": ("price_convergence", lambda v: H.pct(v, 1)),
            "Max. zonal price gap [€/MWh]": ("max_zone_price_gap", lambda v: H.fmt(v, 3)),
            "Wholesale cost DE demand [bn€]": ("consumer_cost_beur", lambda v: H.fmt(v, 2)),
            "Redispatch up DE [TWh]": ("rd_up_de_twh", lambda v: H.fmt(v, 2)),
            "Redispatch down DE [TWh]": ("rd_down_de_twh", lambda v: H.fmt(v, 2)),
            "Redispatch cost [M€]": ("rd_cost_meur", lambda v: H.fmt(v, 0)),
            "Curtailment market DE [TWh]": ("curt_market_de_twh", lambda v: H.fmt(v, 1)),
            "Curtailment after redispatch DE [TWh]": ("curt_grid_de_twh", lambda v: H.fmt(v, 1)),
            "AC expansion DE [GW]": ("ac_exp_gw_de", lambda v: H.fmt(v, 1)),
            "AC expansion DE [GW·km]": ("ac_exp_gwkm_de", lambda v: H.fmt(v, 0)),
            "Grid investment [M€/a]": ("grid_invest_meur", lambda v: H.fmt(v, 0)),
            "Mean line loading DE": ("mean_loading_de", lambda v: H.pct(v, 1)),
            "Electrolysers DE [GW]": ("ely_gw_de", lambda v: H.fmt(v, 2)),
            "Batteries DE [GW]": ("bat_gw_de", lambda v: H.fmt(v, 2)),
            "Congestion rent DE corridors [M€]": ("congestion_rent_de_meur", lambda v: H.fmt(v, 0)),
            "System cost excl. gas imports [bn€/a]": ("cost_excl_gas_beur", lambda v: H.fmt(v, 2)),
            "System cost incl. gas imports [bn€/a]": ("total_cost_beur", lambda v: H.fmt(v, 2)),
            "Load shedding grid [TWh]": ("shedding_grid_twh", lambda v: H.fmt(v, 3)),
        }
        K = {k: m.kpis() for k, m in self.M.items()}
        head = "".join(f"<th scope='col'>{H.esc(l)}</th>" for l in self.labels())
        body = []
        for name, (key, f) in rows.items():
            cells = []
            for k in self.M:
                v = K[k][key]
                extra = ""
                if "status_quo" in K and k != "status_quo" and key != "n_de_zones":
                    dv = v - K["status_quo"][key]
                    extra = f"<div class='sub' style='color:var(--muted);font-size:11.5px'>{'+' if dv >= 0 else '−'}{f(abs(dv))}</div>"
                cells.append(f"<td class='num' data-v='{v}'>{f(v)}{extra}</td>")
            body.append(f"<tr><th scope='row'>{H.esc(name)}</th>{''.join(cells)}</tr>")
        table = (f"<div class='card wide'><div class='ft'>Key indicators</div><p class='note'>Small numbers "
                 f"below each value give the difference to the status quo.</p><div class='tscroll'><table class='tbl'>"
                 f"<thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div></div>")
        self.add("overview", "Overview",
                 "Open a configuration for its full report, or read on for side-by-side comparisons.",
                 f"<div class='cards'>{''.join(cards)}</div>" + H.grid(table))

    def zones(self):
        blocks = []
        for k, m in self.M.items():
            fills = {z: f"@fade:s{i + 1}" for i, z in enumerate(m.de_zones)}
            mp = base_map(m, geo.MAP_EXTENT["de"], zones_fill=fills, height=380)
            vals = m.price_stats["mean"]
            lx, ly, lt = [], [], []
            for z in m.de_zones:
                x, y = zone_anchor(m, z)
                lx.append(x)
                ly.append(y)
                lt.append(f"<b>{m.label(z).split(' ')[0]}</b><br>{vals[z]:.2f}")
            mp.labels(lx, ly, lt, size=11)
            spec = mp.spec()
            spec["layout"]["showlegend"] = False
            fid = self.fig(f"cmp-zone-{k}", spec)
            blocks.append(H.fig_block(fid, SCENARIO_LABELS.get(k, k),
                                      "Zone labels show the mean zonal price [€/MWh].", table=False))
        self.add("zones", "Zone layouts", "The five bidding-zone configurations of Germany. The dotted "
                 "outline is the Schleswig-Holstein focus region.", H.grid(*blocks))

    def prices(self):
        conv = [m.price_convergence * 100 for m in self.M.values()]
        f2 = self.single("cmp-conv", conv, "% of hours", 1)
        cats, vals, cols, syms = [], [], [], []
        data = []
        for k, m in self.M.items():
            ps = m.price_stats.loc[m.de_zones]
            data.append({
                "type": "scatter", "mode": "markers", "name": SCENARIO_LABELS.get(k, k),
                "y": [SCENARIO_LABELS.get(k, k)] * len(ps), "x": C._r(ps["mean"].values, 3),
                "text": [m.label(z) for z in ps.index],
                "marker": {"color": SCENARIO_COLOR.get(k), "size": 12, "line": {"color": "@surface", "width": 2}},
                "hovertemplate": "%{text}: %{x:.3f} €/MWh<extra></extra>",
            })
        data.append({
            "type": "scatter", "mode": "markers", "name": "German load-weighted average",
            "y": self.labels(), "x": C._r([m.de_avg_price for m in self.M.values()], 3),
            "marker": {"color": "@ink", "size": 14, "symbol": "line-ns-open", "line": {"width": 2.5, "color": "@ink"}},
            "hovertemplate": "German average: %{x:.3f} €/MWh<extra></extra>",
        })
        f3 = self.fig("cmp-zoneprices", C.figure(data, {
            "xaxis": {"title": {"text": "mean zonal price [€/MWh]"}},
            "yaxis": {"automargin": True, "autorange": "reversed"}, "showlegend": False,
            "hovermode": "closest"}, height=340))
        cc = [m.consumer_cost.sum() / 1e9 for m in self.M.values()]
        f4 = self.single("cmp-consumer", cc, "bn€", 2)
        rent = []
        for m in self.M.values():
            e = m.exchange
            rent.append({kind: e[e.kind == kind].rent_meur.sum() for kind in ["DE internal", "DE border"]})
        rent = pd.DataFrame(rent, index=self.labels())
        f5 = self.fig("cmp-rent", C.bars(self.labels(), [
            {"name": "Between German zones", "values": rent["DE internal"].values, "color": "@s1"},
            {"name": "German borders", "values": rent["DE border"].values, "color": "@s2"}],
            unit="M€", stacked=True, height=300))
        body = H.grid(
            H.fig_block(f3, "Mean price of every German zone",
                        "Each dot is one zone (hover for its name); the black tick is the load-weighted German "
                        "average. Dots far apart = price divergence. Note the narrow axis: all differences "
                        "are below 1 €/MWh."),
            H.fig_block(f2, "Hours with a single German price", "Spread between German zones < 1 €/MWh."),
            H.fig_block(f4, "Wholesale cost of German electricity demand"),
            H.fig_block(f5, "Congestion rent on corridors",
                        "|flow| × |price difference|, summed over corridors between German zones and "
                        "at the German borders.", wide=True),
        )
        self.add("prices", "Prices & market integration",
                 "Does splitting Germany create meaningful price signals?", body)

    def redispatch(self):
        ups = [m.rd_total["up_de_twh"] for m in self.M.values()]
        dns = [-m.rd_total["down_de_twh"] for m in self.M.values()]
        f1 = self.fig("cmp-rd", C.bars(self.labels(), [
            {"name": "Up", "values": ups, "color": UP},
            {"name": "Down", "values": dns, "color": DOWN}], unit="TWh", stacked=True, height=320))
        f2 = self.single("cmp-rdcost", [m.rd_total["cost_meur"] for m in self.M.values()], "M€", 0)
        cm = [m.curt_total["market_de_twh"] for m in self.M.values()]
        cg = [m.curt_total["grid_de_twh"] for m in self.M.values()]
        f3 = self.fig("cmp-curt", C.bars(self.labels(), [
            {"name": "Market", "values": cm, "color": STAGE_COLOR["market"]},
            {"name": "After redispatch", "values": cg, "color": STAGE_COLOR["grid"]}], unit="TWh", height=320))
        # Redispatch by technology across configurations (down-regulation)
        grp = pd.DataFrame({SCENARIO_LABELS.get(k, k): m.rd_by_group.get("down", pd.Series(dtype=float))
                            for k, m in self.M.items()}).reindex(GEN_GROUP_ORDER).fillna(0)
        grp = grp[grp.abs().sum(axis=1) > 0.01]
        f4 = self.fig("cmp-rdgroup", C.bars(self.labels(), [
            {"name": g, "values": (-grp.loc[g]).values, "color": GEN_GROUP_COLOR[g]} for g in grp.index],
            unit="TWh", stacked=True, height=340))
        body = H.grid(
            H.fig_block(f1, "Redispatch volume in Germany"),
            H.fig_block(f2, "Redispatch cost (all units)"),
            H.fig_block(f4, "German down-regulation by technology"),
            H.fig_block(f3, "Wind & solar curtailment in Germany"),
        )
        self.add("redispatch", "Redispatch & curtailment",
                 "Zone borders that follow grid bottlenecks should move congestion management from "
                 "redispatch into the market.", body)

    def grid(self):
        lt = [m.lines_total for m in self.M.values()]
        f1 = self.single("cmp-gwkm", [x["ac_exp_gwkm_de"] for x in lt], "GW·km", 0)
        f2 = self.fig("cmp-inv", C.bars(self.labels(), [
            {"name": "AC lines", "values": [x["ac_invest_meur"] for x in lt], "color": "@s1"},
            {"name": "DC links", "values": [x["dc_invest_meur"] for x in lt], "color": "@s2"}],
            unit="M€/a", stacked=True, height=300))
        data = []
        for k, m in self.M.items():
            L = m.lines[m.lines.category != "Abroad"]
            data.append({"type": "box", "name": SCENARIO_LABELS.get(k, k), "y": C._r(L.mean_loading.values * 100, 2),
                         "marker": {"color": SCENARIO_COLOR[k], "size": 4}, "line": {"color": SCENARIO_COLOR[k], "width": 1.5},
                         "boxpoints": "outliers", "hovertemplate": "%{y:.1f} %<extra></extra>"})
        f3 = self.fig("cmp-loadbox", C.figure(data, {"yaxis": {"title": {"text": "mean loading [%]"}},
                                                     "showlegend": False}, height=320))
        blocks = [H.fig_block(f1, "AC grid expansion in Germany"),
                  H.fig_block(f2, "Annualised grid investment"),
                  H.fig_block(f3, "Mean loading of German lines", "Box = interquartile range, line = median.",
                              table=False)]
        sq = self.M.get("status_quo")
        if sq is not None:
            for k, m in self.M.items():
                if k == "status_quo":
                    continue
                L = m.lines[m.lines.category != "Abroad"].copy()
                dlt = L.expansion_mw - sq.lines.loc[L.index, "expansion_mw"]
                vabs = max(200.0, float(dlt.abs().quantile(0.98)))
                L["color"] = [C.div(v, vabs) for v in dlt]
                L["d"] = dlt
                L["hover"] = [f"<b>Line {i}</b><br>Δ expansion vs status quo {v:+,.0f} MW" for i, v in zip(L.index, dlt)]
                mp = base_map(m, geo.MAP_EXTENT["de"], height=440)
                L["w"] = dlt.abs()
                mp.segments(L, "color", "w", "hover", wmin=1.0, wmax=7)
                mp.colorbar(-vabs, vabs, "Δ MW", "@divscale")
                spec = mp.spec()
                spec["layout"]["showlegend"] = False
                fid = self.fig(f"cmp-dexp-{k}", spec)
                blocks.append(H.fig_block(fid, f"{SCENARIO_LABELS.get(k, k)}: change in line expansion",
                                          "Red = more expansion than in the status quo, blue = less. "
                                          "Zone borders as thin lines.", table=False))
        self.add("grid", "Grid expansion & loading",
                 "All runs share the same clustered grid, so line-by-line differences come only from the "
                 "bidding-zone design.", H.grid(*blocks))

    def flex(self):
        rows = []
        for k, m in self.M.items():
            ez = m.ely_by_zone.gw
            sh = ez.get(m.focus_zone, 0.0)
            rows.append({"sh": sh, "other": ez.sum() - sh})
        df = pd.DataFrame(rows, index=self.labels())
        f1 = self.fig("cmp-ely", C.bars(self.labels(), [
            {"name": "Zone containing SH", "values": df.sh.values, "color": "@s3"},
            {"name": "Other German zones", "values": df.other.values, "color": "@s1"}],
            unit="GW", stacked=True, height=320))
        bat = [m.bat_by_zone[["existing_gw", "new_gw"]].sum().sum() for m in self.M.values()]
        f2 = self.single("cmp-bat", bat, "GW", 1)
        fs = pd.DataFrame({SCENARIO_LABELS.get(k, k): m.flex_summary[("market", "twh")] for k, m in self.M.items()})
        fs = fs.reindex([o for o in FLEX_ORDER if o in fs.index])
        f3 = self.fig("cmp-flex", C.bars(list(fs.index), [
            {"name": c, "values": fs[c].values, "color": SCENARIO_COLOR[k]}
            for c, k in zip(fs.columns, self.M)], unit="TWh", horizontal=True, height=420))
        body = H.grid(
            H.fig_block(f1, "Electrolyser capacity in Germany",
                        "Split into the zone that contains Schleswig-Holstein and the rest of Germany."),
            H.fig_block(f2, "Battery power in Germany (grid model)"),
            H.fig_block(f3, "Use of flexibility options in the market (Germany)", wide=True),
        )
        self.add("flexibility", "Flexibility, electrolysis & batteries",
                 "Investment and dispatch of flexible assets respond to zonal prices, so this is where the "
                 "zone design has an effect even when prices barely diverge.", body)

    def costs(self):
        sq = self.M.get("status_quo")
        comps = list(next(iter(self.M.values())).costs.index)
        f1 = self.fig("cmp-total", C.bars(self.labels(), [
            {"name": "Excluding gas imports", "values": [m.kpis()["cost_excl_gas_beur"] for m in self.M.values()], "color": "@s1"},
            {"name": "Natural-gas imports", "values": [m.costs["Natural-gas imports (CH4_NG)"] / 1e3 for m in self.M.values()], "color": "@s2"}],
            unit="bn€/a", stacked=True, height=320))
        blocks = [H.fig_block(f1, "Annual system cost",
                              "Natural-gas imports are stacked on top because they mostly reflect the "
                              "market model's gas-storage use (see the data notes), not the zone design.")]
        ex = [m.kpis()["cost_excl_gas_beur"] for m in self.M.values()]
        f0 = self.single("cmp-total-ex", [v - ex[0] for v in ex], "bn€/a vs " + self.labels()[0], 2)
        blocks.append(H.fig_block(f0, "Cost excluding gas imports, relative to the first configuration"))
        if sq is not None:
            others = [k for k in self.M if k != "status_quo"]
            dl = pd.DataFrame({k: self.M[k].costs - sq.costs for k in others}).reindex(
                [c for c in comps if c != "Natural-gas imports (CH4_NG)"])
            dl = dl[dl.abs().max(axis=1) > 1]
            f2 = self.fig("cmp-costdelta", C.bars(list(dl.index), [
                {"name": SCENARIO_LABELS.get(k, k), "values": dl[k].values, "color": SCENARIO_COLOR[k]}
                for k in others], unit="M€/a vs status quo", horizontal=True, height=120 + 70 * len(dl)))
            blocks.append(H.fig_block(f2, "Cost changes against the status quo",
                                      "Positive = more expensive than one DE/LU zone. Natural-gas imports "
                                      "are left out; they are shown in the chart above.", wide=True))
        tb = pd.DataFrame({SCENARIO_LABELS.get(k, k): m.costs for k, m in self.M.items()}).rename_axis("Component [M€/a]")
        tb.loc["Total"] = tb.sum()
        blocks.append(H.table_block(tb, "Cost breakdown", formats={c: (lambda v: H.fmt(v, 0)) for c in tb.columns}))
        self.add("costs", "Costs", "Annual system cost and its components. The market dispatch "
                 "covers the whole model area and dominates the total; small relative differences "
                 "there can outweigh the other components.", H.grid(*blocks))

    def focus(self):
        keys = [("zone", "Zone containing SH", None), ("zone_price", "Zonal price [€/MWh]", 2),
                ("nodal_price", "Mean nodal price [€/MWh]", 2), ("curt_market_twh", "Curtailment market [TWh]", 2),
                ("curt_grid_twh", "Curtailment after redispatch [TWh]", 2), ("rd_down_twh", "Redispatch down [TWh]", 2),
                ("rd_up_twh", "Redispatch up [TWh]", 2), ("ely_gw", "Electrolysers [GW]", 2),
                ("ely_twh", "Electrolysis [TWh]", 2), ("bat_gw", "Batteries [GW]", 2),
                ("line_exp_gwkm", "Line expansion touching SH [GW·km]", 0)]
        tb = pd.DataFrame({SCENARIO_LABELS.get(k, k): {lab: m.focus[key] for key, lab, _ in keys}
                           for k, m in self.M.items()})
        tb.index.name = "Schleswig-Holstein"
        fm = {}
        rows = []
        for key, lab, dg in keys:
            rows.append([lab] + [
                (H.fmt(m.focus[key], dg) if dg is not None else str(m.focus[key])) for m in self.M.values()])
        tbl = pd.DataFrame(rows, columns=["Schleswig-Holstein"] + self.labels()).set_index("Schleswig-Holstein")
        f1 = self.fig("cmp-sh-curt", C.bars(self.labels(), [
            {"name": "Curtailment after redispatch", "values": [m.focus["curt_grid_twh"] for m in self.M.values()], "color": "@s2"},
            {"name": "Redispatch down", "values": [m.focus["rd_down_twh"] for m in self.M.values()], "color": DOWN}],
            unit="TWh", height=320))
        f2 = self.fig("cmp-sh-ely", C.bars(self.labels(), [
            {"name": "Electrolysers", "values": [m.focus["ely_gw"] for m in self.M.values()], "color": "@s3"},
            {"name": "Batteries", "values": [m.focus["bat_gw"] for m in self.M.values()], "color": "@s1"}],
            unit="GW", height=320))
        body = H.grid(H.fig_block(f1, "Grid-related curtailment in SH"),
                      H.fig_block(f2, "Flexible assets in SH"),
                      H.table_block(tbl, "Schleswig-Holstein indicators"))
        self.add("focus", "Schleswig-Holstein", "How the region fares under each design.", body)
