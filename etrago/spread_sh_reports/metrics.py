"""Compute report metrics for one market-zone configuration.

Conventions
-----------
* Energies are snapshot-weighted (5-h steps -> full year), MWh unless noted.
* Market results come from ``market_optimization`` (rolling-horizon UC
  dispatch), network results from ``grid_optimization`` (redispatch + grid
  expansion), investments in flexibility from ``pre_market_optimization``.
* Redispatch, cost and expansion formulas mirror
  ``etrago.analyze.calc_results``.
"""

import json
import logging

import numpy as np
import pandas as pd

from . import geo
from .config import (
    CONGESTION_SHARE, CURTAIL_GROUPS, FLEX_OPTIONS, GEN_GROUPS, HIGH_LOADING,
    POWER_LINKS, VRES_CARRIERS,
)
from .loader import Stage

logger = logging.getLogger(__name__)

TWH = 1e6
MEUR = 1e6


def _wmean(values, weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    return float((np.asarray(values, dtype=float) * weights).sum() / total) if total else np.nan


def _zone_sort_key(zone):
    if zone == "DE/LU":
        return (0, 0, zone)
    if "-Z" in zone:
        return (0, int(zone.split("-Z")[1]), zone)
    return (1, 0, zone)


class ScenarioMetrics:
    """All numbers behind one report. Attributes are plain pandas objects."""

    def __init__(self, scenario, data_dir):
        self.sc = scenario
        self.data_dir = data_dir
        self.pre = Stage(scenario.stage("pre_market_optimization"))
        self.mkt = Stage(scenario.stage("market_optimization"))
        self.grid = Stage(scenario.stage("grid_optimization"))
        self.notes = []
        self.args = json.loads((scenario.path / "args.json").read_text())

    # ------------------------------------------------------------------
    # Topology & zones
    # ------------------------------------------------------------------
    def build_topology(self):
        self.zone_gdf = geo.zones(self.data_dir, self.sc.key)
        self.zone_label = dict(zip(self.zone_gdf.zone, self.zone_gdf.label))

        gb = self.grid.static("buses")
        self.grid_buses = gb
        ac = gb[gb.carrier == "AC"].copy()
        ac["zone"] = geo.assign_zones(ac, self.zone_gdf, self.sc.key)
        self.ac = ac
        self.bus_zone = ac["zone"]

        focus_file = self.sc.path / "busmap_within_focus.csv"
        if focus_file.exists():
            fm = pd.read_csv(focus_file)
            self.focus_buses = set(fm["cluster"].astype(str)) & set(ac.index)
        else:
            self.focus_buses = set()
        ac["focus"] = ac.index.isin(self.focus_buses)

        # Market AC bus -> zone by majority vote over shared generators.
        mb = self.mkt.static("buses")
        mac = mb[mb.carrier == "AC"]
        mg = self.mkt.static("generators")
        gg = self.grid.static("generators")
        common = mg.index.intersection(gg.index)
        votes = pd.DataFrame({
            "mbus": mg.loc[common, "bus"],
            "zone": gg.loc[common, "bus"].map(self.bus_zone),
        }).dropna()
        vote = votes.groupby("mbus")["zone"].agg(lambda s: s.value_counts().idxmax())
        mzone = {}
        for bus, row in mac.iterrows():
            if bus in vote.index:
                mzone[bus] = vote[bus]
            elif self.sc.key == "status_quo" and row.country in ("DE", "LU"):
                mzone[bus] = "DE/LU"
            else:
                mzone[bus] = row.country
        self.mkt_zone = pd.Series(mzone)
        dup = self.mkt_zone[self.mkt_zone.duplicated(keep=False)]
        if len(dup):
            self.notes.append(
                f"Several market buses map to the same zone: {dup.to_dict()}"
            )
        mislabel = [
            b for b in mac.index
            if mac.loc[b, "country"] == "LU" and geo.is_german_zone(self.mkt_zone[b])
        ]
        if mislabel:
            self.notes.append(
                "The market bus labelled 'LU' in the export carries the German "
                f"zone {self.zone_label.get(self.mkt_zone[mislabel[0]], self.mkt_zone[mislabel[0]])} "
                "(Luxembourg lies inside that zone polygon); it is reported "
                "under the zone name."
            )

        zones = sorted(set(self.mkt_zone.values) | set(self.bus_zone.values),
                       key=_zone_sort_key)
        self.zones = zones
        self.de_zones = [z for z in zones if geo.is_german_zone(z)]
        self.focus_zone = next(
            (z for z, f in zip(self.zone_gdf.zone, self.zone_gdf.contains_focus) if f),
            self.de_zones[0],
        )
        for z in zones:
            self.zone_label.setdefault(z, z)

    def label(self, zone):
        return self.zone_label.get(zone, zone)

    def _ac_side(self, links):
        """AC bus of each link (bus0 if AC, else bus1)."""
        ac_set = set(self.ac.index) | set(self.mkt_zone.index)
        bus0_ac = links["bus0"].isin(ac_set)
        return links["bus0"].where(bus0_ac, links["bus1"])

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------
    def compute_prices(self):
        mkt = self.mkt
        w = mkt.weights
        mp = mkt.series("buses", "marginal_price")[list(self.mkt_zone.index)]
        prices = mp.T.groupby(self.mkt_zone).mean().T
        prices = prices[[z for z in self.zones if z in prices]]
        self.prices = prices

        loads = mkt.static("loads")
        ac_loads = loads[(loads.carrier == "AC") & loads.bus.isin(self.mkt_zone.index)]
        lp = mkt.series("loads", "p")
        lp = lp.reindex(columns=ac_loads.index, fill_value=0.0)
        zone_load = lp.T.groupby(ac_loads.bus.map(self.mkt_zone)).sum().T
        self.zone_load = zone_load.reindex(columns=prices.columns, fill_value=0.0)

        rows = []
        for z in prices.columns:
            p = prices[z]
            ld = self.zone_load[z]
            rows.append({
                "zone": z,
                "label": self.label(z),
                "german": geo.is_german_zone(z),
                "mean": _wmean(p, w),
                "load_weighted": _wmean(p, w * ld) if ld.sum() > 0 else np.nan,
                "p05": p.quantile(0.05),
                "p95": p.quantile(0.95),
                "max": p.max(),
                "std": p.std(),
                "low_price_share": _wmean(p <= 1.0, w),
                "demand_twh": (ld * w).sum() / TWH,
            })
        self.price_stats = pd.DataFrame(rows).set_index("zone")

        de = prices[self.de_zones]
        if len(self.de_zones) > 1:
            spread = de.max(axis=1) - de.min(axis=1)
        else:
            spread = pd.Series(0.0, index=prices.index)
        self.de_spread = spread
        self.price_convergence = _wmean(spread < 1.0, w)
        self.price_monthly = prices.groupby(prices.index.month).apply(
            lambda df: pd.Series({c: _wmean(df[c], w.loc[df.index]) for c in df})
        )
        self.price_hourly = de.groupby(de.index.hour).mean()

        # Nodal shadow prices of the grid optimisation (AC buses).
        gp = self.grid.series("buses", "marginal_price")
        cols = [b for b in self.ac.index if b in gp]
        gw = self.grid.weights
        self.nodal_price = pd.Series(
            {b: _wmean(gp[b], gw) for b in cols}
        )

        de_load = self.zone_load[self.de_zones]
        cost = (de * de_load).mul(w, axis=0).sum()
        self.consumer_cost = cost  # EUR per zone
        self.de_avg_price = cost.sum() / (de_load.mul(w, axis=0).sum().sum())

    # ------------------------------------------------------------------
    # Zonal balance (market)
    # ------------------------------------------------------------------
    def compute_balance(self):
        mkt = self.mkt
        w = mkt.weights
        gens = mkt.static("generators")
        el = gens[gens.carrier.isin(GEN_GROUPS) & gens.bus.isin(self.mkt_zone.index)]
        e = mkt.energy(mkt.series("generators", "p").reindex(columns=el.index, fill_value=0))
        df = pd.DataFrame({
            "zone": el.bus.map(self.mkt_zone),
            "group": el.carrier.map(GEN_GROUPS),
            "twh": e / TWH,
        })
        links = mkt.static("links")
        pl = links[links.carrier.isin(POWER_LINKS) & links.bus1.isin(self.mkt_zone.index)]
        if len(pl):
            p1 = mkt.series("links", "p1").reindex(columns=pl.index, fill_value=0)
            le = mkt.energy(-p1)
            df = pd.concat([df, pd.DataFrame({
                "zone": pl.bus1.map(self.mkt_zone),
                "group": pl.carrier.map(GEN_GROUPS),
                "twh": le / TWH,
            })])
        self.generation = df.groupby(["zone", "group"])["twh"].sum().unstack(fill_value=0)

        demand = self.zone_load.mul(w, axis=0).sum() / TWH
        gen_total = self.generation.sum(axis=1).reindex(demand.index, fill_value=0)
        vres = self.generation.reindex(
            columns=["Wind onshore", "Wind offshore", "Solar"], fill_value=0
        ).sum(axis=1).reindex(demand.index, fill_value=0)
        self.balance = pd.DataFrame({
            "label": [self.label(z) for z in demand.index],
            "demand_twh": demand,
            "generation_twh": gen_total,
            "vres_twh": vres,
            "vres_share": vres / demand.replace(0, np.nan),
        })

    # ------------------------------------------------------------------
    # Cross-zonal exchange (market)
    # ------------------------------------------------------------------
    def compute_exchange(self):
        mkt = self.mkt
        w = mkt.weights
        links = mkt.static("links")
        ic = links[links.bus0.isin(self.mkt_zone.index) & links.bus1.isin(self.mkt_zone.index)].copy()
        ic["zone0"] = ic.bus0.map(self.mkt_zone)
        ic["zone1"] = ic.bus1.map(self.mkt_zone)
        ic = ic[ic.zone0 != ic.zone1]
        cap = ic["p_nom_opt"].fillna(ic["p_nom"])
        ic = ic[cap > 1.0]
        cap = cap.loc[ic.index]
        p0 = mkt.series("links", "p0").reindex(columns=ic.index, fill_value=0)
        pmax = mkt.dense("links", "p_max_pu", columns=ic.index)
        pmin = mkt.dense("links", "p_min_pu", columns=ic.index)

        pairs = []
        flows = {}
        prices = self.prices
        for (a, b), grp in ic.groupby(ic.apply(lambda r: tuple(sorted((r.zone0, r.zone1))), axis=1)):
            sign = np.where(grp.zone0 == a, 1.0, -1.0)
            f = (p0[grp.index] * sign).sum(axis=1)  # + = a -> b
            fwd_cap = pd.DataFrame({
                i: (pmax[i] if s > 0 else -pmin[i]) * cap[i]
                for i, s in zip(grp.index, sign)
            }).sum(axis=1)
            bwd_cap = pd.DataFrame({
                i: (-pmin[i] if s > 0 else pmax[i]) * cap[i]
                for i, s in zip(grp.index, sign)
            }).sum(axis=1)
            util = np.where(f >= 0, f / fwd_cap.replace(0, np.nan),
                            -f / bwd_cap.replace(0, np.nan))
            util = pd.Series(util, index=f.index).fillna(0).clip(0, 1)
            dp = (prices[b] - prices[a]) if (a in prices and b in prices) else 0.0
            rent = (f.abs() * (dp.abs() if not np.isscalar(dp) else 0) * w).sum()
            flows[(a, b)] = f
            pairs.append({
                "zone_a": a, "zone_b": b,
                "label": f"{self.label(a)} ↔ {self.label(b)}",
                "kind": ("DE internal" if geo.is_german_zone(a) and geo.is_german_zone(b)
                         else "DE border" if geo.is_german_zone(a) or geo.is_german_zone(b)
                         else "Abroad"),
                "carriers": ", ".join(sorted(set(grp.carrier))),
                "capacity_mw": float(fwd_cap.mean()),
                "a_to_b_twh": float((f.clip(lower=0) * w).sum() / TWH),
                "b_to_a_twh": float((-f.clip(upper=0) * w).sum() / TWH),
                "mean_util": _wmean(util, w),
                "congested_share": _wmean(util >= CONGESTION_SHARE, w),
                "rent_meur": rent / MEUR,
                "mean_abs_price_diff": _wmean(np.abs(dp), w) if not np.isscalar(dp) else 0.0,
            })
        self.exchange = pd.DataFrame(pairs)
        self.exchange_flows = flows

        net = pd.Series(0.0, index=self.zones)
        for r in pairs:
            net[r["zone_a"]] += r["a_to_b_twh"] - r["b_to_a_twh"]
            net[r["zone_b"]] -= r["a_to_b_twh"] - r["b_to_a_twh"]
        self.net_position = net  # + = net exporter (TWh)
        if len(self.balance):
            self.balance["net_export_twh"] = net.reindex(self.balance.index).values

    # ------------------------------------------------------------------
    # Redispatch (grid)
    # ------------------------------------------------------------------
    def compute_redispatch(self):
        g = self.grid
        w = g.weights
        gens = g.static("generators")
        ramp_g = gens[gens.index.str.contains("ramp")]
        p = g.series("generators", "p").reindex(columns=ramp_g.index, fill_value=0)
        mc = g.dense("generators", "marginal_cost", columns=ramp_g.index)
        energy_g = g.energy(p)
        cost_g = (p * mc).mul(w, axis=0).sum()

        links = g.static("links")
        ramp_l = links[links.index.str.contains("ramp")]
        p0 = g.series("links", "p0").reindex(columns=ramp_l.index, fill_value=0)
        p1 = g.series("links", "p1").reindex(columns=ramp_l.index, fill_value=0)
        lmc = g.dense("links", "marginal_cost", columns=ramp_l.index)
        energy_l = g.energy(-p1)
        cost_l = (p0 * lmc).mul(w, axis=0).sum()

        units = pd.concat([
            pd.DataFrame({"carrier": ramp_g.carrier, "bus": ramp_g.bus,
                          "mwh": energy_g, "cost": cost_g}),
            pd.DataFrame({"carrier": ramp_l.carrier, "bus": ramp_l.bus1,
                          "mwh": energy_l, "cost": cost_l}),
        ])
        units["direction"] = np.where(units.index.str.contains("ramp_up"), "up", "down")
        units["group"] = units.carrier.map(GEN_GROUPS).fillna("Coal, oil & other")
        units["zone"] = units.bus.map(self.bus_zone)
        units["german"] = units.zone.map(geo.is_german_zone).fillna(False).astype(bool)
        units["focus"] = units.bus.isin(self.focus_buses)
        self.rd_units = units

        ts_up = pd.concat([p[ramp_g.index[ramp_g.index.str.contains("ramp_up")]],
                           -p1[ramp_l.index[ramp_l.index.str.contains("ramp_up")]]], axis=1)
        ts_dn = pd.concat([p[ramp_g.index[ramp_g.index.str.contains("ramp_down")]],
                           -p1[ramp_l.index[ramp_l.index.str.contains("ramp_down")]]], axis=1)
        de_cols = units.index[units.german]
        up_de = ts_up[ts_up.columns.intersection(de_cols)].sum(axis=1)
        dn_de = ts_dn[ts_dn.columns.intersection(de_cols)].sum(axis=1)
        self.rd_ts = pd.DataFrame({"up": up_de, "down": dn_de})
        weekly = self.rd_ts.mul(w, axis=0).resample("W").sum() / TWH
        self.rd_weekly = weekly
        self.rd_monthly = self.rd_ts.mul(w, axis=0).groupby(self.rd_ts.index.month).sum() / TWH

        de = units[units.german]
        self.rd_by_group = (de.groupby(["direction", "group"])["mwh"].sum().unstack(0, fill_value=0) / TWH)
        self.rd_by_zone = (units.assign(zone=units.zone.where(units.german, "Abroad"))
                           .groupby(["zone", "direction"])["mwh"].sum().unstack(fill_value=0) / TWH)
        self.rd_cost_by_zone = (units.assign(zone=units.zone.where(units.german, "Abroad"))
                                .groupby("zone")["cost"].sum() / MEUR)
        node = de.groupby(["bus", "direction"])["mwh"].sum().unstack(fill_value=0) / TWH
        node = node.reindex(columns=["up", "down"], fill_value=0)
        node["volume"] = node["up"] - node["down"]
        node["net"] = node["up"] + node["down"]
        self.rd_nodes = node
        self.rd_total = {
            "up_de_twh": de[de.direction == "up"].mwh.sum() / TWH,
            "down_de_twh": -de[de.direction == "down"].mwh.sum() / TWH,
            "up_abroad_twh": units[~units.german & (units.direction == "up")].mwh.sum() / TWH,
            "down_abroad_twh": -units[~units.german & (units.direction == "down")].mwh.sum() / TWH,
            "cost_meur": units.cost.sum() / MEUR,
            "cost_de_meur": de.cost.sum() / MEUR,
            "up_focus_twh": de[de.focus & (de.direction == "up")].mwh.sum() / TWH,
            "down_focus_twh": -de[de.focus & (de.direction == "down")].mwh.sum() / TWH,
        }

    # ------------------------------------------------------------------
    # Curtailment of variable renewables
    # ------------------------------------------------------------------
    def compute_curtailment(self):
        mkt, g = self.mkt, self.grid
        w = mkt.weights
        mg = mkt.static("generators")
        gg = g.static("generators")
        vres = mg[mg.carrier.isin(VRES_CARRIERS)].index.intersection(gg.index)
        avail = mkt.dense("generators", "p_max_pu", columns=vres).mul(mg.loc[vres, "p_nom"], axis=1)
        p_mkt = mkt.series("generators", "p").reindex(columns=vres, fill_value=0)
        gp = g.series("generators", "p")
        final = gp.reindex(columns=vres, fill_value=0)
        for suffix in (" ramp_up", " ramp_down"):
            cols = [c + suffix for c in vres]
            final = final + gp.reindex(columns=cols, fill_value=0).values
        curt_m = (avail - p_mkt).clip(lower=0)
        curt_g = (avail - final).clip(lower=0)

        info = pd.DataFrame({
            "carrier": gg.loc[vres, "carrier"],
            "group": gg.loc[vres, "carrier"].map(CURTAIL_GROUPS),
            "bus": gg.loc[vres, "bus"],
            "p_nom": mg.loc[vres, "p_nom"],
        })
        info["zone"] = info.bus.map(self.bus_zone)
        info["german"] = info.zone.map(geo.is_german_zone).fillna(False).astype(bool)
        info["focus"] = info.bus.isin(self.focus_buses)
        info["avail_twh"] = mkt.energy(avail) / TWH
        info["curt_market_twh"] = mkt.energy(curt_m) / TWH
        info["curt_grid_twh"] = g.energy(curt_g) / TWH
        self.curt_units = info

        de = info[info.german]
        self.curt_by_group = de.groupby("group")[["avail_twh", "curt_market_twh", "curt_grid_twh"]].sum()
        self.curt_by_zone = info.assign(zone=info.zone.where(info.german, "Abroad")).groupby("zone")[
            ["avail_twh", "curt_market_twh", "curt_grid_twh"]].sum()
        self.curt_nodes = de.groupby("bus")[["avail_twh", "curt_market_twh", "curt_grid_twh"]].sum()
        de_cols = de.index
        self.curt_ts = pd.DataFrame({
            "market": curt_m[de_cols].sum(axis=1),
            "grid": curt_g[de_cols].sum(axis=1),
            "available": avail[de_cols].sum(axis=1),
        })
        self.curt_monthly = self.curt_ts.mul(w, axis=0).groupby(self.curt_ts.index.month).sum() / TWH
        a = de.avail_twh.sum()
        self.curt_total = {
            "avail_de_twh": a,
            "market_de_twh": de.curt_market_twh.sum(),
            "grid_de_twh": de.curt_grid_twh.sum(),
            "market_de_rate": de.curt_market_twh.sum() / a if a else np.nan,
            "grid_de_rate": de.curt_grid_twh.sum() / a if a else np.nan,
            "grid_focus_twh": de[de.focus].curt_grid_twh.sum(),
            "market_focus_twh": de[de.focus].curt_market_twh.sum(),
            "avail_focus_twh": de[de.focus].avail_twh.sum(),
        }

    # ------------------------------------------------------------------
    # Flexibility, batteries, electrolysis
    # ------------------------------------------------------------------
    def _flex_stage(self, stage, zone_map):
        w = stage.weights
        rows, profiles, units = [], {}, []
        for label, comp, carriers, measure in FLEX_OPTIONS:
            static = stage.static(comp)
            if static.empty:
                continue
            sel = static[static.carrier.isin(carriers) & ~static.index.str.contains("ramp")]
            if sel.empty:
                continue
            if comp == "storage_units":
                bus = sel.bus
                act = stage.series(comp, "p_dispatch").reindex(columns=sel.index, fill_value=0)
                cap = sel.p_nom_opt
            else:
                bus = self._ac_side(sel)
                if measure == "output":
                    act = -stage.series(comp, "p1").reindex(columns=sel.index, fill_value=0)
                else:
                    act = stage.series(comp, "p0").reindex(columns=sel.index, fill_value=0).clip(lower=0)
                cap = sel.p_nom_opt.fillna(sel.p_nom)
            zone = bus.map(zone_map)
            german = zone.map(geo.is_german_zone).fillna(False).astype(bool)
            e = stage.energy(act)
            df = pd.DataFrame({"option": label, "bus": bus, "zone": zone,
                               "german": german, "twh": e / TWH, "gw": cap / 1e3})
            units.append(df)
            de_cols = sel.index[german.values]
            profiles[label] = act[de_cols].sum(axis=1)
        units = pd.concat(units) if units else pd.DataFrame()
        return units, pd.DataFrame(profiles)

    def compute_flexibility(self):
        self.flex_grid_units, self.flex_grid_ts = self._flex_stage(self.grid, self.bus_zone)
        mkt_map = pd.concat([self.bus_zone, self.mkt_zone])
        mkt_map = mkt_map[~mkt_map.index.duplicated()]
        self.flex_mkt_units, self.flex_mkt_ts = self._flex_stage(self.mkt, mkt_map)

        def summary(units):
            de = units[units.german]
            return de.groupby("option")[["twh", "gw"]].sum()

        self.flex_summary = pd.concat(
            {"market": summary(self.flex_mkt_units), "grid": summary(self.flex_grid_units)}, axis=1
        )
        g = self.flex_grid_units
        self.flex_by_zone = g[g.german].groupby(["zone", "option"])["twh"].sum().unstack(fill_value=0)
        self.flex_cap_by_zone = g[g.german].groupby(["zone", "option"])["gw"].sum().unstack(fill_value=0)
        ts = self.flex_mkt_ts
        self.flex_hourly = ts.groupby(ts.index.hour).mean() / 1e3  # GW

        # Price capture in the market: price paid for consumption vs average.
        mkt = self.mkt
        w = mkt.weights
        rows = []
        su = mkt.static("storage_units")
        links = mkt.static("links")
        for z in self.de_zones:
            price = self.prices[z]
            buses = self.mkt_zone.index[self.mkt_zone == z]
            row = {"zone": z, "label": self.label(z), "avg_price": _wmean(price, w)}
            ely = links[(links.carrier == "power_to_H2") & links.bus0.isin(buses)].index
            if len(ely):
                c = mkt.series("links", "p0").reindex(columns=ely, fill_value=0).clip(lower=0).sum(axis=1)
                row["ely_price"] = _wmean(price, w * c) if c.sum() > 0 else np.nan
                cap = links.loc[ely, "p_nom_opt"].sum()
                row["ely_flh"] = (c * w).sum() / cap if cap > 0 else np.nan
            bat = su[(su.carrier == "battery") & su.bus.isin(buses)].index
            if len(bat):
                ch = mkt.series("storage_units", "p_store").reindex(columns=bat, fill_value=0).sum(axis=1)
                dis = mkt.series("storage_units", "p_dispatch").reindex(columns=bat, fill_value=0).sum(axis=1)
                row["bat_charge_price"] = _wmean(price, w * ch) if ch.sum() > 0 else np.nan
                row["bat_discharge_price"] = _wmean(price, w * dis) if dis.sum() > 0 else np.nan
            hp = links[links.carrier.isin(["central_heat_pump", "rural_heat_pump"]) & links.bus0.isin(buses)].index
            if len(hp):
                c = mkt.series("links", "p0").reindex(columns=hp, fill_value=0).clip(lower=0).sum(axis=1)
                row["hp_price"] = _wmean(price, w * c) if c.sum() > 0 else np.nan
            rows.append(row)
        self.price_capture = pd.DataFrame(rows).set_index("zone")

        # Electrolysis (grid stage keeps the market investment, fixed).
        gl = self.grid.static("links")
        ely = gl[gl.carrier == "power_to_H2"].copy()
        ely["zone"] = ely.bus0.map(self.bus_zone)
        ely["german"] = ely.zone.map(geo.is_german_zone).fillna(False).astype(bool)
        ely["focus"] = ely.bus0.isin(self.focus_buses)
        p0 = self.grid.series("links", "p0").reindex(columns=ely.index, fill_value=0).clip(lower=0)
        ely["twh"] = self.grid.energy(p0) / TWH
        ely["gw"] = ely.p_nom_opt / 1e3
        self.electrolysis = ely
        de = ely[ely.german]
        self.ely_by_zone = de.groupby("zone")[["gw", "twh"]].sum()
        self.ely_nodes = de.groupby("bus0")[["gw", "twh"]].sum()

        # Batteries at grid nodes.
        su = self.grid.static("storage_units")
        bat = su[su.carrier == "battery"].copy()
        bat["zone"] = bat.bus.map(self.bus_zone)
        bat["german"] = bat.zone.map(geo.is_german_zone).fillna(False).astype(bool)
        bat["focus"] = bat.bus.isin(self.focus_buses)
        pmin = bat["p_nom_min"].fillna(0) if "p_nom_min" in bat else bat["p_nom"]
        bat["existing_gw"] = np.maximum(pmin, bat["p_nom"]) / 1e3
        bat["new_gw"] = (bat.p_nom_opt / 1e3 - bat.existing_gw).clip(lower=0)
        bat["gwh"] = bat.p_nom_opt * bat.max_hours / 1e3
        dis = self.grid.series("storage_units", "p_dispatch").reindex(columns=bat.index, fill_value=0)
        st = self.grid.series("storage_units", "p_store").reindex(columns=bat.index, fill_value=0)
        bat["discharge_twh"] = self.grid.energy(dis) / TWH
        bat["cycles"] = (bat.discharge_twh * 1e3 / bat.gwh.replace(0, np.nan)).fillna(0)
        self.batteries = bat
        de = bat[bat.german]
        self.bat_by_zone = de.groupby("zone")[["existing_gw", "new_gw", "gwh", "discharge_twh"]].sum()
        self.bat_nodes = de.groupby("bus")[["existing_gw", "new_gw", "gwh", "discharge_twh"]].sum()
        net = (dis[de.index].sum(axis=1) - st[de.index].sum(axis=1)) / 1e3
        self.bat_heat = net.groupby([net.index.month, net.index.hour]).mean().unstack()

    # ------------------------------------------------------------------
    # Lines: expansion and loading
    # ------------------------------------------------------------------
    def compute_lines(self):
        g = self.grid
        w = g.weights
        lines = g.static("lines").copy()
        b = self.grid_buses
        lines["x0"] = lines.bus0.map(b.x)
        lines["y0"] = lines.bus0.map(b.y)
        lines["x1"] = lines.bus1.map(b.x)
        lines["y1"] = lines.bus1.map(b.y)
        lines["zone0"] = lines.bus0.map(self.bus_zone)
        lines["zone1"] = lines.bus1.map(self.bus_zone)
        de0 = lines.zone0.map(geo.is_german_zone).fillna(False)
        de1 = lines.zone1.map(geo.is_german_zone).fillna(False)
        lines["category"] = np.select(
            [de0 & de1 & (lines.zone0 == lines.zone1), de0 & de1, de0 | de1],
            ["Within zone", "Between DE zones", "Cross-border"], "Abroad",
        )
        lines["focus"] = lines.bus0.isin(self.focus_buses) | lines.bus1.isin(self.focus_buses)
        smin = lines["s_nom_min"].fillna(lines["s_nom"]) if "s_nom_min" in lines else lines["s_nom"]
        lines["s_nom_base"] = smin
        lines["expansion_mw"] = (lines.s_nom_opt - smin).clip(lower=0)
        lines["expansion_rel"] = lines.expansion_mw / smin.replace(0, np.nan)
        lines["expansion_gwkm"] = lines.expansion_mw * lines.length / 1e3
        ext = lines.get("s_nom_extendable", pd.Series(True, index=lines.index)).astype(bool)
        lines["invest_meur"] = np.where(ext, lines.expansion_mw * lines.capital_cost, 0) / MEUR

        p0 = g.series("lines", "p0").reindex(columns=lines.index, fill_value=0)
        smax = g.dense("lines", "s_max_pu", columns=lines.index)
        limit = smax.mul(lines.s_nom_opt, axis=1).replace(0, np.nan)
        loading = (p0.abs() / limit).fillna(0)
        lines["mean_loading"] = loading.mul(w, axis=0).sum() / w.sum()
        lines["p95_loading"] = loading.quantile(0.95)
        lines["max_loading"] = loading.max()
        lines["congested_share"] = (loading >= CONGESTION_SHARE).mul(w, axis=0).sum() / w.sum()
        lines["high_share"] = (loading >= HIGH_LOADING).mul(w, axis=0).sum() / w.sum()
        # Loading relative to the existing capacity (before expansion)
        base_limit = smax.mul(smin, axis=1).replace(0, np.nan)
        lines["mean_loading_base"] = (p0.abs() / base_limit).fillna(0).mul(w, axis=0).sum() / w.sum()
        self.lines = lines
        self.line_loading = loading
        de_lines = lines[lines.category != "Abroad"]
        top = de_lines.sort_values("congested_share", ascending=False).head(5).index
        self.loading_duration = {
            i: np.sort(loading[i].values)[::-1] for i in top
        }

        links = g.static("links")
        dc = links[(links.carrier == "DC") & ~links.index.str.contains("ramp")].copy()
        dc["x0"] = dc.bus0.map(b.x)
        dc["y0"] = dc.bus0.map(b.y)
        dc["x1"] = dc.bus1.map(b.x)
        dc["y1"] = dc.bus1.map(b.y)
        dc["zone0"] = dc.bus0.map(self.bus_zone)
        dc["zone1"] = dc.bus1.map(self.bus_zone)
        base = dc["p_nom_min"].fillna(dc["p_nom"]) if "p_nom_min" in dc else dc["p_nom"]
        dc["p_nom_base"] = base
        dc["expansion_mw"] = (dc.p_nom_opt - base).clip(lower=0)
        dext = dc.get("p_nom_extendable", pd.Series(False, index=dc.index)).astype(bool)
        dc["invest_meur"] = np.where(dext, dc.expansion_mw * dc.capital_cost, 0) / MEUR
        dc["expansion_gwkm"] = dc.expansion_mw * dc.length / 1e3
        lp = g.series("links", "p0").reindex(columns=dc.index, fill_value=0)
        lim = dc.p_nom_opt.replace(0, np.nan)
        dl = (lp.abs() / lim).fillna(0)
        dc["mean_loading"] = dl.mul(w, axis=0).sum() / w.sum()
        dc["congested_share"] = (dl >= CONGESTION_SHARE).mul(w, axis=0).sum() / w.sum()
        self.dc = dc

        de = lines[lines.category != "Abroad"]
        self.exp_by_category = lines.groupby("category")[
            ["expansion_mw", "expansion_gwkm", "invest_meur"]].sum()
        self.exp_by_zone = de[de.category == "Within zone"].groupby("zone0")[
            ["expansion_mw", "expansion_gwkm", "invest_meur"]].sum()
        self.exp_by_voltage = de.groupby("v_nom")[["expansion_mw", "expansion_gwkm", "invest_meur"]].sum()
        self.lines_total = {
            "ac_exp_gw_de": de.expansion_mw.sum() / 1e3,
            "ac_exp_gwkm_de": de.expansion_gwkm.sum(),
            "ac_invest_meur": lines.invest_meur.sum(),
            "ac_invest_de_meur": de.invest_meur.sum(),
            "dc_exp_gw": dc.expansion_mw.sum() / 1e3,
            "dc_invest_meur": dc.invest_meur.sum(),
            "rel_exp_de": de.expansion_mw.sum() / de.s_nom_base.sum(),
            "between_zone_gw": de[de.category == "Between DE zones"].expansion_mw.sum() / 1e3,
            "focus_exp_gw": de[de.focus].expansion_mw.sum() / 1e3,
            "focus_exp_gwkm": de[de.focus].expansion_gwkm.sum(),
            "mean_loading_de": _wmean(de.mean_loading, de.length * de.s_nom_opt),
            "congested_lines_de": int((de.congested_share > 0.10).sum()),
            "n_lines_de": int(len(de)),
        }

    # ------------------------------------------------------------------
    # Costs & system indicators
    # ------------------------------------------------------------------
    def compute_costs(self):
        mkt, g = self.mkt, self.grid
        w = mkt.weights
        gens = mkt.static("generators")
        shed_idx = gens.index[gens.carrier.isin(["load shedding", "negative load shedding"])]
        gp = mkt.series("generators", "p").reindex(columns=gens.index, fill_value=0)
        gmc = mkt.dense("generators", "marginal_cost", columns=gens.index)
        gen_cost = (gp * gmc).mul(w, axis=0).sum()
        links = mkt.static("links")
        lp = mkt.series("links", "p0").reindex(columns=links.index, fill_value=0)
        lmc = mkt.dense("links", "marginal_cost", columns=links.index)
        link_cost = (lp.abs() * lmc).mul(w, axis=0).sum().sum()
        su = mkt.static("storage_units")
        su_cost = 0.0
        if "marginal_cost" in su and len(su):
            sp = mkt.series("storage_units", "p").reindex(columns=su.index, fill_value=0)
            su_cost = (sp.mul(w, axis=0).sum() * su.marginal_cost.fillna(0)).sum()
        start = 0.0
        for comp in ("generators", "links"):
            st = mkt.static(comp)
            if "start_up_cost" in st:
                su_ts = mkt.series(comp, "start_up").reindex(columns=st.index, fill_value=0)
                start += (su_ts.sum() * st.start_up_cost.fillna(0)).sum()

        # Grid-stage investments (calc_investment_cost of eTraGo).
        gl = g.static("links")
        ext_links = gl[gl.get("p_nom_extendable", False).astype(bool)
                       & (gl.carrier != "DC") & ~gl.index.str.contains("ramp")]
        link_inv = ((ext_links.p_nom_opt - ext_links.p_nom_min.fillna(0)) * ext_links.capital_cost).sum()
        gsu = g.static("storage_units")
        ext_su = gsu[gsu.get("p_nom_extendable", False).astype(bool)]
        su_inv = ((ext_su.p_nom_opt - ext_su.p_nom_min.fillna(0)) * ext_su.capital_cost).sum()
        gst = g.static("stores")
        ext_st = gst[gst.get("e_nom_extendable", False).astype(bool)]
        st_inv = (ext_st.e_nom_opt * ext_st.capital_cost).sum()
        ely = self.electrolysis
        ely_inv = (ely.p_nom_opt * ely.capital_cost).sum()

        gas_idx = gens.index[gens.carrier == "CH4_NG"]
        other_idx = gens.index.difference(shed_idx).difference(gas_idx)
        items = {
            "Market dispatch (fuel & variable)": (gen_cost[other_idx].sum() + link_cost + su_cost) / MEUR,
            "Natural-gas imports (CH4_NG)": gen_cost[gas_idx].sum() / MEUR,
            "Start-up costs": start / MEUR,
            "Load shedding": gen_cost[shed_idx].sum() / MEUR,
            "Redispatch": self.rd_total["cost_meur"],
            "AC grid expansion": self.lines_total["ac_invest_meur"],
            "DC grid expansion": self.lines_total["dc_invest_meur"],
            "Electrolysers": ely_inv / MEUR,
            "Batteries (grid stage)": su_inv / MEUR,
            "Other storage & sector links": (st_inv + link_inv) / MEUR,
        }
        self.costs = pd.Series(items)

        # Consistency of the gas storage in the rolling-horizon market export:
        # net discharge over the year should equal the drop in stored energy.
        st = mkt.static("stores")
        ch4 = st.index[st.carrier == "CH4"]
        if len(ch4):
            net = (mkt.series("stores", "p").reindex(columns=ch4, fill_value=0)
                   .mul(w, axis=0).sum().sum()) / TWH
            e = mkt.series("stores", "e").reindex(columns=ch4, fill_value=0).sum(axis=1)
            drop = (e.iloc[0] - e.iloc[-1]) / TWH
            self.gas_store_check = {"net_discharge_twh": net, "soc_drop_twh": drop}
            if abs(net - drop) > 10:
                self.notes.append(
                    f"Gas storage in the market export: the CH4 stores discharge {net:,.0f} TWh net over "
                    f"the year while their stored energy changes by only {drop:,.1f} TWh. Natural-gas "
                    "imports (and therefore market dispatch cost) absorb this difference and vary between "
                    "runs for reasons unrelated to the bidding zones; they are reported separately."
                )
        self.objectives = {
            "pre_market": self.pre.network_meta.get("objective"),
            "market": mkt.network_meta.get("objective"),
            "grid": g.network_meta.get("objective"),
            "grid_constant": g.network_meta.get("objective_constant"),
        }

        # Load shedding volumes (validity check).
        def shed(stage):
            gs = stage.static("generators")
            idx = gs.index[gs.carrier == "load shedding"]
            nidx = gs.index[gs.carrier == "negative load shedding"]
            p = stage.series("generators", "p")
            e = stage.energy(p.reindex(columns=idx, fill_value=0)).sum() / TWH
            ne = stage.energy(p.reindex(columns=nidx, fill_value=0)).sum() / TWH
            peak = p.reindex(columns=idx, fill_value=0).sum(axis=1).max()
            return e, ne, peak
        m = shed(mkt)
        gr = shed(g)
        self.shedding = {
            "market_twh": m[0], "market_neg_twh": m[1], "market_peak_mw": m[2],
            "grid_twh": gr[0], "grid_neg_twh": gr[1], "grid_peak_mw": gr[2],
        }
        if gr[0] > 0.1 or m[0] > 0.1:
            self.notes.append(
                f"Load shedding is not negligible (market {m[0]:.2f} TWh, grid {gr[0]:.2f} TWh)."
            )

    # ------------------------------------------------------------------
    # Focus region (Schleswig-Holstein)
    # ------------------------------------------------------------------
    def compute_focus(self):
        fb = self.focus_buses
        ac = self.ac
        loads = self.grid.static("loads")
        acl = loads[(loads.carrier == "AC") & loads.bus.isin(fb)]
        lp = self.grid.series("loads", "p").reindex(columns=acl.index, fill_value=0)
        cu = self.curt_units
        fz = self.focus_zone
        self.focus = {
            "n_buses": len(fb),
            "zone": self.label(fz),
            "zone_price": self.price_stats.loc[fz, "mean"] if fz in self.price_stats.index else np.nan,
            "nodal_price": float(self.nodal_price.reindex(list(fb)).mean()),
            "nodal_price_de": float(self.nodal_price.reindex(ac.index[ac.zone.map(geo.is_german_zone)]).mean()),
            "demand_twh": self.grid.energy(lp).sum() / TWH,
            "vres_gw": cu[cu.focus].p_nom.sum() / 1e3,
            "vres_avail_twh": self.curt_total["avail_focus_twh"],
            "curt_market_twh": self.curt_total["market_focus_twh"],
            "curt_grid_twh": self.curt_total["grid_focus_twh"],
            "rd_up_twh": self.rd_total["up_focus_twh"],
            "rd_down_twh": self.rd_total["down_focus_twh"],
            "ely_gw": self.electrolysis[self.electrolysis.focus].gw.sum(),
            "ely_twh": self.electrolysis[self.electrolysis.focus].twh.sum(),
            "bat_gw": self.batteries[self.batteries.focus].p_nom_opt.sum() / 1e3,
            "line_exp_gw": self.lines_total["focus_exp_gw"],
            "line_exp_gwkm": self.lines_total["focus_exp_gwkm"],
        }

    # ------------------------------------------------------------------
    def kpis(self):
        c = self.costs
        return {
            "n_de_zones": len(self.de_zones),
            "de_avg_price": self.de_avg_price,
            "price_convergence": self.price_convergence,
            "max_zone_price_gap": (self.price_stats.loc[self.de_zones, "mean"].max()
                                   - self.price_stats.loc[self.de_zones, "mean"].min()),
            "consumer_cost_beur": self.consumer_cost.sum() / 1e9,
            "rd_up_de_twh": self.rd_total["up_de_twh"],
            "rd_down_de_twh": self.rd_total["down_de_twh"],
            "rd_cost_meur": self.rd_total["cost_meur"],
            "curt_market_de_twh": self.curt_total["market_de_twh"],
            "curt_grid_de_twh": self.curt_total["grid_de_twh"],
            "curt_grid_de_rate": self.curt_total["grid_de_rate"],
            "ac_exp_gw_de": self.lines_total["ac_exp_gw_de"],
            "ac_exp_gwkm_de": self.lines_total["ac_exp_gwkm_de"],
            "grid_invest_meur": self.lines_total["ac_invest_meur"] + self.lines_total["dc_invest_meur"],
            "mean_loading_de": self.lines_total["mean_loading_de"],
            "congested_lines_de": self.lines_total["congested_lines_de"],
            "ely_gw_de": self.ely_by_zone.gw.sum(),
            "bat_gw_de": self.bat_by_zone[["existing_gw", "new_gw"]].sum().sum(),
            "total_cost_beur": c.sum() / 1e3,
            "cost_excl_gas_beur": (c.sum() - c["Natural-gas imports (CH4_NG)"]) / 1e3,
            "congestion_rent_de_meur": self.exchange[self.exchange.kind != "Abroad"].rent_meur.sum()
            if len(self.exchange) else 0.0,
            "shedding_grid_twh": self.shedding["grid_twh"],
        }

    def compute_all(self):
        steps = [
            ("topology", self.build_topology), ("prices", self.compute_prices),
            ("balance", self.compute_balance), ("exchange", self.compute_exchange),
            ("redispatch", self.compute_redispatch),
            ("curtailment", self.compute_curtailment),
            ("flexibility", self.compute_flexibility), ("lines", self.compute_lines),
            ("costs", self.compute_costs), ("focus", self.compute_focus),
        ]
        for name, fn in steps:
            logger.info("[%s] %s", self.sc.key, name)
            fn()
        # Free the raw time series, keep the results.
        for stage in (self.pre, self.mkt, self.grid):
            stage.clear()
        return self
