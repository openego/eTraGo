# -*- coding: utf-8 -*-
"""
H2 adjustments for the eGon2035 data model (SPREAD.SH).

The eGon2035 data has no H2 transport: every H2_grid bus is an island with
its local H2 demand, and salt-cavern buses are only connected to the power
grid. Outside the focus region this is hidden by the gas clustering, which
merges hundreds of H2 buses into a few copper-plate hubs. Inside the focus
region the H2 buses are protected from clustering and stay small islands, so
electrolysers there can only supply local demand.

Two optional adjustments, applied in ``Etrago.adjust_network``:

* ``electrolyzer_site_limit = "egon_dev"`` replaces the 1 MW limit of
  electrolysers (and fuel cells) at H2_grid buses with the rule of the
  current eGon-data code (hydrogen_etrago/power_to_h2.py): 120 MW at HV/MV
  substations, 5000 MW at EHV substations. The substation type is taken from
  the voltage of the AC bus.
* ``connect_focus_h2 = "retrofit"`` adds an extendable, bidirectional H2
  link along every CH4 pipeline that touches the focus region (repurposed
  CH4 pipeline, capacity limited to a share of the CH4 capacity).
  ``"nearest"`` instead adds one new, unlimited pipeline from every focus
  H2_grid bus to the nearest German H2_grid bus outside. In both modes the
  salt-cavern buses of the focus region are connected to the nearest focus
  H2_grid bus. See ``connect_focus_h2`` for the details.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

H2_POWER_CARRIERS = ("power_to_H2", "H2_to_power")
FOCUS_PIPELINE_CARRIER = "H2_focus_pipeline"
RETROFIT_CARRIER = "H2_focus_retrofit"


def _settings(self):
    return self.args["method"]["market_optimization"]


def _haversine_km(x0, y0, x1, y1):
    """Great-circle distance in km between (x0, y0) and arrays (x1, y1)."""
    lon0, lat0, lon1, lat1 = map(np.radians, (x0, y0, x1, y1))
    a = (
        np.sin((lat1 - lat0) / 2) ** 2
        + np.cos(lat0) * np.cos(lat1) * np.sin((lon1 - lon0) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def set_electrolyzer_site_limit(self):
    """Replace the 1 MW limit of power_to_H2 / H2_to_power links at H2_grid
    buses by 120 MW (AC bus <= 110 kV) or 5000 MW (AC bus >= 220 kV)."""
    rule = _settings(self).get("electrolyzer_site_limit", "data")
    if rule in (None, "data"):
        return
    if rule != "egon_dev":
        raise ValueError(
            "electrolyzer_site_limit must be 'data' or 'egon_dev', "
            f"found {rule!r}"
        )

    n = self.network
    links, buses = n.links, n.buses
    is_ptx = links.carrier == "power_to_H2"
    ac_bus = links.bus0.where(is_ptx, links.bus1)
    h2_bus = links.bus1.where(is_ptx, links.bus0)
    sel = (
        links.carrier.isin(H2_POWER_CARRIERS)
        & (links.p_nom_max == 1.0)
        & h2_bus.map(buses.carrier).eq("H2_grid")
    )
    v_nom = ac_bus[sel].map(buses.v_nom)
    links.loc[sel, "p_nom_max"] = np.where(v_nom <= 110, 120.0, 5000.0)

    counts = links.loc[sel].groupby(["carrier", "p_nom_max"]).size()
    logger.info(
        "Electrolyser site limit 'egon_dev': 1 MW limit replaced on %s links "
        "(%s)",
        int(sel.sum()),
        {f"{c} {int(p)} MW": int(k) for (c, p), k in counts.items()},
    )


def _link_index(n, count):
    """New link ids after the highest numeric id."""
    start = 1 + max(
        [int(i) for i in n.links.index if str(i).lstrip("-").isdigit()]
        or [0]
    )
    if pd.api.types.is_integer_dtype(n.links.index):
        return pd.Index([start + i for i in range(count)], name=n.links.index.name)
    return pd.Index([str(start + i) for i in range(count)], name=n.links.index.name)


def connect_focus_h2(self):
    """Add extendable H2 links that connect the H2 buses of the focus region
    to the German H2 system (see module docstring).

    connect_focus_h2:
        False      nothing is added
        "retrofit" one H2 link per CH4 pipeline with at least one end in the
                   focus region, between the H2_grid buses at the two CH4
                   pipeline ends (every German CH4 bus has an H2_grid bus at
                   the same location). Capacity up to
                   h2_retrofit_capacity_share x CH4 pipeline capacity, cost
                   h2_retrofit_capital_cost per km. The CH4 pipeline keeps its
                   capacity (no coupling of both).
        "nearest"  (or True) one new pipeline from every focus H2_grid bus to
                   the nearest German H2_grid bus outside the focus region,
                   unlimited, cost h2_pipeline_capital_cost per km.
    In both modes every salt-cavern bus in the focus region is connected to
    the nearest focus H2_grid bus (new pipeline, unlimited), unless
    connect_focus_saltcaverns is False.
    """
    settings = _settings(self)
    mode = settings.get("connect_focus_h2", False)
    if mode is True:
        mode = "nearest"
    if not mode:
        return
    if mode not in ("nearest", "retrofit"):
        raise ValueError(
            "connect_focus_h2 must be False, 'retrofit' or 'nearest', "
            f"found {mode!r}"
        )

    focus_region = self.args["network_clustering"]["method"].get(
        "focus_region"
    )
    if not focus_region:
        logger.warning("connect_focus_h2 is set, but no focus_region given.")
        return

    from etrago.cluster.spatial import buses_inside_focus_region

    n = self.network
    buses = n.buses
    focus = set(
        buses_inside_focus_region(self, n, focus_region).astype(str)
    )
    in_focus = pd.Series(buses.index.astype(str).isin(focus), index=buses.index)
    h2 = (buses.carrier == "H2_grid") & (buses.country == "DE")
    inside = buses[h2 & in_focus]
    outside = buses[h2 & ~in_focus]
    caverns = buses[(buses.carrier == "H2_saltcavern") & in_focus]
    if inside.empty:
        logger.warning("connect_focus_h2: no H2_grid buses in the focus region.")
        return

    factor = float(
        self.args["network_clustering"].get("line_length_factor", 1.0)
    )
    new_cost = float(settings.get("h2_pipeline_capital_cost", 16.6))

    def nearest(row, candidates):
        d = _haversine_km(
            row.x, row.y, candidates.x.values, candidates.y.values
        )
        i = int(np.argmin(d))
        return candidates.index[i], float(d[i])

    def dist(a, b):
        return float(
            _haversine_km(
                buses.at[a, "x"], buses.at[a, "y"],
                buses.at[b, "x"], buses.at[b, "y"],
            )
        )

    # rows: bus0, bus1, carrier, km (straight line), p_nom_max, cost per km
    rows = []
    if mode == "retrofit":
        share = float(settings.get("h2_retrofit_capacity_share", 0.6))
        cost = float(settings.get("h2_retrofit_capital_cost", 8.3))
        # CH4 bus -> H2_grid bus at the same location
        h2_by_xy = {
            (round(r.x, 6), round(r.y, 6)): i for i, r in buses[h2].iterrows()
        }
        ch4_to_h2 = {
            i: h2_by_xy.get((round(r.x, 6), round(r.y, 6)))
            for i, r in buses[buses.carrier == "CH4"].iterrows()
        }
        pipes = n.links[
            (n.links.carrier == "CH4")
            & (n.links.bus0.map(in_focus) | n.links.bus1.map(in_focus))
        ]
        skipped = 0
        for _, p in pipes.iterrows():
            a, b = ch4_to_h2.get(p.bus0), ch4_to_h2.get(p.bus1)
            if a is None or b is None or a == b:
                skipped += 1
                continue
            rows.append((a, b, RETROFIT_CARRIER, dist(a, b),
                         share * float(p.p_nom), cost))
        logger.info(
            "connect_focus_h2 'retrofit': %s CH4 pipelines touch the focus "
            "region, %s become H2 links (%s skipped: no H2 bus at an end), "
            "capacity up to %.0f %% of the CH4 pipeline, %.1f EUR/MW/km/a",
            len(pipes), len(pipes) - skipped, skipped, 100 * share, cost,
        )
    else:
        if outside.empty:
            logger.warning("connect_focus_h2: no H2_grid buses outside focus.")
        for bus, row in inside.iterrows() if not outside.empty else []:
            target, d = nearest(row, outside)
            rows.append((bus, target, FOCUS_PIPELINE_CARRIER, d, np.inf,
                         new_cost))

    if settings.get("connect_focus_saltcaverns", True):
        for bus, row in caverns.iterrows():
            # bus0 = H2_grid, bus1 = cavern: the gas clustering maps cavern
            # buses to the H2_grid cluster they are connected to
            target, d = nearest(row, inside)
            rows.append((target, bus, FOCUS_PIPELINE_CARRIER, d, np.inf,
                         new_cost))

    if not rows:
        logger.warning("connect_focus_h2: no links added.")
        return

    length = np.array([max(r[3], 1.0) * factor for r in rows])
    new = pd.DataFrame(
        {
            "bus0": [r[0] for r in rows],
            "bus1": [r[1] for r in rows],
            "carrier": [r[2] for r in rows],
            "p_nom": 0.0,
            "p_nom_extendable": True,
            "p_nom_min": 0.0,
            "p_nom_max": [r[4] for r in rows],
            "p_min_pu": -1.0,
            "p_max_pu": 1.0,
            "efficiency": 1.0,
            "marginal_cost": 0.0,
            "length": length,
            "capital_cost": np.array([r[5] for r in rows]) * length,
            "scn_name": self.args["scn_name"],
            "country": "DE",
        },
        index=_link_index(n, len(rows)),
    )
    # only attributes the links table already knows (scn_name, country are
    # custom eTraGo columns that may be missing)
    new = new[[c for c in new.columns if c in n.links.columns]]

    for carrier in new.carrier.unique():
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)
    n.import_components_from_dataframe(new, "Link")

    summary = new.groupby("carrier").agg(
        links=("bus0", "size"), km=("length", "sum")
    ).round(0)
    logger.info(
        "connect_focus_h2 '%s': added %s H2 links %s",
        mode,
        len(new),
        summary.to_dict("index"),
    )
