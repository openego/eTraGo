# -*- coding: utf-8 -*-
# Copyright 2016-2023  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
"""
Defines the market optimization within eTraGo
"""

import os

if "READTHEDOCS" not in os.environ:
    import logging

    from pypsa.components import component_attrs
    import numpy as np
    import pandas as pd

    from etrago.cluster.electrical import postprocessing, preprocessing
    from etrago.cluster.spatial import group_links
    from etrago.execute import optimize_with_rolling_horizon
    from etrago.tools.constraints import Constraints
    from etrago.tools.market_zones import create_market_zone_busmap

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, ClaraBuettner, CarlosEpia"

from etrago.tools.utilities import adjust_chp_model, adjust_PtH2_model


def reduce_market_snapshots(network, step=5):
    """Subsample snapshots and aggregate their original weights."""
    if not isinstance(step, int) or step < 1:
        raise ValueError("market snapshot_step must be a positive integer")

    original_snapshots = network.snapshots.copy()
    original_weightings = network.snapshot_weightings.copy()
    selected_snapshots = original_snapshots[::step]

    aggregated_weightings = pd.DataFrame(
        index=selected_snapshots,
        columns=original_weightings.columns,
        dtype=float,
    )

    for position, snapshot in enumerate(selected_snapshots):
        start = position * step
        stop = min(start + step, len(original_snapshots))
        aggregated_weightings.loc[snapshot] = (
            original_weightings.iloc[start:stop].sum()
        )

    network.set_snapshots(selected_snapshots)
    network.snapshot_weightings.loc[:, :] = aggregated_weightings.values

    logger.info(
        "Reduced market snapshots from %s to %s using a %s-hour step",
        len(original_snapshots),
        len(selected_snapshots),
        step,
    )

def market_optimization(self):
    logger.info("Start building pre market model")

    unit_commitment = True

    build_market_model(self, unit_commitment)

    market_snapshot_step = self.args["method"]["market_optimization"].get(
        "snapshot_step", 1
    )

    if market_snapshot_step > 1:
        reduce_market_snapshots(
            self.pre_market_model,
            step=market_snapshot_step,
        )
        reduce_market_snapshots(
            self.market_model,
            step=market_snapshot_step,
        )

    self.pre_market_model.determine_network_topology()

    logger.info("Start solving pre market model")

    if self.args["method"]["formulation"] == "pyomo":
        self.pre_market_model.lopf(
            solver_name=self.args["solver"],
            solver_options=self.args["solver_options"],
            pyomo=True,
            extra_functionality=Constraints(
                self.args,
                False,
                apply_on="pre_market_model",
            ).functionality,
            formulation=self.args["model_formulation"],
        )
    elif self.args["method"]["formulation"] == "linopy":
        status, condition = self.pre_market_model.optimize(
            solver_name=self.args["solver"],
            solver_options=self.args["solver_options"],
            extra_functionality=Constraints(
                self.args,
                False,
                apply_on="pre_market_model",
            ).functionality,
            linearized_unit_commitment=True,
        )

        if status != "ok":
            logger.warning(f"""Optimization failed with status {status}
                and condition {condition}""")

    else:
        logger.warning("Method type must be either 'pyomo' or 'linopy'")

    # Export results of pre-market model
    if self.args["export_results_path"]:
        path = self.args["export_results_path"]
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.pre_market_model.export_to_csv_folder(
            path + "/pre_market_optimization"
        )
    logger.info("Preparing short-term UC market model")

    build_shortterm_market_model(self, unit_commitment)

    self.market_model.determine_network_topology()
    logger.info("Start solving short-term UC market model")

    # Set 'linopy' as formulation to make sure that constraints are added
    method_args = self.args["method"]["formulation"]
    self.args["method"]["formulation"] = "linopy"

    optimize_with_rolling_horizon(
        self.market_model,
        self.pre_market_model,
        snapshots=None,
        horizon=self.args["method"]["market_optimization"]["rolling_horizon"][
            "planning_horizon"
        ],
        overlap=self.args["method"]["market_optimization"]["rolling_horizon"][
            "overlap"
        ],
        solver_name=self.args["solver"],
        extra_functionality=Constraints(
            self.args, False, apply_on="market_model"
        ).functionality,
        args=self.args,
    )

    # Reset formulation to previous setting of args
    self.args["method"]["formulation"] = method_args

    # Export results of market model
    if self.args["export_results_path"]:
        path = self.args["export_results_path"]
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.market_model.export_to_csv_folder(path + "/market_optimization")


def build_market_model(self, unit_commitment=False):
    """Builds market model based on imported network from eTraGo


    - import market regions from file or database
    - Cluster network to market regions
    -- consider marginal cost incl. generator noise when grouoping electrical
        generation capacities

    Returns
    -------
    None.

    """
    # Save network in full resolution if not copied before
    if self.network_tsa.buses.empty:
        self.network_tsa = self.network.copy()

    # use existing preprocessing to get only the electricity system
    net, weight, n_clusters, busmap_foreign = preprocessing(
        self, apply_on="market_model"
    )

    market_zones = self.args["method"]["market_optimization"]["market_zones"]

    if market_zones == "status_quo":
        df = pd.DataFrame(
            {
                "country": net.buses.country.unique(),
                "marketzone": net.buses.country.unique(),
            },
            columns=["country", "marketzone"],
        )

        # Germany and Luxembourg form one bidding zone.
        df.loc[(df.country == "DE") | (df.country == "LU"), "marketzone"] = (
            "DE/LU"
        )

        df["cluster"] = df.groupby(df.marketzone).grouper.group_info[0]

        for country in net.buses.country.unique():
            net.buses.loc[net.buses.country == country, "cluster"] = df.loc[
                df.country == country, "cluster"
            ].values[0]

        busmap = pd.Series(
            net.buses.cluster.astype(int).astype(str),
            net.buses.index,
        )
        medoid_idx = pd.Series(dtype=str)

    elif market_zones in {"DE2", "DE3", "DE4", "DE5"}:
        busmap, medoid_idx = create_market_zone_busmap(
            net,
            market_zones,
        )

    else:
        raise ValueError(
            f"Market zone setting {market_zones!r} is not available. "
            "Use one of: 'status_quo', 'DE2', 'DE3', 'DE4', or 'DE5'."
        )

    logger.info("Start market-zone-specific clustering")

    clustering, busmap = postprocessing(
        self,
        busmap,
        busmap_foreign,
        medoid_idx,
        aggregate_generators_carriers=[],
        aggregate_links=False,
        apply_on="market_model",
    )

    net = clustering.network

    # Keep the mapping grid bus -> market bus, e.g. to hand market
    # investments over to the grid optimisation.
    self.market_busmap = pd.Series(busmap).astype(str)
    self.market_busmap.index = self.market_busmap.index.astype(str)

    # Adjust positions foreign buses
    foreign = self.network.buses[self.network.buses.country != "DE"].copy()
    foreign = foreign[foreign.index.isin(self.network.loads.bus)]
    foreign = foreign.drop_duplicates(subset="country")
    foreign = foreign.set_index("country")

    for country in foreign.index:
        bus_for = net.buses.index[net.buses.country == country]
        net.buses.loc[bus_for, "x"] = foreign.at[country, "x"]
        net.buses.loc[bus_for, "y"] = foreign.at[country, "y"]

    # links_col = net.links.columns
    ac = net.lines[net.lines.carrier == "AC"]
    str1 = "transshipment_"
    ac.index = f"{str1}" + ac.index
    net.import_components_from_dataframe(
        ac.loc[:, ["bus0", "bus1", "capital_cost", "length"]]
        .assign(p_nom=ac.s_nom)
        .assign(p_nom_min=ac.s_nom_min)
        .assign(p_nom_max=ac.s_nom_max)
        .assign(p_nom_extendable=ac.s_nom_extendable)
        .assign(p_max_pu=ac.s_max_pu)
        .assign(p_min_pu=-1.0)
        .assign(carrier="DC")
        .set_index(ac.index),
        "Link",
    )
    net.lines.drop(
        net.lines.loc[net.lines.carrier == "AC"].index, inplace=True
    )
    # net.buses.loc[net.buses.carrier == 'AC', 'carrier'] = "DC"

    set_interzonal_transfer_limits(self, net, busmap, ac.index)

    net.generators_t.p_max_pu = self.network_tsa.generators_t.p_max_pu

    # Set stores and storage_units to cyclic
    if len(self.network_tsa.snapshots) > 1000:
        net.stores.loc[net.stores.carrier != "battery_storage", "e_cyclic"] = (
            True
        )
        net.storage_units.cyclic_state_of_charge = True
    net.stores.loc[net.stores.carrier == "dsm", "e_cyclic"] = False
    net.storage_units.cyclic_state_of_charge = True

    self.pre_market_model = net

    if self.args["method"]["market_optimization"].get(
        "fix_interzonal_capacity_in_market", False
    ):
        fix_interzonal_capacity(net)

    gas_clustering_market_model(self)

    if self.args["method"]["market_optimization"].get(
        "split_h2_nodes_by_zone", False
    ):
        split_h2_nodes_by_zone(self.pre_market_model)

    if unit_commitment:
        set_unit_commitment(self, apply_on="pre_market_model")

    self.pre_market_model.links.loc[
        self.pre_market_model.links.carrier.isin(
            ["CH4", "DC", "AC", "H2_grid", "H2_saltcavern"]
        ),
        "p_min_pu",
    ] = -1.0

    if self.args["scn_name"] in [
        "eGon100RE",
        "powerd2025",
        "powerd2030",
        "powerd2035",
    ]:
        self.pre_market_model = adjust_PtH2_model(self)
        logger.info("PtH2-Model adjusted in pre_market_network")

        self.pre_market_model = adjust_chp_model(self)
        logger.info(
            "CHP model in foreign countries adjusted in pre_market_network"
        )

    # Set country tags for market model
    self.buses_by_country(apply_on="pre_market_model")
    self.geolocation_buses(apply_on="pre_market_model")

    self.market_model = self.pre_market_model.copy()

    self.pre_market_model.links, self.pre_market_model.links_t = group_links(
        self.pre_market_model,
        carriers=[
            "central_heat_pump",
            "central_resistive_heater",
            "rural_heat_pump",
            "rural_resistive_heater",
            "BEV_charger",
            "dsm",
            "central_gas_boiler",
            "rural_gas_boiler",
        ],
    )
    self.pre_market_model.links.min_up_time = (
        self.pre_market_model.links.min_up_time.astype(int)
    )
    self.pre_market_model.links.down_up_time = (
        self.pre_market_model.links.min_down_time.astype(int)
    )
    self.pre_market_model.links.down_time_before = (
        self.pre_market_model.links.down_time_before.astype(int)
    )
    self.pre_market_model.links.up_time_before = (
        self.pre_market_model.links.up_time_before.astype(int)
    )
    self.pre_market_model.links.min_down_time = (
        self.pre_market_model.links.min_down_time.astype(int)
    )
    self.pre_market_model.links.min_up_time = (
        self.pre_market_model.links.min_up_time.astype(int)
    )


def build_shortterm_market_model(self, unit_commitment=False):

    self.market_model.storage_units.loc[
        self.market_model.storage_units.p_nom_extendable, "p_nom"
    ] = self.pre_market_model.storage_units.loc[
        self.pre_market_model.storage_units.p_nom_extendable, "p_nom_opt"
    ].clip(
        lower=0
    )
    self.market_model.stores.loc[
        self.market_model.stores.e_nom_extendable, "e_nom"
    ] = self.pre_market_model.stores.loc[
        self.pre_market_model.stores.e_nom_extendable, "e_nom_opt"
    ].clip(
        lower=0
    )

    # Fix oder of bus0 and bus1 of DC links
    dc_links = self.market_model.links[self.market_model.links.carrier == "DC"]
    bus0 = dc_links[dc_links.bus0.astype(int) < dc_links.bus1.astype(int)].bus1
    bus1 = dc_links[dc_links.bus0.astype(int) < dc_links.bus1.astype(int)].bus0
    self.market_model.links.loc[bus0.index, "bus0"] = bus0.values
    self.market_model.links.loc[bus1.index, "bus1"] = bus1.values

    dc_links = self.pre_market_model.links[
        self.pre_market_model.links.carrier == "DC"
    ]
    bus0 = dc_links[dc_links.bus0.astype(int) < dc_links.bus1.astype(int)].bus1
    bus1 = dc_links[dc_links.bus0.astype(int) < dc_links.bus1.astype(int)].bus0
    self.pre_market_model.links.loc[bus0.index, "bus0"] = bus0.values
    self.pre_market_model.links.loc[bus1.index, "bus1"] = bus1.values

    grouped_links = (
        self.market_model.links.loc[self.market_model.links.p_nom_extendable]
        .groupby(["carrier", "bus0", "bus1"])
        .p_nom.sum()
        .reset_index()
    )
    for link in grouped_links.index:
        bus0 = grouped_links.loc[link, "bus0"]
        bus1 = grouped_links.loc[link, "bus1"]
        carrier = grouped_links.loc[link, "carrier"]

        self.market_model.links.loc[
            (self.market_model.links.bus0 == bus0)
            & (self.market_model.links.bus1 == bus1)
            & (self.market_model.links.carrier == carrier),
            "p_nom",
        ] = (
            self.pre_market_model.links.loc[
                (self.pre_market_model.links.bus0 == bus0)
                & (self.pre_market_model.links.bus1 == bus1)
                & (self.pre_market_model.links.carrier == carrier),
                "p_nom_opt",
            ]
            .clip(lower=0)
            .values
        )

    self.market_model.lines.loc[
        self.market_model.lines.s_nom_extendable, "s_nom"
    ] = self.pre_market_model.lines.loc[
        self.pre_market_model.lines.s_nom_extendable, "s_nom_opt"
    ].clip(
        lower=0
    )

    self.market_model.storage_units.p_nom_extendable = False
    self.market_model.stores.e_nom_extendable = False
    self.market_model.links.p_nom_extendable = False
    self.market_model.lines.s_nom_extendable = False

    self.market_model.mremove(
        "Store",
        self.market_model.stores[self.market_model.stores.e_nom == 0].index,
    )
    self.market_model.stores.e_cyclic = False
    self.market_model.storage_units.cyclic_state_of_charge = False

    if unit_commitment:
        set_unit_commitment(self, apply_on="market_model")

    self.market_model.links.loc[
        self.market_model.links.carrier.isin(
            ["CH4", "DC", "AC", "H2_grid", "H2_saltcavern"]
        ),
        "p_min_pu",
    ] = -1.0

    # Set country tags for market model
    self.buses_by_country(apply_on="market_model")
    self.geolocation_buses(apply_on="market_model")


def fix_interzonal_capacity(network):
    """Make all links between two electricity market zones non-extendable.

    The transshipment links between German zones and the interconnectors to
    neighbouring zones inherit ``p_nom_extendable`` from the AC lines / DC
    links. A market does not build transmission capacity, and expanding it in
    the pre-market model removes exactly the congestion a zone split is meant
    to price. Capacities are kept at their existing value (p_nom).
    """
    ac = network.buses.index[network.buses.carrier == "AC"]
    sel = network.links.index[
        network.links.bus0.isin(ac) & network.links.bus1.isin(ac)
    ]
    was = int(network.links.loc[sel, "p_nom_extendable"].sum())
    network.links.loc[sel, "p_nom_extendable"] = False
    logger.info(
        "Inter-zonal and border links fixed to existing capacity in the "
        "market model (%s links, %s of them were extendable, %.1f GW)",
        len(sel),
        was,
        network.links.loc[sel, "p_nom"].sum() / 1e3,
    )


def scale_unit_commitment_to_step(unit_commitment, step):
    """Convert hourly unit-commitment parameters to ``step``-hour snapshots.

    * minimum up/down times are given in hours -> number of snapshots
      (rounded up, at least one snapshot);
    * ramp limits are given per hour -> per snapshot (capped at 1);
    * start-up/shut-down limits cover the first hour plus the ramping in the
      remaining ``step - 1`` hours of the snapshot.
    """
    uc = unit_commitment.copy().astype(float)
    for attr in ["min_up_time", "min_down_time"]:
        uc.loc[attr] = np.ceil(uc.loc[attr] / step)
    ramp = uc.loc["ramp_limit_up"]
    for attr in ["ramp_limit_start_up", "ramp_limit_shut_down"]:
        uc.loc[attr] = (uc.loc[attr] + ramp * (step - 1)).clip(upper=1.0)
    uc.loc["ramp_limit_up"] = (ramp * step).clip(upper=1.0)
    return uc


def set_interzonal_transfer_limits(self, net, busmap, line_names):
    """Give the transshipment links the transfer limits of their lines.

    The transshipment links between market zones aggregate the AC lines
    crossing a zone border (p_nom = sum of s_nom). Their static
    ``p_max_pu`` is copied from the aggregated lines, but the branch capacity
    factor and dynamic line rating of the German lines only exist as time
    series (``lines_t.s_max_pu``), which got lost. This sets

        p_max_pu(t) = sum(s_nom * s_max_pu(t)) / sum(s_nom),  p_min_pu = -p_max_pu

    per link from the underlying lines, so the market sees the same limits as
    the grid model. Optionally an additional factor is applied to links
    between two German market zones (``interzonal_capacity_factor``) as a
    simple NTC-like reduction for loop flows.
    """
    settings = self.args["method"]["market_optimization"]
    use_lines = settings.get("interzonal_line_limits", True)
    factor = float(settings.get("interzonal_capacity_factor", 1.0))
    if not use_lines and factor == 1.0:
        return

    names = pd.Index(line_names).astype(str)  # already "transshipment_<id>"
    names = names[names.isin(net.links.index)]
    if names.empty:
        return
    links = net.links.loc[names]

    src = self.network_tsa
    bm = pd.Series(busmap).astype(str)
    lines = src.lines
    zone0 = lines.bus0.astype(str).map(bm)
    zone1 = lines.bus1.astype(str).map(bm)
    s_max_pu = src.get_switchable_as_dense("Line", "s_max_pu")

    p_max_pu = pd.DataFrame(index=net.snapshots, columns=names, dtype=float)
    for name, link in links.iterrows():
        a, b = str(link.bus0), str(link.bus1)
        sel = lines.index[
            ((zone0 == a) & (zone1 == b)) | ((zone0 == b) & (zone1 == a))
        ]
        s_nom = lines.loc[sel, "s_nom"]
        if use_lines and len(sel) and s_nom.sum() > 0:
            p_max_pu[name] = (
                s_max_pu[sel].mul(s_nom, axis=1).sum(axis=1) / s_nom.sum()
            ).reindex(net.snapshots).values
        else:
            p_max_pu[name] = float(link.p_max_pu)

    if factor != 1.0:
        german = set(
            bm.reindex(
                src.buses.index[
                    (src.buses.carrier == "AC") & (src.buses.country == "DE")
                ].astype(str)
            ).dropna()
        )
        internal = links.index[
            links.bus0.astype(str).isin(german)
            & links.bus1.astype(str).isin(german)
        ]
        p_max_pu[internal] *= factor
        logger.info(
            "Interzonal capacity factor %.2f applied to %s German links",
            factor,
            len(internal),
        )

    p_max_pu = p_max_pu.clip(lower=0.0).fillna(1.0)
    p_max_pu.columns.name = net.links_t.p_max_pu.columns.name
    p_max_pu.index.name = net.links_t.p_max_pu.index.name
    net.links_t.p_max_pu = _concat_keep_names(
        [net.links_t.p_max_pu.drop(columns=names, errors="ignore"), p_max_pu],
        axis=1,
    )
    net.links_t.p_min_pu = _concat_keep_names(
        [net.links_t.p_min_pu.drop(columns=names, errors="ignore"), -p_max_pu],
        axis=1,
    )
    summary = (p_max_pu.mean() * links.p_nom).sum() / links.p_nom.sum()
    logger.info(
        "Interzonal transfer limits set for %s links "
        "(capacity-weighted mean p_max_pu %.3f, was %.3f)",
        len(names),
        summary,
        (links.p_max_pu * links.p_nom).sum() / links.p_nom.sum(),
    )


def split_h2_nodes_by_zone(network, h2_carriers=("H2_grid",)):
    """Give every market zone its own copy of a shared H2 node.

    After the market clustering, electrolysers of several bidding zones can
    feed the same H2 node. Whenever they run at partial load they then share
    one H2 value and equalise the electricity prices of their zones. This
    sensitivity splits such nodes: each zone gets its own H2 bus, its
    electrolysers and H2-to-power plants are re-connected to it, and the
    other components of the node (H2 demand, H2 stores, CH4<->H2 links, ...)
    are distributed in proportion to the zones' electrolyser capacity
    (p_nom_max, or p_nom where p_nom_max is not finite).
    """
    buses = network.buses
    ac = set(buses.index[buses.carrier == "AC"])
    links = network.links
    ely = links[(links.carrier == "power_to_H2") & links.bus0.isin(ac)]

    n_split = 0
    for node, grp in ely.groupby("bus1"):
        if buses.at[node, "carrier"] not in h2_carriers:
            continue
        cap = grp.p_nom_max.where(np.isfinite(grp.p_nom_max), grp.p_nom)
        shares = cap.groupby(grp.bus0).sum()
        if len(shares) < 2:
            continue
        shares = (
            shares / shares.sum()
            if shares.sum() > 0
            else pd.Series(1.0 / len(shares), index=shares.index)
        )
        keep = shares.idxmax()
        new_bus = {z: f"{node}_zone{z}" for z in shares.index if z != keep}

        # Buses
        copies = pd.concat(
            [buses.loc[[node]].rename(index={node: b}) for b in new_bus.values()]
        )
        network.buses = _concat_keep_names([network.buses, copies])

        _split_one_ports(network, "Load", node, shares, keep, new_bus,
                         scale=["p_set", "q_set"], scale_t=["p_set", "q_set"])
        _split_one_ports(network, "Store", node, shares, keep, new_bus,
                         scale=["e_nom", "e_nom_min", "e_nom_max", "e_nom_opt",
                                "e_initial"], scale_t=[])
        _split_one_ports(network, "Generator", node, shares, keep, new_bus,
                         scale=[], scale_t=[])
        _split_links(network, node, shares, keep, new_bus)

        # Electrolysers and H2-to-power plants follow their zone.
        lk = network.links
        for zone, bus in new_bus.items():
            lk.loc[(lk.carrier == "power_to_H2") & (lk.bus1 == node)
                   & (lk.bus0 == zone), "bus1"] = bus
            lk.loc[(lk.carrier == "H2_to_power") & (lk.bus0 == node)
                   & (lk.bus1 == zone), "bus0"] = bus
        n_split += 1
        logger.info(
            "H2 node %s split by zone: %s",
            node,
            {z: round(v, 3) for z, v in shares.items()},
        )

    logger.info("Split %s shared H2 nodes by market zone", n_split)


def _split_one_ports(network, cls, node, shares, keep, new_bus, scale, scale_t):
    df = network.df(cls)
    pnl = network.pnl(cls)
    sel = df.index[df.bus == node]
    if sel.empty:
        return
    copies = []
    for zone, bus in new_bus.items():
        part = df.loc[sel].copy()
        part.index = [f"{i} zone{zone}" for i in sel]
        part["bus"] = bus
        for col in scale:
            if col in part:
                part[col] = part[col] * shares[zone]
        copies.append(part)
        for attr, ts in pnl.items():
            cols = ts.columns.intersection(sel)
            if cols.empty:
                continue
            factor = shares[zone] if attr in scale_t else 1.0
            add = ts[cols] * factor
            add.columns = [f"{i} zone{zone}" for i in cols]
            pnl[attr] = _concat_keep_names([ts, add], axis=1)
    for col in scale:
        if col in df:
            df.loc[sel, col] = df.loc[sel, col] * shares[keep]
    for attr in scale_t:
        if attr in pnl:
            cols = pnl[attr].columns.intersection(sel)
            pnl[attr][cols] = pnl[attr][cols] * shares[keep]
    setattr(network, network.components[cls]["list_name"],
            _concat_keep_names([df] + copies))


def _split_links(network, node, shares, keep, new_bus):
    df = network.links
    pnl = network.links_t
    sel = df.index[
        ((df.bus0 == node) | (df.bus1 == node))
        & ~df.carrier.isin(["power_to_H2", "H2_to_power"])
    ]
    if sel.empty:
        return
    scale = ["p_nom", "p_nom_min", "p_nom_max", "p_nom_opt"]
    copies = []
    for zone, bus in new_bus.items():
        part = df.loc[sel].copy()
        part.index = [f"{i} zone{zone}" for i in sel]
        part.loc[part.bus0 == node, "bus0"] = bus
        part.loc[part.bus1 == node, "bus1"] = bus
        for col in scale:
            if col in part:
                part[col] = part[col] * shares[zone]
        copies.append(part)
        for attr, ts in pnl.items():
            cols = ts.columns.intersection(sel)
            if cols.empty:
                continue
            add = ts[cols].copy()
            add.columns = [f"{i} zone{zone}" for i in cols]
            pnl[attr] = _concat_keep_names([ts, add], axis=1)
    for col in scale:
        if col in df:
            df.loc[sel, col] = df.loc[sel, col] * shares[keep]
    network.links = _concat_keep_names([df] + copies)


def _concat_keep_names(frames, axis=0):
    """pd.concat that keeps the index/column names PyPSA relies on."""
    out = pd.concat(frames, axis=axis)
    out.index.name = frames[0].index.name
    out.columns.name = frames[0].columns.name
    return out


def set_unit_commitment(self, apply_on):

    if apply_on == "market_model":
        network = self.market_model
    elif apply_on == "pre_market_model":
        network = self.pre_market_model
    else:
        print(f"Can not be applied on {apply_on} yet.")
        return

    # set UC constraints
    unit_commitment = pd.DataFrame(
        {
            "OCGT": [1.0, 0.2, 0.2, 0.2, 0.0, 0.0, 9.6],
            "CCGT": [1.0, 0.45, 0.45, 0.45, 3.0, 2.0, 34.2],
            "coal": [1.0, 0.38, 0.38, 0.325, 5.0, 6.0, 35.64],
            "lignite": [1.0, 0.40, 0.40, 0.40, 7.0, 6.0, 19.14],
            "nuclear": [0.3, 0.5, 0.5, 0.5, 6.0, 10.0, 16.5],
        },
        index=[
            "ramp_limit_up",
            "ramp_limit_start_up",
            "ramp_limit_shut_down",
            "p_min_pu",
            "min_up_time",
            "min_down_time",
            "start_up_cost",
        ],
    )

    unit_commitment.index.name = "attribute"

    # The UC parameters above are given per hour. With a reduced market
    # resolution (snapshot_step > 1) PyPSA interprets them per snapshot, so
    # convert them to the snapshot length.
    market_args = self.args["method"]["market_optimization"]
    step = int(market_args.get("snapshot_step", 1) or 1)
    if step > 1 and market_args.get("scale_uc_to_snapshot_step", True):
        unit_commitment = scale_unit_commitment_to_step(unit_commitment, step)
        logger.info(
            "Unit-commitment parameters converted to %s-hour snapshots", step
        )

    committable_attrs = network.generators.carrier.isin(
        unit_commitment
    ).to_frame("committable")

    for attr in unit_commitment.index:
        default = component_attrs["Generator"].default[attr]
        committable_attrs[attr] = network.generators.carrier.map(
            unit_commitment.loc[attr]
        ).fillna(default)
        committable_attrs[attr] = committable_attrs[attr].astype(
            network.generators.carrier.map(unit_commitment.loc[attr]).dtype
        )

    network.generators[committable_attrs.columns] = committable_attrs
    network.generators.min_up_time = network.generators.min_up_time.astype(int)
    network.generators.min_down_time = network.generators.min_down_time.astype(
        int
    )

    # Tadress link carriers i.e. OCGT
    committable_links = network.links.carrier.isin(unit_commitment).to_frame(
        "committable"
    )

    for attr in unit_commitment.index:
        default = component_attrs["Link"].default[attr]
        committable_links[attr] = network.links.carrier.map(
            unit_commitment.loc[attr]
        ).fillna(default)
        committable_links[attr] = committable_links[attr].astype(
            network.links.carrier.map(unit_commitment.loc[attr]).dtype
        )

    network.links[committable_links.columns] = committable_links
    network.links.min_up_time = network.links.min_up_time.astype(int)
    network.links.min_down_time = network.links.min_down_time.astype(int)

    network.generators.loc[
        network.generators.committable, "ramp_limit_down"
    ].fillna(1.0, inplace=True)
    network.links.loc[network.links.committable, "ramp_limit_down"].fillna(
        1.0, inplace=True
    )

    if apply_on == "pre_market_model":
        # Set all start_up and shut_down cost to 0 to simpify unit committment
        network.links.loc[network.links.committable, "start_up_cost"] = 0.0
        network.links.loc[network.links.committable, "shut_down_cost"] = 0.0

        # Set all start_up and shut_down cost to 0 to simpify unit committment
        network.generators.loc[
            network.generators.committable, "start_up_cost"
        ] = 0.0
        network.generators.loc[
            network.generators.committable, "shut_down_cost"
        ] = 0.0

    logger.info(f"Unit commitment set for {apply_on}")


def gas_clustering_market_model(self):
    from etrago.cluster.gas import (
        gas_postprocessing,
        preprocessing as gas_preprocessing,
    )

    if self.network.links[self.network.links.carrier == "H2_grid"].empty:
        logger.warning("H2 grid not clustered for market in this scenario")
        return

    ch4_network, weight_ch4, n_clusters_ch4 = gas_preprocessing(
        self, "CH4", apply_on="market_model"
    )

    df = pd.DataFrame(
        {
            "country": ch4_network.buses.country.unique(),
            "marketzone": ch4_network.buses.country.unique(),
        },
        columns=["country", "marketzone"],
    )

    df.loc[(df.country == "DE") | (df.country == "LU"), "marketzone"] = "DE/LU"

    df["cluster"] = df.groupby(df.marketzone).grouper.group_info[0]

    for i in ch4_network.buses.country.unique():
        ch4_network.buses.loc[ch4_network.buses.country == i, "cluster"] = (
            df.loc[df.country == i, "cluster"].values[0]
        )

    busmap = pd.Series(
        ch4_network.buses.cluster.astype(int).astype(str),
        ch4_network.buses.index,
    )

    if "H2_grid" in self.network.links.carrier.unique():
        h2_network, weight_h2, n_clusters_h2 = gas_preprocessing(
            self, "H2_grid", apply_on="market_model"
        )

        df_h2 = pd.DataFrame(
            {
                "country": h2_network.buses.country.unique(),
                "marketzone": h2_network.buses.country.unique(),
            },
            columns=["country", "marketzone"],
        )

        df_h2.loc[
            (df.country == "DE") | (df_h2.country == "LU"), "marketzone"
        ] = "DE/LU"

        df_h2["cluster"] = df_h2.groupby(df_h2.marketzone).grouper.group_info[
            0
        ] + len(df)

        for i in h2_network.buses.country.unique():
            h2_network.buses.loc[h2_network.buses.country == i, "cluster"] = (
                df_h2.loc[df_h2.country == i, "cluster"].values[0]
            )

        busmap = pd.concat(
            [
                busmap,
                pd.Series(
                    h2_network.buses.cluster.astype(int).astype(str),
                    h2_network.buses.index,
                ),
            ]
        )

    medoid_idx = pd.Series()
    # Set country tags for market model
    self.buses_by_country(apply_on="pre_market_model")
    self.geolocation_buses(apply_on="pre_market_model")

    self.pre_market_model, busmap_new = gas_postprocessing(
        self,
        busmap,
        medoid_idx=medoid_idx,
        apply_on="market_model",
        aggregate_generators_carriers=[],
    )
