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
    import pandas as pd

    from etrago.cluster.electrical import postprocessing, preprocessing

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, ClaraBuettner, CarlosEpia"


def market_optimization(self):
    logger.info("Start building pre market model")
    build_market_model(self)
    logger.info("Start solving pre market model")
    self.pre_market_model.lopf(
        solver_name=self.args["solver"],
        solver_options=self.args["solver_options"],
        pyomo=True,
        extra_functionality=extra_functionality(),
        formulation=self.args["model_formulation"],
    )

    logger.info("Preparing short-term UC market model")
    build_shortterm_market_model(self)
    logger.info("Start solving short-term UC market model")

    optimize_with_rolling_horizon(
        self.market_model,
        self.pre_market_model,
        snapshots=None,
        horizon=72,
        overlap=48,
        solver_name=self.args["solver"],
        extra_functionality=extra_functionality(),
    )

    # quick and dirty csv export of market model results
    path = self.args["csv_export"]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    self.market_model.export_to_csv_folder(path + "/market")


def optimize_with_rolling_horizon(n, pre_market, snapshots=None, horizon=2, overlap=0, **kwargs):
    """
    Optimizes the network in a rolling horizon fashion.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list-like
        Set of snapshots to consider in the optimization. The default is None.
    horizon : int
        Number of snapshots to consider in each iteration. Defaults to 100.
    overlap : int
        Number of snapshots to overlap between two iterations. Defaults to 0.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

    Returns
    -------
    None
    """
    if snapshots is None:
        snapshots = n.snapshots

    if horizon <= overlap:
        raise ValueError("overlap must be smaller than horizon")

    starting_points = range(0, len(snapshots), horizon - overlap)
    for i, start in enumerate(starting_points):
        end = min(len(snapshots), start + horizon)
        sns = snapshots[start:end]
        logger.info(
            f"Optimizing network for snapshot horizon [{sns[0]}:{sns[-1]}] ({i+1}/{len(starting_points)})."
        )

        if i:
            if not n.stores.empty:
                n.stores.e_initial = n.stores_t.e.loc[snapshots[start - 1]]

                # Select seasonal stores
                seasonal_stores = n.stores.index[n.stores.carrier.isin(["central_heat_store", "H2_overground", "CH4"])]

                # Set e_initial from pre_market model for seasonal stores
                n.stores.e_initial[seasonal_stores] = pre_market.stores_t.e.loc[snapshots[start - 1], seasonal_stores]

                # Set e at the end of the horizon by setting e_max_pu and e_min_pu
                n.stores_t.e_max_pu.loc[snapshots[end-1],seasonal_stores] = pre_market.stores_t.e.loc[snapshots[end-1], seasonal_stores].div(
                    pre_market.stores.e_nom_opt[seasonal_stores]
                    )
                n.stores_t.e_min_pu.loc[snapshots[end-1],seasonal_stores] = pre_market.stores_t.e.loc[snapshots[end-1], seasonal_stores].div(
                    pre_market.stores.e_nom_opt[seasonal_stores]
                    )
                n.stores_t.e_min_pu.fillna(0., inplace=True)
                n.stores_t.e_max_pu.fillna(1., inplace=True)

            if not n.storage_units.empty:
                n.storage_units.state_of_charge_initial = (
                    n.storage_units_t.state_of_charge.loc[snapshots[start - 1]]
                )

       
        status, condition = n.optimize(sns, **kwargs)
        
        if status != "ok":
            logger.warning(
                f"Optimization failed with status {status} and condition {condition}"
            )
    return n

def build_market_model(self):
    """Builds market model based on imported network from eTraGo


    - import market regions from file or database
    - Cluster network to market regions
    -- consider marginal cost incl. generator noise when grouoping electrical
        generation capacities

    Returns
    -------
    None.

    """

    # use existing preprocessing to get only the electricity system

    net, weight, n_clusters, busmap_foreign = preprocessing(self)

    df = pd.DataFrame(
        {
            "country": net.buses.country.unique(),
            "marketzone": net.buses.country.unique(),
        },
        columns=["country", "marketzone"],
    )

    df.loc[(df.country == "DE") | (df.country == "LU"), "marketzone"] = "DE/LU"

    df["cluster"] = df.groupby(df.marketzone).grouper.group_info[0]

    for i in net.buses.country.unique():
        net.buses.loc[net.buses.country == i, "cluster"] = df.loc[
            df.country == i, "cluster"
        ].values[0]

    busmap = pd.Series(
        net.buses.cluster.astype(int).astype(str), net.buses.index
    )
    medoid_idx = pd.Series(dtype=str)

    logger.info("Start market zone specifc clustering")

    self.clustering, busmap = postprocessing(
        self,
        busmap,
        busmap_foreign,
        medoid_idx,
        aggregate_generators_carriers=[],
        aggregate_links=False,
    )

    self.update_busmap(busmap)

    net = self.clustering.network
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

    net.generators_t.p_max_pu = self.network.generators_t.p_max_pu

    self.pre_market_model = net


def build_shortterm_market_model(self):
    m = self.pre_market_model

    m.storage_units.p_nom_extendable = False
    m.stores.e_nom_extendable = False
    m.links.p_nom_extendable = False
    m.lines.s_nom_extendable = False

    m.storage_units.p_nom = m.storage_units.p_nom_opt
    m.stores.e_nom = m.stores.e_nom_opt
    m.links.p_nom = m.links.p_nom_opt
    m.lines.s_nom = m.lines.s_nom_opt

    # ToDo maybe ?!
    # somoehow setting seasonal storage (phs, h2 or finding a dynamic
    # definition with respect to the results i.e. the storage behavior)
    # filling level (or similar) for the short-term rolling complicated
    # market problem

    # set UC constraints

    unit_commitment = pd.read_csv("./data/unit_commitment.csv", index_col=0)
    unit_commitment.fillna(0, inplace=True)
    committable_attrs = m.generators.carrier.isin(unit_commitment).to_frame(
        "committable"
    )

    for attr in unit_commitment.index:
        default = component_attrs["Generator"].default[attr]
        committable_attrs[attr] = m.generators.carrier.map(
            unit_commitment.loc[attr]
        ).fillna(default)
        committable_attrs[attr] = committable_attrs[attr].astype(
            m.generators.carrier.map(unit_commitment.loc[attr]).dtype
        )

    m.generators[committable_attrs.columns] = committable_attrs
    m.generators.min_up_time = m.generators.min_up_time.astype(int)
    m.generators.min_down_time = m.generators.min_down_time.astype(int)

    # Tadress link carriers i.e. OCGT
    committable_links = m.links.carrier.isin(unit_commitment).to_frame(
        "committable"
    )

    for attr in unit_commitment.index:
        default = component_attrs["Link"].default[attr]
        committable_links[attr] = m.links.carrier.map(
            unit_commitment.loc[attr]
        ).fillna(default)
        committable_links[attr] = committable_links[attr].astype(
            m.links.carrier.map(unit_commitment.loc[attr]).dtype
        )

    m.links[committable_links.columns] = committable_links
    m.links.min_up_time = m.links.min_up_time.astype(int)
    m.links.min_down_time = m.links.min_down_time.astype(int)

    self.market_model = m


def extra_functionality():
    """Placeholder for extra functionalities within market optimization

    Returns
    -------
    None.

    """

    return None
