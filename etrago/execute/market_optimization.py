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
    from etrago.tools.constraints import Constraints

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
    self.pre_market_model.determine_network_topology()

    logger.info("Start solving pre market model")

    if self.args["method"]["formulation"] == "pyomo":
        self.pre_market_model.lopf(
            solver_name=self.args["solver"],
            solver_options=self.args["solver_options"],
            pyomo=True,
            extra_functionality=Constraints(self.args, False).functionality,
            formulation=self.args["model_formulation"],
        )
    elif self.args["method"]["formulation"] == "linopy":
        status, condition = self.pre_market_model.optimize(
            solver_name=self.args["solver"],
            solver_options=self.args["solver_options"],
            extra_functionality=Constraints(self.args, False).functionality,
            linearized_unit_commitment=True,
        )

        if status != "ok":
            logger.warning(
                f"""Optimization failed with status {status}
                and condition {condition}"""
            )
            self.pre_market_model.model.print_infeasibilities()
            import pdb

            pdb.set_trace()
    else:
        logger.warning("Method type must be either 'pyomo' or 'linopy'")

    # Export results of pre-market model
    if self.args["csv_export"]:
        path = self.args["csv_export"]
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.pre_market_model.export_to_csv_folder(path + "/pre_market")

    logger.info("Preparing short-term UC market model")

    build_shortterm_market_model(self)
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
    if self.args["csv_export"]:
        path = self.args["csv_export"]
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.market_model.export_to_csv_folder(path + "/market")


def optimize_with_rolling_horizon(
    n,
    pre_market,
    snapshots,
    horizon,
    overlap,
    solver_name,
    extra_functionality,
    args,
):
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

    # Make sure that quadratic costs as zero and not NaN
    n.links.marginal_cost_quadratic = 0.0

    starting_points = range(0, len(snapshots), horizon - overlap)
    for i, start in enumerate(starting_points):
        end = min(len(snapshots), start + horizon)
        sns = snapshots[start:end]
        logger.info(
            f"""Optimizing network for snapshot horizon
            [{sns[0]}:{sns[-1]}] ({i+1}/{len(starting_points)})."""
        )

        if not n.stores.empty:
            stores_no_dsm = n.stores[
                ~n.stores.carrier.isin(["dsm", "battery_storage"])
            ].index
            n.stores.loc[stores_no_dsm, "e_initial"] = n.stores_t.e.loc[
                snapshots[start - 1], stores_no_dsm
            ]

            # Select seasonal stores
            seasonal_stores = n.stores.index[
                n.stores.carrier.isin(
                    ["central_heat_store", "H2_overground", "CH4"]
                )
            ]

            # Set e_initial from pre_market model for seasonal stores
            n.stores.e_initial[seasonal_stores] = pre_market.stores_t.e.loc[
                snapshots[start - 1], seasonal_stores
            ]

            # Set e at the end of the horizon
            # by setting e_max_pu and e_min_pu
            n.stores_t.e_max_pu.loc[snapshots[end - 1], seasonal_stores] = (
                pre_market.stores_t.e.loc[
                    snapshots[end - 1], seasonal_stores
                ].div(pre_market.stores.e_nom_opt[seasonal_stores])
                * 1.01
            )
            n.stores_t.e_min_pu.loc[snapshots[end - 1], seasonal_stores] = (
                pre_market.stores_t.e.loc[
                    snapshots[end - 1], seasonal_stores
                ].div(pre_market.stores.e_nom_opt[seasonal_stores])
                * 0.99
            )
            n.stores_t.e_min_pu.fillna(0.0, inplace=True)
            n.stores_t.e_max_pu.fillna(1.0, inplace=True)

        if not n.storage_units.empty:
            n.storage_units.state_of_charge_initial = (
                n.storage_units_t.state_of_charge.loc[snapshots[start - 1]]
            )
            print(i)
            # Make sure that state of charge of batteries and pumped hydro
            # plants are cyclic over the year by using the state_of_charges
            # from the pre_market_model
            if i == 0:
                n.storage_units.state_of_charge_initial = (
                    pre_market.storage_units_t.state_of_charge.iloc[-1]
                )

            elif i == len(starting_points) - 1:
                extra_functionality = Constraints(
                    args, False, apply_on="last_market_model"
                ).functionality

        status, condition = n.optimize(
            sns,
            solver_name=solver_name,
            extra_functionality=extra_functionality,
        )

        if status != "ok":
            logger.warning(
                f"""Optimization failed with status {status}
                and condition {condition}"""
            )
            n.model.print_infeasibilities()
            import pdb

            pdb.set_trace()
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
    net, weight, n_clusters, busmap_foreign = preprocessing(
        self, apply_on="market_model"
    )

    # Define market regions based on settings.
    # Currently the only option is 'status_quo' which means that the current
    # regions are used. When other market zone options are introduced, they
    # can be assinged here.
    if (
        self.args["method"]["market_optimization"]["market_zones"]
        == "status_quo"
    ):
        df = pd.DataFrame(
            {
                "country": net.buses.country.unique(),
                "marketzone": net.buses.country.unique(),
            },
            columns=["country", "marketzone"],
        )

        df.loc[(df.country == "DE") | (df.country == "LU"), "marketzone"] = (
            "DE/LU"
        )

        df["cluster"] = df.groupby(df.marketzone).grouper.group_info[0]

        for i in net.buses.country.unique():
            net.buses.loc[net.buses.country == i, "cluster"] = df.loc[
                df.country == i, "cluster"
            ].values[0]

        busmap = pd.Series(
            net.buses.cluster.astype(int).astype(str), net.buses.index
        )
        medoid_idx = pd.Series(dtype=str)

    else:
        logger.warning(
            f"""
            Market zone setting {self.args['method']['market_zones']}
            is not available. Please use one of ['status_quo']."""
        )

    logger.info("Start market zone specifc clustering")

    self.clustering, busmap = postprocessing(
        self,
        busmap,
        busmap_foreign,
        medoid_idx,
        aggregate_generators_carriers=[],
        aggregate_links=False,
        apply_on="market_model",
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

    net.generators_t.p_max_pu = self.network_tsa.generators_t.p_max_pu

    # set UC constraints
    unit_commitment = pd.read_csv("./data/unit_commitment.csv", index_col=0)
    unit_commitment.fillna(0, inplace=True)
    committable_attrs = net.generators.carrier.isin(unit_commitment).to_frame(
        "committable"
    )

    for attr in unit_commitment.index:

        default = component_attrs["Generator"].default[attr]
        committable_attrs[attr] = net.generators.carrier.map(
            unit_commitment.loc[attr]
        ).fillna(default)
        committable_attrs[attr] = committable_attrs[attr].astype(
            net.generators.carrier.map(unit_commitment.loc[attr]).dtype
        )

    net.generators[committable_attrs.columns] = committable_attrs
    net.generators.min_up_time = net.generators.min_up_time.astype(int)
    net.generators.min_down_time = net.generators.min_down_time.astype(int)

    net.generators.loc[net.generators.committable, "ramp_limit_down"].fillna(
        1.0, inplace=True
    )

    # Set all start_up and shut_down cost to 0 to simpify unit committment
    net.generators.loc[net.generators.committable, "start_up_cost"] = 0.0
    net.generators.loc[net.generators.committable, "shut_down_cost"] = 0.0

    # Tadress link carriers i.e. OCGT
    committable_links = net.links.carrier.isin(unit_commitment).to_frame(
        "committable"
    )

    for attr in unit_commitment.index:

        default = component_attrs["Link"].default[attr]
        committable_links[attr] = net.links.carrier.map(
            unit_commitment.loc[attr]
        ).fillna(default)
        committable_links[attr] = committable_links[attr].astype(
            net.links.carrier.map(unit_commitment.loc[attr]).dtype
        )

    net.links[committable_links.columns] = committable_links
    net.links.min_up_time = net.links.min_up_time.astype(int)
    net.links.min_down_time = net.links.min_down_time.astype(int)
    net.links[committable_links.columns].loc["ramp_limit_down"] = 1.0
    net.links.loc[net.links.carrier.isin(["CH4", "DC", "AC"]), "p_min_pu"] = (
        -1.0
    )

    # Set all start_up and shut_down cost to 0 to simpify unit committment
    net.links.loc[net.links.committable, "start_up_cost"] = 0.0
    net.links.loc[net.links.committable, "shut_down_cost"] = 0.0

    # Set stores and storage_units to cyclic
    net.stores.loc[net.stores.carrier != "battery_storage", "e_cyclic"] = True
    net.storage_units.cyclic_state_of_charge = True

    self.pre_market_model = net

    # Set country tags for market model
    self.buses_by_country(apply_on="pre_market_model")
    self.geolocation_buses(apply_on="pre_market_model")


def build_shortterm_market_model(self):
    m = self.pre_market_model.copy()

    m.storage_units.p_nom_extendable = False
    m.stores.e_nom_extendable = False
    m.links.p_nom_extendable = False
    m.lines.s_nom_extendable = False

    m.storage_units.p_nom = m.storage_units.p_nom_opt
    m.stores.e_nom = m.stores.e_nom_opt
    m.links.p_nom = m.links.p_nom_opt
    m.lines.s_nom = m.lines.s_nom_opt

    m.stores.e_cyclic = False
    m.storage_units.cyclic_state_of_charge = False

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
    m.links.loc[m.links.carrier.isin(["CH4", "DC", "AC"]), "p_min_pu"] = -1.0

    m.generators.loc[m.generators.committable, "ramp_limit_down"].fillna(
        1.0, inplace=True
    )
    m.links.loc[m.links.committable, "ramp_limit_down"].fillna(
        1.0, inplace=True
    )

    self.market_model = m

    # Set country tags for market model
    self.buses_by_country(apply_on="market_model")
    self.geolocation_buses(apply_on="market_model")
