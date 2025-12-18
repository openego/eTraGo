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
    from etrago.cluster.spatial import group_links
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

from etrago.tools.utilities import adjust_chp_model, adjust_PtH2_model


def market_optimization(self):
    logger.info("Start building pre market model")

    unit_commitment = True

    build_market_model(self, unit_commitment)
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
            logger.warning(
                f"""Optimization failed with status {status}
                and condition {condition}"""
            )

    else:
        logger.warning("Method type must be either 'pyomo' or 'linopy'")

    # Export results of pre-market model
    if self.args["csv_export"]:
        path = self.args["csv_export"]
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.pre_market_model.export_to_csv_folder(path + "/pre_market")
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
                ~n.stores.carrier.isin(
                    [
                        "PtH2_waste_heat",
                        "PtH2_O2",
                        "dsm",
                        "battery_storage",
                        "central_heat_store",
                        "H2_overground",
                        "CH4",
                        "H2_underground",
                    ]
                )
            ].index
            if start != 0:
                n.stores.loc[stores_no_dsm, "e_initial"] = n.stores_t.e.loc[
                    snapshots[start - 1], stores_no_dsm
                ]
            else:
                n.stores.loc[stores_no_dsm, "e_initial"] = (
                    pre_market.stores_t.e.loc[
                        snapshots[start - 1], stores_no_dsm
                    ]
                )

            # Select seasonal stores
            seasonal_stores = n.stores.index[
                n.stores.carrier.isin(
                    [
                        "central_heat_store",
                        "H2_overground",
                        "CH4",
                        "H2_underground",
                    ]
                )
            ]

            # Set e_initial from pre_market model for seasonal stores
            n.stores.e_initial[seasonal_stores] = pre_market.stores_t.e.loc[
                snapshots[start - 1], seasonal_stores
            ]

            # Set e at the end of the horizon
            # by setting e_max_pu and e_min_pu
            n.stores_t.e_max_pu.loc[snapshots[end - 1], seasonal_stores] = (
                pre_market.stores_t.e.loc[snapshots[end - 1], seasonal_stores]
                .div(pre_market.stores.e_nom_opt[seasonal_stores])
                .clip(lower=0.0)
                * 1.01
            )
            n.stores_t.e_min_pu.loc[snapshots[end - 1], seasonal_stores] = (
                pre_market.stores_t.e.loc[snapshots[end - 1], seasonal_stores]
                .div(pre_market.stores.e_nom_opt[seasonal_stores])
                .clip(lower=0.0)
                * 0.99
            )
            n.stores_t.e_min_pu.fillna(0.0, inplace=True)
            n.stores_t.e_max_pu.fillna(1.0, inplace=True)

        if not n.storage_units.empty:
            n.storage_units.state_of_charge_initial = (
                n.storage_units_t.state_of_charge.loc[snapshots[start - 1]]
            )
            # Make sure that state of charge of batteries and pumped hydro
            # plants are cyclic over the year by using the state_of_charges
            # from the pre_market_model
            if i == 0:
                n.storage_units.state_of_charge_initial = (
                    pre_market.storage_units_t.state_of_charge.iloc[-1]
                )
                seasonal_storage = pre_market.storage_units[
                    pre_market.storage_units.carrier == "reservoir"
                ].index

                soc_value = pre_market.storage_units_t.state_of_charge.loc[
                    snapshots[end - 1], seasonal_storage
                ]

                args_addition = {
                    "pre_market_seasonal_soc": soc_value,
                }

                extra_functionality = Constraints(
                    {**args, **args_addition}, False, apply_on="market_model"
                ).functionality

            elif i == len(starting_points) - 1:
                if len(snapshots) > 1000:
                    extra_functionality = Constraints(
                        args, False, apply_on="last_market_model"
                    ).functionality
            else:
                seasonal_storage = pre_market.storage_units[
                    pre_market.storage_units.carrier == "reservoir"
                ].index

                soc_value = pre_market.storage_units_t.state_of_charge.loc[
                    snapshots[end - 1], seasonal_storage
                ]

                args_addition = {
                    "pre_market_seasonal_soc": soc_value,
                }

                extra_functionality = Constraints(
                    {**args, **args_addition}, False, apply_on="market_model"
                ).functionality

        status, condition = n.optimize(
            sns,
            solver_name=solver_name,
            extra_functionality=extra_functionality,
            assign_all_duals=True,
            linearized_unit_commitment=True,
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

    gas_clustering_market_model(self)

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
        self, busmap, medoid_idx=medoid_idx, apply_on="market_model"
    )
