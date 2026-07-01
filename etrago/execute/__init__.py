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
execute.py defines optimization and simulation methods for the etrago object.
"""

import os

if "READTHEDOCS" not in os.environ:
    import logging
    import shutil
    import time

    from pypsa.linopf import network_lopf
    from pypsa.pf import sub_network_pf
    import numpy as np
    import pandas as pd

    from etrago.tools.constraints import (
        Constraints,
        _get_crossborder_components,
    )

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = (
    "ulfmueller, s3pp, wolfbunke, mariusves, lukasol, KathiEsterl, "
    "ClaraBuettner, CarlosEpia, AmeliaNadal"
)


def update_electrical_parameters(network, l_snom_pre, t_snom_pre):
    """
    Update electrical parameters of active branch components
    considering s_nom of previous iteration.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    l_snom_pre: pandas.Series
        s_nom of ac-lines in previous iteration.
    t_snom_pre: pandas.Series
        s_nom of transformers in previous iteration.

    Returns
    -------
    None.

    """

    network.lines.x[network.lines.s_nom_extendable] = (
        network.lines.x * l_snom_pre / network.lines.s_nom_opt
    )

    network.transformers.x[network.transformers.s_nom_extendable] = (
        network.transformers.x * t_snom_pre / network.transformers.s_nom_opt
    )

    network.lines.r[network.lines.s_nom_extendable] = (
        network.lines.r * l_snom_pre / network.lines.s_nom_opt
    )

    network.transformers.r[network.transformers.s_nom_extendable] = (
        network.transformers.r * t_snom_pre / network.transformers.s_nom_opt
    )

    network.lines.g[network.lines.s_nom_extendable] = (
        network.lines.g * network.lines.s_nom_opt / l_snom_pre
    )

    network.transformers.g[network.transformers.s_nom_extendable] = (
        network.transformers.g * network.transformers.s_nom_opt / t_snom_pre
    )

    network.lines.b[network.lines.s_nom_extendable] = (
        network.lines.b * network.lines.s_nom_opt / l_snom_pre
    )

    network.transformers.b[network.transformers.s_nom_extendable] = (
        network.transformers.b * network.transformers.s_nom_opt / t_snom_pre
    )

    # Set snom_pre to s_nom_opt for next iteration
    l_snom_pre = network.lines.s_nom_opt.copy()
    t_snom_pre = network.transformers.s_nom_opt.copy()

    return l_snom_pre, t_snom_pre


def run_lopf(etrago, extra_functionality, method):
    """
    Function that performs lopf with or without pyomo

    Parameters
    ----------
    etrago : etrago object
        eTraGo containing all network information and a PyPSA network.
    extra_functionality: dict
        Define extra constranits.
    method: dict
        Choose 'n_iter' and integer for fixed number of iterations or
        'threshold' and derivation of objective in percent for variable number
        of iteration until the threshold of the objective function is reached.

    Returns
    -------
    None.

    """

    x = time.time()

    if method["formulation"] == "pyomo":
        etrago.network.lopf(
            etrago.network.snapshots,
            solver_name=etrago.args["solver"],
            solver_options=etrago.args["solver_options"],
            pyomo=True,
            extra_functionality=extra_functionality,
            formulation=etrago.args["model_formulation"],
        )

        if etrago.network.results["Solver"][0]["Status"] != "ok":
            raise Exception("LOPF not solved.")

    elif method["formulation"] == "linopy":
        status, condition = etrago.network.optimize(
            solver_name=etrago.args["solver"],
            solver_options=etrago.args["solver_options"],
            extra_functionality=extra_functionality,
        )
        if status != "ok":
            logger.warning(f"""Optimization failed with status {status}
                and condition {condition}""")
            etrago.network.model.print_infeasibilities()
            import pdb

            pdb.set_trace()
    else:
        status, termination_condition = network_lopf(
            etrago.network,
            solver_name=etrago.args["solver"],
            solver_options=etrago.args["solver_options"],
            extra_functionality=extra_functionality,
            formulation=etrago.args["model_formulation"],
        )

        if status != "ok":
            raise Exception("LOPF not solved.")

    y = time.time()
    z = (y - x) / 60

    print("Time for LOPF [min]:", round(z, 2))


def iterate_lopf(
    etrago,
    extra_functionality,
    method={"n_iter": 4, "pyomo": True},
):
    """
    Run optimization of lopf. If network extension is included, the specified
    number of iterations is calculated to consider reactance changes.

    Parameters
    ----------
    etrago : etrago object
        eTraGo containing all network information and a PyPSA network.
    extra_functionality: dict
        Define extra constranits.
    method: dict
        Choose 'n_iter' and integer for fixed number of iterations or
        'threshold' and derivation of objective in percent for variable number
        of iteration until the threshold of the objective function is reached.

    """

    args = etrago.args
    lp_path = args["lpfile"]

    if args["export_results_path"]:
        path = args["export_results_path"] + "/grid_optimization"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    if args["temporal_disaggregation"]["active"]:
        if args["lpfile"]:
            lp_path = lp_path[0:-3] + "_temporally_reduced.lp"

    network = etrago.network

    # if network is extendable, iterate lopf
    # to include changes of electrical parameters
    if network.lines.s_nom_extendable.any():
        # Initialise s_nom_pre (s_nom_opt of previous iteration)
        # to s_nom for first lopf:
        l_snom_pre = network.lines.s_nom.copy()
        t_snom_pre = network.transformers.s_nom.copy()

        # calculate fixed number of iterations
        if "n_iter" in method:
            n_iter = method["n_iter"]

            for i in range(1, (1 + n_iter)):
                run_lopf(etrago, extra_functionality, method)

                if args["export_results_path"]:
                    # Store intermediate iterations
                    if i < n_iter:
                        path_it = path + "/lopf_iteration_" + str(i)
                        etrago.network.export_to_csv_folder(path_it)
                    # Store final result
                    else:
                        etrago.network.export_to_csv_folder(path)

                    # Delete previous iteration's results
                    if i > 1:
                        path_prev = path + "/lopf_iteration_" + str(i - 1)
                        if os.path.exists(path_prev):
                            shutil.rmtree(path_prev)

                if i < n_iter:
                    l_snom_pre, t_snom_pre = update_electrical_parameters(
                        network, l_snom_pre, t_snom_pre
                    )

        # Calculate variable number of iterations until threshold of objective
        # function is reached

        if "threshold" in method:
            run_lopf(etrago, extra_functionality, method)

            diff_obj = network.objective * method["threshold"] / 100

            i = 1

            # Stop after 100 iterations to aviod unending loop
            while i <= 100:
                if i == 100:
                    print("Maximum number of iterations reached.")
                    break

                l_snom_pre, t_snom_pre = update_electrical_parameters(
                    network, l_snom_pre, t_snom_pre
                )
                pre = network.objective

                run_lopf(etrago, extra_functionality, method)

                i += 1

                if args["export_results_path"]:
                    path_it = path + "/lopf_iteration_" + str(i)
                    etrago.network.export_to_csv_folder(path_it)

                if abs(pre - network.objective) <= diff_obj:
                    print("Threshold reached after " + str(i) + " iterations.")
                    break

    else:
        run_lopf(etrago, extra_functionality, method)
        if etrago.args["export_results_path"]:
            etrago.network.export_to_csv_folder(path)

    if args["lpfile"]:
        network.model.write(lp_path)

    return network


def lopf(self):
    """
    Functions that runs lopf according to arguments.

    Returns
    -------
    None.

    """

    x = time.time()

    self.conduct_dispatch_disaggregation = False

    if self.args["snapshot_clustering"]["active"] & (
        self.args["snapshot_clustering"]["method"] == "typical_periods"
    ):
        iterate_lopf(
            self,
            Constraints(
                self.args,
                self.conduct_dispatch_disaggregation,
                cluster_temporal=self.cluster_temporal,
                cluster_ts=self.cluster_ts,
            ).functionality,
            method=self.args["method"],
        )
    else:
        iterate_lopf(
            self,
            Constraints(
                self.args, self.conduct_dispatch_disaggregation
            ).functionality,
            method=self.args["method"],
        )

    y = time.time()
    z = (y - x) / 60
    logger.info("Time for LOPF [min]: {}".format(round(z, 2)))


def optimize(self):
    """Run optimization of dispatch and grid and storage expansion based on
    arguments

    Returns
    -------
    None.

    """

    if self.args["method"]["market_optimization"]["active"]:
        self.market_optimization()

        if self.args["scn_name"] in [
            "eGon100RE",
            "powerd2025",
            "powerd2030",
            "powerd2035",
        ]:
            self.network = self.adjust_PtH2_model(apply_on="grid_model")
            logger.info("PtH2-Model adjusted in network")

            self.network = self.adjust_chp_model(apply_on="grid_model")
            logger.info("CHP model in foreign countries adjusted in network")
        # self.market_results_to_grid()

        self.grid_optimization()

    elif self.args["method"]["type"] == "lopf":

        if self.args["scn_name"] in [
            "eGon100RE",
            "powerd2025",
            "powerd2030",
            "powerd2035",
        ]:
            self.network = self.adjust_PtH2_model(apply_on="grid_model")
            logger.info("PtH2-Model adjusted in network")

            self.network = self.adjust_chp_model(apply_on="grid_model")
            logger.info("CHP model in foreign countries adjusted in network")

        self.lopf()

    elif self.args["method"]["type"] == "sclopf":

        self.network = self.adjust_PtH2_model(apply_on="grid_model")
        logger.info("PtH2-Model adjusted in network")

        self.sclopf(
            post_lopf=False,
            n_process=4,
            delta=0.01,
            n_overload=0,
            div_ext_lines=False,
        )
    else:
        print("Method not defined")


def optimize_with_rolling_horizon(
    n,
    pre_market,
    snapshots,
    horizon,
    overlap,
    solver_name,
    extra_functionality,
    args,
    temporal_disaggregation=False,
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
        logger.info(f"""Optimizing network for snapshot horizon
            [{sns[0]}:{sns[-1]}] ({i+1}/{len(starting_points)}).""")

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
            n.stores_t.e_max_pu.loc[
                snapshots[end - 1], seasonal_stores
            ] = pre_market.stores_t.e.loc[
                snapshots[end - 1], seasonal_stores
            ].div(
                pre_market.stores.e_nom_opt[seasonal_stores]
            ).clip(
                lower=0.0
            ) * (
                1 + 1e6
            )
            n.stores_t.e_min_pu.loc[
                snapshots[end - 1], seasonal_stores
            ] = pre_market.stores_t.e.loc[
                snapshots[end - 1], seasonal_stores
            ].div(
                pre_market.stores.e_nom_opt[seasonal_stores]
            ).clip(
                lower=0.0
            ) * (
                1 - 1e6
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

        if temporal_disaggregation:
            args = update_extra_functionality_temporal_disaggregation(
                n, pre_market, args, sns
            )

        if args["method"]["formulation"] == "linopy":
            status, condition = n.optimize(
                sns,
                solver_name=solver_name,
                extra_functionality=extra_functionality,
                assign_all_duals=True,
                solver_options=args["solver_options"],
                linearized_unit_commitment=True,
            )

            if status != "ok":
                logger.warning(f"""Optimization failed with status {status}
                    and condition {condition}""")
                n.model.print_infeasibilities()
                import pdb

                pdb.set_trace()

        else:
            n.lopf(
                sns,
                solver_name=solver_name,
                solver_options=args["solver_options"],
                pyomo=True,
                extra_functionality=extra_functionality,
                formulation=args["model_formulation"],
            )

    return n


def update_extra_functionality_temporal_disaggregation(n, n_tsa, args, sns):

    for key in args["extra_functionality"].keys():
        if key == "cross_border_flow_per_country":
            args = adjust_crossborder_flow_rolling_horizon(n, n_tsa, args, sns)
            logger.info("Cross border flow constraint adjusted")

        elif key in [
            "max_line_ext",
            "max_battery_expansion_germany",
            "fixed_battery_expansion_germany",
            "min_ely_capacity_germany",
        ]:
            logger.info(
                "No adjustment for temporal disaggregation in constraint "
                f"{key} needed."
            )
        else:
            logger.warning(
                "Adjustment for temporal disaggregation in constraint "
                f"{key} needed but not implemented yet. Results of "
                "temporal disaggregation might be incorrect."
            )
    return args


def adjust_crossborder_flow_rolling_horizon(n, n_tsa, args, sns):

    buses_de, buses_for, cb0, cb1, cb0_link, cb1_link = (
        _get_crossborder_components(n)
    )

    line0 = n.lines.loc[cb0]
    line0["country0"] = n.buses.loc[line0.bus0.values, "country"].values
    line0["country1"] = n.buses.loc[line0.bus1.values, "country"].values

    line1 = n.lines.loc[cb1]
    line1["country0"] = n.buses.loc[line1.bus0.values, "country"].values
    line1["country1"] = n.buses.loc[line1.bus1.values, "country"].values

    link0 = n.links.loc[cb0_link]
    link0["country0"] = n.buses.loc[link0.bus0.values, "country"].values
    link0["country1"] = n.buses.loc[link0.bus1.values, "country"].values

    link1 = n.links.loc[cb1_link]
    link1["country0"] = n.buses.loc[link1.bus0.values, "country"].values
    link1["country1"] = n.buses.loc[link1.bus1.values, "country"].values

    n_tsa_sns = n_tsa.snapshots[n_tsa.snapshots.isin(sns)]
    for cntr in args["extra_functionality"][
        "cross_border_flow_per_country"
    ].keys():
        usage_tsa = (
            (
                n_tsa.lines_t.p0.loc[
                    n_tsa_sns, line1[line1.country1 == cntr].index
                ]
                .sum(axis=1)
                .mul(n_tsa.snapshot_weightings.loc[n_tsa_sns, "objective"])
                .sum()
            )
            - (
                n_tsa.lines_t.p0.loc[
                    n_tsa_sns, line0[line0.country0 == cntr].index
                ]
                .sum(axis=1)
                .mul(n_tsa.snapshot_weightings.loc[n_tsa_sns, "objective"])
                .sum()
            )
            + (
                n_tsa.links_t.p0.loc[
                    n_tsa_sns, link1[link1.country1 == cntr].index
                ]
                .sum(axis=1)
                .mul(n_tsa.snapshot_weightings.loc[n_tsa_sns, "objective"])
                .sum()
            )
            - (
                n_tsa.links_t.p0.loc[
                    n_tsa_sns, link0[link0.country0 == cntr].index
                ]
                .sum(axis=1)
                .mul(n_tsa.snapshot_weightings.loc[n_tsa_sns, "objective"])
                .sum()
            )
        )
        if usage_tsa >= 0:
            args["extra_functionality"]["cross_border_flow_per_country"][cntr][
                0
            ] = (0.99 * usage_tsa)
            args["extra_functionality"]["cross_border_flow_per_country"][cntr][
                1
            ] = (1.01 * usage_tsa)
        else:
            args["extra_functionality"]["cross_border_flow_per_country"][cntr][
                1
            ] = (0.99 * usage_tsa)
            args["extra_functionality"]["cross_border_flow_per_country"][cntr][
                0
            ] = (1.01 * usage_tsa)

    return args


def import_gen_from_links(network, drop_small_capacities=True):
    """
    create gas generators from links in order to not lose them when
    dropping non-electric carriers
    """

    if drop_small_capacities:
        # Discard all generators < 1kW
        discard_gen = network.links[network.links["p_nom"] <= 0.001].index
        network.links.drop(discard_gen, inplace=True)
        for df in network.links_t:
            if not network.links_t[df].empty:
                network.links_t[df].drop(
                    columns=discard_gen.values, inplace=True, errors="ignore"
                )
    # Select links that should be represented as generators
    gas_to_add = network.links[
        network.links.carrier.isin(
            [
                "central_gas_CHP",
                "OCGT",
                "H2_to_power",
                "industrial_gas_CHP",
            ]
        )
    ].copy()

    # Rename bus1 column to bus
    gas_to_add.rename(columns={"bus1": "bus"}, inplace=True)

    # Aggregate new generators per bus and carrier
    df = pd.DataFrame()
    df["p_nom"] = gas_to_add.groupby(["bus", "carrier"]).p_nom.sum()
    df["p_nom_opt"] = gas_to_add.groupby(["bus", "carrier"]).p_nom_opt.sum()
    df["p_nom_max"] = gas_to_add.groupby(["bus", "carrier"]).p_nom_max.sum()
    df["p_nom_min"] = gas_to_add.groupby(["bus", "carrier"]).p_nom_min.sum()
    df["p_nom_extendable"] = gas_to_add.groupby(
        ["bus", "carrier"]
    ).p_nom_extendable.any()
    df["marginal_cost"] = gas_to_add.groupby(
        ["bus", "carrier"]
    ).marginal_cost.mean()
    df["efficiency"] = gas_to_add.groupby(["bus", "carrier"]).efficiency.mean()
    df["control"] = "PV"
    df.reset_index(inplace=True)

    if not df.empty:
        df.index = df.bus + " " + df.carrier

    # Aggregate disptach time series for new generators
    gas_to_add["bus1_carrier"] = gas_to_add.bus + " " + gas_to_add.carrier

    if not network.links_t.p1.empty:
        df_t = (
            network.links_t.p1[gas_to_add.index]
            .groupby(gas_to_add.bus1_carrier, axis=1)
            .sum()
            * -1
        )

    # Insert aggregated generators their dispatch time series
    network.madd("Generator", df.index, **df)
    if not network.links_t.p1.empty:
        network.import_series_from_dataframe(df_t, "Generator", "p")
        network.import_series_from_dataframe(
            pd.DataFrame(index=df_t.index, columns=df_t.columns, data=1.0),
            "Generator",
            "status",
        )

    # Drop links now modelled as generator
    network.mremove("Link", gas_to_add.index)

    return


def run_pf_post_lopf(self):
    """
    Function that runs pf_post_lopf according to arguments.

    Returns
    -------
    None.

    """

    if self.args["pf_post_lopf"]["active"]:
        pf_post_lopf(self)


def pf_post_lopf(etrago, calc_losses=False):
    """
    Function that prepares and runs non-linar load flow using PyPSA pf.
    If crossborder lines are DC-links, pf is only applied on german network.
    Crossborder flows are still considerd due to the active behavior of links.
    To return a network containing the whole grid, the optimised solution of
    the foreign components can be added afterwards.

    Parameters
    ----------
    etrago : etrago object
        eTraGo containing all network information and a PyPSA network.
    add_foreign_lopf: boolean
        Choose if foreign results of lopf should be added to the network when
        foreign lines are DC.
    q_allocation: str
        Choose allocation of reactive power. Possible settings are listed in
        distribute_q function.
    calc_losses: bolean
        Choose if line losses will be calculated.

    Returns
    -------

    """

    def drop_foreign_components(network):
        """
        Function that drops foreign components which are only connected via
        DC-links and saves their optimization results in pd.DataFrame.

        Parameters
        ----------
        network : pypsa.Network object
            Container for all network components.

        Returns
        -------
        None.

        """

        # Create series for constant loads
        constant_loads = network.loads[network.loads.p_set != 0]["p_set"]
        for load in constant_loads.index:
            network.loads_t.p_set[load] = constant_loads[load]
        network.loads.p_set = 0

        n_bus = pd.Series(index=network.sub_networks.index)

        for i in network.sub_networks.index:
            n_bus[i] = len(network.buses.index[network.buses.sub_network == i])

        sub_network_DE = n_bus.index[n_bus == n_bus.max()]

        foreign_bus = network.buses[
            (network.buses.sub_network != sub_network_DE.values[0])
            & (network.buses.country != "DE")
        ]

        foreign_comp = {
            "Bus": network.buses[network.buses.index.isin(foreign_bus.index)],
            "Generator": network.generators[
                network.generators.bus.isin(foreign_bus.index)
            ],
            "Load": network.loads[network.loads.bus.isin(foreign_bus.index)],
            "Transformer": network.transformers[
                network.transformers.bus0.isin(foreign_bus.index)
            ],
            "StorageUnit": network.storage_units[
                network.storage_units.bus.isin(foreign_bus.index)
            ],
            "Store": network.stores[
                network.stores.bus.isin(foreign_bus.index)
            ],
        }

        foreign_series = {
            "Bus": network.buses_t.copy(),
            "Generator": network.generators_t.copy(),
            "Load": network.loads_t.copy(),
            "Transformer": network.transformers_t.copy(),
            "StorageUnit": network.storage_units_t.copy(),
            "Store": network.stores_t.copy(),
        }

        for comp in sorted(foreign_series):
            attr = sorted(foreign_series[comp])
            for a in attr:
                if (
                    not foreign_series[comp][a].empty
                    and not (foreign_series[comp][a] == 0.0).all().all()
                ):
                    if a not in ["mu_lower", "mu_upper", "p_max_pu"]:
                        if a in ["q_set", "e_max_pu", "e_min_pu"]:
                            g_in_q_set = foreign_comp[comp][
                                foreign_comp[comp].index.isin(
                                    foreign_series[comp][a].columns
                                )
                            ]
                            foreign_series[comp][a] = foreign_series[comp][a][
                                g_in_q_set.index
                            ]
                        else:
                            foreign_series[comp][a] = foreign_series[comp][a][
                                foreign_comp[comp].index
                            ]

                    else:
                        foreign_series[comp][a] = foreign_series[comp][a][
                            foreign_comp[comp][
                                foreign_comp[comp].index.isin(
                                    network.generators_t.p_max_pu.columns
                                )
                            ].index
                        ]

        # Drop components
        network.buses = network.buses.drop(foreign_bus.index)
        network.generators = network.generators[
            network.generators.bus.isin(network.buses.index)
        ]
        network.loads = network.loads[
            network.loads.bus.isin(network.buses.index)
        ]
        network.transformers = network.transformers[
            network.transformers.bus0.isin(network.buses.index)
        ]
        network.storage_units = network.storage_units[
            network.storage_units.bus.isin(network.buses.index)
        ]
        network.stores = network.stores[
            network.stores.bus.isin(network.buses.index)
        ]

        return foreign_bus, foreign_comp, foreign_series

    x = time.time()
    network = etrago.network
    args = etrago.args

    network.lines.s_nom = network.lines.s_nom_opt
    network.links.loc[network.links.p_nom_extendable, "p_nom"] = (
        network.links.loc[network.links.p_nom_extendable, "p_nom_opt"]
    )

    # generators modeled as links are imported to the generators table
    import_gen_from_links(network)

    if args["spatial_disaggregation"]:
        import_gen_from_links(
            etrago.disaggregated_network, drop_small_capacities=False
        )

    # For the PF, set the P to be the optimised P
    network.generators_t.p_set = network.generators_t.p_set.reindex(
        columns=network.generators.index
    )
    network.generators_t.p_set = network.generators_t.p

    network.storage_units_t.p_set = network.storage_units_t.p_set.reindex(
        columns=network.storage_units.index
    )
    network.storage_units_t.p_set = network.storage_units_t.p

    network.stores_t.p_set = network.stores_t.p_set.reindex(
        columns=network.stores.index
    )
    network.stores_t.p_set = network.stores_t.p

    network.links_t.p_set = network.links_t.p_set.reindex(
        columns=network.links.index
    )
    network.links_t.p_set = network.links_t.p0

    network.determine_network_topology()

    # if foreign lines are DC, execute pf only on sub_network in Germany
    if (args["foreign_lines"]["carrier"] == "DC") or (
        (args["scn_extension"] is not None)
        and ("BE_NO_NEP 2035" in args["scn_extension"])
    ):
        foreign_bus, foreign_comp, foreign_series = drop_foreign_components(
            network
        )

    # Assign generators control strategy
    ac_bus = network.buses[network.buses.carrier == "AC"]
    network.generators.control[network.generators.bus.isin(ac_bus.index)] = (
        "PV"
    )
    network.generators.control[
        network.generators.carrier.str.contains("load shedding")
    ] = "PQ"

    # Assign storage units control strategy
    network.storage_units.control[
        network.storage_units.bus.isin(ac_bus.index)
    ] = "PV"

    # Find out the name of the main subnetwork
    main_subnet = str(network.buses.sub_network.value_counts().argmax())

    # Delete very small p_set and q_set values to avoid problems when solving
    network.generators_t["p_set"][
        np.abs(network.generators_t["p_set"]) < 0.001
    ] = 0
    network.generators_t["q_set"][
        np.abs(network.generators_t["q_set"]) < 0.001
    ] = 0
    network.loads_t["p_set"][np.abs(network.loads_t["p_set"]) < 0.001] = 0
    network.loads_t["q_set"][np.abs(network.loads_t["q_set"]) < 0.001] = 0
    network.storage_units_t["p_set"][
        np.abs(network.storage_units_t["p_set"]) < 0.001
    ] = 0
    network.storage_units_t["q_set"][
        np.abs(network.storage_units_t["p_set"]) < 0.001
    ] = 0

    # execute non-linear pf
    pf_solution = sub_network_pf(
        sub_network=network.sub_networks["obj"][main_subnet],
        snapshots=network.snapshots,
        use_seed=True,
        distribute_slack=True,
    )

    pf_solve = pd.DataFrame(index=pf_solution[0].index)
    pf_solve["converged"] = pf_solution[2].values
    pf_solve["error"] = pf_solution[1].values
    pf_solve["n_iter"] = pf_solution[0].values

    if not pf_solve[~pf_solve.converged].count().max() == 0:
        logger.warning(
            "PF of %d snapshots not converged. \n"
            "Generation time series are set to zero for these time steps.",
            pf_solve[~pf_solve.converged].count().max(),
        )
        network.generators_t.p.loc[~pf_solve.converged, :] *= 0
        network.generators_t.q.loc[~pf_solve.converged, :] *= 0
        network.storage_units_t.p.loc[~pf_solve.converged, :] *= 0
        network.storage_units_t.q.loc[~pf_solve.converged, :] *= 0

    if calc_losses:
        calc_line_losses(network, pf_solve["converged"])

    network = distribute_q(
        network, etrago.args["pf_post_lopf"]["q_allocation"]
    )

    y = time.time()
    z = (y - x) / 60
    print("Time for PF [min]:", round(z, 2))

    # if selected, copy lopf results of neighboring countries to network
    if (
        (args["foreign_lines"]["carrier"] == "DC")
        or (
            (args["scn_extension"] is not None)
            and ("BE_NO_NEP 2035" in args["scn_extension"])
        )
    ) and etrago.args["pf_post_lopf"]["add_foreign_lopf"]:
        for comp in sorted(foreign_series):
            network.import_components_from_dataframe(foreign_comp[comp], comp)

            for attr in sorted(foreign_series[comp]):
                network.import_series_from_dataframe(
                    foreign_series[comp][attr], comp, attr
                )

    if args["export_results_path"]:
        path = args["export_results_path"] + "/pf_post_lopf"
        etrago.network.export_to_csv_folder(path)
        pf_solve.to_csv(os.path.join(path, "pf_solution.csv"), index=True)

        # Save un-solved disaggregated network
        if args["spatial_disaggregation"]:
            for comp_df in [
                etrago.disaggregated_network.transformers,
                etrago.disaggregated_network.lines,
                etrago.disaggregated_network.links,
                etrago.disaggregated_network.buses,
            ]:
                if "geom" in comp_df.columns:
                    comp_df.drop(columns=["geom"], inplace=True)
                if "topo" in comp_df.columns:
                    comp_df.drop(columns=["topo"], inplace=True)
            etrago.disaggregated_network.export_to_netcdf(
                args["export_results_path"] + "/disaggregated_network.nc"
            )

    return network


def distribute_q(network, allocation="p_nom"):
    """
    Function that distributes reactive power at bus to all installed
    generators and storages.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    allocation: str
        Choose key to distribute reactive power:
        'p_nom' to dirstribute via p_nom
        'p' to distribute via p_set.

    Returns
    -------
    None.

    """

    ac_bus = network.buses[network.buses.carrier == "AC"]

    gen_elec = network.generators[
        (network.generators.bus.isin(ac_bus.index))
        & (network.generators.carrier != "load shedding")
    ].carrier.unique()

    network.allocation = allocation
    if allocation == "p":
        if (network.buses.carrier == "AC").all():
            p_sum = (
                network.generators_t["p"]
                .groupby(network.generators.bus, axis=1)
                .sum()
                .add(
                    network.storage_units_t["p"]
                    .abs()
                    .groupby(network.storage_units.bus, axis=1)
                    .sum(),
                    fill_value=0,
                )
            )
            q_sum = (
                network.generators_t["q"]
                .groupby(network.generators.bus, axis=1)
                .sum()
            )

            q_distributed = (
                network.generators_t.p
                / p_sum[network.generators.bus.sort_index()].values
                * q_sum[network.generators.bus.sort_index()].values
            )

            q_storages = (
                network.storage_units_t.p
                / p_sum[network.storage_units.bus.sort_index()].values
                * q_sum[network.storage_units.bus.sort_index()].values
            )
        else:
            print(
                """WARNING: Distribution of reactive power based on active
                  power is currently outdated for sector coupled models. This
                  process will continue with the option allocation = 'p_nom'"""
            )
            allocation = "p_nom"

    if allocation == "p_nom":
        q_bus = (
            network.generators_t["q"]
            .groupby(network.generators.bus, axis=1)
            .sum()
            .add(
                network.storage_units_t.q.groupby(
                    network.storage_units.bus, axis=1
                ).sum(),
                fill_value=0,
            )
        )

        total_q1 = q_bus.sum().sum()
        ac_bus = network.buses[network.buses.carrier == "AC"]

        gen_elec = network.generators[
            (network.generators.bus.isin(ac_bus.index))
            & (network.generators.carrier != "load shedding")
            & (network.generators.p_nom > 0)
        ].sort_index()

        q_distributed = q_bus[gen_elec.bus].multiply(gen_elec.p_nom.values) / (
            (
                gen_elec.p_nom.groupby(network.generators.bus)
                .sum()
                .reindex(network.generators.bus.unique(), fill_value=0)
                .add(
                    network.storage_units.p_nom_opt.groupby(
                        network.storage_units.bus
                    ).sum(),
                    fill_value=0,
                )
            )[gen_elec.bus.sort_index()].values
        )

        q_distributed.columns = gen_elec.index

        q_storages = q_bus[network.storage_units.bus].multiply(
            network.storage_units.p_nom_opt.values
        ) / (
            (
                gen_elec.p_nom.groupby(network.generators.bus)
                .sum()
                .add(
                    network.storage_units.p_nom_opt.groupby(
                        network.storage_units.bus
                    ).sum(),
                    fill_value=0,
                )
            )[network.storage_units.bus].values
        )

        q_storages.columns = network.storage_units.index

    q_distributed[q_distributed.isnull()] = 0
    q_distributed[q_distributed.abs() == np.inf] = 0
    q_storages[q_storages.isnull()] = 0
    q_storages[q_storages.abs() == np.inf] = 0
    network.generators_t.q = q_distributed
    network.storage_units_t.q = q_storages

    total_q2 = q_distributed.sum().sum() + q_storages.sum().sum()
    print(f"Error in q distribution={(total_q2 - total_q1)/total_q1}%")

    return network


def calc_line_losses(network, converged):
    """
    Calculate losses per line with PF result data.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    converged : pd.Series
        List of snapshots with their status (converged or not).

    Returns
    -------
    None.

    """
    # Line losses
    # calculate apparent power S = sqrt(p² + q²) [in MW]
    s0_lines = (network.lines_t.p0**2 + network.lines_t.q0**2).apply(np.sqrt)
    # in case some snapshots did not converge, discard them from the
    # calculation
    s0_lines.loc[converged[converged is False].index, :] = np.nan
    # calculate current I = S / U [in A]
    i0_lines = np.multiply(s0_lines, 1000000) / np.multiply(
        network.lines.v_nom, 1000
    )
    # calculate losses per line and timestep network.\
    # lines_t.line_losses = I² * R [in MW]
    network.lines_t.losses = np.divide(i0_lines**2 * network.lines.r, 1000000)
    # calculate total losses per line [in MW]
    network.lines = network.lines.assign(
        losses=np.sum(network.lines_t.losses).values
    )

    # Transformer losses
    # https://books.google.de/books?id=0glcCgAAQBAJ&pg=PA151&lpg=PA151&dq=
    # wirkungsgrad+transformator+1000+mva&source=bl&ots=a6TKhNfwrJ&sig=
    # r2HCpHczRRqdgzX_JDdlJo4hj-k&hl=de&sa=X&ved=
    # 0ahUKEwib5JTFs6fWAhVJY1AKHa1cAeAQ6AEIXjAI#v=onepage&q=
    # wirkungsgrad%20transformator%201000%20mva&f=false
    # Crastan, Elektrische Energieversorgung, p.151
    # trafo 1000 MVA: 99.8 %
    network.transformers = network.transformers.assign(
        losses=np.multiply(network.transformers.s_nom, (1 - 0.998)).values
    )

    main_subnet = str(network.buses.sub_network.value_counts().argmax())
    price_per_bus = network.buses_t.marginal_price[
        network.buses.sub_network[
            network.buses.sub_network == main_subnet
        ].index
    ]

    # calculate total losses (possibly enhance with adding these values
    # to network container)
    losses_total = sum(network.lines.losses) + sum(network.transformers.losses)
    print("Total lines losses for all snapshots [MW]:", round(losses_total, 2))
    losses_costs = losses_total * np.average(price_per_bus)
    print("Total costs for these losses [EUR]:", round(losses_costs, 2))
    if (~converged).sum() > 0:
        print(
            f"Note: {(~converged).sum()} snapshot(s) was/were excluded "
            + "because the PF did not converge"
        )


def set_slack(network):
    """
    Function that chosses the bus with the maximum installed power as slack.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.

    Returns
    -------
    network : pypsa.Network object
        Container for all network components.

    """

    old_slack = network.generators.index[
        network.generators.control == "Slack"
    ][0]
    # check if old slack was PV or PQ control:
    if network.generators.p_nom[old_slack] > 50 and network.generators.carrier[
        old_slack
    ] in ("solar", "wind"):
        old_control = "PQ"
    elif network.generators.p_nom[
        old_slack
    ] > 50 and network.generators.carrier[old_slack] not in ("solar", "wind"):
        old_control = "PV"
    elif network.generators.p_nom[old_slack] < 50:
        old_control = "PQ"

    old_gens = network.generators
    gens_summed = network.generators_t.p.sum()
    old_gens["p_summed"] = gens_summed
    max_gen_buses_index = (
        old_gens.groupby(["bus"])
        .agg({"p_summed": np.sum})
        .p_summed.sort_values()
        .index
    )

    for bus_iter in range(1, len(max_gen_buses_index) - 1):
        if old_gens[
            (network.generators["bus"] == max_gen_buses_index[-bus_iter])
            & (network.generators["control"] != "PQ")
        ].empty:
            continue
        else:
            new_slack_bus = max_gen_buses_index[-bus_iter]
            break

    network.generators = network.generators.drop(columns=["p_summed"])
    new_slack_gen = (
        network.generators.p_nom[
            (network.generators["bus"] == new_slack_bus)
            & (network.generators["control"] == "PV")
        ]
        .sort_values()
        .index[-1]
    )

    network.generators.at[old_slack, "control"] = old_control
    network.generators.at[new_slack_gen, "control"] = "Slack"

    return network
