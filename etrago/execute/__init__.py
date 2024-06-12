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
    import time

    from pypsa.linopf import network_lopf
    from pypsa.pf import sub_network_pf
    import numpy as np
    import pandas as pd

    from etrago.tools.constraints import Constraints

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

    if etrago.conduct_dispatch_disaggregation is not False:
        # parameters defining the start and end per slices
        no_slices = etrago.args["temporal_disaggregation"]["no_slices"]
        skipped = etrago.network.snapshot_weightings.iloc[0].objective
        transits = np.where(
            etrago.network_tsa.snapshots.isin(
                etrago.conduct_dispatch_disaggregation.index
            )
        )[0]

        if method["pyomo"]:
            # repeat the optimization for all slices
            for i in range(0, no_slices):
                # keep information on the initial state of charge for the
                # respectng slice
                initial = transits[i - 1]
                soc_initial = etrago.conduct_dispatch_disaggregation.loc[
                    [etrago.network_tsa.snapshots[initial]]
                ].transpose()
                etrago.network_tsa.storage_units.state_of_charge_initial = (
                    soc_initial
                )
                etrago.network_tsa.stores.e_initial = soc_initial
                etrago.network_tsa.stores.e_initial.fillna(0, inplace=True)
                # the state of charge at the end of each slice is set within
                # split_dispatch_disaggregation_constraints in constraints.py

                # adapt start and end snapshot of respecting slice
                start = transits[i - 1] + skipped
                end = transits[i] + (skipped - 1)
                if i == 0:
                    start = 0
                if i == no_slices - 1:
                    end = len(etrago.network_tsa.snapshots)

                etrago.network_tsa.lopf(
                    etrago.network_tsa.snapshots[start : end + 1],
                    solver_name=etrago.args["solver"],
                    solver_options=etrago.args["solver_options"],
                    pyomo=True,
                    extra_functionality=extra_functionality,
                    formulation=etrago.args["model_formulation"],
                )

                if etrago.network_tsa.results["Solver"][0]["Status"] != "ok":
                    raise Exception("LOPF not solved.")

        else:
            for i in range(0, no_slices):
                status, termination_condition = network_lopf(
                    etrago.network_tsa,
                    etrago.network_tsa.snapshots[start : end + 1],
                    solver_name=etrago.args["solver"],
                    solver_options=etrago.args["solver_options"],
                    extra_functionality=extra_functionality,
                    formulation=etrago.args["model_formulation"],
                )

                if status != "ok":
                    raise Exception("LOPF not solved.")

        etrago.network_tsa.storage_units.state_of_charge_initial = 0
        etrago.network_tsa.stores.e_initial = 0

    else:
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
                formulation=etrago.args["model_formulation"],
            )
            if status != "ok":
                logger.warning(
                    f"""Optimization failed with status {status}
                    and condition {condition}"""
                )
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
    path = args["csv_export"]
    lp_path = args["lpfile"]

    if (
        args["temporal_disaggregation"]["active"] is True
        and etrago.conduct_dispatch_disaggregation is False
    ):
        if args["csv_export"]:
            path = path + "/temporally_reduced"

        if args["lpfile"]:
            lp_path = lp_path[0:-3] + "_temporally_reduced.lp"

    if etrago.conduct_dispatch_disaggregation is not False:
        if args["lpfile"]:
            lp_path = lp_path[0:-3] + "_dispatch_disaggregation.lp"

        etrago.network_tsa.lines["s_nom"] = etrago.network.lines["s_nom_opt"]
        etrago.network_tsa.lines["s_nom_extendable"] = False

        etrago.network_tsa.links["p_nom"] = etrago.network.links["p_nom_opt"]
        etrago.network_tsa.links["p_nom_extendable"] = False

        etrago.network_tsa.transformers["s_nom"] = etrago.network.transformers[
            "s_nom_opt"
        ]
        etrago.network_tsa.transformers.s_nom_extendable = False

        etrago.network_tsa.storage_units["p_nom"] = (
            etrago.network.storage_units["p_nom_opt"]
        )
        etrago.network_tsa.storage_units["p_nom_extendable"] = False

        etrago.network_tsa.stores["e_nom"] = etrago.network.stores["e_nom_opt"]
        etrago.network_tsa.stores["e_nom_extendable"] = False

        etrago.network_tsa.storage_units.cyclic_state_of_charge = False
        etrago.network_tsa.stores.e_cyclic = False

        args["snapshot_clustering"]["active"] = False
        args["skip_snapshots"] = False
        args["extendable"] = []

        network = etrago.network_tsa

    else:
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

                if args["csv_export"]:
                    path_it = path + "/lopf_iteration_" + str(i)
                    etrago.export_to_csv(path_it)

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

                if args["csv_export"]:
                    path_it = path + "/lopf_iteration_" + str(i)
                    etrago.export_to_csv(path_it)

                if abs(pre - network.objective) <= diff_obj:
                    print("Threshold reached after " + str(i) + " iterations.")
                    break

    else:
        run_lopf(etrago, extra_functionality, method)
        etrago.export_to_csv(path)

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

    if self.args["csv_export"]:
        path = self.args["csv_export"]
        if self.args["temporal_disaggregation"]["active"] is True:
            path = path + "/temporally_reduced"
        self.export_to_csv(path)


def optimize(self):
    """Run optimization of dispatch and grid and storage expansion based on
    arguments

    Returns
    -------
    None.

    """

    if self.args["method"]["market_optimization"]:
        self.market_optimization()

        # self.market_results_to_grid()

        self.grid_optimization()

    elif self.args["method"]["type"] == "lopf":

        self.lopf()

    elif self.args["method"]["type"] == "sclopf":
        self.sclopf(
            post_lopf=False,
            n_process=4,
            delta=0.01,
            n_overload=0,
            div_ext_lines=False,
        )
    else:
        print("Method not defined")


def dispatch_disaggregation(self):
    """
    Function running the tempral disaggregation meaning the optimization
    of dispatch in the temporally fully resolved network; therfore, the problem
    is reduced to smaller subproblems by slicing the whole considered time span
    while keeping inforation on the state of charge of storage units and stores
    to ensure compatibility and to reproduce saisonality.

    Returns
    -------
    None.

    """

    if self.args["temporal_disaggregation"]["active"]:
        x = time.time()

        if self.args["temporal_disaggregation"]["no_slices"]:
            # split dispatch_disaggregation into subproblems
            # keep some information on soc in beginning and end of slices
            # to ensure compatibility and to reproduce saisonality

            # define number of slices and corresponding slice length
            no_slices = self.args["temporal_disaggregation"]["no_slices"]
            slice_len = int(len(self.network.snapshots) / no_slices)

            # transition snapshots defining start and end of slices
            transits = self.network.snapshots[0::slice_len]
            if len(transits) > 1:
                transits = transits[1:]
            if transits[-1] != self.network.snapshots[-1]:
                transits = transits.insert(
                    (len(transits)), self.network.snapshots[-1]
                )
            # for stores, exclude emob and dsm because of their special
            # constraints
            sto = self.network.stores[
                ~self.network.stores.carrier.isin(
                    ["battery_storage", "battery storage", "dsm"]
                )
            ]

            # save state of charge of storage units and stores at those
            # transition snapshots
            self.conduct_dispatch_disaggregation = pd.DataFrame(
                columns=self.network.storage_units.index.append(sto.index),
                index=transits,
            )
            for storage in self.network.storage_units.index:
                self.conduct_dispatch_disaggregation[storage] = (
                    self.network.storage_units_t.state_of_charge[storage]
                )
            for store in sto.index:
                self.conduct_dispatch_disaggregation[store] = (
                    self.network.stores_t.e[store]
                )

            extra_func = self.args["extra_functionality"]
            self.args["extra_functionality"] = {}

        load_shedding = self.args["load_shedding"]
        if not load_shedding:
            self.args["load_shedding"] = True
            self.load_shedding(temporal_disaggregation=True)

        iterate_lopf(
            self,
            Constraints(
                self.args, self.conduct_dispatch_disaggregation
            ).functionality,
            method=self.args["method"],
        )

        # switch to temporally fully resolved network as standard network,
        # temporally reduced network is stored in network_tsa
        network1 = self.network.copy()
        self.network = self.network_tsa.copy()
        self.network_tsa = network1.copy()
        network1 = 0

        # keep original settings

        if self.args["temporal_disaggregation"]["no_slices"]:
            self.args["extra_functionality"] = extra_func
        self.args["load_shedding"] = load_shedding

        self.network.lines["s_nom_extendable"] = self.network_tsa.lines[
            "s_nom_extendable"
        ]
        self.network.links["p_nom_extendable"] = self.network_tsa.links[
            "p_nom_extendable"
        ]
        self.network.transformers.s_nom_extendable = (
            self.network_tsa.transformers.s_nom_extendable
        )
        self.network.storage_units["p_nom_extendable"] = (
            self.network_tsa.storage_units["p_nom_extendable"]
        )
        self.network.stores["e_nom_extendable"] = self.network_tsa.stores[
            "e_nom_extendable"
        ]
        self.network.storage_units.cyclic_state_of_charge = (
            self.network_tsa.storage_units.cyclic_state_of_charge
        )
        self.network.stores.e_cyclic = self.network_tsa.stores.e_cyclic

        if not self.args["csv_export"]:
            path = self.args["csv_export"]
            self.export_to_csv(path)
            self.export_to_csv(path + "/temporal_disaggregaton")

        y = time.time()
        z = (y - x) / 60
        logger.info("Time for LOPF [min]: {}".format(round(z, 2)))


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
                    if a != "p_max_pu":
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
        network.generators.carrier == "load shedding"
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
            "PF of %d snapshots not converged.",
            pf_solve[~pf_solve.converged].count().max(),
        )
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

    if args["csv_export"]:
        path = args["csv_export"] + "/pf_post_lopf"
        etrago.export_to_csv(path)
        pf_solve.to_csv(os.path.join(path, "pf_solution.csv"), index=True)

        if args["spatial_disaggregation"]:
            etrago.disaggregated_network.export_to_csv_folder(
                path + "/disaggregated_network"
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
