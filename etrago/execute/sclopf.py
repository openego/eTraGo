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
sclopf.py defines functions for contingency analysis. 
"""
import os

import numpy as np
import pandas as pd
import time
import datetime
import logging

logger = logging.getLogger(__name__)

from pypsa.opt import l_constraint
from pypsa.opf import (
    define_passive_branch_flows_with_kirchhoff,
    network_lopf_solve,
    define_passive_branch_flows,
    network_lopf_build_model,
    network_lopf_prepare_solver,
)
from pypsa.pf import sub_network_lpf
import multiprocessing as mp
import csv

from etrago.execute import (
    import_gen_from_links,
    update_electrical_parameters,
)

from etrago.tools.constraints import Constraints


logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ClaraBuettner"


def post_contingency_analysis_lopf(etrago, branch_outages, n_process=4):
    network = etrago.network.copy()

    import_gen_from_links(network, drop_small_capacities=False)

    # Drop not-AC subnetwork from network
    n = network.copy()
    main_subnet = str(network.buses.sub_network.value_counts().argmax())
    n.mremove(
        "Bus",
        n.buses[
            ~n.buses.index.isin(
                network.sub_networks["obj"][main_subnet].buses().index
            )
        ].index,
    )

    for one_port in n.iterate_components(
        ["Load", "Generator", "Store", "StorageUnit"]
    ):
        n.mremove(
            one_port.name,
            one_port.df[~one_port.df.bus.isin(n.buses.index)].index,
        )

    for two_port in n.iterate_components(["Line", "Link", "Transformer"]):
        n.mremove(
            two_port.name,
            two_port.df[~two_port.df.bus0.isin(n.buses.index)].index,
        )

        n.mremove(
            two_port.name,
            two_port.df[~two_port.df.bus1.isin(n.buses.index)].index,
        )

    n.lines.s_nom = n.lines.s_nom_opt.copy()

    b_x = 1.0 / n.lines.x_pu

    # Consider outage of each line
    branch_outages = n.lines.index

    # Copy result from LOPF
    n.generators_t.p_set = n.generators_t.p_set.reindex(
        columns=n.generators.index
    )
    n.generators_t.p_set = n.generators_t.p
    n.storage_units_t.p_set = n.storage_units_t.p_set.reindex(
        columns=n.storage_units.index
    )
    n.storage_units_t.p_set = n.storage_units_t.p
    n.links_t.p_set = n.links_t.p_set.reindex(columns=n.links.index)
    n.links_t.p_set = n.links_t.p0

    snapshots_set = {}
    length = int(n.snapshots.size / n_process)
    for i in range(n_process):
        snapshots_set[str(i + 1)] = n.snapshots[i * length : (i + 1) * length]
    manager = mp.Manager()
    d = manager.dict()
    snapshots_set[str(n_process)] = n.snapshots[i * length :]

    def multi_con(n, snapshots, d):
        for sn in snapshots:
            # Check no lines are overloaded with the linear contingency analysis
            # rows: branch outage, index = monitorred line
            p0_test = n.lpf_contingency(
                branch_outages=branch_outages, snapshots=sn
            )

            # check loading as per unit of s_nom in each contingency
            # columns: branch_outages
            load = abs(
                p0_test.divide(n.passive_branches().s_nom_opt, axis=0)
            ).drop(["base"], axis=1)
            # schon im base_case leitungen teilweise überlastet. Liegt das an x Anpassung?
            overload = load - 1  # columns: branch_outages
            overload[overload < 0] = 0

            if not len(overload) == 0:
                d[sn] = overload

    processes = [
        mp.Process(target=multi_con, args=(n, snapshots_set[i], d))
        for i in snapshots_set
    ]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

    if etrago.args["csv_export"] != False:
        if not os.path.exists(
            etrago.args["csv_export"] + "/post_contingency_analysis/"
        ):
            os.mkdir(etrago.args["csv_export"] + "/post_contingency_analysis/")

        for s in d.keys():
            d[s].to_csv(
                etrago.args["csv_export"]
                + "/post_contingency_analysis/"
                + str(s)
                + ".csv"
            )

    return d


def post_contingency_analysis(network, branch_outages, delta=0.05):
    import_gen_from_links(network, drop_small_capacities=False)

    # Drop not-AC subnetwork from network
    n = network.copy()
    main_subnet = str(network.buses.sub_network.value_counts().argmax())

    n.mremove(
        "Bus",
        n.buses[
            ~n.buses.index.isin(
                network.sub_networks["obj"][main_subnet].buses().index
            )
        ].index,
    )

    for one_port in n.iterate_components(
        ["Load", "Generator", "Store", "StorageUnit"]
    ):
        n.mremove(
            one_port.name,
            one_port.df[~one_port.df.bus.isin(n.buses.index)].index,
        )

    for two_port in n.iterate_components(["Line", "Link", "Transformer"]):
        n.mremove(
            two_port.name,
            two_port.df[~two_port.df.bus0.isin(n.buses.index)].index,
        )

        n.mremove(
            two_port.name,
            two_port.df[~two_port.df.bus1.isin(n.buses.index)].index,
        )

    n.lines.s_nom = n.lines.s_nom_opt.copy()

    b_x = 1.0 / n.lines.x_pu

    if np.isnan(b_x).any():
        import pdb

        pdb.set_trace()

    branch_outages = (
        n.lines.index
    )  # branch_outages[branch_outages.isin(n.lines.index)]
    n.generators_t.p_set = n.generators_t.p_set.reindex(
        columns=n.generators.index
    )
    n.generators_t.p_set = n.generators_t.p
    n.storage_units_t.p_set = n.storage_units_t.p_set.reindex(
        columns=n.storage_units.index
    )
    n.storage_units_t.p_set = n.storage_units_t.p
    n.links_t.p_set = n.links_t.p_set.reindex(columns=n.links.index)
    n.links_t.p_set = n.links_t.p0

    d = {}
    for sn in network.snapshots:
        p0_test = n.lpf_contingency(
            branch_outages=branch_outages, snapshots=sn
        )
        # rows: branch outage, index = monitorred line
        # check loading as per unit of s_nom in each contingency
        load_signed = (
            p0_test.divide(n.passive_branches().s_nom_opt, axis=0)
        ).drop(["base"], axis=1)
        load = abs(
            p0_test.divide(n.passive_branches().s_nom_opt, axis=0)
        ).drop(
            ["base"], axis=1
        )  # columns: branch_outages

        load_per_outage_over = load_signed.transpose()[
            load_signed.abs().max() > (1 + delta)
        ].transpose()

        out = load_per_outage_over.columns.values
        mon = load_per_outage_over.abs().idxmax().values

        sign = []
        for i in range(len(out)):
            sign.append(
                np.sign(load_per_outage_over[out[i]][mon[i]]).astype(int)
            )
        combinations = [out, mon, sign]

        # else:
        #     overloaded = (load>(1 + delta))# columns: branch_outages

        #     array_mon = []
        #     array_out = []
        #     array_sign = []

        #     for col in overloaded:
        #         mon = overloaded.index[overloaded[col]].tolist()
        #         out = [col]*len(mon)

        #         sign = np.sign(load_signed[overloaded[col]][col]).values

        #         if mon != []:
        #             array_mon.extend(mon)
        #             array_out.extend(out)
        #             array_sign.extend(sign)
        #     combinations = [array_out, array_mon, array_sign]
        if not len(combinations[0]) == 0:
            d[sn] = combinations

    return d


def network_lpf_contingency_subnetwork(
    network, snapshot=None, branch_outages=None
):
    """
    Computes linear power flow for a selection of branch outages within the
    subnetwork with the highest number of buses

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to network.snapshots
        NB: currently this only works for a single snapshot
    branch_outages : list-like
        A list of passive branches which are to be tested for outages.
        If None, it's take as all network.passive_branches_i()

    Returns
    -------
    p0 : pandas.DataFrame
        num_passive_branch x num_branch_outages DataFrame of new power flows

    """

    main_subnet = str(network.buses.sub_network.value_counts().argmax())
    sn = network.sub_networks.obj[main_subnet]
    sub_network_lpf(sn, snapshot)

    # Store the flows from the base case
    passive_branches = sn.branches()

    if branch_outages is None:
        branch_outages = passive_branches.index

    p0_base = pd.concat(
        {
            c: network.pnl(c).p0.loc[snapshot]
            for c in network.passive_branch_components
        }
    )
    p0 = p0_base.to_frame("base")

    sn._branches = sn.branches()
    sn.calculate_BODF()

    if not isinstance(branch_outages[0], tuple):
        logger.warning(
            f"No type given for {branch_outages}, assuming it is a line"
        )

    for branch in branch_outages:
        if not isinstance(branch, tuple):
            branch = ("Line", branch)
        sn = network.sub_networks.obj[passive_branches.sub_network[branch]]

        branch_i = sn._branches.index.get_loc(branch)

        p0_new = p0_base + pd.Series(
            sn.BODF[:, branch_i] * p0_base[branch], sn._branches.index
        )

        p0[branch] = p0_new

    return p0


def post_contingency_analysis_per_line(
    network, branch_outages, n_process=4, delta=0.01
):
    x = time.time()
    # import_gen_from_links(network, drop_small_capacities=False)

    # # Drop not-AC subnetwork from network
    n = network.copy()
    # main_subnet = str(network.buses.sub_network.value_counts().argmax())

    # Import other sectors into main subnetwork
    from_AC = n.buses.carrier[n.links.bus0] == "AC"
    to_AC = n.buses.carrier[n.links.bus1] == "AC"

    links_from_ac = n.links[(from_AC.values) & (n.links.carrier != "DC")]
    links_to_ac = n.links[(to_AC.values) & (n.links.carrier != "DC")]

    links_from_ac.drop(
        links_from_ac[links_from_ac.index.isin(links_to_ac.index)].index,
        inplace=True,
    )
    ts_from = n.links_t.p0[links_from_ac.index]
    ts_from.columns = "link_from_" + ts_from.columns

    ts_to = n.links_t.p1[links_to_ac.index]
    ts_to.columns = "link_to_" + ts_to.columns

    n.madd(
        "Load",
        "link_from_" + links_from_ac.index.values,
        bus=links_from_ac.bus0.values,
        p_set=ts_from,
    )

    n.madd(
        "Load",
        "link_to_" + links_to_ac.index.values,
        bus=links_to_ac.bus0.values,
        p_set=ts_to,
    )

    buses_to_drop = (
        links_from_ac.bus1.values.tolist()
        + links_to_ac.bus0.values.tolist()
        + n.buses[
            n.buses.carrier.str.contains("heat_store")
        ].index.values.tolist()
    )

    for one_port in n.iterate_components(
        ["Load", "Generator", "Store", "StorageUnit"]
    ):
        n.mremove(
            one_port.name,
            one_port.df[one_port.df.bus.isin(buses_to_drop)].index,
        )

    for two_port in n.iterate_components(["Link", "Transformer"]):
        n.mremove(
            two_port.name,
            two_port.df[two_port.df.bus0.isin(buses_to_drop)].index,
        )

        n.mremove(
            two_port.name,
            two_port.df[two_port.df.bus1.isin(buses_to_drop)].index,
        )

    n.mremove("Bus", buses_to_drop)

    n.lines.s_nom = n.lines.s_nom_opt.copy()

    b_x = 1.0 / n.lines.x_pu

    if np.isnan(b_x).any():
        import pdb

        pdb.set_trace()

    n.generators_t.p_set = n.generators_t.p_set.reindex(
        columns=n.generators.index
    )
    n.generators_t.p_set = n.generators_t.p
    n.storage_units_t.p_set = n.storage_units_t.p_set.reindex(
        columns=n.storage_units.index
    )
    n.storage_units_t.p_set = n.storage_units_t.p
    n.links_t.p_set = n.links_t.p_set.reindex(columns=n.links.index)
    n.links_t.p_set = n.links_t.p0

    snapshots_set = {}
    length = int(n.snapshots.size / n_process)

    for i in range(n_process):
        snapshots_set[str(i + 1)] = n.snapshots[i * length : (i + 1) * length]
    snapshots_set[str(n_process)] = n.snapshots[i * length :]

    manager = mp.Manager()
    d = manager.dict()

    def multi_con(n, snapshots, d):
        for sn in snapshots:
            # Check no lines are overloaded with the linear contingency analysis
            p0_test = network_lpf_contingency_subnetwork(
                network=n, branch_outages=branch_outages, snapshot=sn
            )
            # rows: branch outage, index = monitorred line
            # check loading as per unit of s_nom in each contingency
            load_signed = (
                p0_test.divide(n.passive_branches().s_nom_opt, axis=0)
            ).drop(["base"], axis=1)
            load = abs(
                p0_test.divide(n.passive_branches().s_nom_opt, axis=0)
            ).drop(
                ["base"], axis=1
            )  # columns: branch_outages
            if True:
                load_per_outage_over = load_signed.transpose()[
                    load_signed.abs().max() > (1 + delta)
                ].transpose()

                out = load_per_outage_over.columns.values
                mon = load_per_outage_over.abs().idxmax().values

                sign = []
                for i in range(len(out)):
                    sign.append(
                        np.sign(load_per_outage_over[out[i]][mon[i]]).astype(
                            int
                        )
                    )
                combinations = [out, mon, sign]

            else:
                overloaded = load > (1 + delta)  # columns: branch_outages

                array_mon = []
                array_out = []
                array_sign = []

                for col in overloaded:
                    mon = overloaded.index[overloaded[col]].tolist()
                    out = [col] * len(mon)

                    sign = np.sign(load_signed[overloaded[col]][col]).values

                    if mon != []:
                        array_mon.extend(mon)
                        array_out.extend(out)
                        array_sign.extend(sign)
                combinations = [array_out, array_mon, array_sign]
            if not len(combinations[0]) == 0:
                d[sn] = combinations

    processes = [
        mp.Process(target=multi_con, args=(n, snapshots_set[i], d))
        for i in snapshots_set
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

    y = (time.time() - x) / 60
    logger.info(
        "Post contingengy check finished in " + str(round(y, 2)) + " minutes."
    )

    return d


def iterate_lopf_calc(network, args, l_snom_pre, t_snom_pre):
    """
    Function that runs iterations of lopf without building new models.
    Currently only working with model_formulation = 'kirchhoff'

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    l_snom_pre: pandas.Series
        s_nom of ac-lines in previous iteration
    t_snom_pre: pandas.Series
        s_nom of transformers in previous iteration
    """
    # Delete flow constraints for each possible model formulation
    x = time.time()
    network.model.del_component("cycle_constraints")
    network.model.del_component("cycle_constraints_index")
    network.model.del_component("cycle_constraints_index_0")
    network.model.del_component("cycle_constraints_index_1")

    if args["model_formulation"] == "kirchhoff":
        define_passive_branch_flows_with_kirchhoff(
            network, network.snapshots, skip_vars=True
        )
    else:
        logger.error("Currently only implemented for kirchhoff-formulation.")
    y = time.time()
    logger.info("Flow constraints updated in [min] " + str((y - x) / 60))
    network_lopf_solve(
        network,
        network.snapshots,
        formulation=args["model_formulation"],
        free_memory={"pypsa"},
        solver_options=args["solver_options"],
    )

    return network


def add_all_contingency_constraints(network, combinations, track_time):
    x = time.time()

    branch_outage_keys = []
    contingency_flow = {}
    n_buses = 0

    # choose biggest sub_network
    if len(network.sub_networks.obj.index) > 1:
        for s in network.sub_networks.obj.index:
            n = len(network.sub_networks.obj[s].buses())

            if n > n_buses:
                n_buses = n
                sub = network.sub_networks.obj[s]
    else:
        sub = network.sub_networks.obj[0]

    sub._branches = sub.branches()
    sub.calculate_BODF()
    bodf = pd.DataFrame(
        columns=sub.branches().index.get_level_values(1),
        index=sub.branches().index.get_level_values(1),
        data=sub.BODF,
    )

    sub._branches["_i"] = range(sub._branches.shape[0])
    sub._extendable_branches = sub._branches[sub._branches.s_nom_extendable]
    sub._fixed_branches = sub._branches[~sub._branches.s_nom_extendable]

    # set dict with s_nom containing fixed or extendable s_nom
    s_nom = sub._branches.s_nom.to_dict()

    for idx in sub._extendable_branches.index.values:
        s_nom[idx] = network.model.passive_branch_s_nom[idx]

    BODF_FACTOR = 1  # avoid numerical trouble caused by small BODFs

    for sn in combinations.keys():
        if len(combinations[sn][0]) > 0:
            out = combinations[sn][0]
            mon = combinations[sn][1]
            sign = combinations[sn][2]

            branch_outage_keys.extend(
                [
                    (out[i][0], out[i][1], mon[i][0], mon[i][1], sn)
                    for i in range(len(out))
                ]
            )

            # avoid duplicate values in branch_outage_keys
            branch_outage_keys = list(set(branch_outage_keys))

            # set contingengy flow constraint
            if sub._fixed_branches.empty:
                contingency_flow.update(
                    {
                        (out[i][0], out[i][1], mon[i][0], mon[i][1], sn): [
                            [
                                (
                                    BODF_FACTOR * sign[i],
                                    network.model.passive_branch_p[mon[i], sn],
                                ),
                                (
                                    BODF_FACTOR
                                    * sign[i]
                                    * bodf[out[i][1]][mon[i][1]],
                                    network.model.passive_branch_p[out[i], sn],
                                ),
                                (-1 * BODF_FACTOR * network.lines_t.s_max_pu.loc[sn, mon[i][1]], s_nom[mon[i]]),
                            ],
                            "<=",
                            0,
                        ]
                        for i in range(len(mon))
                    }
                )

            elif sub._extendable_branches.empty:
                contingency_flow.update(
                    {
                        (out[i][0], out[i][1], mon[i][0], mon[i][1], sn): [
                            [
                                (
                                    sign[i],
                                    network.model.passive_branch_p[mon[i], sn],
                                ),
                                (
                                    sign[i] * bodf[out[i][1]][mon[i][1]],
                                    network.model.passive_branch_p[out[i], sn],
                                ),
                            ],
                            "<=",
                            s_nom[mon[i]],
                        ]
                        for i in range(len(mon))
                    }
                )

            else:
                print("Not implemented!")

    z = time.time()
    track_time[datetime.datetime.now()] = "Contingency constraints calculated"
    logger.info(
        "Security constraints calculated in [min] " + str((z - x) / 60)
    )

    # remove constraints from previous iteration
    network.model.del_component("contingency_flow")
    network.model.del_component("contingency_flow_index")

    # Delete rows with small BODFs to avoid nummerical problems
    for c in list(contingency_flow.keys()):
        if (
            (abs(contingency_flow[c][0][1][0]) < 1e-8 * BODF_FACTOR)
            & (abs(contingency_flow[c][0][1][0]) != 0)
        ) | (abs(contingency_flow[c][0][1][0]) > 1.1 * BODF_FACTOR):
            contingency_flow.pop(c)

    # set constraints for new iteration
    l_constraint(
        network.model,
        "contingency_flow",
        contingency_flow,
        # branch_outage_keys)
        contingency_flow.keys(),
    )

    y = time.time()
    logger.info("Security constraints updated in [min] " + str((y - x) / 60))

    return len(contingency_flow)


def split_extended_lines(network, percent):
    split_extended_lines.counter += 1
    expansion_rel = network.lines.s_nom_opt / network.lines.s_nom
    num_lines = expansion_rel[expansion_rel > percent]

    expanded_lines = network.lines[network.lines.index.isin(num_lines.index)]
    new_lines = pd.DataFrame(columns=network.lines.columns)

    for line in num_lines.index:
        data = expanded_lines[expanded_lines.index == line]
        n = 0
        while num_lines[line] > 0:
            if num_lines[line] < 1:
                factor = num_lines[line]
            else:
                factor = 1
            data_new = data.copy()

            for col in ["x", "r", "x_pu"]:
                data_new[col] = (
                    data_new[col] / factor * data.s_nom_opt / data.s_nom
                )
            data_new.s_nom_opt = data_new.s_nom * factor
            if n == 0:
                new_lines = pd.concat([new_lines, data_new])
            else:
                data_new.s_nom_min = 0
                new_lines = pd.concat(
                    [
                        new_lines,
                        data_new.rename(
                            index={
                                line: str(line)
                                + "_"
                                + str(np.ceil(num_lines[line]).astype(int))
                                + "iter_"
                                + str(split_extended_lines.counter)
                            }
                        ),
                    ],
                )

            num_lines[line] = num_lines[line] - factor
            n = n + 1

    network.lines = network.lines.drop(
        new_lines.index, axis="index", errors="ignore"
    )
    # new_lines.index += network.lines.index.astype(int).max() + 1
    network.import_components_from_dataframe(new_lines, "Line")

    l_snom_pre = network.lines.s_nom_opt.copy()
    l_snom_pre[l_snom_pre == 0] = network.lines.s_nom[l_snom_pre == 0]
    t_snom_pre = network.transformers.s_nom_opt.copy()

    return l_snom_pre, t_snom_pre


def calc_new_sc_combinations(combinations, new):
    for sn in new.keys():
        com = [[], [], []]
        com[0] = combinations[sn][0] + list(new[sn][0])
        com[1] = combinations[sn][1] + list(new[sn][1])
        com[2] = combinations[sn][2] + new[sn][2]

        df = pd.DataFrame([com[0], com[1], com[2]])
        data = df.transpose().drop_duplicates().transpose()
        combinations[sn] = data.values.tolist()
    return combinations


def split_parallel_lines(network):
    print("Splitting parallel lines...")

    parallel_lines = network.lines[network.lines.num_parallel > 1]
    s_max_pu = network.lines_t.s_max_pu[
        parallel_lines[
            parallel_lines.index.isin(network.lines_t.s_max_pu.columns)].index]

    new_lines = pd.DataFrame(columns=network.lines.columns)
    new_lines_t = pd.DataFrame(index=network.snapshots)
    for i in parallel_lines.index:
        data_new = parallel_lines[parallel_lines.index == i]
        for col in ["b", "g", "s_nom", "s_nom_min", "s_nom_max", "s_nom_opt"]:
            data_new[col] = data_new[col] / data_new.num_parallel
        for col in ["x", "r"]:
            data_new[col] = data_new[col] * data_new.num_parallel
        data_new.cables = 3
        data_new.num_parallel = 1
        num = parallel_lines.num_parallel[i]
        for n in range(int(num)):
            data_new.index = [str(i) + "_" + str(int(n + 1))]
            new_lines = pd.concat(
                [
                    new_lines,
                    data_new,
                ],
            )
            if i in s_max_pu.columns:
                new_lines_t.loc[:, data_new.index] =  s_max_pu[i]

    network.mremove("Line", parallel_lines.index)

    network.import_components_from_dataframe(new_lines, "Line")

    if not new_lines_t.empty:
        network.import_series_from_dataframe(new_lines_t, "Line", "s_max_pu")

    for i in network.lines.index[
            ~network.lines.index.isin(network.lines_t.s_max_pu.columns)]:
        network.lines_t.s_max_pu[i] = network.lines.s_max_pu[i]
    return network


def plot_sc_lines(out, mon, network):
    line_w = pd.Series(index=network.lines.index, data=1)

    line_w[out] = 5
    line_w[mon] = 5

    line_colors = pd.Series(index=network.lines.index, data="grey")

    line_colors[out] = "red"
    line_colors[mon] = "blue"
    network.plot(
        link_widths=0,
        bus_sizes=0.001,
        line_widths=line_w,
        line_colors=line_colors,
    )


def iterate_sclopf(
    etrago,
    n_process=4,
    delta=0.01,
    n_overload=0,
    post_lopf=False,
    div_ext_lines=False,
):

    if etrago.args["method"]["formulation"] != "pyomo":
        etrago.args["method"]["formulation"] = "pyomo"
        logger.info("""
                    SCLOPF currently only implemented for pyomo.
                    Setting etrago.args["method"]["formulation"] = 'pyomo'
                    """)
    network = etrago.network

    network = split_parallel_lines(network)
    network.lines.s_max_pu = pd.Series(index=network.lines.index, data=1.0)

    args = etrago.args

    track_time = pd.Series()
    l_snom_pre = network.lines.s_nom.copy()
    t_snom_pre = network.transformers.s_nom.copy()
    add_all_contingency_constraints.counter = 0
    n = 0
    track_time[datetime.datetime.now()] = "Iterative SCLOPF started"
    x = time.time()
    #    results_to_csv.counter=0
    split_extended_lines.counter = 0

    # 1. LOPF without SC
    solver_options_lopf = args["solver_options"]
    solver_options_lopf["FeasibilityTol"] = 1e-5
    solver_options_lopf["BarConvTol"] = 1e-6

    # If LOPF was performed beforehand, this can be used as the starting
    # point for the SCLOPF
    if post_lopf:
        if div_ext_lines:
            l_snom_pre, t_snom_pre = split_extended_lines(network, percent=1.5)
            # branch_outages=network.lines[network.lines.country == "DE"].index

            network_lopf_build_model(network, formulation="kirchhoff")

            network_lopf_prepare_solver(network, solver_name="gurobi")

        network.lines.s_nom = network.lines.s_nom_opt.copy()
        network.links.p_nom = network.links.p_nom_opt.copy()
        network.lines.s_nom_extendable = False
        network.links.p_nom_extendable = False
        network.storage_units.p_nom = network.storage_units.p_nom_opt.copy()
        network.storage_units.p_nom_extendable = False
        if div_ext_lines:
            network_lopf_build_model(network, formulation="kirchhoff")

            network_lopf_prepare_solver(network, solver_name="gurobi")
        path_name = "/post_sclopf_iteration_"

    # Run first iteration without any contingency constraint
    else:
        network.lopf(
            network.snapshots,
            solver_name=args["solver"],
            solver_options=solver_options_lopf,
            extra_functionality=Constraints(etrago.args, False).functionality,
            formulation=args["model_formulation"],
            pyomo=True,
        )
        path_name = "/sclopf_iteration_"
        track_time[datetime.datetime.now()] = "Solve SCLOPF"

    # Update electrical parameters if network is extendable
    if network.lines.s_nom_extendable.any():
        l_snom_pre, t_snom_pre = update_electrical_parameters(
            network, l_snom_pre, t_snom_pre
        )
        track_time[datetime.datetime.now()] = "Adjust impedances"
        if not post_lopf:
            if div_ext_lines:
                l_snom_pre, t_snom_pre = split_extended_lines(
                    network, percent=1.5
                )

            #  branch_outages=network.lines[network.lines.country == "DE"].index

            network_lopf_build_model(network, formulation="kirchhoff")

            # Add extra_functionalities depending on args
            Constraints(etrago.args, False).functionality(
                etrago.network, etrago.network.snapshots
            )

            network_lopf_prepare_solver(network, solver_name="gurobi")
    # Calculate security constraints
    nb = 0
    main_subnet = str(network.buses.sub_network.value_counts().argmax())
    branch_outages = network.lines[
        network.lines.sub_network == main_subnet
    ].index

    new = post_contingency_analysis_per_line(
        network, branch_outages, n_process=n_process, delta=delta
    )

    track_time[datetime.datetime.now()] = "Overall post contingency analysis"

    # Initalzie dict of security constraints
    combinations = dict.fromkeys(network.snapshots, [[], [], []])
    size = 0
    for i in range(len(new.keys())):
        size = (
            size
            + len(new.values()[i][0])
            * network.snapshot_weightings.generators[new.keys()[i]]
        )

    while size > n_overload:
        if n < 100:
            print(len(branch_outages))
            # if not post_lopf:

            #     if div_ext_lines:
            #         l_snom_pre, t_snom_pre = split_extended_lines(
            #             network, percent=1.5
            #         )
            #         branch_outages=network.lines[network.lines.country == "DE"].index

            #     network_lopf_build_model(network, formulation="kirchhoff")

            #     network_lopf_prepare_solver(network, solver_name="gurobi")
            logger.info(str(size) + " overloadings")

            combinations = calc_new_sc_combinations(combinations, new)

            nb = (
                int(
                    add_all_contingency_constraints(
                        network, combinations, track_time
                    )
                )
                / 2
            )

            track_time[datetime.datetime.now()] = (
                "Update Contingency constraints"
            )

            logger.info(
                "SCLOPF No. "
                + str(n + 1)
                + " started with "
                + str(2 * nb)
                + " SC-constraints."
            )

            iterate_lopf_calc(network, args, l_snom_pre, t_snom_pre)
            track_time[datetime.datetime.now()] = "Solve SCLOPF"

            # nur mit dieser Reihenfolge (x anpassen, dann lpf_check) kann Netzausbau n-1 sicher werden
            if network.lines.s_nom_extendable.any():
                l_snom_pre, t_snom_pre = update_electrical_parameters(
                    network, l_snom_pre, t_snom_pre
                )
                track_time[datetime.datetime.now()] = "Adjust impedances"
                if not post_lopf:
                    if div_ext_lines:
                        l_snom_pre, t_snom_pre = split_extended_lines(
                            network, percent=1.5
                        )
                    # branch_outages=network.lines[network.lines.country == "DE"].index

                    network_lopf_build_model(network, formulation="kirchhoff")

                    network_lopf_prepare_solver(network, solver_name="gurobi")
                track_time[datetime.datetime.now()] = "Adjust impedances"

            new = post_contingency_analysis_per_line(
                network, branch_outages, n_process, delta
            )

            size = 0
            for i in range(len(new.keys())):
                size = (
                    size
                    + len(new.values()[i][0])
                    * network.snapshot_weightings.generators[new.keys()[i]]
                )

            track_time[datetime.datetime.now()] = (
                "Overall post contingency analysis"
            )
            n += 1

        else:
            print("Maximum number of iterations reached.")
            break

    if args["csv_export"] != False:
        etrago.export_to_csv(args["csv_export"] + "/grid_optimization")

    y = (time.time() - x) / 60

    logger.info(
        "SCLOPF with "
        + str(2 * nb)
        + " constraints solved in "
        + str(n)
        + " iterations in "
        + str(round(y, 2))
        + " minutes."
    )
