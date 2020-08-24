# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
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
lopf.py defines optimization methods for Etrago object.
"""
import os
if 'READTHEDOCS' not in os.environ:
    import time
    import logging
    import pandas as pd
    import numpy as np
    from etrago.tools.utilities import results_to_csv

    logger = logging.getLogger(__name__)


def run_pf_post_lopf(self):
    """ Functions that runs pf_post_lopf accordning to arguments

    Returns
    -------
    None.

    """

    if self.args['pf_post_lopf'] != False:
        # set deafault settings
        pf_settings = {'add_foreign_lopf': True,
                       'q_allocation': 'p_nom',
                       'calc_losses': True}

        # overwirte default values if given in args
        if type(self.args['pf_post_lopf']) == dict:
            for k in self.args['pf_post_lopf'].keys():
                pf_settings[k] = self.args['pf_post_lopf'][k]

        pf_post_lopf(self,
                     pf_settings['add_foreign_lopf'],
                     pf_settings['q_allocation'],
                     pf_settings['calc_losses'])


def pf_post_lopf(etrago, add_foreign_lopf, q_allocation, calc_losses):

    """
    Function that prepares and runs non-linar load flow using PyPSA pf.


    If crossborder lines are DC-links, pf is only applied on german network.
    Crossborder flows are still considerd due to the active behavior of links.
    To return a network containing the whole grid, the optimised solution of the
    foreign components can be added afterwards.

    Parameters
    ----------
    etrago : :class:`etrago.Etrago
        Transmission grid object
    add_foreign_lopf: boolean
        Choose if foreign results of lopf should be added to the network when
        foreign lines are DC
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
        DC-links and saves their optimization results in pandas.DataFrame


        Parameters
        ----------
        network : :class:`pypsa.Network
            Overall container of PyPSA

        Returns
        -------
        None.

        """
        n_bus = pd.Series(index=network.sub_networks.index)

        for i in range(0, len(network.sub_networks.index)-1):
            n_bus[i] = len(network.buses.index[
                network.buses.sub_network.astype(int) == i])

        sub_network_DE = n_bus.index[n_bus == n_bus.max()]

        foreign_bus = network.buses[network.buses.sub_network !=
                                    sub_network_DE.values[0]]

        foreign_comp = {
            'Bus': network.buses[
                network.buses.sub_network != sub_network_DE.values[0]],
            'Generator': network.generators[
                network.generators.bus.isin(foreign_bus.index)],
            'Load': network.loads[
                network.loads.bus.isin(foreign_bus.index)],
            'Transformer': network.transformers[
                network.transformers.bus0.isin(foreign_bus.index)],
            'StorageUnit': network.storage_units[
                network.storage_units.bus.isin(foreign_bus.index)]}

        foreign_series = {
            'Bus': network.buses_t.copy(),
            'Generator': network.generators_t.copy(),
            'Load': network.loads_t.copy(),
            'Transformer':  network.transformers_t.copy(),
            'StorageUnit': network.storage_units_t.copy()}

        for comp in sorted(foreign_series):
            attr = sorted(foreign_series[comp])
            for a in attr:
                if not foreign_series[comp][a].empty:
                    if a != 'p_max_pu':
                        foreign_series[comp][a] = foreign_series[comp][a][
                            foreign_comp[comp].index]

                    else:
                        foreign_series[comp][a] = \
                            foreign_series[comp][a][
                                foreign_comp[comp][
                                    foreign_comp[comp].index.isin(
                                        network.generators_t.p_max_pu.columns)
                                    ].index]

        # Drop compoenents
        network.buses = network.buses.drop(foreign_bus.index)
        network.generators = network.generators[
            network.generators.bus.isin(network.buses.index)]
        network.loads = network.loads[
            network.loads.bus.isin(network.buses.index)]
        network.transformers = network.transformers[
            network.transformers.bus0.isin(network.buses.index)]
        network.storage_units = network.storage_units[
            network.storage_units.bus.isin(network.buses.index)]

        return foreign_bus, foreign_comp, foreign_series

    x = time.time()
    network = etrago.network
    args = etrago.args

    network.lines.s_nom = network.lines.s_nom_opt

    # For the PF, set the P to the optimised P
    network.generators_t.p_set = network.generators_t.p_set.reindex(
        columns=network.generators.index)
    network.generators_t.p_set = network.generators_t.p

    network.storage_units_t.p_set = network.storage_units_t.p_set\
        .reindex(columns=network.storage_units.index)
    network.storage_units_t.p_set = network.storage_units_t.p

    network.links_t.p_set = network.links_t.p_set.reindex(
        columns=network.links.index)
    network.links_t.p_set = network.links_t.p0

    # if foreign lines are DC, execute pf only on sub_network in Germany
    if (args['foreign_lines']['carrier'] == 'DC')\
        or ((args['scn_extension'] != None)
                and ('BE_NO_NEP 2035' in args['scn_extension'])):
        foreign_bus, foreign_comp, foreign_series = \
            drop_foreign_components(network)

    # Set slack bus
    network = set_slack(network)

    # execute non-linear pf
    pf_solution = network.pf(network.snapshots, use_seed=True)

    # if selected, copy lopf results of neighboring countries to network
    if ((args['foreign_lines']['carrier'] == 'DC')
            or ((args['scn_extension'] != None)
                and ('BE_NO_NEP 2035' in args['scn_extension']))
       ) and add_foreign_lopf:
        for comp in sorted(foreign_series):
            network.import_components_from_dataframe(foreign_comp[comp], comp)

            for attr in sorted(foreign_series[comp]):
                network.import_series_from_dataframe(foreign_series
                                                     [comp][attr], comp, attr)

    pf_solve = pd.DataFrame(index=pf_solution['converged'].index)
    pf_solve['converged'] = pf_solution['converged'].values
    pf_solve['error'] = pf_solution['error'].values
    pf_solve['n_iter'] = pf_solution['n_iter'].values

    if not pf_solve[~pf_solve.converged].count().max() == 0:
        logger.warning("PF of %d snapshots not converged.",
                       pf_solve[~pf_solve.converged].count().max())
    if calc_losses:
        calc_line_losses(network)

    network = distribute_q(network, allocation=q_allocation)

    y = time.time()
    z = (y - x) / 60
    print("Time for PF [min]:", round(z, 2))

    if args['csv_export'] != False:
        path = args['csv_export'] + '/pf_post_lopf'
        results_to_csv(network, args, path)
        pf_solve.to_csv(os.path.join(path, 'pf_solution.csv'), index=True)

    return network


def distribute_q(network, allocation='p_nom'):

    """ Function that distributes reactive power at bus to all installed
    generators and storages.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    allocation: str
        Choose key to distribute reactive power:
        'p_nom' to dirstribute via p_nom
        'p' to distribute via p_set

    Returns
    -------


    """
    network.allocation = allocation
    if allocation == 'p':
        p_sum = network.generators_t['p'].\
            groupby(network.generators.bus, axis=1).sum().\
            add(network.storage_units_t['p'].abs().groupby(
                network.storage_units.bus, axis=1).sum(), fill_value=0)
        q_sum = network.generators_t['q'].\
            groupby(network.generators.bus, axis=1).sum()

        q_distributed = network.generators_t.p / \
            p_sum[network.generators.bus.sort_index()].values * \
            q_sum[network.generators.bus.sort_index()].values

        q_storages = network.storage_units_t.p / \
            p_sum[network.storage_units.bus.sort_index()].values *\
            q_sum[network.storage_units.bus.sort_index()].values

    if allocation == 'p_nom':

        q_bus = network.generators_t['q'].\
            groupby(network.generators.bus, axis=1).sum().add(
                network.storage_units_t.q.groupby(
                    network.storage_units.bus, axis=1).sum(), fill_value=0)

        p_nom_dist = network.generators.p_nom_opt.sort_index()
        p_nom_dist[p_nom_dist.index.isin(network.generators.index
                                         [network.generators.carrier ==
                                          'load shedding'])] = 0

        q_distributed = q_bus[
            network.generators.bus].multiply(p_nom_dist.values) /\
            (network.generators.p_nom_opt[
                network.generators.carrier != 'load shedding']
             .groupby(network.generators.bus).sum().add(
                 network.storage_units.p_nom_opt
                 .groupby(network.storage_units.bus).sum(), fill_value=0))[
                     network.generators.bus.sort_index()].values

        q_distributed.columns = network.generators.index

        q_storages = q_bus[network.storage_units.bus]\
            .multiply(network.storage_units.p_nom_opt.values) /\
            ((network.generators.p_nom_opt[
                network.generators.carrier != 'load shedding']
              .groupby(network.generators.bus).sum()
              .add(network.storage_units.p_nom_opt
                   .groupby(network.storage_units.bus).sum(), fill_value=0))[
                         network.storage_units.bus].values)

        q_storages.columns = network.storage_units.index

    q_distributed[q_distributed.isnull()] = 0
    q_distributed[q_distributed.abs() == np.inf] = 0
    q_storages[q_storages.isnull()] = 0
    q_storages[q_storages.abs() == np.inf] = 0
    network.generators_t.q = q_distributed
    network.storage_units_t.q = q_storages

    return network


def calc_line_losses(network):
    """ Calculate losses per line with PF result data

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    s0 : series
        apparent power of line
    i0 : series
        current of line
    -------

    """

    # Line losses
    # calculate apparent power S = sqrt(p² + q²) [in MW]
    s0_lines = ((network.lines_t.p0**2 + network.lines_t.q0**2).
                apply(np.sqrt))
    # calculate current I = S / U [in A]
    i0_lines = np.multiply(s0_lines, 1000000) / \
        np.multiply(network.lines.v_nom, 1000)
    # calculate losses per line and timestep network.\
    # lines_t.line_losses = I² * R [in MW]
    network.lines_t.losses = np.divide(i0_lines**2 * network.lines.r, 1000000)
    # calculate total losses per line [in MW]
    network.lines = network.lines.assign(
        losses=np.sum(network.lines_t.losses).values)

    # Transformer losses
    # https://books.google.de/books?id=0glcCgAAQBAJ&pg=PA151&lpg=PA151&dq=
    # wirkungsgrad+transformator+1000+mva&source=bl&ots=a6TKhNfwrJ&sig=
    # r2HCpHczRRqdgzX_JDdlJo4hj-k&hl=de&sa=X&ved=
    # 0ahUKEwib5JTFs6fWAhVJY1AKHa1cAeAQ6AEIXjAI#v=onepage&q=
    # wirkungsgrad%20transformator%201000%20mva&f=false
    # Crastan, Elektrische Energieversorgung, p.151
    # trafo 1000 MVA: 99.8 %
    network.transformers = network.transformers.assign(
        losses=np.multiply(network.transformers.s_nom, (1 - 0.998)).values)

    # calculate total losses (possibly enhance with adding these values
    # to network container)
    losses_total = sum(network.lines.losses) + sum(network.transformers.losses)
    print("Total lines losses for all snapshots [MW]:", round(losses_total, 2))
    losses_costs = losses_total * np.average(network.buses_t.marginal_price)
    print("Total costs for these losses [EUR]:", round(losses_costs, 2))

    return


def set_slack(network):

    """
    Function that chosses the bus with the maximum installed power as slack

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    network : :class:`pypsa.Network
        Overall container of PyPSA



    """

    old_slack = network.generators.index[network.
                                         generators.control == 'Slack'][0]
    # check if old slack was PV or PQ control:
    if network.generators.p_nom[old_slack] > 50 and network.generators.\
            carrier[old_slack] in ('solar', 'wind'):
        old_control = 'PQ'
    elif network.generators.p_nom[old_slack] > 50 and network.generators.\
            carrier[old_slack] not in ('solar', 'wind'):
        old_control = 'PV'
    elif network.generators.p_nom[old_slack] < 50:
        old_control = 'PQ'

    old_gens = network.generators
    gens_summed = network.generators_t.p.sum()
    old_gens['p_summed'] = gens_summed
    max_gen_buses_index = old_gens.groupby(['bus']).agg(
        {'p_summed': np.sum}).p_summed.sort_values().index

    for bus_iter in range(1, len(max_gen_buses_index) - 1):
        if old_gens[(network.
                     generators['bus'] == max_gen_buses_index[-bus_iter]) &
                    (network.generators['control'] != 'PQ')].empty:
            continue
        else:
            new_slack_bus = max_gen_buses_index[-bus_iter]
            break

    network.generators = network.generators.drop('p_summed', 1)
    new_slack_gen = network.generators.\
        p_nom[(network.generators['bus'] == new_slack_bus) & (
            network.generators['control'] == 'PV')].sort_values().index[-1]

    network.generators.at[old_slack, 'control'] = old_control
    network.generators.at[new_slack_gen, 'control'] = 'Slack'


    return network
