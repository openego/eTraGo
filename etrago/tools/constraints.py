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
Constraints.py includes additional constraints for eTraGo-optimizations
"""
import logging
from pyomo.environ import Constraint
import pandas as pd
import pyomo.environ as po
import numpy as np
from pypsa.linopt import get_var, linexpr, define_constraints, define_variables
from pypsa.descriptors import expand_series
from pypsa.pf import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"


def _get_crossborder_components(network, cntr='all'):
    """
    Identifies foreign buses and crossborder ac- and dc-lines for all foreign
    countries or only one specific.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    cntr : str, optional
        Country code of the returned buses and lines. The default is 'all'.

    Returns
    -------
    buses_de : pandas.base.Index
        Index of buses located in Germany
    buses_for : pandas.base.Index
        Index of buses located in selected country
    cb0 : pandas.base.Index
        Index of ac-lines from foreign country to Germany
    cb1 : pandas.base.Index
        Index of ac-lines from Germany to foreign country
    cb0_link : pandas.base.Index
        Index of dc-lines from foreign country to Germany
    cb1_link : pandas.base.Index
        Index of dc-lines from Germany to foreign country

    """
    buses_de = network.buses.index[network.buses.country_code == 'DE']

    if cntr == 'all':
        buses_for = network.buses.index[network.buses.country_code != 'DE']
    else:
        buses_for = network.buses.index[network.buses.country_code == cntr]

    cb0 = network.lines.index[(network.lines.bus0.isin(buses_for))
                              & (network.lines.bus1.isin(buses_de))]

    cb1 = network.lines.index[(network.lines.bus1.isin(buses_for))
                              & (network.lines.bus0.isin(buses_de))]

    cb0_link = network.links.index[(network.links.bus0.isin(buses_for))
                                   & (network.links.bus1.isin(buses_de))]

    cb1_link = network.links.index[(network.links.bus0.isin(buses_de))
                                   & (network.links.bus1.isin(buses_for))]

    return buses_de, buses_for, cb0, cb1, cb0_link, cb1_link

def _max_line_ext(self, network, snapshots):
    """
    Extra-functionality that limits overall line expansion to a multiple
    of existing capacity.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None

    """
    lines_snom = network.lines.s_nom.sum()
    links_pnom = network.links.p_nom.sum()

    def _rule(m):
        lines_opt = sum(m.passive_branch_s_nom[index]
                        for index in m.passive_branch_s_nom_index)

        links_opt = sum(m.link_p_nom[index]
                        for index in m.link_p_nom_index)

        return (lines_opt + links_opt) <= (lines_snom + links_pnom)\
                           * self.args['extra_functionality']['max_line_ext']

    network.model.max_line_ext = Constraint(rule=_rule)

def _max_line_ext_nmp(self, network, snapshots):
    """
    Extra-functionality that limits overall line expansion to a multiple
    of existing capacity without pyomo.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    lines_snom = network.lines.s_nom.sum()
    links_pnom = network.links.p_nom.sum()
    get_var(network, 'Line', 's_nom')

    def _rule(m):
        lines_opt = sum(m.passive_branch_s_nom[index]
                        for index in m.passive_branch_s_nom_index)

        links_opt = sum(m.link_p_nom[index]
                        for index in m.link_p_nom_index)

        return (lines_opt + links_opt) <= (lines_snom + links_pnom)\
                           * self.args['extra_functionality']['max_line_ext']

    network.model.max_line_ext = Constraint(rule=_rule)

def _min_renewable_share_nmp(self, network, snapshots):
    """
    Extra-functionality that limits the minimum share of renewable generation.
    Add key 'min_renewable_share' and minimal share in p.u. as float
    to args.extra_functionality.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    renewables = ['wind_onshore', 'wind_offshore',
                  'biomass', 'solar', 'run_of_river']

    res = network.generators.index[network.generators.carrier.isin(renewables)]

    renew = get_var(network, 'Generator', 'p').loc[network.snapshots, res].mul(
        network.snapshot_weightings, axis=0)
    total = get_var(network, 'Generator', 'p').mul(
        network.snapshot_weightings, axis=0)

    renew_production = linexpr((1, renew)).sum().sum()
    total_production = linexpr((
        -self.args['extra_functionality']['min_renewable_share'],
        total)).sum().sum()

    expr = renew_production + total_production

    define_constraints(network, expr, '>=', 0, 'Generator', 'min_renew_share')

def _min_renewable_share(self, network, snapshots):
    """
    Extra-functionality that limits the minimum share of renewable generation.
    Add key 'min_renewable_share' and minimal share in p.u. as float
    to args.extra_functionality.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    renewables = ['wind_onshore', 'wind_offshore',
                  'biomass', 'solar', 'run_of_river']

    res = list(network.generators.index[
        network.generators.carrier.isin(renewables)])

    total = list(network.generators.index)

    def _rule(m):

        renewable_production = sum(m.generator_p[gen, sn] * \
                                           network.snapshot_weightings[sn]
                                   for gen in res
                                   for sn in snapshots)
        total_production = sum(m.generator_p[gen, sn] * \
                                       network.snapshot_weightings[sn]
                               for gen in total
                               for sn in snapshots)

        return renewable_production >= total_production *\
                        self.args['extra_functionality']['min_renewable_share']

    network.model.min_renewable_share = Constraint(rule=_rule)

def _cross_border_flow(self, network, snapshots):
    """
    Extra_functionality that limits overall crossborder flows from/to Germany.
    Add key 'cross_border_flow' and array with minimal and maximal percent of
    im- and exports as a fraction of loads in Germany.
    Example: {'cross_border_flow': [-0.1, 0.1]}
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization
    Returns
    -------
    None.

    """

    buses_de, buses_for, cb0, cb1, cb0_link, cb1_link = \
        _get_crossborder_components(network)

    export = pd.Series(
        data=self.args['extra_functionality']['cross_border_flow'])*\
        network.loads_t.p_set.mul(network.snapshot_weightings, axis=0)\
            [network.loads.index[network.loads.bus.isin(buses_de)]].sum().sum()

    def _rule_min(m):
        cb_flow = - sum(m.passive_branch_p['Line', line, sn] *\
                        network.snapshot_weightings[sn]
                        for line in cb0
                        for sn in snapshots) \
                    + sum(m.passive_branch_p['Line', line, sn] *\
                          network.snapshot_weightings[sn]
                          for line in cb1
                          for sn in snapshots)\
                    - sum(m.link_p[link, sn] * \
                          network.snapshot_weightings[sn]
                          for link in cb0_link
                          for sn in snapshots)\
                    + sum(m.link_p[link, sn] * \
                          network.snapshot_weightings[sn]
                          for link in cb1_link
                          for sn in snapshots)
        return cb_flow >= export[0]

    def _rule_max(m):
        cb_flow = - sum(m.passive_branch_p['Line', line, sn] * \
                          network.snapshot_weightings[sn]
                        for line in cb0
                        for sn in snapshots)\
                    + sum(m.passive_branch_p['Line', line, sn] * \
                          network.snapshot_weightings[sn]
                          for line in cb1
                          for sn in snapshots)\
                    - sum(m.link_p[link, sn] * \
                          network.snapshot_weightings[sn]
                          for link in cb0_link
                          for sn in snapshots)\
                    + sum(m.link_p[link, sn] * \
                          network.snapshot_weightings[sn]
                          for link in cb1_link
                          for sn in snapshots)
        return cb_flow <= export[1]

    network.model.cross_border_flows_min = Constraint(rule=_rule_min)
    network.model.cross_border_flows_max = Constraint(rule=_rule_max)

def _cross_border_flow_nmp(self, network, snapshots):
    """
    Extra_functionality that limits overall crossborder flows from/to Germany.
    Add key 'cross_border_flow' and array with minimal and maximal percent of
    im- and exports as a fraction of loads in Germany.
    Example: {'cross_border_flow': [-0.1, 0.1]}
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization
    Returns
    -------
    None.

    """

    buses_de, buses_for, cb0, cb1, cb0_link, cb1_link = \
        _get_crossborder_components(network)

    export = pd.Series(
        data=self.args['extra_functionality']['cross_border_flow'])*\
        network.loads_t.p_set.mul(network.snapshot_weightings, axis=0)\
            [network.loads.index[network.loads.bus.isin(buses_de)]].sum().sum()

    cb0_flow = get_var(network, 'Line', 's').loc[snapshots, cb0].mul(
        network.snapshot_weightings, axis=0)

    cb1_flow = get_var(network, 'Line', 's').loc[snapshots, cb1].mul(
        network.snapshot_weightings, axis=0)

    cb0_link_flow = get_var(network, 'Link', 'p').loc[snapshots, cb0_link].mul(
        network.snapshot_weightings, axis=0)

    cb1_link_flow = get_var(network, 'Link', 'p').loc[snapshots, cb1_link].mul(
        network.snapshot_weightings, axis=0)

    expr = linexpr((-1, cb0_flow)).sum().sum() + \
        linexpr((1, cb1_flow)).sum().sum() +\
        linexpr((-1, cb0_link_flow)).sum().sum() +\
        linexpr((1, cb1_link_flow)).sum().sum()

    define_constraints(network, expr, '>=', export[0], 'Line', 'min_cb_flow')
    define_constraints(network, expr, '<=', export[1], 'Line', 'max_cb_flow')

def _cross_border_flow_per_country_nmp(self, network, snapshots):
    """
    Extra_functionality that limits crossborder flows for each given
    foreign country from/to Germany.
    Add key 'cross_border_flow_per_country' to args.extra_functionality and
    define dictionary of country keys and desired limitations of im/exports as
    a fraction of load in Germany.
    Example: {'cross_border_flow_per_country': {'DK':[-0.05, 0.1], 'FR':[0,0]}}
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization
    Returns
    -------
    None.

    """

    buses_de = network.buses.index[network.buses.country_code == 'DE']

    countries = network.buses.country_code.unique()

    export_per_country = pd.DataFrame(
        data=self.args['extra_functionality']['cross_border_flow_per_country']
        ).transpose()*network.loads_t.p_set.mul(
            network.snapshot_weightings, axis=0)[network.loads.index[
                network.loads.bus.isin(buses_de)]].sum().sum()

    for cntr in export_per_country.index:
        if cntr in countries:
            buses_de, buses_for, cb0, cb1, cb0_link, cb1_link = \
                _get_crossborder_components(network, cntr)

            cb0_flow = get_var(network, 'Line', 's').loc[snapshots, cb0].mul(
                network.snapshot_weightings, axis=0)

            cb1_flow = get_var(network, 'Line', 's').loc[snapshots, cb1].mul(
                network.snapshot_weightings, axis=0)

            cb0_link_flow = get_var(network, 'Link', 'p').loc[
                snapshots, cb0_link].mul(network.snapshot_weightings, axis=0)

            cb1_link_flow = get_var(network, 'Link', 'p').loc[
                snapshots, cb1_link].mul(network.snapshot_weightings, axis=0)

            expr = linexpr((-1, cb0_flow)).sum().sum() + \
                linexpr((1, cb1_flow)).sum().sum() +\
                linexpr((-1, cb0_link_flow)).sum().sum() +\
                linexpr((1, cb1_link_flow)).sum().sum()

            define_constraints(network, expr,
                               '>=', export_per_country[cntr][0],
                               'Line', 'min_cb_flow_' + cntr)
            define_constraints(network, expr,
                               '<=', export_per_country[cntr][1],
                               'Line', 'max_cb_flow_' + cntr)

def _cross_border_flow_per_country(self, network, snapshots):
    """
    Extra_functionality that limits crossborder flows for each given
    foreign country from/to Germany.
    Add key 'cross_border_flow_per_country' to args.extra_functionality and
    define dictionary of country keys and desired limitations of im/exports as
    a fraction of load in Germany.
    Example: {'cross_border_flow_per_country': {'DK':[-0.05, 0.1], 'FR':[0,0]}}
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization
    Returns
    -------
    None.

    """

    buses_de = network.buses.index[network.buses.country_code == 'DE']

    countries = network.buses.country_code.unique()

    export_per_country = pd.DataFrame(
        data=self.args['extra_functionality']['cross_border_flow_per_country']
        ).transpose()*network.loads_t.p_set.mul(
            network.snapshot_weightings, axis=0)[network.loads.index[
                network.loads.bus.isin(buses_de)]].sum().sum()


    for cntr in export_per_country.index:
        if cntr in countries:
            buses_de, buses_for, cb0, cb1, cb0_link, cb1_link = \
                _get_crossborder_components(network, cntr)

            def _rule_min(m):
                cb_flow = - sum(m.passive_branch_p['Line', line, sn] *\
                                        network.snapshot_weightings[sn]
                                for line in cb0
                                for sn in snapshots)\
                                  + sum(m.passive_branch_p['Line', line, sn] *\
                                        network.snapshot_weightings[sn]
                                        for line in cb1
                                        for sn in snapshots)\
                                  - sum(m.link_p[link, sn] * \
                                        network.snapshot_weightings[sn]
                                        for link in cb0_link
                                        for sn in snapshots)\
                                  + sum(m.link_p[link, sn] * \
                                        network.snapshot_weightings[sn]
                                        for link in cb1_link
                                        for sn in snapshots)

                return cb_flow >= export_per_country[0][cntr]

            setattr(network.model, "min_cross_border" + cntr,
                    Constraint(cntr, rule=_rule_min))

            def _rule_max(m):
                cb_flow = - sum(m.passive_branch_p['Line', line, sn] *\
                                        network.snapshot_weightings[sn]
                                for line in cb0
                                for sn in snapshots)\
                                  + sum(m.passive_branch_p['Line', line, sn] *\
                                        network.snapshot_weightings[sn]
                                        for line in cb1
                                        for sn in snapshots)\
                                  - sum(m.link_p[link, sn] * \
                                        network.snapshot_weightings[sn]
                                        for link in cb0_link
                                        for sn in snapshots)\
                                  + sum(m.link_p[link, sn] * \
                                        network.snapshot_weightings[sn]
                                        for link in cb1_link
                                        for sn in snapshots)
                return cb_flow <= export_per_country[1][cntr]

            setattr(network.model, "max_cross_border" + cntr,
                    Constraint(cntr, rule=_rule_max))

def _generation_potential(network, carrier, cntr='all'):
    """
    Function that calculates the generation potential for chosen carriers and
    countries.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    carrier : str
        Energy carrier of generator
    cntr : str, optional
        Country code or 'all'. The default is 'all'.

    Returns
    -------
    gens : pandas.base.Index
        Index of generators with given carrier in chosen country
    potential : float
        Gerneration potential in MW

    """

    if cntr == 'all':
        gens = network.generators.index[network.generators.carrier == carrier]
    else:
        gens = network.generators.index[
            (network.generators.carrier == carrier) &
            (network.generators.bus.astype(str).isin(
                network.buses.index[network.buses.country_code == cntr]))]
    if carrier in ['wind_onshore', 'wind_offshore', 'solar']:
        potential = (network.generators.p_nom[gens]*\
                             network.generators_t.p_max_pu[gens].mul(
                                 network.snapshot_weightings, axis=0)
                    ).sum().sum()
    else:
        potential = network.snapshot_weightings.sum() \
                                * network.generators.p_nom[gens].sum()
    return gens, potential


def _capacity_factor(self, network, snapshots):
    """
    Extra-functionality that limits overall dispatch of generators with chosen
    energy carrier.
    Add key 'capacity_factor' to args.extra_functionality and set limits in
    a dictonary as a fraction of generation potential.
    Example: 'capacity_factor': {'run_of_river': [0, 0.5], 'solar': [0.1, 1]}


    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    arg = self.args['extra_functionality']['capacity_factor']
    carrier = arg.keys()

    for c in carrier:
        factor = arg[c]
        gens, potential = _generation_potential(network, c, cntr='all')

        def _rule_max(m):

            dispatch = sum(m.generator_p[gen, sn] * \
                           network.snapshot_weightings[sn]
                           for gen in gens
                           for sn in snapshots)

            return dispatch <= factor[1] * potential

        setattr(network.model, "max_flh_" + c, Constraint(rule=_rule_max))

        def _rule_min(m):

            dispatch = sum(m.generator_p[gen, sn] * \
                                   network.snapshot_weightings[sn]
                           for gen in gens
                           for sn in snapshots)

            return dispatch >= factor[0] * potential

        setattr(network.model, "min_flh_" + c, Constraint(rule=_rule_min))

def _capacity_factor_nmp(self, network, snapshots):
    """
    Extra-functionality that limits overall dispatch of generators with chosen
    energy carrier.
    Add key 'capacity_factor' to args.extra_functionality and set limits in
    a dictonary as a fraction of generation potential.
    Example: 'capacity_factor': {'run_of_river': [0, 0.5], 'solar': [0.1, 1]}


    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    arg = self.args['extra_functionality']['capacity_factor']
    carrier = arg.keys()

    for c in carrier:
        gens, potential = _generation_potential(network, c, cntr='all')

        generation = get_var(network, 'Generator', 'p').loc[snapshots, gens].\
            mul(network.snapshot_weightings, axis=0)

        define_constraints(network, linexpr((1, generation)).sum().sum(),
                           '>=', arg[c][0] * potential, 'Generator',
                           'min_flh_' + c)
        define_constraints(network, linexpr((1, generation)).sum().sum(),
                           '<=', arg[c][1] * potential, 'Generator',
                           'max_flh_' + c)

def _capacity_factor_per_cntr(self, network, snapshots):
    """
    Extra-functionality that limits dispatch of generators with chosen
    energy carrier located in the chosen country.
    Add key 'capacity_factor_per_cntr' to args.extra_functionality and set
    limits per carrier in a dictonary with country codes as keys.

    Example:
    'capacity_factor_per_cntr': {'DE':{'run_of_river': [0, 0.5],
                                       'wind_onshore': [0.1, 1]},
                                 'DK':{'wind_onshore':[0, 0.7]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    arg = self.args['extra_functionality']['capacity_factor_per_cntr']
    for cntr in arg.keys():
        carrier = arg[cntr].keys()
        for c in carrier:
            factor = arg[cntr][c]
            gens, potential = _generation_potential(network, c, cntr)

            def _rule_max(m):

                dispatch = sum(m.generator_p[gen, sn] * \
                                  network.snapshot_weightings[sn]
                               for gen in gens
                               for sn in snapshots)

                return dispatch <= factor[1] * potential

            setattr(network.model, "max_flh_" + cntr + '_'+ c,
                    Constraint(rule=_rule_max))

            def _rule_min(m):

                dispatch = sum(m.generator_p[gen, sn] * \
                                    network.snapshot_weightings[sn]
                               for gen in gens
                               for sn in snapshots)

                return dispatch >= factor[0] * potential

            setattr(network.model, "min_flh_" + cntr + '_'+ c,
                    Constraint(rule=_rule_min))

def _capacity_factor_per_cntr_nmp(self, network, snapshots):
    """
    Extra-functionality that limits dispatch of generators with chosen
    energy carrier located in the chosen country.
    Add key 'capacity_factor_per_cntr' to args.extra_functionality and set
    limits per carrier in a dictonary with country codes as keys.

    Example:
    'capacity_factor_per_cntr': {'DE':{'run_of_river': [0, 0.5],
                                       'wind_onshore': [0.1, 1]},
                                 'DK':{'wind_onshore':[0, 0.7]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    arg = self.args['extra_functionality']['capacity_factor_per_cntr']
    for cntr in arg.keys():
        carrier = arg[cntr].keys()
        for c in carrier:
            gens, potential = _generation_potential(network, c, cntr)

            generation = get_var(network, 'Generator', 'p').loc[
                snapshots, gens].mul(network.snapshot_weightings, axis=0)

            define_constraints(network, linexpr((1, generation)).sum().sum(),
                               '>=', arg[cntr][c][0] * potential, 'Generator',
                               'min_flh_' + c + '_' + cntr)
            define_constraints(network, linexpr((1, generation)).sum().sum(),
                               '<=', arg[cntr][c][1] * potential, 'Generator',
                               'max_flh_' + c + '_' + cntr)

def _capacity_factor_per_gen(self, network, snapshots):
    """
    Extra-functionality that limits dispatch for each generator with chosen
    energy carrier.
    Add key 'capacity_factor_per_gen' to args.extra_functionality and set
    limits in a dictonary as a fraction of generation potential.
    Example:
    'capacity_factor_per_gen': {'run_of_river': [0, 0.5], 'solar': [0.1, 1]}


    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    arg = self.args['extra_functionality']['capacity_factor_per_gen']
    carrier = arg.keys()
    snapshots = network.snapshots
    for c in carrier:
        factor = arg[c]
        gens = network.generators.index[network.generators.carrier == c]
        for g in gens:
            if c in ['wind_onshore', 'wind_offshore', 'solar']:
                potential = (network.generators.p_nom[g]*
                             network.generators_t.p_max_pu[g].mul(
                                 network.snapshot_weightings, axis=0)
                            ).sum().sum()
            else:
                potential = network.snapshot_weightings.sum() \
                                * network.generators.p_nom[g].sum()

            def _rule_max(m):

                dispatch = sum(m.generator_p[g, sn] * \
                                   network.snapshot_weightings[sn]
                               for sn in snapshots)

                return dispatch <= factor[1] * potential

            setattr(network.model, "max_flh_" + g,
                    Constraint(gens, rule=_rule_max))

            def _rule_min(m):

                dispatch = sum(m.generator_p[g, sn] * \
                                   network.snapshot_weightings[sn]
                               for sn in snapshots)

                return dispatch >= factor[0] * potential

            setattr(network.model, "min_flh_" + g,
                    Constraint(gens, rule=_rule_min))

def _capacity_factor_per_gen_nmp(self, network, snapshots):
    """
    Extra-functionality that limits dispatch for each generator with chosen
    energy carrier.
    Add key 'capacity_factor_per_gen' to args.extra_functionality and set
    limits in a dictonary as a fraction of generation potential.
    Example:
    'capacity_factor_per_gen': {'run_of_river': [0, 0.5], 'solar': [0.1, 1]}


    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    arg = self.args['extra_functionality']['capacity_factor_per_gen']
    carrier = arg.keys()
    snapshots = network.snapshots
    for c in carrier:
        gens = network.generators.index[network.generators.carrier == c]
        for g in gens:
            if c in ['wind_onshore', 'wind_offshore', 'solar']:
                potential = (network.generators.p_nom[g]*
                             network.generators_t.p_max_pu[g].mul(
                                 network.snapshot_weightings, axis=0)
                            ).sum().sum()
            else:
                potential = network.snapshot_weightings.sum() \
                                * network.generators.p_nom[g].sum()

            generation = get_var(network, 'Generator', 'p').loc[
                snapshots, g].mul(network.snapshot_weightings, axis=0)

            define_constraints(network, linexpr((1, generation)).sum(),
                               '>=', arg[c][0]*potential, 'Generator',
                               'min_flh_' + g)
            define_constraints(network, linexpr((1, generation)).sum(),
                               '<=', arg[c][1]*potential, 'Generator',
                               'max_flh_' + g)

def _capacity_factor_per_gen_cntr(self, network, snapshots):
    """
    Extra-functionality that limits dispatch of each generator with chosen
    energy carrier located in the chosen country.
    Add key 'capacity_factor_per_gen_cntr' to args.extra_functionality and set
    limits per carrier in a dictonary with country codes as keys.

    Example:
    'capacity_factor_per_gen_cntr': {'DE':{'run_of_river': [0, 0.5],
                                       'wind_onshore': [0.1, 1]},
                                 'DK':{'wind_onshore':[0, 0.7]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    arg = self.args['extra_functionality']\
                ['capacity_factor_per_gen_cntr']
    for cntr in arg.keys():

        carrier = arg[cntr].keys()
        snapshots = network.snapshots
        for c in carrier:
            factor = arg[cntr][c]
            gens = network.generators.index[
                (network.generators.carrier == c) &
                (network.generators.bus.astype(str).isin(
                    network.buses.index[network.buses.country_code == cntr]))]
            for g in gens:
                if c in ['wind_onshore', 'wind_offshore', 'solar']:
                    potential = (network.generators.p_nom[g]*
                                 network.generators_t.p_max_pu[g].mul(
                                     network.snapshot_weightings, axis=0)
                                 ).sum().sum()
                else:
                    potential = network.snapshot_weightings.sum() \
                                * network.generators.p_nom[g].sum()

                def _rule_max(m):

                    dispatch = sum(m.generator_p[g, sn] * \
                                   network.snapshot_weightings[sn]
                                   for sn in snapshots)

                    return dispatch <= factor[1] * potential

                setattr(network.model, "max_flh_" + cntr + '_'+ g,
                        Constraint(gens, rule=_rule_max))

                def _rule_min(m):

                    dispatch = sum(m.generator_p[g, sn] * \
                                   network.snapshot_weightings[sn]
                                   for sn in snapshots)

                    return dispatch >= factor[0] * potential

                setattr(network.model, "min_flh_" + cntr + '_'+ g,
                        Constraint(rule=_rule_min))

def _capacity_factor_per_gen_cntr_nmp(self, network, snapshots):
    """
    Extra-functionality that limits dispatch of each generator with chosen
    energy carrier located in the chosen country.
    Add key 'capacity_factor_per_gen_cntr' to args.extra_functionality and set
    limits per carrier in a dictonary with country codes as keys.

    Example:
    'capacity_factor_per_gen_cntr': {'DE':{'run_of_river': [0, 0.5],
                                       'wind_onshore': [0.1, 1]},
                                 'DK':{'wind_onshore':[0, 0.7]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    arg = self.args['extra_functionality']['capacity_factor_per_gen_cntr']
    for cntr in arg.keys():

        carrier = arg[cntr].keys()

        for c in carrier:
            gens = network.generators.index[
                (network.generators.carrier == c) &
                (network.generators.bus.astype(str).isin(
                    network.buses.index[network.buses.country_code == cntr]))]
            for g in gens:
                if c in ['wind_onshore', 'wind_offshore', 'solar']:
                    potential = (network.generators.p_nom[g]*
                                 network.generators_t.p_max_pu[g].mul(
                                     network.snapshot_weightings, axis=0)
                                 ).sum().sum()
                else:
                    potential = network.snapshot_weightings.sum() \
                                * network.generators.p_nom[g].sum()

                generation = get_var(network, 'Generator', 'p').loc[
                    snapshots, g].mul(network.snapshot_weightings, axis=0)

                define_constraints(network, linexpr((1, generation)).sum(),
                                   '>=', arg[cntr][c][0]*potential,
                                   'Generator', 'min_flh_' + g)
                define_constraints(network, linexpr((1, generation)).sum(),
                                   '<=', arg[cntr][c][1]*potential,
                                   'Generator', 'max_flh_' + g)


def snapshot_clustering_daily_bounds(self, network, snapshots):
    # This will bound the storage level to 0.5 max_level every 24th hour.
    sus = network.storage_units
    # take every first hour of the clustered days
    network.model.period_starts = \
        network.snapshot_weightings.index[0::24]

    network.model.storages = sus.index

    print('Setting daily_bounds constraint')

    def day_rule(m, s, p):
        """
        Sets the soc of the every first hour to the
        soc of the last hour of the day (i.e. + 23 hours)
        """
        return (m.state_of_charge[s, p] ==
                m.state_of_charge[s, p + pd.Timedelta(hours=23
                                                      )])

    network.model.period_bound = Constraint(
        network.model.storages,
        network.model.period_starts, rule=day_rule)

def snapshot_clustering_daily_bounds_nmp(self, network, snapshots):

    c = 'StorageUnit'

    period_starts = snapshots[0::24]
    period_ends = period_starts + pd.Timedelta(hours=23)

    eh = expand_series(
        network.snapshot_weightings[period_ends], network.storage_units.index) #elapsed hours

    eff_stand = expand_series(1-network.df(c).standing_loss, period_ends).T
    eff_dispatch = expand_series(network.df(c).efficiency_dispatch, period_ends).T
    eff_store = expand_series(network.df(c).efficiency_store, period_ends).T

    soc = get_var(network, c, 'state_of_charge').loc[period_ends, :]

    soc_peroid_start = get_var(network, c, 'state_of_charge').loc[period_starts]

    coeff_var = [(-1, soc),
                 (-1/eff_dispatch * eh, get_var(network, c, 'p_dispatch').loc[period_ends, :]),
                 (eff_store * eh, get_var(network, c, 'p_store').loc[period_ends, :])]

    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    def masked_term(coeff, var, cols):
        return linexpr((coeff[cols], var[cols]))\
               .reindex(index=axes[0], columns=axes[1], fill_value='').values

    lhs += masked_term(eff_stand, soc_peroid_start, network.storage_units.index)

    rhs = -get_as_dense(network, c, 'inflow', period_ends).mul(eh)

    define_constraints(network, lhs, '==', rhs, 'daily_bounds')

def snapshot_clustering_seasonal_storage(self, network, snapshots, simplified):

    sus = network.storage_units

    network.model.period_starts = \
        network.snapshot_weightings.index[0::24]

    network.model.storages = sus.index

    candidates = \
        network.cluster.index.get_level_values(0).unique()

    # create set for inter-temp constraints and variables
    network.model.candidates = po.Set(
        initialize=candidates, ordered=True)

    # create intra soc variable for each storage and each hour
    network.model.state_of_charge_intra = po.Var(
        sus.index, network.snapshots)

    def intra_soc_rule(m, s, h):
        """
        Sets soc_inter of first hour of every day to 0. Other hours
        are set by technical coherences of storage units

        According to:
        L. Kotzur et al: 'Time series aggregation for energy
        system design:
        Modeling seasonal storage', 2018, equation no. 18
        """

        if h.hour == 0:
            expr = (m.state_of_charge_intra[s, h] == 0)
        else:
            expr = (
                m.state_of_charge_intra[s, h] ==
                m.state_of_charge_intra[s, h-pd.DateOffset(hours=1)]
                * (1 - network.storage_units.at[s, 'standing_loss'])
                -(m.storage_p_dispatch[s, h-pd.DateOffset(hours=1)]/
                  network.storage_units.at[s, 'efficiency_dispatch'] -
                  network.storage_units.at[s, 'efficiency_store'] *
                  m.storage_p_store[s, h-pd.DateOffset(hours=1)]))
        return expr

    network.model.soc_intra = po.Constraint(
        network.model.storages, network.snapshots,
        rule=intra_soc_rule)

    # create inter soc variable for each storage and each candidate
    network.model.state_of_charge_inter = po.Var(
        sus.index, network.model.candidates,
        within=po.NonNegativeReals)

    def inter_storage_soc_rule(m, s, i):
        """
        Define the state_of_charge_inter as the state_of_charge_inter of
        the day before minus the storage losses plus the state_of_charge_intra
        of one hour after the last hour of the representative day.
        For the last reperesentive day, the soc_inter is the same as
        the first day due to cyclic soc condition

        According to:
        L. Kotzur et al: 'Time series aggregation for energy system design:
        Modeling seasonal storage', 2018, equation no. 19
        """

        if i == network.model.candidates[-1]:
            last_hour = network.cluster["last_hour_RepresentativeDay"][i]
            expr = po.Constraint.Skip

        else:
            last_hour = network.cluster["last_hour_RepresentativeDay"][i]
            expr = (
                m.state_of_charge_inter[s, i+1] ==
                m.state_of_charge_inter[s, i]
                * (1 - network.storage_units.at[s, 'standing_loss'])**24
                + m.state_of_charge_intra[s, last_hour]\
                    * (1 - network.storage_units.at[s, 'standing_loss'])\
                    -(m.storage_p_dispatch[s, last_hour]/\
                    network.storage_units.at[s, 'efficiency_dispatch'] -
                      network.storage_units.at[s, 'efficiency_store'] *
                      m.storage_p_store[s, last_hour]))

        return expr

    network.model.inter_storage_soc_constraint = po.Constraint(
        sus.index, network.model.candidates,
        rule=inter_storage_soc_rule)

    #new definition of the state_of_charge used in pypsa
    network.model.del_component('state_of_charge_constraint')
    network.model.del_component('state_of_charge_constraint_index')
    network.model.del_component('state_of_charge_constraint_index_0')
    network.model.del_component('state_of_charge_constraint_index_1')

    def total_state_of_charge(m, s, h):
        """
        Define the state_of_charge as the sum of state_of_charge_inter
        and state_of_charge_intra

        According to:
        L. Kotzur et al: 'Time series aggregation for energy system design:
        Modeling seasonal storage', 2018
        """

        return(m.state_of_charge[s, h] ==
               m.state_of_charge_intra[s, h] + m.state_of_charge_inter[
                   s, network.cluster_ts['Candidate_day'][h]])

    network.model.total_storage_constraint = po.Constraint(
        sus.index, network.snapshots, rule=total_state_of_charge)
    
    if simplified == True: 
        
        def simplified_lower(m,s,h):
            
            intra_min = min(m.state_of_charge_intra[s, :])
            
            return (m.state_of_charge_inter[s, network.cluster_ts['Candidate_day'][h]]
                    * (1 - network.storage_units.at[s, 'standing_loss'])**24
                    + intra_min >= 0)

        network.model.simplified_lower = po.Constraint(
            sus.index, network.cluster_ts.index,
            rule=simplified_lower)
        
        def simplified_upper(m,s,h):
            
            intra_max = max(m.state_of_charge_intra[s, :])
            
            if network.storage_units.p_nom_extendable[s]:
                p_nom = m.storage_p_nom[s]
            else:
                p_nom = network.storage_units.p_nom[s]
          
            return (m.state_of_charge_inter[s, network.cluster_ts['Candidate_day'][h]]
                    + intra_max <= p_nom)
            

        network.model.simplified_upper = po.Constraint(
            sus.index, network.cluster_ts.index,
            rule=simplified_upper)
        
    else: 
        
        def state_of_charge_lower(m, s, h):
            """
            Define the state_of_charge as the sum of state_of_charge_inter
            and state_of_charge_intra
    
            According to:
            L. Kotzur et al: 'Time series aggregation for energy system design:
            Modeling seasonal storage', 2018
            """
    
          # Choose datetime of representive day
            date = str(network.snapshots[
                network.snapshots.dayofyear -1 ==
                network.cluster['RepresentativeDay'][h.dayofyear]][0]).split(' ')[0]
            hour = str(h).split(' ')[1]
    
            intra_hour = pd.to_datetime(date + ' ' + hour)
    
            return(m.state_of_charge_intra[s, intra_hour] +
                   m.state_of_charge_inter[s, network.cluster_ts['Candidate_day'][h]]
                   # * (1 - network.storage_units.at
                   # [s, 'standing_loss']*elapsed_hours)**24
                   >= 0)
    
        network.model.state_of_charge_lower = po.Constraint(
            sus.index, network.cluster_ts.index, rule=state_of_charge_lower)
    
        network.model.del_component('state_of_charge_upper')
        network.model.del_component('state_of_charge_upper_index')
        network.model.del_component('state_of_charge_upper_index_0')
        network.model.del_component('state_of_charge_upper_index_1')
    
        def state_of_charge_upper(m, s, h):
            date = str(network.snapshots[
                network.snapshots.dayofyear -1 ==
                network.cluster['RepresentativeDay'][h.dayofyear]][0]).split(' ')[0]
    
            hour = str(h).split(' ')[1]
    
            intra_hour = pd.to_datetime(date + ' ' + hour)
    
            if network.storage_units.p_nom_extendable[s]:
                p_nom = m.storage_p_nom[s]
            else:
                p_nom = network.storage_units.p_nom[s]
    
            return (m.state_of_charge_intra[s, intra_hour] +
                    m.state_of_charge_inter[s, network.cluster_ts['Candidate_day'][h]]
                    # * (1 - network.storage_units.at[s,
                    # 'standing_loss']*elapsed_hours)**24
                    <= p_nom * network.storage_units.at[s, 'max_hours'])
        
        network.model.state_of_charge_upper = po.Constraint(
            sus.index, network.cluster_ts.index,
            rule=state_of_charge_upper)

    def cyclic_state_of_charge(m, s):
        """
        Defines cyclic condition like pypsas 'state_of_charge_contraint'.
        There are small differences to original results.
        """
        last_day = network.cluster.index[-1]

        last_calc_hour = network.cluster[
            'last_hour_RepresentativeDay'][last_day]

        last_inter = m.state_of_charge_inter[s, last_day]

        last_intra = m.state_of_charge_intra[s, last_calc_hour]

        first_day = network.cluster.index[0]

        first_calc_hour = network.cluster[
            'last_hour_RepresentativeDay'][first_day] - pd.DateOffset(hours=23)

        first_inter = m.state_of_charge_inter[s, first_day]

        first_intra = m.state_of_charge_intra[s, first_calc_hour]

        return  (first_intra + first_inter == \
               ((last_intra + last_inter)
                * (1 - network.storage_units.at[s, 'standing_loss'])
                -(m.storage_p_dispatch[s, last_calc_hour]/
                  network.storage_units.at[s, 'efficiency_dispatch']
                  -m.storage_p_store[s, last_calc_hour] *
                  network.storage_units.at[s, 'efficiency_store'])))

    network.model.cyclic_storage_constraint = po.Constraint(
        sus.index, rule=cyclic_state_of_charge)

def snapshot_clustering_seasonal_storage_nmp(self, n, sns):

    sus = n.storage_units

    c = 'StorageUnit'

    period_starts = sns[0::24]


    candidates = \
        n.cluster.index.get_level_values(0).unique()



    soc_total = get_var(n, c, 'state_of_charge')

    # inter_soc
    # Set lower and upper bound for soc_inter
    lb = pd.DataFrame(index=candidates, columns=sus.index, data=0)
    ub = pd.DataFrame(index=candidates, columns=sus.index, data=np.inf)

    # Create soc_inter variable for each storage and each day
    define_variables(n, lb, ub, 'StorageUnit', 'soc_inter')
    
    # Define soc_intra
    # Set lower and upper bound for soc_intra
    lb = pd.DataFrame(index=sns, columns=sus.index, data=-np.inf)
    ub = pd.DataFrame(index=sns, columns=sus.index, data=np.inf)

    # Set soc_intra to 0 at first hour of every day
    lb.loc[period_starts, :] = 0
    ub.loc[period_starts, :] = 0

    # Create intra soc variable for each storage and each hour
    define_variables(n, lb, ub, 'StorageUnit', 'soc_intra')
    soc_intra = get_var(n, c, 'soc_intra')

    last_hour = n.cluster["last_hour_RepresentativeDay"].values
    first_hour = n.cluster["first_hour_RepresentativeDay"].values

    soc_inter = get_var(n, c, 'soc_inter')
    next_soc_inter = soc_inter.shift(-1).fillna(soc_inter.loc[candidates[0]])
    
    last_soc_intra = soc_intra.loc[last_hour].set_index(candidates)

    eff_stand = expand_series(1-n.df(c).standing_loss, candidates).T
    eff_dispatch = expand_series(n.df(c).efficiency_dispatch, candidates).T
    eff_store = expand_series(n.df(c).efficiency_store, candidates).T
    
    dispatch = get_var(n, c, 'p_dispatch').loc[first_hour].set_index(candidates)
    store = get_var(n, c, 'p_store').loc[first_hour].set_index(candidates)
    next_dispatch =  dispatch.shift(-1).fillna(dispatch.loc[candidates[0]])
    next_store = store.shift(-1).fillna(dispatch.loc[candidates[0]])

    coeff_var = [(-1, next_soc_inter),
                 (eff_stand.pow(24), soc_inter),
                 (eff_stand, last_soc_intra),
                 (-1/eff_dispatch, next_dispatch),
                 (eff_store, next_store)]

    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    define_constraints(n, lhs, '==', 0, c, 'soc_inter_constraints')

    coeff_var = [(-1, soc_total),
                 (1, soc_intra),
                 (1, soc_inter.loc[n.cluster_ts.loc[sns, 'Candidate_day']].set_index(sns))
                 ]
    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    define_constraints(n, lhs, '==', 0, c, 'soc_intra_constraints')


class Constraints:

    def __init__(self, args):
        self.args = args

    def functionality(self, network, snapshots):
        """ Add constraints to pypsa-model using extra-functionality.
        Serveral constraints can be choosen at once. Possible constraints are
        set and described in the above functions.

        Parameters
        ----------
        network : :class:`pypsa.Network
            Overall container of PyPSA
        snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

        """

        for constraint in self.args['extra_functionality'].keys():
            try:
                type(network.model)
                try:
                    eval('_'+constraint+'(self, network, snapshots)')
                    logger.info("Added extra_functionality {}".format(
                        constraint))
                except:
                    logger.warning("Constraint {} not defined".format(
                        constraint)+\
                            ". New constraints can be defined in" +
                                   " etrago/tools/constraint.py.")
            except:
                try:
                    eval('_'+constraint+'_nmp(self, network, snapshots)')
                    logger.info("Added extra_functionality {} without pyomo".
                                format(constraint))
                except:
                    logger.warning("Constraint {} not defined".format(constraint))


        if self.args['snapshot_clustering']['active']:

            if self.args['snapshot_clustering']['storage_constraints'] \
                == 'daily_bounds':

                    if self.args['method']['pyomo']:
                        snapshot_clustering_daily_bounds(self, network, snapshots)
                    else:
                        snapshot_clustering_daily_bounds_nmp(self, network, snapshots)

            elif self.args['snapshot_clustering']['storage_constraints'] \
                == 'soc_constraints':
                    if self.args['method']['pyomo']:
                        snapshot_clustering_seasonal_storage(self, network, snapshots, simplified=False)
                    else:
                        snapshot_clustering_seasonal_storage_nmp(self, network, snapshots)
            elif self.args['snapshot_clustering']['storage_constraints'] \
                == 'soc_constraints_simplified':
                    if self.args['method']['pyomo']:
                        snapshot_clustering_seasonal_storage(self, network, snapshots, simplified=True)
                    else:
                        snapshot_clustering_seasonal_storage_nmp(self, network, snapshots)

            else:
                logger.error('snapshot clustering constraints must be in' +
                             ' [daily_bounds, soc_constraints]')
