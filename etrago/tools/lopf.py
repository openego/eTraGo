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

    from pypsa.linopf import network_lopf
    from etrago.tools.constraints import Constraints
    from etrago.tools.utilities import (results_to_csv)

    logger = logging.getLogger(__name__)

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"

def update_electrical_parameters(network, l_snom_pre, t_snom_pre):

    """
    Update electrical parameters of active branch components
    considering s_nom of previous iteration

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    l_snom_pre: pandas.Series
        s_nom of ac-lines in previous iteration
    t_snom_pre: pandas.Series
        s_nom of transformers in previous iteration
    """

    network.lines.x[network.lines.s_nom_extendable] = \
    network.lines.x * l_snom_pre / network.lines.s_nom_opt

    network.transformers.x[network.transformers.s_nom_extendable] = \
    network.transformers.x * t_snom_pre /\
    network.transformers.s_nom_opt

    network.lines.r[network.lines.s_nom_extendable] = \
    network.lines.r * l_snom_pre / network.lines.s_nom_opt

    network.transformers.r[network.transformers.s_nom_extendable] = \
    network.transformers.r * t_snom_pre /\
    network.transformers.s_nom_opt

    network.lines.g[network.lines.s_nom_extendable] = \
    network.lines.g * network.lines.s_nom_opt / l_snom_pre

    network.transformers.g[network.transformers.s_nom_extendable] = \
    network.transformers.g * network.transformers.s_nom_opt /\
    t_snom_pre

    network.lines.b[network.lines.s_nom_extendable] = \
    network.lines.b * network.lines.s_nom_opt / l_snom_pre

    network.transformers.b[network.transformers.s_nom_extendable] = \
    network.transformers.b * network.transformers.s_nom_opt /\
    t_snom_pre

    # Set snom_pre to s_nom_opt for next iteration
    l_snom_pre = network.lines.s_nom_opt.copy()
    t_snom_pre = network.transformers.s_nom_opt.copy()

    return l_snom_pre, t_snom_pre


def run_lopf(etrago, extra_functionality, method):
    """ Function that performs lopf with or without pyomo


    Parameters
    ----------
    etrago : :class:`etrago.Etrago
        Transmission grid object
    extra_functionality: str
        Define extra constranits.
    method: dict
        Choose 'n_iter' and integer for fixed number of iterations or
        'threshold' and derivation of objective in percent for variable number
        of iteration until the threshold of the objective function is reached

    Returns
    -------
    None.

    """

    x = time.time()
    if method['pyomo']:
        etrago.network.lopf(
            etrago.network.snapshots,
            solver_name=etrago.args['solver'],
            solver_options=etrago.args['solver_options'],
            extra_functionality=extra_functionality,
            formulation=etrago.args['model_formulation'])

        if etrago.network.results["Solver"][0]["Status"] != 'ok':
            raise  Exception('LOPF not solved.')

    else:
        status, termination_condition = network_lopf(
            etrago.network,
            solver_name=etrago.args['solver'],
            solver_options=etrago.args['solver_options'],
            extra_functionality=extra_functionality,
            formulation=etrago.args['model_formulation'])

        if status != 'ok':
            raise  Exception('LOPF not solved.')
    y = time.time()
    z = (y - x) / 60

    print("Time for LOPF [min]:", round(z, 2))

def iterate_lopf(etrago, extra_functionality, method={'n_iter':4, 'pyomo':True},
                 ):

    """
    Run optimization of lopf. If network extension is included, the specified
    number of iterations is calculated to consider reactance changes.

    Parameters
    ----------
    etrago : :class:`etrago.Etrago
        Transmission grid object
    extra_functionality: str
        Define extra constranits.
    method: dict
        Choose 'n_iter' and integer for fixed number of iterations or
        'threshold' and derivation of objective in percent for variable number
        of iteration until the threshold of the objective function is reached

    """

    network = etrago.network
    args = etrago.args
    # if network is extendable, iterate lopf
    # to include changes of electrical parameters
    if network.lines.s_nom_extendable.any():

        # Initialise s_nom_pre (s_nom_opt of previous iteration)
        # to s_nom for first lopf:
        l_snom_pre = network.lines.s_nom.copy()
        t_snom_pre = network.transformers.s_nom.copy()

        # calculate fixed number of iterations
        if 'n_iter' in method:
            n_iter = method['n_iter']

            for i in range(1, (1+n_iter)):

                run_lopf(etrago, extra_functionality, method)

                if args['csv_export'] != False:
                    path = args['csv_export'] + '/lopf_iteration_'+ str(i)
                    results_to_csv(network, args, path)

                if i < n_iter:
                    l_snom_pre, t_snom_pre = \
                    update_electrical_parameters(network,
                                                 l_snom_pre, t_snom_pre)

        # Calculate variable number of iterations until threshold of objective
        # function is reached

        if 'threshold' in method:

            run_lopf(etrago, extra_functionality, method)

            diff_obj = network.objective*method['threshold']/100

            i = 1

            # Stop after 100 iterations to aviod unending loop
            while i <= 100:

                if i == 100:
                    print('Maximum number of iterations reached.')
                    break

                l_snom_pre, t_snom_pre = \
                    update_electrical_parameters(network,
                                                 l_snom_pre, t_snom_pre)
                pre = network.objective

                run_lopf(etrago, extra_functionality, method)

                i += 1

                if args['csv_export'] != False:
                    path = args['csv_export'] + '/lopf_iteration_'+ str(i)
                    results_to_csv(network, args, path)

                if abs(pre-network.objective) <= diff_obj:
                    print('Threshold reached after ' + str(i) + ' iterations.')
                    break

    else:
        run_lopf(etrago, extra_functionality, method)

    if args['csv_export'] != False:
        path = args['csv_export']
        results_to_csv(network, args, path)

    if not args['lpfile'] is False:
        network.model.write(
            args['lpfile'], io_options={
                'symbolic_solver_labels': True})

    return network

def lopf(self):

    # TODO: Check if Constraints can be added to etrago object

    # set deafault settings
    lopf_settings = {'type': 'lopf', 'n_iter':5, 'pyomo':True}

    # overwirte default values if given in args
    if type(self.args['method']) == dict:
        for k in self.args['method'].keys():
            lopf_settings[k] = self.args['method'][k]

    else:
        lopf_settings['type'] = self.args['method']

    x = time.time()
    if lopf_settings['type'] == 'lopf':
        try:
            from vresutils.benchmark import memory_logger
            with memory_logger(filename=self.args['csv_export']+'_memory.log',
                               interval=30.) as mem:
                iterate_lopf(self,
                             Constraints(self.args).functionality,
                             method=lopf_settings)
        except:
            iterate_lopf(self,
                         Constraints(self.args).functionality,
                         method=lopf_settings)

        print("Maximum memory usage: {} MB".format(round(mem.mem_usage[0], 1)))

    elif self.args['method']['type'] == 'ilopf':
        from pypsa.linopf import ilopf
        # Temporary set all line types
        self.network.lines.type = 'Al/St 240/40 4-bundle 380.0'
        x = time.time()
        ilopf(self.network, solver_name=self.args['solver'],
              solver_options=self.args['solver_options'])

    y = time.time()
    z = (y - x) / 60
    logger.info("Time for LOPF [min]: {}".format(round(z, 2)))
