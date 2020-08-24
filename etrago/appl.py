# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
"""
This is the application file for the tool eTraGo.
Define your connection parameters and power flow settings before executing
the function etrago.
"""


import datetime
import os
import os.path

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, lukasol, wolfbunke, mariusves, s3pp"


if 'READTHEDOCS' not in os.environ:
    # Sphinx does not run this code.
    # Do not import internal packages directly

    from etrago.network import Etrago

    import oedialect

args = {
    # Setup and Configuration:
    'db': 'oedb',  # database session
    'gridversion': 'v0.4.6',  # None for model_draft or Version number
    'method': 'lopf',  # lopf or pf
    'pf_post_lopf': False,  # perform a pf after a lopf simulation
    'start_snapshot': 1,
    'end_snapshot': 2,
    'solver': 'gurobi',  # glpk, cplex or gurobi
    'solver_options': {'BarConvTol': 1.e-5, 'FeasibilityTol': 1.e-5,
                       'method':2, 'crossover':0,
                       'logFile': 'solver.log'},  # {} for default options
    'model_formulation': 'kirchhoff', # angles or kirchhoff
    'scn_name': 'NEP 2035',  # a scenario: Status Quo, NEP 2035, eGo 100
    'add_sectors': [],
    # Scenario variations:
    'scn_extension': ['BE_NO_NEP 2035', 'nep2035_b2'],  # None or array of extension scenarios
    'scn_decommissioning': 'nep2035_b2',  # None or decommissioning scenario
    # Export options:
    'lpfile': False,  # save pyomo's lp file: False or /path/tofolder
    'csv_export': 'test_2007_nmp',  # save results as csv: False or /path/tofolder
    # Settings:
    'extendable': ['network', 'storage'],  # Array of components to optimize
    'generator_noise': 789456,  # apply generator noise, False or seed number
    'extra_functionality':{},  # Choose function name or {}
    # Clustering:2
    'network_clustering_kmeans': {'n_clusters': 10},  # False or the value k for clustering
    'network_clustering_ehv': False,  # clustering of HV buses to EHV buses.
    'disaggregation': None,  # None, 'mini' or 'uniform'
    'snapshot_clustering': False,  # False or the number of 'periods'
    # Simplifications:
    'skip_snapshots': False,
    'branch_capacity_factor': {'HV': 0.5, 'eHV': 0.7},  # p.u. branch derating
    'load_shedding': False,  # meet the demand at value of loss load cost
    'foreign_lines': {'carrier': 'AC', 'capacity': 'osmTGmod'},
    'comments': None}

def run_etrago(args, json_path):
    """The etrago function works with following arguments:


    Parameters
    ----------

    db : str
        ``'oedb'``,
        Name of Database session setting stored in *config.ini* of *.egoio*

    gridversion : NoneType or str
        ``'v0.2.11'``,
        Name of the data version number of oedb: state ``'None'`` for
        model_draft (sand-box) or an explicit version number
        (e.g. 'v0.2.10') for the grid schema.

    method : str
        ``'lopf'``,
        Choose between a non-linear power flow ('pf') or
        a linear optimal power flow ('lopf').

    pf_post_lopf : bool
        False,
        Option to run a non-linear power flow (pf) directly after the
        linear optimal power flow (and thus the dispatch) has finished.

    start_snapshot : int
        1,
        Start hour of the scenario year to be calculated.

    end_snapshot : int
        2,
        End hour of the scenario year to be calculated.

    solver : str
        'glpk',
        Choose your preferred solver. Current options: 'glpk' (open-source),
        'cplex' or 'gurobi'.

    solver_options: dict
        Choose settings of solver to improve simulation time and result.
        Options are described in documentation of choosen solver.

    model_formulation: str
        'angles'
        Choose formulation of pyomo-model.
        Current options: angles, cycles, kirchhoff, ptdf

    scn_name : str
        'Status Quo',
        Choose your scenario. Currently, there are three different
        scenarios: 'Status Quo', 'NEP 2035', 'eGo100'. If you do not
        want to use the full German dataset, you can use the excerpt of
        Schleswig-Holstein by adding the acronym SH to the scenario
        name (e.g. 'SH Status Quo').

   scn_extension : NoneType or list
       None,
       Choose extension-scenarios which will be added to the existing
       network container. Data of the extension scenarios are located in
       extension-tables (e.g. model_draft.ego_grid_pf_hv_extension_bus)
       with the prefix 'extension_'.
       Currently there are three overlay networks:
           'nep2035_confirmed' includes all planed new lines confirmed by the
           Bundesnetzagentur
           'nep2035_b2' includes all new lines planned by the
           Netzentwicklungsplan 2025 in scenario 2035 B2
           'BE_NO_NEP 2035' includes planned lines to Belgium and Norway and
           adds BE and NO as electrical neighbours

    scn_decommissioning : str
        None,
        Choose an extra scenario which includes lines you want to decommise
        from the existing network. Data of the decommissioning scenarios are
        located in extension-tables
        (e.g. model_draft.ego_grid_pf_hv_extension_bus) with the prefix
        'decommissioning_'.
        Currently, there are two decommissioning_scenarios which are linked to
        extension-scenarios:
            'nep2035_confirmed' includes all lines that will be replaced in
            confirmed projects
            'nep2035_b2' includes all lines that will be replaced in
            NEP-scenario 2035 B2

    lpfile : obj
        False,
        State if and where you want to save pyomo's lp file. Options:
        False or '/path/tofolder'.import numpy as np

    csv_export : obj
        False,
        State if and where you want to save results as csv files.Options:
        False or '/path/tofolder'.

    extendable : list
        ['network', 'storages'],
        Choose components you want to optimize.
        Settings can be added in /tools/extendable.py.
        The most important possibilities:
            'network': set all lines, links and transformers extendable
            'german_network': set lines and transformers in German grid
                            extendable
            'foreign_network': set foreign lines and transformers extendable
            'transformers': set all transformers extendable
            'overlay_network': set all components of the 'scn_extension'
                               extendable
            'storages': allow to install extendable storages
                        (unlimited in size) at each grid node in order to meet
                        the flexibility demand.
            'network_preselection': set only preselected lines extendable,
                                    method is chosen in function call


    generator_noise : bool or int
        State if you want to apply a small random noise to the marginal costs
        of each generator in order to prevent an optima plateau. To reproduce
        a noise, choose the same integer (seed number).

    extra_functionality : dict or None
        None,
        Choose extra functionalities and their parameters for PyPSA-model.
        Settings can be added in /tools/constraints.py.
        Current options are:
            'max_line_ext': float
                Maximal share of network extension in p.u.
            'min_renewable_share': float
                Minimal share of renewable generation in p.u.
            'cross_border_flow': array of two floats
                Limit cross-border-flows between Germany and its neigbouring
                countries, set values in p.u. of german loads in snapshots
                for all countries
                (positiv: export from Germany)
            'cross_border_flows_per_country': dict of cntr and array of floats
                Limit cross-border-flows between Germany and its neigbouring
                countries, set values in p.u. of german loads in snapshots
                for each country
                (positiv: export from Germany)
            'max_curtailment_per_gen': float
                Limit curtailment of all wind and solar generators in Germany,
                values set in p.u. of generation potential.
            'max_curtailment_per_gen': float
                Limit curtailment of each wind and solar generator in Germany,
                values set in p.u. of generation potential.
            'capacity_factor': dict of arrays
                Limit overall energy production for each carrier,
                set upper/lower limit in p.u.
            'capacity_factor_per_gen': dict of arrays
                Limit overall energy production for each generator by carrier,
                set upper/lower limit in p.u.
            'capacity_factor_per_cntr': dict of dict of arrays
                Limit overall energy production country-wise for each carrier,
                set upper/lower limit in p.u.
            'capacity_factor_per_gen_cntr': dict of dict of arrays
                Limit overall energy production country-wise for each generator
                by carrier, set upper/lower limit in p.u.

    network_clustering_kmeans : bool or int
        False,
        State if you want to apply a clustering of all network buses down to
        only ``'k'`` buses. The weighting takes place considering generation
        and load
        at each node. If so, state the number of k you want to apply. Otherwise
        put False. This function doesn't work together with
        ``'line_grouping = True'``.

    load_cluster : bool or obj
        state if you want to load cluster coordinates from a previous run:
        False or /path/tofile (filename similar to ./cluster_coord_k_n_result).

    network_clustering_ehv : bool
        False,
        Choose if you want to cluster the full HV/EHV dataset down to only the
        EHV buses. In that case, all HV buses are assigned to their closest EHV
        sub-station, taking into account the shortest distance on power lines.

    snapshot_clustering : bool or int
        False,
        State if you want to cluster the snapshots and run the optimization
        only on a subset of snapshot periods. The int value defines the number
        of periods (i.e. days) which will be clustered to.
        Move to PyPSA branch:features/snapshot_clustering

    branch_capacity_factor : dict
        {'HV': 0.5, 'eHV' : 0.7},
        Add a factor here if you want to globally change line capacities
        (e.g. to "consider" an (n-1) criterion or for debugging purposes).

    load_shedding : bool
        False,
        State here if you want to make use of the load shedding function which
        is helpful when debugging: a very expensive generator is set to each
        bus and meets the demand when regular
        generators cannot do so.

    foreign_lines : dict
        {'carrier':'AC', 'capacity': 'osmTGmod}'
        Choose transmission technology and capacity of foreign lines:
            'carrier': 'AC' or 'DC'
            'capacity': 'osmTGmod', 'ntc_acer' or 'thermal_acer'

    comments : str
        None

    Returns
    -------
    network : `pandas.DataFrame<dataframe>`
        eTraGo result network based on `PyPSA network
        <https://www.pypsa.org/doc/components.html#network>`_
    """
    etrago = Etrago(args, json_path)

    # TODO: Check if the following function calls should be moved to Etrago.__init__
    # import network from database
    etrago.build_network_from_db()

    # adjust network, e.g. set (n-1)-security factor
    etrago.adjust_network()

    # ehv network clustering
    etrago.ehv_clustering()

    # k-mean clustering
    etrago.kmean_clustering()

    # skip snapshots
    etrago.skip_snapshots()

    # snapshot clustering
    etrago.snapshot_clustering()

    # start linear optimal powerflow calculations
    etrago.lopf()

    # TODO: check if should be combined with etrago.lopf()
    etrago.pf_post_lopf()

    # spaital disaggregation
    etrago.disaggregation()
    # TODO: Do we need a temporal desaggregation???

    # calculate central etrago results
    etrago.calc_results()

    return etrago


if __name__ == '__main__':
    # execute etrago function
    print(datetime.datetime.now())
    etrago = run_etrago(args, json_path='args.json')
    print(datetime.datetime.now())
    # plots
    # make a line loading plot
    # plot_line_loading(network)
    # plot stacked sum of nominal power for each generator type and timestep
    # plot_stacked_gen(network, resolution="MW")
    # plot to show extendable storages
    # storage_distribution(network)
    # extension_overlay_network(network)
