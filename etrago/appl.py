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
import numpy as np
import sys

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, lukasol, wolfbunke, mariusves, s3pp"


if "READTHEDOCS" not in os.environ:
    # Sphinx does not run this code.
    # Do not import internal packages directly

    from etrago import Etrago

args = {
    # Setup and Configuration:
    "db": "egon-data-clara",  # database session
    "gridversion": None,  # None for model_draft or Version number
    "method": {  # Choose method and settings for optimization
        "type": "lopf",  # type of optimization, currently only 'lopf'
        "n_iter": 4,  # abort criterion of iterative optimization, 'n_iter' or 'threshold'
        "pyomo": True,
    },  # set if pyomo is used for model building
    "pf_post_lopf": {
        "active": False,  # choose if perform a pf after a lopf simulation
        "add_foreign_lopf": True,  # keep results of lopf for foreign DC-links
        "q_allocation": "p_nom",
    },  # allocate reactive power via 'p_nom' or 'p'
    "start_snapshot": 1,
    "end_snapshot": 8760,
    "solver": "gurobi",  # glpk, cplex or gurobi
    "solver_options": {
        'BarConvTol': 1.e-5,
        'FeasibilityTol': 1.e-5,
        'method':2,
        'crossover':0,
        'logFile': 'solver_etragos.log',
        'threads': 4},
    "model_formulation": "kirchhoff",  # angles or kirchhoff
    "scn_name": "eGon2035",  # a scenario: eGon2035 or eGon100RE
    # Scenario variations:
    "scn_extension": None,  # None or array of extension scenarios
    "scn_decommissioning": None,  # None or decommissioning scenario
    # Export options:
    "lpfile": 'lp-file.lp',  # save pyomo's lp file: False or /path/tofolder
    "csv_export": "results",  # save results as csv: False or /path/tofolder
    # Settings:
    "extendable": {
        "extendable_components": ["network"],  # Array of components to optimize
        "upper_bounds_grid": {  # Set upper bounds for grid expansion
            # lines in Germany
            "grid_max_D": None,  # relative to existing capacity
            "grid_max_abs_D": {  # absolute capacity per voltage level
                "380": {"i": 1020, "wires": 4, "circuits": 4},
                "220": {"i": 1020, "wires": 4, "circuits": 4},
                "110": {"i": 1020, "wires": 4, "circuits": 2},
                "dc": 0,
            },
            # border crossing lines
            "grid_max_foreign": 2,  # relative to existing capacity
            "grid_max_abs_foreign": None,  # absolute capacity per voltage level
        },
    },
    "generator_noise": 789456,  # apply generator noise, False or seed number
    "extra_functionality": {},  # Choose function name or {} # 'cross_border_flow':[-0.1, 1.0]
    # Spatial Complexity:
    "network_clustering": {
        "random_state": 42,  # random state for replicability of kmeans results
        "active": True,  # choose if clustering is activated
        "method": "kmeans",  # choose clustering method: kmeans or kmedoids-dijkstra
        "n_clusters_AC": 100,  # total number of resulting AC nodes (DE+foreign)
        "cluster_foreign_AC": False,  # take foreign AC buses into account, True or False
        "method_gas": "kmeans",  # choose clustering method: kmeans (kmedoids-dijkstra not yet implemented)
        "n_clusters_gas": 17,  # total number of resulting CH4 nodes (DE+foreign)
        "cluster_foreign_gas": False,  # take foreign CH4 buses into account, True or False
        "k_busmap": False,  # False or path/to/busmap.csv
        "kmeans_gas_busmap": False,  # False or path/to/ch4_busmap.csv
        "line_length_factor": 1,  #
        "remove_stubs": False,  # remove stubs bevore kmeans clustering
        "use_reduced_coordinates": False,  #
        "bus_weight_tocsv": None,  # None or path/to/bus_weight.csv
        "bus_weight_fromcsv": None,  # None or path/to/bus_weight.csv
        "gas_weight_tocsv": None,  # None or path/to/gas_bus_weight.csv
        "gas_weight_fromcsv": None,  # None or path/to/gas_bus_weight.csv
        "n_init": 10,  # affects clustering algorithm, only change when neccesary
        "max_iter": 100,  # affects clustering algorithm, only change when neccesary
        "tol": 1e-6,
    },  # affects clustering algorithm, only change when neccesary
    "sector_coupled_clustering": {
        "active": False,  # choose if clustering is activated
        "carrier_data": {  # select carriers affected by sector coupling
            "H2_ind_load": {"base": ["H2_grid"], "strategy": "consecutive"},
            "central_heat": {"base": ["CH4", "AC"], "strategy": "consecutive"},
            "rural_heat": {"base": ["CH4", "AC"], "strategy": "consecutive"},
        },
    },
    "network_clustering_ehv": False,  # clustering of HV buses to EHV buses.
    "disaggregation": None,  # None, 'mini' or 'uniform'
    # Temporal Complexity:
    "snapshot_clustering": {
        "active": False,  # choose if clustering is activated
        "method": "segmentation",  # 'typical_periods' or 'segmentation'
        "extreme_periods": None, # consideration of extreme timesteps; e.g. 'append'
        "how": "daily",  # type of period, currently only 'daily' - only relevant for 'typical_periods'
        "storage_constraints": "soc_constraints",  # additional constraints for storages  - only relevant for 'typical_periods'
        "n_clusters": 5,  #  number of periods - only relevant for 'typical_periods'
        "n_segments": 5,
    },  # number of segments - only relevant for segmentation
    "skip_snapshots": 5,  # False or number of snapshots to skip
    "dispatch_disaggregation": False, # choose if full complex dispatch optimization should be conducted
    # Simplifications:
    "branch_capacity_factor": {"HV": 0.5, "eHV": 0.7},  # p.u. branch derating
    "load_shedding": False,  # meet the demand at value of loss load cost
    "foreign_lines": {
        "carrier": "DC",  # 'DC' for modeling foreign lines as links
        "capacity": "osmTGmod",
    },  # 'osmTGmod', 'tyndp2020', 'ntc_acer' or 'thermal_acer'
    "comments": None,
}


def run_etrago(args, json_path):
    """The etrago function works with following arguments:


     Parameters
     ----------

     db : str
         ``'oedb'``,
         Name of Database session setting stored in *config.ini* of *.egoio*

     gridversion : NoneType or str
         ``'v0.4.6'``,
         Name of the data version number of oedb: state ``'None'`` for
         model_draft (sand-box) or an explicit version number
         (e.g. 'v0.4.6') for the grid schema.

     method : dict
         {'type': 'lopf', 'n_iter': 5, 'pyomo': True},
         Choose 'lopf' for 'type'. In case of extendable lines, several lopfs
         have to be performed. Choose either 'n_init' and a fixed number of
         iterations or 'thershold' and a threashold of the objective function as
         abort criteria.
         Set 'pyomo' to False for big optimization problems, currently only
         possible when solver is 'gurobi'.

     pf_post_lopf :dict
         {'active': True, 'add_foreign_lopf': True, 'q_allocation': 'p_nom'},
         Option to run a non-linear power flow (pf) directly after the
         linear optimal power flow (and thus the dispatch) has finished.
         If foreign lines are modeled as DC-links (see foreign_lines), results
         of the lopf can be added by setting 'add_foreign_lopf'.
         Reactive power can be distributed either by 'p_nom' or 'p'.

     start_snapshot : int
         1,
         Start hour of the scenario year to be calculated.

     end_snapshot : int
         2,
         End hour of the scenario year to be calculated.
         If temporal clustering is used, the selected snapshots should cover
         whole days.

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
         'eGon2035',
         Choose your scenario. Currently, there are two different
         scenarios: 'eGon2035', 'eGon100RE'.

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

    extendable : dict
        {'extendable_components': ['as_in_db'],
            'upper_bounds_grid': {
                'grid_max_D': None,
                'grid_max_abs_D': {
                    '380':{'i':1020, 'wires':4, 'circuits':4},
                    '220':{'i':1020, 'wires':4, 'circuits':4},
                    '110':{'i':1020, 'wires':4, 'circuits':2},
                    'dc':0},
                'grid_max_foreign': 4,
                'grid_max_abs_foreign': None}},
        ['network', 'storages'],
        Choose components you want to optimize and set upper bounds for grid expansion.
        The list 'extendable_components' defines a set of components to optimize.
        Settings can be added in /tools/extendable.py.
        The most important possibilities:
            'as_in_db': leaves everything as it is defined in the data coming
                        from the database
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
        Upper bounds for grid expansion can be set for lines in Germany can be
        defined relative to the existing capacity using 'grid_max_D'.
        Alternatively, absolute maximum capacities between two buses can be
        defined per voltage level using 'grid_max_abs_D'.
        Upper bounds for bordercrossing lines can be defined accrodingly
        using 'grid_max_foreign' or 'grid_max_abs_foreign'.

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

    network_clustering : dict
          {'active': True, method: 'kmedoids-dijkstra', 'n_clusters_AC': 30,
           'cluster_foreign_AC': False, method_gas: 'kmeans',
           'n_clusters_gas': 30, 'cluster_foreign_gas': False,
           'k_busmap': False, 'kmeans_gas_busmap': False, 'line_length_factor': 1,
           'remove_stubs': False, 'use_reduced_coordinates': False,
           'bus_weight_tocsv': None, 'bus_weight_fromcsv': None,
           'gas_weight_tocsv': None, 'gas_weight_fromcsv': None, 'n_init': 10,
           'max_iter': 100, 'tol': 1e-6},
        State if you want to apply a clustering of all network buses.
        When ``'active'`` is set to True, the AC buses are clustered down to
        ``'n_clusters_AC'`` and ``'n_clusters_gas'``buses. If ``'cluster_foreign_AC'`` is set to False,
        the AC buses outside Germany are not clustered, and the buses inside
        Germany are clustered to complete ``'n_clusters'`` buses.
        The weighting takes place considering generation and load at each node. CH-4 nodes also take
        non-transport capacities into account.
        ``'cluster_foreign_gas'`` controls whether gas buses of Germanies
        neighboring countries are considered for clustering.
        With ``'method'`` you can choose between two clustering methods:
        k-means Clustering considering geopraphical locations of buses or
        k-medoids Dijkstra Clustering considering electrical distances between buses.
        With ``'k_busmap'`` you can choose if you want to load cluster
        coordinates from a previous run.
        Option ``'remove_stubs'`` reduces the overestimating of line meshes.
        The other options affect the kmeans algorithm and should only be
        changed carefully, documentation and possible settings are described
        in sklearn-package (sklearn/cluster/k_means_.py).
        This function doesn't work together with
        ``'network_clustering_kmedoids_dijkstra`` and ``'line_grouping = True'``.

    sector_coupled_clustering : nested dict
        {'active': True, 'carrier_data': {
         'H2_ind_load': {'base': ['H2_grid'], 'strategy': "consecutive"},
         'central_heat': {'base': ['CH4', 'AC'], 'strategy': "consecutive"},
         'rural_heat': {'base': ['CH4', 'AC']}, 'strategy': "consecutive"}
        }
        State if you want to apply clustering of sector coupled carriers, such
        as central_heat or rural_heat. The approach builds on already clustered
        buses (e.g. CH4 and AC) and builds clusters around the topology of the
        buses with carrier ``'base'`` for all buses of a specific carrier, e.g.
        ``'H2_ind_load'``. With ``'strategy'`` it is possible to apply either
        ``'consecutive'`` or ``'simultaneous'`` clustering. The consecutive
        strategy clusters around the buses of the first carrier in the list.
        The links to other buses are preserved. All buses, that have no
        connection to the first carrier will then be clustered around the buses
        of the second carrier in the list. The simultanous strategy looks for
        links connecting the buses of the carriers in the list and aggregates
        buses in case they have the same set of links connected. For example,
        a heat bus connected to CH4 via gas boiler and to AC via heat pump will
        only form a cluster with other buses, if these have the same links to
        the same clusters of CH4 and AC.

    network_clustering_ehv : bool
        False,
        Choose if you want to cluster the full HV/EHV dataset down to only the
        EHV buses. In that case, all HV buses are assigned to their closest EHV
        sub-station, taking into account the shortest distance on power lines.

    snapshot_clustering : dict
        {'active': False, 'method':'typical_periods', 'how': 'daily',
         'extreme_periods': None, 'storage_constraints': '', 'n_clusters': 5, 'n_segments': 5},
        State if you want to apply a temporal clustering and run the optimization
        only on a subset of snapshot periods.
        You can choose between a method clustering to typical periods, e.g. days
        or a method clustering to segments of adjacent hours.
        With ``'extreme_periods'`` you define the consideration of timesteps with
        extreme residual load while temporal aggregation.
        With ``'how'``, ``'storage_constraints'`` and ``'n_clusters'`` you choose
        the length of the periods, constraints considering the storages and the number
        of clusters for the usage of the method typical_periods.
        With ``'n_segments'`` you choose the number of segments for the usage of
        the method segmentation.

    skip_snapshots : bool or int
        State if you only want to consider every n-th timestep
        to reduce temporal complexity.

    dispatch_disaggregation : bool
        State if you to apply a second lopf considering dispatch only
        to disaggregate the dispatch to the whole temporal complexity.

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

    # import network from database
    etrago.build_network_from_db()

    # manual fixes for database-network from 16th of August

    etrago.network.lines.type = ""
    etrago.network.buses.v_mag_pu_set.fillna(1.0, inplace=True)
    etrago.network.storage_units.lifetime = 40
    etrago.network.transformers.lifetime = 40
    etrago.network.lines.lifetime = 40
    # only temporal fix until either the
    # PyPSA network clustering function
    # is changed (taking the mean) or our
    # data model is altered, which will
    # happen in the next data creation run
    etrago.network.links_t.p_max_pu.fillna(1.0, inplace=True)
    etrago.network.links_t.efficiency.fillna(1.0, inplace=True)
    etrago.network.links_t.p_max_pu.fillna(0.0, inplace=True)

    etrago.network.lines_t.s_max_pu = (
        etrago.network.lines_t.s_max_pu.transpose()
        [etrago.network.lines_t.s_max_pu.columns.isin(
            etrago.network.lines.index)].transpose())

    for t in etrago.network.iterate_components():
        if "p_min_pu" in t.df:
            t.df["p_min_pu"].fillna(0.0, inplace=True)

    for t in etrago.network.iterate_components():
        if "p_max_pu" in t.df:
            t.df["p_max_pu"].fillna(1., inplace=True)

    # Temporary fix until egon-data issue #815 is solved
    foreign_generators = etrago.network.generators[
        etrago.network.generators.bus.isin(
            etrago.network.buses.index[etrago.network.buses.country!="DE"]) &
        (etrago.network.generators.index.isin(
            etrago.network.generators_t.p_max_pu.columns))].index

    if etrago.args["end_snapshot"] == 8760:
        for i in foreign_generators:
            etrago.network.generators_t.p_max_pu[i] = np.repeat(
                etrago.network.generators_t.p_max_pu[i][:int(8760/3)].values, 3)

    # Temporary fix missing marginal costs of foreign generators
    etrago.network.generators.loc[etrago.network.generators.carrier=='solar', 'marginal_cost']  = 0
    etrago.network.generators.loc[etrago.network.generators.carrier=='solar_rooftop', 'marginal_cost']  = 0
    etrago.network.generators.loc[etrago.network.generators.carrier=='wind_onshore', 'marginal_cost'] = 1.3
    etrago.network.generators.loc[etrago.network.generators.carrier=='wind_offshore', 'marginal_cost'] =2.5
    etrago.network.generators.loc[etrago.network.generators.carrier=='nuclear', 'marginal_cost'] += 1.7
    etrago.network.generators.loc[etrago.network.generators.carrier=='lignite', 'marginal_cost'] += 4 + 0.393* 76.5
    etrago.network.generators.loc[etrago.network.generators.carrier=='coal', 'marginal_cost'] += 20.2 + 0.335 *  76.5

    # only electricity sector, no DSM and no DLR

    etrago.drop_sectors(['CH4', 'H2_saltcavern', 'H2_grid', 'H2_ind_load', 'dsm', 'central_heat',
     'rural_heat', 'central_heat_store', 'rural_heat_store', 'Li ion'])

    # no DLR
    etrago.network.lines_t.s_max_pu[etrago.network.lines_t.s_max_pu != 1] = 1

    etrago.adjust_network()
    
    # avoid usage of cheap storages in foreign countries to provoke network expansion
    aus = etrago.network.buses[etrago.network.buses.country!='DE']
    sto_aus = etrago.network.storage_units[etrago.network.storage_units.bus.isin(aus.index)]
    sto_aus_bat = sto_aus[sto_aus.carrier=='battery']
    etrago.network.storage_units.loc[etrago.network.storage_units.index.isin(sto_aus_bat.index),'p_nom'] = 0
    etrago.network.storage_units.loc[etrago.network.storage_units.index.isin(sto_aus_bat.index),'p_nom_extendable'] = True
    etrago.network.storage_units.loc[etrago.network.storage_units.index.isin(sto_aus_bat.index),'capital_cost'] = 64763.66650832

    '''# Set foreign batteries extendable
    etrago.network.storage_units["country"] = etrago.network.buses.loc[
        etrago.network.storage_units.bus.values, "country"
    ].values
    etrago.network.storage_units[
        (etrago.network.storage_units.country!='DE')&(etrago.network.storage_units.carrier=='battery')
        ].capital_cost = etrago.network.storage_units[(etrago.network.storage_units.country=='DE')&(etrago.network.storage_units.carrier=='battery')
        ].capital_cost.mean()
    etrago.network.storage_units[
        (etrago.network.storage_units.country!='DE')&(etrago.network.storage_units.carrier=='battery')
        ].p_nom_extendable = True'''

    # ehv network clustering
    #etrago.ehv_clustering()

    etrago.export_to_csv("before_spatial")

    print(' ')
    print('start spatial clustering')
    print(datetime.datetime.now())
    print(' ')

    # spatial clustering
    etrago.spatial_clustering()

    print(' ')
    print('stop spatial clustering')
    print(datetime.datetime.now())
    print(' ')

    etrago.export_to_csv("after_spatial")

    #etrago.spatial_clustering_gas()

    etrago.args["load_shedding"] = True
    etrago.load_shedding()

    # snapshot clustering
    # etrago.snapshot_clustering()

    # skip snapshots
    etrago.skip_snapshots()

    # start linear optimal powerflow calculations
    etrago.lopf()

    etrago.export_to_csv("network_results")
    
    from etrago.tools.utilities import modular_weight
    print(' ')
    print('Modularity')
    print(modular_weight(etrago.busmap['orig_network'],etrago.busmap['busmap']))
    print(' ')

    # conduct lopf with full complex timeseries for dispatch disaggregation
    # etrago.dispatch_disaggregation()

    # TODO: check if should be combined with etrago.lopf()
    # needs to be adjusted for new sectors
    # etrago.pf_post_lopf()

    # spatial disaggregation
    # needs to be adjusted for new sectors
    # etrago.disaggregation()

    # calculate central etrago results
    # etrago.calc_results()

    return etrago


if __name__ == "__main__":
    # execute etrago function
    
    old_stdout = sys.stdout
    log_file = open('console.log',"w")
    sys.stdout = log_file
    
    print(datetime.datetime.now())
    etrago = run_etrago(args, json_path=None)
    print(datetime.datetime.now())
    
    sys.stdout = old_stdout
    log_file.close()
    
    etrago.session.close()
    # plots
    # make a line loading plot
    # plot_line_loading(network)
    # plot stacked sum of nominal power for each generator type and timestep
    # plot_stacked_gen(network, resolution="MW")
    # plot to show extendable storages
    # storage_distribution(network)
    # extension_overlay_network(network)
