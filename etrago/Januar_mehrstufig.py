#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:43:58 2022

@author: student
"""

import datetime
import os
import os.path
import numpy as np
import geopandas
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pypsa.descriptors import get_switchable_as_dense as as_dense


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
#April 2160 2880
#Januar 1 745
args = {
    # Setup and Configuration:
    "db": "etrago-data",  # database session
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
    "start_snapshot": 2160,
    "end_snapshot": 2185,
    "solver": "gurobi",  # glpk, cplex or gurobi
    "solver_options": {
        "BarConvTol": 1.0e-5,
        "FeasibilityTol": 1.0e-5,
        "method": 2,
        "crossover": 0,
        "logFile": "solver_etragos.log",
        "threads": 4,
    },
    "model_formulation": "kirchhoff",  # angles or kirchhoff
    "scn_name": "eGon2035",  # a scenario: eGon2035 or eGon100RE
    # Scenario variations:
    "scn_extension": None,  # None or array of extension scenarios
    "scn_decommissioning": None,  # None or decommissioning scenario
    # Export options:
    "lpfile": False,  # save pyomo's lp file: False or /path/to/lpfile.lp
    "csv_export": "results",  # save results as csv: False or /path/tofolder
    # Settings:
    "extendable": {
        "extendable_components": ["as_in_db"],  # Array of components to optimize
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
            "grid_max_foreign": 4,  # relative to existing capacity
            "grid_max_abs_foreign": None,  # absolute capacity per voltage level
        },
    },
    "generator_noise": 789456,  # apply generator noise, False or seed number
    "extra_functionality": {},  # Choose function name or {}
    # Spatial Complexity:
    "network_clustering": {
        "random_state": 42,  # random state for replicability of kmeans results
        "active": True,  # choose if clustering is activated
        "method": "kmedoids-dijkstra",  # choose clustering method: kmeans or kmedoids-dijkstra
        "n_clusters_AC": 30,  # total number of resulting AC nodes (DE+foreign)
        "cluster_foreign_AC": False,  # take foreign AC buses into account, True or False
        "method_gas": "kmedoids-dijkstra",  # choose clustering method: kmeans or kmedoids-dijkstra
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
        "tol": 1e-6, # affects clustering algorithm, only change when neccesary
        "CPU_cores": 4, # number of cores used during clustering. "max" for all cores available.
    },
    "sector_coupled_clustering": {
        "active": False,  # choose if clustering is activated
        "carrier_data": {  # select carriers affected by sector coupling
            "central_heat": {"base": ["CH4", "AC"], "strategy": "simultaneous"},
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
    "skip_snapshots": False,  # False or number of snapshots to skip
    "dispatch_disaggregation": False, # choose if full complex dispatch optimization should be conducted
    # Simplifications:
    "branch_capacity_factor": {"HV": 0.5, "eHV": 0.7},  # p.u. branch derating
    "load_shedding": False,  # meet the demand at value of loss load cost
    "foreign_lines": {
        "carrier": "AC",  # 'DC' for modeling foreign lines as links
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
        False or '/path/tofile.lp'

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
                    'dc':lines=etrago.network.lines[['s_nom', 's_nom_min']]0},
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
                Limit AC cross-border-flows between Germany and its neigbouring
                countries, set values in MWh for all snapshots, e.g. [-x, y]
                (with x Import, y Export, positiv: export from Germany)
            'cross_border_flows_per_country': dict of cntr and array of floats
                Limit AC cross-border-flows between Germany and its neigbouring
                countries, set values in in MWh for each country, e.g. [-x, y]
                (with x Import, y Export, positiv: export from Germany)
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
         'central_heat': {'base': ['CH4', 'AC'], 'strategy': "simultaneous"},
        }
        State if you want to apply clustering of sector coupled carriers, such
        as central_heat. The approach builds on already clustered
        buses (e.g. CH4 and AC) and builds clusters around the topology of the
        buses with carrier ``'base'`` for all buses of a specific carrier, e.g.
        ``'central_heat'``. With ``'strategy'`` it is possible to apply either
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
        False,lines=etrago.network.lines[['s_nom', 's_nom_min']]
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
    network : `pandas.Drun_etragoataFrame<dataframe>`
        eTraGo result network based on `PyPSA network
        <https://www.pypsa.org/doc/components.html#network>`_

    """
    etrago = Etrago(args, json_path)
    #from etrago import Etrago
    #etrago = Etrago(csv_folder_name='/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/network')
   # etrago.network.import_from_csv_folder('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/market')
   # market=etrago.network.copy()
   # etrago = Etrago(csv_folder_name='/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/network')
    #import network from database
    etrago.build_network_from_db()
    etrago.drop_sectors(['dsm', 'CH4', 'H2_saltcavern', 'H2_grid', 'central_heat',
      'central_heat_store', 'Li ion', 'H2_ind_load', 'rural_heat', 'rural_heat_store'])
    etrago.network.storage_units.lifetime = np.inf
    etrago.network.transformers.lifetime = 40  # only temporal fix
    etrago.network.lines.lifetime = 40  # only temporal fix until either the
    # PyPSA network clustering function
    # is changed (taking the mean) or our
    # data model is altered, which will
    # happen in the next data creation run

    etrago.network.lines_t.s_max_pu = (
        etrago.network.lines_t.s_max_pu.transpose()
        [etrago.network.lines_t.s_max_pu.columns.isin(
            etrago.network.lines.index)].transpose())

    # Set gas grid links bidirectional
    etrago.network.links.loc[etrago.network.links[
        etrago.network.links.carrier=='CH4'].index, 'p_min_pu'] = -1.

    # Set efficiences of CHP
    etrago.network.links.loc[etrago.network.links[
        etrago.network.links.carrier.str.contains('CHP')].index, 'efficiency'] = 0.43
    
    #Manuelle Eingabe von leeren Länder Einträgen
    etrago.network.buses['country']= etrago.network.buses.apply(lambda x: 'DE' if x.country == None else x.country, axis=1)    

    #mögliche Offshore Anbindungen in der Zukunft
    etrago.network.generators_=etrago.network.generators.drop(['9286','9152'])
    etrago.network.generators_t.p_max_pu= etrago.network.generators_t.p_max_pu.drop(['9286','9152'], axis=1)
    
    #Speicher gleicher Stand wie am Beginn
    etrago.network.storage_units.cyclic_state_of_charge= True
    
    etrago.adjust_network() 
    
    # spatial clustering
    etrago.spatial_clustering()
    
    
    
    # Fehler im Clustering s_nom teils ungleich (<) s_nom_min, behoben? dev pullen
    lines=etrago.network.lines[['s_nom', 's_nom_min']]
    etrago.network.lines.s_nom = etrago.network.lines.s_nom_min
    
    #etrago.spatial_clustering_gas()
    # import pdb; pdb.set_trace()

    #Speicherverluste = 0 auslandspeicher auch gleichsetzen?
    etrago.network.storage_units.efficiency_dispatch = 1   
    etrago.network.storage_units.efficiency_store = 1
    etrago.network.storage_units.standing_loss = 0
    
    #Markt Network zuweisen, um 2 LOPF zu ermöglichen
    etrago.network_market = etrago.network.copy()  
    market =  etrago.network_market
    
    # Marktzonen Zuweisung
  
    #Buses df to geodf | filter AC bus
    geo_bus= geopandas.GeoDataFrame(market.buses, geometry= geopandas.points_from_xy(market.buses.x, market.buses.y))
    geo_bus= geo_bus.loc[geo_bus['carrier'] == 'AC']
    
    #geodf europ. Länder ohne Rus | Koordinatensystem ändern | einstampfen
    europe = geopandas.read_file('/home/student/Documents/Masterthesis/EU Grenzen/Nut/NUTS_RG_01M_2016_3035_LEVL_0.shp')
    #rus = geopandas.read_file('/home/student/Documents/Masterthesis/EU Grenzen/Nut/Russland/RUS_adm0.shp')
    #rus = rus.to_crs(4326)
    europe = europe.to_crs(4326)
    europe = europe[['NUTS_NAME','FID','geometry']]
    
    # Knoten und Länder zusammenführen | Marktzonen zuweisen und reassignen 
    buses_with_country = geo_bus.sjoin(europe, how="left", predicate='intersects')   
    buses_with_country.loc[buses_with_country['FID'].isnull(), ['FID']] = buses_with_country['country']
    buses_with_country.loc[(buses_with_country['FID'] == 'DE') | (buses_with_country['FID'] == 'LU'), ['zone']] = 'DE/LU'
    buses_with_country.loc[buses_with_country['zone'].isnull(), ['zone']] = buses_with_country['FID']
    
    # Länderknoten zuweisen und Generatoren Zonen zuweisen    
    market.buses= buses_with_country
    market.buses.index.names =['bus']
    market.generators=market.generators.reset_index().merge(market.buses[['FID','zone']], how='left', on='bus').set_index(market.generators.index.names)
   
    #Bus mit Zone filtern und zu Series machen
    zones = buses_with_country[['zone']]
    zones = zones.squeeze()
    
    #Liste deropy( Zonen erstellen
    lst_zone = buses_with_country['zone'].unique()
    lst_zone = lst_zone.tolist()
  
    
    #Komponenten mit einem Knoten neuen Knoten zuweisen
    for c in market.iterate_components(market.one_port_components):
        c.df.bus = c.df.bus.map(zones)
    

    #Komponenten mit zwei Knoten neue Knoten zuweisen | alle Komponenten mit Bus0=Bus1 entfernen| Lines müssten noch da sein
    for c in market.iterate_components(market.branch_components):
        c.df.bus0 = c.df.bus0.map(zones)
        c.df.bus1 = c.df.bus1.map(zones)
        internal = c.df.bus0 == c.df.bus1
        market.mremove(c.name, c.df.loc[internal].index)
    
    buses_with_country=buses_with_country.drop(['NUTS_NAME'], axis=1)
    #Knoten entfernen und neue Zonenknoten einfügen
    #market.buses.rename('{}_market'.format)
    market.mremove("Bus", market.buses.index)
    market.madd("Bus", lst_zone)
   
    etrago.network.buses= buses_with_country
    etrago.network.buses.index.names =['bus']
    etrago.network.generators=etrago.network.generators.reset_index().merge(etrago.network.buses[['FID', 'zone']], how='left', on='bus').set_index(etrago.network.generators.index.names)
    
    etrago.export_to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/network_mehrstufig')
    market.export_to_csv_folder('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/market_mehrstufig')
    
    # from etrago import Etrago
    # etrago = Etrago(csv_folder_name='/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/network_mehrstufig')
    
    # import pypsa
    # market=pypsa.Network('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/market_mehrstufig')
    
    market.lines.s_nom = market.lines.s_nom*0.1
    market.lines.s_nom_max= market.lines.s_nom
    market.lines.s_nom_min= market.lines.s_nom_max
    
    market.lopf(solver_name= 'gurobi', pyomo=True)
        
    
  
    ### Redispatchmodell #####
    
    # #Leistung der Generatores pu gleich dem Marktresultat setzen    
    g_p = market.generators_t.p / market.generators.p_nom
    etrago.network.generators_t.p_min_pu = g_p
    etrago.network.generators_t.p_max_pu = g_p
        
    # g_up = etrago.network.generators.loc[(etrago.network.generators['FID'] == 'DE')].copy()
    # g_down = etrago.network.generators.loc[(etrago.network.generators['FID'] == 'DE')].copy()
    
    # g_up.index = g_up.index.map(lambda x: x + " ramp up")
    # g_down.index = g_down.index.map(lambda x: x + " ramp down")
   
   
    # up = (
    #     as_dense(market, "Generator", "p_max_pu") * market.generators.loc[market.generators['FID'] =='DE'].p_nom - market.generators_t.p).clip(0) / market.generators.loc[market.generators['FID'] =='DE'].p_nom
    # up=up.dropna(axis='columns')
    # up.index=up.index.astype('datetime64[ns]')
    
    # #import pdb; pdb.set_trace()
    # down =  -market.generators_t.p / market.generators.loc[market.generators['FID'] =='DE'].p_nom
    # down= down.dropna(axis='columns')
    # down.index=up.index.astype('datetime64[ns]')
    
    # #import pdb; pdb.set_trace()
    
    # up.columns = up.columns.map(lambda x: x + " ramp up")
    # down.columns = down.columns.map(lambda x: x + " ramp down")
    
    # #etrago.network.generators_t.p_max_pu.index = etrago.network.generators_t.p_max_pu.index.astype(object)
    # etrago.network.madd("Generator", g_up.index, p_max_pu=up, **g_up.drop(["p_max_pu"], axis=1))
    # #import pdb; pdb.set_trace()
        
    # etrago.network.madd("Generator", g_down.index, p_min_pu=down, p_max_pu=0, **g_down.drop(["p_max_pu", "p_min_pu"], axis=1))
   
    # ehv network clustering
    etrago.ehv_clustering()


    etrago.args["load_shedding"] = False
    etrago.load_shedding()

    # snapshot clustering
    etrago.snapshot_clustering()

    # skip snapshots
    etrago.skip_snapshots()

    # start linear optimal powerflow calculations
    # needs to be adjusted for new sectors
    #etrago.network.generators.loc[etrago.network.generators.index.str.contains("ramp up"), "marginal_cost"] *= 2
    #etrago.network.generators.loc[etrago.network.generators.index.str.contains("ramp down"), "marginal_cost"] *= -0.5
    #market.export_to_csv_folder('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/market')
    #etrago.export_to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/network')
   # from etrago import Etrago
    #etrago.network.import_from_csv_folder('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/market')
   # market=etrago.network.copy()
   # etrago = Etrago(csv_folder_name='/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/network')
    
    import pdb; pdb.set_trace()
    etrago.lopf()
    
    # fig, axs = plt.subplots(
    # 1, 3, figsize=(20, 10), subplot_kw={"projection": ccrs.AlbersEqualArea()}
    # )
    
    # mkt = (
    #     etrago.network.generators_t.p[market.generators.index].groupby(etrago.network.generators.bus, axis=1)
    #     .sum()
    #     .div(2e6)
    
    # )
    # etrago.network.plot(ax=axs[0],bus_sizes= mkt.sum(), title="Market simulation", geomap=False)
    
    # #up
    # redispatch_up = (
    #     etrago.network.generators_t.p.filter(like="ramp up").groupby(etrago.network.generators.bus,axis=1)
    #     .sum()
    #     .div(2e6)
    # )
    
    
    # etrago.network.plot(
    #     ax=axs[1], bus_sizes=redispatch_up.sum(), bus_colors="blue", title="Redispatch: ramp up",geomap=False
    # )
    
    # #down
    # redispatch_down = (
    #     etrago.network.generators_t.p.filter(like="ramp down").groupby(etrago.network.generators.bus,axis=1)
    #     .sum()
    #     .div(-2e6)
    # )
    
    # etrago.network.plot(
    #     ax=axs[2], bus_sizes=redispatch_down.sum(), bus_colors="red",title="Redispatch: ramp down / curtail", geomap=False,  
    # );
      
    # conduct lopf with full complex timeseries for dispatch disaggregation
    etrago.dispatch_disaggregation()

    # start power flow based on lopf results
    etrago.pf_post_lopf()

    # spatial disaggregation
    # needs to be adjusted for new sectors
    # etrago.disaggregation()

    # calculate central etrago results
    
    etrago.calc_results()
    
    # lines= etrago.network.lines
    # #lines.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/lines.csv')
    # lines.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/lines.csv')
    
    # generator= etrago.network.generators
    # #generator.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/generator.csv')
    # generator.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/generator.csv')
    
    # generator_t_p= etrago.network.generators_t.p
    # #generator_t_p.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/generator_t_p.csv')
    # generator_t_p.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/generator_t_p.csv')
    
    # #objective= etrago.network.objective
    # #objective.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/objective.csv')
    
    # results=etrago.results
    # #results.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/results.csv')
    # results.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/results.csv')
    
    
    
    # redispatch_down_p = etrago.network.generators_t.p.filter(like="ramp down").sum()
    # redispatch_down_p.loc['Total'] = redispatch_down_p.sum()
    # #redispatch_down_p.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/redispatch_down_p.csv')
    # redispatch_down_p.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/redispatch_down_p.csv')
    
    # redispatch_up_p = etrago.network.generators_t.p.filter(like="ramp up").sum().T
    # redispatch_up_p.loc['Total'] = redispatch_up_p.sum()
    # #redispatch_up_p.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/redispatch_up_p.csv')
    # redispatch_up_p.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/redispatch_up_p.csv')
    
    # redispatch_market_p= etrago.network.generators_t.p[etrago.network.generators[etrago.network.generators.index.str.contains('ramp')==False].index].sum().T
    # redispatch_market_p.loc['Total'] = redispatch_market_p.sum()
    # #redispatch_market_p.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/redispatch_market_p.csv')
    # redispatch_market_p.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/redispatch_market_p.csv')    
    
    # #redispatch_p_sum.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/redispatch_p_sum.csv')
    # #redispatch_p_sum.to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/redispatch_p_sum.csv')
   
    # #market.export_to_csv_folder('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/market')
    # #market.export_to_csv_folder('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/market')
    
    # #etrago.export_to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/network')
    # #etrago.export_to_csv('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/Januar/network')
    
    # # from etrago import Etrago
    # # etrago.network.import_from_csv_folder('/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/market')
    # # market=etrago.network.copy()
    # # etrago = Etrago(csv_folder_name='/home/student/eTraGo/git/eTraGo/etrago/eTraGo_szenarien/Redispatch_funktioniert/April/network')
    
    etrago.plot_grid(line_colors='expansion_abs')
    return etrago, market, lines


if __name__ == "__main__":
    # execute etrago function
    print(datetime.datetime.now())
    etrago, market, lines = run_etrago(args, json_path=None)
    print(datetime.datetime.now())
    etrago.session.close()
    # plots
    # make a line loading plot
    # plot_line_loading(network)
    # plot stacked sum of nominal power for each generator type and timestep
    # plot_stacked_gen(network, resolution="MW")
    # plot to show extendable storages
    # storage_distribution(network)
    # extension_overlay_network(network)