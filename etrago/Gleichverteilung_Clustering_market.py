## -*- coding: utf-8 -*-
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
        "active": True,  # choose if perform a pf after a lopf simulation
        "add_foreign_lopf": True,  # keep results of lopf for foreign DC-links
        "q_allocation": "p_nom",
    },  # allocate reactive power via 'p_nom' or 'p'
    "start_snapshot": 2170,
    "end_snapshot": 2180,
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
        "n_clusters_AC": 314,  # total number of resulting AC nodes (DE+foreign)
        "cluster_foreign_AC": False,  # take foreign AC buses into account, True or False
        "method_gas": "kmedoids-dijkstra",  # choose clustering method: kmeans or kmedoids-dijkstra
        "n_clusters_gas": 80,  # total number of resulting CH4 nodes (DE+foreign)
        "cluster_foreign_gas": False,  # take foreign CH4 buses into account, True or False
        "k_elec_busmap": False,  # False or path/to/busmap.csv
        "k_gas_busmap": False,  # False or path/to/ch4_busmap.csv
        "line_length_factor": 1,  #
        "remove_stubs": False,  # remove stubs bevore kmeans clustering
        "use_reduced_coordinates": False,  #
        "bus_weight_tocsv": None,  # None or path/to/bus_weight.csv
        "bus_weight_fromcsv": None,  # None or path/to/bus_weight.csv
        "gas_weight_tocsv": None,  # None or path/to/gas_bus_weight.csv
        "gas_weight_fromcsv": None,  # None or path/to/gas_bus_weight.csv
        "n_init": 10,  # affects clustering algorithm, only change when neccesary
        "max_iter": 100,  # affects clustering algorithm, only change when neccesary
        "tol": 1e-6,  # affects clustering algorithm, only change when neccesary
        "CPU_cores": 4  ,  # number of cores used during clustering. "max" for all cores available.
    },
    "sector_coupled_clustering": {
        "active": True,  # choose if clustering is activated
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
        "extreme_periods": None,  # consideration of extreme timesteps; e.g. 'append'
        "how": "daily",  # type of period, currently only 'daily' - only relevant for 'typical_periods'
        "storage_constraints": "soc_constraints",  # additional constraints for storages  - only relevant for 'typical_periods'
        "n_clusters": 5,  #  number of periods - only relevant for 'typical_periods'
        "n_segments": 5,
    },  # number of segments - only relevant for segmentation
    "skip_snapshots": False,  # False or number of snapshots to skip
    "dispatch_disaggregation": False,  # choose if full complex dispatch optimization should be conducted
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
            'network': set all lines, links and transformers in electrical
                            grid extendable
            'german_network': set lines and transformers in German electrical
                            grid extendable
            'foreign_network': set foreign lines and transformers in electrical
                            grid extendable
            'transformers': set all transformers extendable
            'storages' / 'stores': allow to install extendable storages
                        (unlimited in size) at each grid node in order to meet
                        the flexibility demand.
            'overlay_network': set all components of the 'scn_extension'
                               extendable
            'network_preselection': set only preselected lines extendable,
                                    method is chosen in function call
        Upper bounds for electrical grid expansion can be defined for lines in
        Germany relative to the existing capacity using 'grid_max_D'.
        Alternatively, absolute maximum capacities between two electrical buses
        can be defined per voltage level using 'grid_max_abs_D'.
        Upper bounds for bordercrossing electrical lines can be defined accrodingly
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
    
    # Sektoren auf Strom begrenzen
    ## es gab mal Probleme beim Dev mergen mit gas constraints von Amelia.. 
    etrago.drop_sectors(['Li ion'])
    
    # etrago.drop_sectors(['dsm', 'CH4', 'H2_saltcavern', 'H2_grid', 'central_heat',
    #   'central_heat_store', 'Li ion', 'H2_ind_load', 'rural_heat', 'rural_heat_store'])
    etrago.network.storage_units.lifetime = np.inf
    etrago.network.transformers.lifetime = 40  # only temporal fix
    etrago.network.lines.lifetime = 40  # only temporal fix until either the
    # PyPSA network clustering function
    # is changed (taking the mean) or our
    # data model is altered, which will
    # happen in the next data creation run

    etrago.network.lines_t.s_max_pu = etrago.network.lines_t.s_max_pu.transpose()[
        etrago.network.lines_t.s_max_pu.columns.isin(etrago.network.lines.index)
    ].transpose()

    # Set gas grid links bidirectional
    etrago.network.links.loc[
        etrago.network.links[etrago.network.links.carrier == "CH4"].index, "p_min_pu"
    ] = -1.0

    # Set efficiences of CHP
    etrago.network.links.loc[
        etrago.network.links[etrago.network.links.carrier.str.contains("CHP")].index,
        "efficiency",
    ] = 0.43

    etrago.network.links_t.p_min_pu.fillna(0.0, inplace=True)
    etrago.network.links_t.p_max_pu.fillna(1.0, inplace=True)
    etrago.network.links_t.efficiency.fillna(1.0, inplace=True)

    # Netzwerkanpassungen
    etrago.adjust_network()

    # ehv network clustering
    etrago.ehv_clustering()
    
    #Export des Netzwerkes, bitte eigenen Speicherpfad angeben
    etrago.export_to_csv('/home/student/Documents/Masterthesis/eTraGo_szenarien/Ausgleichenergie/all_allsec')
    
    #Problematische nicht gekoppellte Gas Knoten und Verbindungenlöschen, mit neuen Datenmodell probieren und anschließend entfernen
    # etrago.network.buses= etrago.network.buses.drop(['48044','48111'])
    # etrago.network.links= etrago.network.links.drop(etrago.network.links.loc[((etrago.network.links.bus0 == '48044') | (etrago.network.links.bus1 == '48044') |  (etrago.network.links.bus1 == '48111') |  (etrago.network.links.bus0 == '48111'))].index)
    # etrago.network.storage_units= etrago.network.storage_units.drop(etrago.network.storage_units.loc[((etrago.network.storage_units.bus == '48044')  |  (etrago.network.storage_units.bus == '48111'))].index)
    # etrago.network.stores= etrago.network.stores.drop(etrago.network.stores.loc[((etrago.network.stores.bus == '48044')  |  (etrago.network.stores.bus == '48111'))].index)
    # etrago.network.generators= etrago.network.generators.drop(etrago.network.generators.loc[((etrago.network.generators.bus == '48044')  |  (etrago.network.generators.bus == '48111'))].index)
    # etrago.network.loads= etrago.network.loads.drop(etrago.network.loads.loc[((etrago.network.loads.bus == '48044')  | (etrago.network.loads.bus == '48111'))].index)
    
    # spatial clustering, erste Aggregation auf 300 In- und 14 Auslandsknoten / Ausgangsgröße für Netzberechnung
    etrago.spatial_clustering()
    etrago.export_to_csv('/home/student/Documents/Masterthesis/eTraGo_szenarien/Ausgleichenergie/300_allsec_strom')   
    
    # from etrago import Etrago
    # etrago = Etrago(csv_folder_name='/home/student/Documents/Masterthesis/eTraGo_szenarien/Ausgleichenergie/300_allsec_strom')
    
    etrago.spatial_clustering_gas()
    
    #Export des Netzwerkes, bitte eigenen Speicherpfad angeben
    etrago.export_to_csv('/home/student/Documents/Masterthesis/eTraGo_szenarien/Ausgleichenergie/300_allsec')
    
    # Zweite Aggregation auf 14 Auslandsknoten und 1 Inlandsknoten
    etrago.args["network_clustering"]["n_clusters_AC"] = 15
    
   
    etrago.spatial_clustering() 
    etrago.export_to_csv('/home/student/Documents/Masterthesis/eTraGo_szenarien/Ausgleichenergie/15_allsec_strom')
    
    # Eventueller reimport des Networks
   
    ch4_h2 = pd.read_csv('/home/student/eTraGo/git/eTraGo/etrago/kmeans_gasgrid_80_result.csv')
    #ch4 = pd.read_csv('/home/student/eTraGo/git/eTraGo/etrago/kmeans_ch4_busmap_14_result.csv')
    # ch4_h2.set_index('bus1', inplace=True)
    # ch4_h2.index = ch4_h2.index.astype(str)
    # ch4 = ch4_h2.loc[etrago.network.buses.carrier == 'CH4']
    # h2 = ch4_h2.loc[etrago.network.buses.carrier.str.contains('H2')]
    #ch4 = ch4.index.unique()
    #h2 = h2.index.unique()
    liste = etrago.network.buses.loc[(etrago.network.buses.carrier == 'H2_grid')  | (etrago.network.buses.carrier == 'CH4')]
    liste = liste.loc[liste.duplicated(subset=['x','y'],keep=False)]
    etrago.ch4_h2_mapping= pd.DataFrame(None)
    etrago.ch4_h2_mapping.index= liste.loc[liste.carrier == 'CH4'].index
    etrago.ch4_h2_mapping['0'] = liste.loc[liste.carrier == 'H2_grid'].index
    etrago.ch4_h2_mapping.index.names =['CH4_bus']
    etrago.ch4_h2_mapping['0'] = etrago.ch4_h2_mapping['0'].astype(int)
    etrago.ch4_h2_mapping = etrago.ch4_h2_mapping.squeeze()
    #conn = psycopg2.connect("dbname=test user=postgres")
    
    # from egoio.tools import db
    
    # conn = db.connection(section=etrago.args["db"])
    
    # with conn.connect().execution_options(autocommit=True) as con:
    #     con.execute("DELETE FROM grid.egon_etrago_ch4_h2")
    etrago.args["network_clustering"]["n_clusters_gas"] = 14  
    etrago.spatial_clustering_gas()
    etrago.export_to_csv('/home/student/Documents/Masterthesis/eTraGo_szenarien/Ausgleichenergie/15_allsec')
    # Speicherwirkungsgrad und Verluste optimal, nur für Testzwecke
    # etrago.network.storage_units.efficiency_dispatch = 1   
    # etrago.network.storage_units.efficiency_store = 1
    # etrago.network.storage_units.standing_loss = 0
    
    # Marktzonenzuweisung
    
    # Knoten zu GeoDataFrame macehn
    geo_bus= geopandas.GeoDataFrame(etrago.network.buses, geometry= geopandas.points_from_xy(etrago.network.buses.x, etrago.network.buses.y))
    # geo_bus= geo_bus.loc[geo_bus['carrier'] == 'AC']
    
    # geodf europ. Länder ohne Russland | Koordinatensystem ändern | einstampfen auf benötigte Spalten | hier wären auch andere Grenzen möglich, daür die Datei ändern
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
    etrago.network.buses= buses_with_country
    etrago.network.buses.index.names =['bus']
    etrago.network.generators=etrago.network.generators.reset_index().merge(etrago.network.buses[['FID','zone']], how='left', on='bus').set_index(etrago.network.generators.index.names)
    
    # Bus mit Zone filtern und zu Series machen
    zones = buses_with_country[['zone']]
    zones = zones.squeeze()
    
    # Liste der Zonen erstellen
    lst_zone = buses_with_country['zone'].unique()
    lst_zone = lst_zone.tolist()
    
    # Network vor Optimerung exportieren mit allen Snapshots
    etrago.export_to_csv('/home/student/Documents/Masterthesis/eTraGo_szenarien/Ausgleichenergie/market_every_snap_allsec')
    
    
    
    # Load Shedding einstellen, für schwer lösbare Konstellation nötig
    etrago.args["load_shedding"] = False
    etrago.load_shedding()   
    
    # Speicherstand gleich Anfangsspeicherstand
    etrago.network.storage_units.cyclic_state_of_charge= True
    
    # snapshot clustering
    etrago.snapshot_clustering()
    
    # skip snapshots
    etrago.skip_snapshots()

    # Start Optimierung
    etrago.lopf()
    
    # export Daten des optimierten Netzes
    etrago.export_to_csv('/home/student/Documents/Masterthesis/eTraGo_szenarien/Ausgleichenergie/market_allsec')
    
    # conduct lopf with full complex timeseries for dispatch disaggregation
    etrago.dispatch_disaggregation()

    # start power flow based on lopf results
    # etrago.pf_post_lopf()

    # spatial disaggregation
    # needs to be adjusted for new sectors
    # etrago.disaggregation()

    # calculate central etrago results
    etrago.calc_results()

    etrago.plot_grid(line_colors='expansion_abs')
    return etrago


if __name__ == "__main__":
    # execute etrago function
    print(datetime.datetime.now())
    etrago = run_etrago(args, json_path=None)
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
