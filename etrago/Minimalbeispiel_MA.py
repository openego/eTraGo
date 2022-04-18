
import datetime
import os
import os.path
import sys

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from etrago import Etrago
from etrago.tools.utilities import convert_capital_costs
from etrago.tools.utilities import load_shedding

def import_data():
    
    ### Electricity Sector
    
    # wind generator 
    
    wind = pd.read_csv('Daten_Minibsp/gen_wind.csv')
    wind = pd.Series(wind['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    
    pv = pd.read_csv('Daten_Minibsp/gen_pv.csv')
    pv = pd.Series(pv['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    
    # electrical load
    
    el_load = pd.read_csv('Daten_Minibsp/el_load.csv')
    el_load = pd.Series(el_load['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    
    # DSM
    
    dsm = pd.read_csv('Daten_Minibsp/dsm_link.csv')
    dsm_link = pd.DataFrame(index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    dsm_link['p_max_pu']=dsm['p_max_pu'].values
    dsm_link['p_min_pu']=dsm['p_min_pu'].values
    dsm_link['p_nom']=dsm['p_nom'].values
    dsm = pd.read_csv('Daten_Minibsp/dsm_store.csv')
    dsm_store = pd.DataFrame(index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    dsm_store['e_max_pu']=dsm['e_max_pu'].values
    dsm_store['e_min_pu']=dsm['e_min_pu'].values
    dsm_store['e_nom']=dsm['e_nom'].values
    
    # DLR
    
    line12 = pd.Series(1, index=pd.date_range(
    "2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    line23 = pd.read_csv('Daten_Minibsp/line23.csv')
    line23 = pd.Series(line23['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    line31 = pd.read_csv('Daten_Minibsp/line31.csv')
    line31 = pd.Series(line31['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    
    ### Heat Sector
    
    # solarthermal generator
    
    solarthermal = pd.read_csv('Daten_Minibsp/gen_thermal.csv')
    solarthermal = pd.Series(solarthermal['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    
    # heat load
    
    central_heat_load = pd.read_csv('Daten_Minibsp/central_heat_load.csv')
    central_heat_load = pd.Series(central_heat_load['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    rural_heat_load = pd.read_csv('Daten_Minibsp/rural_heat_load.csv')
    rural_heat_load = pd.Series(rural_heat_load['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    
    ### Gas Sector
    
    # gas loads
    
    h2_load = pd.read_csv('Daten_Minibsp/h2_load.csv')
    h2_load = pd.Series(h2_load['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    ch4_load = pd.read_csv('Daten_Minibsp/ch4_load.csv')
    ch4_load = pd.Series(ch4_load['0'].values, index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H"))
    
    return wind, pv, el_load, dsm_link, dsm_store, line12, line23, line31, solarthermal, \
            central_heat_load, rural_heat_load, h2_load, ch4_load

##############################################################################

def import_network(wind, pv, el_load, dsm_link, dsm_store, line12, \
    line23, line31, solarthermal, central_heat_load, rural_heat_load, h2_load, ch4_load):
    
    network = pypsa.Network() 
    
    network.set_snapshots(pd.date_range(
    "2011-01-01 00:00","2011-12-31 23:00",freq="H"))

    ########################### Electricity Sector ###########################
    
    ###### Buses
    
    network.add(
        "Bus",
        name = 'AC 1',
        carrier = "AC",
        v_nom = 380., 
        x = 10.6414678044968,
        y = 53.9191626912878)

    network.add(
        "Bus",
        name = 'AC 2',
        carrier = "AC",
        v_nom = 380.,
        x = 10.7626283,
        y = 53.9116822)

    network.add(
        "Bus",
        name = 'AC 3',
        carrier = "AC",
        v_nom = 380.,
        x = 10.7395055967076,
        y = 54.0311768296654)
    
    ###### Lines with DLR 

    network.add(
        "Line",
        name = "Line12",
        bus0 = "AC 1",
        bus1 = "AC 2",
        s_nom_extendable = True,
        s_nom_min = 20,
        s_nom = 20, 
        length = 5, 
        capital_cost = 85*5, # Quelle: open_eGo-Abschlussbericht
        s_max_pu = line12,
        x = 0.3*5) # Quelle: etwa Werte in Datenbank bzw. Vorschläge bei pypsa-Komponenten

    network.add(
        "Line",
        name = "Line23",
        bus0 = "AC 2",
        bus1 = "AC 3",
        s_nom_extendable = True,
        s_nom_min = 20, 
        s_nom = 20,
        length = 5,
        capital_cost = 85*5,
        s_max_pu = line23,
        x = 0.3*5)

    network.add(
        "Line",
        name = "Line31",
        bus0 = "AC 3",
        bus1 = "AC 1",
        s_nom_extendable = True,
        s_nom_min = 20,
        s_nom = 20, 
        length = 5,
        capital_cost = 85*5,
        s_max_pu = line31,
        x = 0.3*5)
    
    ###### Generation: Wind Onshore Turbine
    
    network.add(
        "Generator",
        name = "Gen Wind",
        carrier = "wind",
        bus = "AC 3",
        p_nom = 150,
        p_max_pu = wind,
        marginal_cost = 0,
        control = 'PV') 
    
    network.add(
        "Generator",
        name = "Gen PV",
        carrier = "pv",
        bus = "AC 2",
        p_nom = 150,
        p_max_pu = pv,
        marginal_cost = 0,
        control = 'PV') 
    
    ###### Load (with DSM-potential)

    network.add(
        "Load",
        name = "AC Load",
        bus = "AC 1",
        p_set = el_load)
    network.loads['carrier'] = 'AC'

    ###### Storage Unit: Battery

    network.add(
        "StorageUnit",
        name = "Battery Storage",
        carrier = "extendable_battery_storage",
        bus = "AC 2",
        capital_cost = 19600, # Quelle: pypsa-Tabelle (118000 EUR/MWh / 6h)
        p_nom_extendable = True,
        max_hours = 6,
        cyclic_state_of_charge = True,
        standing_loss = 0.007, 
        efficiency_store = 0.93, 
        efficiency_dispatch = 0.93)

    ###### DSM
    
    network.add(
        "Bus",
        name = 'DSM-Bus',
        carrier = "DSM",
        v_nom = 380.,
        x = 10.58,
        y = 53.9191626912878)
    
    network.add(
        "Link",
        name = 'DSM-Link',
        bus0 = "AC 1",
        bus1 = "DSM-Bus",
        p_nom = dsm_link['p_nom'].iloc[0],
        p_min_pu = dsm_link['p_min_pu'],
        p_max_pu = dsm_link['p_max_pu'])

    network.add(
        "Store",
        name = "DSM-Store",
        bus = "DSM-Bus",
        e_cyclic=True,
        e_nom = dsm_store['e_nom'].iloc[0],
        e_min_pu = dsm_store['e_min_pu'],
        e_max_pu = dsm_store['e_max_pu'])
        
    ############################### Gas Sector ###############################
    
    ###### Buses 
    
    network.add(
        "Bus",
        name = "CH4 1",
        carrier = "CH4",
        x = 10.614455,
        y = 53.989616)

    network.add(
        "Bus",
        name = "CH4 2",
        carrier = "CH4",
        x = 10.475106,
        y = 53.78447)

    network.add(
        "Bus",
        name = "H2_Grid",
        carrier = "H2",
        x = 10.6100082297444,
        y = 54.0102799788641)

    network.add(
        "Bus",
        name = "H2_Saltcavern",
        carrier = "H2",
        x = 10.5,
        y = 53.96)        
    
    ###### H2- and CH4- conversions / connections

    network.add(
        "Link",
        name = 'H2 Feedin',  
        bus0 = "H2_Grid",
        bus1 = "CH4 1", 
        p_nom = 10) 
    
    network.add(
        "Link",
        name = 'Methanation',
        bus0 = "H2_Grid",
        bus1 = "CH4 1",
        p_nom_extendable = True, 
        capital_cost = 252000, # Quelle: pypsa-Tabelle
        efficiency = 0.8) # Quelle: pypsa-Tabelle

    network.add(
        "Link",
        name = 'SMR',
        bus0 = "CH4 1",
        bus1 = "H2_Grid",
        p_nom_extendable = True,
        capital_cost = 540000, # Quelle: pypsa-Tabelle
        efficiency = 0.74) # Quelle: pypsa-Tabelle
    
    network.add(
        "Link",
        name = 'Gas Pipeline',
        bus0 = "CH4 1",
        bus1 = "CH4 2",
        p_nom = 100, 
        p_min_pu =-1)

    ###### Gas Generation: Biogas
        
    network.add(
        "Generator",
        name = "Gen Biogas",
        carrier = "biogas",
        bus = "CH4 2",
        p_nom = 300, 
        marginal_cost = 65 # Quelle: pypsa-Tabelle (und weitere Recherchen)
        )
    
    ###### Connections to Electricity Sector

    network.add(
        "Link",
        name = 'Power2H2 1',
        bus0 = "AC 3",
        bus1 = "H2_Grid",
        p_nom_extendable = True,
        capital_cost = 375000, # Quelle: pypsa-Tabelle
        efficiency = 0.7) # Quelle: pypsa-Tabelle

    network.add(
        "Link",
        name = 'Fuel Cell 1',
        bus0 = "H2_Grid",
        bus1 = "AC 3",
        p_nom_extendable = True,
        capital_cost = 1025000, # Quelle: pypsa-Tabelle
        efficiency = 0.5) # Quelle: pypsa-Tabelle
    
    network.add(
        "Link",
        name = 'Gas Turbine',
        bus0 = "CH4 1",
        bus1 = "AC 3",
        p_nom = 100, 
        p_min_pu = 0,
        marginal_cost = 4.5, # Quelle: pypsa-Tabelle
        efficiency = 0.42) # Quelle: pypsa-Tabelle
    
    network.add(
        "Link",
        name = 'Power2H2 2',
        bus0 = "AC 1",
        bus1 = "H2_Saltcavern",
        p_nom_extendable = True,
        capital_cost = 375000,  # Quelle: pypsa-Tabelle
        efficiency = 0.7) # Quelle: pypsa-Tabelle
    
    network.add(
        "Link",
        name = 'Fuel Cell 2',
        bus0 = "H2_Saltcavern",
        bus1 = "AC 1",
        p_nom_extendable = True, 
        capital_cost = 1025000, # Quelle: pypsa-Tabelle
        efficiency = 0.5) # Quelle: pypsa-Tabelle
    
    ##### CHP (elektrisch)
            
    network.add("Link",
            "CHP_el",
            bus0="CH4 2",
            bus1="AC 2",
            carrier="CHP_el",
            p_nom = 100, 
            efficiency = 0.6) 
    
    ###### Gas Loads
    
    network.add(
        "Load",
        name = "CH4 Load",
        bus = "CH4 1",
        p_set = ch4_load)
    network.loads['carrier'].loc['CH4 Load'] = 'gas'
    
    network.add(
        "Load",
        name = "H2 Load",
        bus = "H2_Grid",
        p_set = h2_load)
    network.loads['carrier'].loc['CH2 Load'] = 'gas'

    ###### H2 Store
    
    # H2 Steel Tank
    
    network.add(
        "Bus",
        name = "H2 steel tank",
        carrier = "H2",
        x = 10.6100082297444,
        y = 54.0102799788641)

    network.add(
        "Store",
        name = "H2 Steel Tank",
        bus = "H2 steel tank",
        e_cyclic=True,
        e_nom_extendable = True,
        capital_cost = 750) # Quelle: pypsa-Tabelle

    network.add(
        "Link",
        name = "H2 charger 1",
        bus0 = "H2_Grid",
        bus1 = "H2 steel tank",
        p_nom_extendable = True)

    network.add(
        "Link",
        name = "H2 discharger 1",
        bus0 = "H2 steel tank",
        bus1 = "H2_Grid",
        p_nom_extendable = True)
    
    # H2 Saltcavern
    
    network.add(
        "Bus",
        name = "H2 saltcavern",
        carrier = "H2",
        x = 10.5,
        y = 53.96) 
    
    network.add(
        "Store",
        name = "H2 Saltcavern",
        bus = "H2 saltcavern",
        e_cyclic=True,
        e_nom_extendable = True, 
        e_nom_max = 240000, # Quelle: egon-data
        capital_cost = 8400) # Quelle: egon-data

    network.add(
        "Link",
        name = "H2 charger 2",
        bus0 = "H2_Saltcavern",
        bus1 = "H2 saltcavern",
        p_nom_extendable = True)

    network.add(
        "Link",
        name = "H2 discharger 2",
        bus0 = "H2 saltcavern",
        bus1 = "H2_Saltcavern",
        p_nom_extendable = True)
    
    # CH4 Grid Capacity
    
    network.add(
        "Bus",
        name = "gas grid",
        carrier = "gas",
        x = 10.475106,
        y = 53.784470)
    
    network.add(
        "Store",
        name = "Gas Grid",
        bus = "gas grid",
        e_cyclic=True,
        e_nom = 206) # Quelle: egon-data

    network.add(
        "Link",
        name = "CH4 charger 1",
        bus0 = "CH4 2",
        bus1 = "gas grid",
        p_nom_extendable = True)

    network.add(
        "Link",
        name = "CH4 discharger 1",
        bus0 = "gas grid",
        bus1 = "CH4 2",
        p_nom_extendable = True)

    ############################## Heat Sector ###############################
    
    ###### Heat Bus
    
    network.add(
        "Bus",
        name = "Central Heat",
        carrier = "heat",
        x = 10.7320610006821,
        y = 53.8850833996917)
    
    network.add(
        "Bus",
        name = "Rural Heat",
        carrier = "heat",
        x = 10.679828485803428,
        y = 53.83210392503918)
        
    ###### CHP (Wärme)
                
    network.add("Link",
            "CHP_heat",
            bus0="CH4 2",
            bus1="Central Heat",
            carrier="CHP_heat",
            p_nom = 100) 
    
    ###### Heat Load

    network.add(
        "Load",
        name = "Central Heat Load",
        bus = "Central Heat",
        p_set = central_heat_load)
    network.loads['carrier'].loc['Central Heat Load'] = 'heat'
    
    network.add(
        "Load",
        name = "Rural Heat Load",
        bus = "Rural Heat",
        p_set = rural_heat_load)
    network.loads['carrier'].loc['Rural Heat Load'] = 'heat'
    
    ###### Heat Generation: solarthermal

    network.add(
        "Generator",
        name = "Gen Solarthermal",
        carrier = "solarthermal",
        bus = "Central Heat",
        p_nom = 300, 
        p_max_pu = solarthermal, 
        marginal_cost = 0)
    
    ###### Power to Heat: Wärmepumpe

    network.add(
        "Link",
        name = "Heat Pump 1",
        bus0 = "AC 2",
        bus1 = "Central Heat",
        p_nom = 100, 
        efficiency = 3.6) # Quelle: pypsa-Tabelle
    
    network.add(
        "Link",
        name = "Heat Pump 2",
        bus0 = "AC 2",
        bus1 = "Rural Heat",
        p_nom = 100,
        efficiency = 3.6) # Quelle: pypsa-Tabelle
    
    ###### Gas to Heat: Gas Boiler 
    
    network.add(
        "Link",
        name = "Gas Boiler 1",
        bus0 = "CH4 2",
        bus1 = "Rural Heat",
        p_nom = 100, 
        efficiency = 1.04) # Quelle: pypsa-Tabelle
    
    network.add(
        "Link",
        name = "Gas Boiler 2",
        bus0 = "CH4 2",
        bus1 = "Central Heat",
        p_nom = 100, 
        efficiency = 1.04) # Quelle: pypsa-Tabelle

    ###### Heat Stores
    
    # central heat bus
    
    network.add(
        "Bus",
        name = "central heat store",
        carrier = "heat",
        x = 10.732061000682,
        y = 53.8850833996917)

    network.add(
        "Store",
        name = "Central Heat Store",
        bus = "central heat store",
        e_cyclic=True,
        e_nom_extendable = True,
        capital_cost = 520)

    network.add(
        "Link",
        name = "heat store charger 1",
        bus0 = "Central Heat",
        bus1 = "central heat store",
        p_nom_extendable = True,
        efficiency = 0.84) 

    network.add(
        "Link",
        name = "heat store discharger 1",
        bus0 = "central heat store",
        bus1 = "Central Heat",
        p_nom_extendable = True,
        efficiency = 0.84) 
    
    # rural heat bus
    
    network.add(
        "Bus",
        name = "rural heat store",
        carrier = "heat",
        x = 10.679828485803428,
        y = 53.83210392503918)
    
    network.add(
        "Store",
        name = "Rural Heat Store",
        bus = "rural heat store",
        e_cyclic=True,
        e_nom_extendable = True,
        capital_cost = 520) 

    network.add(
        "Link",
        name = "heat store charger 2",
        bus0 = "Rural Heat",
        bus1 = "rural heat store",
        p_nom_extendable = True,
        efficiency = 0.84) 

    network.add(
        "Link",
        name = "heat store discharger 2",
        bus0 = "rural heat store",
        bus1 = "Rural Heat",
        p_nom_extendable = True,
        efficiency = 0.84) 
    
    return network

args = {
    # Setup and Configuration:
    'db': 'sh-egon-data',  # database session
    'gridversion': None,  # None for model_draft or Version number
    'method': { # Choose method and settings for optimization
        'type': 'lopf', # type of optimization, currently only 'lopf'
        'n_iter': 5, # abort criterion of iterative optimization, 'n_iter' or 'threshold' 
        'pyomo': False}, # set if pyomo is used for model building
    'pf_post_lopf': {
        'active': False, # choose if perform a pf after a lopf simulation
        'add_foreign_lopf': True, # keep results of lopf for foreign DC-links
        'q_allocation': 'p_nom'}, # allocate reactive power via 'p_nom' or 'p'
    'start_snapshot': 1,
    'end_snapshot': 8760,
    'solver': 'gurobi',  # glpk, cplex or gurobi
    'solver_options': {}, # {} for default options, specific for solver
    'model_formulation': 'kirchhoff', # angles or kirchhoff
    'scn_name': 'eGon2035',  # a scenario: eGon2035 or eGon100RE
    # Scenario variations:
    'scn_extension': None,  # None or array of extension scenarios
    'scn_decommissioning': None,  # None or decommissioning scenario
    # Export options:
    'lpfile': False,  # save pyomo's lp file: False or /path/tofolder
    'csv_export': 'results',  # save results as csv: False or /path/tofolder
    # Settings:
    'extendable': ['network', 'storage'],  # Array of components to optimize 
    'generator_noise': 789456,  # apply generator noise, False or seed number
    'extra_functionality':{},  # Choose function name or {}
    # Clustering:
    'network_clustering_kmeans': {
        'active': False, # choose if clustering is activated
        'n_clusters': 2, # number of resulting nodes
        'kmeans_busmap': False, # False or path/to/busmap.csv
        'line_length_factor': 1, #
        'remove_stubs': False, # remove stubs before kmeans clustering
        'use_reduced_coordinates': False, #
        'bus_weight_tocsv': None, # None or path/to/bus_weight.csv
        'bus_weight_fromcsv': None, # None or path/to/bus_weight.csv
        'n_init': 10, # affects clustering algorithm, only change when neccesary
        'max_iter': 100, # affects clustering algorithm, only change when neccesary
        'tol': 1e-6, # affects clustering algorithm, only change when neccesary
        'n_jobs': -1}, # affects clustering algorithm, only change when neccesary
    'network_clustering_ehv': False,  # clustering of HV buses to EHV buses.
    'disaggregation': 'uniform',  # None, 'mini' or 'uniform'
    'snapshot_clustering': { 
        'active': False, # choose if clustering is activated
        'method':'typical_periods', # 'typical_periods' or 'segmentation'
        'extreme_periods': 'None', # optional adding of extreme period
        'how': 'daily', # type of period - only relevant for 'typical_periods'
        'storage_constraints': 'soc_constraints', # additional constraints for storages  - only relevant for 'typical_periods'
        'n_clusters': 5, #  number of periods - only relevant for 'typical_periods'
        'n_segments': 5}, # number of segments - only relevant for segmentation
    # Simplifications:
    'skip_snapshots': False, # False or number of snapshots to skip
    'branch_capacity_factor': {'HV': 0.5, 'eHV': 0.7},  # p.u. branch derating
    'load_shedding': False,  # meet the demand at value of loss load cost
    'foreign_lines': {'carrier': 'AC', # 'DC' for modeling foreign lines as links
                      'capacity': 'osmTGmod'}, # 'osmTGmod', 'ntc_acer' or 'thermal_acer'
    'comments': None}

def optimize_etrago(args, number, path):
    
    # to procede in loop
    if path == 'skip_snapshots':
        args['skip_snapshots'] = number
    elif path == 'typical_days' or path == 'typical_weeks' or path == 'typical_months' or path == 'segmentation':
        args['snapshot_clustering']['active'] = True
        if args['snapshot_clustering']['method'] == 'typical_periods':
            args['snapshot_clustering']['n_clusters'] = number
        elif args['snapshot_clustering']['method'] == 'segmentation':
            args['snapshot_clustering']['n_segments'] = number
        
    path = path +'/'+ str(number) +'/'
        
    wind, pv, el_load, dsm_link, dsm_store, line12, line23, line31, solarthermal, \
        central_heat_load, rural_heat_load, h2_load, ch4_load = import_data()
         
    etrago = Etrago(args, json_path=None)
    etrago.args['extendable'] = ['network', 'storage']
    
    etrago.network = import_network(wind, pv, el_load, dsm_link, dsm_store, line12, \
                            line23, line31, solarthermal, central_heat_load, rural_heat_load, h2_load, ch4_load)
    etrago.network.lines['v_nom'] = 380 
    
    # adjust network, e.g. annualize costs  
    etrago.load_shedding()
    etrago.convert_capital_costs()
    etrago.set_random_noise(0.01)

    # save original timeseries
    original_snapshots = etrago.network.snapshots
    original_weighting = etrago.network.snapshot_weightings

    # save format for dispatch using original timeseries
    gen_p = etrago.network.generators_t.p.copy()
    lines_lower = etrago.network.lines_t.mu_lower.copy()
    lines_upper = etrago.network.lines_t.mu_upper.copy()
    lines_p0 = etrago.network.lines_t.p0.copy()
    lines_p1 = etrago.network.lines_t.p1.copy()
    links_lower = etrago.network.links_t.mu_lower.copy()
    links_upper = etrago.network.links_t.mu_upper.copy()
    links_p0 = etrago.network.links_t.p0.copy()
    links_p1 = etrago.network.links_t.p1.copy()
    stun_p = etrago.network.storage_units_t.p.copy()
    stun_p_dispatch = etrago.network.storage_units_t.p_dispatch.copy()
    stun_p_store = etrago.network.storage_units_t.p_store.copy()
    stun_state_of_charge = etrago.network.storage_units_t.state_of_charge.copy()
    store_e = etrago.network.stores_t.e.copy()
    store_p = etrago.network.stores_t.p.copy()
    bus_price = etrago.network.buses_t.marginal_price.copy()
    bus_p = etrago.network.buses_t.p.copy()
    bus_vang = etrago.network.buses_t.v_ang.copy()
    loads = etrago.network.loads_t.p.copy()
    
    print(' ')
    print('Start Snapshot Clustering')
    t1 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    # skip snapshots
    etrago.skip_snapshots()
    
    # snapshot clustering
    etrago.snapshot_clustering()
    
    print(' ')
    print('Stop Snapshot Clustering')
    t2 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    print(' ')
    print('Start LOPF Level 1')
    t3 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    # LOPF 1
    
    if args['skip_snapshots'] != False:
        print(' ')
        print(etrago.network.snapshots)
        print(' ')
        
    etrago.args['csv_export'] = path+'/results/level1/'
    etrago.args['lpfile'] = path+'/lp-level1.lp'
    
    # optimization of network- and storage expansion with aggregated timeseries
    # start linear optimal powerflow calculations
    etrago.lopf()
    
    print(' ')
    print('Stop LOPF Level 1')
    t4 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    # save results of this optimization
    etrago.export_to_csv(path+'Level-1')
    
    # calculate central etrago results
    etrago.calc_results()
    print(' ')
    print(etrago.results)
    etrago.results.to_csv(path+'results-1')
    print(' ') 
    
    #####
    # LOPF 2
        
    # drop dispatch from LOPF1
    
    etrago.network.generators_t.p = gen_p
    etrago.network.lines_t.mu_lower = lines_lower
    etrago.network.lines_t.mu_upper = lines_upper
    etrago.network.lines_t.p0 = lines_p0
    etrago.network.lines_t.p1 = lines_p1
    etrago.network.links_t.mu_lower = links_lower
    etrago.network.links_t.mu_upper = links_upper
    etrago.network.links_t.p0 = links_p0
    etrago.network.links_t.p1 = links_p1
    etrago.network.storage_units_t.p = stun_p
    etrago.network.storage_units_t.p_dispatch = stun_p_dispatch
    etrago.network.storage_units_t.p_store = stun_p_store
    etrago.network.storage_units_t.state_of_charge = stun_state_of_charge
    etrago.network.storage_units_t.soc_intra = etrago.network.storage_units_t.state_of_charge.copy()
    etrago.network.stores_t.e = store_e
    etrago.network.stores_t.p = store_p
    etrago.network.stores_t.soc_intra_store = etrago.network.stores_t.e.copy()
    etrago.network.buses_t.marginal_price = bus_price
    etrago.network.buses_t.p = bus_p
    etrago.network.buses_t.v_ang = bus_vang
    etrago.network.loads_t.p = loads
    
    # use network and storage expansion from LOPF 1
    
    etrago.network.lines['s_nom'] = etrago.network.lines['s_nom_opt']
    etrago.network.lines['s_nom_extendable'] = False
    
    etrago.network.storage_units['p_nom'] = etrago.network.storage_units['p_nom_opt']
    etrago.network.storage_units['p_nom_extendable'] = False
    
    etrago.network.stores['e_nom'] = etrago.network.stores['e_nom_opt']
    etrago.network.stores['e_nom_extendable'] = False
    
    etrago.network.links['p_nom'] = etrago.network.links['p_nom_opt']
    etrago.network.links['p_nom_extendable'] = False
    
    etrago.args['extendable'] = []
    etrago.args['load_shedding'] = False
    
    # use original timeseries
    
    etrago.network.snapshots = original_snapshots
    etrago.network.snapshot_weightings = original_weighting
    
    # adapt settings for LOPF 2
    
    etrago.args['snapshot_clustering']['active']=False
    etrago.args['skip_snapshots']=False
    
    etrago.args['csv_export'] = path+'/results/level2/'
    etrao.args['lpfile'] = path+'/lp-level2.lp'
    
    print(' ')
    print('Start LOPF Level 2')
    t5 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    # optimization of dispatch with complex timeseries
    etrago.lopf()
    
    print(' ')
    print('Stop LOPF Level 2')
    t6 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    # save results of this optimization
    etrago.export_to_csv(path+'Level-2')
    
    # calculate central etrago results
    etrago.calc_results()
    print(' ')
    print(etrago.results)
    etrago.results.to_csv(path+'results-2')
    print(' ')
    
    t = pd.Series([t1, t2, t3, t4, t5, t6]) 
    t.to_csv(path+'time')
    
    return etrago

##############################################################################

# für Minimalbeispiel Änderungen in 

# load_shedding()
# convert_capital_costs()
# set_random_noise()
# chp_constraints
# calc_results()

args['load_shedding'] = False
args['method']['pyomo'] = True
args['method']['n_iter'] = 5 # bei 5 nur noch Änderungen im 100 Euro-Bereich, siehe auch Abschlussbericht
args['solver_options'] = {}

args['snapshot_clustering']['active'] = True # True, False
args['snapshot_clustering']['method'] = 'typical_periods' # 'typical_periods', 'segmentation'
args['snapshot_clustering']['extreme_periods'] = 'None' # 'None', 'append', 'replace_cluster_center'
args['snapshot_clustering']['how'] = 'daily' # 'daily', 'weekly', 'monthly'
args['snapshot_clustering']['storage_constraints'] = 'soc_constraints' # 'soc_constraints', ''
args['skip_snapshots'] = False # True, False

##############################################################################

skip_snapshots = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 150] 

typical_days = [365, 350, 300, 250, 200, 150, 120, 100, 60, 40, 20, 10, 5]

typical_weeks = [52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 1]

typical_months = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

segmentation = [8760, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 500, 200, 100]

if args['snapshot_clustering']['active'] == True and args['skip_snapshots'] == True:
    raise ValueError("Decide for temporal aggregation method!")
elif args['snapshot_clustering']['active'] == False and args['skip_snapshots'] == False:
    loop = [0]
    path = 'original'
if args['snapshot_clustering']['active'] == True:
    if args['snapshot_clustering']['method'] == 'typical_periods':
        if args['snapshot_clustering']['how'] == 'daily':
            loop = typical_days
            path = 'typical_days'
        elif args['snapshot_clustering']['how'] == 'weekly':
            loop = typical_weeks
            path = 'typical_weeks'
        elif args['snapshot_clustering']['how'] == 'monthly':
            loop = typical_months
            path = 'typical_months'
    elif args['snapshot_clustering']['method'] == 'segmentation':
        loop = segmentation
        path = 'segmentation'
elif args['skip_snapshots'] == True:
    loop = skip_snapshots
    path = 'skip_snapshots'

for no in loop:

    old_stdout = sys.stdout
    path_log = path +'/'+ str(no)
    os.makedirs(path_log, exist_ok=True)
    log_file = open(path_log+'/console.log',"w")
    sys.stdout = log_file
    
    print(' ')
    print('pyomo:')
    print(args['method'])
    print('snapshot_clustering:')
    print(args['snapshot_clustering'])
    print(args['snapshot_clustering']['method'])
    print(args['snapshot_clustering']['how'])
    print(args['snapshot_clustering']['extreme_periods'])
    print('skip_snapshots:')
    print(args['skip_snapshots'])
    print(' ') 
    
    print(' ')
    print('Calculation using: '+ path)
    print('with number = '+ str(no))
    print(' ')
    
    etrago = optimize_etrago(args=args, number = no, path = path)
    
    sys.stdout = old_stdout
    log_file.close()








     

