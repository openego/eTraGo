#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:26:19 2022

@author: student
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from etrago import Etrago
from pypsa.descriptors import get_switchable_as_dense as as_dense
import geopandas

def import_data():

    # Electricity Sector

    # wind generator
    gen = pd.read_csv('gen.csv')
    gen = gen.set_index('Generator')
    gen_p = pd.read_csv('gen_p.csv')
    gen_p = gen_p.set_index('snapshot')
    
    wind = pd.read_csv('gen_wind_p.csv')
    # wind = pd.Series(wind['0'].values, index=pd.date_range("2021-06-10 00:00","2021-12-31 23:00",freq="H"))
    wind = wind.set_index('Generator')
    wind = wind.squeeze().values

    konv = pd.read_csv('gen_turb_p.csv')
    # konv = pd.Series(konv['0'].values, index=pd.date_range("2021-06-10 00:00","2021-12-31 23:00",freq="H"))
    konv = konv.set_index('Generator')
    konv = konv.squeeze().values

    # electrical load

    el_load = pd.read_csv('load_p.csv')
    # el_load = pd.Series(el_load['0'].values, index=pd.date_range("2021-06-10 00:00","2021-12-31 23:00",freq="H"))
    el_load = el_load.set_index('Load')
    el_load = el_load.squeeze().values
    
    price_m = pd.read_csv('price_m.csv')
    price_m = price_m.set_index('snapshot')
    return gen, wind, konv, el_load, price_m, gen_p


def import_network(gen, wind, konv, el_load, price_m, gen_p):

    network = pypsa.Network()
    network.set_snapshots(pd.date_range(
        start='10/06/2021-00:00', freq='H', periods=2))

### Electrical Sector###

    network.add(
        'Bus',
        name='AC 1',
        carrier='AC',
        v_nm=380,
        x=10.6414678044968,
        y=53.9191626912878)

    network.add(
        'Bus',
        name='AC 2',
        carrier="AC",
        v_nom=380.,
        x=10.7626283,
        y=53.9116822)

    network.add(
        'Bus',
        name='AC 3',
        carrier="AC",
        v_nom=380.,
        x=10.7395055967076,
        y=54.0311768296654)

    network.add(
        'Line',
        name='Line12',
        bus0='AC 1',
        bus1='AC 2',
        s_nom_min = 20,
        s_nom_max=30,
        s_nom_extendable=True,
        length=5,
        capital_cost=500,
        x=0.1)

    network.add(
        "Line",
        name="Line23",
        bus0="AC 2",
        bus1="AC 3",
        s_nom_min = 20,
        s_nom_max=150,
        s_nom_extendable=True,
        length=5,
        capital_cost=500,
        x=0.1)

    network.add(
        "Line",
        name="Line31",
        bus0="AC 3",
        bus1="AC 1",
        s_nom_min = 20,
        s_nom_max=150,
        s_nom_extendable=True,
        length=5,
        capital_cost=500,
        x=0.1)

    network.add(
        "Generator",
        name="wind turbine",
        carrier="wind",
        bus="AC 1",
        p_nom=100,
        p_min_pu = wind/gen.loc['wind turbine','p_nom'],
        p_max_pu = wind/gen.loc['wind turbine','p_nom'],
        control='PV',
        p_set= wind)

    network.add(
        "Generator",
        name="gas turbine",
        carrier="gas",
        bus="AC 3",
        p_nom= 300,
        control='PV',
        p_set= konv,
        p_min_pu = konv/gen.loc['gas turbine','p_nom'],
        p_max_pu = konv/gen.loc['gas turbine','p_nom'],
        marginal_cost=450
        )

    network.add(
        "Load",
        name="AC load 1",
        carrier="AC",
        bus="AC 2",
        p_set = [100, 125])
    
    network.buses['country'] = 'DE'
    

    return network



args = {
    # Setup and Configuration:
    'db': 'etrago-data',  # database session
    'gridversion': None,  # None for model_draft or Version number
    'method': {  # Choose method and settings for optimization
        'type': 'lopf',  # type of optimization, currently only 'lopf'
        'n_iter': 4,  # abort criterion of iterative optimization, 'n_iter' or 'threshold'
        'pyomo': True},  # set if pyomo is used for model building
    'pf_post_lopf': {
        'active': False,  # choose if perform a pf after a lopf simulation
        'add_foreign_lopf': True,  # keep results of lopf for foreign DC-links
        'q_allocation': 'p_nom'},  # allocate reactive power via 'p_nom' or 'p'
    'start_snapshot': 1,
    'end_snapshot': 3,
    'solver': 'gurobi',  # glpk, cplex or gurobi
    'solver_options': {},
    'model_formulation': 'kirchhoff',  # angles or kirchhoff
    'scn_name': 'eGon2035',  # a scenario: eGon2035 or eGon100RE
    # Scenario variations:
    'scn_extension': None,  # None or array of extension scenarios
    'scn_decommissioning': None,  # None or decommissioning scenario
    # Export options:
    'lpfile': False,  # save pyomo's lp file: False or /path/tofolder
    'csv_export': 'results',  # save results as csv: False or /path/tofolder
    # Settings:
    'extendable': [],  # Array of components to optimize
    'generator_noise': 789456,  # apply generator noise, False or seed number
    'extra_functionality': {},  # Choose function name or {}
    # Clustering:
    'network_clustering_kmeans': {
        'active': False,  # choose if clustering is activated
        'n_clusters': 10,  # number of resulting nodes
        'kmeans_busmap': False,  # False or path/to/busmap.csv
        'line_length_factor': 1,
        'remove_stubs': False,  # remove stubs bevore kmeans clustering
        'use_reduced_coordinates': False,
        'bus_weight_tocsv': None,  # None or path/to/bus_weight.csv
        'bus_weight_fromcsv': None,  # None or path/to/bus_weight.csv
        'n_init': 10,  # affects clustering algorithm, only change when neccesary
        'max_iter': 100,  # affects clustering algorithm, only change when neccesary
        'tol': 1e-6,  # affects clustering algorithm, only change when neccesary
        'n_jobs': -1},  # affects clustering algorithm, only change when neccesary
    'network_clustering_ehv': False,  # clustering of HV buses to EHV buses.
    'disaggregation': None,  # None, 'mini' or 'uniform'
    'snapshot_clustering': {
        'active': False,  # choose if clustering is activated
        'n_clusters': 2,  # number of periods
        'how': 'daily',  # type of period, currently only 'daily'
        'storage_constraints': 'soc_constraints'},  # additional constraints for storages
    # Simplifications:
    'skip_snapshots': False,  # False or number of snapshots to skip
    'branch_capacity_factor': {'HV': 1, 'eHV': 1},  # p.u. branch derating
    'load_shedding': False,  # meet the demand at value of loss load cost
    'foreign_lines': {'carrier': 'DC',  # 'DC' for modeling foreign lines as links
                      'capacity': 'osmTGmod'},  # 'osmTGmod', 'ntc_acer' or 'thermal_acer'
    'comments': None}

  
etrago = Etrago(args, json_path=None)
gen, wind, konv, el_load, price_m, gen_p = import_data()
etrago.network = import_network(gen, wind, konv, el_load, price_m, gen_p)



g_up = etrago.network.generators.copy()
g_down = etrago.network.generators.copy()

g_up.index = g_up.index.map(lambda x: x + " ramp up")
g_down.index = g_down.index.map(lambda x: x + " ramp down")

up = ( etrago.network.generators.p_nom - gen_p
).clip(0) / etrago.network.generators.p_nom
up.index=up.index.astype('datetime64[ns]')
#import pdb; pdb.set_trace()
down = -(gen_p) / etrago.network.generators['p_nom']
down.index=up.index.astype('datetime64[ns]')

up.columns = up.columns.map(lambda x: x + " ramp up")
down.columns = down.columns.map(lambda x: x + " ramp down")

#etrago.network.generators_t.p_max_pu.index = etrago.network.generators_t.p_max_pu.index.astype(object)


etrago.network.madd("Generator", g_up.index, p_max_pu=up, **g_up.drop(["p_max_pu"], axis=1))
#import pdb; pdb.set_trace()
#etrago.network.generators_t.p_max_pu.merge(up, left_on='snapshot', right_on='snapshot')


etrago.network.madd("Generator", g_down.index, p_min_pu=down, p_max_pu=0, **g_down.drop(["p_max_pu", "p_min_pu"], axis=1))


etrago.network.generators.loc[etrago.network.generators.index.str.contains("ramp up"), "marginal_cost"] *= 1
etrago.network.generators.loc[etrago.network.generators.index.str.contains("ramp down"), "marginal_cost"] *= -1
# Haltepunkt: 

    
etrago.network.lopf(solver_name='gurobi')

etrago.calc_results()


gen = etrago.network.generators
gen_p = etrago.network.generators_t.p
gen_p_nom = gen['p_nom']
gen_p_max_pu= etrago.network.generators_t.p_max_pu


gen_wind = gen.filter(regex='wind turbine', axis=0)
gen_wind_p = gen_p.filter(regex='wind turbine', axis=1)
gen_wind_p = gen_wind_p.T

gen_turb = gen.filter(regex='gas turbine', axis=0)
gen_turb_p = gen_p.filter(regex='gas turbine', axis=1)
gen_turb_p = gen_turb_p.T

load = etrago.network.loads
load_p = etrago.network.loads_t.p
load_p = load_p.T

line = etrago.network.lines
m= etrago.network
price = price_m.values.sum() + etrago.network.objective

gen_p.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_2/gen_p.csv' )
gen_p_nom.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_2/gen_p_nom.csv' )
gen_p_max_pu.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_2/gen_p_max_pu.csv' )

fig, axs = plt.subplots(
    1, 3, figsize=(20, 10), subplot_kw={"projection": ccrs.AlbersEqualArea()}
    )

market = (
    etrago.network.generators_t.p[m.generators.index]
    .T.squeeze()
    .groupby(etrago.network.generators.bus)
    .sum()
    .div(2e6)

)
market= market.T
etrago.network.plot(ax=axs[0],bus_sizes= market.iloc[0], title="Market simulation", geomap=True)

up
redispatch_up = (
    etrago.network.generators_t.p.filter(like="ramp up")
    .T.squeeze()
    .groupby(etrago.network.generators.bus)
    .sum()
    .div(2e6)
)
redispatch_up = redispatch_up.T

etrago.network.plot(
    ax=axs[1], bus_sizes=redispatch_up.iloc[0], bus_colors="blue", title="Redispatch: ramp up",geomap=True
)

#down
redispatch_down = (
    etrago.network.generators_t.p.filter(like="ramp down")
    .T.squeeze()
    .groupby(etrago.network.generators.bus)
    .sum()
    .div(-2e6)
)
redispatch_down = redispatch_down.T
etrago.network.plot(
    ax=axs[2], bus_sizes=redispatch_down.iloc[0], bus_colors="red",title="Redispatch: ramp down / curtail", geomap=True,  
);