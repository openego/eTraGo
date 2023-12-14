import json
import pypsa
from etrago import Etrago

import pandas as pd
import networkx as nx

###############################################################################

def investment_Germany(network, sto=True):
    
    de_buses = network.buses[network.buses.country == "DE"]
    
    de_lines = network.lines[(network.lines.bus0.isin(de_buses.index))&
                             (network.lines.bus1.isin(de_buses.index))&
                             network.lines.s_nom_extendable==True]
    
        
    de_links = network.links[(network.links.bus0.isin(de_buses.index))&
                             (network.links.bus1.isin(de_buses.index))&
                             network.links.p_nom_extendable==True]
    
    lines = ((de_lines.s_nom_opt-de_lines.s_nom_min
                         )*de_lines.capital_cost).sum()
    links = ((de_links.p_nom_opt-de_links.p_nom_min
                         )*de_links.capital_cost).sum()
    lines = lines + links
    
    if sto:
    
        de_sto = network.storage_units[(network.storage_units.carrier== 'battery')&
                                       network.storage_units.bus.isin(de_buses.index)]
        
        sto = ((de_sto.p_nom_opt-de_sto.p_nom_min)*de_sto.capital_cost).sum()
    
    else:
        sto= 0
    
    return lines + sto

def foreign_sto_expansion(network):
    
    bus = network.buses[network.buses.country!='DE']
    
    sto = network.stores[network.stores.bus.isin(bus.index)]
    sto_u = network.storage_units[network.storage_units.bus.isin(bus.index)]
    
    sto = ((sto.e_nom_opt -
            sto.e_nom_min
            )[sto.e_nom_extendable]*sto.capital_cost).sum()
    
    sto_u = ((sto_u.p_nom_opt - sto_u.p_nom_min
            )[sto_u.p_nom_extendable]\
                *sto_u.capital_cost).sum()
                
    return sto + sto_u

###############################################################################

etrago = Etrago(csv_folder_name="Calcs/max_line_load_dijkstra-case/no_sto_exp/original/1/")

kmeans = Etrago(csv_folder_name="Calcs/max_line_load_dijkstra-case/no_sto_exp/kmeans/300/")

dijkstra = Etrago(csv_folder_name="Calcs/max_line_load_dijkstra-case/no_sto_exp/kmedoids-dijkstra/300/")

etrago.calc_results()
etrago.results

kmeans.calc_results()
kmeans.results

dijkstra.calc_results()
dijkstra.results

df = pd.DataFrame(index=['system costs', 'operational costs', 'invest costs', 'grid expansion costs inside Germany', 'foreign grid expansion costs', 'battery expansion costs'], columns=['original network', 'k-means clustered network', 'rel. error k-means', 'k-medoids Dijkstra clustered network', 'rel. error k-medoids Dijkstra'])

df.at['system costs', 'original network'] = etrago.results.loc['annual system costs'].value
df.at['operational costs', 'original network'] = etrago.results.loc['annual marginal costs'].value
df.at['invest costs', 'original network'] = etrago.results.loc['annual investment costs'].value
df.at['transformer expansion costs', 'original network'] = ((etrago.network.transformers.s_nom_opt-etrago.network.transformers.s_nom_min)*etrago.network.transformers.capital_cost)[etrago.network.transformers.s_nom_extendable].sum()
df.at['grid expansion costs inside Germany', 'original network'] = investment_Germany(etrago.network, sto=False) #etrago.results.loc['annual ac grid investment costs'].value #
df.at['battery expansion costs', 'original network'] = etrago.results.loc['annual electrical storage investment costs'].value 
df.at['foreign grid expansion costs', 'original network'] = df.at['invest costs', 'original network'] - df.at['transformer expansion costs', 'original network'] - df.at['grid expansion costs inside Germany', 'original network'] - df.at['battery expansion costs', 'original network']   #etrago.results.loc['annual dc grid investment costs'].value # etrago.results.loc['annual electrical grid investment costs'].value-investment_Germany(etrago.network, sto=False) 
df.at['grid expansion inside Germany', 'original network'] = etrago.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'original network'] = etrago.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-means clustered network'] = kmeans.results.loc['annual system costs'].value
df.at['operational costs', 'k-means clustered network'] = kmeans.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-means clustered network'] = kmeans.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-means clustered network'] = investment_Germany(kmeans.network, sto=False)
df.at['foreign grid expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual electrical storage investment costs'].value 
df.at['grid expansion inside Germany', 'k-means clustered network'] = kmeans.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-means clustered network'] = kmeans.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual system costs'].value
df.at['operational costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-medoids Dijkstra clustered network'] = investment_Germany(dijkstra.network, sto=False)
df.at['foreign grid expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual electrical storage investment costs'].value
df.at['grid expansion inside Germany', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'rel. error k-means'] = ((kmeans.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-means'] = ((kmeans.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-means'] = ((kmeans.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-means'] = ((investment_Germany(kmeans.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
#df.at['battery expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.at['system costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-medoids Dijkstra'] = ((investment_Germany(dijkstra.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
#df.at['battery expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.fillna(0, inplace=True)

max_load_no_sto = df.copy()

###############################################################################

etrago = Etrago(csv_folder_name="Calcs/max_line_load_dijkstra-case/sto_exp/original/1/")

kmeans = Etrago(csv_folder_name="Calcs/max_line_load_dijkstra-case/sto_exp/kmeans/300/")

dijkstra = Etrago(csv_folder_name="Calcs/max_line_load_dijkstra-case/sto_exp/kmedoids-dijkstra/300/")

etrago.calc_results()
etrago.results

kmeans.calc_results()
kmeans.results

dijkstra.calc_results()
dijkstra.results

df = pd.DataFrame(index=['system costs', 'operational costs', 'invest costs', 'grid expansion costs inside Germany', 'foreign grid expansion costs', 'battery expansion costs'], columns=['original network', 'k-means clustered network', 'rel. error k-means', 'k-medoids Dijkstra clustered network', 'rel. error k-medoids Dijkstra'])

df.at['system costs', 'original network'] = etrago.results.loc['annual system costs'].value
df.at['operational costs', 'original network'] = etrago.results.loc['annual marginal costs'].value
df.at['invest costs', 'original network'] = etrago.results.loc['annual investment costs'].value
df.at['transformer expansion costs', 'original network'] = ((etrago.network.transformers.s_nom_opt-etrago.network.transformers.s_nom_min)*etrago.network.transformers.capital_cost)[etrago.network.transformers.s_nom_extendable].sum()
df.at['grid expansion costs inside Germany', 'original network'] = investment_Germany(etrago.network, sto=False) #etrago.results.loc['annual ac grid investment costs'].value #
df.at['battery expansion costs', 'original network'] = etrago.results.loc['annual electrical storage investment costs'].value 
df.at['foreign grid expansion costs', 'original network'] = df.at['invest costs', 'original network'] - df.at['transformer expansion costs', 'original network'] - df.at['grid expansion costs inside Germany', 'original network'] - df.at['battery expansion costs', 'original network']   #etrago.results.loc['annual dc grid investment costs'].value # etrago.results.loc['annual electrical grid investment costs'].value-investment_Germany(etrago.network, sto=False) 
df.at['grid expansion inside Germany', 'original network'] = etrago.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'original network'] = etrago.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-means clustered network'] = kmeans.results.loc['annual system costs'].value
df.at['operational costs', 'k-means clustered network'] = kmeans.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-means clustered network'] = kmeans.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-means clustered network'] = investment_Germany(kmeans.network, sto=False)
df.at['foreign grid expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual electrical storage investment costs'].value 
df.at['grid expansion inside Germany', 'k-means clustered network'] = kmeans.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-means clustered network'] = kmeans.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual system costs'].value
df.at['operational costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-medoids Dijkstra clustered network'] = investment_Germany(dijkstra.network, sto=False)
df.at['foreign grid expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual electrical storage investment costs'].value
df.at['grid expansion inside Germany', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'rel. error k-means'] = ((kmeans.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-means'] = ((kmeans.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-means'] = ((kmeans.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-means'] = ((investment_Germany(kmeans.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
df.at['battery expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.at['system costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-medoids Dijkstra'] = ((investment_Germany(dijkstra.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
df.at['battery expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.fillna(0, inplace=True)

max_load_sto = df.copy()

###############################################################################

etrago = Etrago(csv_folder_name="Calcs/max_wind_GER-case/3h/original/1")
# etrago = Etrago(csv_folder_name="Calcs/other_extreme_cases/3h/original/ohne_sto_exp/")

kmeans = Etrago(csv_folder_name="Calcs/max_wind_GER-case/3h/kmeans")

dijkstra = Etrago(csv_folder_name="Calcs/max_wind_GER-case/3h/kmedoids-dijkstra")

etrago.calc_results()
etrago.results

kmeans.calc_results()
kmeans.results

dijkstra.calc_results()
dijkstra.results

df = pd.DataFrame(index=['system costs', 'operational costs', 'invest costs', 'grid expansion costs inside Germany', 'foreign grid expansion costs', 'battery expansion costs'], columns=['original network', 'k-means clustered network', 'rel. error k-means', 'k-medoids Dijkstra clustered network', 'rel. error k-medoids Dijkstra'])

df.at['system costs', 'original network'] = etrago.results.loc['annual system costs'].value
df.at['operational costs', 'original network'] = etrago.results.loc['annual marginal costs'].value
df.at['invest costs', 'original network'] = etrago.results.loc['annual investment costs'].value
df.at['transformer expansion costs', 'original network'] = ((etrago.network.transformers.s_nom_opt-etrago.network.transformers.s_nom_min)*etrago.network.transformers.capital_cost)[etrago.network.transformers.s_nom_extendable].sum()
df.at['grid expansion costs inside Germany', 'original network'] = investment_Germany(etrago.network, sto=False) #etrago.results.loc['annual ac grid investment costs'].value #
df.at['battery expansion costs', 'original network'] = etrago.results.loc['annual electrical storage investment costs'].value 
df.at['foreign grid expansion costs', 'original network'] = df.at['invest costs', 'original network'] - df.at['transformer expansion costs', 'original network'] - df.at['grid expansion costs inside Germany', 'original network'] - df.at['battery expansion costs', 'original network']   #etrago.results.loc['annual dc grid investment costs'].value # etrago.results.loc['annual electrical grid investment costs'].value-investment_Germany(etrago.network, sto=False) 
df.at['grid expansion inside Germany', 'original network'] = etrago.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'original network'] = etrago.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-means clustered network'] = kmeans.results.loc['annual system costs'].value
df.at['operational costs', 'k-means clustered network'] = kmeans.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-means clustered network'] = kmeans.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-means clustered network'] = investment_Germany(kmeans.network, sto=False)
df.at['foreign grid expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual electrical storage investment costs'].value 
df.at['grid expansion inside Germany', 'k-means clustered network'] = kmeans.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-means clustered network'] = kmeans.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual system costs'].value
df.at['operational costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-medoids Dijkstra clustered network'] = investment_Germany(dijkstra.network, sto=False)
df.at['foreign grid expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual electrical storage investment costs'].value
df.at['grid expansion inside Germany', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'rel. error k-means'] = ((kmeans.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-means'] = ((kmeans.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-means'] = ((kmeans.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-means'] = ((investment_Germany(kmeans.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
df.at['battery expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.at['system costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-medoids Dijkstra'] = ((investment_Germany(dijkstra.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
df.at['battery expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.fillna(0, inplace=True)

max_wind = df.copy()

###############################################################################

etrago = Etrago(csv_folder_name="Calcs/feedin-case/2h/original")
etrago.network.set_snapshots(['2011-05-26 13:00:00'])

kmeans = Etrago(csv_folder_name="Calcs/feedin-case/2h/kmeans")
kmeans.network.set_snapshots(['2011-05-26 13:00:00'])

dijkstra = Etrago(csv_folder_name="Calcs/feedin-case/2h/kmedoids-dijkstra")
dijkstra.network.set_snapshots(['2011-05-26 13:00:00'])

etrago.calc_results()
etrago.results

kmeans.calc_results()
kmeans.results

dijkstra.calc_results()
dijkstra.results

df = pd.DataFrame(index=['system costs', 'operational costs', 'invest costs', 'grid expansion costs inside Germany', 'foreign grid expansion costs', 'battery expansion costs'], columns=['original network', 'k-means clustered network', 'rel. error k-means', 'k-medoids Dijkstra clustered network', 'rel. error k-medoids Dijkstra'])

df.at['system costs', 'original network'] = etrago.results.loc['annual system costs'].value
df.at['operational costs', 'original network'] = etrago.results.loc['annual marginal costs'].value
df.at['invest costs', 'original network'] = etrago.results.loc['annual investment costs'].value
df.at['transformer expansion costs', 'original network'] = ((etrago.network.transformers.s_nom_opt-etrago.network.transformers.s_nom_min)*etrago.network.transformers.capital_cost)[etrago.network.transformers.s_nom_extendable].sum()
df.at['grid expansion costs inside Germany', 'original network'] = investment_Germany(etrago.network, sto=False) #etrago.results.loc['annual ac grid investment costs'].value #
df.at['battery expansion costs', 'original network'] = etrago.results.loc['annual electrical storage investment costs'].value 
df.at['foreign grid expansion costs', 'original network'] = df.at['invest costs', 'original network'] - df.at['transformer expansion costs', 'original network'] - df.at['grid expansion costs inside Germany', 'original network'] - df.at['battery expansion costs', 'original network']   #etrago.results.loc['annual dc grid investment costs'].value # etrago.results.loc['annual electrical grid investment costs'].value-investment_Germany(etrago.network, sto=False) 
df.at['grid expansion inside Germany', 'original network'] = etrago.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'original network'] = etrago.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-means clustered network'] = kmeans.results.loc['annual system costs'].value
df.at['operational costs', 'k-means clustered network'] = kmeans.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-means clustered network'] = kmeans.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-means clustered network'] = investment_Germany(kmeans.network, sto=False)
df.at['foreign grid expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual electrical storage investment costs'].value 
df.at['grid expansion inside Germany', 'k-means clustered network'] = kmeans.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-means clustered network'] = kmeans.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual system costs'].value
df.at['operational costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-medoids Dijkstra clustered network'] = investment_Germany(dijkstra.network, sto=False)
df.at['foreign grid expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual electrical storage investment costs'].value
df.at['grid expansion inside Germany', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'rel. error k-means'] = ((kmeans.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-means'] = ((kmeans.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-means'] = ((kmeans.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-means'] = ((investment_Germany(kmeans.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
df.at['battery expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.at['system costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-medoids Dijkstra'] = ((investment_Germany(dijkstra.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
df.at['battery expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.fillna(0, inplace=True)

feedin = df.copy()

###############################################################################

etrago = Etrago(csv_folder_name="Calcs/load-case/2h/original")
etrago.network.set_snapshots(['2011-01-31 08:00:00'])

kmeans = Etrago(csv_folder_name="Calcs/load-case/2h/kmeans")
kmeans.network.set_snapshots(['2011-01-31 08:00:00'])

dijkstra = Etrago(csv_folder_name="Calcs/load-case/2h/kmedoids-dijkstra")
dijkstra.network.set_snapshots(['2011-01-31 08:00:00'])

etrago.calc_results()
etrago.results

kmeans.calc_results()
kmeans.results

dijkstra.calc_results()
dijkstra.results

df = pd.DataFrame(index=['system costs', 'operational costs', 'invest costs', 'grid expansion costs inside Germany', 'foreign grid expansion costs', 'battery expansion costs'], columns=['original network', 'k-means clustered network', 'rel. error k-means', 'k-medoids Dijkstra clustered network', 'rel. error k-medoids Dijkstra'])

df.at['system costs', 'original network'] = etrago.results.loc['annual system costs'].value
df.at['operational costs', 'original network'] = etrago.results.loc['annual marginal costs'].value
df.at['invest costs', 'original network'] = etrago.results.loc['annual investment costs'].value
df.at['transformer expansion costs', 'original network'] = ((etrago.network.transformers.s_nom_opt-etrago.network.transformers.s_nom_min)*etrago.network.transformers.capital_cost)[etrago.network.transformers.s_nom_extendable].sum()
df.at['grid expansion costs inside Germany', 'original network'] = investment_Germany(etrago.network, sto=False) #etrago.results.loc['annual ac grid investment costs'].value #
df.at['battery expansion costs', 'original network'] = etrago.results.loc['annual electrical storage investment costs'].value 
df.at['foreign grid expansion costs', 'original network'] = df.at['invest costs', 'original network'] - df.at['transformer expansion costs', 'original network'] - df.at['grid expansion costs inside Germany', 'original network'] - df.at['battery expansion costs', 'original network']   #etrago.results.loc['annual dc grid investment costs'].value # etrago.results.loc['annual electrical grid investment costs'].value-investment_Germany(etrago.network, sto=False) 
df.at['grid expansion inside Germany', 'original network'] = etrago.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'original network'] = etrago.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-means clustered network'] = kmeans.results.loc['annual system costs'].value
df.at['operational costs', 'k-means clustered network'] = kmeans.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-means clustered network'] = kmeans.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-means clustered network'] = investment_Germany(kmeans.network, sto=False)
df.at['foreign grid expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-means clustered network'] = kmeans.results.loc['annual electrical storage investment costs'].value 
df.at['grid expansion inside Germany', 'k-means clustered network'] = kmeans.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-means clustered network'] = kmeans.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual system costs'].value
df.at['operational costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual marginal costs'].value
df.at['invest costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual investment costs'].value
df.at['grid expansion costs inside Germany', 'k-medoids Dijkstra clustered network'] = investment_Germany(dijkstra.network, sto=False)
df.at['foreign grid expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual dc grid investment costs'].value
df.at['battery expansion costs', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['annual electrical storage investment costs'].value
df.at['grid expansion inside Germany', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical ac grid expansion'].value
df.at['foreign grid expansion', 'k-medoids Dijkstra clustered network'] = dijkstra.results.loc['abs. electrical dc grid expansion'].value

df.at['system costs', 'rel. error k-means'] = ((kmeans.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-means'] = ((kmeans.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-means'] = ((kmeans.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-means'] = ((investment_Germany(kmeans.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
df.at['battery expansion costs', 'rel. error k-means'] = ((kmeans.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.at['system costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual system costs'].value-etrago.results.loc['annual system costs'].value)/etrago.results.loc['annual system costs'].value)*100
df.at['operational costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual marginal costs'].value-etrago.results.loc['annual marginal costs'].value)/etrago.results.loc['annual marginal costs'].value)*100
df.at['invest costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual investment costs'].value-etrago.results.loc['annual investment costs'].value)/etrago.results.loc['annual investment costs'].value)*100
df.at['grid expansion costs inside Germany', 'rel. error k-medoids Dijkstra'] = ((investment_Germany(dijkstra.network, sto=False)-etrago.results.loc['annual ac grid investment costs'].value)/etrago.results.loc['annual ac grid investment costs'].value)*100
df.at['foreign grid expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual dc grid investment costs'].value-etrago.results.loc['annual dc grid investment costs'].value)/etrago.results.loc['annual dc grid investment costs'].value)*100
df.at['battery expansion costs', 'rel. error k-medoids Dijkstra'] = ((dijkstra.results.loc['annual electrical storage investment costs'].value-etrago.results.loc['annual electrical storage investment costs'].value)/etrago.results.loc['annual electrical storage investment costs'].value)*100

df.fillna(0, inplace=True)

load = df.copy()

################################################################################

'''loads1 = etrago.network.generators[etrago.network.generators.carrier=='load shedding']
etrago.network.generators_t.p.transpose().loc[loads1.index].sum().sum()

loads2 = dijkstra.network.generators[dijkstra.network.generators.carrier=='load shedding']
dijkstra.network.generators_t.p.transpose().loc[loads2.index].sum().sum()

loads3 = kmeans.network.generators[kmeans.network.generators.carrier=='load shedding']
kmeans.network.generators_t.p.transpose().loc[loads3.index].sum().sum()'''

################################################################################

three = Etrago(csv_folder_name="Calcs/dijkstra_300")
(abs(three.network.lines_t.p0)).sum(axis=1).idxmax()

################################################################################

sns = (
    pd.date_range("2019-01-01 00:00", "2019-12-31 23:00", freq="H")
)

start = 581
end = start+1

sns[start:end]