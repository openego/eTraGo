#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:33:08 2025

@author: student
"""
import matplotlib.pyplot as plt
import pandas as pd


#import pypsa

#market_DE2= pypsa.Network()
#market_DE2.import_from_csv_folder("/home/student/Masterarbeit_Dateien/Ergebnisse/results_DE2_marketzones_assigned/market") 


#market_DE3 = pypsa.Network()
#market_DE3.import_from_csv_folder("/home/student/Masterarbeit_Dateien/Ergebnisse/results_DE3_latest/market") 

#market_DE4 = pypsa.Network()
#market_DE4.import_from_csv_folder("/home/student/Masterarbeit_Dateien/Ergebnisse/results_DE4_marketzones_assigned/market")

#market_sq = pypsa.Network()
#market_sq.import_from_csv_folder("/home/student/Masterarbeit_Dateien/Ergebnisse/results_status_quo_latest/market")

#net_DE2= pypsa.Network()
#net_DE2.import_from_csv_folder("/home/student/Masterarbeit_Dateien/Ergebnisse/results_DE2_marketzones_assigned") 


#net_DE3 = pypsa.Network()
#net_DE3.import_from_csv_folder("/home/student/Masterarbeit_Dateien/Ergebnisse/results_DE3_latest") 

#net_DE4 = pypsa.Network()
#net_DE4.import_from_csv_folder("/home/student/Masterarbeit_Dateien/Ergebnisse/results_DE4_marketzones_assigned")

#net_sq = pypsa.Network()
#net_sq.import_from_csv_folder("/home/student/Masterarbeit_Dateien/Ergebnisse/results_status_quo_latest")

#net_nodal = pypsa.Network()
#net_nodal.import_from_csv_folder("/home/student/Masterarbeit_Dateien/Ergebnisse/results_nodal_latest")


#from etrago import Etrago

#etrago_nodal = Etrago(csv_folder_name="/home/student/Masterarbeit_Dateien/Ergebnisse/results_nodal_latest")

#etrago_DE2 = Etrago(csv_folder_name="/home/student/Masterarbeit_Dateien/Ergebnisse/results_DE2_marketzones_assigned")

#etrago_DE3 = Etrago(csv_folder_name="/home/student/Masterarbeit_Dateien/Ergebnisse/results_DE3_latest")

#etrago_DE4 = Etrago(csv_folder_name="/home/student/Masterarbeit_Dateien/Ergebnisse/results_DE4_marketzones_assigned")

#etrago_status_quo = Etrago(csv_folder_name="/home/student/Masterarbeit_Dateien/Ergebnisse/results_status_quo_latest")


"""
import pandas as pd
from calc_results_germany import calc_etrago_results
from calc_results_germany import _calc_storage_expansion
from calc_results_germany import _calc_store_expansion
from calc_results_germany import _calc_sectorcoupling_link_expansion
from calc_results_germany import _calc_network_expansion
from calc_results_germany import _calc_network_expansion_length
from calc_results_germany import calc_investment_cost
from calc_results_germany import calc_marginal_cost
from calc_results_germany import german_network
"""

def assign_market_zones_to_buses(network, market_zones):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    if market_zones == "DE2":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp"
    elif market_zones == "DE3":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_12_DE3.shp"
    elif market_zones == "DE4":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp"
    elif market_zones == "DE5":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp"
    elif market_zones == "none":
        shapefile_path = None
    else:
        raise ValueError("Ungültiger Wert für market_zone. Erlaubt sind: 'DE2', 'DE3', 'DE4', 'DE5', oder 'none'.")


    if shapefile_path is None:
        raise ValueError("Ungültiger Wert für market_zones")

    zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # Buserstellen mit Geometrie
    buses = network.buses.copy()
    bus_points = gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses.x, buses.y),
        crs="EPSG:4326"
    )

    # Spatial Join: Busse mit Zonen verbinden
    joined = gpd.sjoin(bus_points, zones, how="left", predicate="within")

    # 'id' ist hier die Zone
    network.buses["zone"] = joined["id"]

    # Diagnose
    missing = network.buses["zone"].isna().sum()
    print(f"{missing} Bussen konnte keine Zone zugewiesen werden.")

    return network


def calculate_and_extract_results(etrago_obj):
    """Berechnet die Ergebnisse und extrahiert die relevanten Daten."""
    calc_etrago_results(etrago_obj)
    return etrago_obj.results


def compare_configurations(etrago_DE2, etrago_DE3, etrago_status_quo):
    """Vergleicht die Ergebnisse der beiden Konfigurationen."""
    # Berechne die Ergebnisse für jede Konfiguration
    results_DE2 = calculate_and_extract_results(etrago_DE2)
    results_DE3 = calculate_and_extract_results(etrago_DE3)
    results_status_quo = calculate_and_extract_results(etrago_status_quo)

    # Erstelle ein DataFrame für den Vergleich
    comparison_df = pd.DataFrame({
        "DE2": results_DE2,
        "DE3": results_DE3,
        "Status Quo": results_status_quo
    })

    return comparison_df




def get_translink_capacities(self):
    """
    Analyze the links in the network and print the capacities in MW of the links
    between market zones".

    Prints
    ------
    The names of the links and their corresponding 'p_nom' values.
    """
    # Filtern der Links, deren Index mit "transshipment_" beginnt
    links = self.market_model.links
    filtered_links = links[links.index.str.startswith('transshipment_')]

    # Erstellen der neuen Spalten 'bus0_country' und 'bus1_country'
    buses = self.market_model.buses

    for index, row in filtered_links.iterrows():
        bus0_country = buses.loc[row['bus0'], 'country']
        bus1_country = buses.loc[row['bus1'], 'country']
        link_name = f"{bus0_country}-{bus1_country}"
        p_nom = row['p_nom']
        print(f"{link_name}, {p_nom}")


import pandas as pd
import re
import matplotlib.pyplot as plt



def total_dispatch(self, timesteps=range(1752)):
    """
    Calculate and print the dispatch summary for the given timesteps.

    Parameters
    ----------
    etrago_nodal : Etrago object
        The Etrago model object.
    timesteps : range or list
        The timesteps to consider in the dispatch calculation.
    """
        
    # Berechnen Sie den Dispatch pro Carrier für die angegebenen Zeitschritte
    dispatch_per_carrier_net = calc_dispatch_per_carrier(german_network(self.network), timesteps, dispatch_type="total")
    dispatch_per_carrier_market = calc_dispatch_per_carrier(german_network(self.market_model), timesteps, dispatch_type="total")
    
    # Gruppieren Sie die Ergebnisse nach 'carrier' und summieren Sie die Werte
    sum_per_carrier_net = ((dispatch_per_carrier_net.groupby('carrier').sum())*5)/1e6
    sum_per_carrier_market = ((dispatch_per_carrier_market.groupby('carrier').sum())*5)/1e6
    
    # Definieren Sie die erneuerbaren Carrier einschließlich Biomasse
    renewable_carriers = [
        'solar', 'solar_rooftop', 'wind_offshore', 'wind_onshore',
        'reservoir', 'run_of_river', 'biomass', 'central_biomass_CHP', 'industrial_biomass_CHP'
    ]

    # Filtern Sie die erneuerbaren Carrier und summieren Sie deren Werte
    sum_total_twh_net = sum_per_carrier_net.sum()
    sum_total_twh_market = sum_per_carrier_market.sum()
    
    sum_renewables_twh_net = sum_per_carrier_net[sum_per_carrier_net.index.isin(renewable_carriers)].sum()
    sum_renewables_twh_market = sum_per_carrier_market[sum_per_carrier_market.index.isin(renewable_carriers)].sum()
    # Rechnen Sie die Summe von MWh in TWh um
   
    
    
    #Ausgabe der Ergebnisse
        
    print("\nStromproduktion nach Marktoptimierung gesamt:")
    print(f"{sum_total_twh_market} TWh")
    print("\nDavon Erneuerbar:")
    print(f"{sum_renewables_twh_market} TWh")
    
    print("\nStromproduktion nach Netzoptimierung gesamt:")
    print(f"{sum_total_twh_net} TWh")
    print("\nDavon Erneuerbar:")
    print(f"{sum_renewables_twh_net} TWh")
    
    return sum_per_carrier_market, sum_per_carrier_net


  

def compare_dispatch(etrago_objects):
    """
    Compare the dispatch results from multiple Etrago objects.

    Parameters
    ----------
    etrago_objects : dict
        A dictionary of Etrago model objects with keys as scenario names.
        
        Example usage:
         etrago_objects = {
             'status_quo': etrago_status_quo,
             'DE2': etrago_DE2,
             'DE3': etrago_DE3,
             'DE4': etrago_DE4,
             'DE5': etrago_DE5,
             'nodal': etrago_nodal
             }

    Returns
    -------
    pd.DataFrame
        A DataFrame with carriers as rows and scenarios as columns containing the sum of generated energy.
    """
    # Initialize a dictionary to store the results for each scenario
    scenario_results = {}

    for scenario_name, etrago_obj in etrago_objects.items():
        # Call total_dispatch for each Etrago object
        sum_per_carrier = total_dispatch(etrago_obj)

        # Store the results in the dictionary
        scenario_results[scenario_name] = sum_per_carrier

    # Combine the results into a single DataFrame
    combined_df = pd.concat(scenario_results, axis=1)

    
    # Define soft colors for scenarios
    scenario_colors = {
        'status_quo': '#7986CB',  # Soft Blue
        'nodal': '#BDE0FE',       # Soft Orange
        'DE2': '#A5D6A7',         # Soft Green
        'DE3': '#C8E6C9',         # Lighter Soft Green
        'DE4': '#E8F5E9',         # Even Lighter Soft Green
        'DE5': '#F1F8E9'          # Lightest Soft Green
    }

    # Separate carriers into two groups
    carriers_gt_15 = combined_df[combined_df.sum(axis=1) > 15]
    carriers_lt_15 = combined_df[combined_df.sum(axis=1) <= 15]

    # Plot for carriers > 15 TWh
    plt.figure(figsize=(14, 8))
    num_carriers_gt_15 = len(carriers_gt_15.index)
    num_scenarios = len(carriers_gt_15.columns)
    bar_width = 0.8 / num_scenarios
    x_positions = range(num_carriers_gt_15)

    for i, (scenario, values) in enumerate(carriers_gt_15.items()):
        plt.bar([x + i * bar_width for x in x_positions], values, width=bar_width, label=scenario, color=scenario_colors[scenario])

    plt.xticks([x + (num_scenarios - 1) * bar_width / 2 for x in x_positions], carriers_gt_15.index, rotation=45)
    plt.ylabel('Erzeugter Strom (TWh)')
    plt.title('Stromerzeugung (> 15 TWh) und Import je nach Marktkonfiguration')
    plt.legend(title='Marktkonfiguration')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Plot for carriers < 15 TWh
    plt.figure(figsize=(14, 8))
    num_carriers_lt_15 = len(carriers_lt_15.index)
    num_scenarios = len(carriers_lt_15.columns)
    bar_width = 0.8 / num_scenarios
    x_positions = range(num_carriers_lt_15)

    for i, (scenario, values) in enumerate(carriers_lt_15.items()):
        plt.bar([x + i * bar_width for x in x_positions], values, width=bar_width, label=scenario, color=scenario_colors[scenario])

    plt.xticks([x + (num_scenarios - 1) * bar_width / 2 for x in x_positions], carriers_lt_15.index, rotation=45)
    plt.xlabel('Carrier')
    plt.ylabel('Erzeugte Energie (TWh)')
    plt.title('Erzeugte Energie pro Carrier (< 15 TWh) und Szenario')
    plt.legend(title='Szenario')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def total_dispatch_by_zone(self, timesteps=range(1752)):
    """
    Berechnet die Stromproduktion je Carrier und Zone für die gegebenen Zeitschritte.
    Gibt die Gesamt- und EE-Produktion je Zone aus.

    Parameters
    ----------
    timesteps : range oder Liste
        Die Zeitschritte, die berücksichtigt werden sollen.

    Returns
    -------
    dispatch_per_zone : DataFrame
        Stromproduktion in TWh je Zone und Carrier.
    """

    # Dispatch pro Carrier und Bus berechnen (gibt Series zurück)
    dispatch_series = calc_dispatch_per_carrier(self.german_network(), timesteps, dispatch_type="total")

    # In DataFrame umwandeln
    dispatch_df = dispatch_series.reset_index()
    dispatch_df.columns = ['bus', 'carrier', 'dispatch']

    # Sicherstellen, dass 'zone' in buses vorhanden ist
    buses = self.german_network().buses
    if 'zone' not in buses.columns:
        raise ValueError("Spalte 'zone' fehlt in network.buses. Bitte zuerst Zonen zuweisen.")

    # Zone pro Bus zuordnen
    dispatch_df = dispatch_df.merge(buses['zone'], left_on='bus', right_index=True)

    # Nach zone und carrier gruppieren und summieren, dann auf TWh skalieren
    dispatch_per_zone = ((dispatch_df.groupby(['zone', 'carrier'])['dispatch'].sum()) * 5) / 1e6

    # Tabelle für Ausgabe pivotieren
    table = dispatch_per_zone.unstack().fillna(0)

    # Definiere Erneuerbare
    renewables = [
        'solar', 'solar_rooftop', 'wind_offshore', 'wind_onshore',
        'reservoir', 'run_of_river', 'biomass', 'central_biomass_CHP', 'industrial_biomass_CHP'
    ]

    # Ergebnis-Ausgabe
    print("\nStromproduktion pro Zone (TWh gesamt / Erneuerbar):")
    for zone in table.index:
        total = table.loc[zone].sum()
        renew = table.loc[zone][renewables].sum()
        print(f"{zone}: {total:.2f} TWh gesamt, davon {renew:.2f} TWh erneuerbar ({(renew/total*100):.1f}%)")

    return dispatch_per_zone


import pypsa
import pandas as pd

def calculate_annual_marginal_costs_by_carrier(self):
    network = self.german_network()

    # Gesamt-Marginalkosten je Generator-Zeitpunkt berechnen
    marginal_costs_df = (
        network.generators_t.p.mul(network.snapshot_weightings.objective, axis=0)
        .mul(pypsa.descriptors.get_switchable_as_dense(network, "Generator", "marginal_cost"))
    )

    # Liste aller Carrier, die berücksichtigt werden sollen
    carriers = [
        "CH4", "CH4_NG", "CH4_biogas", "OCGT", "biomass", "central_biomass_CHP", "central_gas_CHP",
        "central_biomass_CHP_heat", "coal", "geo_thermal", "industrial_biomass_CHP", "industrial_gas_CHP", "load_shedding",
        "lignite", "nuclear", "oil", "others", "reservoir", "run_of_river", 
        "solar", "solar_rooftop", "solar_thermal_collector", 
        "wind_offshore", "wind_onshore"
    ]

    # Ergebnisdictionary: Carrier -> jährliche marginale Kosten
    annual_marginal_costs = {}

    for carrier in carriers:
        # Spalten mit dem entsprechenden Carrier auswählen
        relevant_cols = [col for col in marginal_costs_df.columns if carrier in col]
        if relevant_cols:
            carrier_df = marginal_costs_df[relevant_cols]
            # Summe über alle Zeitpunkte und Generatoren
            total_cost = carrier_df.sum().sum()
            annual_marginal_costs[carrier] = total_cost
        else:
            annual_marginal_costs[carrier] = 0.0  # falls kein Generator dieses Typs vorhanden ist

    return annual_marginal_costs
    


# Beispielaufruf der Funktion
# Angenommen, 'self' ist dein Etrago-Objekt
# total_dispatch(self)

#import pdb; pdb.set_trace()

# Vergleiche die Konfigurationen
#comparison_results = compare_configurations(etrago_DE2, etrago_DE3, etrago_status_quo)

# Ausgabe der Vergleichsergebnisse
#print(comparison_results)

# Optional: Speichere die Ergebnisse in einer CSV-Datei
#comparison_results.to_csv("comparison_results_germany.csv")
