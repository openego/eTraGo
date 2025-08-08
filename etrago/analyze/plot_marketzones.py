#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import numpy as np
from shapely.geometry import Point
import pandas as pd

def plot_marketzone_clustering(self, market_zones):
    # Optionales Einlesen des Shapefiles je nach market_zone
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

    # Erstellen einer GeoAxesSubplot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

    # Netzbusse für Deutschland laden
    market_bus_de = self.buses[
        (self.buses.country == "DE")
        & (self.buses.carrier == "AC")
    ]

    # Geodataframe für Busse erstellen
    gdf_buses = gpd.GeoDataFrame(
        market_bus_de,
        geometry=gpd.points_from_xy(market_bus_de.x, market_bus_de.y),
        crs="EPSG:4326"
    )

    if shapefile_path:
        zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

        # Farbpalette und Farben zuweisen
        colors = plt.cm.tab20(range(len(zones)))
        zones['color'] = [mcolors.rgb2hex(color[:3]) for color in colors]

        # Zonen plotten
        zones.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        zones.plot(ax=ax, facecolor=zones['color'], alpha=0.3)

        # Durchschnittspreis pro Zone berechnen und einzeichnen
        for idx, zone in zones.iterrows():
            buses_in_zone = gdf_buses[gdf_buses.within(zone['geometry'])]
            if not buses_in_zone.empty:
                prices_in_zone = self.buses_t.marginal_price[buses_in_zone.index]
                avg_price = prices_in_zone.mean().mean()
                center = zone['geometry'].centroid
                ax.text(center.x, center.y, f"{avg_price:.2f}", fontsize=7, ha='center', color='black')
    else:
        # Durchschnittspreis für ganz Deutschland berechnen und in Kartenmitte plotten
        market_price = self.buses_t.marginal_price[market_bus_de.index]
        avg_price = market_price.mean().mean()

        # Mittelpunkt Deutschlands als Koordinate annehmen (z. B. geografischer Mittelpunkt bei Niederdorla)
        center_lon, center_lat = 10.4515, 51.1657
        ax.text(center_lon, center_lat, f"{avg_price:.2f} ", fontsize=7, ha='center', color='black')

    # Netzwerk plotten
    self.plot(
        ax=ax,
        link_widths=self.links.apply(
            lambda row: 0 if row['carrier'] in ['DC', 'CH4', 'central_resistive_heater', 'rural_heat_pump', 'H2_to_power']
            else 0,
            axis=1
        ),
        bus_sizes=self.buses.apply(
            lambda row: 0 if row['carrier'] == 'AC' else 0,
            axis=1
        )
    )
    plt.show()

    # --- Säulendiagramm: Preisverteilung ---
    bins = [0, 5, 10, 40, 100, np.inf]
    labels = ['0-5', '6-10', '11-40', '41-100', '>100']

    if shapefile_path:
        zone_distributions = []
        zone_labels = []

        for idx, zone in zones.iterrows():
            buses_in_zone = gdf_buses[gdf_buses.within(zone['geometry'])]
            if not buses_in_zone.empty:
                prices_in_zone = self.buses_t.marginal_price[buses_in_zone.index]
                prices_in_zone = prices_in_zone.values.flatten()
                prices_in_zone = prices_in_zone[~np.isnan(prices_in_zone)]

                price_distribution, _ = np.histogram(prices_in_zone, bins=bins)
                zone_distributions.append(price_distribution)
                zone_labels.append(f'Zone {idx}')

        if zone_distributions:
            x = np.arange(len(labels))
            width = 0.8 / len(zone_distributions)

            fig2, ax2 = plt.subplots(figsize=(12, 8))
            for i, (distribution, label, color) in enumerate(zip(zone_distributions, zone_labels, zones['color'])):
                offset = width * i
                ax2.bar(x + offset, distribution, width, label=label, color=color, alpha=0.6)

            ax2.set_ylabel('Anzahl der Stunden', fontsize=30)
            ax2.set_ylim(0, 4200)
            ax2.set_yticks(np.arange(0, 4001, 500))  # Y-Ticks alle 500
            ax2.yaxis.grid(True, linestyle='--', linewidth=0.8, color='gray', alpha = 0.3)  # Horizontale gestrichelte Linien

            ax2.set_xticks(x + width)
            ax2.set_xticklabels(labels)
            #ax2.set_xlabel('Preisintervalle (€/MWh)', fontsize=15)
            ax2.tick_params(axis='both', which='major', labelsize=25)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    else:
        # Kein Zonen-Shapefile → gesamtes Deutschland analysieren
        prices_in_de = self.buses_t.marginal_price[market_bus_de.index]
        prices_in_de = prices_in_de.values.flatten()
        prices_in_de = prices_in_de[~np.isnan(prices_in_de)]

        price_distribution, _ = np.histogram(prices_in_de, bins=bins)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(labels, price_distribution, width=0.6, color='steelblue', alpha=0.7)

        ax2.set_ylabel('Anzahl der Stunden', fontsize=25)
        ax2.set_ylim(0, 4200)
        ax2.set_yticks(np.arange(0, 4001, 500))  # Y-Ticks alle 500
        ax2.yaxis.grid(True, linestyle='--', linewidth=0.8, color='gray', alpha = 0.3 )  # Horizontale gestrichelte Linien

        #ax2.set_xlabel('Preisintervalle (€/MWh)', fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



"""
def plot_marketzone_clustering(self, market_zones):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import numpy as np
    
    # Optionales Einlesen des Shapefiles je nach market_zone
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

    # Erstellen einer GeoAxesSubplot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

    # Fa`lls gewünscht: Marktzonen laden und plotten
    if shapefile_path:
        zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

        # Farbpalette und Farben zuweisen
        colors = plt.cm.tab20(range(len(zones)))
        zones['color'] = [mcolors.rgb2hex(color[:3]) for color in colors]

        # Zonen plotten
        zones.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        zones.plot(ax=ax, facecolor=zones['color'], alpha=0.3)

    # Netzwerk plotten
    self.plot(
        ax=ax,
        link_widths=0,
    
        bus_sizes=self.buses.apply(
            bus_sizes = self.buses.apply(
                lambda row: 0.05 if row.name in self.ramps_down("wind_offshore").index else 0,
                axis=1
        ),
        line_widths=0.5
    ))
    plt.show()
    

    # p_nom Werte neben den Links anzeigen, wenn country == "DE"
   # for idx, row in self.market_model.links.iterrows():
    #    if row['country'] == 'DE' and row['carrier'] == 'DC':
     #       bus0_coords = self.market_model.buses.loc[row['bus0'], ['x', 'y']]
      #      bus1_coords = self.market_model.buses.loc[row['bus1'], ['x', 'y']]
       #     mid_x = (bus0_coords['x'] + bus1_coords['x']) / 2
        #    mid_y = (bus0_coords['y'] + bus1_coords['y']) / 2
         #   p_nom_int = int(row['p_nom'])/1000  # Nur die Zahlen vor dem Komma verwenden
          #  ax.text(mid_x, mid_y, f"{p_nom_int} MW", fontsize=5, ha='center')
    #plt.show()
"""    

import geopandas as gpd
import pandas as pd
import requests
import os
from shapely.geometry import Point

def total_dispatch_by_zone(self, timesteps=range(1752), market_zones="DE4"):
    """    
    Berechnet die Stromproduktion je Carrier und Zone und plottet sie auf einer Karte mit EE-Anteil.

    Parameters
    ----------
    timesteps : range oder Liste
        Die Zeitschritte, die berücksichtigt werden sollen.
    market_zones : str
        Marktzonen-Konfiguration: 'DE2', 'DE3', 'DE4', 'DE5'

    Returns
    -------
    dispatch_per_zone : DataFrame
        Stromproduktion in TWh je Zone und Carrier.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import numpy as np

    # --- 1. Dispatch berechnen ---
    dispatch_series = calc_dispatch_per_carrier(german_network(self), timesteps, dispatch_type="total")
    dispatch_df = dispatch_series.reset_index()
    dispatch_df.columns = ['bus', 'carrier', 'dispatch']

    buses = self.german_network().buses
    if 'zone' not in buses.columns:
        raise ValueError("Spalte 'zone' fehlt in network.buses. Bitte zuerst Zonen zuweisen.")
    dispatch_df = dispatch_df.merge(buses['zone'], left_on='bus', right_index=True)

    # in TWh skalieren
    dispatch_per_zone = ((dispatch_df.groupby(['zone', 'carrier'])['dispatch'].sum()) * 5) / 1e6
    table = dispatch_per_zone.unstack().fillna(0)

    # Erneuerbare Carrier
    renewables = [
        'solar', 'solar_rooftop', 'wind_offshore', 'wind_onshore',
        'reservoir', 'run_of_river', 'biomass', 'central_biomass_CHP', 'industrial_biomass_CHP'
    ]

    print("\nStromproduktion pro Zone (TWh gesamt / Erneuerbar):")
    for zone in table.index:
        total = table.loc[zone].sum()
        renew = table.loc[zone][renewables].sum()
        print(f"{zone}: {total:.2f} TWh gesamt, davon {renew:.2f} TWh erneuerbar ({(renew/total*100):.1f}%)")

    # --- 2. Plot ---
    # Shapefile wählen
    if market_zones == "DE2":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp"
    elif market_zones == "DE3":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_12_DE3.shp"
    elif market_zones == "DE4":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp"
    elif market_zones == "DE5":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp"
    else:
        raise ValueError("Ungültiger Wert für market_zones. Erlaubt sind: 'DE2', 'DE3', 'DE4', 'DE5'.")

    zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # Farben zuweisen
    colors = plt.cm.tab20(range(len(zones)))
    zones['color'] = [mcolors.rgb2hex(color[:3]) for color in colors]

    # Plot starten
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

    # Marktzonen zeichnen
    zones.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
    zones.plot(ax=ax, facecolor=zones['color'], alpha=0.3)

    # Netz zeichnen wie in plot_marketzone_clustering
    self.market_model.plot(
        ax=ax,
        link_widths=0,
        bus_sizes=0
    )

    # Textplot: TWh und EE-Anteil in Zone-Mitte
    for _, row in zones.iterrows():
        zone_name = row["id"]
        if zone_name not in table.index:
            continue
        total = table.loc[zone_name].sum()
        renew = table.loc[zone_name][renewables].sum()
        share = renew / total * 100 if total > 0 else 0
        centroid = row.geometry.centroid
        ax.text(
            centroid.x, centroid.y,
            f"{total:.1f} TWh\n{share:.1f}% EE",
            fontsize=20, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )


    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return dispatch_per_zone

def calculate_inter_zonal_trade(network, market_zones):
    """Calculate electricity trade between German bidding zones and neighboring countries

    Parameters
    ----------
    network : object
        The network model (e.g., etrago_DE3.network or etrago_DE3.market_model)
    market_zones : str
        The number of German market zones to consider (e.g., "DE2", "DE3", "DE4", "DE5")

    Returns
    -------
    dict
        Electricity trade for each German bidding zone in TWh
    """
    from shapely.geometry import point
    
    # Define shapefile paths based on market_zones
    shapefile_paths = {
        "DE2": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp",
        "DE3": "/home/student/Masterarbeit_Dateien/Shape-Files/BZR_config_12_DE3.shp",
        "DE4": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp",
        "DE5": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp"
    }

    if market_zones not in shapefile_paths:
        raise ValueError("Ungültiger Wert für market_zones. Erlaubt sind: 'DE2', 'DE3', 'DE4', 'DE5'.")

    shapefile_path = shapefile_paths[market_zones]

    # Load shapefile for the specified zones
    zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # Explode multi-polygons
    zones = zones.explode(index_parts=False).reset_index(drop=True)

    # Convert net.buses to a GeoDataFrame
    geometry = [
        Point(xy) for xy in zip(network.buses["x"].values, network.buses["y"].values)
    ]
    geo_buses = gpd.GeoDataFrame(network.buses, geometry=geometry, crs="EPSG:4326")

    # Spatial join to assign zones
    geo_buses = gpd.sjoin(geo_buses, zones[["geometry", "id"]], how="left", predicate="within")

    def assign_zone(row):
        if row["country"] == "DE":
            if pd.notnull(row["id"]):
                return f"DE{int(row['id'])}"
            else:
                return "DE"
        else:
            return row["country"]

    geo_buses["marketzone"] = geo_buses.apply(assign_zone, axis=1)

    # Assign clusters based on the zones
    geo_buses["cluster"] = geo_buses.groupby("marketzone").ngroup()

    # Ensure the marketzone column is added to network.buses only if it doesn't exist
    if "marketzone" not in network.buses.columns:
        network.buses = network.buses.join(geo_buses[["marketzone"]], lsuffix='_left', rsuffix='_right')

    # Initialize result dictionary for each German zone
    result = {f"DE{i}": pd.Series(dtype=float) for i in range(1, int(market_zones[-1]) + 1)}

    # Calculate trade between zones
    for _, line in network.lines.iterrows():
        bus0_zone = network.buses.loc[line["bus0"], "marketzone"]
        bus1_zone = network.buses.loc[line["bus1"], "marketzone"]

        if bus0_zone != bus1_zone:
            flow = (
                network.lines_t.p0[line.name]
                .mul(network.snapshot_weightings.generators)
                .sum()
            ) * 1e-6

            if bus0_zone.startswith("DE"):
                result[bus0_zone][bus1_zone] = result[bus0_zone].get(bus1_zone, 0) + flow
            elif bus1_zone.startswith("DE"):
                result[bus1_zone][bus0_zone] = result[bus1_zone].get(bus0_zone, 0) - flow

    return result


def plot_zone_net_flows(self, market_zones):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import pandas as pd
    import numpy as np
    from shapely.geometry import Point
    from matplotlib.patches import Patch

    # 1. Shapefile-Pfade definieren (Berechnung und Plot getrennt)
    calc_shapefile_paths = {
        "DE2": "/home/student/Masterarbeit_Dateien/Shape-Files/BZR_config_2_DE2.shp",
        "DE3": "/home/student/Masterarbeit_Dateien/Shape-Files/BZR_config_12_DE3.shp",
        "DE4": "/home/student/Masterarbeit_Dateien/Shape-Files/BZR_config_13_DE4.shp",
        "DE5": "/home/student/Masterarbeit_Dateien/Shape-Files/BZR_config_14_DE5.shp"
    }

    plot_shapefile_paths = {
        "DE2": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp",
        "DE3": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_12_DE3.shp",
        "DE4": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp",
        "DE5": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp"
    }

    # 2. Shapefiles laden
    if market_zones != "none":
        calc_zones = gpd.read_file(calc_shapefile_paths[market_zones]).to_crs(epsg=4326)
        calc_zones = calc_zones.explode(index_parts=False).reset_index(drop=True)
        plot_zones = gpd.read_file(plot_shapefile_paths[market_zones]).to_crs(epsg=4326)
        plot_zones = plot_zones.explode(index_parts=False).reset_index(drop=True)
    else:
        calc_zones = None
        plot_zones = None

    # 3. GeoDataFrame für Busse erstellen
    geometry = [Point(xy) for xy in zip(self.buses["x"], self.buses["y"])]
    geo_buses = gpd.GeoDataFrame(self.buses, geometry=geometry, crs="EPSG:4326")

    # 4. Zonen-Zuordnung
    if market_zones != "none":
        geo_buses = gpd.sjoin(geo_buses, calc_zones[["geometry", "id"]], how="left", predicate="within")

        def assign_zone(row):
            if row["country"] == "DE" and pd.notnull(row["id"]):
                return f"DE{int(row['id'])}"
            return "Other"

        geo_buses["marketzone"] = geo_buses.apply(assign_zone, axis=1)
        self.buses["marketzone"] = geo_buses["marketzone"]
    else:
        self.buses["marketzone"] = self.buses["country"].apply(lambda c: "DE" if c == "DE" else "Other")

    # 5. Liste der deutschen Zonen
    if market_zones != "none":
        num_zones = int(market_zones[-1])
        german_zones = [f"DE{i}" for i in range(1, num_zones + 1)]
    else:
        german_zones = ["DE"]

    ac_flows = {zone: 0.0 for zone in german_zones}
    dc_flows = {zone: 0.0 for zone in german_zones}

    # 6. AC-Flüsse berechnen
    for line_idx, line in self.lines.iterrows():
        try:
            bus0_zone = self.buses.loc[line["bus0"], "marketzone"]
            bus1_zone = self.buses.loc[line["bus1"], "marketzone"]
        except KeyError:
            continue

        if bus0_zone.startswith("DE") and bus1_zone == "Other":
            flow = self.lines_t.p0[line_idx].mul(self.snapshot_weightings.generators).sum() * 1e-6
            ac_flows[bus0_zone] += flow
        elif bus0_zone == "Other" and bus1_zone.startswith("DE"):
            flow = self.lines_t.p1[line_idx].mul(self.snapshot_weightings.generators).sum() * 1e-6
            ac_flows[bus1_zone] -= flow
        elif bus0_zone.startswith("DE") and bus1_zone.startswith("DE") and bus0_zone != bus1_zone:
            flow = self.lines_t.p0[line_idx].mul(self.snapshot_weightings.generators).sum() * 1e-6
            ac_flows[bus0_zone] += flow
            ac_flows[bus1_zone] -= flow

    # 7. DC-Flüsse berechnen
    dc_links = self.links[self.links.carrier == "DC"]

    for link_idx, link in dc_links.iterrows():
        try:
            bus0_zone = self.buses.loc[link["bus0"], "marketzone"]
            bus1_zone = self.buses.loc[link["bus1"], "marketzone"]
        except KeyError:
            continue

        if bus0_zone.startswith("DE") and bus1_zone == "Other":
            flow = self.links_t.p0[link_idx].mul(self.snapshot_weightings.generators).sum() * 1e-6
            dc_flows[bus0_zone] += flow
        elif bus0_zone == "Other" and bus1_zone.startswith("DE"):
            flow = self.links_t.p1[link_idx].mul(self.snapshot_weightings.generators).sum() * 1e-6
            dc_flows[bus1_zone] -= flow
        elif bus0_zone.startswith("DE") and bus1_zone.startswith("DE") and bus0_zone != bus1_zone:
            flow = self.links_t.p0[link_idx].mul(self.snapshot_weightings.generators).sum() * 1e-6
            dc_flows[bus0_zone] += flow
            dc_flows[bus1_zone] -= flow

    net_flows = {zone: ac_flows[zone] + dc_flows[zone] for zone in german_zones}

    # 8. Plot erstellen
    fig, ax = plt.subplots(figsize=(12, 8), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-2.5, 16, 46.8, 58], crs=ccrs.PlateCarree())

    # 9. Zonen zeichnen
    if plot_zones is not None:
        colors = plt.cm.tab20(range(len(plot_zones)))
        plot_zones['color'] = [mcolors.rgb2hex(color[:3]) for color in colors]
        plot_zones.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        plot_zones.plot(ax=ax, facecolor=plot_zones['color'], alpha=0.2)

    # 10. Netzwerk zeichnen
    self.plot(ax=ax, line_widths=0, link_widths=0, bus_sizes=0)
    
    # 11. Nettoflüsse textlich einfügen
    if plot_zones is not None:  # Add this check
        for idx, zone in plot_zones.iterrows():
            if market_zones != "none":
                zone_id = f"DE{int(zone['id'])}"
            else:
                zone_id = "DE"
    
            if zone_id in net_flows:
                net_flow = net_flows[zone_id]
                color = 'green' if net_flow >= 0 else 'red'
                center = zone['geometry'].centroid
                ax.text(center.x, center.y, f"{abs(net_flow):.1f} TWh", fontsize=9, ha='center', va='center',
                        color=color, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.8))
    else:
        # Fall für ganz Deutschland
        net_flow = dc_export_de_lu(self)
        german_buses = geo_buses[geo_buses["country"] == "DE"]
        de_center_x = german_buses.geometry.x.mean()
        de_center_y = german_buses.geometry.y.mean()

        color = 'green' if net_flow >= 0 else 'red'
        ax.text(de_center_x, de_center_y, f"{abs(net_flow*1e-6):.1f} TWh",
                   fontsize= 9, ha='center', va='center',
                   color=color, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.8))
        
    
    '''    
    exports_DE2 = ac_export_per_country(etrago_obj) + dc_export_per_country(etrago_obj)
    
    for country, value in exports_DE2.items():
        if country == "DE" or value == 0:
            continue  # Deutschland überspringen
        if value > 0:
            color = 'green'
        elif value < 0:
            color = 'red'
        # Koordinaten des AC-Busses des Landes
        country_buses = etrago_obj.network.buses[etrago_obj.network.buses.country == country]
        if country_buses.empty:
            continue
        lon = country_buses['x'].mean()
        lat = country_buses['y'].mean()
        ax.text(lon, lat, f"{abs(value):.1f} TWh", fontsize=22, color=color,
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.8))

    '''

    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Netto-Export'),
        Patch(facecolor='red', edgecolor='black', label='Netto-Import')
    ]
   

    

    ax.legend(handles= legend_elements, loc='upper right', bbox_to_anchor=(0.42, 0.55), fontsize = 10)   
    # Begrenzung der Kartendarstellung
    #ax.set_extent([5.5, 15.5, 46.5, 55.5], crs=ccrs.PlateCarree())
    

    plt.tight_layout()
    plt.show()
    
    



def ac_export_per_country(self):
    """Calculate electricity exports and imports over AC lines per country

    Returns
    -------
    pd.Series
        Electricity export (if positive) or import (if negative) from DE+LU to each neighboring country in TWh
    """
    # Buses in DE or LU zählen als 'Exportland'
    de_buses = self.network.buses[self.network.buses.country.isin(["DE", "LU"])]
    for_buses = self.network.buses[~self.network.buses.country.isin(["DE", "LU"])]

    result = pd.Series(index=for_buses.country.unique(), dtype=float)

    for c in for_buses.country.unique():
        target_buses = for_buses[for_buses.country == c].index

        exp = self.network.lines[
            (self.network.lines.bus0.isin(de_buses.index)) &
            (self.network.lines.bus1.isin(target_buses))
        ]
        imp = self.network.lines[
            (self.network.lines.bus1.isin(de_buses.index)) &
            (self.network.lines.bus0.isin(target_buses))
        ]

        exp_sum = (
            self.network.lines_t.p0[exp.index]
            .sum(axis=1)
            .mul(self.network.snapshot_weightings.generators)
            .sum()
        )

        imp_sum = (
            self.network.lines_t.p1[imp.index]
            .sum(axis=1)
            .mul(self.network.snapshot_weightings.generators)
            .sum()
        )

        result[c] = (exp_sum + imp_sum) * 1e-6  # in TWh

    return result

def dc_export_per_country(self):
    """Calculate electricity exports and imports over DC lines per country

    Returns
    -------
    float
        Electricity export (if negative: import) from Germany in TWh

    """
    de_buses = self.buses[self.buses.country == "DE"]

    for_buses = self.buses[self.buses.country != "DE"]

    result = pd.Series(index=for_buses.country.unique())

    for c in for_buses.country.unique():
        exp = self.links[
            (self.links.carrier == "DC")
            & (self.links.bus0.isin(de_buses.index))
            & (
                self.links.bus1.isin(
                    for_buses[for_buses.country == c].index
                )
            )
        ]
        imp = self.links[
            (self.links.carrier == "DC")
            & (self.links.bus1.isin(de_buses.index))
            & (
                self.links.bus0.isin(
                    for_buses[for_buses.country == c].index
                )
            )
        ]

        result[c] = (
            self.links_t.p0[exp.index]
            .sum(axis=1)
            .mul(self.snapshot_weightings.generators)
            .sum()
            + self.links_t.p1[imp.index]
            .sum(axis=1)
            .mul(self.snapshot_weightings.generators)
            .sum()
        ) * 1e-6

    return result


def plot_country_exports_per_configuration(market_sq, market_DE2, market_DE3, market_DE4):
    """
    
    Plottet für jedes Nachbarland den Nettoexport Deutschlands (AC + DC) für verschiedene Modellkonfigurationen.
    Nur Länder mit Handelswerten ungleich Null werden dargestellt.
    BE, GB, NO und RU werden explizit ausgeschlossen.

    Parameters
    ----------
    etrago_status_quo : etrago object
    etrago_DE2 : etrago object
    etrago_DE3 : etrago object
    etrago_DE4 : etrago object
    etrago_nodal : etrago object
    """
    def get_exports(market):
        exports = dc_export_per_country(market)
        return exports.drop(labels=['DE', 'LU', 'GB', 'NO', 'RU'], errors='ignore') 

    configs = {
        "Status Quo": market_sq,
        "DE2": market_DE2,
        "DE3": market_DE3,
        "DE4": market_DE4,
        #"Nodal": etrago_nodal
    }

    # Exporte je Konfiguration berechnen
    all_exports = {}
    all_countries = set()

    for name, model in configs.items():
        try:
            exports = get_exports(model)
            # Nur Länder mit Werten ungleich Null behalten
            exports = exports[exports != 0]
            all_exports[name] = exports
            all_countries.update(exports.index)
        except Exception as e:
            print(f"⚠️ Fehler bei {name}: {e}")
            all_exports[name] = pd.Series()

    # Nur Länder, die in mindestens einer Konfiguration vorkommen
    all_countries = sorted(all_countries)
    df = pd.DataFrame(index=all_countries, columns=configs.keys())

    for name, exports in all_exports.items():
        df[name] = exports.reindex(all_countries)

    # Entferne Zeilen, die in allen Spalten Null sind
    df = df.loc[~(df == 0).all(axis=1)]

    # Falls keine Daten übrig sind, abbrechen
    if df.empty:
        print("Keine Handelsdaten zum Plotten verfügbar.")
        return

    # Erstelle eine Farbpalette mit Blautönen
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, len(df.columns)))
    colors = [mcolors.rgb2hex(color) for color in blues]

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(14, 6))

    # Position der Balken
    bar_width = 0.15
    x = np.arange(len(df.index))

    # Plotten der Balken
    for i, (config, color) in enumerate(zip(df.columns, colors)):
        ax.bar(x + i*bar_width,
               df[config],
               width=bar_width,
               label=config,
               color=color)

    # Achsenbeschriftungen und Titel
    ax.set_xticks(x + (len(df.columns)-1)*bar_width/2)
    ax.set_xticklabels(df.index, fontsize = 16)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel("TWh", fontsize = 18)

    ax.set_yticks([-60, -40, -20, 0, 20, 40, 60])  # Hier die gewünschten y-Werte angeben
    ax.set_yticklabels([-60, -40, -20, 0, 20, 40, 60], fontsize=16)  # Hier die gewünschten Beschriftungen angeben

    # Legende
    ax.legend(loc='upper left', fontsize=16)

    # Grid für bessere Lesbarkeit
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Anpassung der x-Achse für bessere Lesbarkeit
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Daten aus dem Bild mit DE4 als Dezimalzahlen
data = {
    'Carrier': [
        'OCGT', 'biomass', 'central_biomass_CHP', 'central_gas_CHP',
        'industrial_biomass_CHP', 'industrial_gas_CHP', 'load_shedding', 'oil', 'others',
        'reservoir', 'run_of_river', 'solar', 'solar_rooftop', 'wind_offshore', 'wind_onshore'
    ],
    'status Quo Markt': [
        0.089135, 1.765038, 1.410584, 11.602485, 8.779279, 1.476175, 
        0.000126, 0.273327, 6.914386, 2.306651, 24.843909, 40.923547, 
        96.037084, 104.153102, 257.918936
    ],
    'DE2 Markt': [
        0.140645, 1.945268, 1.344615, 11.561617, 8.393769, 1.300017,
        0.000536, 0.278178, 6.633533, 2.234094, 24.144473, 39.743945,
        94.509561, 102.844942, 254.693021
    ],
    'DE3 Markt': [
        0.167331, 1.878833, 1.337527, 11.419106, 8.337737, 1.219248,
        0.000579, 0.274658, 6.767534, 2.264720, 24.487233, 40.387639,
        95.402664, 101.588257, 255.931050
    ],
    'DE4 Markt': [
        0.04909348,1.153281, 1.313722, 11.28458,
        8.201612, 12.55294, 0.0, 0.1131882, 2.3444,
        7.008, 25.09967, 41.36727, 96.73906, 111.0096, 266.8184
    ]
}

# Erstellen Sie einen DataFrame
df = pd.DataFrame(data)

# Gruppieren der Biomass-Carrier
biomass_carriers = ['biomass', 'central_biomass_CHP', 'industrial_biomass_CHP']
biomass_row = {
    'Carrier': 'biomass',
    'status Quo Markt': df[df['Carrier'].isin(biomass_carriers)]['status Quo Markt'].sum(),
    'DE2 Markt': df[df['Carrier'].isin(biomass_carriers)]['DE2 Markt'].sum(),
    'DE3 Markt': df[df['Carrier'].isin(biomass_carriers)]['DE3 Markt'].sum(),
    'DE4 Markt': df[df['Carrier'].isin(biomass_carriers)]['DE4 Markt'].sum()
}

# Gruppieren der Gas-Carrier
gas_carriers = ['OCGT', 'central_gas_CHP', 'industrial_gas_CHP']
gas_row = {
    'Carrier': 'natural gas',
    'status Quo Markt': df[df['Carrier'].isin(gas_carriers)]['status Quo Markt'].sum(),
    'DE2 Markt': df[df['Carrier'].isin(gas_carriers)]['DE2 Markt'].sum(),
    'DE3 Markt': df[df['Carrier'].isin(gas_carriers)]['DE3 Markt'].sum(),
    'DE4 Markt': df[df['Carrier'].isin(gas_carriers)]['DE4 Markt'].sum()
}

# Wind Carrier einzeln behalten
wind_offshore_row = df[df['Carrier'] == 'wind_offshore'].iloc[0].to_dict()
wind_onshore_row = df[df['Carrier'] == 'wind_onshore'].iloc[0].to_dict()

# Erstellen Sie eine Liste mit den neuen Zeilen
new_rows = [biomass_row, gas_row, wind_offshore_row, wind_onshore_row]

# Others: Alle anderen Carrier zusammenfassen
others_carriers = [carrier for carrier in df['Carrier'] if carrier not in biomass_carriers + gas_carriers + ['wind_offshore', 'wind_onshore']]
others_row = {
    'Carrier': 'others',
    'status Quo Markt': df[df['Carrier'].isin(others_carriers)]['status Quo Markt'].sum(),
    'DE2 Markt': df[df['Carrier'].isin(others_carriers)]['DE2 Markt'].sum(),
    'DE3 Markt': df[df['Carrier'].isin(others_carriers)]['DE3 Markt'].sum(),
    'DE4 Markt': df[df['Carrier'].isin(others_carriers)]['DE4 Markt'].sum()
}
new_rows.append(others_row)

# Erstellen Sie einen neuen DataFrame mit den gruppierten Daten
grouped_df = pd.DataFrame(new_rows)

# Multiplizieren Sie alle Werte mit 1000 für GWh
for col in ['status Quo Markt', 'DE2 Markt', 'DE3 Markt', 'DE4 Markt']:
    grouped_df[col] = grouped_df[col] * 1000

# Berechnen Sie die Differenzen
grouped_df['DE2'] = grouped_df['DE2 Markt'] - grouped_df['status Quo Markt']
grouped_df['DE3'] = grouped_df['DE3 Markt'] - grouped_df['status Quo Markt']
grouped_df['DE4'] = grouped_df['DE4 Markt'] - grouped_df['status Quo Markt']

# Weiche Blautöne für die Säulen
colors = ['#6BAFDB', '#3D8BD9', '#0E67B9']  # Weiche Blautöne

# Erstellen Sie das Säulendiagramm
fig, ax = plt.subplots(figsize=(12, 7))

# Breite der Säulen
bar_width = 0.2

# Position der Säulen auf der x-Achse
r1 = np.arange(len(grouped_df['Carrier']))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotten der Säulen mit weichen Blautönen
ax.bar(r1, grouped_df['DE2'], color=colors[0], width=bar_width, edgecolor='grey', label='DE2')
ax.bar(r2, grouped_df['DE3'], color=colors[1], width=bar_width, edgecolor='grey', label='DE3')
ax.bar(r3, grouped_df['DE4'], color=colors[2], width=bar_width, edgecolor='grey', label='DE4')

# Beschriftungen und Titel
ax.set_ylabel('GWh', fontsize = 17)
ax.set_xticks([x + bar_width for x in range(len(grouped_df['Carrier']))])
ax.set_xticklabels(grouped_df['Carrier'], rotation=0, ha='center', fontsize = 17)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.tick_params(axis='y', labelsize=14)  # y-Achsen-Beschriftungen

# Legende
ax.legend(prop={'size': 12})

# Anzeigen des Diagramms
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Daten aus dem Bild mit DE4 als Dezimalzahlen
data = {
    'Carrier': [
        'H2_to_power', 'OCGT', 'biomass', 'central_biomass_CHP', 'central_gas_CHP',
        'industrial_biomass_CHP', 'industrial_gas_CHP', 'load_shedding', 'oil', 'others',
        'reservoir', 'run_of_river', 'solar', 'solar_rooftop', 'wind_offshore', 'wind_onshore'
    ],
    'status Quo Markt': [
        0.000113, 0.012187, 1.212365, 1.384359, 11.592796,
        8.597678, 13.744073, 0.000000, 0.115418, 2.344400,
        7.008000, 25.099675, 41.376271, 96.739059, 113.279031, 266.186084
    ],
    'DE2 Markt': [
        0.003771, 0.005853, 1.164415, 1.323526, 11.500313,
        8.249843, 12.656710, 0.000000, 0.113463, 2.344400,
        7.008000, 25.099675, 41.376271, 96.739059, 104.915332, 265.037219
    ],
    'DE3 Markt': [
        0.002697, 0.007260, 1.155810, 1.315185, 11.458404,
        8.188919, 12.666167, 0.000000, 0.114308, 2.344400,
        7.008000, 25.099675, 41.376271, 96.739059, 104.799975, 265.495740
    ],
    'DE4 Markt': [
        0.04909348, 0.0, 1.153281, 1.313722, 11.28458,
        8.201612, 12.55294, 0.0, 0.1131882, 2.3444,
        7.008, 25.09967, 41.36727, 96.73906, 111.0096, 266.8184
    ]
}

# Erstellen Sie einen DataFrame
df = pd.DataFrame(data)

# Gruppieren der Biomass-Carrier
biomass_carriers = ['biomass', 'central_biomass_CHP', 'industrial_biomass_CHP']
biomass_row = {
    'Carrier': 'biomass',
    'status Quo Markt': df[df['Carrier'].isin(biomass_carriers)]['status Quo Markt'].sum(),
    'DE2 Markt': df[df['Carrier'].isin(biomass_carriers)]['DE2 Markt'].sum(),
    'DE3 Markt': df[df['Carrier'].isin(biomass_carriers)]['DE3 Markt'].sum(),
    'DE4 Markt': df[df['Carrier'].isin(biomass_carriers)]['DE4 Markt'].sum()
}

# Gruppieren der Gas-Carrier
gas_carriers = ['OCGT', 'central_gas_CHP', 'industrial_gas_CHP']
gas_row = {
    'Carrier': 'natural gas',
    'status Quo Markt': df[df['Carrier'].isin(gas_carriers)]['status Quo Markt'].sum(),
    'DE2 Markt': df[df['Carrier'].isin(gas_carriers)]['DE2 Markt'].sum(),
    'DE3 Markt': df[df['Carrier'].isin(gas_carriers)]['DE3 Markt'].sum(),
    'DE4 Markt': df[df['Carrier'].isin(gas_carriers)]['DE4 Markt'].sum()
}

# Wind Carrier einzeln behalten
wind_offshore_row = df[df['Carrier'] == 'wind_offshore'].iloc[0].to_dict()
wind_onshore_row = df[df['Carrier'] == 'wind_onshore'].iloc[0].to_dict()

# Erstellen Sie eine Liste mit den neuen Zeilen
new_rows = [biomass_row, gas_row, wind_offshore_row, wind_onshore_row]

# Others: Alle anderen Carrier zusammenfassen
others_carriers = [carrier for carrier in df['Carrier'] if carrier not in biomass_carriers + gas_carriers + ['wind_offshore', 'wind_onshore']]
others_row = {
    'Carrier': 'others',
    'status Quo Markt': df[df['Carrier'].isin(others_carriers)]['status Quo Markt'].sum(),
    'DE2 Markt': df[df['Carrier'].isin(others_carriers)]['DE2 Markt'].sum(),
    'DE3 Markt': df[df['Carrier'].isin(others_carriers)]['DE3 Markt'].sum(),
    'DE4 Markt': df[df['Carrier'].isin(others_carriers)]['DE4 Markt'].sum()
}
new_rows.append(others_row)

# Erstellen Sie einen neuen DataFrame mit den gruppierten Daten
grouped_df = pd.DataFrame(new_rows)

# Multiplizieren Sie alle Werte mit 1000 für GWh
for col in ['status Quo Markt', 'DE2 Markt', 'DE3 Markt', 'DE4 Markt']:
    grouped_df[col] = grouped_df[col] 

# Berechnen Sie die Differenzen
grouped_df['DE2'] = grouped_df['DE2 Markt'] - grouped_df['status Quo Markt']
grouped_df['DE3'] = grouped_df['DE3 Markt'] - grouped_df['status Quo Markt']
grouped_df['DE4'] = grouped_df['DE4 Markt'] - grouped_df['status Quo Markt']

# Weiche Blautöne für die Säulen
colors = ['#6BAFDB', '#3D8BD9', '#0E67B9']  # Weiche Blautöne

# Erstellen Sie das Säulendiagramm
fig, ax = plt.subplots(figsize=(12, 7))

# Breite der Säulen
bar_width = 0.2

# Position der Säulen auf der x-Achse
r1 = np.arange(len(grouped_df['Carrier']))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotten der Säulen mit weichen Blautönen
ax.bar(r1, grouped_df['DE2'], color=colors[0], width=bar_width, edgecolor='grey', label='DE2 Diff')
ax.bar(r2, grouped_df['DE3'], color=colors[1], width=bar_width, edgecolor='grey', label='DE3 Diff')
ax.bar(r3, grouped_df['DE4'], color=colors[2], width=bar_width, edgecolor='grey', label='DE4 Diff')

# Beschriftungen und Titel
ax.set_ylabel('Differenz zum Status Quo Szenario [TWh]', fontsize = 17)
ax.set_xticks([x + bar_width for x in range(len(grouped_df['Carrier']))])
ax.set_xticklabels(grouped_df['Carrier'], rotation=0, ha='center', fontsize = 17)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.tick_params(axis='y', labelsize=14)  # y-Achsen-Beschriftungen

# Legende
ax.legend(prop={'size': 17})

# Anzeigen des Diagramms
plt.tight_layout()
plt.show()

def translink_capacity_zubau(self):
    
    trans_links = self.market_model.links[self.market_model.links.carrier == "DC"]
    
    DK = trans_links[trans_links.country == "DK"].p_nom.sum() - trans_links[trans_links.country == "DK"].p_nom_min.sum()
    SE = trans_links[trans_links.country == "SE"].p_nom.sum() - trans_links[trans_links.country == "SE"].p_nom_min.sum()
    PL = trans_links[trans_links.country == "PL"].p_nom.sum() - trans_links[trans_links.country == "PL"].p_nom_min.sum()
    CZ = trans_links[trans_links.country == "CZ"].p_nom.sum() - trans_links[trans_links.country == "CZ"].p_nom_min.sum()
    AT = trans_links[trans_links.country == "AT"].p_nom.sum() - trans_links[trans_links.country == "AT"].p_nom_min.sum()
    CH = trans_links[trans_links.country == "CH"].p_nom.sum() - trans_links[trans_links.country == "CH"].p_nom_min.sum()
    FR = trans_links[trans_links.country == "FR"].p_nom.sum() - trans_links[trans_links.country == "FR"].p_nom_min.sum()
    BE = trans_links[trans_links.country == "BE"].p_nom.sum() - trans_links[trans_links.country == "BE"].p_nom_min.sum()
    NL = trans_links[trans_links.country == "NL"].p_nom.sum() - trans_links[trans_links.country == "NL"].p_nom_min.sum()
    
    print(f" DK: {DK}\n SE: {SE}\n PL: {PL}\n CZ: {CZ}\n AT: {AT}\n CH: {CH}\n FR: {FR}\n BE: {BE}\n NL: {NL}")

    return
"""




def plot_dispatch_difference_by_bus(
    networkA,
    networkB,
    carrier="wind_offshore",
    market_zones="DE3",
    filename=None,
    base_scaling=0.5  # Nutzerdefinierbare Skalierungsbasis
):
    """
    Plottet die ungewichtete Dispatch-Differenz (Summe über 8760h) eines Carriers pro Bus zwischen zwei Netzwerken.
    Farben: grün = höherer Dispatch in A, rot = niedrigerer Dispatch in A.
    Die Kreisgröße wird automatisch auf die maximal auftretende Differenz skaliert.
    """
    import matplotlib.patches as mpatches
    import pandas as pd


    # A: Generatoren mit diesem Carrier aus beiden Netzwerken
    gensA = networkA.generators.query("carrier == @carrier")
    gensB = networkB.generators.query("carrier == @carrier")

    # Nur Generatoren mit gleichem Index in beiden Netzen vergleichen
    common_index = gensA.index.intersection(gensB.index)
    if common_index.empty:
        print("⚠️ Keine gemeinsamen Generatoren gefunden!")
        return

    # Dispatch-Zeitreihen
    pA = networkA.generators_t.p[common_index]
    pB = networkB.generators_t.p[common_index]

    # Differenz über 8760 Stunden summieren
    diff = pA.sum(axis=0) - pB.sum(axis=0)  # Series mit Generator-Index

    # Mapping Generator → Bus
    bus_map = gensA.loc[common_index, "bus"]
    dispatch_grouped = diff.groupby(bus_map).sum()

    # Farben nach Vorzeichen
    colors_buses = {
    bus: mcolors.to_rgba("green", alpha=0.2) if val > 0 else mcolors.to_rgba("red", alpha=0.2)
    for bus, val in dispatch_grouped.items()
    }
    # Absolutwerte für Kreisgrößen
    dispatch_abs = dispatch_grouped.abs()

    # ❗ automatische Skalierung
    scaling = 1 / dispatch_abs.max() * base_scaling

    # Optional: Hintergrundkarte mit Marktzonen
    if market_zones in ["DE2", "DE3", "DE4", "DE5"]:
        shapefile_path = f"/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_*_{market_zones}.shp"
        shapefile_path = shapefile_path.replace("*", {"DE2": "2", "DE3": "12", "DE4": "13", "DE5": "14"}[market_zones])
        zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    else:
        zones = None

    # Plot vorbereiten
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, subplot_kw={"projection": ccrs.PlateCarree()})

    if zones is not None:
        # Farbpalette und Farben zuweisen
        colors = plt.cm.tab20(range(len(zones)))
        zones['color'] = [mcolors.rgb2hex(color[:3]) for color in colors]

        # Zonen plotten
        zones.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        zones.plot(ax=ax, facecolor=zones['color'], alpha=0.3)
        

    # Netzelemente plotten (aus networkA)
    networkA.plot(
        geomap=True,
        bus_sizes=dispatch_abs * scaling,
        bus_colors=colors_buses,
        line_widths=0,
        link_widths= 0,
        margin=0.01,
        ax=ax
    )

    ax.set_title(f"Dispatch-Differenz {carrier} pro Bus")
    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
    
    # Farben-Legende
    patch_green = mpatches.Patch(color='green', alpha=0.5, label='Reduktion Dispatch')
    patch_red = mpatches.Patch(color='red', alpha=0.5, label='Erhöhung Dispatch')
    
    #ax.legend(handles=[patch_green, patch_red], loc='upper left')
    
    
    if filename:
        plt.savefig(f"{filename}.png", bbox_inches="tight")
        print(f"✅ Plot gespeichert unter {filename}.png")
    else:
        plt.show()

def dc_export_de_lu(self):
    """
    Calculate electricity exports and imports over DC lines

    Returns
    -------
    float
        Electricity export (if negative: import) from Germany
    """

    
    network = self
    de_buses = network.buses[
        (network.buses.country == "DE") |
        (network.buses.country == "LU")
    ]
    
    for_buses = network.buses[
        ~network.buses.country.isin(["DE", "LU"])
    ]
    
    exp = network.links[
        (network.links.carrier == "DC")
        & (network.links.bus0.isin(de_buses.index))
        & (network.links.bus1.isin(for_buses.index))
    ]
    imp = network.links[
        (network.links.carrier == "DC")
        & (network.links.bus1.isin(de_buses.index))
        & (network.links.bus0.isin(for_buses.index))
    ]
    return (
        network.links_t.p0[exp.index]
        .sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
        + network.links_t.p1[imp.index]
        .sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
    )


def plot_marketzone_clustering(self, market_zones, scaling=0.0001):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import numpy as np

    # Optionales Einlesen des Shapefiles je nach market_zone
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

    fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

    # Marktzonen zeichnen
    if shapefile_path:
        zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        colors = plt.cm.tab20(range(len(zones)))
        zones['color'] = [mcolors.rgb2hex(c[:3]) for c in colors]
        zones.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        zones.plot(ax=ax, facecolor=zones['color'], alpha=0.3)

    # Offshore-Abregelung in MWh (negativ), umgerechnet in GWh
    offshore_buses = ramps_down(self, "wind_offshore")
    bus_values_gwh = -offshore_buses / 1000
    bus_values_gwh = bus_values_gwh.reindex(self.buses.index).fillna(0)

    # Farbskala von min bis 0
    norm = mcolors.Normalize(vmin=-1160, vmax=0.0)
    cmap = plt.cm.viridis
    bus_colors = bus_values_gwh.map(lambda x: cmap(norm(x)))

    # Jetzt die Netzdarstellung mit Buses als Punkte
    self.plot(
        ax=ax,
        link_widths=0,
        bus_sizes=bus_values_gwh.abs() * scaling,
        bus_colors=bus_colors,
        line_widths=0.5,
        line_colors="grey"
    )
    
    # ScalarMappable definieren
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Colorbar mit größeren Abmessungen
    cbar = fig.colorbar(
        sm,
        ax=ax,                 # oder ax=ax, je nach deinem Aufruf
        orientation='vertical',
        pad=0.02,
        fraction=0.1,          # Breite als Anteil der Höhe der Achse
        shrink=0.9,             # Länge als Anteil der Höhe der Achse (1.0 = voll)
        aspect=20               # Verhältnis von Länge zu Breite; größer = dünner
    )
    
    # Beschriftung und Tick‑Labels größer machen
    cbar.set_label("Abregelung (GWh)", rotation=90, labelpad=10, fontsize=18)
    cbar.ax.tick_params(labelsize=17)
    
    # Ticks setzen wie gehabt
    ticks = np.linspace(bus_values_gwh.min(), 0.0, num=5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

   
    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
    ax.axis('off')
    plt.tight_layout()
    plt.show()

