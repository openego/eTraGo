from etrago import Etrago
import pypsa
from sqlalchemy import create_engine
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

'''focus_region = ["Region Hannover"]
con = create_engine("postgresql+psycopg2://egon:data@127.0.0.1:59732/egon-data")
query = "SELECT gen, geometry FROM boundaries.vg250_krs"
focus_gdf = gpd.read_postgis(query, con, geom_col="geometry")
focus_gdf = focus_gdf[focus_gdf["gen"].isin(focus_region)]'''

'''buses_orig = e.busmap["orig_network"].buses[e.busmap["orig_network"].buses.carrier=="AC"]
buses_orig["geom"] =  buses_orig.apply(lambda x: Point(x["x"], x["y"]), axis=1)
buses_orig = gpd.GeoDataFrame(buses_orig, geometry="geom", crs=4326)
buses_area_orig = gpd.clip(buses_orig, focus_gdf)
ac_buses_original = len(buses_orig)
ac_buses_original_area = len(buses_area_orig)'''

# original network:
# 13131  AC buses
# 13042 AC buses in DE
# 104 buses in focus region
# 56 davon ohne load and generation (ohne load shedding)

focus_gdf = gpd.read_file("/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/focus-region/hannover.gpkg")
focus_gdf = focus_gdf.to_crs(epsg=4326)

### only clustering - local tests #############################################

path = '/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming/tests/sig-100-neu-cluster/'
calcs= [300] # [50, 100, 150, 200, 250, 300]

def plot_clusters_row(path, calcs, focus_gdf):
    
    for i in range(0, len(calcs)):
    
        npath = path+str(calcs[i])
        e = Etrago(csv_folder_name=npath)
        e.plot_clusters(focus_gdf=focus_gdf, transmission_lines=True, save_path=npath+'/clusters.png')
        
#plot_clusters_row(path, calcs, focus_gdf)

def plot_weight_distr(weights):
    
    # Bus-Spalte entfernen
    weight_columns = [col for col in weights.columns if col != "bus"]

    # Histogramme überlagern
    plt.figure(figsize=(8,5))
    #for col in weight_columns:
        #plt.hist(weights[col], bins=30, alpha=0.5, label=col)
    weights[weight_columns].plot(kind="box")

    plt.xlabel("Funktion für Fokusregion")
    plt.ylabel("finale Gewichtung")
    plt.title("Vergleich der Gewichtungsverteilungen")
    plt.legend()
    plt.tight_layout()
    plt.show()

path = '/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming/tests/sig-100-neu-cluster/weighting_tests.csv'
weights = pd.read_csv(path)
#weights.set_index('Bus', inplace=True)

#plot_weight_distr(weights)

def network_weights(n, weights, path):
    weight_columns = [col for col in weights.columns if col != "bus"]
    
    for col in weight_columns:
        descr = '1-durch-dist' if col == '1/dist' else col

        # Werte übernehmen und Index als String
        values = weights[col].copy()
        values.index = values.index.astype(str)

        # Nur AC-Busse auswählen
        ac_buses = n.buses[n.buses['carrier'] == 'AC'].index.astype(str)

        # Bus-Werte vorbereiten: AC-Busse behalten, andere NaN
        bus_values = []
        for bus in n.buses.index.astype(str):
            if bus in values.index:
                bus_values.append(values[bus])
            else:
                bus_values.append(np.nan)
        bus_values = np.array(bus_values)

        # Größen: nur AC-Busse sichtbar
        bus_sizes = np.array([0.001 if bus in ac_buses else 0 for bus in n.buses.index.astype(str)])

        # GeoAxes
        fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={"projection": ccrs.PlateCarree()})

        n.plot(
            ax=ax,
            bus_colors=bus_values,
            bus_sizes=bus_sizes,
            line_widths=0,
            link_widths=0,
            bus_cmap="viridis",
            title=f"Gewichtung der Knoten: {col}"
        )

        ax.set_extent([5, 16, 47, 56], crs=ccrs.PlateCarree())
        ax.coastlines(color="gray", linewidth=0.5)

        # Farblegende
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=np.nanmin(bus_values), vmax=np.nanmax(bus_values)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label('Gewichtung', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{path}{descr}.png", dpi=200)
        plt.close(fig)
        
network = pypsa.Network('/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming/tests/pypsa-original')
path = '/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming/tests/sig-100-neu-cluster'
weights = pd.read_csv(path+'/weighting_tests.csv')        
#weights.set_index('Bus', inplace=True)
#network_weights(network, weights, path+'/weighting_analysis/')

### optimization calcs server #################################################

def get_results(path, calc, descr, focus_gdf=focus_gdf):
    
    results = pd.DataFrame(columns = ["n_buses", "n_buses_de", "n_buses_area", "total_system_costs", "marginal_costs", "investment_costs", 
                                      "ac_grid_investment_costs", "dc_grid_investment_costs", "other_link_expansion_costs",
                                      "battery_expansion_costs", "H2_overground_expansion_costs"], index = calc)
    
    for r in calc:
        
        e = Etrago(csv_folder_name=path + "/" + r)
        e.calc_results()
        
        e.plot_clusters(focus_gdf, transmission_lines=True, save_path="/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/plot_clusters/"+descr+r)
        
        buses = e.network.buses[e.network.buses.carrier=="AC"]
        buses["geom"] =  buses.apply(lambda x: Point(x["x"], x["y"]), axis=1)
        buses = gpd.GeoDataFrame(buses, geometry="geom", crs=4326)
        buses_area = gpd.clip(buses, focus_gdf)
        results.at[r, "n_buses"] = len(buses)
        results.at[r, "n_buses_de"] = len(e.network.buses[e.network.buses.carrier=='AC'][e.network.buses.country=='DE'])
        results.at[r, "n_buses_area"] = len(buses_area)
        
        results.at[r, "total_system_costs"] = e.results.at["annual system costs", "value"] /1000000000
        results.at[r, "marginal_costs"] = e.results.at["annual marginal costs", "value"] /1000000000
        results.at[r, "investment_costs"] = e.results.at["annual investment costs", "value"] /1000000000
        results.at[r, "ac_grid_investment_costs"] = e.results.at["annual ac grid investment costs", "value"] /1000000
        results.at[r, "dc_grid_investment_costs"] = e.results.at["annual dc grid investment costs", "value"] /1000000
        
        links = e.network.links[e.network.links.p_nom_extendable][e.network.links.carrier!='DC']
        results.at[r, "other_link_expansion_costs"] = ((links.p_nom_opt - links.p_nom_min)*links.capital_cost).sum() /1000000
        
        results.at[r, "battery_expansion_costs"] = e.results.at["annual electrical storage investment costs", "value"] /1000000
        results.at[r, "H2_overground_expansion_costs"] = e.results.at["annual store investment costs", "value"] /1000000    
    
    '''ax1 = results.drop(labels="orig")["area/total"].plot(title="Buses in interest area")
    ax1 = results.drop(labels="orig")["n_buses"].plot()
    ax1 = results.drop(labels="orig")["n_buses_area"].plot()
    ax1.legend()
    ax1.figure.savefig(path+"buses_in_interest_area.png", dpi=300, bbox_inches="tight")
    
    ax = results.total_system_costs.dropna().plot(title="System Costs")
    ax = results.marginal_costs.dropna().plot()
    ax = results.ac_grid_investment_costs.dropna().plot()
    ax = results.dc_grid_investment_costs.dropna().plot()
    ax.set_ylabel("Cost [€]")
    ax.legend()
    ax.figure.savefig(path+"system_costs.png", dpi=300, bbox_inches="tight")'''
    
    return e, results

def get_results_de(path, calc, focus_gdf=focus_gdf):
    
    def marginal_costs_de(network):
        
        buses_de_all = network.buses[network.buses.country=='DE']
        
        gen = network.generators[e.network.generators.bus.isin(buses_de_all.index)]
        gen_t = network.generators_t.p[gen.index]
        gen_res = (
            gen_t.mul(
                network.snapshot_weightings.objective, axis=0
            )
            .mul(
                pypsa.descriptors.get_switchable_as_dense(
                    network, "Generator", "marginal_cost"
                )
            )
            .sum()
            .sum()
        )
        
        links_de = network.links[(e.network.links.bus0.isin(buses_de_all.index))&(e.network.links.bus1.isin(buses_de_all.index))]
        links = links_de[links_de.carrier!='DC']
        links_p0 = network.links_t.p0[links.index]
        link_res = (
            abs(links_p0)
            .mul(network.snapshot_weightings.objective, axis=0)
            .mul(
                pypsa.descriptors.get_switchable_as_dense(
                    network, "Link", "marginal_cost"
                )
            )
            .sum()
            .sum()
        )
        
        bat = network.storage_units[network.storage_units.bus.isin(buses_de_all.index)][network.storage_units.p_nom_extendable]
        bat_t = network.storage_units_t.p[bat.index]
        stor_res = (
            bat_t.mul(
                network.snapshot_weightings.objective, axis=0
            )
            .sum(axis=0)
            .mul(network.storage_units.marginal_cost)
            .sum()
        )
        costs = gen_res + link_res + stor_res
        
        return costs
    
    results = pd.DataFrame(columns = ["n_buses", "n_buses_de", "n_buses_area", "total_system_costs", "marginal_costs", "investment_costs", 
                                      "ac_grid_investment_costs", "dc_grid_investment_costs", "other_link_expansion_costs",
                                      "battery_expansion_costs", "H2_overground_expansion_costs"], index = calc)
    
    for r in calc:
        
        e = Etrago(csv_folder_name=path + "/" + r)
        e.calc_results()
        
        buses = e.network.buses[e.network.buses.carrier=="AC"]
        buses_de = buses[buses.country=='DE']
        buses_de_all = e.network.buses[e.network.buses.country=='DE']
        
        buses["geom"] =  buses.apply(lambda x: Point(x["x"], x["y"]), axis=1)
        buses = gpd.GeoDataFrame(buses, geometry="geom", crs=4326)
        buses_area = gpd.clip(buses, focus_gdf)
        results.at[r, "n_buses"] = len(buses)
        results.at[r, "n_buses_de"] = len(buses_de)
        results.at[r, "n_buses_area"] = len(buses_area)
        
        results.at[r, "marginal_costs"] = marginal_costs_de(e.network) /1000000000
        
        lines_de = e.network.lines[(e.network.lines.bus0.isin(buses_de.index))&(e.network.lines.bus1.isin(buses_de.index))]
        lines_de_ext = lines_de[lines_de.s_nom_extendable]
        results.at[r, "ac_grid_investment_costs"] = ((lines_de_ext.s_nom_opt-lines_de_ext.s_nom_min)*lines_de_ext.capital_cost).sum()/1000000
        results.at[r, "dc_grid_investment_costs"] = 0
        
        links_de = e.network.links[(e.network.links.bus0.isin(buses_de_all.index))&(e.network.links.bus1.isin(buses_de_all.index))]
        other_links = links_de[links_de.carrier!='DC']
        results.at[r, "other_link_expansion_costs"] = ((other_links.p_nom_opt - other_links.p_nom_min)*other_links.capital_cost).sum()/1000000
    
        bat = e.network.storage_units[e.network.storage_units.bus.isin(buses_de.index)][e.network.storage_units.p_nom_extendable]
        results.at[r, "battery_expansion_costs"] = ((bat.p_nom_opt-bat.p_nom_min)*bat.capital_cost).sum() /1000000
        
        sto_links = links_de[links_de.carrier=='power_to_H2']
        sto = e.network.stores[e.network.stores.bus.isin(sto_links.bus1)][e.network.stores.e_nom_extendable]
        results.at[r, "H2_overground_expansion_costs"] = ((sto.e_nom_opt-sto.e_nom_min)*sto.capital_cost).sum()/1000000
        
        results.at[r, "investment_costs"] = (results.at[r, "ac_grid_investment_costs"] + results.at[r, "other_link_expansion_costs"] + results.at[r, "battery_expansion_costs"] + results.at[r, "H2_overground_expansion_costs"])/1000
        results.at[r, "total_system_costs"] = (results.at[r, "investment_costs"] + results.at[r, "marginal_costs"])
    
    return results

def get_results_focus(path, calc, focus_gdf=focus_gdf):
    
    def marginal_costs_focus(network):
        
        buses=network.buses        
        buses["geom"] =  buses.apply(lambda x: Point(x["x"], x["y"]), axis=1)
        buses = gpd.GeoDataFrame(buses, geometry="geom", crs=4326)
        buses = gpd.clip(buses, focus_gdf)
        
        gen = network.generators[e.network.generators.bus.isin(buses.index)]
        gen_t = network.generators_t.p[gen.index]
        gen_res = (
            gen_t.mul(
                network.snapshot_weightings.objective, axis=0
            )
            .mul(
                pypsa.descriptors.get_switchable_as_dense(
                    network, "Generator", "marginal_cost"
                )
            )
            .sum()
            .sum()
        )
        
        links = network.links[(e.network.links.bus0.isin(buses.index))&(e.network.links.bus1.isin(buses.index))]
        links = links[links.carrier!='DC']
        links_p0 = network.links_t.p0[links.index]
        link_res = (
            abs(links_p0)
            .mul(network.snapshot_weightings.objective, axis=0)
            .mul(
                pypsa.descriptors.get_switchable_as_dense(
                    network, "Link", "marginal_cost"
                )
            )
            .sum()
            .sum()
        )
        
        bat = network.storage_units[network.storage_units.bus.isin(buses.index)][network.storage_units.p_nom_extendable]
        bat_t = network.storage_units_t.p[bat.index]
        stor_res = (
            bat_t.mul(
                network.snapshot_weightings.objective, axis=0
            )
            .sum(axis=0)
            .mul(network.storage_units.marginal_cost)
            .sum()
        )
        costs = gen_res + link_res + stor_res
        
        return costs
    
    results = pd.DataFrame(columns = ["n_buses", "n_buses_de", "n_buses_area", "total_system_costs", "marginal_costs", "investment_costs", 
                                      "ac_grid_investment_costs", "dc_grid_investment_costs", "other_link_expansion_costs",
                                      "battery_expansion_costs", "H2_overground_expansion_costs"], index = calc)
    
    for r in calc:
        
        e = Etrago(csv_folder_name=path + "/" + r)
        e.calc_results()
        
        buses = e.network.buses
        buses["geom"] =  buses.apply(lambda x: Point(x["x"], x["y"]), axis=1)
        buses = gpd.GeoDataFrame(buses, geometry="geom", crs=4326)
        buses = gpd.clip(buses, focus_gdf)
        
        results.at[r, "n_buses"] = len(e.network.buses[e.network.buses.carrier=='AC'])
        results.at[r, "n_buses_de"] = len(e.network.buses[e.network.buses.carrier=='AC'][e.network.buses.country=='DE'])
        results.at[r, "n_buses_area"] = len(buses[buses.carrier=='AC'])
        
        results.at[r, "marginal_costs"] = marginal_costs_focus(e.network) /1000000000
        
        lines = e.network.lines[(e.network.lines.bus0.isin(buses.index))&(e.network.lines.bus1.isin(buses.index))]
        lines_ext = lines[lines.s_nom_extendable]
        results.at[r, "ac_grid_investment_costs"] = ((lines_ext.s_nom_opt-lines_ext.s_nom_min)*lines_ext.capital_cost).sum() /1000000
        results.at[r, "dc_grid_investment_costs"] = 0
        
        links = e.network.links[(e.network.links.bus0.isin(buses.index))&(e.network.links.bus1.isin(buses.index))]
        other_links = links[links.carrier!='DC']
        results.at[r, "other_link_expansion_costs"] = ((other_links.p_nom_opt - other_links.p_nom_min)*other_links.capital_cost).sum() /1000000
    
        bat = e.network.storage_units[e.network.storage_units.bus.isin(buses.index)][e.network.storage_units.p_nom_extendable]
        results.at[r, "battery_expansion_costs"] = ((bat.p_nom_opt-bat.p_nom_min)*bat.capital_cost).sum() /1000000
        
        sto_links = links[links.carrier=='power_to_H2']
        sto = e.network.stores[e.network.stores.bus.isin(sto_links.bus1)][e.network.stores.e_nom_extendable]
        results.at[r, "H2_overground_expansion_costs"] = ((sto.e_nom_opt-sto.e_nom_min)*sto.capital_cost).sum() /1000000
        
        results.at[r, "investment_costs"] = (results.at[r, "ac_grid_investment_costs"] + results.at[r, "other_link_expansion_costs"] + results.at[r, "battery_expansion_costs"] + results.at[r, "H2_overground_expansion_costs"]) /1000
        results.at[r, "total_system_costs"] = (results.at[r, "investment_costs"] + results.at[r, "marginal_costs"])
    
    return results

def plot_no_buses(results1, label1, results2, label2, results3, label3, path):
    
    # Plot
    plt.figure(figsize=(8, 5))
    
    plt.plot(results1.index, results1["n_buses_area"], 'x--', label=label1)
    common_index = results1.index.intersection(results2.index)
    plt.plot(common_index, results2.loc[common_index, "n_buses_area"], 'x--', label=label2)
    common_index = results1.index.intersection(results3.index)
    plt.plot(common_index, results3.loc[common_index, "n_buses_area"], 'x--', label=label3)
    
    plt.axhline(y=48, color='red', linestyle='-', linewidth=1.5, label='focus buses with cap.')
    
    plt.ylabel('Anzahl der Knoten in der Fokusregion')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_costs(results1, label1, results2, label2, results3, label3, path):
    
    # Drei Diagramme untereinander, gemeinsame x-Achse
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    plt.suptitle("Annualisierte Kosten in Milliarden Euro")

    # Plot 1
    ax1.plot(results1.index, results1.total_system_costs, marker='x', linestyle='--', label=label1)
    common_index = results1.index.intersection(results2.index)
    ax1.plot(common_index, results2.loc[common_index, "total_system_costs"], marker='x', linestyle='--', label=label2)
    common_index = results1.index.intersection(results3.index)
    ax1.plot(common_index, results3.loc[common_index, "total_system_costs"], 'x--', label=label3)
    ax1.legend()
    ax1.grid(True)

    # Plot 2
    ax2.plot(results1.index, results1.marginal_costs, marker='x', linestyle='--', label=label1)
    common_index = results1.index.intersection(results2.index)
    ax2.plot(common_index, results2.loc[common_index, "marginal_costs"], marker='x', linestyle='--', label=label2)
    common_index = results1.index.intersection(results3.index)
    ax2.plot(common_index, results3.loc[common_index, "marginal_costs"], 'x--', label=label3)
    ax2.legend()
    ax2.grid(True)

    # Plot 3
    ax3.plot(results1.index, results1.investment_costs, marker='x', linestyle='--', label=label1)
    common_index = results1.index.intersection(results2.index)
    ax3.plot(common_index, results2.loc[common_index, "investment_costs"], marker='x', linestyle='--', label=label2)
    common_index = results1.index.intersection(results3.index)
    ax3.plot(common_index, results3.loc[common_index, "investment_costs"], 'x--', label=label3)
    ax3.legend()
    ax3.grid(True)

    ax1.set_ylabel("Systemkosten")
    ax2.set_ylabel("Erzeugungskosten")
    ax3.set_ylabel("Investkosten")
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    
def plot_investment_costs(results1, label1, results2, label2, results3, label3, path):
    
    # Drei Diagramme untereinander, gemeinsame x-Achse
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(8, 8))
    plt.suptitle("Investitionskosten in Millionen Euro")

    # Plot 1
    ax1.plot(results1.index, results1.ac_grid_investment_costs, marker='x', linestyle='--', label=label1)
    common_index = results1.index.intersection(results2.index)
    ax1.plot(common_index, results2.loc[common_index, "ac_grid_investment_costs"], marker='x', linestyle='--', label=label2)
    common_index = results1.index.intersection(results3.index)
    ax1.plot(common_index, results3.loc[common_index, "ac_grid_investment_costs"], 'x--', label=label3)
    ax1.legend()
    ax1.grid(True)

    # Plot 2
    ax2.plot(results1.index, results1.dc_grid_investment_costs, marker='x', linestyle='--', label=label1)
    common_index = results1.index.intersection(results2.index)
    ax2.plot(common_index, results2.loc[common_index, "dc_grid_investment_costs"], marker='x', linestyle='--', label=label2)
    common_index = results1.index.intersection(results3.index)
    ax3.plot(common_index, results3.loc[common_index, "dc_grid_investment_costs"], 'x--', label=label3)
    ax2.legend()
    ax2.grid(True)

    # Plot 3
    ax3.plot(results1.index, results1.battery_expansion_costs, marker='x', linestyle='--', label=label1)
    common_index = results1.index.intersection(results2.index)
    ax3.plot(common_index, results2.loc[common_index, "battery_expansion_costs"], marker='x', linestyle='--', label=label2)
    common_index = results1.index.intersection(results3.index)
    ax1.plot(common_index, results3.loc[common_index, "battery_expansion_costs"], 'x--', label=label3)
    ax3.legend()
    ax3.grid(True)

    # Plot 4
    ax4.plot(results1.index, results1.other_link_expansion_costs, marker='x', linestyle='--', label=label1)
    common_index = results1.index.intersection(results2.index)
    ax4.plot(common_index, results2.loc[common_index, "other_link_expansion_costs"], marker='x', linestyle='--', label=label2)
    common_index = results1.index.intersection(results3.index)
    ax4.plot(common_index, results3.loc[common_index, "other_link_expansion_costs"], 'x--', label=label3)
    ax4.legend()
    ax4.grid(True)
    
    # Plot 5
    ax5.plot(results1.index, results1.H2_overground_expansion_costs, marker='x', linestyle='--', label=label1)
    common_index = results1.index.intersection(results2.index)
    ax5.plot(common_index, results2.loc[common_index, "H2_overground_expansion_costs"], marker='x', linestyle='--', label=label2)
    common_index = results1.index.intersection(results3.index)
    ax5.plot(common_index, results3.loc[common_index, "H2_overground_expansion_costs"], 'x--', label=label3)
    ax5.legend()
    ax5.grid(True)

    ax1.set_ylabel("AC-Leitungen")
    ax2.set_ylabel("DC-Leitungen")
    ax3.set_ylabel("Batteriespeicher")
    ax4.set_ylabel("andere Links")
    ax5.set_ylabel("H2")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# sig100
path="/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/sig-100/"
calc = ["AC-50", "AC-100", "AC-150", "AC-200", "AC-250", "AC-300"]
e_sig100, results_sig100 = get_results(path, calc, "sig-100_")
results_sig100_de = get_results_de(path, calc)
results_sig100_focus = get_results_focus(path, calc)

# sig20
path="/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/sig-20/"
calc = ["AC-50", "AC-150", "AC-250"]
e_sig20, results_sig20 = get_results(path, calc, "sig-20_")
results_sig20_de = get_results_de(path, calc)
results_sig20_focus = get_results_focus(path, calc)

# no Zooming
path="/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/no-zooming/"
calc = ["AC-50", "AC-100", "AC-200", "AC-300"]
e_ohne, results_ohne = get_results(path, calc, "ohne_")
results_ohne_de = get_results_de(path, calc)
results_ohne_focus = get_results_focus(path, calc)

plot_path = "/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/"
plot_no_buses(results_sig100, "sig100", results_sig20, "sig20", results_ohne, "no-zooming", plot_path+'no_buses.png')
plot_costs(results_sig100, "sig100", results_sig20, "sig20", results_ohne, "no-zooming", plot_path+'costs.png')
plot_investment_costs(results_sig100, "sig100", results_sig20, "sig20", results_ohne, "no-zooming", plot_path+'invest_costs.png')

plot_path = "/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/de_"
plot_no_buses(results_sig100_de, "sig100", results_sig20_de, "sig20", results_ohne_de, "no-zooming", plot_path+'no_buses.png')
plot_costs(results_sig100_de, "sig100", results_sig20_de, "sig20", results_ohne_de, "no-zooming", plot_path+'costs.png')
plot_investment_costs(results_sig100_de, "sig100", results_sig20_de, "sig20", results_ohne_de, "no-zooming", plot_path+'invest_costs.png')

plot_path = "/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/focus_"
plot_no_buses(results_sig100_focus, "sig100", results_sig20_focus, "sig20", results_ohne_focus, "no-zooming", plot_path+'no_buses.png')
plot_costs(results_sig100_focus, "sig100", results_sig20_focus, "sig20", results_ohne_focus, "no-zooming", plot_path+'costs.png')
plot_investment_costs(results_sig100_focus, "sig100", results_sig20_focus, "sig20", results_ohne_focus, "no-zooming", plot_path+'invest_costs.png')



