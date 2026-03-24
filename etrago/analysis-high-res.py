
import numpy as np
import pandas as pd
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

focus_gdf = gpd.read_file("/home/dozeumesk/eTraGo/git/eTraGo/etrago/focus-region/hannover.gpkg")
focus_gdf = focus_gdf.to_crs(epsg=4326)

# Auswertung

res=[30, 300, 1000, 3000, 5000, 8000, 10000]

full = Etrago(csv_folder_name= "/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/full-res/AC-30/lpf")
buses=full.network.buses        
buses["geom"] =  buses.apply(lambda x: Point(x["x"], x["y"]), axis=1)
buses = gpd.GeoDataFrame(buses, geometry="geom", crs=4326)
buses_full = gpd.clip(buses, focus_gdf)

half = Etrago(csv_folder_name= "/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/full-res/AC-300/lpf")
buses=half.network.buses        
buses["geom"] =  buses.apply(lambda x: Point(x["x"], x["y"]), axis=1)
buses = gpd.GeoDataFrame(buses, geometry="geom", crs=4326)
buses_half = gpd.clip(buses, focus_gdf)

## Lastflussfehler mit Fokus auf Region in Abhängigkeit der räumlichen Auflösung

## Lastflussfehler-Karte

def line_flow_error_clustered(n_ref, n_cluster, busmap):
    
    import pdb; pdb.set_trace()

    snapshot = n_ref.snapshots[0]

    # -------------------------
    # 1. Referenzleitungen laden
    # -------------------------

    lines = n_ref.lines.copy()

    flows_ref = n_ref.lines_t.p0.loc[snapshot]

    lines["flow"] = flows_ref

    # Clusterzuordnung der Busse
    lines["cluster0"] = lines.bus0.map(busmap)
    lines["cluster1"] = lines.bus1.map(busmap)

    # -------------------------
    # 2. Leitungen innerhalb desselben Clusters entfernen
    # -------------------------

    lines = lines[lines.cluster0 != lines.cluster1]

    # -------------------------
    # 3. Referenzflüsse zwischen Clusterpaaren aggregieren
    # -------------------------

    agg_ref = (
        lines
        .groupby(["cluster0", "cluster1"])["flow"]
        .sum()
    )

    # auch umgekehrte Richtung berücksichtigen
    agg_ref_rev = (
        lines
        .groupby(["cluster1", "cluster0"])["flow"]
        .sum()
    )

    agg_ref = agg_ref.add(agg_ref_rev, fill_value=0)

    # -------------------------
    # 4. Clusterflüsse
    # -------------------------

    flows_cluster = n_cluster.lines_t.p0.loc[snapshot]

    cluster_lines = n_cluster.lines.copy()

    cluster_lines["flow_cluster"] = flows_cluster

    # lookup Schlüssel
    keys = list(zip(cluster_lines.bus0, cluster_lines.bus1))

    ref_flow = pd.Series(
        [agg_ref.get(k, np.nan) for k in keys],
        index=cluster_lines.index
    )

    # -------------------------
    # 5. relativer Fehler
    # -------------------------

    eps = 1e-6

    rel_error = (
        np.abs(np.abs(cluster_lines.flow_cluster) - np.abs(ref_flow))
        / (np.abs(ref_flow) + eps)
    )

    return rel_error

busmap = pd.read_csv("/home/dozeumesk/eTraGo/git/eTraGo/etrago/Zooming-Tests/full-res/kmedoids-dijkstra_elecgrid_busmap_30_result.csv")
busmap.index=busmap.bus
busmap=busmap.drop(['foreign', 'medoid_idx', 'bus'], axis=1)
line_flow_error_clustered(full.network, half.network, busmap=busmap)

# Plausibilitätschecks

vorher = Etrago(csv_folder_name= "Zooming-Tests/full-res/AC-30/vor-lpf")
n_vorher = vorher.network

lpf = Etrago(csv_folder_name= "Zooming-Tests/full-res/AC-30/lpf")
n_lpf = lpf.network

def check_p_set_completeness(n):
    
    components = {
        "Generator": "generators_t",
        "Link": "links_t",
        "StorageUnit": "storage_units_t",
        "Store": "stores_t"
    }
    
    result = {}
    n_snapshots = len(n.snapshots)
    
    for comp, container in components.items():
        df = getattr(n, container, None)
        info = {'exists': False, 'complete': False, 'missing': []}
        
        if df is not None:
            # Prüfen ob 'p_set' vorhanden ist
            if 'p_set' in df:
                info['exists'] = True
                
                p_set = df['p_set']
                
                # Wenn Series → ein Asset, ansonsten DataFrame
                if isinstance(p_set, pd.Series):
                    if p_set.count() == n_snapshots:
                        info['complete'] = True
                    else:
                        info['missing'] = [p_set.name]
                else:
                    # DataFrame
                    missing_cols = [col for col in p_set.columns if p_set[col].count() < n_snapshots]
                    info['missing'] = missing_cols
                    info['complete'] = len(missing_cols) == 0
        
        result[comp] = info
    
    return result

status = check_p_set_completeness(n_vorher)

for comp, info in status.items():
    print(f"{comp}: exists={info['exists']}, complete={info['complete']}, missing={info['missing']}")


def check_lpf_vs_pset(n_vorher, n_lpf, atol=1e-3):

    mapping = {
        "Generator": ("generators_t", "p_set", "p"),
        "StorageUnit": ("storage_units_t", "p_set", "p"),
        "Store": ("stores_t", "p_set", "p"),
        "Link": ("links_t", "p_set", "p0"),
    }

    results = []

    for comp, (container, set_attr, lpf_attr) in mapping.items():

        df_set = getattr(n_vorher, container, None)
        df_lpf = getattr(n_lpf, container, None)

        if df_set is None or df_lpf is None:
            results.append([comp, 0, 0, 0, np.nan])
            continue

        if set_attr not in df_set or lpf_attr not in df_lpf:
            results.append([comp, 0, 0, 0, np.nan])
            continue

        p_set = df_set[set_attr]
        p_lpf = df_lpf[lpf_attr]

        common = p_set.columns.intersection(p_lpf.columns)

        if len(common) == 0:
            results.append([comp, 0, 0, 0, np.nan])
            continue

        diff = (p_set[common] - p_lpf[common]).abs()
        max_diff_asset = diff.max()

        n_total = len(common)
        n_match = (max_diff_asset <= atol).sum()
        n_mismatch = n_total - n_match
        max_delta = max_diff_asset.max()

        results.append([comp, n_total, n_match, n_mismatch, max_delta])

    return pd.DataFrame(
        results,
        columns=["Component", "Assets", "Match", "Mismatch", "Max Δ"]
    )

summary = check_lpf_vs_pset(n_vorher, n_lpf)