
from sqlalchemy import create_engine
from etrago import Etrago
import pypsa
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

focus_gdf = gpd.read_file("/home/dozeumesk/eTraGo/git/eTraGo/etrago/focus-region/hannover.gpkg")
focus_gdf = focus_gdf.to_crs(epsg=4326)

# Plausibilitätschecks

res=30

vorher = Etrago(csv_folder_name= 'Zooming-Tests/full-res/Server1/AC-'+str(res)+'/vor-lpf')
n_vorher = vorher.network

lpf = Etrago(csv_folder_name= 'Zooming-Tests/full-res/Server1/AC-'+str(res)+'/lpf')
n_lpf = lpf.network

def check_completeness(n, n_lpf, atol=1e-3):

    components = {
        "Generator":   ("generators",    "generators_t",    "p_set", "p"),
        "Link":        ("links",         "links_t",         "p_set", "p0"),
        "StorageUnit": ("storage_units", "storage_units_t", "p_set", "p"),
        "Store":       ("stores",        "stores_t",        "p_set", "p"),
    }

    n_snapshots = len(n.snapshots)

    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Schritt 1: Haben alle Komponenten in n_vorher ein p_set?")
    print("=" * 60)

    for comp, (component, container, set_attr, lpf_attr) in components.items():
        df_comp = getattr(n, component, None)
        df      = getattr(n, container, None)

        if df_comp is None or df_comp.empty:
            print(f"  [{comp}] — keine Assets vorhanden, übersprungen")
            continue

        no_timeseries  = []  # Asset nicht in Zeitreihe
        incomplete     = []  # Asset in Zeitreihe, aber NaN vorhanden

        if df is None or set_attr not in df:
            # gar keine Zeitreihe vorhanden → alle Assets fehlen
            no_timeseries = list(df_comp.index)
        else:
            p_set = df[set_attr]
            for idx in df_comp.index:
                if idx not in p_set.columns:
                    no_timeseries.append(idx)
                else:
                    n_valid = p_set[idx].count()
                    if n_valid < n_snapshots:
                        incomplete.append((idx, n_valid, n_snapshots))

        n_total     = len(df_comp)
        n_ok        = n_total - len(no_timeseries) - len(incomplete)

        print(f"\n  [{comp}] — {n_total} Assets total")
        print(f"    ✓ vollständig:          {n_ok}")
        print(f"    ✗ kein Zeitreihen-p_set: {len(no_timeseries)}", end="")
        if no_timeseries:
            if "carrier" in df_comp.columns:
                carriers = df_comp.loc[no_timeseries, "carrier"].unique()
                print(f"  → Carrier: {list(carriers)}", end="")
            else:
                sample = no_timeseries[:3]
                print(f"  → z.B. {sample}{'...' if len(no_timeseries) > 3 else ''}", end="")
        print()

    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Schritt 2: Übereinstimmung n_vorher ↔ n_lpf")
    print("=" * 60)

    for comp, (component, container, set_attr, lpf_attr) in components.items():
        print(f"\n  [{comp}]")

        df_n   = getattr(n,     container, None)
        df_lpf = getattr(n_lpf, container, None)

        # ---- a) Komponenten vollständig in n_lpf vorhanden? ----------- #
        if df_n is None or set_attr not in df_n:
            print(f"    a) ✗ kein p_set in n_vorher — übersprungen")
            continue

        assets_n = df_n[set_attr].columns if not isinstance(df_n[set_attr], pd.Series) else [df_n[set_attr].name]

        if df_lpf is None:
            print(f"    a) ✗ Container '{container}' fehlt komplett in n_lpf")
            continue

        if set_attr not in df_lpf:
            assets_lpf = []
        else:
            assets_lpf = df_lpf[set_attr].columns if not isinstance(df_lpf[set_attr], pd.Series) else [df_lpf[set_attr].name]

        only_in_n   = set(assets_n) - set(assets_lpf)
        only_in_lpf = set(assets_lpf) - set(assets_n)

        if not only_in_n and not only_in_lpf:
            print(f"    a) ✓ alle {len(assets_n)} Assets in beiden Netzen vorhanden")
        else:
            print(f"    a) ✗ Differenz: {len(only_in_n)} nur in n_vorher, {len(only_in_lpf)} nur in n_lpf")
            if only_in_n:
                sample = list(only_in_n)[:3]
                print(f"         nur in n_vorher: {sample}{'...' if len(only_in_n) > 3 else ''}")
            if only_in_lpf:
                sample = list(only_in_lpf)[:3]
                print(f"         nur in n_lpf:    {sample}{'...' if len(only_in_lpf) > 3 else ''}")

        # ---- b) p_set(n_vorher) == p_set(n_lpf)? --------------------- #
        if set_attr not in df_lpf:
            print(f"    b) ✗ kein p_set in n_lpf")
        else:
            p_set_n   = df_n[set_attr]
            p_set_lpf = df_lpf[set_attr]
            common_b  = p_set_n.columns.intersection(p_set_lpf.columns)

            if len(common_b) == 0:
                print(f"    b) ✗ keine gemeinsamen Assets für Vergleich")
            else:
                diff_b       = (p_set_n[common_b] - p_set_lpf[common_b]).abs()
                max_b        = diff_b.max()
                n_mismatch_b = (max_b > atol).sum()
                if n_mismatch_b == 0:
                    print(f"    b) ✓ p_set identisch in n_vorher und n_lpf ({len(common_b)} Assets)")
                else:
                    print(f"    b) ✗ p_set weicht ab: {n_mismatch_b}/{len(common_b)} Assets, max Δ={max_b.max():.4f}")

        # ---- c) p(n_lpf) == p_set(n_lpf)? ---------------------------- #
        if set_attr not in df_lpf or lpf_attr not in df_lpf:
            print(f"    c) ✗ p_set oder {lpf_attr} fehlt in n_lpf")
        else:
            p_set_lpf2 = df_lpf[set_attr]
            p_lpf      = df_lpf[lpf_attr]
            common_c   = p_set_lpf2.columns.intersection(p_lpf.columns)

            if len(common_c) == 0:
                print(f"    c) ✗ keine gemeinsamen Assets für Vergleich")
            else:
                diff_c       = (p_set_lpf2[common_c] - p_lpf[common_c]).abs()
                max_c        = diff_c.max()
                n_mismatch_c = (max_c > atol).sum()
                if n_mismatch_c == 0:
                    print(f"    c) ✓ p == p_set in n_lpf ({len(common_c)} Assets)")
                else:
                    print(f"    c) ✗ p ≠ p_set in n_lpf: {n_mismatch_c}/{len(common_c)} Assets, max Δ={max_c.max():.4f}")

    print()
    print("=" * 60)
    print("Prüfung abgeschlossen.")
    print("=" * 60)

check_completeness(n_vorher, n_lpf)

'''# Auswertung

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
line_flow_error_clustered(full.network, half.network, busmap=busmap)'''

