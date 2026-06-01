
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

cluster = Etrago(csv_folder_name= 'Zooming-Tests/full-res/Server1/AC-'+str(res)+'/vor-cluster')
n_cluster = cluster.network

vorher = Etrago(csv_folder_name= 'Zooming-Tests/full-res/Server1/AC-'+str(res)+'/vor-lpf')
n_vorher = vorher.network

lpf = Etrago(csv_folder_name= 'Zooming-Tests/full-res/Server1/AC-'+str(res)+'/lpf')
n_lpf = lpf.network

def check_completeness(n_cluster, n_vorher, n_lpf, atol=1e-3):

    components = {
        "Generator":   ("generators",    "generators_t",    "p_set", "p"),
        "Link":        ("links",         "links_t",         "p_set", "p0"),
        "StorageUnit": ("storage_units", "storage_units_t", "p_set", "p"),
        "Store":       ("stores",        "stores_t",        "p_set", "p"),
    }

    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Schritt 1: Haben alle Komponenten ein p_set?")
    print("(geprüft für: n_cluster, n_vorher, n_lpf)")
    print("=" * 60)

    for net_name, net in [("n_cluster", n_cluster), ("n_vorher", n_vorher), ("n_lpf", n_lpf)]:
        print(f"\n  >>> Netz: {net_name}")
        n_snapshots = len(net.snapshots)

        for comp, (component, container, set_attr, lpf_attr) in components.items():
            df_comp = getattr(net, component, None)
            df      = getattr(net, container, None)

            if df_comp is None or df_comp.empty:
                print(f"    [{comp}] — keine Assets vorhanden, übersprungen")
                continue

            no_timeseries = []
            incomplete    = []

            if df is None or set_attr not in df:
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

            n_total = len(df_comp)
            n_ok    = n_total - len(no_timeseries) - len(incomplete)

            print(f"\n    [{comp}] — {n_total} Assets total")
            print(f"      ✓ vollständig:           {n_ok}")
            print(f"      ✗ kein Zeitreihen-p_set: {len(no_timeseries)}", end="")
            if no_timeseries:
                if "carrier" in df_comp.columns:
                    carriers = df_comp.loc[no_timeseries, "carrier"].unique()
                    print(f"  → Carrier: {list(carriers)}", end="")
                else:
                    sample = no_timeseries[:3]
                    print(f"  → z.B. {sample}{'...' if len(no_timeseries) > 3 else ''}", end="")
            print()
            if incomplete:
                print(f"      ✗ unvollständige p_set:  {len(incomplete)}")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Schritt 2: Übereinstimmung zwischen allen Netz-Paaren")
    print("=" * 60)

    net_pairs = [
        ("n_cluster", n_cluster, "n_vorher", n_vorher),
        ("n_vorher",  n_vorher,  "n_lpf",    n_lpf),
        ("n_cluster", n_cluster, "n_lpf",    n_lpf),
    ]

    for name_a, net_a, name_b, net_b in net_pairs:
        print(f"\n  >>> Vergleich: {name_a}  ↔  {name_b}")

        for comp, (component, container, set_attr, lpf_attr) in components.items():
            print(f"\n    [{comp}]")

            df_a = getattr(net_a, container, None)
            df_b = getattr(net_b, container, None)

            # ---- a) Asset-Mengen identisch? --------------------------- #
            if df_a is None or set_attr not in df_a:
                print(f"      a) ✗ kein p_set in {name_a} — übersprungen")
                continue
            if df_b is None or set_attr not in df_b:
                print(f"      a) ✗ kein p_set in {name_b} — übersprungen")
                continue

            assets_a = set(df_a[set_attr].columns)
            assets_b = set(df_b[set_attr].columns)

            only_in_a = assets_a - assets_b
            only_in_b = assets_b - assets_a

            if not only_in_a and not only_in_b:
                print(f"      a) ✓ alle {len(assets_a)} Assets in beiden Netzen vorhanden")
            else:
                print(f"      a) ✗ Differenz: {len(only_in_a)} nur in {name_a}, {len(only_in_b)} nur in {name_b}")
                if only_in_a:
                    sample = list(only_in_a)[:3]
                    print(f"           nur in {name_a}: {sample}{'...' if len(only_in_a) > 3 else ''}")
                if only_in_b:
                    sample = list(only_in_b)[:3]
                    print(f"           nur in {name_b}: {sample}{'...' if len(only_in_b) > 3 else ''}")

            # ---- b) p_set(net_a) == p_set(net_b)? -------------------- #
            p_set_a  = df_a[set_attr]
            p_set_b  = df_b[set_attr]
            common_b = p_set_a.columns.intersection(p_set_b.columns)

            if len(common_b) == 0:
                print(f"      b) ✗ keine gemeinsamen Assets für Vergleich")
            else:
                diff_b       = (p_set_a[common_b] - p_set_b[common_b]).abs()
                max_b        = diff_b.max()
                n_mismatch_b = (max_b > atol).sum()
                if n_mismatch_b == 0:
                    print(f"      b) ✓ p_set identisch ({len(common_b)} Assets)")
                else:
                    print(f"      b) ✗ p_set weicht ab: {n_mismatch_b}/{len(common_b)} Assets, max Δ={max_b.max():.4f}")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Schritt 3: p == p_set innerhalb jedes Netzes")
    print("=" * 60)

    for net_name, net in [("n_cluster", n_cluster), ("n_vorher", n_vorher), ("n_lpf", n_lpf)]:
        print(f"\n  >>> Netz: {net_name}")

        for comp, (component, container, set_attr, lpf_attr) in components.items():
            df = getattr(net, container, None)

            if df is None or set_attr not in df or lpf_attr not in df:
                print(f"    [{comp}] ✗ p_set oder {lpf_attr} fehlt — übersprungen")
                continue

            p_set   = df[set_attr]
            p       = df[lpf_attr]
            common  = p_set.columns.intersection(p.columns)

            if len(common) == 0:
                print(f"    [{comp}] ✗ keine gemeinsamen Assets für Vergleich")
            else:
                diff        = (p_set[common] - p[common]).abs()
                max_diff    = diff.max()
                n_mismatch  = (max_diff > atol).sum()
                if n_mismatch == 0:
                    print(f"    [{comp}] ✓ p == p_set ({len(common)} Assets)")
                else:
                    print(f"    [{comp}] ✗ p ≠ p_set: {n_mismatch}/{len(common)} Assets, max Δ={max_diff.max():.4f}")

    print()
    print("=" * 60)
    print("Prüfung abgeschlossen.")
    print("=" * 60)

check_completeness(n_cluster, n_vorher, n_lpf)

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

