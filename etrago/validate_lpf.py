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

focus_gdf = gpd.read_file("/home/dozeumesk/eTraGo/git-zooming/eTraGo/etrago/focus-region/hannover.gpkg")
focus_gdf = focus_gdf.to_crs(epsg=4326)

res=50

vor_cluster = Etrago(csv_folder_name= 'Zooming-Tests/lokale_tests/AC-'+str(res)+'/01-vor-cluster')
n_vor_cluster = vor_cluster.network

nach_cluster = Etrago(csv_folder_name= 'Zooming-Tests/lokale_tests/AC-'+str(res)+'/02-nach-cluster')
n_nach_cluster = nach_cluster.network

dc_results = Etrago(csv_folder_name= 'Zooming-Tests/lokale_tests/AC-'+str(res)+'/03-mit-dc-results')
n_dc_results = dc_results.network

vor_lpf = Etrago(csv_folder_name= 'Zooming-Tests/lokale_tests/AC-'+str(res)+'/04-vor-lpf')
n_vor_lpf = vor_lpf.network

nach_lpf = Etrago(csv_folder_name= 'Zooming-Tests/lokale_tests/AC-'+str(res)+'/05-nach-lpf')
n_nach_lpf = nach_lpf.network

final = Etrago(csv_folder_name= 'Zooming-Tests/lokale_tests/AC-'+str(res)+'/06-final-results')
n_final = final.network

dropped = Etrago(csv_folder_name= 'Zooming-Tests/lokale_tests/AC-'+str(res)+'/07-dropped-sectors')
n_dropped = dropped.network

def validate_power_flow_pipeline(
    n_vor_cluster,
    n_nach_cluster,
    n_dc_results,
    n_vor_lpf,
    n_nach_lpf,
    n_final,
    n_dropped,
    atol=1e-3,
):
    """
    Validiert die einzelnen Zwischenzustände der run_power_flow-Pipeline.
    """

    dispatch_components = {
        "Generator":   ("generators",    "generators_t",    "p_set", "p"),
        "Link":        ("links",         "links_t",         "p_set", "p0"),
        "StorageUnit": ("storage_units", "storage_units_t", "p_set", "p"),
        "Store":       ("stores",        "stores_t",        "p_set", "p"),
    }

    all_components = {
        "Bus":          "buses",
        "Line":         "lines",
        "Transformer":  "transformers",
        "Link":         "links",
        "Generator":    "generators",
        "Load":         "loads",
        "StorageUnit":  "storage_units",
        "Store":        "stores",
    }

    # Repräsentativer Bus je Komponente (für Scope-Filterung nach
    # Carrier/Land). Bei zweipoligen Elementen (Line/Transformer/Link)
    # wird bus0 als Referenz genutzt.
    bus_col_map = {
    "Bus": None,
    "Generator": ("bus", None),
    "Load": ("bus", None),
    "StorageUnit": ("bus", None),
    "Store": ("bus", None),
    "Line": ("bus0", "bus1"),
    "Transformer": ("bus0", "bus1"),
    "Link": ("bus0", "bus1"),
    }

    SCOPES = [
        ("Gesamt",                     "all"),
        ("Nur AC",                     "ac"),
        ("AC in DE",                   "ac_de"),
        ("AC im Ausland",              "ac_foreign"),
        ("Andere Sektoren (nicht-AC)", "other"),
    ]

    # ------------------------------------------------------------------ #
    def scope_index(net, comp_name, df_comp, scope):
        """Index der Assets einer Komponente, die in den gegebenen Scope fallen.
    
        Für zweipolige Komponenten (Line, Transformer, Link) werden BEIDE
        Bus-Enden berücksichtigt:
          - ac_de:      BEIDE Enden sind AC und in DE
          - ac_foreign: AC, und MINDESTENS EIN Ende im Ausland
                        (grenzüberschreitende Elemente zählen zum Ausland)
    
        Links mit carrier == "DC" zählen ebenfalls als AC-/Strom-Scope.
    
        Falls ein Bus-Ende nicht (mehr) in net.buses existiert (z.B. weil
        ausländische Buses zuvor via drop_foreign_components entfernt
        wurden), wird dieses Ende als AC + Ausland gewertet — das Element
        zählt dann zu ac_foreign.
        """
        if df_comp is None or df_comp.empty:
            return pd.Index([])
    
        bus_cols = bus_col_map[comp_name]
    
        if comp_name == "Bus":
            carrier0 = df_comp["carrier"]
            country0 = df_comp["country"] if "country" in df_comp else pd.Series("", index=df_comp.index)
            carrier1 = carrier0
            country1 = country0
            bus0_missing = pd.Series(False, index=df_comp.index)
            bus1_missing = pd.Series(False, index=df_comp.index)
        else:
            bus_col0, bus_col1 = bus_cols
            if bus_col0 not in df_comp.columns:
                return pd.Index([])
    
            buses0 = df_comp[bus_col0]
            bus0_missing = ~buses0.isin(net.buses.index)
            carrier0 = buses0.map(net.buses["carrier"])
            country0 = buses0.map(net.buses["country"]) if "country" in net.buses else pd.Series("", index=df_comp.index)
    
            if bus_col1 is not None and bus_col1 in df_comp.columns:
                buses1 = df_comp[bus_col1]
                bus1_missing = ~buses1.isin(net.buses.index)
                carrier1 = buses1.map(net.buses["carrier"])
                country1 = buses1.map(net.buses["country"]) if "country" in net.buses else pd.Series("", index=df_comp.index)
            else:
                # einpolige Komponente: bus1 == bus0
                bus1_missing = bus0_missing
                carrier1 = carrier0
                country1 = country0
    
            # fehlende Buses (z.B. zuvor gedroppte Foreign Buses) als
            # AC + Ausland werten
            carrier0 = carrier0.where(~bus0_missing, "AC")
            carrier1 = carrier1.where(~bus1_missing, "AC")
            country0 = country0.where(~bus0_missing, "__FOREIGN_MISSING__")
            country1 = country1.where(~bus1_missing, "__FOREIGN_MISSING__")
    
        both_ac = (carrier0 == "AC") & (carrier1 == "AC")
    
        # DC-Links zählen ebenfalls als AC-/Strom-Scope, unabhängig vom
        # Bus-Carrier
        if comp_name == "Link" and "carrier" in df_comp.columns:
            is_dc_link = df_comp["carrier"] == "DC"
            both_ac = both_ac | is_dc_link
    
        if scope == "all":
            mask = pd.Series(True, index=df_comp.index)
        elif scope == "ac":
            mask = both_ac
        elif scope == "ac_de":
            mask = both_ac & (country0 == "DE") & (country1 == "DE")
        elif scope == "ac_foreign":
            mask = both_ac & ((country0 != "DE") | (country1 != "DE"))
        elif scope == "other":
            mask = ~both_ac
        else:
            raise ValueError(f"Unbekannter Scope: {scope}")
    
        return df_comp.index[mask]

    def effective_p_set(net, component, container, set_attr):
        """Statischer Default, überschrieben durch vorhandene Zeitreihe."""
        df_comp = getattr(net, component, None)
        df = getattr(net, container, None)

        if df_comp is None or df_comp.empty:
            return pd.DataFrame(index=net.snapshots)

        static_vals = (
            df_comp[set_attr] if set_attr in df_comp
            else pd.Series(0.0, index=df_comp.index)
        )
        eff = pd.DataFrame(
            np.tile(static_vals.values, (len(net.snapshots), 1)),
            index=net.snapshots,
            columns=df_comp.index,
        )

        if df is not None and set_attr in df and not df[set_attr].empty:
            ts = df[set_attr]
            common_cols = ts.columns.intersection(eff.columns)
            eff[common_cols] = ts[common_cols]

        return eff

    def effective_p(net, container, lpf_attr):
        df = getattr(net, container, None)
        if df is None or lpf_attr not in df:
            return pd.DataFrame(index=net.snapshots)
        return df[lpf_attr]

    # ------------------------------------------------------------------ #
    def component_overview(net):
        for comp, attr in all_components.items():
            df = getattr(net, attr, None)
            n = len(df) if df is not None else 0
            print(f"    {comp:<12}: {n}")

    def check_p_set_completeness(net, restrict_idx=None):
        for comp, (component, container, set_attr, lpf_attr) in dispatch_components.items():
            df_comp = getattr(net, component, None)
            if df_comp is None or df_comp.empty:
                print(f"    [{comp}] — keine Assets vorhanden, übersprungen")
                continue

            eff = effective_p_set(net, component, container, set_attr)
            if restrict_idx is not None:
                cols = eff.columns.intersection(restrict_idx)
                eff = eff[cols]
                n_total = len(cols)
            else:
                n_total = len(df_comp)

            if n_total == 0:
                print(f"    [{comp}] — keine Assets in diesem Scope, übersprungen")
                continue

            nan_counts = eff.isna().sum()
            assets_with_nan = nan_counts[nan_counts > 0]
            n_nan = len(assets_with_nan)
            n_ok = n_total - n_nan

            status = "✓" if n_nan == 0 else "✗"
            print(f"    [{comp}] {status} {n_ok}/{n_total} vollständig", end="")
            if n_nan:
                sample = assets_with_nan.index.tolist()[:3]
                print(f"  — Lücken bei: {sample}{'...' if n_nan > 3 else ''}", end="")
            print()

    def check_dispatch_matches_pset(net, restrict_idx=None):
        for comp, (component, container, set_attr, lpf_attr) in dispatch_components.items():
            df_comp = getattr(net, component, None)
            if df_comp is None or df_comp.empty:
                print(f"    [{comp}] — keine Assets vorhanden, übersprungen")
                continue

            eff_p_set = effective_p_set(net, component, container, set_attr)
            p = effective_p(net, container, lpf_attr)

            common = eff_p_set.columns.intersection(p.columns)
            if restrict_idx is not None:
                common = common.intersection(restrict_idx)

            if len(common) == 0:
                print(f"    [{comp}] ✗ keine gemeinsamen Assets für Vergleich ({lpf_attr} fehlt oder keine relevanten Assets)")
                continue

            diff = (eff_p_set[common] - p[common]).abs()
            max_diff = diff.max()
            n_mismatch = (max_diff > atol).sum()

            if n_mismatch == 0:
                print(f"    [{comp}] ✓ p == p_set ({len(common)} Assets geprüft)")
            else:
                mismatched = max_diff[max_diff > atol]
                sample = mismatched.index.tolist()[:3]
                print(f"    [{comp}] ✗ p ≠ p_set: {n_mismatch}/{len(common)} Assets, "
                      f"max Δ={max_diff.max():.4f} — z.B. {sample}")

    # ------------------------------------------------------------------ #
    def detailed_scope_report(net, check_mode):
        """
        check_mode:
          'pset'          -> p_set-Vollständigkeit pro Scope
          'dispatch_ac'-> Übersicht für alle Scopes, aber
                              p/p0 == p_set NUR für Scope 'AC'
        """
        for scope_label, scope_key in SCOPES:
            print(f"\n  >>> Scope: {scope_label}")

            print("    Komponentenübersicht:")
            for comp, attr in all_components.items():
                df_comp = getattr(net, attr, None)
                idx = scope_index(net, comp, df_comp, scope_key)
                print(f"      {comp:<12}: {len(idx)}")

            if check_mode == "pset":
                print("    p_set-Vollständigkeit:")
                for comp, (component, container, set_attr, lpf_attr) in dispatch_components.items():
                    df_comp = getattr(net, component, None)
                    idx = scope_index(net, comp, df_comp, scope_key)
                    if len(idx) == 0:
                        print(f"      [{comp}] — keine Assets in diesem Scope, übersprungen")
                        continue

                    eff = effective_p_set(net, component, container, set_attr)
                    eff = eff[eff.columns.intersection(idx)]
                    n_total = len(eff.columns)

                    nan_counts = eff.isna().sum()
                    assets_with_nan = nan_counts[nan_counts > 0]
                    n_nan = len(assets_with_nan)
                    n_ok = n_total - n_nan

                    status = "✓" if n_nan == 0 else "✗"
                    print(f"      [{comp}] {status} {n_ok}/{n_total} vollständig", end="")
                    if n_nan:
                        sample = assets_with_nan.index.tolist()[:3]
                        print(f"  — Lücken bei: {sample}{'...' if n_nan > 3 else ''}", end="")
                    print()

            elif check_mode == "dispatch_ac_de":
                if scope_key == "ac_de":
                    print("    p == p_set (Dispatch-Check):")
                    idx_by_comp = {
                        comp: scope_index(net, comp, getattr(net, dispatch_components[comp][0], None), scope_key)
                        for comp in dispatch_components
                    }
                    # Vereinigter Index reicht hier nicht, da je comp eigener bus-bezug
                    # -> check_dispatch_matches_pset comp-weise mit passendem restrict_idx
                    for comp, (component, container, set_attr, lpf_attr) in dispatch_components.items():
                        df_comp = getattr(net, component, None)
                        idx = scope_index(net, comp, df_comp, scope_key)
                        if len(idx) == 0:
                            print(f"      [{comp}] — keine Assets in diesem Scope, übersprungen")
                            continue
                        eff_p_set = effective_p_set(net, component, container, set_attr)
                        p = effective_p(net, container, lpf_attr)
                        common = eff_p_set.columns.intersection(p.columns).intersection(idx)
                        if len(common) == 0:
                            print(f"      [{comp}] ✗ keine gemeinsamen Assets für Vergleich")
                            continue
                        diff = (eff_p_set[common] - p[common]).abs()
                        max_diff = diff.max()
                        n_mismatch = (max_diff > atol).sum()
                        if n_mismatch == 0:
                            print(f"      [{comp}] ✓ p == p_set ({len(common)} Assets geprüft)")
                        else:
                            mismatched = max_diff[max_diff > atol]
                            sample = mismatched.index.tolist()[:3]
                            print(f"      [{comp}] ✗ p ≠ p_set: {n_mismatch}/{len(common)} Assets, "
                                  f"max Δ={max_diff.max():.4f} — z.B. {sample}")
            elif check_mode == "dispatch_ac":
                if scope_key in ("ac_de", "ac_foreign"):
                    print("    p == p_set (Dispatch-Check):")
                    for comp, (component, container, set_attr, lpf_attr) in dispatch_components.items():
                        df_comp = getattr(net, component, None)
                        idx = scope_index(net, comp, df_comp, scope_key)
                        if len(idx) == 0:
                            print(f"      [{comp}] — keine Assets in diesem Scope, übersprungen")
                            continue
                        eff_p_set = effective_p_set(net, component, container, set_attr)
                        p = effective_p(net, container, lpf_attr)
                        common = eff_p_set.columns.intersection(p.columns).intersection(idx)
                        if len(common) == 0:
                            print(f"      [{comp}] ✗ keine gemeinsamen Assets für Vergleich")
                            continue
                        diff = (eff_p_set[common] - p[common]).abs()
                        max_diff = diff.max()
                        n_mismatch = (max_diff > atol).sum()
                        if n_mismatch == 0:
                            print(f"      [{comp}] ✓ p == p_set ({len(common)} Assets geprüft)")
                        else:
                            mismatched = max_diff[max_diff > atol]
                            sample = mismatched.index.tolist()[:3]
                            print(f"      [{comp}] ✗ p ≠ p_set: {n_mismatch}/{len(common)} Assets, "
                                  f"max Δ={max_diff.max():.4f} — z.B. {sample}")
                                

    # ------------------------------------------------------------------ #
    # Hauptablauf
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Stufe 1: vor Clustering")
    print("=" * 60)
    print()
    print("  Komponentenübersicht:")
    component_overview(n_vor_cluster)
    print("\n  p_set-Vollständigkeit:")
    check_p_set_completeness(n_vor_cluster)
    print()

    print("=" * 60)
    print("Stufe 2: nach Clustering")
    print("=" * 60)
    print()
    print("  Komponentenübersicht:")
    component_overview(n_nach_cluster)
    print("\n  p_set-Vollständigkeit:")
    check_p_set_completeness(n_nach_cluster)
    print()
    
    print("=" * 60)
    print("Stufe 3: nach Clustering mit DC-Results)")
    print("=" * 60)
    detailed_scope_report(n_dc_results, check_mode="pset")
    print()

    print("=" * 60)
    print("Stufe 4: vor lpf (Ausland entfernt, sonstig vorbereitet für lpf)")
    print("=" * 60)
    detailed_scope_report(n_vor_lpf, check_mode="pset")
    print()
    
    print("=" * 60)
    print("Stufe 5: nach_lpf (Ausland noch entfernt)")
    print("=" * 60)
    detailed_scope_report(n_nach_lpf, check_mode="dispatch_ac")
    print()

    print("=" * 60)
    print("Stufe 6: final (Ausland wieder hinzugefügt)")
    print("=" * 60)
    print("  Komponentenübersicht:")
    detailed_scope_report(n_final, check_mode="dispatch_ac")
    print()
    

    print("=" * 60)
    print("Stufe 7: nur noch AC, andere Sektoren gedropped")
    print("=" * 60)
    detailed_scope_report(n_dropped, check_mode="dispatch_ac")
    print()

    print("=" * 60)
    print("Validierung abgeschlossen.")
    print("=" * 60)

validate_power_flow_pipeline(n_vor_cluster, n_nach_cluster, n_dc_results, n_vor_lpf, n_nach_lpf, n_final, n_dropped)