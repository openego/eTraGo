
import numpy as np
import pandas as pd
from etrago import Etrago

vorher = Etrago(csv_folder_name= "Zooming-Tests/full-res/AC-20/vor-lpf")
n_vorher = vorher.network

lpf = Etrago(csv_folder_name= "Zooming-Tests/full-res/AC-20/lpf")
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