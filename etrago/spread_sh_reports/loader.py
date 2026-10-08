"""Lazy, memoised access to the PyPSA CSV exports of one optimisation stage."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


class Stage:
    """Read-only view on ``<run>/<stage>/`` exported by ``export_to_csv_folder``."""

    def __init__(self, path):
        self.path = Path(path)

    def exists(self, name):
        return (self.path / f"{name}.csv").exists()

    @lru_cache(maxsize=None)
    def static(self, component):
        """Static component table (e.g. ``generators``), index as str."""
        file = self.path / f"{component}.csv"
        if not file.exists():
            return pd.DataFrame()
        df = pd.read_csv(file, index_col=0, low_memory=False)
        df.index = df.index.astype(str)
        for col in ("bus", "bus0", "bus1"):
            if col in df:
                df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True)
        return df

    @lru_cache(maxsize=None)
    def series(self, component, attr):
        """Time-varying attribute (e.g. ``generators``, ``p``) as float frame."""
        file = self.path / f"{component}-{attr}.csv"
        if not file.exists():
            return pd.DataFrame(index=self.snapshots)
        df = pd.read_csv(file, index_col=0)
        df.columns = df.columns.astype(str)
        snapshots = self.snapshots
        if pd.api.types.is_integer_dtype(df.index):
            # PyPSA exports some series by snapshot position.
            df.index = snapshots[df.index.values]
        else:
            df.index = pd.to_datetime(df.index)
        df.index.name = "snapshot"
        return df.astype(float)

    @property
    @lru_cache(maxsize=None)
    def snapshot_table(self):
        df = pd.read_csv(self.path / "snapshots.csv", index_col=0)
        df["snapshot"] = pd.to_datetime(df["snapshot"])
        return df.set_index("snapshot")

    @property
    def snapshots(self):
        return self.snapshot_table.index

    @property
    def weights(self):
        """Objective weighting in hours per snapshot."""
        return self.snapshot_table["objective"].astype(float)

    @property
    @lru_cache(maxsize=None)
    def network_meta(self):
        file = self.path / "network.csv"
        if not file.exists():
            return {}
        return pd.read_csv(file).iloc[0].to_dict()

    def dense(self, component, attr, default_col=None, columns=None):
        """Time series with static fall-back, like PyPSA's
        ``get_switchable_as_dense``."""
        static = self.static(component)
        columns = static.index if columns is None else pd.Index(columns)
        ts = self.series(component, attr)
        default = static[default_col or attr] if (default_col or attr) in static else 0.0
        if isinstance(default, pd.Series):
            default = default.reindex(columns)
            out = pd.DataFrame(
                np.tile(default.values.astype(float), (len(self.snapshots), 1)),
                index=self.snapshots,
                columns=columns,
            )
        else:
            out = pd.DataFrame(default, index=self.snapshots, columns=columns)
        common = columns.intersection(ts.columns)
        if len(common):
            out[common] = ts[common].values
        return out

    def energy(self, frame):
        """Weighted sum over time (MW -> MWh)."""
        return frame.mul(self.weights.reindex(frame.index).values, axis=0).sum()

    def clear(self):
        self.static.cache_clear()
        self.series.cache_clear()
