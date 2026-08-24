#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:11:44 2026

@author: Mohsen Mansouri
"""

# -*- coding: utf-8 -*-

from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

ZENODO_MARKET_ZONE_RECORD = "https://zenodo.org/records/14833526/files"

MARKET_ZONE_SHAPEFILES = {
    "DE2": {
        "base": "BZR_config_2_DE2",
        "md5": {
            ".dbf": "078993bc32af1695da0b221b337b15f6",
            ".prj": "c742bee3d4edfc2948a2ad08de1790a5",
            ".shp": "95283edee5b17e9b1a4c4d5ef54f1043",
            ".shx": "b5f52663ac6c905e8ff529f09d8c74db",
        },
    },
    "DE3": {
        "base": "BZR_config_12_DE3",
        "md5": {
            ".dbf": "2d16f00cdf277d4bfb616fa8544dedeb",
            ".prj": "c742bee3d4edfc2948a2ad08de1790a5",
            ".shp": "61a74508d5912859db2251de7fc94070",
            ".shx": "45609c8e76b0313884fe19d03f3c14a9",
        },
    },
    "DE4": {
        "base": "BZR_config_13_DE4",
        "md5": {
            ".dbf": "0e10162d94e445b1a66df2bfff937e75",
            ".prj": "c742bee3d4edfc2948a2ad08de1790a5",
            ".shp": "69020b6ecfa73ca27d2340909a27358d",
            ".shx": "b09d076bd243943076cbcfe80e5b5885",
        },
    },
    "DE5": {
        "base": "BZR_config_14_DE5",
        "md5": {
            ".dbf": "e8dbcb76ff1c774c96d24e5cd9bb45b2",
            ".prj": "c742bee3d4edfc2948a2ad08de1790a5",
            ".shp": "8608842d2f785ccac70215c4338e6d5b",
            ".shx": "005524edb5e03d339b5b992c75e2a960",
        },
    },
}


def _market_zone_cache_dir():
    """
    Package-relative cache directory:
    etrago/data/shapes_biddingzones
    """
    return Path(__file__).resolve().parents[1] / "data" / "shapes_biddingzones"


def _md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_market_zone_shapefile(market_zones):
    """
    Download the required shapefile components from Zenodo and return the .shp path.
    """
    if market_zones not in MARKET_ZONE_SHAPEFILES:
        raise ValueError(
            "Invalid market_zones setting. "
            "Allowed values are: 'DE2', 'DE3', 'DE4', 'DE5'."
        )

    cache_dir = _market_zone_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    info = MARKET_ZONE_SHAPEFILES[market_zones]
    base = info["base"]

    for ext, expected_md5 in info["md5"].items():
        filename = f"{base}{ext}"
        target = cache_dir / filename
        url = f"{ZENODO_MARKET_ZONE_RECORD}/{filename}?download=1"

        needs_download = True

        if target.exists():
            if _md5sum(target) == expected_md5:
                needs_download = False
            else:
                logger.warning(
                    "MD5 mismatch for %s. Re-downloading from Zenodo.",
                    target,
                )

        if needs_download:
            logger.info("Downloading %s from Zenodo.", filename)
            urlretrieve(url, target)

            actual_md5 = _md5sum(target)
            if actual_md5 != expected_md5:
                raise RuntimeError(
                    f"MD5 check failed for {filename}. "
                    f"Expected {expected_md5}, got {actual_md5}."
                )

    return cache_dir / f"{base}.shp"


def load_market_zones_from_zenodo(market_zones):
    """
    Load market-zone geometries for DE2, DE3, DE4 or DE5.
    """
    shapefile_path = download_market_zone_shapefile(market_zones)

    zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    zones = zones.explode(index_parts=False).reset_index(drop=True)

    return zones


def assign_market_zones_to_bus_dataframe(buses, zones):
    """
    Assign market-zone IDs to a bus DataFrame.

    Returns a GeoDataFrame with:
    - id
    - marketzone
    - cluster
    """
    geometry = [Point(xy) for xy in zip(buses["x"].values, buses["y"].values)]

    geo_buses = gpd.GeoDataFrame(
        buses.copy(),
        geometry=geometry,
        crs="EPSG:4326",
    )

    geo_buses = gpd.sjoin(
        geo_buses,
        zones[["geometry", "id"]],
        how="left",
        predicate="within",
    )

    # Assign nearest zone to German buses that are not inside a zone polygon.
    buses_no_zone = geo_buses[
        (geo_buses["country"] == "DE") & (geo_buses["id"].isna())
    ]

    if not buses_no_zone.empty:
        zones_projected = zones.to_crs(epsg=3035)
        buses_projected = geo_buses.to_crs(epsg=3035)

        for idx in buses_no_zone.index:
            distances = zones_projected.geometry.distance(
                buses_projected.loc[idx].geometry
            )
            nearest_zone_idx = distances.idxmin()
            geo_buses.at[idx, "id"] = zones.loc[nearest_zone_idx, "id"]

    def assign_zone(row):
        if pd.notnull(row["id"]):
            return f"Zone_{int(row['id'])}"
        return row["country"]

    geo_buses["marketzone"] = geo_buses.apply(assign_zone, axis=1)
    geo_buses["cluster"] = geo_buses.groupby("marketzone").ngroup()

    return geo_buses


def create_market_zone_busmap(net, market_zones):
    """
    Use inside build_market_model().

    Adds cluster column to net.buses and returns:
    - busmap
    - medoid_idx
    """
    zones = load_market_zones_from_zenodo(market_zones)
    geo_buses = assign_market_zones_to_bus_dataframe(net.buses, zones)

    net.buses["cluster"] = geo_buses["cluster"].reindex(net.buses.index)

    busmap = pd.Series(
        net.buses.cluster.astype(int).astype(str),
        net.buses.index,
    )

    medoid_idx = pd.Series(dtype=str)

    return busmap, medoid_idx


def assign_market_zone_column_to_network(network, market_zones):
    """
    Use inside standalone assign_market_zones_to_buses().

    Adds:
    - network.buses["zone"]
    - network.buses["marketzone"]
    """
    if market_zones == "none":
        print("market_zones='none': no market-zone shapefile assigned.")
        return network

    zones = load_market_zones_from_zenodo(market_zones)
    geo_buses = assign_market_zones_to_bus_dataframe(network.buses, zones)

    network.buses["zone"] = geo_buses["id"].reindex(network.buses.index)
    network.buses["marketzone"] = geo_buses["marketzone"].reindex(
        network.buses.index
    )

    missing = network.buses["zone"].isna().sum()
    print(f"{missing} Bussen konnte keine Zone zugewiesen werden.")

    return network
