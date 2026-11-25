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


# Reconstructed zone files (T. Noldin, 11.11.2026): whole ding0 MV grid
# districts following their eGon2035 110 kV bus, zones numbered as in ACER
# Decision 11/2022 (J1 = 1, ..., J5 = 5). One GeoPackage per configuration.
RECONSTRUCTED_MARKET_ZONE_URL = (
    "https://cloud.fossilexit.net/s/CgggMcBSM4GsaxK/download"
    "?path=/20261112_BZR_configs_reconstructed&files={filename}"
)

RECONSTRUCTED_MARKET_ZONE_FILES = {
    "DE2": {
        "file": "BZR_config_2_DE2_rekonstruiert_20261111.gpkg",
        "md5": "9ea2a670903166c55dbb821adf28a3d0",
    },
    "DE3": {
        "file": "BZR_config_12_DE3_rekonstruiert_20261111.gpkg",
        "md5": "54a01a946d71908ba1857a4f5f70cd3d",
    },
    "DE4": {
        "file": "BZR_config_13_DE4_rekonstruiert_20261111.gpkg",
        "md5": "49ca5e3c4d8da020faa91ba7ef74d955",
    },
    "DE5": {
        "file": "BZR_config_14_DE5_rekonstruiert_20261111.gpkg",
        "md5": "67806fec745fbeafec6871efc627b59c",
    },
}

MARKET_ZONE_SOURCES = ("Basic", "Reconstructed")


def download_reconstructed_market_zone_file(market_zones):
    """
    Download the reconstructed GeoPackage of one configuration (MD5-checked)
    into etrago/data/shapes_biddingzones and return its path.
    """
    if market_zones not in RECONSTRUCTED_MARKET_ZONE_FILES:
        raise ValueError(
            "Invalid market_zones setting. "
            "Allowed values are: 'DE2', 'DE3', 'DE4', 'DE5'."
        )

    cache_dir = _market_zone_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    info = RECONSTRUCTED_MARKET_ZONE_FILES[market_zones]
    target = cache_dir / info["file"]

    if target.exists() and _md5sum(target) == info["md5"]:
        return target

    if target.exists():
        logger.warning("MD5 mismatch for %s. Downloading it again.", target)
    logger.info("Downloading %s.", info["file"])
    urlretrieve(
        RECONSTRUCTED_MARKET_ZONE_URL.format(filename=info["file"]), target
    )

    actual_md5 = _md5sum(target)
    if actual_md5 != info["md5"]:
        raise RuntimeError(
            f"MD5 check failed for {info['file']}. "
            f"Expected {info['md5']}, got {actual_md5}."
        )

    return target


def load_market_zones_from_zenodo(market_zones, source="Reconstructed"):
    """
    Load market-zone geometries for DE2, DE3, DE4 or DE5.

    source (case-insensitive): "Reconstructed" (default; GeoPackages of the
    reconstructed ACER zones, zones numbered J1 = 1, ...) or "Basic"
    (shapefiles of Zenodo record 14833526, used up to the 16.11.2026 round).
    """
    source = {s.lower(): s for s in MARKET_ZONE_SOURCES}.get(
        str(source).lower(), source
    )
    if source not in MARKET_ZONE_SOURCES:
        raise ValueError(
            "market_zone_shapes must be 'Basic' or 'Reconstructed', "
            f"found {source!r}"
        )

    if source == "Reconstructed":
        path = download_reconstructed_market_zone_file(market_zones)
    else:
        path = download_market_zone_shapefile(market_zones)

    logger.info(
        "Market zones %s from %s (source %s, md5 %s)",
        market_zones,
        path.name,
        source,
        _md5sum(path),
    )

    zones = gpd.read_file(path).to_crs(epsg=4326)
    zones["id"] = zones["id"].astype(int)
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


def create_market_zone_busmap(net, market_zones, source="Reconstructed"):
    """
    Use inside build_market_model().

    Adds cluster column to net.buses and returns:
    - busmap
    - medoid_idx
    """
    zones = load_market_zones_from_zenodo(market_zones, source=source)
    geo_buses = assign_market_zones_to_bus_dataframe(net.buses, zones)

    net.buses["cluster"] = geo_buses["cluster"].reindex(net.buses.index)

    busmap = pd.Series(
        net.buses.cluster.astype(int).astype(str),
        net.buses.index,
    )

    medoid_idx = pd.Series(dtype=str)

    return busmap, medoid_idx


def assign_market_zone_column_to_network(
    network, market_zones, source="Reconstructed"
):
    """
    Use inside standalone assign_market_zones_to_buses().

    Adds:
    - network.buses["zone"]
    - network.buses["marketzone"]
    """
    if market_zones == "none":
        print("market_zones='none': no market-zone shapefile assigned.")
        return network

    zones = load_market_zones_from_zenodo(market_zones, source=source)
    geo_buses = assign_market_zones_to_bus_dataframe(network.buses, zones)

    network.buses["zone"] = geo_buses["id"].reindex(network.buses.index)
    network.buses["marketzone"] = geo_buses["marketzone"].reindex(
        network.buses.index
    )

    missing = network.buses["zone"].isna().sum()
    print(f"{missing} Bussen konnte keine Zone zugewiesen werden.")

    return network
