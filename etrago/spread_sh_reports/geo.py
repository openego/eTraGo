"""Bidding-zone geometry, bus-to-zone assignment and map outlines."""

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from .config import COUNTRY_SHAPE, FOCUS_SHAPE, MODEL_COUNTRIES, ZONE_SHAPEFILES

DE_CENTER = (10.45, 51.16)
COMPASS = ["East", "North-east", "North", "North-west", "West", "South-west",
           "South", "South-east"]

# Map window covering the modelled countries (lon/lat).
MAP_EXTENT = {"de": (5.4, 15.6, 47.1, 55.3), "europe": (-8.5, 25.0, 42.0, 64.0)}
FOCUS_EXTENT = (7.8, 11.5, 53.3, 55.15)


@lru_cache(maxsize=None)
def countries(data_dir):
    gdf = gpd.read_file(Path(data_dir) / COUNTRY_SHAPE).to_crs(epsg=4326)
    code = gdf["ISO_A2_EH"].where(gdf["ISO_A2_EH"] != "-99", gdf["ISO_A2"])
    gdf["code"] = code.replace({"UK": "GB"})
    return gdf[["code", "NAME", "geometry"]]


FOCUS_DISTRICTS = (
    "Flensburg", "Kiel", "Lübeck", "Neumünster", "Dithmarschen",
    "Herzogtum Lauenburg", "Nordfriesland", "Ostholstein", "Pinneberg",
    "Plön", "Rendsburg-Eckernförde", "Schleswig-Flensburg", "Segeberg",
    "Steinburg", "Stormarn",
)


@lru_cache(maxsize=None)
def focus_outline(data_dir, districts=FOCUS_DISTRICTS):
    """Outline of the focus region (the districts given in args.json)."""
    gdf = gpd.read_file(Path(data_dir) / FOCUS_SHAPE).to_crs(epsg=4326)
    # vg250 holds all German districts; keep the Schleswig-Holstein ones.
    gdf = gdf[gdf["gen"].isin(districts) & (gdf["sn_l"] == "01")]
    geom = gdf.geometry.buffer(0).union_all() if hasattr(gdf.geometry, "union_all") \
        else gdf.geometry.buffer(0).unary_union
    return geom.simplify(0.01, preserve_topology=True)


def _compass(lon, lat):
    dx = (lon - DE_CENTER[0]) * np.cos(np.radians(lat))
    dy = lat - DE_CENTER[1]
    if np.hypot(dx, dy) < 0.6:
        return "Central"
    angle = (np.degrees(np.arctan2(dy, dx)) + 360 + 22.5) % 360
    return COMPASS[int(angle // 45)]


@lru_cache(maxsize=None)
def zones(data_dir, key):
    """Zone polygons for a configuration.

    Returns a GeoDataFrame with ``zone`` (stable id e.g. ``DE3-Z1``),
    ``label`` (readable name) and ``geometry``. For the status quo the single
    DE/LU zone is built from the country outlines.
    """
    data_dir = Path(data_dir)
    focus = focus_outline(data_dir)
    if key not in ZONE_SHAPEFILES:
        c = countries(data_dir)
        geom = c[c.code.isin(["DE", "LU"])].geometry.union_all() \
            if hasattr(c.geometry, "union_all") else c[c.code.isin(["DE", "LU"])].geometry.unary_union
        return gpd.GeoDataFrame(
            {"zone": ["DE/LU"], "label": ["DE/LU"], "shape_id": [0],
             "contains_focus": [True]},
            geometry=[geom], crs="EPSG:4326",
        )

    raw = gpd.read_file(data_dir / ZONE_SHAPEFILES[key]).to_crs(epsg=4326)
    raw = raw.dissolve(by="id", as_index=False)
    rows = []
    for _, row in raw.iterrows():
        geom = row.geometry.buffer(0)
        pt = geom.representative_point()
        share_focus = geom.intersection(focus).area / focus.area
        rows.append({
            "shape_id": int(row["id"]),
            "zone": f"{key}-Z{int(row['id'])}",
            "direction": _compass(pt.x, pt.y),
            "contains_focus": share_focus > 0.5,
            "geometry": geom,
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    labels = []
    for _, row in gdf.iterrows():
        name = row["direction"]
        if (gdf["direction"] == name).sum() > 1:
            name = f"{name} {row['shape_id']}"
        if row["contains_focus"]:
            name += " (SH)"
        labels.append(f"Z{row['shape_id']} {name}")
    gdf["label"] = labels
    return gdf[["zone", "label", "shape_id", "contains_focus", "geometry"]]


def assign_zones(buses, zone_gdf, key):
    """Return a Series bus -> zone id for AC buses.

    Mirrors ``etrago.tools.market_zones``: buses inside a zone polygon get
    that zone, German buses outside every polygon get the nearest zone,
    all others keep their country code. In the status quo DE and LU form
    the joint ``DE/LU`` zone.
    """
    buses = buses.copy()
    out = buses["country"].astype(str).copy()
    if key not in ZONE_SHAPEFILES:
        out[out.isin(["DE", "LU"])] = "DE/LU"
        return out
    pts = gpd.GeoDataFrame(
        buses[["country"]],
        geometry=[Point(xy) for xy in zip(buses["x"], buses["y"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, zone_gdf[["zone", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    inside = joined["zone"].dropna()
    out.loc[inside.index] = inside
    missing = joined.index[(joined["country"] == "DE") & joined["zone"].isna()]
    if len(missing):
        zp = zone_gdf.to_crs(epsg=3035)
        bp = pts.loc[missing].to_crs(epsg=3035)
        for idx, geom in bp.geometry.items():
            out.loc[idx] = zp.loc[zp.distance(geom).idxmin(), "zone"]
    return out


def is_german_zone(zone):
    return zone == "DE/LU" or "-Z" in str(zone)


def polygon_coords(geom, tolerance=0.02):
    """Flatten (Multi)Polygon exteriors into lon/lat lists separated by None."""
    if geom is None or geom.is_empty:
        return [], []
    geom = geom.simplify(tolerance, preserve_topology=True)
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    xs, ys = [], []
    for poly in polys:
        if not isinstance(poly, Polygon) or poly.area < 1e-3:
            continue
        x, y = poly.exterior.coords.xy
        xs += [round(v, 4) for v in x] + [None]
        ys += [round(v, 4) for v in y] + [None]
    return xs, ys


@lru_cache(maxsize=None)
def germany_land(data_dir):
    """Detailed German land area from the vg250 districts (land parts only)."""
    gdf = gpd.read_file(Path(data_dir) / FOCUS_SHAPE).to_crs(epsg=4326)
    if "gf" in gdf:
        gdf = gdf[gdf["gf"].isin([3, 4])]
    geom = gdf.geometry.buffer(0)
    geom = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    return geom.simplify(0.004, preserve_topology=True)


def background(data_dir, extent):
    """Country outlines clipped to the map window (Germany from vg250)."""
    from shapely.geometry import box

    window = box(extent[0] - 8, extent[2] - 6, extent[1] + 8, extent[3] + 6)
    c = countries(data_dir)
    shapes = []
    for _, row in c.iterrows():
        geom = row.geometry
        if row["code"] == "DE":
            geom = germany_land(data_dir)
        geom = geom.intersection(window)
        if geom.is_empty:
            continue
        xs, ys = polygon_coords(geom, 0.004 if row["code"] == "DE" else 0.01)
        shapes.append({"code": row["code"], "name": row["NAME"], "x": xs,
                       "y": ys, "model": row["code"] in MODEL_COUNTRIES})
    return shapes


def country_anchor(data_dir, code, fallback):
    """Representative point for a neighbouring market zone."""
    overrides = {"FR": (2.4, 47.0), "NO": (9.0, 60.6), "SE": (15.0, 59.5),
                 "GB": (-1.5, 52.6), "DK": (9.6, 55.9)}
    if code in overrides:
        return overrides[code]
    c = countries(data_dir)
    hit = c[c.code == code]
    if hit.empty:
        return fallback
    p = hit.geometry.iloc[0].representative_point()
    return (p.x, p.y)
