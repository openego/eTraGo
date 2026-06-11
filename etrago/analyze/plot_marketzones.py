#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import numpy as np
from shapely.geometry import Point
import pandas as pd

from etrago.tools.market_zones import (
    load_market_zones_from_zenodo,
    assign_market_zone_column_to_network,
    assign_market_zones_to_bus_dataframe,
)


def _save_or_show_plot(filename=None, dpi=600):
    """
    Save the current matplotlib figure if filename is given.
    Otherwise show it on screen.
    """
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _load_zones_for_plot(market_zones):
    """
    Load DE2, DE3, DE4, or DE5 market-zone geometries from Zenodo.

    Returns None if market_zones is 'none'.
    """
    market_zones = str(market_zones).upper()

    if market_zones == "NONE":
        return None

    if market_zones in ["DE2", "DE3", "DE4", "DE5"]:
        return load_market_zones_from_zenodo(market_zones)

    raise ValueError(
        "Invalid value for market_zones. "
        "Allowed values are: 'DE2', 'DE3', 'DE4', 'DE5', or 'none'."
    )


def _add_zone_colors(zones):
    """
    Add a color column to the market-zone GeoDataFrame for plotting.
    """
    zones = zones.copy()
    colors = plt.cm.tab20(range(len(zones)))
    zones["color"] = [mcolors.rgb2hex(color[:3]) for color in colors]
    return zones


def assign_market_zones_to_buses(network, market_zones):
    """
    Assign DE2, DE3, DE4, or DE5 market zones to network.buses
    using Zenodo shapefiles.
    """
    return assign_market_zone_column_to_network(network, market_zones)


def _zone_key(value):
    """
    Convert zone IDs to stable comparable keys.
    """
    if pd.isna(value):
        return value

    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _safe_percentage(value, total):
    """
    Avoid division by zero in percentage calculations.
    """
    if total == 0:
        return 0.0

    return 100 * value / total


def plot_marketzone_clustering(
    self,
    market_zones,
    filename=None,
):
    """
    Plot market-zone clustering and average marginal prices.

    If market_zones is one of 'DE2', 'DE3', 'DE4', or 'DE5',
    the market-zone geometries are loaded from Zenodo.

    If market_zones is 'none', Germany is treated as one single zone.

    Parameters
    ----------
    self : pypsa.Network
        Usually etrago.market_model.
    market_zones : str
        'DE2', 'DE3', 'DE4', 'DE5', or 'none'.
    filename : str, optional
        If given, the map is saved to this file.
        The price-distribution plot is saved with '_price_distribution'
        added to the filename.
    """
    from pathlib import Path

    def _save_or_show(filename_to_use=None):
        plt.tight_layout()

        if filename_to_use is not None:
            plt.savefig(
                filename_to_use,
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def _derived_filename(filename_to_use, suffix):
        if filename_to_use is None:
            return None

        path = Path(filename_to_use)
        file_suffix = path.suffix if path.suffix else ".png"

        return path.with_name(f"{path.stem}{suffix}{file_suffix}")

    # ------------------------------------------------------------
    # 1. Load market-zone geometries from Zenodo
    # ------------------------------------------------------------
    zones = _load_zones_for_plot(market_zones)

    # ------------------------------------------------------------
    # 2. Prepare map
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10, 6),
        dpi=600,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # ------------------------------------------------------------
    # 3. Select German AC buses
    # ------------------------------------------------------------
    market_bus_de = self.buses[
        (self.buses.country == "DE") & (self.buses.carrier == "AC")
    ]

    gdf_buses = gpd.GeoDataFrame(
        market_bus_de,
        geometry=gpd.points_from_xy(
            market_bus_de.x,
            market_bus_de.y,
        ),
        crs="EPSG:4326",
    )

    average_prices = {}

    # ------------------------------------------------------------
    # 4. Plot zones and average marginal price per zone
    # ------------------------------------------------------------
    if zones is not None:
        zones = _add_zone_colors(zones)

        zones.boundary.plot(
            ax=ax,
            edgecolor="black",
            linewidth=0.5,
        )
        zones.plot(
            ax=ax,
            facecolor=zones["color"],
            alpha=0.3,
        )

        for _, zone in zones.iterrows():
            zone_id = _zone_key(zone["id"])

            buses_in_zone = gdf_buses[gdf_buses.within(zone["geometry"])]

            if buses_in_zone.empty:
                continue

            price_columns = self.buses_t.marginal_price.columns.intersection(
                buses_in_zone.index
            )

            if len(price_columns) == 0:
                continue

            prices_in_zone = self.buses_t.marginal_price[price_columns]
            avg_price = prices_in_zone.mean().mean()
            average_prices[zone_id] = avg_price

            center = zone["geometry"].centroid

            ax.text(
                center.x,
                center.y,
                f"{avg_price:.2f}",
                fontsize=7,
                ha="center",
                color="black",
            )

    # ------------------------------------------------------------
    # 5. Fallback for market_zones='none'
    # ------------------------------------------------------------
    else:
        price_columns = self.buses_t.marginal_price.columns.intersection(
            market_bus_de.index
        )

        prices_in_de = self.buses_t.marginal_price[price_columns]
        avg_price = prices_in_de.mean().mean()
        average_prices["Germany"] = avg_price

        center_lon = 10.4515
        center_lat = 51.1657

        ax.text(
            center_lon,
            center_lat,
            f"{avg_price:.2f}",
            fontsize=7,
            ha="center",
            color="black",
        )

    # ------------------------------------------------------------
    # 6. Plot network without visible bus/link sizes
    # ------------------------------------------------------------
    self.plot(
        ax=ax,
        link_widths=0,
        bus_sizes=0,
        line_widths=0.5,
        line_colors="grey",
    )

    ax.set_extent(
        [5.5, 15.5, 47, 55.5],
        crs=ccrs.PlateCarree(),
    )
    ax.axis("off")

    _save_or_show(filename)

    # ------------------------------------------------------------
    # 7. Price-distribution bar chart
    # ------------------------------------------------------------
    bins = [0, 5, 10, 40, 100, np.inf]
    labels = ["0-5", "6-10", "11-40", "41-100", ">100"]

    distribution_filename = _derived_filename(
        filename,
        "_price_distribution",
    )

    if zones is not None:
        zone_distributions = []
        zone_labels = []
        zone_colors = []

        for _, zone in zones.iterrows():
            zone_id = _zone_key(zone["id"])

            buses_in_zone = gdf_buses[gdf_buses.within(zone["geometry"])]

            if buses_in_zone.empty:
                continue

            price_columns = self.buses_t.marginal_price.columns.intersection(
                buses_in_zone.index
            )

            if len(price_columns) == 0:
                continue

            prices_in_zone = self.buses_t.marginal_price[price_columns]
            prices_in_zone = prices_in_zone.values.flatten()
            prices_in_zone = prices_in_zone[~np.isnan(prices_in_zone)]

            price_distribution, _ = np.histogram(
                prices_in_zone,
                bins=bins,
            )

            zone_distributions.append(price_distribution)
            zone_labels.append(f"Zone {zone_id}")
            zone_colors.append(zone["color"])

        if zone_distributions:
            x = np.arange(len(labels))
            width = 0.8 / len(zone_distributions)

            fig2, ax2 = plt.subplots(
                figsize=(12, 8),
            )

            for i, (distribution, label, color) in enumerate(
                zip(
                    zone_distributions,
                    zone_labels,
                    zone_colors,
                )
            ):
                offset = width * i

                ax2.bar(
                    x + offset,
                    distribution,
                    width,
                    label=label,
                    color=color,
                    alpha=0.6,
                )

            ax2.set_ylabel(
                "Number of hours",
                fontsize=30,
            )
            ax2.set_ylim(0, 4200)
            ax2.set_yticks(
                np.arange(0, 4001, 500),
            )
            ax2.yaxis.grid(
                True,
                linestyle="--",
                linewidth=0.8,
                color="gray",
                alpha=0.3,
            )

            ax2.set_xticks(
                x + width * (len(zone_distributions) - 1) / 2,
            )
            ax2.set_xticklabels(labels)
            ax2.tick_params(
                axis="both",
                which="major",
                labelsize=25,
            )
            ax2.legend(fontsize=12)

            plt.xticks(rotation=45)
            _save_or_show(distribution_filename)

    else:
        price_columns = self.buses_t.marginal_price.columns.intersection(
            market_bus_de.index
        )

        prices_in_de = self.buses_t.marginal_price[price_columns]
        prices_in_de = prices_in_de.values.flatten()
        prices_in_de = prices_in_de[~np.isnan(prices_in_de)]

        price_distribution, _ = np.histogram(
            prices_in_de,
            bins=bins,
        )

        fig2, ax2 = plt.subplots(
            figsize=(10, 6),
        )

        ax2.bar(
            labels,
            price_distribution,
            width=0.6,
            color="steelblue",
            alpha=0.7,
        )

        ax2.set_ylabel(
            "Number of hours",
            fontsize=25,
        )
        ax2.set_ylim(0, 4200)
        ax2.set_yticks(
            np.arange(0, 4001, 500),
        )
        ax2.yaxis.grid(
            True,
            linestyle="--",
            linewidth=0.8,
            color="gray",
            alpha=0.3,
        )
        ax2.tick_params(
            axis="both",
            which="major",
            labelsize=20,
        )

        plt.xticks(rotation=45)
        _save_or_show(distribution_filename)

    return average_prices


def total_dispatch_by_zone(
    self,
    timesteps=None,
    market_zones="DE4",
    filename=None,
):
    """
    Calculate electricity production per carrier and German market zone
    and plot the result on a map.

    This function is intended for the market model, for example:

        total_dispatch_by_zone(
            etrago.market_model,
            market_zones="DE5",
            filename="results_DE5_test/plots/total_dispatch_by_zone_DE5.png",
        )

    Unlike the previous version, this function does not call
    german_network(self). This avoids PyPSA snapshot-index errors when the
    network contains time-series tables with different snapshot indices.

    Parameters
    ----------
    self : pypsa.Network
        Usually etrago.market_model.
    timesteps : range or list, optional
        Timesteps used for dispatch aggregation. If None, all available
        snapshots of the network are used.
    market_zones : str
        'DE2', 'DE3', 'DE4', or 'DE5'.
    filename : str, optional
        If given, the plot is saved to this file.

    Returns
    -------
    pandas.Series
        Dispatch per zone and carrier in TWh.
    """
    from etrago.analyze.plot import calc_dispatch_per_carrier

    market_zones = str(market_zones).upper()

    if market_zones not in ["DE2", "DE3", "DE4", "DE5"]:
        raise ValueError(
            "Invalid value for market_zones. "
            "Allowed values are: 'DE2', 'DE3', 'DE4', 'DE5'."
        )

    if timesteps is None:
        timesteps = range(len(self.snapshots))

    # Assign market-zone information directly to the provided PyPSA network.
    # This modifies self.buses by adding at least the columns 'zone'
    # and 'marketzone'.
    assign_market_zones_to_buses(
        self,
        market_zones,
    )

    if "zone" not in self.buses.columns:
        raise ValueError(
            "Column 'zone' is missing in network.buses. "
            "Market-zone assignment failed."
        )

    # Work only with German buses. This replaces german_network(self),
    # but avoids copying the whole PyPSA network and therefore avoids
    # snapshot-index mismatch errors.
    buses_de = self.buses[self.buses["country"] == "DE"].copy()

    if buses_de.empty:
        raise ValueError(
            "No German buses found in the network. "
            "Cannot calculate dispatch by German market zone."
        )

    missing_zone_buses = buses_de[buses_de["zone"].isna()]

    if not missing_zone_buses.empty:
        examples = list(missing_zone_buses.index[:10])

        raise ValueError(
            f"{len(missing_zone_buses)} German buses have no market-zone "
            f"assignment. Examples: {examples}"
        )

    buses_de["_zone_key"] = buses_de["zone"].apply(_zone_key)

    # Calculate dispatch for the full network first, then keep only German
    # buses. This is safer than making a copied German subnetwork.
    dispatch_series = calc_dispatch_per_carrier(
        self,
        timesteps,
        dispatch_type="total",
    )

    dispatch_df = dispatch_series.reset_index()
    dispatch_df.columns = [
        "bus",
        "carrier",
        "dispatch",
    ]

    dispatch_df = dispatch_df[dispatch_df["bus"].isin(buses_de.index)]

    dispatch_df = dispatch_df.merge(
        buses_de[["_zone_key"]],
        left_on="bus",
        right_index=True,
        how="left",
    )

    dispatch_df = dispatch_df.rename(
        columns={
            "_zone_key": "zone",
        }
    )

    dispatch_df = dispatch_df[dispatch_df["zone"].notna()]

    if dispatch_df.empty:
        raise ValueError(
            "No German dispatch data could be matched to market zones."
        )

    # Keep the original scaling logic:
    # dispatch is converted to TWh using * 5 / 1e6.
    # This is consistent with the existing eTraGo post-processing logic.
    dispatch_per_zone = (
        dispatch_df.groupby(["zone", "carrier"])["dispatch"].sum() * 5 / 1e6
    )

    table = dispatch_per_zone.unstack().fillna(0)

    renewables = [
        "solar",
        "solar_rooftop",
        "wind_offshore",
        "wind_onshore",
        "reservoir",
        "run_of_river",
        "biomass",
        "central_biomass_CHP",
        "industrial_biomass_CHP",
    ]

    print("\nElectricity production per zone:")

    for zone in table.index:
        total = table.loc[zone].sum()

        renewable_generation = (
            table.loc[zone]
            .reindex(
                renewables,
                fill_value=0,
            )
            .sum()
        )

        renewable_share = _safe_percentage(
            renewable_generation,
            total,
        )

        print(
            f"{zone}: "
            f"{total:.2f} TWh total, "
            f"{renewable_generation:.2f} TWh renewable "
            f"({renewable_share:.1f}%)"
        )

    zones = _load_zones_for_plot(market_zones)
    zones = _add_zone_colors(zones)

    fig, ax = plt.subplots(
        figsize=(10, 6),
        dpi=600,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    zones.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )

    zones.plot(
        ax=ax,
        facecolor=zones["color"],
        alpha=0.3,
    )

    self.plot(
        ax=ax,
        link_widths=0,
        bus_sizes=0,
    )

    for _, row in zones.iterrows():
        zone_name = _zone_key(row["id"])

        if zone_name not in table.index:
            continue

        total = table.loc[zone_name].sum()

        renewable_generation = (
            table.loc[zone_name]
            .reindex(
                renewables,
                fill_value=0,
            )
            .sum()
        )

        renewable_share = _safe_percentage(
            renewable_generation,
            total,
        )

        centroid = row.geometry.centroid

        ax.text(
            centroid.x,
            centroid.y,
            f"{total:.1f} TWh\n{renewable_share:.1f}% RES",
            fontsize=18,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.3",
                alpha=0.9,
            ),
        )

    ax.set_extent(
        [5.5, 15.5, 47, 55.5],
        crs=ccrs.PlateCarree(),
    )

    ax.axis("off")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(
            filename,
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()

    return dispatch_per_zone


def calculate_inter_zonal_trade(network, market_zones):
    """
    Calculate electricity trade between German bidding zones and neighboring
    countries.

    Market-zone geometries are loaded from Zenodo.
    """
    market_zones = str(market_zones).upper()

    if market_zones not in ["DE2", "DE3", "DE4", "DE5"]:
        raise ValueError(
            "Invalid value for market_zones. "
            "Allowed values are: 'DE2', 'DE3', 'DE4', 'DE5'."
        )

    zones = _load_zones_for_plot(market_zones)

    geo_buses = assign_market_zones_to_bus_dataframe(
        network.buses,
        zones,
    )

    def assign_zone(row):
        if row["country"] == "DE":
            if pd.notnull(row["id"]):
                return f"DE{int(row['id'])}"
            return "DE"

        return row["country"]

    geo_buses["marketzone"] = geo_buses.apply(
        assign_zone,
        axis=1,
    )

    network.buses["marketzone"] = geo_buses["marketzone"].reindex(
        network.buses.index,
    )

    result = {
        f"DE{i}": pd.Series(dtype=float)
        for i in range(1, int(market_zones[-1]) + 1)
    }

    for _, line in network.lines.iterrows():
        bus0_zone = network.buses.loc[line["bus0"], "marketzone"]
        bus1_zone = network.buses.loc[line["bus1"], "marketzone"]

        if bus0_zone == bus1_zone:
            continue

        flow = (
            network.lines_t.p0[line.name]
            .mul(network.snapshot_weightings.generators)
            .sum()
        ) * 1e-6

        if str(bus0_zone).startswith("DE"):
            result[bus0_zone][bus1_zone] = (
                result[bus0_zone].get(bus1_zone, 0) + flow
            )
        elif str(bus1_zone).startswith("DE"):
            result[bus1_zone][bus0_zone] = (
                result[bus1_zone].get(bus0_zone, 0) - flow
            )

    return result


def plot_zone_net_flows(self, market_zones, filename=None):
    """
    Plot net electricity flows for German market zones.

    For DE2, DE3, DE4, and DE5, market-zone geometries are loaded from
    Zenodo via _load_zones_for_plot(). For 'none', Germany is treated as
    one single zone.

    Parameters
    ----------
    self : pypsa.Network
        Network object.
    market_zones : str
        Market-zone configuration: 'DE2', 'DE3', 'DE4', 'DE5', or 'none'.
    """
    from matplotlib.patches import Patch

    market_zones = str(market_zones).upper()

    if market_zones not in ["DE2", "DE3", "DE4", "DE5", "NONE"]:
        raise ValueError(
            "Invalid value for market_zones. "
            "Allowed values are: 'DE2', 'DE3', 'DE4', 'DE5', or 'none'."
        )

    # ------------------------------------------------------------
    # 1. Load market-zone geometries from Zenodo
    # ------------------------------------------------------------
    if market_zones != "NONE":
        zones = _load_zones_for_plot(market_zones)
        plot_zones = zones.copy()
    else:
        zones = None
        plot_zones = None

    # ------------------------------------------------------------
    # 2. Assign buses to market zones
    # ------------------------------------------------------------
    if market_zones != "NONE":
        geo_buses = assign_market_zones_to_bus_dataframe(
            self.buses,
            zones,
        )

        def assign_zone(row):
            if row["country"] == "DE" and pd.notnull(row["id"]):
                return f"DE{int(row['id'])}"
            return "Other"

        geo_buses["marketzone"] = geo_buses.apply(
            assign_zone,
            axis=1,
        )

        self.buses["marketzone"] = geo_buses["marketzone"].reindex(
            self.buses.index,
        )

        zone_ids = sorted(
            int(zone_id) for zone_id in zones["id"].dropna().unique()
        )
        german_zones = [f"DE{zone_id}" for zone_id in zone_ids]

    else:
        geometry = [Point(xy) for xy in zip(self.buses["x"], self.buses["y"])]

        geo_buses = gpd.GeoDataFrame(
            self.buses,
            geometry=geometry,
            crs="EPSG:4326",
        )

        self.buses["marketzone"] = self.buses["country"].apply(
            lambda country: "DE" if country == "DE" else "Other"
        )

        german_zones = ["DE"]

    ac_flows = {zone: 0.0 for zone in german_zones}
    dc_flows = {zone: 0.0 for zone in german_zones}

    # ------------------------------------------------------------
    # 3. Calculate AC net flows
    # ------------------------------------------------------------
    for line_idx, line in self.lines.iterrows():
        try:
            bus0_zone = self.buses.loc[line["bus0"], "marketzone"]
            bus1_zone = self.buses.loc[line["bus1"], "marketzone"]
        except KeyError:
            continue

        if bus0_zone not in german_zones and bus1_zone not in german_zones:
            continue

        if bus0_zone == bus1_zone:
            continue

        if bus0_zone in german_zones and bus1_zone == "Other":
            flow = (
                self.lines_t.p0[line_idx]
                .mul(self.snapshot_weightings.generators)
                .sum()
            ) * 1e-6
            ac_flows[bus0_zone] += flow

        elif bus0_zone == "Other" and bus1_zone in german_zones:
            flow = (
                self.lines_t.p1[line_idx]
                .mul(self.snapshot_weightings.generators)
                .sum()
            ) * 1e-6
            ac_flows[bus1_zone] -= flow

        elif bus0_zone in german_zones and bus1_zone in german_zones:
            flow = (
                self.lines_t.p0[line_idx]
                .mul(self.snapshot_weightings.generators)
                .sum()
            ) * 1e-6
            ac_flows[bus0_zone] += flow
            ac_flows[bus1_zone] -= flow

    # ------------------------------------------------------------
    # 4. Calculate DC net flows
    # ------------------------------------------------------------
    dc_links = self.links[self.links.carrier == "DC"]

    for link_idx, link in dc_links.iterrows():
        try:
            bus0_zone = self.buses.loc[link["bus0"], "marketzone"]
            bus1_zone = self.buses.loc[link["bus1"], "marketzone"]
        except KeyError:
            continue

        if bus0_zone not in german_zones and bus1_zone not in german_zones:
            continue

        if bus0_zone == bus1_zone:
            continue

        if bus0_zone in german_zones and bus1_zone == "Other":
            flow = (
                self.links_t.p0[link_idx]
                .mul(self.snapshot_weightings.generators)
                .sum()
            ) * 1e-6
            dc_flows[bus0_zone] += flow

        elif bus0_zone == "Other" and bus1_zone in german_zones:
            flow = (
                self.links_t.p1[link_idx]
                .mul(self.snapshot_weightings.generators)
                .sum()
            ) * 1e-6
            dc_flows[bus1_zone] -= flow

        elif bus0_zone in german_zones and bus1_zone in german_zones:
            flow = (
                self.links_t.p0[link_idx]
                .mul(self.snapshot_weightings.generators)
                .sum()
            ) * 1e-6
            dc_flows[bus0_zone] += flow
            dc_flows[bus1_zone] -= flow

    net_flows = {
        zone: ac_flows[zone] + dc_flows[zone] for zone in german_zones
    }

    # ------------------------------------------------------------
    # 5. Create map
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(12, 8),
        dpi=600,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    ax.set_extent(
        [-2.5, 16, 46.8, 58],
        crs=ccrs.PlateCarree(),
    )

    # ------------------------------------------------------------
    # 6. Plot market-zone geometries
    # ------------------------------------------------------------
    if plot_zones is not None:
        plot_zones = _add_zone_colors(plot_zones)

        plot_zones.boundary.plot(
            ax=ax,
            edgecolor="black",
            linewidth=0.5,
        )
        plot_zones.plot(
            ax=ax,
            facecolor=plot_zones["color"],
            alpha=0.2,
        )

    # ------------------------------------------------------------
    # 7. Plot network
    # ------------------------------------------------------------
    self.plot(
        ax=ax,
        line_widths=0,
        link_widths=0,
        bus_sizes=0,
    )

    # ------------------------------------------------------------
    # 8. Add net-flow labe```ls
    # ------------------------------------------------------------
    if plot_zones is not None:
        for _, zone in plot_zones.iterrows():
            zone_id = f"DE{int(zone['id'])}"

            if zone_id not in net_flows:
                continue

            net_flow = net_flows[zone_id]
            color = "green" if net_flow >= 0 else "red"
            center = zone["geometry"].centroid

            ax.text(
                center.x,
                center.y,
                f"{abs(net_flow):.1f} TWh",
                fontsize=9,
                ha="center",
                va="center",
                color=color,
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round",
                    alpha=0.8,
                ),
            )

    else:
        net_flow = dc_export_de_lu(self)
        german_buses = geo_buses[geo_buses["country"] == "DE"]

        de_center_x = german_buses.geometry.x.mean()
        de_center_y = german_buses.geometry.y.mean()

        color = "green" if net_flow >= 0 else "red"

        ax.text(
            de_center_x,
            de_center_y,
            f"{abs(net_flow * 1e-6):.1f} TWh",
            fontsize=9,
            ha="center",
            va="center",
            color=color,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round",
                alpha=0.8,
            ),
        )

    # ------------------------------------------------------------
    # 9. Legend and layout
    # ------------------------------------------------------------
    legend_elements = [
        Patch(
            facecolor="green",
            edgecolor="black",
            label="Net export",
        ),
        Patch(
            facecolor="red",
            edgecolor="black",
            label="Net import",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.42, 0.55),
        fontsize=10,
    )

    ax.axis("off")
    _save_or_show_plot(filename)

    return net_flows


def ac_export_per_country(self):
    """Calculate electricity exports and imports over AC lines per country

    Returns
    -------
    pd.Series
        Electricity export (if positive) or import (if negative) from DE+LU to each neighboring country in TWh
    """
    # Buses in DE or LU zählen als 'Exportland'
    de_buses = self.network.buses[
        self.network.buses.country.isin(["DE", "LU"])
    ]
    for_buses = self.network.buses[
        ~self.network.buses.country.isin(["DE", "LU"])
    ]

    result = pd.Series(index=for_buses.country.unique(), dtype=float)

    for c in for_buses.country.unique():
        target_buses = for_buses[for_buses.country == c].index

        exp = self.network.lines[
            (self.network.lines.bus0.isin(de_buses.index))
            & (self.network.lines.bus1.isin(target_buses))
        ]
        imp = self.network.lines[
            (self.network.lines.bus1.isin(de_buses.index))
            & (self.network.lines.bus0.isin(target_buses))
        ]

        exp_sum = (
            self.network.lines_t.p0[exp.index]
            .sum(axis=1)
            .mul(self.network.snapshot_weightings.generators)
            .sum()
        )

        imp_sum = (
            self.network.lines_t.p1[imp.index]
            .sum(axis=1)
            .mul(self.network.snapshot_weightings.generators)
            .sum()
        )

        result[c] = (exp_sum + imp_sum) * 1e-6  # in TWh

    return result


def dc_export_per_country(self):
    """Calculate electricity exports and imports over DC lines per country

    Returns
    -------
    float
        Electricity export (if negative: import) from Germany in TWh

    """
    de_buses = self.buses[self.buses.country == "DE"]

    for_buses = self.buses[self.buses.country != "DE"]

    result = pd.Series(index=for_buses.country.unique())

    for c in for_buses.country.unique():
        exp = self.links[
            (self.links.carrier == "DC")
            & (self.links.bus0.isin(de_buses.index))
            & (self.links.bus1.isin(for_buses[for_buses.country == c].index))
        ]
        imp = self.links[
            (self.links.carrier == "DC")
            & (self.links.bus1.isin(de_buses.index))
            & (self.links.bus0.isin(for_buses[for_buses.country == c].index))
        ]

        result[c] = (
            self.links_t.p0[exp.index]
            .sum(axis=1)
            .mul(self.snapshot_weightings.generators)
            .sum()
            + self.links_t.p1[imp.index]
            .sum(axis=1)
            .mul(self.snapshot_weightings.generators)
            .sum()
        ) * 1e-6

    return result


def plot_country_exports_per_configuration(
    market_sq, market_DE2, market_DE3, market_DE4
):
    """

    Plottet für jedes Nachbarland den Nettoexport Deutschlands (AC + DC) für verschiedene Modellkonfigurationen.
    Nur Länder mit Handelswerten ungleich Null werden dargestellt.
    BE, GB, NO und RU werden explizit ausgeschlossen.

    Parameters
    ----------
    etrago_status_quo : etrago object
    etrago_DE2 : etrago object
    etrago_DE3 : etrago object
    etrago_DE4 : etrago object
    etrago_nodal : etrago object
    """

    def get_exports(market):
        exports = dc_export_per_country(market)
        return exports.drop(
            labels=["DE", "LU", "GB", "NO", "RU"], errors="ignore"
        )

    configs = {
        "Status Quo": market_sq,
        "DE2": market_DE2,
        "DE3": market_DE3,
        "DE4": market_DE4,
        # "Nodal": etrago_nodal
    }

    # Exporte je Konfiguration berechnen
    all_exports = {}
    all_countries = set()

    for name, model in configs.items():
        try:
            exports = get_exports(model)
            # Nur Länder mit Werten ungleich Null behalten
            exports = exports[exports != 0]
            all_exports[name] = exports
            all_countries.update(exports.index)
        except Exception as e:
            print(f"⚠️ Fehler bei {name}: {e}")
            all_exports[name] = pd.Series()

    # Nur Länder, die in mindestens einer Konfiguration vorkommen
    all_countries = sorted(all_countries)
    df = pd.DataFrame(index=all_countries, columns=configs.keys())

    for name, exports in all_exports.items():
        df[name] = exports.reindex(all_countries)

    # Entferne Zeilen, die in allen Spalten Null sind
    df = df.loc[~(df == 0).all(axis=1)]

    # Falls keine Daten übrig sind, abbrechen
    if df.empty:
        print("Keine Handelsdaten zum Plotten verfügbar.")
        return

    # Erstelle eine Farbpalette mit Blautönen
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, len(df.columns)))
    colors = [mcolors.rgb2hex(color) for color in blues]

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(14, 6))

    # Position der Balken
    bar_width = 0.15
    x = np.arange(len(df.index))

    # Plotten der Balken
    for i, (config, color) in enumerate(zip(df.columns, colors)):
        ax.bar(
            x + i * bar_width,
            df[config],
            width=bar_width,
            label=config,
            color=color,
        )

    # Achsenbeschriftungen und Titel
    ax.set_xticks(x + (len(df.columns) - 1) * bar_width / 2)
    ax.set_xticklabels(df.index, fontsize=16)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("TWh", fontsize=18)

    ax.set_yticks(
        [-60, -40, -20, 0, 20, 40, 60]
    )  # Hier die gewünschten y-Werte angeben
    ax.set_yticklabels(
        [-60, -40, -20, 0, 20, 40, 60], fontsize=16
    )  # Hier die gewünschten Beschriftungen angeben

    # Legende
    ax.legend(loc="upper left", fontsize=16)

    # Grid für bessere Lesbarkeit
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Anpassung der x-Achse für bessere Lesbarkeit
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_dispatch_difference_by_bus(
    networkA,
    networkB,
    carrier="wind_offshore",
    market_zones="DE3",
    filename=None,
    base_scaling=0.5,  # Nutzerdefinierbare Skalierungsbasis
):
    """
    Plottet die ungewichtete Dispatch-Differenz (Summe über 8760h) eines Carriers pro Bus zwischen zwei Netzwerken.
    Farben: grün = höherer Dispatch in A, rot = niedrigerer Dispatch in A.
    Die Kreisgröße wird automatisch auf die maximal auftretende Differenz skaliert.
    """
    import matplotlib.patches as mpatches
    import pandas as pd

    # A: Generatoren mit diesem Carrier aus beiden Netzwerken
    gensA = networkA.generators.query("carrier == @carrier")
    gensB = networkB.generators.query("carrier == @carrier")

    # Nur Generatoren mit gleichem Index in beiden Netzen vergleichen
    common_index = gensA.index.intersection(gensB.index)
    if common_index.empty:
        print("⚠️ Keine gemeinsamen Generatoren gefunden!")
        return

    # Dispatch-Zeitreihen
    pA = networkA.generators_t.p[common_index]
    pB = networkB.generators_t.p[common_index]

    # Differenz über 8760 Stunden summieren
    diff = pA.sum(axis=0) - pB.sum(axis=0)  # Series mit Generator-Index

    # Mapping Generator → Bus
    bus_map = gensA.loc[common_index, "bus"]
    dispatch_grouped = diff.groupby(bus_map).sum()

    # Farben nach Vorzeichen
    colors_buses = {
        bus: (
            mcolors.to_rgba("green", alpha=0.2)
            if val > 0
            else mcolors.to_rgba("red", alpha=0.2)
        )
        for bus, val in dispatch_grouped.items()
    }
    # Absolutwerte für Kreisgrößen
    dispatch_abs = dispatch_grouped.abs()

    # ❗ automatische Skalierung
    scaling = 1 / dispatch_abs.max() * base_scaling

    # Optional: Hintergrundkarte mit Marktzonen
    zones = _load_zones_for_plot(market_zones)

    # Plot vorbereiten
    fig, ax = plt.subplots(
        figsize=(10, 6), dpi=300, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    if zones is not None:
        zones = _add_zone_colors(zones)

        # Zonen plotten
        zones.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)
        zones.plot(ax=ax, facecolor=zones["color"], alpha=0.3)

    # Netzelemente plotten (aus networkA)
    networkA.plot(
        geomap=True,
        bus_sizes=dispatch_abs * scaling,
        bus_colors=colors_buses,
        line_widths=0,
        link_widths=0,
        margin=0.01,
        ax=ax,
    )

    ax.set_title(f"Dispatch-Differenz {carrier} pro Bus")
    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())

    # Farben-Legende
    patch_green = mpatches.Patch(
        color="green", alpha=0.5, label="Reduktion Dispatch"
    )
    patch_red = mpatches.Patch(
        color="red", alpha=0.5, label="Erhöhung Dispatch"
    )

    # ax.legend(handles=[patch_green, patch_red], loc='upper left')

    if filename:
        plt.savefig(f"{filename}.png", bbox_inches="tight")
        print(f"✅ Plot gespeichert unter {filename}.png")
    else:
        plt.show()


def dc_export_de_lu(self):
    """
    Calculate electricity exports and imports over DC lines

    Returns
    -------
    float
        Electricity export (if negative: import) from Germany
    """

    network = self
    de_buses = network.buses[
        (network.buses.country == "DE") | (network.buses.country == "LU")
    ]

    for_buses = network.buses[~network.buses.country.isin(["DE", "LU"])]

    exp = network.links[
        (network.links.carrier == "DC")
        & (network.links.bus0.isin(de_buses.index))
        & (network.links.bus1.isin(for_buses.index))
    ]
    imp = network.links[
        (network.links.carrier == "DC")
        & (network.links.bus1.isin(de_buses.index))
        & (network.links.bus0.isin(for_buses.index))
    ]
    return (
        network.links_t.p0[exp.index]
        .sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
        + network.links_t.p1[imp.index]
        .sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
    )
