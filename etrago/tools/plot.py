# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
"""
Plot.py defines functions necessary to plot results of eTraGo.
"""
import logging
import os
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import pandas as pd
import numpy as np
from math import sqrt, log10
from pyproj import Proj, transform
import tilemapbase

cartopy_present = True
try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.mpl.geoaxes
    import requests
except ImportError:
    cartopy_present = False
import geopandas as gpd
from pypsa.plot import draw_map_cartopy
from shapely.geometry import LineString, MultiPoint, Point, Polygon

logger = logging.getLogger(__name__)

if "READTHEDOCS" not in os.environ:
    from geoalchemy2.shape import to_shape

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, MarlonSchlemminger, mariusves, lukasol"


def set_epsg_network(network):
    """
    Change EPSG from 4326 to 3857. Needed when using osm-background.

    Parameters
    ----------
    network : PyPSA network container

    Returns
    -------
    """

    inProj = Proj(init="epsg:4326")
    outProj = Proj(init="epsg:3857")
    x1, y1 = network.buses.x.values, network.buses.y.values
    x2, y2 = transform(inProj, outProj, x1, y1)
    network.buses.x, network.buses.y = x2, y2
    network.epsg = 3857
    set_epsg_network.counter = set_epsg_network.counter + 1


def plot_osm(x, y, zoom, alpha=0.4):
    """
    Plots openstreetmap as background of network-plots

    Parameters
    ----------
    x : array of two floats
        Define x-axis boundaries (lat) of osm plot
    y : array of two floats
        Define y-axis boundaries (long) of osm plot
    zoom : int
        Define zoom of osm, higher values for higher resolution
    alpha : float
        Sets osm-visibility, increase value if osm covers network-plot

    Returns
    -------
    """

    tilemapbase.init(create=True)

    extent = tilemapbase.Extent.from_lonlat(x[0], x[1], y[0], y[1])
    extent = extent.to_aspect(1.0)
    extent = extent.to_project_3857()

    fig, ax = plt.subplots()
    plt.axis("off")
    plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), zoom=zoom)
    plotter.plot(ax, alpha=alpha)
    # ax.plot(x, y, "ro-")
    return fig, ax


def coloring():

    colors = {
        "load": "red",
        "DC": "blue",
        "power_to_H2": "cyan",
        "H2_overground": "cyan",
        "H2_underground": "cyan",
        "H2": "cyan",
        "dsm-cts": "dodgerblue",
        "dsm-ind-osm": "dodgerblue",
        "dsm-ind-sites": "dodgerblue",
        "dsm": "dodgerblue",
        "central_heat_pump": "mediumpurple",
        "central_resistive_heater": "blueviolet",
        "rural_heat_pump": "violet",
        "CH4": "yellow",
        "CH4_to_H2": "yellowgreen",
        "industrial_gas_CHP": "olive",
        "rural_gas_boiler": "sandybrown",
        "central_gas_CHP": "darkorange",
        "central_gas_CHP_heat": "darkorange",
        "central_gas_boiler": "saddlebrown",
        "OCGT": "seagreen",
        "H2_to_power": "darkcyan",
        "H2_feedin": "lime",
        "H2_to_CH4": "seagreen",
        "central_heat_store_charger": "firebrick",
        "central_heat_store": "firebrick",
        "rural_heat_store_charger": "salmon",
        "rural_heat_store": "salmon",
        "central_heat_store_discharger": "firebrick",
        "rural_heat_store_discharger": "salmon",
        "rural_heat": "orange",
        "central_heat": "orangered",
        "H2_grid": "green",
        "H2_saltcavern": "darkgreen",
        "central_heat_store": "firebrick",
        "rural_heat_store": "salmon",
        "AC": "blue",
        "nuclear": "palegreen",
        "oil": "silver",
        "other_non_renewable": "dimgrey",
        "other_renewable": "lightsteelblue",
        "reservoir": "indigo",
        "run_of_river": "slateblue",
        "solar": "gold",
        "wind_offshore": "lightblue",
        "wind_onshore": "blue",
        "coal": "grey",
        "lignite": "brown",
        "biomass": "olive",
        "solar_thermal_collector": "wheat",
        "geo thermal": "peru",
        "load shedding": "black",
        "central_biomass_CHP": "darkorange",
        "industrial_biomass_CHP": "darkorange",
        "solar_rooftop": "goldenrod",
        "gas": "yellow",
        "central_biomass_CHP_heat": "darkorange",
        "geo_thermal": "peru",
        "battery": "blue",
        "pumped_hydro": "indigo",
        "BEV charger": "indigo",
    }

    return colors


def plot_line_loading_diff(networkA, networkB, timestep=0, osm=False):
    """
    Plot difference in line loading between two networks
    (with and without switches) as color on lines

    Positive values mean that line loading with switches is bigger than without
    Plot switches as small dots

    Parameters
    ----------
    networkA : PyPSA network container
        Holds topology of grid with switches
        including results from powerflow analysis
    networkB : PyPSA network container
        Holds topology of grid without switches
        including results from powerflow analysis
    filename : str
        Specify filename
        If not given, figure will be show directly
    timestep : int
        timestep to show, default is 0
    osm : bool or dict, e.g. {'x': [1,20], 'y': [47, 56], 'zoom' : 6}
        If not False, osm is set as background
        with the following settings as dict:
                'x': array of two floats, x axis boundaries (lat)
                'y': array of two floats, y axis boundaries (long)
                'zoom' : resolution of osm
    """
    if osm != False:
        if set_epsg_network.counter == 0:
            set_epsg_network(networkA)
            set_epsg_network(networkB)
        plot_osm(osm["x"], osm["y"], osm["zoom"])
    # new colormap to make sure 0% difference has the same color in every plot
    def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
        """
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero

        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        """
        cdict = {"red": [], "green": [], "blue": [], "alpha": []}

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack(
            [
                np.linspace(0.0, midpoint, 128, endpoint=False),
                np.linspace(midpoint, 1.0, 129, endpoint=True),
            ]
        )

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict["red"].append((si, r, r))
            cdict["green"].append((si, g, g))
            cdict["blue"].append((si, b, b))
            cdict["alpha"].append((si, a, a))

        newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)

        return newcmap

    # calculate difference in loading between both networks
    loading_switches = abs(
        networkA.lines_t.p0.mul(networkA.snapshot_weightings, axis=0)
        .loc[networkA.snapshots[timestep]]
        .to_frame()
    )
    loading_switches.columns = ["switch"]
    loading_noswitches = abs(
        networkB.lines_t.p0.mul(networkB.snapshot_weightings, axis=0)
        .loc[networkB.snapshots[timestep]]
        .to_frame()
    )
    loading_noswitches.columns = ["noswitch"]
    diff_network = loading_switches.join(loading_noswitches)
    diff_network["noswitch"] = diff_network["noswitch"].fillna(diff_network["switch"])
    diff_network[networkA.snapshots[timestep]] = (
        diff_network["switch"] - diff_network["noswitch"]
    )

    # get switches
    new_buses = pd.Series(index=networkA.buses.index.values)
    new_buses.loc[
        set(networkA.buses.index.values) - set(networkB.buses.index.values)
    ] = 0.1
    new_buses = new_buses.fillna(0)

    # plot network with difference in loading and shifted colormap
    loading = (
        diff_network.loc[:, networkA.snapshots[timestep]] / (networkA.lines.s_nom)
    ) * 100
    midpoint = 1 - max(loading) / (max(loading) + abs(min(loading)))
    shifted_cmap = shiftedColorMap(plt.cm.jet, midpoint=midpoint, name="shifted")
    ll = networkA.plot(
        line_colors=loading,
        line_cmap=shifted_cmap,
        title="Line loading",
        bus_sizes=new_buses,
        bus_colors="blue",
        line_widths=0.55,
        geomap=False,
    )

    cb = plt.colorbar(ll[1])
    cb.set_label("Difference in line loading in % of s_nom")


def network_expansion_diff(networkA, networkB, filename=None, boundaries=[], osm=False):
    """Plot relative network expansion derivation of AC- and DC-lines.

    Parameters
    ----------
    networkA: PyPSA network container
        Holds topology of grid including results from powerflow analysis
    networkB: PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename: str or None
        Save figure in this direction
    boundaries: array
       Set boundaries of heatmap axis
    osm : bool or dict, e.g. {'x': [1,20], 'y': [47, 56], 'zoom' : 6}
        If not False, osm is set as background
        with the following settings as dict:
                'x': array of two floats, x axis boundaries (lat)
                'y': array of two floats, y axis boundaries (long)
                'zoom' : resolution of osm

    """
    if osm != False:
        if set_epsg_network.counter == 0:
            set_epsg_network(networkA)
            set_epsg_network(networkB)
        plot_osm(osm["x"], osm["y"], osm["zoom"])

    cmap = plt.cm.jet

    extension_lines = 100 * (
        (networkA.lines.s_nom_opt - networkB.lines.s_nom_opt) / networkA.lines.s_nom_opt
    )

    extension_links = 100 * (
        (networkA.links.p_nom_opt - networkB.links.p_nom_opt) / networkA.links.p_nom_opt
    )

    ll = networkA.plot(
        line_colors=extension_lines,
        link_colors=extension_links,
        line_cmap=cmap,
        bus_sizes=0,
        title="Derivation of AC- and DC-line extension",
        line_widths=2,
        geomap=False,
    )

    if not boundaries:
        v = np.linspace(
            min(extension_lines.min(), extension_links.min()),
            max(extension_lines.max(), extension_links.max()),
            101,
        )
        boundaries = [
            min(extension_lines.min(), extension_links.min()).round(0),
            max(extension_lines.max(), extension_links.max()).round(0),
        ]

    else:
        v = np.linspace(boundaries[0], boundaries[1], 101)

    if not extension_links.empty:
        cb_Link = plt.colorbar(ll[2], boundaries=v, ticks=v[0:101:10])
        cb_Link.set_clim(vmin=boundaries[0], vmax=boundaries[1])

        cb_Link.remove()

    cb = plt.colorbar(ll[1], boundaries=v, ticks=v[0:101:10], fraction=0.046, pad=0.04)

    cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])
    cb.set_label("line extension derivation  in %")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def plot_residual_load(network):
    """Plots residual load summed of all exisiting buses.

    Parameters
    ----------
    network : PyPSA network containter
    """

    renewables = network.generators[
        network.generators.carrier.isin(
            ["wind_onshore", "wind_offshore", "solar", "run_of_river", "wind"]
        )
    ]
    renewables_t = network.generators.p_nom[
        renewables.index
    ] * network.generators_t.p_max_pu[renewables.index].mul(
        network.snapshot_weightings, axis=0
    )
    load = network.loads_t.p_set.mul(network.snapshot_weightings, axis=0).sum(axis=1)
    all_renew = renewables_t.sum(axis=1)
    residual_load = load - all_renew
    plot = residual_load.plot(
        title="Residual load", drawstyle="steps", lw=2, color="red", legend=False
    )
    plot.set_ylabel("MW")
    # sorted curve
    sorted_residual_load = residual_load.sort_values(ascending=False).reset_index()
    plot1 = sorted_residual_load.plot(
        title="Sorted residual load", drawstyle="steps", lw=2, color="red", legend=False
    )
    plot1.set_ylabel("MW")


def plot_stacked_gen(network, bus=None, resolution="GW", filename=None):
    """
    Plot stacked sum of generation grouped by carrier type


    Parameters
    ----------
    network : PyPSA network container
    bus: string
        Plot all generators at one specific bus. If none,
        sum is calulated for all buses
    resolution: string
        Unit for y-axis. Can be either GW/MW/KW

    Returns
    -------
    Plot
    """
    if resolution == "GW":
        reso_int = 1e3
    elif resolution == "MW":
        reso_int = 1
    elif resolution == "KW":
        reso_int = 0.001

    # sum for all buses
    if bus is None:
        p_by_carrier = (
            pd.concat(
                [
                    network.generators_t.p[
                        network.generators[network.generators.control != "Slack"].index
                    ],
                    network.generators_t.p.mul(network.snapshot_weightings, axis=0)[
                        network.generators[network.generators.control == "Slack"].index
                    ]
                    .iloc[:, 0]
                    .apply(lambda x: x if x > 0 else 0),
                ],
                axis=1,
            )
            .groupby(network.generators.carrier, axis=1)
            .sum()
        )

        load = network.loads_t.p.sum(axis=1)
        if hasattr(network, "foreign_trade"):
            trade_sum = network.foreign_trade.sum(axis=1)
            p_by_carrier["imports"] = trade_sum[trade_sum > 0]
            p_by_carrier["imports"] = p_by_carrier["imports"].fillna(0)
    # sum for a single bus
    elif bus is not None:
        filtered_gens = network.generators[network.generators["bus"] == bus]
        p_by_carrier = (
            network.generators_t.p.mul(network.snapshot_weightings, axis=0)
            .groupby(filtered_gens.carrier, axis=1)
            .abs()
            .sum()
        )
        filtered_load = network.loads[network.loads["bus"] == bus]
        load = network.loads_t.p.mul(network.snapshot_weightings, axis=0)[
            filtered_load.index
        ]

    colors = coloring()
    #    TODO: column reordering based on available columns

    fig, ax = plt.subplots(1, 1)

    fig.set_size_inches(12, 6)
    colors = [colors[col] for col in p_by_carrier.columns]
    if len(colors) == 1:
        colors = colors[0]
    (p_by_carrier / reso_int).plot(kind="area", ax=ax, linewidth=0, color=colors)
    (load / reso_int).plot(ax=ax, legend="load", lw=2, color="darkgrey", style="--")
    ax.legend(ncol=4, loc="upper left")

    ax.set_ylabel(resolution)
    ax.set_xlabel("")

    matplotlib.rcParams.update({"font.size": 22})

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def plot_gen_diff(
    networkA,
    networkB,
    leave_out_carriers=[
        "geothermal",
        "oil",
        "other_non_renewable",
        "reservoir",
        "waste",
    ],
):
    """
    Plot difference in generation between two networks grouped by carrier type


    Parameters
    ----------
    networkA : PyPSA network container with switches
    networkB : PyPSA network container without switches
    leave_out_carriers : list of carriers to leave out (default to all small
    carriers)

    Returns
    -------
    Plot
    """

    def gen_by_c(network):
        gen = (
            pd.concat(
                [
                    network.generators_t.p.mul(etwork.snapshot_weightings, axis=0)[
                        network.generators[network.generators.control != "Slack"].index
                    ],
                    network.generators_t.p.mul(network.snapshot_weightings, axis=0)[
                        network.generators[network.generators.control == "Slack"].index
                    ]
                    .iloc[:, 0]
                    .apply(lambda x: x if x > 0 else 0),
                ],
                axis=1,
            )
            .groupby(network.generators.carrier, axis=1)
            .sum()
        )
        return gen

    gen = gen_by_c(networkB)
    gen_switches = gen_by_c(networkA)
    diff = gen_switches - gen

    colors = coloring()
    diff.drop(leave_out_carriers, axis=1, inplace=True)
    colors = [colors[col] for col in diff.columns]

    plot = diff.plot(kind="line", color=colors, use_index=False)
    plot.legend(loc="upper left", ncol=5, prop={"size": 8})
    x = []
    for i in range(0, len(diff)):
        x.append(i)
    plt.xticks(x, x)
    plot.set_xlabel("Timesteps")
    plot.set_ylabel("Difference in Generation in MW")
    plot.set_title("Difference in Generation")
    plt.tight_layout()


def plot_voltage(network, boundaries=[], osm=False):
    """
    Plot voltage at buses as hexbin


    Parameters
    ----------
    network : PyPSA network container
    boundaries: list of 2 values, setting the lower and upper bound of colorbar
    osm : bool or dict, e.g. {'x': [1,20], 'y': [47, 56], 'zoom' : 6}
        If not False, osm is set as background
        with the following settings as dict:
                'x': array of two floats, x axis boundaries (lat)
                'y': array of two floats, y axis boundaries (long)
                'zoom' : resolution of osm

    Returns
    -------
    Plot
    """
    if osm != False:
        if set_epsg_network.counter == 0:
            set_epsg_network(network)
        plot_osm(osm["x"], osm["y"], osm["zoom"])

    x = np.array(network.buses["x"])
    y = np.array(network.buses["y"])

    alpha = np.array(network.buses_t.v_mag_pu.loc[network.snapshots[0]])

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6, 4)
    cmap = plt.cm.jet
    if not boundaries:
        plt.hexbin(x, y, C=alpha, cmap=cmap, gridsize=100)
        cb = plt.colorbar()
    elif boundaries:
        v = np.linspace(boundaries[0], boundaries[1], 101)
        norm = matplotlib.colors.BoundaryNorm(v, cmap.N)
        plt.hexbin(x, y, C=alpha, cmap=cmap, gridsize=100, norm=norm)
        cb = plt.colorbar(boundaries=v, ticks=v[0:101:10], norm=norm)
        cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])
    cb.set_label("Voltage Magnitude per unit of v_nom")

    network.plot(ax=ax, line_widths=pd.Series(0.5, network.lines.index), bus_sizes=0)
    plt.show()


def curtailment(network, carrier="solar", filename=None):
    """
    Plot curtailment of selected carrier


    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    carrier: str
        Plot curtailemt of this carrier
    filename: str or None
        Save figure in this direction


    Returns
    -------
    Plot
    """
    p_by_carrier = network.generators_t.p.groupby(
        network.generators.carrier, axis=1
    ).sum()
    capacity = network.generators.groupby("carrier").sum().at[carrier, "p_nom"]
    p_available = network.generators_t.p_max_pu.multiply(network.generators["p_nom"])
    p_available_by_carrier = p_available.groupby(
        network.generators.carrier, axis=1
    ).sum()
    p_curtailed_by_carrier = p_available_by_carrier - p_by_carrier
    print(p_curtailed_by_carrier.sum())
    p_df = pd.DataFrame(
        {
            carrier + " available": p_available_by_carrier[carrier],
            carrier + " dispatched": p_by_carrier[carrier],
            carrier + " curtailed": p_curtailed_by_carrier[carrier],
        }
    )

    p_df[carrier + " capacity"] = capacity
    p_df[carrier + " curtailed"][p_df[carrier + " curtailed"] < 0.0] = 0.0

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 6)
    p_df[[carrier + " dispatched", carrier + " curtailed"]].plot(
        kind="area", ax=ax, linewidth=3
    )
    p_df[[carrier + " available", carrier + " capacity"]].plot(ax=ax, linewidth=3)

    ax.set_xlabel("")
    ax.set_ylabel("Power [MW]")
    ax.set_ylim([0, capacity * 1.1])
    ax.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def calc_dispatch_per_carrier(network, timesteps):
    """Function that calculates dispatch per carrier in given timesteps

    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    timesteps : array
        Timesteps considered in dispatch calculation

    Returns
    -------
    dist : pandas.Series
        dispatch per carrier

    """

    index = [
        (network.generators.bus[idx], network.generators.carrier[idx])
        for idx in network.generators.index
    ]

    dist = pd.Series(
        index=pd.MultiIndex.from_tuples(index, names=["bus", "carrier"]), dtype=float
    )

    for i in dist.index:
        gens = network.generators[
            (network.generators.bus == i[0]) & (network.generators.carrier == i[1])
        ].index
        dist[i] = (
            (network.generators_t.p[gens].transpose()[network.snapshots[timesteps]])
            .sum()
            .sum()
        )

    return dist


def calc_storage_expansion_per_bus(network, carriers=["battery", "H2_overground", "H2_underground", "rural_heat_store", "central_heat_store"]):
    """Function that calculates storage expansion per bus and technology

    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis

    Returns
    -------
    dist : pandas.Series
        storage expansion per bus and technology

    """
    index = [(idx, "battery") for idx in network.buses.index]
    for c in carriers:
        if c != "battery":
            index.extend([(idx, c) for idx in network.buses.index])
    # index.extend([(idx, 'hydrogen_storage') for idx in network.buses.index])

    dist = pd.Series(
        index=pd.MultiIndex.from_tuples(index, names=["bus", "carrier"]), dtype=float
    )

    if "battery" in carriers:
        batteries = network.storage_units[network.storage_units.carrier == "battery"]
        battery_distribution = (
            network.storage_units.p_nom_opt[batteries.index]
            .groupby(network.storage_units.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        ).mul(6)
        dist.iloc[
            dist.index.get_level_values("carrier") == "battery"
        ] = battery_distribution.sort_index().values
    if "H2_overground" in carriers:        
        h2_overground = network.stores[network.stores.carrier == "H2_overground"]
        h2_over_distribution = (
            network.stores.e_nom_opt[h2_overground.index]
            .groupby(network.stores.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        )
        dist.iloc[
            dist.index.get_level_values("carrier") == "H2_overground"
        ] = h2_over_distribution.sort_index().values
        
    if "H2_overground" in carriers:        
        h2_underground = network.stores[network.stores.carrier == "H2_underground"]
        h2_under_distribution = (
            network.stores.e_nom_opt[h2_underground.index]
            .groupby(network.stores.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        )
        dist.iloc[
            dist.index.get_level_values("carrier") == "H2_underground"
        ] = h2_under_distribution.sort_index().values
        
    if "rural_heat_store" in carriers:        
        rural_heat = network.stores[network.stores.carrier == "rural_heat_store"]
        rural_heat_distribution = (
            network.stores.e_nom_opt[rural_heat.index]
            .groupby(network.stores.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        )

        dist.iloc[
            dist.index.get_level_values("carrier") == "rural_heat_store"
        ] = rural_heat_distribution.sort_index().values
    if "central_heat_store" in carriers:
        central_heat = network.stores[network.stores.carrier == "central_heat_store"]
        central_heat_distribution = (
            network.stores.e_nom_opt[central_heat.index]
            .groupby(network.stores.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        )
        dist.iloc[
            dist.index.get_level_values("carrier") == "central_heat_store"
        ] = central_heat_distribution.sort_index().values
    # hydrogen_distribution =\
    #     network.storage_units.p_nom_opt[hydrogen.index].groupby(
    #         network.storage_units.bus).sum().reindex(
    #             network.buses.index, fill_value=0.)




    # dist.iloc[dist.index.get_level_values('carrier') == 'hydrogen_storage'] = \
    #         hydrogen_distribution.sort_index().values
    # network.carriers.color['hydrogen_storage'] = 'orange'
    # network.carriers.color['battery_storage'] = 'blue'

    return dist


def gen_dist_diff(
    networkA,
    networkB,
    techs=None,
    snapshot=0,
    n_cols=3,
    gen_size=0.2,
    filename=None,
    buscmap=plt.cm.jet,
):
    """
    Difference in generation distribution
    Green/Yellow/Red colors mean that the generation at a location
    is bigger with switches than without
    Blue colors mean that the generation at a location is smaller with switches
    than without

    Parameters
    ----------
    networkA : PyPSA network container
        Holds topology of grid with switches
        including results from powerflow analysis
    networkB : PyPSA network container
        Holds topology of grid without switches
        including results from powerflow analysis
    techs : dict
        type of technologies which shall be plotted
    snapshot : int
        snapshot
    n_cols : int
        number of columns of the plot
    gen_size : num
        size of generation bubbles at the buses
    filename : str
        Specify filename
        If not given, figure will be show directly
    """
    if techs is None:
        techs = networkA.generators.carrier.unique()
    else:
        techs = techs

    n_graphs = len(techs)
    n_cols = n_cols

    if n_graphs % n_cols == 0:
        n_rows = n_graphs // n_cols
    else:
        n_rows = n_graphs // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)

    size = 4

    fig.set_size_inches(size * n_cols, size * n_rows)

    for i, tech in enumerate(techs):
        i_row = i // n_cols
        i_col = i % n_cols

        ax = axes[i_row, i_col]

        gensA = networkA.generators[networkA.generators.carrier == tech]
        gensB = networkB.generators[networkB.generators.carrier == tech]

        gen_distribution = networkA.generators_t.p.mul(
            networkA.snapshot_weightings, axis=0
        )[gensA.index].loc[networkA.snapshots[snapshot]].groupby(
            networkA.generators.bus
        ).sum().reindex(
            networkA.buses.index, fill_value=0.0
        ) - networkB.generators_t.p.mul(
            networkB.snapshot_weightings, axis=0
        )[
            gensB.index
        ].loc[
            networkB.snapshots[snapshot]
        ].groupby(
            networkB.generators.bus
        ).sum().reindex(
            networkB.buses.index, fill_value=0.0
        )

        networkA.plot(
            ax=ax,
            bus_sizes=gen_size * abs(gen_distribution),
            bus_colors=gen_distribution,
            line_widths=0.1,
            bus_cmap=buscmap,
        )

        ax.set_title(tech)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def nodal_gen_dispatch(
    network,
    networkB=None,
    techs=["wind_onshore", "solar"],
    item="energy",
    direction=None,
    scaling=1,
    filename=None,
    osm=False,
):
    """
    Plot nodal dispatch or capacity. If networkB is given, difference in
    dispatch is plotted.

    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    networkB : PyPSA network container
        If given and item is 'energy', difference in dispatch between network
        and networkB is plotted. If item is 'capacity', networkB is ignored.
        default None
    techs : None or list,
        Techs to plot. If None, all techs are plotted.
        default ['wind_onshore', 'solar']
    item : str
        Specifies the plotted item. Options are 'energy' and 'capacity'.
        default 'energy'
    direction : str
        Only considered if networkB is given and item is 'energy'. Specifies
        the direction of change in dispatch between network and networkB.
        If 'positive', generation per tech which is higher in network than in
        networkB is plotted.
        If 'negative', generation per tech whcih is lower in network than
        in networkB is plotted.
        If 'absolute', total change per node is plotted.
        Green nodes have higher dispatch in network than in networkB.
        Red nodes have lower dispatch in network than in networkB.
        default None
    scaling : int
        Scaling to change plot sizes.
        default 1
    filename : path to folder
    osm : bool or dict, e.g. {'x': [1,20], 'y': [47, 56], 'zoom' : 6}
        If not False, osm is set as background
        with the following settings as dict:
                'x': array of two floats, x axis boundaries (lat)
                'y': array of two floats, y axis boundaries (long)
                'zoom' : resolution of osm
    """

    if osm != False:
        if set_epsg_network.counter == 0:
            set_epsg_network(network)
        fig, ax = plot_osm(osm["x"], osm["y"], osm["zoom"])
    else:
        fig, ax = plt.subplots(1, 1)

    if techs:
        gens = network.generators[network.generators.carrier.isin(techs)]
    elif techs is None:
        gens = network.generators
        techs = gens.carrier.unique()
    if item == "capacity":
        dispatch = gens.p_nom.groupby(
            [network.generators.bus, network.generators.carrier]
        ).sum()
    elif item == "energy":
        if networkB:
            dispatch_network = (
                network.generators_t.p[gens.index]
                .mul(network.snapshot_weightings.generators, axis=0)
                .groupby([network.generators.bus, network.generators.carrier], axis=1)
                .sum()
            )
            dispatch_networkB = (
                networkB.generators_t.p[gens.index]
                .mul(networkB.snapshot_weightings.generators, axis=0)
                .groupby([networkB.generators.bus, networkB.generators.carrier], axis=1)
                .sum()
            )
            dispatch = dispatch_network - dispatch_networkB

            if direction == "positive":
                dispatch = dispatch[dispatch > 0].fillna(0)
            elif direction == "negative":
                dispatch = dispatch[dispatch < 0].fillna(0)
            elif direction == "absolute":
                pass
            else:
                return "No valid direction given."
            dispatch = dispatch.sum()

        elif networkB is None:
            dispatch = (
                network.generators_t.p[gens.index]
                .mul(network.snapshot_weightings.generators, axis=0)
                .sum()
                .groupby([network.generators.bus, network.generators.carrier])
                .sum()
            )
    scaling = 1 / (max(abs(dispatch.groupby(level=0).sum()))) * scaling
    if direction != "absolute":
        colors = coloring()
        subcolors = {a: colors[a] for a in techs}
        dispatch = dispatch.abs() + 1e-9
    else:
        dispatch = dispatch.sum(level=0)
        colors = {s[0]: "green" if s[1] > 0 else "red" for s in dispatch.iteritems()}
        dispatch = dispatch.abs()
        subcolors = {"negative": "red", "positive": "green"}
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    network.plot(
        bus_sizes=dispatch * scaling,
        bus_colors=colors,
        line_widths=0.2,
        margin=0.01,
        ax=ax,
    )

    fig.subplots_adjust(right=0.8)
    plt.subplots_adjust(wspace=0, hspace=0.001)

    patchList = []
    for key in subcolors:
        data_key = mpatches.Patch(color=subcolors[key], label=key)
        patchList.append(data_key)

    ax.legend(handles=patchList, loc="upper left")
    ax.autoscale()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

    return


def nodal_production_balance(network, timesteps, scaling=0.00001):
    """Function that calculates residual load per node in given timesteps

    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    timesteps : array
        timesteps considered in calculation
    scaling : float, optional
        Scaling factor for bus size. The default is 0.00001.

    Returns
    -------
    bus_sizes : pandas.Series
         scaled residual load per node
    bus_colors : pandas.Series
        'green' for producer and 'red' for consumer

    """

    gen = (
        mul_weighting(network, network.generators_t.p)
        .groupby(network.generators.bus, axis=1)
        .sum()
        .loc[network.snapshots[timesteps]]
    )
    load = (
        mul_weighting(network, network.loads_t.p)
        .groupby(network.loads.bus, axis=1)
        .sum()
        .loc[network.snapshots[timesteps]]
    )

    residual_load = (gen - load).sum()

    bus_colors = pd.Series(
        {s[0]: "green" if s[1] > 0 else "red" for s in residual_load.iteritems()}
    )

    bus_sizes = residual_load.abs() * scaling

    return bus_sizes, bus_colors


def storage_p_soc(network, mean="1H", filename=None):
    """
    Plots the dispatch and state of charge (SOC) of extendable storages.

    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    mean : str
        Defines over how many snapshots the p and soc values will averaged.
    filename : path to folder

    """

    sbatt = network.storage_units.index[
        (network.storage_units.p_nom_opt > 1)
        & (network.storage_units.capital_cost > 10)
        & (network.storage_units.max_hours == 6)
    ]
    shydr = network.storage_units.index[
        (network.storage_units.p_nom_opt > 1)
        & (network.storage_units.capital_cost > 10)
        & (network.storage_units.max_hours == 168)
    ]

    cap_batt = (
        network.storage_units.max_hours[sbatt] * network.storage_units.p_nom_opt[sbatt]
    ).sum()
    cap_hydr = (
        network.storage_units.max_hours[shydr] * network.storage_units.p_nom_opt[shydr]
    ).sum()

    fig, ax = plt.subplots(1, 1)

    if (
        network.storage_units.p_nom_opt[sbatt].sum() < 1
        and network.storage_units.p_nom_opt[shydr].sum() < 1
    ):
        print("No storage unit to plot")

    elif (
        network.storage_units.p_nom_opt[sbatt].sum() > 1
        and network.storage_units.p_nom_opt[shydr].sum() < 1
    ):

        (
            network.storage_units_t.p[sbatt].resample(mean).mean().sum(axis=1)
            / network.storage_units.p_nom_opt[sbatt].sum()
        ).plot(ax=ax, label="Battery dispatch", color="orangered")
        # instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx()
        (
            (
                network.storage_units_t.state_of_charge[sbatt]
                .resample(mean)
                .mean()
                .sum(axis=1)
                / cap_batt
            )
            * 100
        ).plot(ax=ax2, label="Battery state of charge", color="blue")
    elif (
        network.storage_units.p_nom_opt[sbatt].sum() < 1
        and network.storage_units.p_nom_opt[shydr].sum() > 1
    ):
        (
            network.storage_units_t.p[shydr].resample(mean).mean().sum(axis=1)
            / network.storage_units.p_nom_opt[shydr].sum()
        ).plot(ax=ax, label="Hydrogen dispatch", color="teal")
        # instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx()
        (
            (
                network.storage_units_t.state_of_charge[shydr]
                .resample(mean)
                .mean()
                .sum(axis=1)
                / cap_hydr
            )
            * 100
        ).plot(ax=ax2, label="Hydrogen state of charge", color="green")
    else:
        (
            network.storage_units_t.p[sbatt].resample(mean).mean().sum(axis=1)
            / network.storage_units.p_nom_opt[sbatt].sum()
        ).plot(ax=ax, label="Battery dispatch", color="orangered")

        (
            network.storage_units_t.p[shydr].resample(mean).mean().sum(axis=1)
            / network.storage_units.p_nom_opt[shydr].sum()
        ).plot(ax=ax, label="Hydrogen dispatch", color="teal")
        # instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx()
        (
            (
                network.storage_units_t.state_of_charge[shydr]
                .resample(mean)
                .mean()
                .sum(axis=1)
                / cap_hydr
            )
            * 100
        ).plot(ax=ax2, label="Hydrogen state of charge", color="green")

        (
            (
                network.storage_units_t.state_of_charge[sbatt]
                .resample(mean)
                .mean()
                .sum(axis=1)
                / cap_batt
            )
            * 100
        ).plot(ax=ax2, label="Battery state of charge", color="blue")

    ax.set_xlabel("")
    ax.set_ylabel("Storage dispatch in p.u. \n <- charge - discharge ->")
    ax2.set_ylabel("Storage state of charge in % ")
    ax2.set_ylim([0, 100])
    ax.set_ylim([-1, 1])
    ax.legend(loc=2)
    ax2.legend(loc=1)
    ax.set_title("Storage dispatch and state of charge")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

    return


def storage_soc_sorted(network, filename=None):
    """
    Plots the soc (state-pf-charge) of extendable storages

    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis

    filename : path to folder

    """
    sbatt = network.storage_units.index[
        (network.storage_units.p_nom_opt > 1)
        & (network.storage_units.capital_cost > 10)
        & (network.storage_units.max_hours == 6)
    ]
    shydr = network.storage_units.index[
        (network.storage_units.p_nom_opt > 1)
        & (network.storage_units.capital_cost > 10)
        & (network.storage_units.max_hours == 168)
    ]

    cap_batt = (
        network.storage_units.max_hours[sbatt] * network.storage_units.p_nom_opt[sbatt]
    ).sum()
    cap_hydr = (
        network.storage_units.max_hours[shydr] * network.storage_units.p_nom_opt[shydr]
    ).sum()

    fig, ax = plt.subplots(1, 1)

    if (
        network.storage_units.p_nom_opt[sbatt].sum() < 1
        and network.storage_units.p_nom_opt[shydr].sum() < 1
    ):
        print("No storage unit to plot")
    elif (
        network.storage_units.p_nom_opt[sbatt].sum() > 1
        and network.storage_units.p_nom_opt[shydr].sum() < 1
    ):
        (
            network.storage_units_t.p[sbatt]
            .sum(axis=1)
            .sort_values(ascending=False)
            .reset_index()
            / network.storage_units.p_nom_opt[sbatt].sum()
        )[0].plot(ax=ax, label="Battery storage", color="orangered")
    elif (
        network.storage_units.p_nom_opt[sbatt].sum() < 1
        and network.storage_units.p_nom_opt[shydr].sum() > 1
    ):
        (
            network.storage_units_t.p[shydr]
            .sum(axis=1)
            .sort_values(ascending=False)
            .reset_index()
            / network.storage_units.p_nom_opt[shydr].sum()
        )[0].plot(ax=ax, label="Hydrogen storage", color="teal")
    else:
        (
            network.storage_units_t.p[sbatt]
            .sum(axis=1)
            .sort_values(ascending=False)
            .reset_index()
            / network.storage_units.p_nom_opt[sbatt].sum()
        )[0].plot(ax=ax, label="Battery storage", color="orangered")
        (
            network.storage_units_t.p[shydr]
            .sum(axis=1)
            .sort_values(ascending=False)
            .reset_index()
            / network.storage_units.p_nom_opt[shydr].sum()
        )[0].plot(ax=ax, label="Hydrogen storage", color="teal")

    ax.set_xlabel("")
    ax.set_ylabel("Storage dispatch in p.u. \n <- charge - discharge ->")
    ax.set_ylim([-1.05, 1.05])
    ax.legend()
    ax.set_title("Sorted duration curve of storage dispatch")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, figsize=(3, 4), bbox_inches="tight")
        plt.close()

    return


def mul_weighting(network, timeseries):
    """Returns timeseries considering snapshot_weighting

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    timeseries : pd.Series
        timeseries not considering snapshot_weighting

    Returns
    -------
    pd.Series
         timeseries considering snapshot_weightings

    """
    return timeseries.mul(network.snapshot_weightings.generators, axis=0)


def calc_ac_loading(network, timesteps):
    """Calculates loading of AC-lines

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    timesteps : range
        Defines which timesteps are considered. If more than one, an
        average line loading is calculated.

    Returns
    -------
    pandas.Series
        ACC line loading in MVA

    """

    loading_lines = (
        mul_weighting(network, network.lines_t.p0)
        .loc[network.snapshots[timesteps]]
        .sum()
    )

    if not network.lines_t.q0.empty:

        loading_lines = (
            loading_lines**2
            + mul_weighting(network, network.lines_t.q0)
            .loc[network.snapshots[timesteps]]
            .abs()
            .sum()
            ** 2
        ).apply(sqrt)

    return loading_lines / network.lines.s_nom_opt


def calc_dc_loading(network, timesteps):
    """Calculates loading of DC-lines


    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    timesteps : range
        Defines which timesteps are considered. If more than one, an
        average line loading is calculated.

    Returns
    -------
    pandas.Series
        DC line loading in MW

    """
    # Aviod covering of bidirectional links
    network.links["linked_to"] = 0
    for i, row in network.links.iterrows():
        if not (
            network.links.index[
                (network.links.bus0 == row["bus1"])
                & (network.links.bus1 == row["bus0"])
                & (network.links.length == row["length"])
            ]
        ).empty:

            l = network.links.index[
                (network.links.bus0 == row["bus1"])
                & (network.links.bus1 == row["bus0"])
                & (network.links.length == row["length"])
            ]

            network.links.at[i, "linked_to"] = l.values[0]

    network.links.linked_to = network.links.linked_to.astype(str)
    # Set p_nom_max and line_loading for one directional links
    link_load = network.links_t.p0[network.links.index[network.links.linked_to == "0"]]

    p_nom_opt_max = network.links.p_nom_opt[network.links.linked_to == "0"]

    # Set p_nom_max and line_loading for bidirectional links
    for i, row in network.links[network.links.linked_to != "0"].iterrows():
        load = pd.DataFrame(index=network.links_t.p0.index, columns=["to", "from"])
        load["to"] = network.links_t.p0[row["linked_to"]]
        load["from"] = network.links_t.p0[i]
        link_load[i] = load.abs().max(axis=1)
        p_nom_opt_max[i] = max(
            row.p_nom_opt,
            network.links.p_nom_opt[network.links.index == row["linked_to"]].values[0],
        )

    return (
        mul_weighting(network, link_load)
        .loc[network.snapshots[timesteps]]
        .abs()
        .sum()[network.links.index]
        / p_nom_opt_max
    ).dropna()


def plotting_colors(network):
    """Add color values to network.carriers

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    None.

    """
    # if network.carriers.columns[1] != 'co2_emissions':
    #     network.carriers = network.carriers.set_index(
    #         network.carriers.columns[1])
    colors = coloring()
    for i in colors.keys():
        network.carriers.loc[i, "color"] = colors[i]
    #     if i in colors.keys():
    #         network.carriers.color[i] = colors[i]
    # network.carriers.color['hydrogen_storage'] = 'sandybrown'
    # network.carriers.color['battery_storage'] = 'blue'
    # network.carriers.color[network.carriers.color == ''] = 'grey'


def calc_network_expansion(network, method="abs", ext_min=0.1):
    """Calculates absolute or relative expansion per AC- and DC-line

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    method : str, optional
        Choose 'rel' or 'abs'. The default is 'abs'.
    ext_min : float, optional
        Remove lines extended less than this value. The default is 0.1.

    Returns
    -------
    all_network : :class:`pypsa.Network
        Whole network including not extended lines
    extension_lines : pandas.Series
        AC-line expansion
    extension_links : pandas.Series
        DC-line expansion

    """
    all_network = network.copy()

    network.lines = network.lines[
        network.lines.s_nom_extendable
        & (
            (network.lines.s_nom_opt - network.lines.s_nom_min) / network.lines.s_nom
            >= ext_min
        )
    ]
    network.links = network.links[
        network.links.p_nom_extendable
        & (
            (network.links.p_nom_opt - network.links.p_nom_min) / network.links.p_nom
            >= ext_min
        )
    ]

    for i, row in network.links.iterrows():
        linked = network.links[
            (row["bus1"] == network.links.bus0) & (row["bus0"] == network.links.bus1)
        ]
        if not linked.empty:
            if row["p_nom_opt"] < linked.p_nom_opt.values[0]:
                network.links.p_nom_opt[i] = linked.p_nom_opt.values[0]

    if method == "rel":

        extension_lines = (
            100
            * (network.lines.s_nom_opt - network.lines.s_nom_min)
            / network.lines.s_nom
        )

        extension_links = (
            100
            * (network.links.p_nom_opt - network.links.p_nom_min)
            / (network.links.p_nom)
        )
    if method == "abs":
        extension_lines = network.lines.s_nom_opt - network.lines.s_nom_min

        extension_links = network.links.p_nom_opt - network.links.p_nom_min

    return all_network, extension_lines, extension_links


def plot_background_grid(network, ax):
    """Plots grid topology in background of other network.plot

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    ax : matplotlib.axes._subplots.AxesSubplot
        axes of plot

    Returns
    -------
    None.

    """

    link_widths = pd.Series(index=network.links.index, data=0)

    link_widths.loc[network.links.carrier == "DC"] = 0.3

    network.plot(
        ax=ax,
        line_colors="grey",
        link_colors="grey",
        bus_sizes=0,
        line_widths=0.5,
        link_widths=link_widths,
        geomap=True,
        projection=ccrs.PlateCarree(),
        color_geomap=True,
        boundaries=[1.5, 16, 46.8, 58],
    )


def plot_carrier(network, carrier_links=["AC"], carrier_buses=["AC"], cartopy=True):
    """
    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    carrier_links : list
        List of links to be plotted. The default is ["AC"].
    carrier_buses : list
        List of buses to be plotted. The default is ["AC"].
    cartopy : bool, optional
        Provide data about the availability of Cartopy. The default is True.

    Returns
    -------
    None.

    """

    colors = coloring()
    line_colors = "lightblue"

    # Set background
    if cartopy == True:
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        draw_map_cartopy(ax, color_geomap=True)
    else:
        fig, ax = plt.subplots()

    link_width = pd.Series(index=network.links.index, data=2)

    if len(carrier_links) > 0:

        link_width.loc[~network.links.carrier.isin(carrier_links)] = 0

    bus_sizes = pd.Series(index=network.buses.index, data=0.0005)

    if len(carrier_buses) > 0:

        bus_sizes.loc[~network.buses.carrier.isin(carrier_buses)] = 0

    link_colors = network.links.carrier.map(colors)

    bus_colors = network.buses.carrier.map(colors)

    if "AC" in carrier_links:
        line_widths = 1
    else:
        line_widths = 0

    title = ""

    network.plot(
        geomap=True,
        bus_sizes=bus_sizes,
        link_widths=link_width,
        line_widths=line_widths,
        title=title,
        link_colors=link_colors,
        line_colors=line_colors,
        bus_colors=bus_colors,
        ax=ax,
    )

    patchList = []
    for key in carrier_links:
        if key != "AC":
            data_key = mpatches.Patch(color=colors[key], label=f"Link {key}")
        else:
            data_key = mpatches.Patch(color=line_colors, label=f"Line {key}")
        patchList.append(data_key)
    for key in carrier_buses:
        data_key = mpatches.Patch(color=colors[key], label=f"Bus {key}")
        patchList.append(data_key)

    ax.legend(handles=patchList, loc="lower left", ncol=1)
    ax.autoscale()


def plot_grid(
    self,
    line_colors,
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,
    boundaries=None,
    filename=None,
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
):
    """Function that plots etrago.network and results for lines and buses



    Parameters
    ----------
    line_colors : str
        Set static line color or attribute to plot e.g. 'expansion_abs'
        Current options:
            'line_loading': mean line loading in p.u. in selected timesteps
            'v_nom': nominal voltage of lines
            'expansion_abs': absolute network expansion in MVA
            'expansion_rel': network expansion in p.u. of existing capacity
            'q_flow_max': maximal reactive flows
    bus_sizes : float, optional
        Size of buses. The default is 0.001.
    bus_colors : str, optional
        Set static bus color or attribute to plot. The default is 'grey'.
        Current options:
            'nodal_production_balance': net producer/consumer in selected timeteps
            'storage_expansion': storage expansion per bus and technology
            'storage_distribution': installed storage units per bus
            'gen_dist': dispatch per carrier in selected timesteps
    timesteps : array, optional
        Timesteps consideredd in time depended plots. The default is range(2).
    osm : bool or dict, e.g. {'x': [1,20], 'y': [47, 56], 'zoom' : 6}
        If not False, osm is set as background
        with the following settings as dict:
                'x': array of two floats, x axis boundaries (lat)
                'y': array of two floats, y axis boundaries (long)
                'zoom' : resolution of osm. The default is False.
    boundaries: array
       Set fixed boundaries of heatmap axis. The default is None.
    filename: str or None
        Save figure in this direction. The default is None.
    disaggregated : bool, optional
        Choose if disaggregated network is shown. The default is False.
    ext_min: float
        Choose minimum relative line extension shown in plot in p.u..
    ext_width: float or bool
        Choose if line_width respects line extension. Turn off with 'False' or
        set linear factor to decremise extension line_width.
        The default is False.

    Returns
    -------
    None.

    """
    # Choose network or disaggregated_network
    if disaggregated:
        network = self.disaggregated_network.copy()
    else:
        network = self.network.copy()

    # Set colors for plotting
    plotting_colors(network)

    # Set default values
    flow = None
    line_widths = 2
    link_widths = 0

    # Plot osm map in background
    if osm != False:
        if network.srid == 4326:
            set_epsg_network(network)
        fig, ax = plot_osm(osm["x"], osm["y"], osm["zoom"])

    else:
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5, 5)
        )

    # Set line colors
    if line_colors == "line_loading":
        title = (
            "Mean loading from "
            + str(network.snapshots[timesteps[0]])
            + " to "
            + str(network.snapshots[timesteps[-1]])
        )
        rep_snapshots = network.snapshot_weightings[network.snapshots[timesteps]].sum()
        line_colors = calc_ac_loading(network, timesteps).abs() / rep_snapshots
        link_colors = calc_dc_loading(network, timesteps).abs() / rep_snapshots
        label = "line loading in p.u."
        # Only active flow direction is displayed!
        flow = pd.Series(index=network.branches().index, dtype="float64")
        flow.iloc[flow.index.get_level_values("component") == "Line"] = (
            mul_weighting(network, network.lines_t.p0)
            .loc[network.snapshots[timesteps]]
            .sum()
            / network.lines.s_nom
            / rep_snapshots
        ).values
        flow.iloc[flow.index.get_level_values("component") == "Link"] = (
            calc_dc_loading(network, timesteps) / rep_snapshots
        ).values
    elif line_colors == "v_nom":
        title = "Voltage levels"
        label = "v_nom in kV"
        line_colors = network.lines.v_nom
        link_colors = network.links.v_nom
    elif line_colors == "expansion_abs":
        title = "Network expansion"
        label = "network expansion in MVA"
        all_network, line_colors, link_colors = calc_network_expansion(
            network, method="abs", ext_min=ext_min
        )
        plot_background_grid(all_network, ax)
        if ext_width != False:
            line_widths = 0.5 + (line_colors / ext_width)
            link_widths = 0.5 + (link_colors / ext_width)
    elif line_colors == "expansion_rel":
        title = "Network expansion"
        label = "network expansion in %"
        all_network, line_colors, link_colors = calc_network_expansion(
            network, method="rel", ext_min=ext_min
        )
        plot_background_grid(all_network, ax)
        if ext_width != False:
            line_widths = 0.5 + (line_colors / ext_width)
            link_widths = 0.5 + (link_colors / ext_width)
    elif line_colors == "q_flow_max":
        title = "Maximmal reactive power flows"
        label = "flow in Mvar"
        line_colors = abs(network.lines_t.q0.abs().max() / (network.lines.s_nom))
        link_colors = pd.Series(data=0, index=network.links.index)
    elif line_colors == "dlr":
        title = "Dynamic line rating"
        label = "MWh above nominal capacity"
        plot_background_grid(network, ax)
        line_loading = network.lines_t.p0.mul(1 / network.lines.s_nom_opt)
        dlr_usage = (
            line_loading[line_loading.abs() > 1]
            .fillna(0)
            .mul(network.snapshot_weightings.generators, axis=0)
            .abs()
            .sum()
        )
        line_colors = dlr_usage
        if ext_width != False:
            line_widths = 0.5 + (line_colors / ext_width)
        link_colors = pd.Series(data=0, index=network.links.index)

    elif line_colors == "blue":
        title = ""
        label = ""
        line_colors = "blue"
        link_colors = "blue"
        plot_background_grid(network, ax)
        link_widths = 0
        line_widths = 0

    else:
        logger.warning("line_color {} undefined".format(line_colors))

    # Set bus colors

    if bus_colors == "nodal_production_balance":
        bus_scaling = bus_sizes
        bus_sizes, bus_colors = nodal_production_balance(
            network, timesteps, scaling=bus_scaling
        )
        bus_legend = "Nodal production balance"
        bus_unit = "TWh"
    elif bus_colors == "storage_expansion":
        bus_scaling = bus_sizes
        bus_sizes = bus_scaling * calc_storage_expansion_per_bus(network)
        
        
        #scale heat store representation
        bus_sizes[bus_sizes.index.get_level_values('carrier').str.contains('heat')] /=10
        
        bus_legend = "Storage expansion"
        bus_unit = "GW"
    elif bus_colors == "h2_battery_storage_expansion":
        bus_scaling = bus_sizes
        bus_sizes = bus_scaling * calc_storage_expansion_per_bus(network, carriers=["battery", "H2_overground", "H2_underground"])
        bus_legend = "Storage expansion"
        bus_unit = "GW"
    elif bus_colors == "storage_distribution":
        bus_scaling = bus_sizes
        bus_sizes = (
            network.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum()
            * bus_scaling
        )
        bus_legend = "Storage distribution"
        bus_unit = "TW"
    elif bus_colors == "gen_dist":
        bus_scaling = bus_sizes
        bus_sizes = bus_scaling * calc_dispatch_per_carrier(network, timesteps)
        bus_legend = "Dispatch"
        bus_unit = "TW"
    elif bus_colors == "flexibility_usage":
        # import pdb; pdb.set_trace()
        bus_scaling = bus_sizes
        flex_links = network.links[
            network.links.carrier.isin(
                [
                    "dsm",
                    "BEV charger",
                    #'central_heat_store_charger',
                    #'central_heat_store_discharger',
                    #'rural_heat_store_charger',
                    #'rural_heat_store_discharger'
                ]
            )
        ]
        flex_links["p0_sum"] = (
            network.links_t.p0[flex_links.index]
            .mul(network.snapshot_weightings.generators, axis=0)
            .abs()
            .sum()
        )
        flex_links["p1_sum"] = (
            network.links_t.p1[flex_links.index]
            .mul(network.snapshot_weightings.generators, axis=0)
            .sum()
        )
        bus_sizes = bus_scaling * flex_links.groupby(["bus0", "carrier"]).p0_sum.sum()
        bus_unit = "TWh"
        bus_legend = "flexibility_usage"
    elif (
        bus_colors == "PowerToH2"
    ):  # PowerToH2 plots p_nom_opt of links with carrier=power to H2
        bus_scaling = bus_sizes
        bus_sizes = (
            bus_scaling
            * network.links[(network.links.carrier == "power_to_H2")]
            .groupby("bus0")
            .sum()
            .p_nom_opt
        )
        bus_colors = coloring()["power_to_H2"]
        bus_legend = "PowerToH2"
        bus_unit = "TW"
    elif bus_colors == "h2_storage_expansion":
        bus_scaling = bus_sizes
        bus_sizes = bus_scaling * calc_storage_expansion_per_bus(network)
        bus_sizes = bus_sizes.reset_index()
        bus_sizes = bus_sizes[bus_sizes.carrier.str.contains("H2")]
        bus_sizes.set_index(["bus", "carrier"], inplace=True)
        bus_legend = "Storage expansion"
        bus_unit = "GW"
    else:
        logger.warning("bus_color {} undefined".format(bus_colors))

    if type(link_widths) != int:
        link_widths.loc[network.links.carrier != "DC"] = 0

    ll = network.plot(
        line_colors=line_colors,
        link_colors=link_colors,
        line_cmap=plt.cm.jet,
        link_cmap=plt.cm.jet,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        line_widths=line_widths,
        link_widths=link_widths,
        flow=flow,
        title=title,
        geomap=False,
        projection=ccrs.PlateCarree(),
        color_geomap=True,
        boundaries=[1.5, 16, 46.8, 58],
    )

    # legends for bus sizes and colors
    if type(bus_sizes) != float:
        handles = make_legend_circles_for(
            [bus_sizes.min(), bus_sizes.max()], scale=1, facecolor="gray"
        )
        labels = [
            ("{} " + bus_unit).format(s)
            for s in (
                round(bus_sizes.min() / bus_scaling / 1000, 0),
                round(bus_sizes.max() / bus_scaling / 1000, 0),
            )
        ]

        l2 = ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.01, 1.01),
            labelspacing=1.0,
            framealpha=1.0,
            title=bus_legend,
            handler_map=make_handler_map_to_scale_circles_as_in(ax),
        )
        ax.add_artist(l2)

        handles = []
        if bus_legend == "Nodal production balance":
            positive = mpatches.Patch(color="green", label="generation")
            negative = mpatches.Patch(color="red", label="consumption")
            handles = [positive, negative]
        elif legend_entries != "all":
            for i in legend_entries:
                patch = mpatches.Patch(color=network.carriers.color[i], label=i)
                handles.append(patch)
        else:
            for i in network.carriers.color.index:
                patch = mpatches.Patch(color=network.carriers.color[i], label=i)
                handles.append(patch)

        l3 = plt.legend(
            handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(0, 0)
        )
        ax.add_artist(l3)

    if type(line_colors) != str:
        print(111111)
        # Set fixed boundaries if selected in parameters
        if not boundaries:
            boundaries = [
                min(line_colors.min(), link_colors.min()),
                max(line_colors.max(), link_colors.max()),
            ]

        # Create ticks for legend
        v = np.linspace(boundaries[0], boundaries[1], 101)

        # colorbar for line heatmap
        cb = plt.colorbar(
            ll[1], boundaries=v, ticks=v[0:101:10], fraction=0.046, pad=0.04
        )
        # Set legend label
        cb.set_label(label)

    # Show plot or save to file
    if filename is None:
        if type(bus_sizes) != float:
            logger.warning("Legend of bus sizes will change when zooming")
        plt.show()
    else:
        from matplotlib import pylab

        pylab.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


set_epsg_network.counter = 0

### the following functions are copied from pypsa-eur-sec ###
### see here: https://github.com/PyPSA/pypsa-eur-sec/blob/master/scripts/plot_network.py
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse


def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0] * (
            72.0 / fig.dpi
        )

    ellipses = []
    if not dont_resize_actively:

        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2.0 * radius * dist

        fig.canvas.mpl_connect("resize_event", update_width_height)
        ax.callbacks.connect("xlim_changed", update_width_height)
        ax.callbacks.connect("ylim_changed", update_width_height)

    def legend_circle_handler(
        legend, orig_handle, xdescent, ydescent, width, height, fontsize
    ):
        w, h = 2.0 * orig_handle.get_radius() * axes2pt()
        e = Ellipse(
            xy=(0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
            width=w,
            height=w,
        )
        ellipses.append((e, orig_handle.get_radius()))
        return e

    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}


def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale) ** 0.5, **kw) for s in sizes]


###

if __name__ == "__main__":
    pass


def plot_clusters(
    self,
    carrier="AC",
    save_path=False,
    cartopy=True,
    transmission_lines=False,
    gas_pipelines=False,
):
    """
    Parameters
    ----------
    carrier : str, optional
        This variable set the carrier of the buses that will be plotted. The
        default is "AC".
    cartopy : bool, optional
        Set it to True when cartopy is installed and the map is supposed
        to include country's boundaries and bodies of water
    save_path : bool, optional
        Path to save the generated plot. The default is False.
    transmission_lines : bool, optional
        The default is False. Define if the original transmission lines are
        plotted or not.
    gas_pipelines : bool, optional
        The default is False. Define if the original gas pipelines are
        plotted or not.
    Returns
    -------
    None.
    """
    # TODO: Make this function available for other carriers
    # Create geometries
    new_geom = self.network.buses[
        [
            "carrier",
            "x",
            "y",
        ]
    ]
    new_geom = new_geom[new_geom["carrier"] == carrier]
    new_geom["geom"] = new_geom.apply(lambda x: Point(x["x"], x["y"]), axis=1)
    map_buses = self.busmap["orig_network"].buses[
        [
            "carrier",
            "x",
            "y",
        ]
    ]
    map_buses = map_buses[map_buses["carrier"] == carrier]
    map_buses["geom"] = map_buses.apply(lambda x: Point(x["x"], x["y"]), axis=1)
    map_buses["cluster"] = map_buses.index.map(self.busmap["busmap"])
    map_buses["cluster_geom"] = map_buses["cluster"].map(new_geom.geom)
    map_buses["line"] = map_buses.apply(
        lambda x: LineString((x["geom"], x["cluster_geom"])), axis=1
    )

    # Set background
    if cartopy == True:
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        draw_map_cartopy(ax, color_geomap=True)
    else:
        fig, ax = plt.subplots()

    ax.set_title(f'Clustering {self.args["network_clustering"]["method"]}')

    # Draw original transmission lines
    if transmission_lines:
        # AC lines
        lines = self.busmap["orig_network"].lines
        if (
            self.busmap["orig_network"]
            .lines["geom"]
            .apply(lambda x: isinstance(x, str))
            .any()
        ):
            lines["geom"] = gpd.GeoSeries.from_wkt(lines["geom"])
        lines = gpd.GeoDataFrame(self.busmap["orig_network"].lines, geometry="geom")
        lines = lines[
            lines["bus0"].isin(map_buses.index) & lines["bus1"].isin(map_buses.index)
        ]
        lines["geom"] = lines.apply(
            lambda x: x["geom"]
            if not pd.isna(x["geom"])
            else LineString(
                [map_buses["geom"][x["bus0"]], map_buses["geom"][x["bus1"]]]
            ),
            axis=1,
        )
        lines.plot(ax=ax, color="grey", linewidths=0.8, zorder=1)
        # DC lines
        dc_lines = self.busmap["orig_network"].links
        dc_lines = dc_lines[dc_lines["carrier"] == "DC"]
        dc_lines["point0"] = dc_lines["bus0"].map(map_buses["geom"])
        dc_lines["point1"] = dc_lines["bus1"].map(map_buses["geom"])
        dc_lines["line_geom"] = dc_lines.apply(
            lambda x: LineString([x["point0"], x["point1"]]), axis=1
        )
        dc_lines = gpd.GeoDataFrame(dc_lines, geometry="line_geom")
        dc_lines.plot(ax=ax, color="grey", linewidths=0.8, zorder=1)

    if gas_pipelines:
        # CH4 pipelines
        pipelines = self.busmap["orig_network"].links
        if (
            self.busmap["orig_network"]
            .links["geom"]
            .apply(lambda x: isinstance(x, str))
            .any()
        ):
            pipelines["geom"] = gpd.GeoSeries.from_wkt(pipelines["geom"])
        pipelines = pipelines[pipelines["carrier"] == "CH4"]
        pipelines = gpd.GeoDataFrame(pipelines, geometry="geom")
        pipelines.plot(ax=ax, color="grey", linewidths=0.8, zorder=1)

    # Assign a random color to each cluster
    colors = {
        color: np.random.rand(
            3,
        )
        for color in map_buses.cluster.unique()
    }
    map_buses["color"] = map_buses["cluster"].map(colors)

    # Draw original and clustered buses
    map_buses = gpd.GeoDataFrame(map_buses, geometry="line")
    map_buses.plot(ax=ax, color=map_buses["color"], linewidths=0.25, zorder=2)
    map_buses = gpd.GeoDataFrame(map_buses, geometry="geom")
    map_buses.plot(
        ax=ax, color=map_buses["color"], markersize=0.8, marker="o", zorder=3
    )
    map_buses = gpd.GeoDataFrame(map_buses, geometry="cluster_geom")
    map_buses.plot(
        ax=ax,
        color=map_buses["color"],
        markersize=10,
        marker="o",
        edgecolor="black",
        zorder=3,
    )

    if save_path:
        plt.savefig(save_path, dpi=800)

    return

def plot_flexibility(etrago, flexibility, agg='5H'):

    fig, ax = plt.subplots()
    potential = pd.DataFrame(columns=['p_min', 'p_max'])#, "e_min", "e_max"])
    used = pd.DataFrame()

    if flexibility == 'dsm':
        l = etrago.network.links[etrago.network.links.carrier=='dsm']
        s = etrago.network.stores[etrago.network.stores.carrier=='dsm']
        potential["p_min"] = etrago.network.links_t.p_min_pu[l.index].mul(l.p_nom, axis=1).sum(axis=1).resample(agg).mean()
        potential["p_max"] = etrago.network.links_t.p_max_pu[l.index].mul(l.p_nom, axis=1).sum(axis=1).resample(agg).mean()
        used["p"] = etrago.network.links_t.p0[l.index].clip(lower=0).sum(axis=1).resample(agg).mean()
        #potential["e_min"] = etrago.network.stores_t.e_min_pu[s.index].mul(s.e_nom, axis=1).sum(axis=1)
        #potential["e_max"] = etrago.network.stores_t.e_max_pu[s.index].mul(s.e_nom, axis=1).sum(axis=1)
    elif flexibility == 'bev':
        l = etrago.network.links[etrago.network.links.carrier=='BEV charger']    
        s = etrago.network.stores[etrago.network.stores.carrier=='battery storage']
        potential["p_max"] = etrago.network.links_t.p_max_pu[l.index].mul(l.p_nom, axis=1).sum(axis=1).resample(agg).mean()
        potential["p_min"] = 0
        used["p"] = etrago.network.links_t.p0[l.index].sum(axis=1).resample(agg).mean()

    elif flexibility == 'dlr':
        l = etrago.network.lines[etrago.network.lines.index.isin(etrago.network.lines_t.s_max_pu.columns)]
        potential["p_max"] = etrago.network.lines_t.s_max_pu[l.index].mul(l.s_nom_opt, axis=1).sum(axis=1).resample(agg).mean()
        used["p"] = etrago.network.lines_t.p0[l.index].sum(axis=1).resample(agg).mean()

    elif flexibility == 'all':
        line_loading = etrago.network.lines_t.p0.mul(1 / etrago.network.lines.s_nom_opt)
        line_loading[line_loading.abs() > 1].fillna(0)
        l = etrago.network.lines[etrago.network.lines.index.isin(etrago.network.lines_t.s_max_pu.columns)]
        potential["p_max"] = etrago.network.lines_t.s_max_pu[l.index].mul(l.s_nom_opt, axis=1).sum(axis=1).resample(agg).mean()
        used["dlr"] = line_loading[line_loading.abs() > 1].fillna(0).abs().mul(l.s_nom_opt, axis=1).sum(axis=1).resample(agg).mean()

        l = etrago.network.links[etrago.network.links.carrier=='dsm']
        s = etrago.network.stores[etrago.network.stores.carrier=='dsm']
        potential["p_max"] += etrago.network.links_t.p_max_pu[l.index].mul(l.p_nom, axis=1).sum(axis=1).resample(agg).mean()
        used["dsm"] = etrago.network.links_t.p0[l.index].clip(lower=0).sum(axis=1).resample(agg).mean()

        l = etrago.network.links[etrago.network.links.carrier=='BEV charger']    
        s = etrago.network.stores[etrago.network.stores.carrier=='battery storage']
        potential["p_max"] += etrago.network.links_t.p_max_pu[l.index].mul(l.p_nom, axis=1).sum(axis=1).resample(agg).mean()
        used["e_Mob"] = etrago.network.links_t.p0[l.index].sum(axis=1).resample(agg).mean()

        l = etrago.network.links[etrago.network.links.carrier.isin(['rural_heat_store_charger', 'central_heat_store_charger'])]   

        used["heat"] = etrago.network.links_t.p0[l.index].clip(lower=0).sum(axis=1).resample(agg).mean()
        l = etrago.network.links[etrago.network.links.carrier.isin(['rural_heat_store_discharger', 'central_heat_store_discharger'])]   

    else:
        print("Flexibility option not defined")

    if flexibility != 'all':
        potential.plot(ax = ax, kind="area")
        used.plot(ax=ax)
    else:
        used.plot(ax=ax, kind="area")
