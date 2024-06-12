# -*- coding: utf-8 -*-
# Copyright 2016-2023  Flensburg University of Applied Sciences,
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
from math import log10, sqrt
import logging
import os

from etrago.execute import import_gen_from_links
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse
from pyproj import Proj, transform
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

cartopy_present = True
try:
    import cartopy.crs as ccrs
except ImportError:
    cartopy_present = False
from pypsa.plot import draw_map_cartopy

logger = logging.getLogger(__name__)

if "READTHEDOCS" not in os.environ:
    from geoalchemy2.shape import to_shape
    from pyproj import Proj, transform
    from shapely.geometry import LineString, MultiPoint, Point, Polygon
    import geopandas as gpd
    import tilemapbase

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = """ulfmueller, MarlonSchlemminger, mariusves, lukasol, ClaraBuettner,
CarlosEpia, pieterhexen, gnn, fwitte, lukasol, KathiEsterl, BartelsJ"""


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
    plotter = tilemapbase.Plotter(
        extent, tilemapbase.tiles.build_OSM(), zoom=zoom
    )
    plotter.plot(ax, alpha=alpha)
    # ax.plot(x, y, "ro-")
    return fig, ax, extent.xrange, extent.yrange


def coloring():
    """
    Return a dictionary with a color assign to each kind of carrier used in
    etrago.network. This is used for plotting porpuses.

    Returns
    -------
    colors : dict
        Color for each kind of carrier.

    """

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
        "CH4_biogas": "yellow",
        "CH4_NG": "yellow",
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
        "heat": "firebrick",
        "rural_heat_store_charger": "salmon",
        "rural_heat_store": "salmon",
        "central_heat_store_discharger": "firebrick",
        "rural_heat_store_discharger": "salmon",
        "rural_heat": "orange",
        "central_heat": "orangered",
        "H2_grid": "green",
        "H2_saltcavern": "darkgreen",
        "central_heat_store": "firebrick",
        "heat": "firebrick",
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
        "BEV_charger": "indigo",
        "others": "dimgrey",
        "central_coal_CHP": "darkorange",
        "central_lignite_CHP": "darkorange",
        "central_oil_CHP": "darkorange",
        "central_others_CHP": "darkorange",
        "industrial_coal_CHP": "darkorange",
        "industrial_lignite_CHP": "darkorange",
        "industrial_oil_CHP": "darkorange",
        "industrial_others_CHP": "darkorange",
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

        * 'x': array of two floats, x axis boundaries (lat)
        * 'y': array of two floats, y axis boundaries (long)
        * 'zoom' : resolution of osm

    """
    if osm is not False:
        if set_epsg_network.counter == 0:
            set_epsg_network(networkA)
            set_epsg_network(networkB)
        plot_osm(osm["x"], osm["y"], osm["zoom"])

    # new colormap to make sure 0% difference has the same color in every plot
    def shiftedColorMap(
        cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"
    ):
        """
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero

        Parameters
        -----------
        cmap :
            The matplotlib colormap to be altered
        start :
            Offset from lowest point in the colormap's range.
            Defaults to 0.0 (no lower ofset). Should be between
            0.0 and `midpoint`.
        midpoint :
            The new center of the colormap. Defaults to
            0.5 (no shift). Should be between 0.0 and 1.0. In
            general, this should be  1 - vmax/(vmax + abs(vmin))
            For example if your data range from -15.0 to +5.0 and
            you want the center of the colormap at 0.0, `midpoint`
            should be set to  1 - 5/(5 + 15)) or 0.75
        stop :
            Offset from highets point in the colormap's range.
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
    diff_network["noswitch"] = diff_network["noswitch"].fillna(
        diff_network["switch"]
    )
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
        diff_network.loc[:, networkA.snapshots[timestep]]
        / (networkA.lines.s_nom)
    ) * 100
    midpoint = 1 - max(loading) / (max(loading) + abs(min(loading)))
    shifted_cmap = shiftedColorMap(
        plt.cm.jet, midpoint=midpoint, name="shifted"
    )
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


def network_expansion_diff(
    networkA, networkB, filename=None, boundaries=[], osm=False
):
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

        * 'x': array of two floats, x axis boundaries (lat)
        * 'y': array of two floats, y axis boundaries (long)
        * 'zoom' : resolution of osm

    """
    if osm is not False:
        if set_epsg_network.counter == 0:
            set_epsg_network(networkA)
            set_epsg_network(networkB)
        plot_osm(osm["x"], osm["y"], osm["zoom"])

    cmap = plt.cm.jet

    extension_lines = 100 * (
        (networkA.lines.s_nom_opt - networkB.lines.s_nom_opt)
        / networkA.lines.s_nom_opt
    )

    extension_links = 100 * (
        (networkA.links.p_nom_opt - networkB.links.p_nom_opt)
        / networkA.links.p_nom_opt
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

    cb = plt.colorbar(
        ll[1], boundaries=v, ticks=v[0:101:10], fraction=0.046, pad=0.04
    )

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

    Returns
    -------
    Plot
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
    load = network.loads_t.p_set.mul(network.snapshot_weightings, axis=0).sum(
        axis=1
    )
    all_renew = renewables_t.sum(axis=1)
    residual_load = load - all_renew
    plot = residual_load.plot(
        title="Residual load",
        drawstyle="steps",
        lw=2,
        color="red",
        legend=False,
    )
    plot.set_ylabel("MW")
    # sorted curve
    sorted_residual_load = residual_load.sort_values(
        ascending=False
    ).reset_index()
    plot1 = sorted_residual_load.plot(
        title="Sorted residual load",
        drawstyle="steps",
        lw=2,
        color="red",
        legend=False,
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
                        network.generators[
                            network.generators.control != "Slack"
                        ].index
                    ],
                    network.generators_t.p.mul(
                        network.snapshot_weightings, axis=0
                    )[
                        network.generators[
                            network.generators.control == "Slack"
                        ].index
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
    (p_by_carrier / reso_int).plot(
        kind="area", ax=ax, linewidth=0, color=colors
    )
    (load / reso_int).plot(
        ax=ax, legend="load", lw=2, color="darkgrey", style="--"
    )
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
    leave_out_carriers :
        list of carriers to leave out (default to all small carriers)

    Returns
    -------
    Plot
    """

    def gen_by_c(network):
        gen = (
            pd.concat(
                [
                    network.generators_t.p.mul(
                        network.snapshot_weightings, axis=0
                    )[
                        network.generators[
                            network.generators.control != "Slack"
                        ].index
                    ],
                    network.generators_t.p.mul(
                        network.snapshot_weightings, axis=0
                    )[
                        network.generators[
                            network.generators.control == "Slack"
                        ].index
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

        * 'x': array of two floats, x axis boundaries (lat)
        * 'y': array of two floats, y axis boundaries (long)
        * 'zoom' : resolution of osm

    Returns
    -------
    Plot
    """
    if osm is not False:
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

    network.plot(
        ax=ax, line_widths=pd.Series(0.5, network.lines.index), bus_sizes=0
    )
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
    p_available = network.generators_t.p_max_pu.multiply(
        network.generators["p_nom"]
    )
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
    p_df[[carrier + " available", carrier + " capacity"]].plot(
        ax=ax, linewidth=3
    )

    ax.set_xlabel("")
    ax.set_ylabel("Power [MW]")
    ax.set_ylim([0, capacity * 1.1])
    ax.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def calc_dispatch_per_carrier(network, timesteps, dispatch_type="total"):
    """Function that calculates dispatch per carrier in given timesteps

    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    timesteps : array
        Timesteps considered in dispatch calculation
    dispatch_type: str
        This parameter can receive 3 different options: total, ramp_up and
        ramp_down. The default is total, which plots the total energy supplied
        by carrier by the given timesteps.

    Returns
    -------
    dist : pandas.Series
        dispatch per carrier

    """
    # up_time_before and down_time_before are irrelevant for this plot but
    # create problems when grouping, therefore they are set to 0.
    if ("up_time_before" in (network.generators.columns)) | (
        "down_time_before" in (network.generators.columns)
    ):
        network.generators["up_time_before"] = 0
        network.generators["down_time_before"] = 0

    # If ramp_up or ramp_down are passed to the argument dispath_type, all the
    # not relevant generators and their time series are discarted.
    if dispatch_type in ("ramp_up", "ramp_down"):
        if dispatch_type == "ramp_up":
            discard_gen = network.generators[
                ~network.generators.index.str.contains("ramp_up")
            ].index
            discard_gen_l = network.links[
                ~network.links.index.str.contains("ramp_up")
            ].index
        if dispatch_type == "ramp_down":
            discard_gen = network.generators[
                ~network.generators.index.str.contains("ramp_down")
            ].index
            discard_gen_l = network.links[
                ~network.links.index.str.contains("ramp_down")
            ].index
        network.mremove("Generator", discard_gen)
        network.mremove("Link", discard_gen_l)
        network.generators_t.p = network.generators_t.p * -1

    import_gen_from_links(network)

    ac_buses = network.buses[network.buses.carrier == "AC"].index
    network.generators = network.generators[
        network.generators.bus.isin(ac_buses)
    ]
    network.generators_t.p = network.generators_t.p.loc[
        :, network.generators_t.p.columns.isin(network.generators.index)
    ]

    index = [
        (network.generators.bus[idx], network.generators.carrier[idx])
        for idx in network.generators.index
    ]

    dist = pd.Series(
        index=pd.MultiIndex.from_tuples(index, names=["bus", "carrier"]),
        dtype=float,
    )

    for i in dist.index:
        gens = network.generators[
            (network.generators.bus == i[0])
            & (network.generators.carrier == i[1])
        ].index
        dist[i] = (
            (
                network.generators_t.p[gens].transpose()[
                    network.snapshots[timesteps]
                ]
            )
            .sum()
            .sum()
        )

    return dist


def calc_storage_expansion_per_bus(
    network,
    carriers=[
        "battery",
        "H2_overground",
        "H2_underground",
        "rural_heat_store",
        "central_heat_store",
    ],
):
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
        index=pd.MultiIndex.from_tuples(index, names=["bus", "carrier"]),
        dtype=float,
    )

    if "battery" in carriers:
        batteries = network.storage_units[
            network.storage_units.carrier == "battery"
        ]
        battery_distribution = (
            (
                network.storage_units.p_nom_opt[batteries.index]
                - network.storage_units.p_nom_min[batteries.index]
            )
            .groupby(network.storage_units.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        ).mul(6)

        battery_distribution.index = pd.MultiIndex.from_tuples(
            [(idx, "battery") for idx in battery_distribution.index]
        )

        dist.loc[
            dist.index.get_level_values("carrier") == "battery"
        ] = battery_distribution
    if "H2_overground" in carriers:
        h2_overground = network.stores[
            network.stores.carrier == "H2_overground"
        ]
        h2_over_distribution = (
            network.stores.e_nom_opt[h2_overground.index]
            .groupby(network.stores.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        )

        h2_over_distribution.index = pd.MultiIndex.from_tuples(
            [(idx, "H2_overground") for idx in h2_over_distribution.index]
        )

        dist.loc[
            dist.index.get_level_values("carrier") == "H2_overground"
        ] = h2_over_distribution

    if "H2_overground" in carriers:
        h2_underground = network.stores[
            network.stores.carrier == "H2_underground"
        ]
        h2_under_distribution = (
            network.stores.e_nom_opt[h2_underground.index]
            .groupby(network.stores.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        )

        h2_under_distribution.index = pd.MultiIndex.from_tuples(
            [(idx, "H2_underground") for idx in h2_under_distribution.index]
        )

        dist.loc[
            dist.index.get_level_values("carrier") == "H2_underground"
        ] = h2_under_distribution

    if "rural_heat_store" in carriers:
        rural_heat = network.stores[
            network.stores.carrier == "rural_heat_store"
        ]
        rural_heat_distribution = (
            network.stores.e_nom_opt[rural_heat.index]
            .groupby(network.stores.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        )

        rural_heat_distribution.index = pd.MultiIndex.from_tuples(
            [
                (idx, "rural_heat_store")
                for idx in rural_heat_distribution.index
            ]
        )

        dist.loc[
            dist.index.get_level_values("carrier") == "rural_heat_store"
        ] = rural_heat_distribution
    if "central_heat_store" in carriers:
        central_heat = network.stores[
            network.stores.carrier == "central_heat_store"
        ]
        central_heat_distribution = (
            network.stores.e_nom_opt[central_heat.index]
            .groupby(network.stores.bus)
            .sum()
            .reindex(network.buses.index, fill_value=0.0)
        )

        central_heat_distribution.index = pd.MultiIndex.from_tuples(
            [
                (idx, "central_heat_store")
                for idx in central_heat_distribution.index
            ]
        )

        dist.loc[
            dist.index.get_level_values("carrier") == "central_heat_store"
        ] = central_heat_distribution

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

    Returns
    -------
    None.
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

        * 'x': array of two floats, x axis boundaries (lat)
        * 'y': array of two floats, y axis boundaries (long)
        * 'zoom' : resolution of osm

    Returns
    -------
    None.
    """

    if osm is not False:
        if set_epsg_network.counter == 0:
            set_epsg_network(network)
        fig, ax, xrange, yrange = plot_osm(osm["x"], osm["y"], osm["zoom"])
    elif (osm is False) and cartopy_present:
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5, 5)
        )

    else:
        fig, ax = plt.subplots(figsize=(5, 5))

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
                .groupby(
                    [network.generators.bus, network.generators.carrier],
                    axis=1,
                )
                .sum()
            )
            dispatch_networkB = (
                networkB.generators_t.p[gens.index]
                .mul(networkB.snapshot_weightings.generators, axis=0)
                .groupby(
                    [networkB.generators.bus, networkB.generators.carrier],
                    axis=1,
                )
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
        colors = {
            s[0]: "green" if s[1] > 0 else "red" for s in dispatch.iteritems()
        }
        dispatch = dispatch.abs()
        subcolors = {"negative": "red", "positive": "green"}

    network.plot(
        geomap=(cartopy_present | osm),
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

    import_gen_from_links(network)

    ac_buses = network.buses[network.buses.carrier == "AC"].index
    network.generators = network.generators[
        network.generators.bus.isin(ac_buses)
    ]
    network.generators_t.p = network.generators_t.p.loc[
        :, network.generators_t.p.columns.isin(network.generators.index)
    ]

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
        {
            s[0]: "green" if s[1] > 0 else "red"
            for s in residual_load.iteritems()
        }
    )

    bus_sizes = residual_load.abs() * scaling
    bus_sizes = pd.Series(data=bus_sizes, index=network.buses.index).fillna(0)

    bus_colors = pd.Series(data=bus_colors, index=network.buses.index).fillna(
        "grey"
    )
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

    Returns
    -------
    None.

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
        network.storage_units.max_hours[sbatt]
        * network.storage_units.p_nom_opt[sbatt]
    ).sum()
    cap_hydr = (
        network.storage_units.max_hours[shydr]
        * network.storage_units.p_nom_opt[shydr]
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

    Returns
    -------
    None.

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
        AC line loading in MVA

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
    dc_links = network.links.loc[network.links.carrier == "DC", :]

    link_load = network.links_t.p0[
        network.links.index[network.links.carrier == "DC"]
    ]

    dc_load = pd.Series(index=network.links.index, data=0.0)
    dc_load.loc[dc_links.index] = (
        (
            mul_weighting(network, link_load)
            .loc[network.snapshots[timesteps]]
            .abs()
            .sum()[dc_links.index]
            / dc_links.p_nom_opt
        )
        .fillna(0)
        .values
    )

    return dc_load


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
    network : :class:`pypsa.Network
        Whole network including not extended lines
    extension_lines : pandas.Series
        AC-line expansion
    extension_links : pandas.Series
        DC-line expansion

    """

    network_c = network.copy()

    network_c.lines = network_c.lines[
        network_c.lines.s_nom_extendable
        & (
            (network_c.lines.s_nom_opt - network_c.lines.s_nom_min)
            / network_c.lines.s_nom
            >= ext_min
        )
    ]
    network_c.links = network_c.links[
        network_c.links.p_nom_extendable
        & (network_c.links.carrier == "DC")
        & (
            (network_c.links.p_nom_opt - network_c.links.p_nom_min)
            / network_c.links.p_nom
            >= ext_min
        )
    ]

    for i, row in network_c.links.iterrows():
        linked = network_c.links[
            (row["bus1"] == network_c.links.bus0)
            & (row["bus0"] == network_c.links.bus1)
        ]
        if not linked.empty:
            if row["p_nom_opt"] < linked.p_nom_opt.values[0]:
                network_c.links.p_nom_opt[i] = linked.p_nom_opt.values[0]

    if method == "rel":
        extension_lines = (
            100
            * (network_c.lines.s_nom_opt - network_c.lines.s_nom_min)
            / network_c.lines.s_nom
        )

        extension_links = pd.DataFrame(
            data=network_c.links, index=network_c.links.index
        )

        extension_links = (
            100
            * (network_c.links.p_nom_opt - network_c.links.p_nom_min)
            / (network_c.links.p_nom)
        )
        extension_links = extension_links.fillna(0)

    if method == "abs":
        extension_lines = network_c.lines.s_nom_opt - network_c.lines.s_nom_min

        extension_links = pd.DataFrame(
            data=network_c.links, index=network_c.links.index
        )

        extension_links = network_c.links.p_nom_opt - network_c.links.p_nom_min

    extension_lines = pd.Series(
        data=extension_lines, index=network.lines.index
    ).fillna(0)
    extension_links = pd.Series(
        data=extension_links, index=network.links.index
    ).fillna(0)
    return network, extension_lines, extension_links


def plot_background_grid(network, ax, geographical_boundaries, osm):
    """Plots grid topology in background of other network.plot

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    ax : matplotlib.axes._subplots.AxesSubplot
        axes of plot
    geographical_boundaries : list
        Set georaphical boundaries for the plots
    osm : False or dict.
        False if not osm background map is required or dictionary with
        x, y and zoom information.

    Returns
    -------
    None.

    """
    link_widths = pd.Series(index=network.links.index, data=0)
    link_widths.loc[network.links.carrier == "DC"] = 0.3

    if osm is not False:
        network.plot(
            ax=ax,
            line_colors="grey",
            link_colors="grey",
            bus_sizes=0,
            line_widths=0.5,
            link_widths=link_widths,
            geomap=False,
            boundaries=geographical_boundaries,
        )
    else:
        if cartopy_present:
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
                boundaries=geographical_boundaries,
            )
        else:
            network.plot(
                ax=ax,
                line_colors="grey",
                link_colors="grey",
                bus_sizes=0,
                line_widths=0.5,
                link_widths=link_widths,
                geomap=False,
            )


def demand_side_management(self, buses, snapshots, agg="5h", used=False, apply_on="grid_model"):
    """Calculate shifting potential of demand side management

    Parameters
    ----------
    buses : array
        List of electricity buses.
    snapshots : array
        List of snapshots.
    agg : str, optional
        Temporal resolution. The default is '5h'.
    used : boolean, optional
        State if usage should be included in the results. The default is False.
    apply_on : str, optional
        Choose which network is plotted. The available networks depend on your
        settings. The default is 'grid_model'

    Returns
    -------
    df : pandas.DataFrame
        Shifting potential (and usage) of power (MW) and energy (MWh)

    """
    # Choose which network is plotted
    if apply_on == "grid_model":
        network = self.network.copy()
    elif apply_on == "disaggreagted_network":
        network = self.disaggregated_network.copy()
    elif apply_on == "market_model":
        network = self.market_model.copy()
    elif apply_on == "pre_market_model":
        network = self.pre_market_model.copy()
    else:
        logger.warning(
            """Parameter apply_on must be one of ['grid_model', 'market_model'
            'pre_market_model', 'disaggregated_network'.""")

    df = pd.DataFrame(index=network.snapshots[snapshots])

    link = network.links[
        (network.links.carrier == "dsm")
        & (network.links.bus0.isin(buses))
    ]
    s = network.stores[
        (network.stores.carrier == "dsm")
        & (network.stores.bus.isin(link.bus1.values))
    ]

    df["p_min"] = (
        network.links_t.p_min_pu[link.index]
        .mul(link.p_nom, axis=1)
        .sum(axis=1)
        .resample(agg)
        .mean()
        .iloc[snapshots]
    )
    df["p_max"] = (
        network.links_t.p_max_pu[link.index]
        .mul(link.p_nom, axis=1)
        .sum(axis=1)
        .resample(agg)
        .mean()
        .iloc[snapshots]
    )

    df["e_min"] = (
        network.stores_t.e_min_pu[s.index]
        .mul(s.e_nom, axis=1)
        .sum(axis=1)
        .iloc[snapshots]
    )
    df["e_max"] = (
        network.stores_t.e_max_pu[s.index]
        .mul(s.e_nom, axis=1)
        .sum(axis=1)
        .iloc[snapshots]
    )

    if used:
        df["p"] = (
            network.links_t.p0[link.index]
            .clip(lower=0)
            .sum(axis=1)
            .resample(agg)
            .mean()[snapshots]
        )
        df["e"] = network.stores_t.e[s.index].sum(axis=1).iloc[snapshots]

    return df


def bev_flexibility_potential(
    self,
    buses,
    snapshots,
    agg="5h",
    used=False,
    apply_on="grid_model",
):
    """Calculate shifting potential of electric vehicles

    Parameters
    ----------
    buses : array
        List of electricity buses.
    snapshots : array
        List of snapshots.
    agg : str, optional
        Temporal resolution. The default is '5h'.
    used : boolean, optional
        State if usage should be included in the results. The default is False.
    apply_on : str, optional
        Choose which network is plotted. The available networks depend on your
        settings. The default is 'grid_model'

    Returns
    -------
    df : pandas.DataFrame
        Shifting potential (and usage) of power (MW) and energy (MWh)

    """
    # Choose which network is plotted
    if apply_on == "grid_model":
        network = self.network.copy()
    elif apply_on == "disaggreagted_network":
        network = self.disaggregated_network.copy()
    elif apply_on == "market_model":
        network = self.market_model.copy()
    elif apply_on == "pre_market_model":
        network = self.pre_market_model.copy()
    else:
        logger.warning(
            """Parameter apply_on must be one of ['grid_model', 'market_model'
            'pre_market_model', 'disaggregated_network'.""")

    # Initialize DataFrame
    df = pd.DataFrame(index=network.snapshots[snapshots])

    # Select BEV buses and links
    bev_buses = network.buses[
        network.buses.carrier.str.contains("Li ion")
    ]
    bev_links = network.links[
        (network.links.bus1.isin(bev_buses.index.values))
        & (network.links.bus0.isin(buses))
    ]
    bev_buses = bev_links.bus1.values

    # Maximum loading of BEV charger in MW per BEV bus
    bev_links_t = (
        network.links_t.p_max_pu[bev_links.index]
        .mul(bev_links.p_nom, axis=1)
        .iloc[snapshots]
    )
    bev_links_t.columns = bev_links_t.columns.map(bev_links.bus1)

    # BEV loads per bus
    bev_loads = network.loads[network.loads.bus.isin(bev_buses)]
    bev_loads_t = network.loads_t.p_set[bev_loads.index].iloc[snapshots]
    bev_loads_t.columns = bev_loads_t.columns.map(bev_loads.bus)

    # Maximal positive shifting df is max. loading of charger minus fixed loads
    df["p_max"] = (bev_links_t - bev_loads_t).sum(axis=1).resample(agg).mean()

    # Maximal negative shifting is minus fixed loads
    df["p_min"] = bev_loads_t.mul(-1).sum(axis=1).resample(agg).mean()

    # Select BEV stores (batteries of vehicles)
    bev_stores = network.stores[network.stores.bus.isin(bev_buses)]

    # Calculate maximum and minumum state of charges of battries
    df["e_max"] = (
        network.stores_t.e_max_pu[bev_stores.index]
        .mul(bev_stores.e_nom, axis=1)
        .iloc[snapshots]
        .sum(axis=1)
        .resample(agg)
        .mean()
    )
    df["e_min"] = (
        network.stores_t.e_min_pu[bev_stores.index]
        .mul(bev_stores.e_nom, axis=1)
        .iloc[snapshots]
        .sum(axis=1)
        .resample(agg)
        .mean()
    )

    if used:
        bev_links_t_used = network.links_t.p0[bev_links.index].iloc[
            snapshots
        ]

        bev_links_t_used.columns = bev_links_t_used.columns.map(bev_links.bus1)

        bev_usage = bev_links_t_used - bev_loads_t

        df["p"] = (
            bev_usage.clip(lower=0).sum(axis=1).resample(agg).mean()
            + bev_usage.clip(upper=0)  # always > 0
            .sum(axis=1)
            .resample(agg)
            .mean()
        )  # always < 0
        df["e"] = (
            network.stores_t.e[bev_stores.index]
            .sum(axis=1)
            .resample(agg)
            .mean()
            .iloc[snapshots]
        )

    return df


def heat_stores(
    self,
    buses,
    snapshots,
    agg="5h",
    used=False,
    apply_on="grid_model",
):
    """Calculate shifting potential (and usage) of heat stores

    Parameters
    ----------
    buses : array
        List of electricity buses.
    snapshots : array
        List of snapshots.
    agg : str, optional
        Temporal resolution. The default is '5h'.
    used : boolean, optional
        State if usage should be included in the results. The default is False.
    apply_on : str, optional
        Choose which network is plotted. The available networks depend on your
        settings. The default is 'grid_model'

    Returns
    -------
    df : pandas.DataFrame
        Shifting potential (and usage) of power (MW) and energy (MWh)

    """
    # Choose which network is plotted
    if apply_on == "grid_model":
        network = self.network.copy()
    elif apply_on == "disaggreagted_network":
        network = self.disaggregated_network.copy()
    elif apply_on == "market_model":
        network = self.market_model.copy()
    elif apply_on == "pre_market_model":
        network = self.pre_market_model.copy()
    else:
        logger.warning(
            """Parameter apply_on must be one of ['grid_model', 'market_model'
            'pre_market_model', 'disaggregated_network'.""")

    df = pd.DataFrame(index=network.snapshots[snapshots])

    heat_buses = network.links[
        network.links.bus0.isin(
            network.buses[
                (network.buses.carrier == "AC")
                & (network.buses.index.isin(buses))
            ].index
        )
        & network.links.bus1.isin(
            network.buses[
                network.buses.carrier.str.contains("heat")
            ].index
        )
    ].bus1.unique()

    l_charge = network.links[
        (network.links.carrier.str.contains("heat_store_charger"))
        & (network.links.bus0.isin(heat_buses))
    ]
    l_discharge = network.links[
        (network.links.carrier.str.contains("heat_store_discharger"))
        & (network.links.bus1.isin(heat_buses))
    ]

    s = network.stores[
        (network.stores.carrier.str.contains("heat_store"))
        & (network.stores.bus.isin(l_charge.bus1.values))
    ]

    df["p_min"] = l_discharge.p_nom_opt.mul(-1 * l_discharge.efficiency).sum()
    df["p_max"] = l_charge.p_nom_opt.mul(l_charge.efficiency).sum()

    df["e_min"] = 0
    df["e_max"] = s.e_nom_opt.sum()

    if used:
        df["p"] = (
            network.links_t.p1[l_charge.index]
            .mul(-1)
            .sum(axis=1)
            .resample(agg)
            .mean()[snapshots]
            + network.links_t.p0[l_discharge.index]
            .mul(-1)
            .sum(axis=1)
            .resample(agg)
            .mean()[snapshots]
        )
        df["e"] = network.stores_t.e[s.index].sum(axis=1).iloc[snapshots]

    return df


def hydrogen_stores(
    self,
    buses,
    snapshots,
    agg="5h",
    used=False,
    apply_on="grid_model",
):
    """Calculate shifting potential (and usage) of heat stores

    Parameters
    ----------
    buses : array
        List of electricity buses.
    snapshots : array
        List of snapshots.
    agg : str, optional
        Temporal resolution. The default is '5h'.
    used : boolean, optional
        State if usage should be included in the results. The default is False.
    apply_on : str, optional
        Choose which network is plotted. The available networks depend on your
        settings. The default is 'grid_model'

    Returns
    -------
    df : pandas.DataFrame
        Shifting potential (and usage) of power (MW) and energy (MWh)

    """
    # Choose which network is plotted
    if apply_on == "grid_model":
        network = self.network.copy()
    elif apply_on == "disaggreagted_network":
        network = self.disaggregated_network.copy()
    elif apply_on == "market_model":
        network = self.market_model.copy()
    elif apply_on == "pre_market_model":
        network = self.pre_market_model.copy()
    else:
        logger.warning(
            """Parameter apply_on must be one of ['grid_model', 'market_model'
            'pre_market_model', 'disaggregated_network'.""")

    df = pd.DataFrame(index=network.snapshots[snapshots])

    h2_buses = network.links[
        network.links.bus0.isin(
            network.buses[
                (network.buses.carrier == "AC")
                & (network.buses.index.isin(buses))
            ].index
        )
        & network.links.bus1.isin(
            network.buses[
                network.buses.carrier.str.contains("H2")
            ].index
        )
    ].bus1.unique()

    s = network.stores[network.stores.bus.isin(h2_buses)]

    df["p_min"] = network.stores_t.p[s.index].sum(axis=1).min()
    df["p_max"] = network.stores_t.p[s.index].sum(axis=1).max()

    df["e_min"] = 0
    df["e_max"] = s.e_nom_opt.sum()

    if used:
        df["p"] = network.stores_t.p[s.index].sum(axis=1).iloc[snapshots]
        df["e"] = network.stores_t.e[s.index].sum(axis=1).iloc[snapshots]

    return df


def flexibility_usage(
    self, flexibility, agg="5h", snapshots=[], buses=[], pre_path=None, apply_on="grid_model",
):
    """Plots temporal distribution of potential and usage for flexibilities

    Parameters
    ----------
    flexibility : str
        Name of flexibility option.
    agg : str, optional
        Temporal resolution. The default is "5h".
    snapshots : list, optional
        Considered snapshots, if empty all are considered. The default is [].
    buses : list, optional
        Considered components at AC buses, if empty all are considered.
        The default is [].
    pre_path : str, optional
        State of and where you want to store the figure. The default is None.
    apply_on : str, optional
        Choose which network is plotted. The available networks depend on your
        settings. The default is 'grid_model'

    Returns
    -------
    None.

    """
    # Choose which network is plotted
    if apply_on == "grid_model":
        network = self.network.copy()
    elif apply_on == "disaggreagted_network":
        network = self.disaggregated_network.copy()
    elif apply_on == "market_model":
        network = self.market_model.copy()
    elif apply_on == "pre_market_model":
        network = self.pre_market_model.copy()
    else:
        logger.warning(
            """Parameter apply_on must be one of ['grid_model', 'market_model'
            'pre_market_model', 'disaggregated_network'.""")

    colors = coloring()
    colors["dlr"] = "orange"
    colors["h2_store"] = colors["H2_underground"]
    colors["heat"] = colors["central_heat_store"]

    if not buses:
        buses = network.buses.index

    if len(snapshots) == 0:
        snapshots = range(1, len(network.snapshots))

    if flexibility == "dsm":
        df = demand_side_management(
            self,
            buses,
            snapshots,
            agg,
            used=True,
            apply_on=apply_on,
        )

    elif flexibility == "BEV charger":
        df = bev_flexibility_potential(
            self,
            buses,
            snapshots,
            agg,
            used=True,
            apply_on=apply_on,
        )

    elif flexibility == "heat":
        df = heat_stores(
            self,
            buses,
            snapshots,
            agg,
            used=True,
            apply_on=apply_on,
        )

    elif flexibility == "battery":
        df = pd.DataFrame(index=network.snapshots[snapshots])

        su = network.storage_units[
            (network.storage_units.carrier == "battery")
            & (network.storage_units.bus.isin(buses))
        ]

        df["p_min"] = su.p_nom_opt.sum() * (-1)
        df["p_max"] = su.p_nom_opt.sum()
        df["p"] = (
            network.storage_units_t.p[su.index]
            .sum(axis=1)
            .iloc[snapshots]
        )

        df["e_min"] = 0
        df["e_max"] = su.p_nom_opt.mul(su.max_hours).sum()
        df["e"] = (
            network.storage_units_t.state_of_charge[su.index]
            .sum(axis=1)
            .iloc[snapshots]
        )

    elif flexibility == "h2_store":
        df = hydrogen_stores(
            self,
            buses,
            snapshots,
            agg,
            used=True,
            apply_on=apply_on,
        )

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.fill_between(
        df.index, df.p_min, df.p_max, color=colors[flexibility], alpha=0.2
    )
    ax.plot(df.index, df.p, color=colors[flexibility])
    ax.set_ylabel("shifted power in MW")
    ax.set_xlim(df.index[0], df.index[-1])
    if pre_path:
        fig.savefig(pre_path + f"shifted_p_{flexibility}")

    fig_e, ax_e = plt.subplots(figsize=(15, 5))
    ax_e.fill_between(
        df.index, df.e_min, df.e_max, color=colors[flexibility], alpha=0.2
    )
    ax_e.plot(df.index, df.e, color=colors[flexibility])
    ax_e.set_ylabel("stored energy in MWh")
    ax_e.set_xlim(df.index[0], df.index[-1])
    if pre_path:
        fig_e.savefig(pre_path + f"stored_e_{flexibility}")


def plot_carrier(etrago, carrier_links=["AC"], carrier_buses=["AC"], apply_on="grid_model"):
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
    # Choose network or disaggregated_network
    if apply_on == "grid_model":
        network = etrago.network.copy()
    elif apply_on == "disaggreagted_network":
        network = etrago.disaggregated_network.copy()
    elif apply_on == "market_model":
        network = etrago.market_model.copy()
    elif apply_on == "pre_market_model":
        network = etrago.pre_market_model.copy()
    else:
        logger.warning(
            """Parameter apply_on must be one of ['grid_model', 'market_model'
            'pre_market_model', 'disaggregated_network'.""")

    colors = coloring()
    line_colors = "lightblue"

    # Set background
    if cartopy_present:
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
        geomap=cartopy_present,
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
    apply_on="grid_model",
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion=False,
    geographical_boundaries=[-2.5, 16, 46.8, 58],
):
    """Function that plots etrago.network and results for lines and buses

    Parameters
    ----------
    line_colors : str
        Set static line color or attribute to plot e.g. 'expansion_abs'
        Current options:

        * 'line_loading': mean line loading in p.u. in selected timesteps
        * 'v_nom': nominal voltage of lines
        * 'expansion_abs': absolute network expansion in MVA
        * 'expansion_rel': network expansion in p.u. of existing capacity
        * 'q_flow_max': maximal reactive flows
        * 'dlr': energy above nominal capacity
        * 'grey': plot all lines and DC links grey colored

    bus_sizes : float, optional
        Size of buses. The default is 0.001.
    bus_colors : str, optional
        Set static bus color or attribute to plot. The default is 'grey'.
        Current options:

        * 'nodal_production_balance': net producer/consumer in selected timeteps
        * 'storage_expansion': storage expansion per bus and technology
        * 'storage_distribution': installed storage units per bus
        * 'h2_battery_storage_expansion': storage expansion per bus and
           technology for underground and overground H2 and batteries.
        * 'gen_dist': dispatch per carrier in selected timesteps
        * 'ramp_up': re-dispatch up per carrier in selected timesteps
        * 'ramp_down': re-dispatch down per carrier in selected timesteps
        * 'PowerToH2': location and sizes of electrolyzers
        * 'flexibility_usage': use of DSM and BEV charger
        * 'PowerToH2_correlation': indication of degree of correlation to 
        market or nodal price of electrolyzers

    timesteps : array, optional
        Timesteps consideredd in time depended plots. The default
        is range(2).
    osm : bool or dict, e.g. {'x': [1,20], 'y': [47, 56], 'zoom' : 6}
        If not False, osm is set as background
        with the following settings as dict:

        * 'x': array of two floats, x axis boundaries (lat)
        * 'y': array of two floats, y axis boundaries (long)
        * 'zoom' : resolution of osm. The default is False.

    boundaries: array
       Set fixed boundaries of heatmap axis. The default is None.
    filename: str or None
        Save figure in this direction. The default is None.
    apply_on : str, optional
        Choose which network is plotted. The available networks depend on your
        settings. The default is 'grid_model'
    ext_min: float
        Choose minimum relative line extension shown in plot in p.u..
    ext_width: float or bool
        Choose if line_width respects line extension. Turn off with
        'False' or set linear factor to decremise extension line_width.
        The default is False.
    legend_entries : list, optional
        Set the legends for buses to be plotted. The default is 'all'.
    scaling_store_expansion : dict, optional
        Set scaling values to be used per technology for the plots
        storage_expansion and h2_battery_storage_expansion. The default is
        False, it could be assinged like this:
        {"H2": 50, "heat": 0.1, "battery": 10}
    geographical_boundaries : list, optional
        Set georaphical boundaries for the plots. This parameter is overwritten
        when osm is used. The default is [-2.5, 16, 46.8, 58]

    Returns
    -------
    None.

    """
    # Choose network or disaggregated_network
    if apply_on == "grid_model":
        network = self.network.copy()
    elif apply_on == "disaggreagted_network":
        network = self.disaggregated_network.copy()
    elif apply_on == "market_model":
        network = self.market_model.copy()
    elif apply_on == "pre_market_model":
        network = self.pre_market_model.copy()
    else:
        logger.warning(
            """Parameter apply_on must be one of ['grid_model', 'market_model'
            'pre_market_model', 'disaggregated_network'.""")

    # Set colors for plotting
    plotting_colors(network)

    # Set default values
    flow = None
    title = ""
    line_widths = 2
    link_widths = 0

    # Plot osm map in background
    if osm is not False:
        if network.srid == 4326:
            set_epsg_network(network)
        fig, ax, xrange, yrange = plot_osm(osm["x"], osm["y"], osm["zoom"])
        geographical_boundaries = [xrange[0], xrange[1], yrange[0], yrange[1]]

    elif (osm is False) and cartopy_present:
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5, 5)
        )

    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    fig.set_tight_layout(True)

    # Set line colors
    if line_colors == "line_loading":
        title = "Mean line loading"
        rep_snapshots = network.snapshot_weightings["objective"][
            network.snapshots[timesteps]
        ].sum()
        line_colors = calc_ac_loading(network, timesteps).abs() / rep_snapshots
        link_colors = calc_dc_loading(network, timesteps).abs() / rep_snapshots
        if ext_width is not False:
            link_widths = link_colors.apply(
                lambda x: 10 + (x / ext_width) if x != 0 else 0
            )
            line_widths = 10 + (line_colors / ext_width)
        else:
            link_widths = link_colors.apply(lambda x: 10 if x != 0 else 0)
            line_widths = 10
        label = "line loading in p.u."
        plot_background_grid(network, ax, geographical_boundaries, osm)
        # Only active flow direction is displayed!
        flow = pd.Series(1, index=network.branches().index, dtype="float64")
        flow.iloc[flow.index.get_level_values("component") == "Line"] = (
            mul_weighting(network, network.lines_t.p0)
            .loc[network.snapshots[timesteps]]
            .sum()
            / network.lines.s_nom
            / rep_snapshots
        ).values

        dc_loading = calc_dc_loading(network, timesteps) / rep_snapshots
        dc_loading.index = pd.MultiIndex.from_tuples(
            [("Link", name) for name in dc_loading.index],
            names=["component", "name"],
        )
        flow.loc["Link", :] = dc_loading

        flow = flow[
            (flow.index.get_level_values("component") == "Line")
            | (
                flow.index.isin(
                    link_widths[
                        link_widths.index.isin(
                            network.links[network.links.carrier == "DC"].index
                        )
                    ].index,
                    level=1,
                )
            )
        ]
        flow[flow < 0] = -1
        flow[flow > 0] = 1

    elif line_colors == "v_nom":
        title = "Voltage levels"
        label = "v_nom in kV"
        line_colors = network.lines.v_nom
        link_colors = pd.Series(data=0, index=network.links.index)
        plot_background_grid(network, ax, geographical_boundaries, osm)
    elif line_colors == "expansion_abs":
        title = "Network expansion"
        label = "network expansion in GVA"
        all_network, line_colors, link_colors = calc_network_expansion(
            network, method="abs", ext_min=ext_min
        )
        plot_background_grid(all_network, ax, geographical_boundaries, osm)

        if ext_width is not False:
            line_widths = line_colors / ext_width
            link_widths = link_colors.apply(
                lambda x: x / ext_width if x != 0 else 0
            )
        else:
            dc_link = network.links.index[network.links.carrier == "DC"]
            link_widths = pd.Series(0, index=network.links.index)
            link_widths.loc[dc_link] = 1.5
            line_widths = line_colors.apply(lambda x: 1.5 if x != 0 else 0)

        link_colors = link_colors.mul(1e-3)
        line_colors = line_colors.mul(1e-3)

    elif line_colors == "expansion_rel":
        title = "Network expansion"
        label = "network expansion in %"
        all_network, line_colors, link_colors = calc_network_expansion(
            network, method="rel", ext_min=ext_min
        )
        plot_background_grid(all_network, ax, geographical_boundaries, osm)
        if ext_width is not False:
            line_widths = 0.5 + (line_colors / ext_width)
            link_widths = link_colors.apply(
                lambda x: 0.5 + x / ext_width if x != 0 else 0
            )
        else:
            dc_link = network.links.index[network.links.carrier == "DC"]
            link_widths = pd.Series(0, index=network.links.index)
            link_widths.loc[dc_link] = 2
            line_widths = line_colors.apply(lambda x: 1.5 if x != 0 else 0)
    elif line_colors == "q_flow_max":
        title = "Maximum reactive power flows"
        label = "flow in pu"
        line_colors = abs(
            network.lines_t.q0.abs().max() / (network.lines.s_nom)
        )
        if ext_width is not False:
            line_widths = 0.5 + (line_colors / ext_width)
        link_colors = pd.Series(data=0, index=network.links.index)
        plot_background_grid(network, ax, geographical_boundaries, osm)
    elif line_colors == "dlr":
        title = "Dynamic line rating"
        label = "TWh above nominal capacity"
        plot_background_grid(network, ax, geographical_boundaries, osm)

        # calc min capacity per line in the given period: Since lines with
        # different original voltage level could be aggregated during the
        # clustering, the security factors can be values in between the values
        # provided in the args for branch_capacity_factor.
        network.lines.s_max_pu = network.lines_t.s_max_pu.min()
        line_loading = network.lines_t.p0.mul(
            1 / (network.lines.s_nom_opt * network.lines.s_max_pu)
        ).abs()
        line_loading = line_loading.iloc[timesteps, :]
        # keep only the capacity allowed by dlr
        line_loading = line_loading - 1
        dlr_usage = (
            line_loading[line_loading > 0]
            .fillna(0)
            .mul(network.snapshot_weightings.generators, axis=0)
            .sum()
        )
        dlr_usage = (
            dlr_usage * network.lines.s_nom * network.lines.s_max_pu / 1000000
        )
        dlr_usage = dlr_usage.round(decimals=0)
        line_colors = dlr_usage
        if ext_width is not False:
            line_widths = 0.2 + (line_colors / ext_width)
        link_colors = pd.Series(data=0, index=network.links.index)

    elif line_colors == "grey":
        title = ""
        label = ""
        line_colors = "grey"
        link_colors = "grey"
        plot_background_grid(network, ax, geographical_boundaries, osm)
        link_widths = 0
        line_widths = 0

    else:
        logger.warning("line_color {} undefined".format(line_colors))

    # Set bus colors
    bus_legend = False

    if bus_colors == "nodal_production_balance":
        bus_scaling = bus_sizes
        bus_sizes, bus_colors = nodal_production_balance(
            network, timesteps, scaling=bus_scaling
        )
        bus_legend = "Nodal production balance"
        bus_unit = "TWh"
    elif bus_colors == "storage_expansion":
        if not isinstance(scaling_store_expansion, dict):
            raise Exception(
                """To plot storage_expansion, the argument\
            scaling_store_expansion must be a dictionary like:
                            {"H2": 50,
                            "heat": 0.1,
                            "battery": 10}"""
            )
        bus_scaling = bus_sizes
        bus_sizes = bus_scaling * calc_storage_expansion_per_bus(network)
        for store_carrier in scaling_store_expansion.keys():
            bus_sizes[
                bus_sizes.index.get_level_values("carrier").str.contains(
                    store_carrier
                )
            ] *= scaling_store_expansion[store_carrier]
        bus_legend = "Storage expansion"
        bus_unit = "GW"
    elif bus_colors == "h2_battery_storage_expansion":
        bus_scaling = bus_sizes
        bus_sizes = bus_scaling * calc_storage_expansion_per_bus(
            network, carriers=["battery", "H2_overground", "H2_underground"]
        )
        if (
            ("battery" not in scaling_store_expansion.keys())
            | ("H2_overground" not in scaling_store_expansion.keys())
            | ("H2_underground" not in scaling_store_expansion.keys())
        ):
            raise Exception(
                """To plot h2_battery_storage_expansion, the argument\
            scaling_store_expansion must be a dictionary like:
                            {"H2_overground": 1,
                             "H2_underground": 1,
                             "battery": 1,}"""
            )

        for store_carrier in ["battery", "H2_overground", "H2_underground"]:
            bus_sizes[
                bus_sizes.index.get_level_values("carrier").str.contains(
                    store_carrier
                )
            ] *= scaling_store_expansion[store_carrier]
        bus_legend = "Battery and H2 storage expansion"
        bus_unit = "GW"
    elif bus_colors == "storage_distribution":
        bus_scaling = bus_sizes
        bus_sizes = (
            network.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum()
            * bus_scaling
        )
        bus_legend = "Storage distribution"
        bus_unit = "TW"
    elif bus_colors in ("gen_dist", "ramp_up", "ramp_down"):
        bus_scaling = bus_sizes
        bus_sizes = bus_scaling * calc_dispatch_per_carrier(
            network, timesteps, bus_colors
        )
        bus_legend = "Dispatch"
        bus_unit = "TW"
    elif bus_colors == "flexibility_usage":
        bus_scaling = bus_sizes
        flex_links = network.links[
            network.links.carrier.isin(
                [
                    "dsm",
                    "BEV charger",
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
        bus_sizes = (
            bus_scaling * flex_links.groupby(["bus0", "carrier"]).p0_sum.sum()
        )
        bus_unit = "TWh"
        bus_legend = "flexibility_usage"
    elif bus_colors == "h2_storage_expansion":
        bus_scaling = bus_sizes
        bus_sizes = bus_scaling * calc_storage_expansion_per_bus(network)
        bus_sizes = bus_sizes.reset_index()
        bus_sizes = bus_sizes[bus_sizes.carrier.str.contains("H2")]
        bus_sizes.set_index(["bus", "carrier"], inplace=True)
        bus_legend = "Storage expansion"
        bus_unit = "GW"
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
        if len(bus_sizes) == 0:
            print("There is no PowerToH2 to plot")
        bus_colors = coloring()["power_to_H2"]
        bus_legend = "PowerToH2"
        bus_unit = "TW"
    elif (
        bus_colors == "PowerToH2_correlation"
    ):  # PowerToH2 plots p_nom_opt of links with carrier=power to H2
        bus_scaling = bus_sizes
        bus_sizes = (
            bus_scaling
            * network.links[(network.links.carrier == "power_to_H2")]
            .groupby("bus0")
            .sum()
            .p_nom_opt
        )
        if len(bus_sizes) == 0:
            print("There is no PowerToH2 to plot")
        bus_colors = coloring()["power_to_H2"]
        bus_legend = "PowerToH2"
        bus_unit = "TW"
        

        
        market_bus_de = self.market_model.buses[(self.market_model.buses.country=="DE") & (self.market_model.buses.carrier=="AC")].index
        market_price = self.market_model.buses_t.marginal_price[market_bus_de]
        
        bus_colors = pd.Series(index=network.buses.index, data=0)
        for bus in network.links[(network.links.carrier == "power_to_H2")].bus0:
            
            nodal_price = network.buses_t.marginal_price[bus]
            
            ely = network.links_t.p0[network.links[(network.links.carrier == "power_to_H2")
                                                   & (network.links.bus0==bus)].index]
            df_corr = pd.DataFrame()
    
            df_corr["ely"] = ely
            df_corr["market"] = market_price
            df_corr["nodal_price"] = nodal_price
    
            bus_colors[bus] = (df_corr.corr(method = 'spearman').loc["nodal_price", "ely"]/ (                            
                               df_corr.corr(method = 'spearman').loc["nodal_price", "ely"])+
                               df_corr.corr(method = 'spearman').loc["market", "ely"])
            
        bus_colors = bus_colors.abs()
            
        # ely.corr
        # ely_corr_market = ely.corrwith(
        #     market_price, method = 'spearman', axis=1)
        # ely_corr_nodal = ely.corr(nodal_price, method = 'spearman')
    elif bus_colors == "grey":
        bus_scaling = bus_sizes
        bus_sizes = pd.Series(
            data=network.buses.carrier, index=network.buses.index
        )
        bus_sizes[bus_sizes != "AC"] = 0
        bus_sizes[bus_sizes == "AC"] = 1 * bus_scaling
        bus_scaling = bus_sizes
    else:
        logger.warning("bus_color {} undefined".format(bus_colors))

    if cartopy_present & (osm is False):
        ll = network.plot(
            line_colors=line_colors,
            link_colors=link_colors,
            line_cmap=plt.cm.viridis,
            bus_alpha=0.9,
            link_cmap=plt.cm.viridis,
            bus_sizes=bus_sizes,
            bus_colors=bus_colors,
            bus_cmap = plt.cm.viridis,
            line_widths=line_widths,
            link_widths=link_widths,
            flow=flow,
            #title=title,
            geomap=False,
            projection=ccrs.PlateCarree(),
            color_geomap=True,
            boundaries=geographical_boundaries,
        )
    else:
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
            boundaries=geographical_boundaries,
        )
    l3 = None

    # legends for bus sizes and colors
    if bus_legend:
        handles = []
        labels = []
        if scaling_store_expansion:
            if not isinstance(legend_entries, list):
                if bus_legend == "Storage expansion":
                    legend_entries = list(scaling_store_expansion.keys())
                if bus_legend == "Battery and H2 storage expansion":
                    legend_entries = [
                        "battery",
                        "H2_overground",
                        "H2_underground",
                    ]
            for i in legend_entries:
                try:
                    max_value = bus_sizes[
                        bus_sizes.index.get_level_values(
                            "carrier"
                        ).str.contains(i)
                    ].max()
                except KeyError:
                    max_value = bus_sizes.max()
                handles.append(
                    make_legend_circles_for(
                        [max_value],
                        scale=1,
                        facecolor=network.carriers.color[i],
                    )[0]
                )
                labels.append(
                    f"""
                    {round(max_value/bus_scaling/scaling_store_expansion[i]/
                           1000, 0)} {bus_unit} """
                    + i
                )
        else:
            if len(bus_sizes) > 0:
                max_value = bus_sizes.max()
            else:
                max_value = 0
            labels.append(f"{round(max_value / bus_scaling /1000, 0)} GWh ")
            handles.append(
                make_legend_circles_for(
                    [max_value],
                    scale=1,
                    facecolor="grey",
                )[0]
            )

        l2 = ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.01, 1.01),
            labelspacing=1.0,
            framealpha=1.0,
            title=bus_legend,
            handler_map=make_handler_map_to_scale_circles_as_in(ax),
            prop={"size": 8},
        )
        ax.add_artist(l2)

        plt.setp(l2.get_title(), fontsize="9")

        if not scaling_store_expansion:
            handles = []
            if bus_legend == "Nodal production balance":
                positive = mpatches.Patch(color="green", label="generation")
                negative = mpatches.Patch(color="red", label="consumption")
                handles = [positive, negative]

            elif bus_legend == "PowerToH2":
                pth = mpatches.Patch(color="cyan", label="PowerToH2")
                handles = [pth]
            elif legend_entries != "all":
                for i in legend_entries:
                    patch = mpatches.Patch(
                        color=network.carriers.color[i], label=i
                    )
                    handles.append(patch)
            else:
                for i in bus_sizes.index.get_level_values("carrier").unique():
                    patch = mpatches.Patch(
                        color=network.carriers.color[i], label=i
                    )
                    handles.append(patch)

            l3 = plt.legend(
                handles=handles,
                loc="upper left",
                ncol=2,
                bbox_to_anchor=(0, 0),
            )
            ax.add_artist(l3)

    if type(line_colors) != str:
        # Set fixed boundaries if selected in parameters
        if not boundaries:
            boundaries = [
                min(round(line_colors.min(), 1), round(link_colors.min(), 1)),
                max(round(line_colors.max()), round(link_colors.max())),
            ]

        # Create ticks for legend
        v = [
            round(x, 1) for x in np.linspace(boundaries[0], boundaries[1], 101)
        ]
        for l_collection in ll:
            l_collection.set_clim(boundaries[0], boundaries[1])

        # colorbar for line heatmap
        cb = plt.colorbar(
            ll[1],
            values=v,
            ticks=v[0:101:10],
            #fraction=0.028,
            pad=0.04,
            orientation="horizontal"
        )
        # Set legend label
        cb.set_label(label)
        
    elif type(bus_colors) != str:
        #import pdb; pdb.set_trace()
        ll[0].set_clim([0, bus_colors.max()]) 
        plt.colorbar(ll[0], fraction=0.04, pad=0.004, label="correlation factor", ax=ax)


    # Show plot or save to file
    if filename is None:
        if not isinstance(bus_sizes, (pd.Series, float)):
            logger.warning("Legend of bus sizes will change when zooming")
        plt.tight_layout()
        plt.show()
    else:
        from matplotlib import pylab

        if l3 is None:
            pylab.savefig(filename, dpi=300, bbox_inches="tight")
        else:
            pylab.savefig(
                filename, dpi=300, bbox_inches="tight", bbox_extra_artists=[l3]
            )
        plt.close()


set_epsg_network.counter = 0

# the following functions are copied from pypsa-eur-sec. see:
# https://github.com/PyPSA/pypsa-eur-sec/blob/master/scripts/plot_network.py


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
    map_buses["geom"] = map_buses.apply(
        lambda x: Point(x["x"], x["y"]), axis=1
    )
    map_buses["cluster"] = map_buses.index.map(self.busmap["busmap"])
    map_buses["cluster_geom"] = map_buses["cluster"].map(new_geom.geom)
    map_buses["line"] = map_buses.apply(
        lambda x: LineString((x["geom"], x["cluster_geom"])), axis=1
    )

    # Set background
    if cartopy_present:
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
        lines = gpd.GeoDataFrame(
            self.busmap["orig_network"].lines, geometry="geom"
        )
        lines = lines[
            lines["bus0"].isin(map_buses.index)
            & lines["bus1"].isin(map_buses.index)
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


def plot_gas_generation(
    self, t_resolution="20H", save_path=False
):  # FIXXXXXXXXXXXXXXXX
    """
    Plots timeseries data for gas generation

    Parameters
    ----------
    self : :class:`Etrago
        Overall container of Etrago
    t_resolution : str, optional
        sets the resampling rate of timeseries data to allow for smoother
        line plots
    save_path : bool, optional
        Path to save the generated plot. The default is False.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    colors = coloring()

    ch4_gens_feedin = self.network.generators_t.p[
        [col for col in self.network.generators_t.p.columns if "CH4" in col]
    ]  # active power at bus
    ch4_links_feedin = -self.network.links_t.p1[
        self.network.links.loc[self.network.links.carrier == "H2_to_CH4"].index
    ]  # p1 is output p of H2_to_CH4
    h2_links_feedin = -self.network.links_t.p1[
        self.network.links.loc[self.network.links.carrier == "H2_feedin"].index
    ]

    total_gen_per_t = ch4_gens_feedin.sum(axis=1) / 1e3
    total_link_per_t = ch4_links_feedin.sum(axis=1) / 1e3
    total_h2_per_t = h2_links_feedin.sum(axis=1) / 1e3

    (total_gen_per_t + total_link_per_t + total_h2_per_t).resample(
        t_resolution
    ).mean().plot(
        ax=ax,
        title="Gas Generation",
        ylabel="[GW]",
        legend=True,
        label="Total Gas Dispatch",
    )
    total_gen_per_t.plot(
        ax=ax, label="CH4 Generator Dispatch", legend=True, color=colors["CH4"]
    )
    total_h2_per_t.resample(t_resolution).mean().plot(
        ax=ax,
        label="H2_feedin Dispatch",
        legend=True,
        color=colors["H2_feedin"],
    )
    total_link_per_t.resample(t_resolution).mean().plot(
        ax=ax,
        label="H2_to_CH4 Link Dispatch",
        legend=True,
        color=colors["H2_to_CH4"],
    )

    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_gas_summary(self, t_resolution="20H", stacked=True, save_path=False):
    """
    Plots timeseries data for gas loads (and generation)

    Parameters
    ----------
    self : :class:`Etrago
        Overall container of Etrago
    t_resolution : str, optional
        sets the resampling rate of timeseries data to allow for smoother
        line plots
    stacked : bool, optional
        If True all TS data will be shown as stacked area plot. Total gas
        generation will then also be plotted to check for matching demand and
        generation.
    save_path : bool, optional
        Path to save the generated plot. The default is False.

    Returns
    -------
    None.

    """
    colors = coloring()

    ch4_load_carrier = ["rural_gas_boiler", "CH4_for_industry", "CH4"]

    rel_ch4_loads = self.network.links.loc[
        self.network.links.bus0.isin(
            self.network.buses.loc[self.network.buses.carrier == "CH4"].index
        )
    ].carrier.unique()
    rel_ch4_loads = np.delete(rel_ch4_loads, np.where(rel_ch4_loads == "CH4"))

    data = self.network.links_t.p0[
        self.network.links.loc[
            self.network.links.carrier == rel_ch4_loads[0]
        ].index.to_list()
    ]

    if stacked:
        data = (
            pd.DataFrame(data.sum(axis=1)).resample(t_resolution).mean() / 1e3
        )
        data = data.rename(columns={0: rel_ch4_loads[0]})

        for i in rel_ch4_loads[1:]:
            loads = self.network.links_t.p0[
                self.network.links.loc[
                    self.network.links.carrier == i
                ].index.to_list()
            ]
            data[i] = loads.sum(axis=1).resample(t_resolution).mean() / 1e3

        for i in ch4_load_carrier:
            loads = self.network.loads_t.p[
                self.network.loads.loc[
                    self.network.loads.carrier == i
                ].index.to_list()
            ]
            data[i] = loads.sum(axis=1).resample(t_resolution).mean() / 1e3

        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        data.plot.area(
            ax=ax,
            title="Stacked Gas Loads and Generation by carrier",
            ylabel="[GW]",
            legend=True,
            stacked=True,
        )

        ch4_gens_feedin = self.network.generators_t.p[
            [
                col
                for col in self.network.generators_t.p.columns
                if "CH4" in col
            ]
        ]  # active power at bus
        ch4_links_feedin = -self.network.links_t.p1[
            self.network.links.loc[
                self.network.links.carrier == "H2_to_CH4"
            ].index
        ]  # p1 is output p of H2_to_CH4
        h2_links_feedin = -self.network.links_t.p1[
            self.network.links.loc[
                self.network.links.carrier == "H2_feedin"
            ].index
        ]

        total_gen_per_t = ch4_gens_feedin.sum(axis=1) / 1e3
        total_link_per_t = ch4_links_feedin.sum(axis=1) / 1e3
        total_h2_per_t = h2_links_feedin.sum(axis=1) / 1e3

        (total_gen_per_t + total_link_per_t + total_h2_per_t).resample(
            t_resolution
        ).mean().plot.line(
            ax=ax,
            legend=True,
            label="Total_Gas_generation",
            color=colors["CH4"],
            linestyle="dashed",
        )

        stores = self.network.stores.loc[self.network.stores.carrier == "CH4"]
        a = self.network.stores_t.p[stores.index].sum(axis=1) / 1e3
        (total_gen_per_t + total_link_per_t + total_h2_per_t + a).resample(
            t_resolution
        ).mean().plot.line(
            ax=ax,
            legend=True,
            label="Total_Gas_generation + Gas Storage dispatch",
            color="black",
            linestyle="dashed",
        )

    else:
        data = data.sum(axis=1).resample(t_resolution).mean() / 1e3
        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        data.plot(
            ax=ax,
            title="Gas Loads by carrier",
            label=rel_ch4_loads[0],
            ylabel="[GW]",
            legend=True,
        )

        for i in rel_ch4_loads[1:]:
            data = self.network.links_t.p0[
                self.network.links.loc[
                    self.network.links.carrier == i
                ].index.to_list()
            ]
            data = data.sum(axis=1).resample(t_resolution).mean() / 1e3
            data.plot(ax=ax, label=i, legend=True)

        data = self.network.loads_t.p[
            self.network.loads.loc[
                self.network.loads.carrier == ch4_load_carrier[0]
            ].index.to_list()
        ]
        data = data.sum(axis=1).resample(t_resolution).mean() / 1e3
        data.plot(ax=ax, label=ch4_load_carrier[0], ylabel="[GW]", legend=True)

        for i in ch4_load_carrier[1:]:
            data = self.network.loads_t.p[
                self.network.loads.loc[
                    self.network.loads.carrier == i
                ].index.to_list()
            ]
            data = data.sum(axis=1).resample(t_resolution).mean() / 1e3
            data.plot(ax=ax, label=i, legend=True)

    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_h2_generation(self, t_resolution="20H", save_path=False):
    """
    Plots timeseries data for H2 generation

    Parameters
    ----------
    self : :class:`Etrago
        Overall container of Etrago
    t_resolution : str, optional
        sets the resampling rate of timeseries data to allow for smoother
        line plots
    save_path : bool, optional
        Path to save the generated plot. The default is False.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    colors = coloring()

    h2_CH4_gen = -self.network.links_t.p1[
        self.network.links.loc[self.network.links.carrier == "CH4_to_H2"].index
    ]
    h2_power_gen = -self.network.links_t.p1[
        self.network.links.loc[
            self.network.links.carrier == "power_to_H2"
        ].index
    ]

    (h2_CH4_gen.sum(axis=1) / 1e3 + h2_power_gen.sum(axis=1) / 1e3).resample(
        t_resolution
    ).mean().plot(
        ax=ax,
        title="H2 Generation",
        legend=True,
        ylabel="[GW]",
        label="Total dispatch",
        lw=5,
    )
    (h2_CH4_gen.sum(axis=1) / 1e3).resample(t_resolution).mean().plot(
        ax=ax,
        label="CH4_to_H2 Dispatch",
        legend=True,
        color=colors["CH4_to_H2"],
    )
    (h2_power_gen.sum(axis=1) / 1e3).resample(t_resolution).mean().plot(
        ax=ax,
        label="power_to_H2 Dispatch",
        legend=True,
        color=colors["power_to_H2"],
    )

    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_h2_summary(self, t_resolution="20H", stacked=True, save_path=False):
    """
    Plots timeseries data for H2 loads (and generation)

    Parameters
    ----------
    self : :class:`Etrago
        Overall container of Etrago
    t_resolution : str, optional
        sets the resampling rate of timeseries data to allow for smoother
        line plots
    stacked : bool, optional
        If True all TS data will be shown as stacked area plot. Total H2
        generation will then also be plotted to check for matching demand and
        generation.
    save_path : bool, optional
        Path to save the generated plot. The default is False.

    Returns
    -------
    None.

    """

    rel_h2_links = ["H2_feedin", "H2_to_CH4", "H2_to_power"]
    rel_h2_loads = ["H2_for_industry", "H2_hgv_load"]

    data = self.network.links_t.p0[
        self.network.links.loc[
            self.network.links.carrier == rel_h2_links[0]
        ].index.to_list()
    ]

    if stacked:
        data = (
            pd.DataFrame(data.sum(axis=1)).resample(t_resolution).mean() / 1e3
        )
        data = data.rename(columns={0: rel_h2_links[0]})

        for i in rel_h2_links[1:]:
            loads = self.network.links_t.p0[
                self.network.links.loc[
                    self.network.links.carrier == i
                ].index.to_list()
            ]
            data[i] = loads.sum(axis=1).resample(t_resolution).mean() / 1e3

        DE_loads = self.network.loads.loc[
            self.network.loads.bus.isin(
                self.network.buses.loc[
                    self.network.buses.country == "DE"
                ].index
            )
        ]
        for i in rel_h2_loads:
            loads = self.network.loads_t.p[
                DE_loads.loc[DE_loads.carrier == i].index.to_list()
            ]
            data[i] = loads.sum(axis=1).resample(t_resolution).mean() / 1e3

        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        data.plot.area(
            ax=ax,
            title="Stacked H2 Loads by carrier",
            ylabel="[GW]",
            legend=True,
            stacked=True,
        )

        h2_CH4_gen = -self.network.links_t.p1[
            self.network.links.loc[
                self.network.links.carrier == "CH4_to_H2"
            ].index
        ]
        h2_power_gen = -self.network.links_t.p1[
            self.network.links.loc[
                self.network.links.carrier == "power_to_H2"
            ].index
        ]
        (
            h2_CH4_gen.sum(axis=1) / 1e3 + h2_power_gen.sum(axis=1) / 1e3
        ).resample(t_resolution).mean().plot(
            ax=ax,
            legend=True,
            label="H2 Generation",
            color="black",
            linestyle="dashed",
        )

    else:
        data = data.sum(axis=1).resample(t_resolution).mean() / 1e3
        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        data.plot(
            ax=ax,
            title="H2 Loads by carrier",
            label=rel_h2_links[0],
            ylabel="[GW]",
            legend=True,
        )

        for i in rel_h2_links[1:]:
            data = self.network.links_t.p0[
                self.network.links.loc[
                    self.network.links.carrier == i
                ].index.to_list()
            ]
            data = data.sum(axis=1).resample(t_resolution).mean() / 1e3
            data.plot(ax=ax, label=i, legend=True)

        DE_loads = self.network.loads.loc[
            self.network.loads.bus.isin(
                self.network.buses.loc[
                    self.network.buses.country == "DE"
                ].index
            )
        ]
        data = self.network.loads_t.p[
            DE_loads.loc[DE_loads.carrier == rel_h2_loads[0]].index.to_list()
        ]
        data = data.sum(axis=1).resample(t_resolution).mean() / 1e3
        data.plot(ax=ax, label=rel_h2_loads[0], ylabel="[GW]", legend=True)

    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_heat_loads(self, t_resolution="20H", save_path=False):
    """
    Plots timeseries data for heat loads

    Parameters
    ----------
    self : :class:`Etrago
        Overall container of Etrago
    t_resolution : str, optional
        sets the resampling rate of timeseries data to allow for smoother
        line plots
    save_path : bool, optional
        Path to save the generated plot. The default is False.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    central_h = self.network.loads.loc[
        self.network.loads.carrier == "central_heat"
    ]
    rural_h = self.network.loads.loc[
        self.network.loads.carrier == "rural_heat"
    ]
    central_h_loads = self.network.loads_t.p[central_h.index].sum(axis=1)
    rural_h_loads = self.network.loads_t.p[rural_h.index].sum(axis=1)

    ((central_h_loads + rural_h_loads) / 1e3).resample(
        t_resolution
    ).mean().plot(
        ax=ax,
        title="Central and rural heat loads",
        label="central_heat + rural_heat",
        legend=True,
        ylabel="[GW]",
    )
    (central_h_loads / 1e3).resample(t_resolution).mean().plot(
        ax=ax, label="central_heat", legend=True
    )
    (rural_h_loads / 1e3).resample(t_resolution).mean().plot(
        ax=ax, label="rural_heat", legend=True
    )

    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_heat_summary(self, t_resolution="20H", stacked=True, save_path=False):
    """
    Plots timeseries data for heat generation (and demand)

    Parameters
    ----------
    self : :class:`Etrago
        Overall container of Etrago
    t_resolution : str, optional
        sets the resampling rate of timeseries data to allow for smoother
        line plots
    stacked : bool, optional
        If True all TS data will be shown as stacked area plot. Total heat
        demand will then also be plotted to check for matching generation and
        demand.
    save_path : bool, optional
        Path to save the generated plot. The default is False.

    Returns
    -------
    None.

    """

    heat_gen_techs = [
        "central_resistive_heater",
        "central_heat_pump",
        "rural_heat_pump",
        "central_gas_CHP_heat",
        "central_gas_boiler",
        "rural_gas_boiler",
    ]

    heat_gen_ids = self.network.generators.loc[
        self.network.generators.carrier.isin(
            [
                "solar_thermal_collector",
                "geo_thermal",
                "central_biomass_CHP_heat",
            ]
        )
    ].index
    heat_gen_dispatch = (
        self.network.generators_t.p.T.loc[heat_gen_ids].sum(axis=0) / 1e3
    )

    links_id_hc = self.network.links.loc[
        self.network.links.carrier.isin(
            ["central_heat_store_charger", "rural_heat_store_charger"]
        )
    ].index
    heat_store_charger_dispatch = (
        self.network.links_t.p0.T.loc[links_id_hc].sum(axis=0) / 1e3
    )

    links_id_hdc = self.network.links.loc[
        self.network.links.carrier.isin(
            ["central_heat_store_discharger", "rural_heat_store_discharger"]
        )
    ].index
    heat_store_discharger_dispatch = (
        self.network.links_t.p1.T.loc[links_id_hdc].sum(axis=0) / 1e3
    )

    heat_store_dispatch_hb = (
        -heat_store_discharger_dispatch - heat_store_charger_dispatch
    )

    central_h = self.network.loads.loc[
        self.network.loads.carrier == "central_heat"
    ]
    rural_h = self.network.loads.loc[
        self.network.loads.carrier == "rural_heat"
    ]
    central_h_loads = self.network.loads_t.p[central_h.index].sum(axis=1) / 1e3
    rural_h_loads = self.network.loads_t.p[rural_h.index].sum(axis=1) / 1e3

    data = (
        self.network.links_t.p1[
            self.network.links.loc[
                self.network.links.carrier == heat_gen_techs[0]
            ].index.to_list()
        ]
        / 1e3
    )

    if stacked:
        data = pd.DataFrame(-(data.sum(axis=1)))
        data = data.rename(columns={0: heat_gen_techs[0]})

        for i in heat_gen_techs[1:]:
            loads = self.network.links_t.p1[
                self.network.links.loc[
                    self.network.links.carrier == i
                ].index.to_list()
            ]
            data[i] = -(loads).sum(axis=1) / 1e3

        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        data.resample(t_resolution).mean().plot.area(
            ax=ax,
            title="Stacked heat generation and demand",
            ylabel="[GW]",
            legend=True,
            stacked=True,
        )

        (
            data.sum(axis=1) + heat_store_dispatch_hb + heat_gen_dispatch
        ).resample(t_resolution).mean().plot.line(
            ax=ax,
            legend=True,
            label="Total heat generation + heat store dispatch",
            color="yellow",
        )

    else:
        data = -data.sum(axis=1) / 1e3

        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

        data.resample(t_resolution).mean().plot(
            ax=ax,
            title="Heat generation and demand",
            label=heat_gen_techs[0],
            ylabel="[GW]",
            legend=True,
        )

        for i in heat_gen_techs[1:]:
            data = self.network.links_t.p1[
                self.network.links.loc[
                    self.network.links.carrier == i
                ].index.to_list()
            ]
            data = -data.sum(axis=1) / 1e3
            data.resample(t_resolution).mean().plot(
                ax=ax, label=i, legend=True
            )

        heat_store_dispatch_hb.resample(t_resolution).mean().plot.line(
            ax=ax,
            legend=True,
            label="Heat store dispatch",
            color="yellow",
            linestyle="dashed",
        )

    (central_h_loads + rural_h_loads).resample(t_resolution).mean().plot.line(
        ax=ax,
        legend=True,
        label="Total heat demand",
        color="black",
        linestyle="dashed",
    )

    if save_path:
        plt.savefig(save_path, dpi=300)


def shifted_energy(self, carrier, buses):
    """Calulate shifted energy for a specific carrier

    Parameters
    ----------
    carrier : str
        Name of energy carrier
    buses : list
        List of considered bus indices

    Returns
    -------
    shifted : pandas.Series
        Shifted energy per time step

    """

    buses = self.network.links[
        self.network.links.bus0.isin(
            self.network.buses[
                (self.network.buses.carrier == "AC")
                & (self.network.buses.index.isin(buses))
            ].index
        )
        & self.network.links.bus1.isin(
            self.network.buses[
                self.network.buses.carrier.str.contains(carrier)
            ].index
        )
    ].bus1.unique()

    supply = self.network.links_t.p1[
        self.network.links[
            (self.network.links.bus1.isin(buses))
            & ~(self.network.links.carrier.str.contains("charger"))
        ].index
    ].mul(-1).sum(axis=1) + (
        self.network.generators_t.p[
            self.network.generators[
                self.network.generators.bus.isin(buses)
            ].index
        ].sum(axis=1)
    )

    demand = self.network.loads_t.p[
        self.network.loads[self.network.loads.bus.isin(buses)].index
    ].sum(axis=1) + (
        self.network.links_t.p0[
            self.network.links[
                (self.network.links.bus0.isin(buses))
                & ~(self.network.links.carrier.str.contains("charger"))
            ].index
        ].sum(axis=1)
    )

    shifted = supply - demand
    return shifted


def flexibility_duration_curve(etrago, etrago_lowflex, filename=None):
    """Plot duration curves of flexibility options

    Parameters
    ----------
    etrago : Etrago
        Object including network with flexibility options
    etrago_lowflex : Etrago
        Object including network with less flexibility options
    filename : str, optional
        Name of file to save plot. The default is None.

    Returns
    -------
    None.

    """
    colors = coloring()

    value = "p"

    df = pd.DataFrame()

    dsm_stores = etrago.network.stores[
        etrago.network.stores.carrier.str.contains("dsm")
    ]
    df["dsm_positive"] = (
        etrago.network.stores_t[value][dsm_stores.index]
        .clip(lower=0)
        .sum(axis=1)
    )
    df["dsm_negative"] = (
        etrago.network.stores_t[value][dsm_stores.index]
        .clip(upper=0)
        .sum(axis=1)
    )

    emob_static = etrago_lowflex.network.loads[
        etrago_lowflex.network.loads.carrier == "land transport EV"
    ]

    emob_static_t = etrago_lowflex.network.loads_t.p_set[emob_static.index]

    emob_static_t = emob_static_t.loc[:, emob_static.index]

    emob_static_t.columns = emob_static.bus.values

    emob_flex = etrago.network.links[
        etrago.network.links.carrier.str.contains("BEV")
    ]

    emob_flex_t = etrago.network.links_t.p0[emob_flex.index]

    emob_flex_t = emob_flex_t.loc[:, emob_flex.index]

    emob_flex_t.columns = emob_flex.bus0.values

    emob_flex_t - emob_static_t
    df["BEV charger_positive"] = (
        (emob_flex_t - emob_static_t).clip(lower=0).sum(axis=1)
    )
    df["BEV charger_negative"] = (
        (emob_flex_t - emob_static_t).clip(upper=0).sum(axis=1)
    )

    heat_stores = etrago.network.stores[
        etrago.network.stores.carrier.str.contains("heat")
    ]
    df["heat_positive"] = (
        etrago.network.stores_t[value][heat_stores.index]
        .clip(lower=0)
        .sum(axis=1)
    )
    df["heat_negative"] = (
        etrago.network.stores_t[value][heat_stores.index]
        .clip(upper=0)
        .sum(axis=1)
    )

    h2_stores = etrago.network.stores[
        etrago.network.stores.carrier.str.contains("H2")
    ]
    df["H2_positive"] = (
        etrago.network.stores_t[value][h2_stores.index]
        .clip(lower=0)
        .sum(axis=1)
    )
    df["H2_negative"] = (
        etrago.network.stores_t[value][h2_stores.index]
        .clip(upper=0)
        .sum(axis=1)
    )

    fig, ax = plt.subplots(figsize=(15, 8))
    for c in df.columns:
        result = pd.Series(dtype=float)
        color = colors[c.split("_")[0]]
        for p in range(0, 100):
            result[p * df[c].abs().max() * np.sign(df[c].sum()) / 100] = (
                df[c][df[c].abs() > p * 0.01 * df[c].abs().max()].size
                / df[c].size
            ) * 100

        data_to_plot = pd.DataFrame(
            index=result.values, data=result.index * 1e-3
        )
        data_to_plot.columns = [c.split("_")[0]]
        data_to_plot.plot(ax=ax, color=color, linewidth=3.0)
    plt.axhline(y=0.0, color="grey", linestyle="dotted")
    ax.set_xlim(0, 80)
    ax.set_xlabel("time in %")
    ax.set_ylabel("flexibility usage in GW")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if filename:
        fig.savefig(filename, dpi=600)
        plt.close()
