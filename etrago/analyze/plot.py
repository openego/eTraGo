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
from math import sqrt
import logging
import os

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse
from pypsa.plot import draw_map_cartopy
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

cartopy_present = True
try:
    import cartopy.crs as ccrs
except ImportError:
    cartopy_present = False


logger = logging.getLogger(__name__)

if "READTHEDOCS" not in os.environ:
    from geoalchemy2.shape import to_shape  # noqa: F401
    from pyproj import Proj, transform
    from shapely.geometry import LineString, Point
    import geopandas as gpd
    import tilemapbase

    from etrago.execute import import_gen_from_links

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = """ulfmueller, MarlonSchlemminger, mariusves, lukasol,
ClaraBuettner, CarlosEpia, pieterhexen, gnn, fwitte, lukasol, KathiEsterl,
BartelsJ"""


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


def curtailment(network, carrier="wind_onshore", filename=None):
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
    
    network = network.copy()
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
       # network.generators_t.p = network.generators_t.p * -1

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
        index=pd.MultiIndex.from_tuples(
            index, names=["bus", "carrier"]
        ).unique(),
        dtype=float,
    ).sort_index()

    for i in dist.index:
        gens = network.generators[
            (network.generators.bus == i[0])
            & (network.generators.carrier == i[1])
        ].index
        dist.loc[i] = (
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

        dist.loc[dist.index.get_level_values("carrier") == "battery"] = (
            battery_distribution
        )
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

        dist.loc[dist.index.get_level_values("carrier") == "H2_overground"] = (
            h2_over_distribution
        )

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
    dispatch_grouped = dispatch.groupby(level=0).sum()
    if dispatch_grouped.empty or max(abs(dispatch_grouped)) == 0:
        print("⚠️ Keine dispatch-Differenz vorhanden – Plot wird übersprungen.")
        return
    scaling = 1 / max(abs(dispatch_grouped)) * scaling

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
        line_widths=0.01,
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


def demand_side_management(
    self, buses, snapshots, agg="5h", used=False, apply_on="grid_model"
):
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
            'pre_market_model', 'disaggregated_network'."""
        )

    df = pd.DataFrame(index=network.snapshots[snapshots])

    link = network.links[
        (network.links.carrier == "dsm") & (network.links.bus0.isin(buses))
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
            'pre_market_model', 'disaggregated_network'."""
        )

    # Initialize DataFrame
    df = pd.DataFrame(index=network.snapshots[snapshots])

    # Select BEV buses and links
    bev_buses = network.buses[network.buses.carrier.str.contains("Li ion")]
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
        bev_links_t_used = network.links_t.p0[bev_links.index].iloc[snapshots]

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
            'pre_market_model', 'disaggregated_network'."""
        )

    df = pd.DataFrame(index=network.snapshots[snapshots])

    heat_buses = network.links[
        network.links.bus0.isin(
            network.buses[
                (network.buses.carrier == "AC")
                & (network.buses.index.isin(buses))
            ].index
        )
        & network.links.bus1.isin(
            network.buses[network.buses.carrier.str.contains("heat")].index
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
            'pre_market_model', 'disaggregated_network'."""
        )

    df = pd.DataFrame(index=network.snapshots[snapshots])

    h2_buses = network.links[
        network.links.bus0.isin(
            network.buses[
                (network.buses.carrier == "AC")
                & (network.buses.index.isin(buses))
            ].index
        )
        & network.links.bus1.isin(
            network.buses[network.buses.carrier.str.contains("H2")].index
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
    self,
    flexibility,
    agg="5h",
    snapshots=[],
    buses=[],
    pre_path=None,
    apply_on="grid_model",
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
            'pre_market_model', 'disaggregated_network'."""
        )

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
            network.storage_units_t.p[su.index].sum(axis=1).iloc[snapshots]
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


def plot_carrier(
    etrago, carrier_links=["AC"], carrier_buses=["AC"], apply_on="grid_model"
):
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
            'pre_market_model', 'disaggregated_network'."""
        )

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
    geographical_boundaries=[1.5, 19, 46, 61.5],
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

        * 'nodal_production_balance': net producer/consumer in selected
           time steps
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
    elif apply_on == "self":
        network = self
    else:
        logger.warning(
            """Parameter apply_on must be one of ['grid_model', 'market_model'
            'pre_market_model', 'disaggregated_network'."""
        )

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
        # Set a maximum size for bus_sizes
        max_bus_size = 3  # Define your maximum size here
        bus_sizes = bus_sizes.clip(upper=max_bus_size)
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

        market_bus_de = self.market_model.buses[
            (self.market_model.buses.country == "DE")
            & (self.market_model.buses.carrier == "AC")
        ].index
        market_price = self.market_model.buses_t.marginal_price[market_bus_de]

        bus_colors = pd.Series(index=network.buses.index, data=0)
        for bus in network.links[
            (network.links.carrier == "power_to_H2")
        ].bus0:

            nodal_price = network.buses_t.marginal_price[bus]

            ely = network.links_t.p0[
                network.links[
                    (network.links.carrier == "power_to_H2")
                    & (network.links.bus0 == bus)
                ].index
            ]
            df_corr = pd.DataFrame()

            df_corr["ely"] = ely
            df_corr["market"] = market_price
            df_corr["nodal_price"] = nodal_price

            bus_colors[bus] = (
                df_corr.corr(method="spearman").loc["nodal_price", "ely"]
                / (df_corr.corr(method="spearman").loc["nodal_price", "ely"])
                + df_corr.corr(method="spearman").loc["market", "ely"]
            )

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
            line_cmap=plt.cm.viridis_r,
            bus_alpha=0.9,
            link_cmap=plt.cm.viridis_r,
            bus_sizes=bus_sizes,
            bus_colors=bus_colors,
            bus_cmap=plt.cm.viridis,
            line_widths=line_widths,
            link_widths=link_widths,
            flow=flow,
            # title=title,
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
                    scale_for_legend = max_value / (bus_scaling)  # 300 ist tuning-wert

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
            labels.append(f"{(max_value / bus_scaling )/ 1000000:.1f} TW ")
            
            handles.append(
                make_legend_circles_for(
                    [max_value],
                    scale=bus_scaling,
                    facecolor="grey",
                )[0]
            )

        l2 = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.8, 1.03),
            labelspacing=1.0,
            framealpha=1.0,
            title=bus_legend,
            handler_map=make_handler_map_to_scale_circles_as_in(ax),
            prop={"size": 18},
        )
        ax.add_artist(l2)
        print(len(bus_sizes), max_value, bus_scaling)
        plt.setp(l2.get_title(), fontsize="13")

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
                loc="upper right",
                ncol=2,
                bbox_to_anchor=(0, 0),
            )
            ax.add_artist(l3)

    if type(line_colors) is not str:
        # Set fixed boundaries if selected in parameters
        if not boundaries:
            boundaries = [
                min(round(line_colors.min(), 1), round(link_colors.min(), 1)),
                500,#max(round(line_colors.max()), round(link_colors.max())),
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
            # fraction=0.028,
            pad=0.04,
            orientation="horizontal",
        )
        # Set legend label
        cb.set_label(label, fontsize = 12)

    elif type(bus_colors) is not str:
        # import pdb; pdb.set_trace()
        ll[0].set_clim([0, bus_colors.max()])
        plt.colorbar(
            ll[0], fraction=0.04, pad=0.004, label="correlation factor", ax=ax
        )

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
            lambda x: (
                x["geom"]
                if not pd.isna(x["geom"])
                else LineString(
                    [
                        map_buses["geom"][x["bus0"]],
                        map_buses["geom"][x["bus1"]],
                    ]
                )
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

def total_dispatch(self):
    """
    Calculate and print the dispatch summary for the given timesteps.

    Parameters
    ----------
    etrago_nodal : Etrago object
        The Etrago model object.
    timesteps : range or list
        The timesteps to consider in the dispatch calculation.
    """
        
    # Berechnen Sie den Dispatch pro Carrier für die angegebenen Zeitschritte
    dispatch_per_carrier_net = calc_dispatch_per_carrier(german_network(self.network), timesteps = range(1752), dispatch_type = "total")
   
    dispatch_per_carrier_market = calc_dispatch_per_carrier(german_network(self.market_model), timesteps = range(0, 8760, 5), dispatch_type="total")
    
    # Gruppieren Sie die Ergebnisse nach 'carrier' und summieren Sie die Werte
    sum_per_carrier_net = ((dispatch_per_carrier_net.groupby('carrier').sum())*5)/1e6
    sum_per_carrier_market = ((dispatch_per_carrier_market.groupby('carrier').sum())*5)/1e6
    
    # Definieren Sie die erneuerbaren Carrier einschließlich Biomasse
    renewable_carriers = [
        'solar', 'solar_rooftop', 'wind_offshore', 'wind_onshore',
        'reservoir', 'run_of_river', 'biomass', 'central_biomass_CHP', 'industrial_biomass_CHP'
    ]

    # Filtern Sie die erneuerbaren Carrier und summieren Sie deren Werte
    sum_total_twh_net = sum_per_carrier_net.sum()
    sum_total_twh_market = sum_per_carrier_market.sum()
    
    sum_renewables_twh_net = sum_per_carrier_net[sum_per_carrier_net.index.isin(renewable_carriers)].sum()
    sum_renewables_twh_market = sum_per_carrier_market[sum_per_carrier_market.index.isin(renewable_carriers)].sum()
    # Rechnen Sie die Summe von MWh in TWh um
   
    
    
    #Ausgabe der Ergebnisse
        
    print("\nStromproduktion nach Marktoptimierung gesamt:")
    print(f"{sum_total_twh_market} TWh")
    print("\nDavon Erneuerbar:")
    print(f"{sum_renewables_twh_market} TWh")
    
    print("\nStromproduktion nach Netzoptimierung gesamt:")
    print(f"{sum_total_twh_net} TWh")
    print("\nDavon Erneuerbar:")
    print(f"{sum_renewables_twh_net} TWh")
    
    return sum_per_carrier_market, sum_per_carrier_net

def german_network(self):
    """Cut out all network components in Germany

    Returns
    -------
    new_network : pypsa.Network
        Network with all components in Germany

    """
   
    keep_cntr = ["DE", "LU"]
    new_idx = self.buses[
        self.buses.country.isin(keep_cntr)
    ].index

    new_network = self.copy()

    # drop components of other countries
    new_network.mremove(
        "Bus", new_network.buses[~new_network.buses.index.isin(new_idx)].index
    )

    new_network.mremove(
        "Line",
        new_network.lines[
            ~new_network.lines.index.isin(
                new_network.lines[
                    (
                        new_network.lines.bus0.isin(new_idx)
                        & new_network.lines.bus1.isin(new_idx)
                    )
                ].index
            )
        ].index,
    )
    new_network.mremove(
        "Link",
        new_network.links[
            ~new_network.links.index.isin(
                new_network.links[
                    (
                        new_network.links.bus0.isin(new_idx)
                        & new_network.links.bus1.isin(new_idx)
                    )
                ].index
            )
        ].index,
    )

    new_network.mremove(
        "Transformer",
        new_network.transformers[
            ~new_network.transformers.index.isin(
                new_network.transformers[
                    (
                        new_network.transformers.bus0.isin(new_idx)
                        & new_network.transformers.bus1.isin(new_idx)
                    )
                ].index
            )
        ].index,
    )

    new_network.mremove(
        "Generator",
        new_network.generators[
            ~new_network.generators.index.isin(
                new_network.generators[
                    new_network.generators.bus.isin(new_idx)
                ].index
            )
        ].index,
    )

    new_network.mremove(
        "Load",
        new_network.loads[
            ~new_network.loads.index.isin(
                new_network.loads[new_network.loads.bus.isin(new_idx)].index
            )
        ].index,
    )

    new_network.mremove(
        "Store",
        new_network.stores[
            ~new_network.stores.index.isin(
                new_network.stores[new_network.stores.bus.isin(new_idx)].index
            )
        ].index,
    )

    new_network.mremove(
        "StorageUnit",
        new_network.storage_units[
            ~new_network.storage_units.index.isin(
                new_network.storage_units[
                    new_network.storage_units.bus.isin(new_idx)
                ].index
            )
        ].index,
    )

    return new_network

def total_load(self):
    network = german_network(self.network)
    market_model = german_network(self.market_model)
    
    buses_net = network.buses
    buses_market = market_model.buses
    loads_net = network.loads_t.p
    loads_market = market_model.loads_t.p
    links_net = network.links
    links_market = market_model.links
    link_loads_net = network.links_t.p0
    link_loads_market = market_model.links_t.p0
    
    
    ac_buses_net = buses_net[buses_net['carrier'] == 'AC'].index.astype(str)
    ac_buses_market = buses_market[buses_market['carrier'] == 'AC'].index.astype(str)
    ac_columns_net = [col for col in loads_net.columns if col.split()[0] in ac_buses_net and col.split()[1] == 'AC']
    ac_columns_market = [col for col in loads_market.columns if col.split()[0] in ac_buses_market and col.split()[1] == 'AC']
    ac_loads_net= loads_net[ac_columns_net]
    ac_loads_market= loads_market[ac_columns_market]
    
    
    links_to_ac_buses_net = links_net[links_net['bus0'].isin(ac_buses_net)]
    links_to_ac_buses_market = links_market[links_market['bus0'].isin(ac_buses_market)] # DC buses checken
    links_to_ac_buses_market = links_to_ac_buses_market[links_to_ac_buses_market.carrier != "DC"]
    link_ac_loads_net = link_loads_net[link_loads_net.columns.intersection(links_to_ac_buses_net.index)]
    link_ac_loads_market = link_loads_market[link_loads_market.columns.intersection(links_to_ac_buses_market.index)]
    
    ac_loads_market_5h = ac_loads_market.iloc[::5]
    link_ac_loads_market_5h = link_ac_loads_market.iloc[::5]

    total_load_net = ac_loads_net.sum().sum()*5*1e-6 + link_ac_loads_net.sum().sum()*5*1e-6
    total_load_market =  ac_loads_market_5h.sum().sum()*5*1e-6 + link_ac_loads_market_5h.sum().sum()*5*1e-6
    
    print(f"Netz {total_load_net} TWh\nMarkt {total_load_market} TWh")
    print(f"Statische Lasten Netz {ac_loads_net.sum().sum()*5*1e-6} TWh\n Flexible Lasten Netz {link_ac_loads_net.sum().sum()*5*1e-6} TWh")
    return  

def assign_market_zones_to_buses(network, market_zones):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    if market_zones == "DE2":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp"
    elif market_zones == "DE3":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_12_DE3.shp"
    elif market_zones == "DE4":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp"
    elif market_zones == "DE5":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp"
    elif market_zones == "none":
        shapefile_path = None
    else:
        raise ValueError("Ungültiger Wert für market_zone. Erlaubt sind: 'DE2', 'DE3', 'DE4', 'DE5', oder 'none'.")


    if shapefile_path is None:
        raise ValueError("Ungültiger Wert für market_zones")

    zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # Buserstellen mit Geometrie
    buses = network.buses.copy()
    bus_points = gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses.x, buses.y),
        crs="EPSG:4326"
    )

    # Spatial Join: Busse mit Zonen verbinden
    joined = gpd.sjoin(bus_points, zones, how="left", predicate="within")

    # 'id' ist hier die Zone
    network.buses["zone"] = joined["id"]

    # Diagnose
    missing = network.buses["zone"].isna().sum()
    print(f"{missing} Bussen konnte keine Zone zugewiesen werden.")

    return network


def total_load_by_zone(self, market_zones="DE4", plot=True):
    """
    Berechnet und plottet die Gesamtlast je Zone (Netz- und Marktmodell).

    Parameter
    ---------
    etrago_obj : object
        Das eTraGo-Objekt mit .network und .market_model
    market_zones : str
        Marktzonenkonfiguration ('DE2', 'DE3', 'DE4', 'DE5')
    plot : bool
        Wenn True, wird eine Karte mit Lasten je Zone angezeigt
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs

    assign_market_zones_to_buses(self.market_model, market_zones)
    #network = german_network(etrago_obj.network)
    market_model = german_network(self.market_model)

    #buses_net = network.buses
    buses_market = market_model.buses
    #loads_net = network.loads_t.p
    loads_market = market_model.loads_t.p
    #links_net = network.links
    links_market = market_model.links
    #link_loads_net = network.links_t.p0
    link_loads_market = market_model.links_t.p0

    #if 'zone' not in buses_net.columns or 'zone' not in buses_market.columns:
    if 'zone' not in buses_market.columns:
        raise ValueError("Spalte 'zone' fehlt in den buses. Bitte zuerst Zonen zuweisen.")

    zones_list = sorted(buses_market['zone'].dropna().unique())
    results = {}

    for zone in zones_list:
        # --- Netz ---
        #ac_buses_net = buses_net[(buses_net['carrier'] == 'AC') & (buses_net['zone'] == zone)].index.astype(str)
        #ac_columns_net = [col for col in loads_net.columns if col.split()[0] in ac_buses_net and col.split()[1] == 'AC']
        #ac_loads_net = loads_net[ac_columns_net]
        #links_to_ac_net = links_net[links_net['bus0'].isin(ac_buses_net)]
        #links_to_ac_net = links_to_ac_net[links_to_ac_net.carrier != "DC"]
        #link_ac_loads_net = link_loads_net[link_loads_net.columns.intersection(links_to_ac_net.index)]
        #total_net = ac_loads_net.sum().sum() * 5 * 1e-6 + link_ac_loads_net.sum().sum() * 5 * 1e-6

        # --- Markt ---
        ac_buses_market = buses_market[(buses_market['carrier'] == 'AC') & (buses_market['zone'] == zone)].index.astype(str)
        ac_columns_market = [col for col in loads_market.columns if col.split()[0] in ac_buses_market and col.split()[1] == 'AC']
        ac_loads_market = loads_market[ac_columns_market].iloc[::5]
        links_to_ac_market = links_market[links_market['bus0'].isin(ac_buses_market)]
        links_to_ac_market = links_to_ac_market[links_to_ac_market.carrier != "DC"]
        link_ac_loads_market = link_loads_market[link_loads_market.columns.intersection(links_to_ac_market.index)].iloc[::5]
        total_market = ac_loads_market.sum().sum() * 5 * 1e-6 + link_ac_loads_market.sum().sum() * 5 * 1e-6

        results[zone] = {'Netz': total_net, 'Markt': total_market}

    # --- Druck in Konsole ---
    print(f"\n--- Gesamtlast je Zone ({market_zones}) ---")
    
    for zone, data in results.items():
        print(f"Zone {zone}: Netzmodell = {data['Netz']:.2f} TWh, Marktmodell = {data['Markt']:.2f} TWh")

        # --- Plot (einfach wie in total_dispatch_by_zone) ---
    
    # Shape laden
    if market_zones == "DE2":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp"
    elif market_zones == "DE3":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_12_DE3.shp"
    elif market_zones == "DE4":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp"
    elif market_zones == "DE5":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp"
    else:
        raise ValueError("Ungültiger Wert für market_zones.")

    zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # Farben zuweisen wie in total_dispatch_by_zone
    colors = plt.cm.tab20(range(len(zones)))
    zones["color"] = [mcolors.rgb2hex(color[:3]) for color in colors]

    # Plot vorbereiten
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

    # Zonen einfärben
    zones.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)
    zones.plot(ax=ax, facecolor=zones["color"], alpha=0.3)
    
    # Netz zeichnen wie in plot_marketzone_clustering
    etrago_obj.market_model.plot(
        ax=ax,
        link_widths=0,
        bus_sizes=0
    )

    # Zahlen plotten (Markt oben, Netz unten)
    for _, row in zones.iterrows():
        zone = row["id"]
        if zone not in results:
            continue
        centroid = row.geometry.centroid
        markt = results[zone]["Markt"]
        netz = results[zone]["Netz"]

        ax.text(centroid.x, centroid.y + 0.2,
                f"{markt:.1f} TWh", color="blue",
                fontsize=20, ha="center", weight="bold")
        ax.text(centroid.x, centroid.y - 0.2,
                f"{netz:.1f} TWh", color="saddlebrown",
                fontsize=20, ha="center", weight="bold")

    # Legende
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="blue", lw=4, label="Marktmodell"),
        Line2D([0], [0], color="saddlebrown", lw=4, label="Netzmodell"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=20)

    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
    ax.axis("off")
    plt.tight_layout()
    plt.show()


    return results




def total_load_and_renewables_by_zone(self, market_zones="DE4", plot=True):
    """
    Berechnet und plottet die Gesamtlast und installierte Leistung erneuerbarer Erzeuger je Zone oder für ganz Deutschland.

    Parameter
    ---------
    market_zones : str
        Marktzonenkonfiguration ('DE2', 'DE3', 'DE4', 'DE5', 'none')
        Bei 'none' wird die Berechnung für ganz Deutschland ohne Zonenaufteilung durchgeführt
    plot : bool
        Wenn True, wird eine Karte mit Lasten und erneuerbaren Kapazitäten angezeigt
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import pandas as pd
    from cartopy.feature import NaturalEarthFeature

    

    # Define renewable carriers
    renewables = [
        'solar', 'solar_rooftop', 'wind_offshore', 'wind_onshore',
        'reservoir', 'run_of_river', 'biomass', 'central_biomass_CHP',
        'industrial_biomass_CHP'
    ]

    if market_zones != "none":
        # Assign market zones to buses
        assign_market_zones_to_buses(self.market_model, market_zones)
        
        # Initialize market model
        market_model = german_network(self.market_model)
        
        buses_market = market_model.buses
        loads_market = market_model.loads_t.p
        links_market = market_model.links
        link_loads_market = market_model.links_t.p0

        # Check if 'zone' column exists
        if 'zone' not in buses_market.columns:
            raise ValueError("Spalte 'zone' fehlt in den buses. Bitte zuerst Zonen zuweisen.")

        # Calculate total load and renewable capacity per zone
        zones_list = sorted(buses_market['zone'].dropna().unique())
        results = {}

        for zone in zones_list:
            # Calculate total load for the zone
            ac_buses_market = buses_market[(buses_market['carrier'] == 'AC') & (buses_market['zone'] == zone)].index.astype(str)
            ac_columns_market = [col for col in loads_market.columns if col.split()[0] in ac_buses_market and col.split()[1] == 'AC']
            ac_loads_market = loads_market[ac_columns_market].iloc[::5]
            total_load = ac_loads_market.sum().sum() * 5 * 1e-6  # Convert to TWh

            # Berechnung der Link-Lasten für die aktuelle Zone
            links_to_ac_market = links_market[links_market['bus0'].isin(ac_buses_market)]
            links_to_ac_market = links_to_ac_market[links_to_ac_market.carrier != "DC"]
            link_ac_loads_market = link_loads_market[link_loads_market.columns.intersection(links_to_ac_market.index)].iloc[::5]
            link_load = link_ac_loads_market.sum().sum() * 5 * 1e-6  # Convert to TWh
            
            # Addiere die Link-Lasten zur Gesamtlast der Zone
            total_load += link_load

            # Calculate total renewable capacity for the zone
            generators_in_zone = market_model.generators[market_model.generators.bus.isin(ac_buses_market)]
            renewable_capacity = generators_in_zone[generators_in_zone.carrier.isin(renewables)]['p_nom'].sum() * 1e-3  # Convert to GW

            results[zone] = {
                'Total Load': total_load,
                'Renewable Capacity': renewable_capacity
            }
            
            # Berechne Gesamtsummen für Deutschland (für Prozentanteile)
        total_load_germany = sum([data['Total Load'] for data in results.values()])
        total_renewables_germany = sum([data['Renewable Capacity'] for data in results.values()])
        
        print(f"\n--- Prozentuale Anteile je Zone ({market_zones}) ---")
        for zone, data in results.items():
            load_pct = 100 * data['Total Load'] / total_load_germany
            renew_pct = 100 * data['Renewable Capacity'] / total_renewables_germany
            print(f"Zone {zone}: Last = {load_pct:.1f} %, EE-Kapazität = {renew_pct:.1f} %")

        # Print results to console
        #print(f"\n--- Gesamtlast und erneuerbare Kapazität je Zone ({market_zones}) ---")
        #for zone, data in results.items():
        #    print(f"Zone {zone}: Gesamtlast = {data['Total Load']:.2f} TWh, Erneuerbare Kapazität = {data['Renewable Capacity']:.2f} GW")

    else:
        # Calculate for all of Germany without zone assignment
        # Total load calculation
        
        # Initialize market model
        market_model = german_network(self.market_model)
        
        buses_market = market_model.buses
        loads_market = market_model.loads_t.p
        links_market = market_model.links
        link_loads_market = market_model.links_t.p0
        
        ac_buses_market = buses_market[buses_market['carrier'] == 'AC'].index.astype(str)
        ac_columns_market = [col for col in loads_market.columns if col.split()[0] in ac_buses_market and col.split()[1] == 'AC']
        ac_loads_market = loads_market[ac_columns_market].iloc[::5]
        total_load = ac_loads_market.sum().sum() * 5 * 1e-6  # Convert to TWh
        
        # Berechnung der Link-Lasten für ganz Deutschland
        links_market = market_model.links
        link_loads_market = market_model.links_t.p0
        
        # Filtere nur AC-Links (keine DC-Links)
        ac_links_market = links_market[links_market.carrier != "DC"]
        
        # Summiere die Link-Lasten
        link_ac_loads_market = link_loads_market[link_loads_market.columns.intersection(ac_links_market.index)].iloc[::5]
        total_link_load = link_ac_loads_market.sum().sum() * 5 * 1e-6  # Convert to TWh
        
        # Addiere die Link-Lasten zur Gesamtlast
        total_load += total_link_load

        # Total renewable capacity calculation
        generators_in_germany = market_model.generators
        renewable_capacity = generators_in_germany[generators_in_germany.carrier.isin(renewables)]['p_nom'].sum() * 1e-3  # Convert to GW

        results = {
            'Germany': {
                'Total Load': total_load,
                'Renewable Capacity': renewable_capacity
            }
        }
        

        # Print results to console
        print("\n--- Gesamtlast und erneuerbare Kapazität für ganz Deutschland ---")
        print(f"Gesamtlast = {total_load:.2f} TWh, Erneuerbare Kapazität = {renewable_capacity:.2f} GW")

    # Plotting
    if plot:
        if market_zones != "none":
            # Load shapefile based on market zones configuration
            if market_zones == "DE2":
                shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp"
            elif market_zones == "DE3":
                shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_12_DE3.shp"
            elif market_zones == "DE4":
                shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp"
            elif market_zones == "DE5":
                shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp"
            else:
                raise ValueError("Ungültiger Wert für market_zones.")

            zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

            # Assign colors
            colors = plt.cm.tab20(range(len(zones)))
            zones["color"] = [mcolors.rgb2hex(color[:3]) for color in colors]

            # Prepare plot
            fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

            # Plot zones
            zones.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)
            zones.plot(ax=ax, facecolor=zones["color"], alpha=0.3)

            # Plot network
            self.market_model.plot(
                ax=ax,
                link_widths=0,
                bus_sizes=0
            )

            # Plot numbers (Load and Renewable Capacity)
            for _, row in zones.iterrows():
                zone = row["id"]
                if zone not in results:
                    continue
                centroid = row.geometry.centroid
                load_pct = 100 * results[zone]["Total Load"] / total_load_germany
                renew_pct = 100 * results[zone]["Renewable Capacity"] / total_renewables_germany
            
                # Plot Load-Anteil - blue
                ax.text(centroid.x, centroid.y + 0.2,
                        f"{load_pct:.1f}%", color="blue",
                        fontsize=20, ha="center", weight="bold")
            
                # Plot EE-Anteil - green
                ax.text(centroid.x, centroid.y - 0.2,
                        f"{renew_pct:.1f}%", color="green",
                        fontsize=20, ha="center", weight="bold")

                

        else:
            # Create a simple map of Germany with values plotted in the center
            fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

          
            

            # Plot network (optional)
            self.market_model.plot(
                ax=ax,
                link_widths=0,
                bus_sizes=0
            )
            # Set extent to Germany
            ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
            
            # Plot values in the center of Germany (approximately)
            center_x, center_y = 10.5, 50  # Rough center of Germany

            # Plot load (TWh) - blue
            ax.text(center_x, center_y + 1,
                    f"{results['Germany']['Total Load']:.1f} TWh",
                    color="blue", fontsize=20, ha="center", weight="bold")

            # Plot renewable capacity (GW) - green
            ax.text(center_x, center_y,
                    f"{results['Germany']['Renewable Capacity']:.1f} GW",
                    color="green", fontsize=20, ha="center", weight="bold")

        # Add legend for both cases
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="Jährlicher Strombedarf"),
            Line2D([0], [0], color="green", lw=2, label="installierte Leistung EE"),
        ]
        #ax.legend(handles=legend_elements, loc="upper left", fontsize=20)
        ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    return results

def total_load_wind_and_solar_by_zone(self, market_zones="DE4", plot=True):
    """
    Berechnet und plottet die Gesamtlast und installierte Leistung erneuerbarer Erzeuger je Zone oder für ganz Deutschland.

    Parameter
    ---------
    market_zones : str
        Marktzonenkonfiguration ('DE2', 'DE3', 'DE4', 'DE5', 'none')
        Bei 'none' wird die Berechnung für ganz Deutschland ohne Zonenaufteilung durchgeführt
    plot : bool
        Wenn True, wird eine Karte mit Lasten und erneuerbaren Kapazitäten angezeigt
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import pandas as pd
    from cartopy.feature import NaturalEarthFeature

    

    # Define renewable carriers
    solar = [
        'solar', 'solar_rooftop'
    ]
    
    wind = [
        'wind_onshore', 'wind_offshore'
    ]

    if market_zones != "none":
        # Assign market zones to buses
        assign_market_zones_to_buses(self.market_model, market_zones)
        
        # Initialize market model
        market_model = german_network(self.market_model)
        
        buses_market = market_model.buses
        loads_market = market_model.loads_t.p
        links_market = market_model.links
        link_loads_market = market_model.links_t.p0

        # Check if 'zone' column exists
        if 'zone' not in buses_market.columns:
            raise ValueError("Spalte 'zone' fehlt in den buses. Bitte zuerst Zonen zuweisen.")

        # Calculate total load and renewable capacity per zone
        zones_list = sorted(buses_market['zone'].dropna().unique())
        results = {}

        for zone in zones_list:
            # Calculate total load for the zone
            ac_buses_market = buses_market[(buses_market['carrier'] == 'AC') & (buses_market['zone'] == zone)].index.astype(str)
            ac_columns_market = [col for col in loads_market.columns if col.split()[0] in ac_buses_market and col.split()[1] == 'AC']
            ac_loads_market = loads_market[ac_columns_market].iloc[::5]
            total_load = ac_loads_market.sum().sum() * 5 * 1e-6  # Convert to TWh

            # Berechnung der Link-Lasten für die aktuelle Zone
            links_to_ac_market = links_market[links_market['bus0'].isin(ac_buses_market)]
            links_to_ac_market = links_to_ac_market[links_to_ac_market.carrier != "DC"]
            link_ac_loads_market = link_loads_market[link_loads_market.columns.intersection(links_to_ac_market.index)].iloc[::5]
            link_load = link_ac_loads_market.sum().sum() * 5 * 1e-6  # Convert to TWh
            
            # Addiere die Link-Lasten zur Gesamtlast der Zone
            total_load += link_load

            # Calculate total renewable capacity for the zone
            generators_in_zone = market_model.generators[market_model.generators.bus.isin(ac_buses_market)]
            solar_capacity = generators_in_zone[generators_in_zone.carrier.isin(solar)]['p_nom'].sum() * 1e-3  # Convert to GW
            wind_capacity = generators_in_zone[generators_in_zone.carrier.isin(wind)]['p_nom'].sum() * 1e-3  # Convert to GW

            results[zone] = {
                'Total Load': total_load,
                'Solar Capacity': solar_capacity,
                'Wind Capacity': wind_capacity
            }
            
            # Berechne Gesamtsummen für Deutschland (für Prozentanteile)
        total_load_germany = sum([data['Total Load'] for data in results.values()])
        total_solar_germany = sum([data['Solar Capacity'] for data in results.values()])
        total_wind_germany = sum([data['Wind Capacity'] for data in results.values()])
        
        
        print(f"\n--- Prozentuale Anteile je Zone ({market_zones}) ---")
        for zone, data in results.items():
            load_pct = 100 * data['Total Load'] / total_load_germany
            solar_pct = 100 * data['Solar Capacity'] / total_solar_germany
            wind_pct = 100 * data['Wind Capacity'] / total_wind_germany
            print(f"Zone {zone}: Last = {load_pct:.1f} %, PV-Kapazität = {solar_pct:.1f} %, Wind-Kapazität = {wind_pct:.1f} %")

        # Print results to console
        #print(f"\n--- Gesamtlast und erneuerbare Kapazität je Zone ({market_zones}) ---")
        #for zone, data in results.items():
        #    print(f"Zone {zone}: Gesamtlast = {data['Total Load']:.2f} TWh, Erneuerbare Kapazität = {data['Renewable Capacity']:.2f} GW")

    else:
        # Calculate for all of Germany without zone assignment
        # Total load calculation
        
        # Initialize market model
        market_model = german_network(self.market_model)
        
        buses_market = market_model.buses
        loads_market = market_model.loads_t.p
        links_market = market_model.links
        link_loads_market = market_model.links_t.p0
        
        ac_buses_market = buses_market[buses_market['carrier'] == 'AC'].index.astype(str)
        ac_columns_market = [col for col in loads_market.columns if col.split()[0] in ac_buses_market and col.split()[1] == 'AC']
        ac_loads_market = loads_market[ac_columns_market].iloc[::5]
        total_load = ac_loads_market.sum().sum() * 5 * 1e-6  # Convert to TWh
        
        # Berechnung der Link-Lasten für ganz Deutschland
        links_market = market_model.links
        link_loads_market = market_model.links_t.p0
        
        # Filtere nur AC-Links (keine DC-Links)
        ac_links_market = links_market[links_market.carrier != "DC"]
        
        # Summiere die Link-Lasten
        link_ac_loads_market = link_loads_market[link_loads_market.columns.intersection(ac_links_market.index)].iloc[::5]
        total_link_load = link_ac_loads_market.sum().sum() * 5 * 1e-6  # Convert to TWh
        
        # Addiere die Link-Lasten zur Gesamtlast
        total_load += total_link_load

        # Total renewable capacity calculation
        generators_in_germany = market_model.generators
        solar_capacity = generators_in_germany[generators_in_germany.carrier.isin(solar)]['p_nom'].sum() * 1e-3  # Convert to GW
        wind_capacity = generators_in_germany[generators_in_germany.carrier.isin(wind)]['p_nom'].sum() * 1e-3  # Convert to GW

        
        results = {
            'Germany': {
                'Total Load': total_load,
                'Solar Capacity': solar_capacity,
                'Wind Capacity': wind_capacity
            }
        }
        

        # Print results to console
        print("\n--- Gesamtlast und erneuerbare Kapazität für ganz Deutschland ---")
        print(f"Gesamtlast = {total_load:.2f} TWh, PV Kapazität = {solar_capacity:.2f} GW, Wind Kapazität = {wind_capacity:.2f} GW")

    # Plotting
    if plot:
        if market_zones != "none":
            # Load shapefile based on market zones configuration
            if market_zones == "DE2":
                shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp"
            elif market_zones == "DE3":
                shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_12_DE3.shp"
            elif market_zones == "DE4":
                shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp"
            elif market_zones == "DE5":
                shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp"
            else:
                raise ValueError("Ungültiger Wert für market_zones.")

            zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

            # Assign colors
            colors = plt.cm.tab20(range(len(zones)))
            zones["color"] = [mcolors.rgb2hex(color[:3]) for color in colors]

            # Prepare plot
            fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

            # Plot zones
            zones.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)
            zones.plot(ax=ax, facecolor=zones["color"], alpha=0.3)

            # Plot network
            self.market_model.plot(
                ax=ax,
                link_widths=0,
                bus_sizes=0
            )

            # Plot numbers (Load and Renewable Capacity)
            for _, row in zones.iterrows():
                zone = row["id"]
                if zone not in results:
                    continue
                centroid = row.geometry.centroid
                load_pct = 100 * results[zone]["Total Load"] / total_load_germany
                solar_pct = 100 * results[zone]["Solar Capacity"] / total_solar_germany
                wind_pct = 100 * results[zone]["Wind Capacity"] / total_wind_germany
            
                # Plot Load-Anteil - blue
                ax.text(centroid.x, centroid.y + 0.4,
                        f"{load_pct:.1f}%", color="black",
                        fontsize=20, ha="center", weight="bold")
            
                # Plot EE-Anteil - green
                ax.text(centroid.x, centroid.y - 0.0,
                        f"{solar_pct:.1f}%", color="darkorange",
                        fontsize=20, ha="center", weight="bold")
                
                
                ax.text(centroid.x, centroid.y - 0.4,
                        f"{wind_pct:.1f}%", color="blue",
                        fontsize=20, ha="center", weight="bold")
                

        else:
            # Create a simple map of Germany with values plotted in the center
            fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

          
            

            # Plot network (optional)
            self.market_model.plot(
                ax=ax,
                link_widths=0,
                bus_sizes=0
            )
            # Set extent to Germany
            ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
            
            # Plot values in the center of Germany (approximately)
            center_x, center_y = 10.5, 50  # Rough center of Germany

            # Plot load (TWh) - blue
            ax.text(center_x, center_y + 1,
                    f"{results['Germany']['Total Load']:.1f} TWh",
                    color="black", fontsize=20, ha="center", weight="bold")

            # Plot renewable capacity (GW) - green
            ax.text(center_x, center_y+ 0.5,
                    f"{results['Germany']['Solar Capacity']:.1f} GW",
                    color="darkorange", fontsize=20, ha="center", weight="bold")
            
            ax.text(center_x, center_y,
                    f"{results['Germany']['Wind Capacity']:.1f} GW",
                    color="blue", fontsize=20, ha="center", weight="bold")
            
        # Add legend for both cases
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="black", lw=2, label="Jährlicher Strombedarf"),
            Line2D([0], [0], color="darkorange", lw=2, label="inst. PV Leistung"),
            Line2D([0], [0], color="blue", lw=2, label="inst. Wind Leistung"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=20)
        ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    return results


def total_dispatch_by_zone(self, timesteps=range(1752), market_zones="DE4"):
    """    
    Berechnet die Stromproduktion je Carrier und Zone und plottet sie auf einer Karte mit EE-Anteil.

    Parameters
    ----------
    timesteps : range oder Liste
        Die Zeitschritte, die berücksichtigt werden sollen.
    market_zones : str
        Marktzonen-Konfiguration: 'DE2', 'DE3', 'DE4', 'DE5'

    Returns
    -------
    dispatch_per_zone : DataFrame
        Stromproduktion in TWh je Zone und Carrier.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import numpy as np

    # --- 1. Dispatch berechnen ---
    dispatch_series = calc_dispatch_per_carrier(german_network(self), timesteps, dispatch_type="total")
    dispatch_df = dispatch_series.reset_index()
    dispatch_df.columns = ['bus', 'carrier', 'dispatch']

    buses = german_network(self).buses
    if 'zone' not in buses.columns:
        raise ValueError("Spalte 'zone' fehlt in network.buses. Bitte zuerst Zonen zuweisen.")
    dispatch_df = dispatch_df.merge(buses['zone'], left_on='bus', right_index=True)

    # in TWh skalieren
    dispatch_per_zone = ((dispatch_df.groupby(['zone', 'carrier'])['dispatch'].sum()) * 5) / 1e6
    table = dispatch_per_zone.unstack().fillna(0)

    # Erneuerbare Carrier
    renewables = [
        'solar', 'solar_rooftop', 'wind_offshore', 'wind_onshore',
        'reservoir', 'run_of_river', 'biomass', 'central_biomass_CHP', 'industrial_biomass_CHP'
    ]

    print("\nStromproduktion pro Zone (TWh gesamt / Erneuerbar):")
    for zone in table.index:
        total = table.loc[zone].sum()
        renew = table.loc[zone][renewables].sum()
        print(f"{zone}: {total:.2f} TWh gesamt, davon {renew:.2f} TWh erneuerbar ({(renew/total*100):.1f}%)")

    # --- 2. Plot ---
    # Shapefile wählen
    if market_zones == "DE2":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp"
    elif market_zones == "DE3":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_12_DE3.shp"
    elif market_zones == "DE4":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp"
    elif market_zones == "DE5":
        shapefile_path = "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp"
    else:
        raise ValueError("Ungültiger Wert für market_zones. Erlaubt sind: 'DE2', 'DE3', 'DE4', 'DE5'.")

    zones = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # Farben zuweisen
    colors = plt.cm.tab20(range(len(zones)))
    zones['color'] = [mcolors.rgb2hex(color[:3]) for color in colors]

    # Plot starten
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600, subplot_kw={"projection": ccrs.PlateCarree()})

    # Marktzonen zeichnen
    zones.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
    zones.plot(ax=ax, facecolor=zones['color'], alpha=0.3)

    # Netz zeichnen wie in plot_marketzone_clustering
    self.plot(
        ax=ax,
        link_widths=0,
        bus_sizes=0
    )

    # Textplot: TWh und EE-Anteil in Zone-Mitte
    for _, row in zones.iterrows():
        zone_name = row["id"]
        if zone_name not in table.index:
            continue
        total = table.loc[zone_name].sum()
        renew = table.loc[zone_name][renewables].sum()
        share = renew / total * 100 if total > 0 else 0
        centroid = row.geometry.centroid
        ax.text(
            centroid.x, centroid.y,
            f"{total:.1f} TWh\n{share:.1f}% EE",
            fontsize=20, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )


    ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return dispatch_per_zone


def total_load_t_de(self):
    
    market_model = german_network(self)

    #buses_net = network.buses
    buses_market = market_model.buses
    #loads_net = network.loads_t.p
    loads_market = market_model.loads_t.p
    #links_net = network.links
    links_market = market_model.links
    #link_loads_net = network.links_t.p0
    link_loads_market = market_model.links_t.p0


    #ac_buses_net = buses_net[buses_net['carrier'] == 'AC'].index.astype(str)
    ac_buses_market = buses_market[buses_market['carrier'] == 'AC'].index.astype(str)
    #ac_columns_net = [col for col in loads_net.columns if col.split()[0] in ac_buses_net and col.split()[1] == 'AC']
    ac_columns_market = [col for col in loads_market.columns if col.split()[0] in ac_buses_market and col.split()[1] == 'AC']
    #ac_loads_net= loads_net[ac_columns_net]
    ac_loads_market= loads_market[ac_columns_market]


    #links_to_ac_buses_net = links_net[links_net['bus0'].isin(ac_buses_net)]
    links_to_ac_buses_market = links_market[links_market['bus0'].isin(ac_buses_market)] # DC buses checken
    links_to_ac_buses_market = links_to_ac_buses_market[links_to_ac_buses_market.carrier != "DC"]
    #link_ac_loads_net = link_loads_net[link_loads_net.columns.intersection(links_to_ac_buses_net.index)]
    link_ac_loads_market = link_loads_market[link_loads_market.columns.intersection(links_to_ac_buses_market.index)]

    

    #total_load_net = ac_loads_net.sum().sum()*5*1e-6 + link_ac_loads_net.sum().sum()*5*1e-6
    total_load_market = ac_loads_market.copy()
    total_load_market.iloc[:, 0] += link_ac_loads_market.sum(axis=1)

        
    return total_load_market

def residual_t(self):
    ac_buses = german_network(self).buses[german_network(self).buses.carrier == "AC"].index
    ac_gens = self.generators[self.generators.bus.isin(ac_buses)]
    gen_t = self.generators_t.p.loc[:, self.generators_t.p.columns.isin(ac_gens.index)]
    gen_t_sum = gen_t.sum(axis=1)
    
    
    residual_t = total_load_t_de(self).sub(gen_t_sum, axis = 0)*-1
    
    return residual_t

import pandas as pd

def prices_t(self):
    """Calculate electricity exports and imports over DC lines

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamps as rows, and three columns for export, import and net position
    """
    network = self
    de_buses = network.buses[network.buses.country == "DE"]
    for_buses = network.buses[network.buses.country != "DE"]
    # Get the German AC bus
    ac_bus_de = network.buses[(network.buses.carrier == "AC") & (network.buses.country == "DE")]

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
    
    bus_dict = {}
    
    # Füge die Busse für exp Links hinzu (bus1)
    for link_index in exp.index:
        bus_dict[link_index] = exp.loc[link_index, 'bus1']

    # Füge die Busse für imp Links hinzu (bus0)
    for link_index in imp.index:
        bus_dict[link_index] = imp.loc[link_index, 'bus0']


    bus_prices_t = network.buses_t.marginal_price
    export_series = network.links_t.p0[exp.index]
    import_series = network.links_t.p1[imp.index]
    all_exports = pd.concat([export_series, import_series*-1], axis = 1)
    
    
    #all_exports_sum = export_series.sum(axis=1)
    #df =pd.DataFrame({'net_position': all_exports_sum})
    prices = pd.DataFrame(index=all_exports.index)
    prices['highest_price'] = None
    prices['corresponding_bus'] = None
    
    for timestamp in all_exports.index:
        # Calculate the sum of all columns for this timestamp
        row_sum = all_exports.loc[timestamp].sum()

        # Check if the sum is negative
        if row_sum < 0:
            # Find which columns are negative at this timestamp
            negative_columns = all_exports.loc[timestamp][all_exports.loc[timestamp] < 0].index

            # Get the corresponding buses from the bus_dict
            corresponding_buses = []
            for column in negative_columns:
                if column in bus_dict:
                    corresponding_buses.append(bus_dict[column])

            # If there are corresponding buses, find the one with the highest marginal price
            if corresponding_buses:
                # Get the marginal prices for these buses at this timestamp
                bus_prices = bus_prices_t.loc[timestamp, corresponding_buses]

                # Find the bus with the highest price
                max_price = bus_prices.max()
                max_bus = bus_prices.idxmax()

                # Store the results
                prices.at[timestamp, 'highest_price'] = max_price
                prices.at[timestamp, 'corresponding_bus'] = max_bus
            else:
                # If no corresponding buses found, use German AC bus price
                prices.at[timestamp, 'price'] = bus_prices_t.loc[timestamp, ac_bus_de.index[0]]
                prices.at[timestamp, 'bus'] = ac_bus_de.index[0]
        else:
            # If sum is not negative, use German AC bus price
            prices.at[timestamp, 'highest_price'] = bus_prices_t.loc[timestamp, ac_bus_de.index[0]]
            prices.at[timestamp, 'corresponding_bus'] = ac_bus_de.index[0]
           
        
    return prices


def zone_prices_t(self, market_zones="none"):
    """Calculate electricity exports and imports over DC lines, considering market zones if specified.

    Parameters
    ----------
    market_zones : str, optional
        Specifies whether to consider market zones. Can be "none" or "DE3". Default is "none".

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamps as rows, and columns for highest_price, corresponding_bus, and german_bus_price
    """
    network = self

    if market_zones != "none":
        # Assign market zones to buses
        network = assign_market_zones_to_buses(self, market_zones)

    de_buses = network.buses[network.buses.country == "DE"]
    for_buses = network.buses[network.buses.country != "DE"]

    # Get the German AC bus
    ac_bus_de = network.buses[(network.buses.carrier == "AC") & (network.buses.country == "DE")]
    if ac_bus_de.empty:
        raise ValueError("No German AC bus found.")

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

    bus_dict = {}

    # Add buses for export links (bus1)
    for link_index in exp.index:
        bus_dict[link_index] = exp.loc[link_index, 'bus1']
    # Add buses for import links (bus0)
    for link_index in imp.index:
        bus_dict[link_index] = imp.loc[link_index, 'bus0']

    bus_prices_t = network.buses_t.marginal_price
    export_series = network.links_t.p0[exp.index]
    import_series = network.links_t.p1[imp.index]
    all_exports = pd.concat([export_series, import_series * -1], axis=1)

    prices = pd.DataFrame(index=all_exports.index)

    if market_zones == "none":
        # Original behavior
        prices['highest_price'] = None
        prices['corresponding_bus'] = None
        prices['german_bus_price'] = None

        for timestamp in all_exports.index:
            row_sum = all_exports.loc[timestamp].sum()
            if row_sum < 0:
                negative_columns = all_exports.loc[timestamp][all_exports.loc[timestamp] < 0].index
                corresponding_buses = []
                for column in negative_columns:
                    if column in bus_dict:
                        corresponding_buses.append(bus_dict[column])
                if corresponding_buses:
                    bus_prices = bus_prices_t.loc[timestamp, corresponding_buses]
                    max_price = bus_prices.max()
                    max_bus = bus_prices.idxmax()
                    prices.at[timestamp, 'highest_price'] = max_price
                    prices.at[timestamp, 'corresponding_bus'] = max_bus
                else:
                    prices.at[timestamp, 'highest_price'] = bus_prices_t.loc[timestamp, ac_bus_de.index[0]]
                    prices.at[timestamp, 'corresponding_bus'] = ac_bus_de.index[0]
            else:
                prices.at[timestamp, 'highest_price'] = bus_prices_t.loc[timestamp, ac_bus_de.index[0]]
                prices.at[timestamp, 'corresponding_bus'] = ac_bus_de.index[0]

            # Add the price of the German bus at this timestamp
            prices.at[timestamp, 'german_bus_price'] = bus_prices_t.loc[timestamp, ac_bus_de.index[0]]

    else:
        # Behavior for market zones
        zones = network.buses.zone.unique()
        for zone in zones:
            # Skip NaN zones
            if pd.isna(zone):
                continue

            # Convert zone to integer
            zone_int = int(zone)

            zone_buses = network.buses[network.buses.zone == zone]
            zone_prices = pd.DataFrame(index=all_exports.index)
            zone_prices[f'highest_price_{zone_int}'] = None
            zone_prices[f'corresponding_bus_{zone_int}'] = None
            zone_prices[f'german_bus_price_{zone_int}'] = None

            for timestamp in all_exports.index:
                row_sum = all_exports.loc[timestamp].sum()
                if row_sum < 0:
                    negative_columns = all_exports.loc[timestamp][all_exports.loc[timestamp] < 0].index
                    corresponding_buses = []
                    for column in negative_columns:
                        if column in bus_dict:
                            corresponding_buses.append(bus_dict[column])
                    if corresponding_buses:
                        bus_prices = bus_prices_t.loc[timestamp, corresponding_buses]
                        max_price = bus_prices.max()
                        max_bus = bus_prices.idxmax()
                        zone_prices.at[timestamp, f'highest_price_{zone_int}'] = max_price
                        zone_prices.at[timestamp, f'corresponding_bus_{zone_int}'] = max_bus
                    else:
                        zone_prices.at[timestamp, f'highest_price_{zone_int}'] = bus_prices_t.loc[timestamp, ac_bus_de.index[0]]
                        zone_prices.at[timestamp, f'corresponding_bus_{zone_int}'] = ac_bus_de.index[0]
                else:
                    zone_prices.at[timestamp, f'highest_price_{zone_int}'] = bus_prices_t.loc[timestamp, ac_bus_de.index[0]]
                    zone_prices.at[timestamp, f'corresponding_bus_{zone_int}'] = ac_bus_de.index[0]

                # Add the price of the German bus at this timestamp
                zone_prices.at[timestamp, f'german_bus_price_{zone_int}'] = bus_prices_t.loc[timestamp, ac_bus_de.index[0]]

            prices = pd.concat([prices, zone_prices], axis=1)

    return prices

def check_net_position(network):
    """Check if the net position is positive or negative for each timestamp

    Parameters
    ----------
    network : object
        The network object containing the necessary data

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamps as rows, and columns for export, import, net position, and a check if net position is positive
    """
    # Get the DataFrame from the dc_export function
    df = network.dc_export()

    # Initialize a list to store the results
    net_position_check = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        if row['net_position'] > 0:
            net_position_check.append('positive')
        elif row['net_position'] < 0:
            net_position_check.append('negative')
        else:
            net_position_check.append('zero')

    # Add the net_position_check list as a new column to the DataFrame
    df['net_position_check'] = net_position_check

    return df


def electrolyser_dispatch(self):
    
    ac_buses_de = self.buses[(self.buses.carrier=="AC") & (self.buses.country == "DE")]
    electrolyser = self.links[self.links.carrier=="power_to_H2" ]
    electrolyser_de = electrolyser[electrolyser.bus0.isin(ac_buses_de.index)]
    electrolyser_t = self.links_t.p0[electrolyser.index]
    
    return electrolyser_t
    
def fuel_cell_dispatch(self):
    
    ac_buses_de = self.buses[(self.buses.carrier=="AC") & (self.buses.country == "DE")]
    electrolyser = self.links[self.links.carrier=="H2_to_power" ]
    electrolyser_de = electrolyser[electrolyser.bus1.isin(ac_buses_de.index)]
    electrolyser_t = self.links_t.p1[electrolyser.index]
    
    return electrolyser_t

def CH4_to_H2(self):
    buses_de = self.buses[(self.buses.country == "DE")]
    links = self.links[self.links.carrier=="CH4_to_H2" ]
    links_de = links[links.bus0.isin(buses_de.index)]
    links_t = self.links_t.p0[links_de.index]
    
    return links_t


def plot_dispatch_and_prices(networks_dict, start, end):
    """
    Für jedes Szenario (außer "Nodal"):
      - oben: Zonenpreise (nur AC-Busse in DE) je Zone vs Status Quo
      - unten: Elektrolyser-Auslastung [%] vs Status Quo, mit schraffierter Differenz

    Automatische Ableitung der Zone aus den x/y-Koordinaten der AC-Busse:
      * len=1 → einzige Zone "Status Quo"
      * len=2 → DE2: höhere y → "Zone Nord", andere → "Zone Süd"
      * len=3 → DE3: niedrigste y → "Zone Süd",
                 höchste x → "Zone Nordost",
                 verbleibende → "Zone Nordwest"
      * len=4 → DE4: niedrigste y → "Zone Süd",
                 höchste x → "Zone Nordost",
                 niedrigste x → "Zone West",
                 verbleibende → "Zone Nordwest"
    """
    # 1) Elektrolyser-Dispatch sammeln
    dispatch_df = {}
    for name, net in networks_dict.items():
        if name == "Nodal":
            continue
        s = electrolyser_dispatch(net)
        s.index = pd.to_datetime(s.index)
        dispatch_df[name] = s
    dispatch_df = pd.DataFrame(dispatch_df)
    util_df = dispatch_df / dispatch_df.max() * 100
    util_win = util_df.loc[start:end]

    # 2) Zonenpreise sammeln
    price_df = {}
    for name, net in networks_dict.items():
        if name == "Nodal":
            continue

        p = net.buses_t.marginal_price.copy()
        p.index = pd.to_datetime(p.index)

        ac = net.buses[
            (net.buses.carrier == "AC") & (net.buses.country == "DE")
        ][["x","y"]]

        # Zone-Mapping
        if len(ac) == 1:
            idx = ac.index[0]
            mapping = {idx: "Status Quo"}

        elif len(ac) == 2:
            hi = ac["y"].idxmax()
            lo = ac["y"].idxmin()
            mapping = {hi: "Zone Nord", lo: "Zone Süd"}

        elif len(ac) == 3:
            south = ac["y"].idxmin()
            northeast = ac["x"].idxmax()
            other = (set(ac.index) - {south, northeast}).pop()
            mapping = {
                south: "Zone Süd",
                northeast: "Zone Nordost",
                other: "Zone Nordwest"
            }

        elif len(ac) == 4:
            south = ac["y"].idxmin()
            northeast = ac["x"].idxmax()
            west = ac["x"].idxmin()
            other = (set(ac.index) - {south, northeast, west}).pop()
            mapping = {
                south: "Zone Süd",
                northeast: "Zone Nordost",
                west: "Zone West",
                other: "Zone Nordwest"
            }

        else:
            raise ValueError(f"Erwarte 1–4 AC-Busse, aber gefunden: {len(ac)}")

        # Preise filtern und umbenennen
        p_ac = p[list(mapping.keys())].rename(columns=mapping)

        # Mittelwert je Zone
        price_df[name] = p_ac.groupby(axis=1, level=0).mean().loc[start:end]

    # Referenz-Serien
    ref_util = util_win["SQ"]
    ref_price = price_df["SQ"]

    # 3) Plots je Szenario
    for name in util_win.columns:
        if name in ("SQ","Nodal"):
            continue
        scen_util = util_win[name]
        scen_price = price_df[name]

        fig, (ax_price, ax_util) = plt.subplots(
            2,1, sharex=True, figsize=(12,6),
            gridspec_kw={"height_ratios":[1,1]}
        )

        # --- Oben: Zonenpreise ---
        for zone in ref_price.columns:
            ax_price.plot(
                ref_price.index, ref_price[zone],
                linestyle="--", label=f"{zone} (SQ)"
            )
        for zone in scen_price.columns:
            ax_price.plot(
                scen_price.index, scen_price[zone],
                label=zone
            )
        ax_price.set_ylabel("Preis [€/MWh]", fontsize=14)
        ax_price.set_title(f"{name} – Zonenpreise und Elektrolyseurdispatch vs Status Quo", fontsize=20)
        ax_price.grid(True)
        ax_price.legend(fontsize=12, loc="lower left")
        ax_price.tick_params(labelsize=12)

        # --- Unten: Elektrolyser-Auslastung ---
        l1, = ax_util.plot(ref_util.index, ref_util, color="black", label="Status Quo [%]")
        l2, = ax_util.plot(scen_util.index, scen_util, color="blue", label=f"{name} [%]")

        # Schraffierte Differenz
        ax_util.fill_between(ref_util.index, scen_util, ref_util,
                             where=(scen_util>ref_util), interpolate=True,
                             facecolor="lightblue", alpha=0.4,  edgecolor="blue")
        ax_util.fill_between(ref_util.index, scen_util, ref_util,
                             where=(ref_util>scen_util), interpolate=True,
                             facecolor="orange", alpha=0.4,  edgecolor="red")

        # Proxy-Patches für Legende
        blue_patch   = mpatches.Patch(facecolor='lightblue', alpha=0.4, 
                                      label=f"{name} > SQ")
        orange_patch = mpatches.Patch(facecolor='orange', alpha=0.4, 
                                      label="SQ > " + name)

        ax_util.set_ylabel("Dispatch [%]", fontsize=14)
        ax_util.set_xlabel("Zeit", fontsize=14)
        ax_util.grid(True)

        # Legende mit Linien und Patches
        ax_util.legend(handles=[l1, l2, blue_patch, orange_patch],
                       fontsize=12, loc="lower left")
        ax_util.tick_params(labelsize=12)

        plt.tight_layout()
        plt.show()
        
    # ---  Nodal-Auswertung ---------------
    
    net_nodal = networks_dict.get("Nodal")
    if net_nodal is not None:
        # 1) AC-Busse in DE auswählen
        ac_buses_nodal = net_nodal.buses[
            (net_nodal.buses.carrier == "AC") &
            (net_nodal.buses.country == "DE")
        ]

        # 2) marginal_price für diese Busse (1752 × N)
        p = net_nodal.buses_t.marginal_price[ac_buses_nodal.index].copy()
        p.index = pd.to_datetime(p.index)

        # 3) Series Bus 159 und Durchschnitt über alle AC-Busse
        #    wir wissen jetzt: p["159"] funktioniert
        p_bus159 = p["159"]
        p_mean   = p.mean(axis=1)

        # 4) Wiederholung 5× auf 8760 Stunden
        if len(p_bus159) * 5 != 8760:
            raise ValueError("Nodal dispatch muss 1752 Einträge haben!")
        vals_159  = p_bus159.repeat(5).reset_index(drop=True)
        vals_mean = p_mean.repeat(5).reset_index(drop=True)
        full_idx  = pd.date_range(start=net_nodal.snapshots.min(),
                                  periods=8760, freq="H")
        series_159  = pd.Series(vals_159.values,  index=full_idx)
        series_mean = pd.Series(vals_mean.values, index=full_idx)

        # 5) Auf das gewählte Fenster beschränken
        s1 = series_159.loc[start:end]
        s2 = series_mean.loc[start:end]

        # 6) Plot
        plt.figure(figsize=(12,4))
        plt.plot(s1.index, s1, label="Bus 159",           color="green")
        plt.plot(s2.index, s2, label="Ø aller AC‑Buses", color="purple")
        plt.ylabel("Preis [€/MWh]", fontsize=14)
        plt.xlabel("Zeit", fontsize=14)
        plt.title("Nodal‑Preise: Bus 159 vs Durchschnitt", fontsize=16)
        plt.grid(True)
        plt.tick_params(labelsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
        

def plot_dispatch_and_prices(networks_dict, start, end):
    """
    Für jedes Szenario (einschließlich 'Nodal'):
      - Oben: Zonenpreise (nur AC-Busse in DE) vs. Status Quo
      - Unten: Elektrolyser-Auslastung [%] vs. Status Quo, mit schraffierter Differenz

    Für 'Nodal':
      • Bus '159' und Ø aller AC‑Buses, jeweils aus p["159"] bzw. p.mean(axis=1)
      • Werte sind alle 5 Stunden → hier 5× repeat auf stündlich (8760)

    networks_dict: Dict[str, Netzobjekt] mit Keys "SQ","DE2","DE3","DE4","Nodal"
    start,end: z.B. "2011-01-21 18:00", "2011-01-23 18:00"
    """
    # 1) Dispatch einsammeln + bei Nodal expandieren
    dispatch_df = {}
    for name, net in networks_dict.items():
        s = electrolyser_dispatch(net)
        s.index = pd.to_datetime(s.index)
        if name == "Nodal":
            # expand 1752 → 8760
            if len(s) * 5 != 8760:
                raise ValueError("Nodal dispatch must have 1752 entries!")
            s = pd.Series(
                s.repeat(5).values,
                index=pd.date_range(start=net.snapshots.min(), periods=8760, freq="H")
            )
        dispatch_df[name] = s

    dispatch_df = pd.DataFrame(dispatch_df)
    util_df = dispatch_df / dispatch_df.max() * 100
    util_win = util_df.loc[start:end]

    # 2) Zonenpreise sammeln
    price_df = {}
    for name, net in networks_dict.items():
        # raw shadow prices [time x bus]
        p = net.buses_t.marginal_price.copy()
        p.index = pd.to_datetime(p.index)

        # only german AC buses with coords
        ac = net.buses[
            (net.buses.carrier == "AC") & (net.buses.country == "DE")
        ][["x","y"]]

        if name == "Nodal":
            # p has 1752 rows, expand same as dispatch
            p = p[ac.index]
            # Bus 159 and mean
            p_bus159 = p["159"]
            p_mean   = p.mean(axis=1)
            # expand
            vals159 = p_bus159.repeat(5).reset_index(drop=True)
            valsm  = p_mean.repeat(5).reset_index(drop=True)
            full_idx = pd.date_range(start=net.snapshots.min(), periods=8760, freq="H")
            s159 = pd.Series(vals159.values, index=full_idx)
            sm  = pd.Series(valsm.values, index=full_idx)
            # restrict to window
            price_df[name] = pd.DataFrame({
                "Bus 159": s159.loc[start:end],
                "Ø aller Buses": sm.loc[start:end]
            })
        else:
            # determine zone mapping by coord
            idxs = ac.index.tolist()
            if len(idxs) == 1:
                mapping = {idxs[0]: "Status Quo"}
            elif len(idxs) == 2:
                hi = ac["y"].idxmax(); lo = ac["y"].idxmin()
                mapping = {hi: "Zone Nord", lo: "Zone Süd"}
            elif len(idxs) == 3:
                south = ac["y"].idxmin()
                northeast = ac["x"].idxmax()
                other = (set(idxs) - {south, northeast}).pop()
                mapping = {
                    south: "Zone Süd",
                    northeast: "Zone Nordost",
                    other: "Zone Nordwest"
                }
            elif len(idxs) == 4:
                south = ac["y"].idxmin()
                northeast = ac["x"].idxmax()
                west = ac["x"].idxmin()
                other = (set(idxs) - {south, northeast, west}).pop()
                mapping = {
                    south: "Zone Süd",
                    northeast: "Zone Nordost",
                    west: "Zone West",
                    other: "Zone Nordwest"
                }
            else:
                raise ValueError(f"Expected 1–4 AC buses, found {len(idxs)}")

            p_ac = p[list(mapping.keys())].rename(columns=mapping)
            price_df[name] = p_ac.groupby(axis=1, level=0).mean().loc[start:end]

    # reference series
    ref_util  = util_win["SQ"]
    ref_price = price_df["SQ"]

    # 3) Plot je Szenario
    for name in util_win.columns:
        if name == "SQ":
            continue
        scen_util  = util_win[name]
        scen_price = price_df[name]

        fig, (ax_price, ax_util) = plt.subplots(
            2,1, sharex=True, figsize=(12,6),
            gridspec_kw={"height_ratios":[1,1]}
        )

        # --- Oben: Preise ---
        # SQ dashed
        for zone in ref_price.columns:
            ax_price.plot(
                ref_price.index, ref_price[zone],
                linestyle="--", color = 'black', label=f"{zone} (SQ)"
            )
        # scenario solid
        for zone in scen_price.columns:
            ax_price.plot(
                scen_price.index, scen_price[zone],
                label=zone
            )
        ax_price.set_ylabel("Preis [€/MWh]", fontsize=14)
        ax_price.set_title(f"{name} – Stromgestehungskosten und Elektrolyserdispatch vs Status Quo", fontsize=16)
        ax_price.grid(True)
        ax_price.legend(fontsize=12, loc="upper left")
        ax_price.tick_params(labelsize=12)

        # --- Unten: Elektrolyser-Auslastung ---
        l1, = ax_util.plot(
            ref_util.index, ref_util, 
            color="black", label="Status Quo [%]"
        )
        l2, = ax_util.plot(
            scen_util.index, scen_util,
            color="blue", label=f"{name} [%]"
        )
        ax_util.fill_between(
            ref_util.index, scen_util, ref_util,
            where=(scen_util>ref_util),
            interpolate=True,
            facecolor="lightblue", alpha=0.4,
            edgecolor="blue"
        )
        ax_util.fill_between(
            ref_util.index, scen_util, ref_util,
            where=(ref_util>scen_util),
            interpolate=True,
            facecolor="orange", alpha=0.4,
            edgecolor="red"
        )

        # add shading legend
        blue_patch   = mpatches.Patch(facecolor='lightblue', alpha=0.7, 
                                      label=f"{name} > SQ")
        orange_patch = mpatches.Patch(facecolor='orange', alpha=0.4, 
                                      label="SQ > " + name)

        ax_util.set_ylabel("Dispatch [%]", fontsize=14)
        ax_util.grid(True)
        ax_util.legend(handles=[l1, l2, blue_patch, orange_patch], fontsize=12, loc="lower left")
        ax_util.tick_params(labelsize=12)

        plt.tight_layout()
        plt.show()



def plot_dispatch_and_prices(networks_dict, start, end):
    """
    Für jedes Szenario (einschließlich 'Nodal'):
      - Oben: Zonenpreise (nur AC-Busse in DE) vs. Status Quo
      - Unten: Elektrolyser-Auslastung [%] vs. Status Quo, mit schraffierter Differenz

    Für 'Nodal':
      • Bus '159' und Ø aller AC‑Buses, jeweils aus p["159"] bzw. p.mean(axis=1)
      • Werte sind alle 5 Stunden → hier 5× repeat auf stündlich (8760)

    networks_dict: Dict[str, Netzobjekt] mit Keys "SQ","DE2","DE3","DE4","Nodal"
    start,end: z.B. "2011-01-21 18:00", "2011-01-23 18:00"
    """
    # 1) Dispatch einsammeln + bei Nodal expandieren
    dispatch_df = {}
    for name, net in networks_dict.items():
        s = electrolyser_dispatch(net)
        s.index = pd.to_datetime(s.index)
        if name == "Nodal":
            # expand 1752 → 8760
            if len(s) * 5 != 8760:
                raise ValueError("Nodal dispatch must have 1752 entries!")
            s = pd.Series(
                s.repeat(5).values,
                index=pd.date_range(start=net.snapshots.min(), periods=8760, freq="H")
            )
        dispatch_df[name] = s

    dispatch_df = pd.DataFrame(dispatch_df)
    util_df = dispatch_df / dispatch_df.max() * 100
    util_win = util_df.loc[start:end]

    # 2) Zonenpreise sammeln
    price_df = {}
    for name, net in networks_dict.items():
        # raw shadow prices [time x bus]
        p = net.buses_t.marginal_price.copy()
        p.index = pd.to_datetime(p.index)

        # only german AC buses with coords
        ac = net.buses[
            (net.buses.carrier == "AC") & (net.buses.country == "DE")
        ][["x","y"]]

        if name == "Nodal":
            # p has 1752 rows, expand same as dispatch
            p = p[ac.index]
            # Bus 159 and mean
            p_bus159 = p["159"]
            p_mean   = p.mean(axis=1)
            # expand
            vals159 = p_bus159.repeat(5).reset_index(drop=True)
            valsm  = p_mean.repeat(5).reset_index(drop=True)
            full_idx = pd.date_range(start=net.snapshots.min(), periods=8760, freq="H")
            s159 = pd.Series(vals159.values, index=full_idx)
            sm  = pd.Series(valsm.values, index=full_idx)
            # restrict to window
            price_df[name] = pd.DataFrame({
                "Bus 159": s159.loc[start:end],
                "Ø aller Buses": sm.loc[start:end]
            })
        else:
            # determine zone mapping by coord
            idxs = ac.index.tolist()
            if len(idxs) == 1:
                mapping = {idxs[0]: "Status Quo"}
            elif len(idxs) == 2:
                hi = ac["y"].idxmax(); lo = ac["y"].idxmin()
                mapping = {hi: "Zone Nord", lo: "Zone Süd"}
            elif len(idxs) == 3:
                south = ac["y"].idxmin()
                northeast = ac["x"].idxmax()
                other = (set(idxs) - {south, northeast}).pop()
                mapping = {
                    south: "Zone Süd",
                    northeast: "Zone Nordost",
                    other: "Zone Nordwest"
                }
            elif len(idxs) == 4:
                south = ac["y"].idxmin()
                northeast = ac["x"].idxmax()
                west = ac["x"].idxmin()
                other = (set(idxs) - {south, northeast, west}).pop()
                mapping = {
                    south: "Zone Süd",
                    northeast: "Zone Nordost",
                    west: "Zone West",
                    other: "Zone Nordwest"
                }
            else:
                raise ValueError(f"Expected 1–4 AC buses, found {len(idxs)}")

            p_ac = p[list(mapping.keys())].rename(columns=mapping)
            price_df[name] = p_ac.groupby(axis=1, level=0).mean().loc[start:end]

    # reference series
    ref_util  = util_win["SQ"]
    ref_price = price_df["SQ"]

    # 3) Plot je Szenario
    for name in util_win.columns:
        if name == "SQ":
            continue
        scen_util  = util_win[name]
        scen_price = price_df[name]

        fig, (ax_price, ax_util) = plt.subplots(
            2,1, sharex=True, figsize=(12,6),
            gridspec_kw={"height_ratios":[1,1]}
        )

        # --- Oben: Preise ---
        # SQ dashed
        for zone in ref_price.columns:
            ax_price.plot(
                ref_price.index, ref_price[zone],
                linestyle="--", color = 'black', label=f"{zone} (SQ)"
            )
        # scenario solid
        for zone in scen_price.columns:
            ax_price.plot(
                scen_price.index, scen_price[zone],
                label=zone
            )
        ax_price.set_ylabel("Preis [€/MWh]", fontsize=14)
        ax_price.set_title(f"{name} – Stromgestehungskosten und Elektrolyserdispatch vs Status Quo", fontsize=16)
        ax_price.grid(True)
        ax_price.legend(fontsize=12, loc="upper left")
        ax_price.tick_params(labelsize=12)

        # --- Unten: Elektrolyser-Auslastung ---
        # --- Unten: Elektrolyser‑Dispatch je Zone in MW ---
        net = networks_dict[name]
        s = electrolyser_dispatch(net)            # DataFrame oder Series mit Dispatch je Bus
        s.index = pd.to_datetime(s.index)

        if name == "Nodal":
            # unverändert: Bus 159 und Ø aller AC‑Buses hochskalieren
            ac = net.buses[(net.buses.carrier=="AC") & (net.buses.country=="DE")]
            p = net.buses_t.marginal_price[ac.index].copy()
            p.index = pd.to_datetime(p.index)

            p159 = p["159"]
            pmean = p.mean(axis=1)
            # expand 5→8760
            vals159  = p159.repeat(5).reset_index(drop=True)
            valsmean = pmean.repeat(5).reset_index(drop=True)
            full_idx = pd.date_range(start=net.snapshots.min(), periods=8760, freq="H")
            dispatch_for_plot = pd.DataFrame({
                "Bus 159":        pd.Series(vals159.values,  index=full_idx),
                "Ø aller AC‑Buses": pd.Series(valsmean.values, index=full_idx)
            }).loc[start:end]

        else:
            # 2–4 AC-Busse zonal clustern
            ac = net.buses[(net.buses.carrier=="AC") & (net.buses.country=="DE")][["x","y"]]
            idxs = ac.index.tolist()

            if   len(idxs)==1:
                mapping = {idxs[0]: "Status Quo"}
            elif len(idxs)==2:
                mapping = { ac["y"].idxmax(): "Zone Nord",
                            ac["y"].idxmin(): "Zone Süd" }
            elif len(idxs)==3:
                south     = ac["y"].idxmin()
                northeast = ac["x"].idxmax()
                other     = (set(idxs)-{south,northeast}).pop()
                mapping = { south:      "Zone Süd",
                            northeast:  "Zone Nordost",
                            other:      "Zone Nordwest" }
            else:  # len==4
                south     = ac["y"].idxmin()
                northeast = ac["x"].idxmax()
                west      = ac["x"].idxmin()
                other     = (set(idxs)-{south,northeast,west}).pop()
                mapping = { south:     "Zone Süd",
                            northeast: "Zone Nordost",
                            west:      "Zone West",
                            other:     "Zone Nordwest" }

            # s muss ein DataFrame mit Spalten=Bus-Indizes sein
            # hier nach Zonen umbenennen und direkt aus dem Zeitfenster holen
            dispatch_for_plot = (
                s[list(mapping.keys())]
                .rename(columns=mapping)
                .loc[start:end]
            )

        # jetzt pro Zone plotten
        for zone, series in dispatch_for_plot.items():
            ax_util.plot(series.index, series, label=zone)

        ax_util.set_ylabel("Dispatch [MW]", fontsize=14)
        ax_util.set_xlabel("Zeit", fontsize=14)
        ax_util.grid(True)
        ax_util.legend(fontsize=10, loc="upper right")
        ax_util.tick_params(labelsize=12)
        
        
def plot_nodal_prices(self):
    # 1) AC-Busse identifizieren
    ac_index = self.buses[self.buses.carrier == "AC"].index

    # 2) Durchschnittliche nodalen Preise über die Zeit für AC-Busse
    avg_prices_ac = self.buses_t.marginal_price[ac_index].mean(axis=0)

    # 3) Vollständige Series mit NaNs für Nicht-AC, damit self.plot den Rest ignoriert
    prices_all = pd.Series(index=self.buses.index, dtype=float)
    prices_all.loc[ac_index] = avg_prices_ac

    # 4) Knotengrößen: 0 für Nicht-AC, z.B. 0.02 für AC
    sizes_all = pd.Series(0.0, index=self.buses.index)
    sizes_all.loc[ac_index] = 0.02

    # 5) Invertierte Colormap & Norm nur basierend auf AC-Preisen
    cmap = plt.cm.viridis_r              # umgekehrte Viridis-Palette
    norm = plt.Normalize(vmin=avg_prices_ac.min(), vmax=avg_prices_ac.max())

    # 6) Plot aufsetzen
    fig, ax = plt.subplots(
        figsize=(10, 6), dpi=600,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    self.plot(
        ax=ax,
        link_widths=0,
        bus_sizes=sizes_all.values,
        bus_colors=prices_all.values,
        bus_cmap=cmap,
        bus_norm=norm,
        line_widths=1,
        line_colors="grey"
    )

    # 7) Colorbar für AC-Preise
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(avg_prices_ac.values)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    
    # Schriftgröße der Tick‑Labels anpassen
    cbar.ax.tick_params(labelsize=18)   # z.B. 14pt
    
    # Schriftgröße und Padding des Colorbar‑Labels anpassen
    cbar.set_label(
        'Ø Knotenpreis [€/MWh]',
        fontsize=20,     # z.B. 16pt
        labelpad=12      # Abstand zum Colorbar‑Strich
    )
    
    plt.show()
    
    
def ramps_down(self):
    
    network = self
    ramp_up = calc_dispatch_per_carrier(network, timesteps=range(1752), dispatch_type = "ramp_up")
    ramp_down = calc_dispatch_per_carrier(network, timesteps=range(1752), dispatch_type = "ramp_down")
    
    ramp_down_total = ramp_down*5 - ramp_up*5
    
                
    return ramp_down_total
    


def ramps(self):
    network = self
    # hole die beiden Series mit MultiIndex (Bus, Carrier)
    ramp_up = calc_dispatch_per_carrier(network,
                                        timesteps=range(1752),
                                        dispatch_type="ramp_up")
    ramp_down = calc_dispatch_per_carrier(network,
                                          timesteps=range(1752),
                                          dispatch_type="ramp_down")

    # Summiere jeweils über alle Busse für jeden Carrier
    # Dabei wird angenommen, dass der zweite Index-Level "Carrier" heißt.
    ramp_up_per_carrier = ramp_up.groupby(level="carrier").sum()
    ramp_down_per_carrier = ramp_down.groupby(level="carrier").sum()

    # Baue den DataFrame auf
    df = pd.DataFrame({
        "ramp_up":   ramp_up_per_carrier*5,
        "ramp_down": ramp_down_per_carrier*5
    })

    # Optional: falls du Carrier als Index behalten, ansonsten
    # df = df.reset_index().rename(columns={"index": "Carrier"})
    return df


def prepare_bus_values(networks_dict, carrier):
    """
    Berechnet einmalig die Bus-Werte (Abregelung in GWh negativ) 
    und gibt ein dict title -> pd.Series zurück.
    
    Beispielaufruf:
        prepare_bus_values(networks_dict = {
        "Status Quo": (net_sq,  "Status Quo"),
        "DE2":         (net_DE2, "DE2"),
        "DE3":         (net_DE3, "DE3"),
        "DE4":         (net_DE4, "DE4"),
        }, carrier = "OCGT")
    """
    import pandas as pd

    bus_values = {}
    for title, (net, _) in networks_dict.items():
        off = ramps_down(net, carrier)
        vals = -off / 1000                           # GWh mit Minus
        vals = vals.reindex(net.buses.index).fillna(0)
        bus_values[title] = vals
        
    return bus_values


def prepare_electrolyser_redispatch_per_bus(networks_dict):
    """
    Berechnet pro Szenario (title) den jährlichen Redispatch je Bus,
    indem zunächst der Link‑Dispatch im Markt vs. Netz verglichen
    und danach auf die AC-Busse des Netz‑Objekts aggregiert wird.
    Liefert dict title -> pd.Series (Index: alle Bus‑IDs des Netz),
    füllt mit 0 für Busse ohne Elektrolyseur.
    """
    import pandas as pd

    bus_values = {}
    for title, (net_m, net_n) in networks_dict.items():
        
        # --- 2) Elektrolyseur‑Links je Netz ---
        links_m = net_m.links[net_m.links.carrier == "power_to_H2"]
        links_n = net_n.links[net_n.links.carrier == "power_to_H2"]

        # --- 3) Dispatch‑Zeitreihen je Link (MWh) aufsummieren ---
        disp_m = net_m.links_t.p0[links_m.index].sum(axis=0)
        disp_n = (net_n.links_t.p0[links_n.index].sum(axis=0))*5

        # --- 4) Differenz pro Link (Market – Net) ---
        diff_link = disp_n.subtract(disp_m, fill_value=0)

        # --- 5) Auf AC‑Busse des Netz‑Objekts aggregieren ---
        #    bus_map: Serie link_id -> bus0
        bus_map = net_n.links.bus0.reindex(diff_link.index)
        sum_per_bus = diff_link.groupby(bus_map).sum()

        # --- 6) Umrechnung MWh → GWh & vollständiger Bus‑Index ---
        redispatch = sum_per_bus.div(1000.0)
        redispatch = redispatch.reindex(net_n.buses.index).fillna(0)

        bus_values[title] = redispatch

    return bus_values

    
    
def plot_multi_network_redispatch(
    networks_dict,
    bus_values,
    carriers,
    ramp_type="ramp_down",
    scaling=0.0001
):
    """
    Plottet für jedes Szenario in networks_dict die aggregated ramps
    (up/down) für die angegebenen carriers.

    Parameters
    ----------
    networks_dict : dict
        Szenario-Definitionen, z. B. {"Status Quo": (net_sq,"Status Quo"), ...}
    bus_values : dict
        Dict title -> {"ramp_up": df_up, "ramp_down": df_down}
        mit DataFrames Index=Bus, Columns=Carrier, NaN==0
    carriers : str oder list
        Carrier-Namen, die geplottet werden sollen
    ramp_type : {"ramp_up","ramp_down"}
        Welche Ramp-Art zu verwenden (beide DataFrames sind positiv!)
    scaling : float
        Multiplikator für bus_sizes
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import numpy as np
    import pandas as pd

    # Carrier-Liste erzwingen
    if isinstance(carriers, str):
        carriers = [carriers]

    # 1) Baue je Szenario eine Series, clippe zuerst, dann summe
    plot_series = {}
    for title, dfs in bus_values.items():
        # dfs ist {"ramp_up": df_up, "ramp_down": df_down}
        df = dfs[ramp_type].fillna(0) /1000

        # Kontrolliere, ob Spalten existieren
        missing = [c for c in carriers if c not in df.columns]
        if missing:
            raise KeyError(f"Carrier {missing} fehlen in bus_values['{title}']['{ramp_type}']. "
                           f"Verfügbare: {list(df.columns)}")

        sub = df[carriers]
        if ramp_type == "ramp_up":
            # nur positive Teile (ist ohnehin positiv)
            clipped = sub.clip(lower=0)
            label = "Positiver Redispatch"
        else:
            # ramp_down ist positiv gespeichert → auch nur clip
            clipped = sub.clip(upper=0)
            clipped = -clipped
            label = "Negativer Redispatch"

        s = clipped.sum(axis=1)
        plot_series[title] = s

    # 2) Normierung 0…Max
    all_vals = np.concatenate([s.values for s in plot_series.values()])
    vmin, vmax = 0.0, all_vals.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis_r

    # 3) Subplots nebeneinander, keinen Zwischenraum
    n = len(networks_dict)
    fig, axs = plt.subplots(
        1, n,
        figsize=(6 * n, 7.5),
        dpi=100,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    plt.subplots_adjust(wspace=0, bottom=0.12)

    # 4) Pro Szenario plotten
    for ax, (title, (net, market_zone)) in zip(axs, networks_dict.items()):
        # Shapefile laden, falls definiert
        if market_zone.upper() != 'NONE':
            shp_map = {
                "DE2": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_2_DE2.shp",
                "DE3": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_12_DE3.shp",
                "DE4": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_13_DE4.shp",
                "DE5": "/home/student/Masterarbeit_Dateien/Shape-Files/shape_files_old/BZR_config_14_DE5.shp",
            }
            path = shp_map.get(market_zone)
            if path:
                zones = gpd.read_file(path).to_crs(epsg=4326)
                cols = plt.cm.tab20(range(len(zones)))
                zones['color'] = [mcolors.rgb2hex(c[:3]) for c in cols]
                zones.boundary.plot(ax=ax, edgecolor='black', lw=0.5)
                zones.plot(ax=ax, facecolor=zones['color'], alpha=0.3)

        vals       = plot_series[title]
        bus_colors = vals.map(lambda x: cmap(norm(x)))
        bus_sizes  = vals * scaling

        net.plot(
            ax=ax,
            link_widths=0,
            bus_sizes=bus_sizes,
            bus_colors=bus_colors,
            line_widths=0.5,
            line_colors="grey"
        )

        ax.set_title(title, fontsize=28)
        ax.set_extent([5.5, 15.5, 47, 55.5], crs=ccrs.PlateCarree())
        ax.axis('off')

    # 5) Flache Colorbar direkt unter den Subplots
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cax = fig.add_axes([0.25, 0.15, 0.5, 0.03])
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label(f"{label} von Offshore  in GWh", fontsize=28, labelpad=6)
    cbar.ax.tick_params(labelsize=20)

    # Ticks: immer positiv berechnet, aber im Label "-" voranstellen, wenn down
    ticks = np.linspace(vmin, vmax, num=5)
    if ramp_type == "ramp_down":
        tick_labels = [f"-{t:.1f}" for t in ticks]
    else:
        tick_labels = [f"{t:.1f}" for t in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    plt.show()