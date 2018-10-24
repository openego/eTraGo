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

import os
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import pandas as pd
import numpy as np
import time
import math
from math import sqrt, log10

if 'READTHEDOCS' not in os.environ:
    from geoalchemy2.shape import to_shape

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, MarlonSchlemminger, mariusves, lukasol"

basemap_present = True
try:
    from mpl_toolkits.basemap import Basemap
except:
    basemap_present = False


def add_coordinates(network):
    """
    Add coordinates to nodes based on provided geom

    Parameters
    ----------
    network : PyPSA network container

    Returns
    -------
    Altered PyPSA network container ready for plotting
    """
    for idx, row in network.buses.iterrows():
        wkt_geom = to_shape(row['geom'])
        network.buses.loc[idx, 'x'] = wkt_geom.x
        network.buses.loc[idx, 'y'] = wkt_geom.y

    return network


def coloring():
    colors = {'biomass': 'green',
              'coal': 'k',
              'gas': 'orange',
              'eeg_gas': 'olive',
              'geothermal': 'purple',
              'lignite': 'brown',
              'oil': 'darkgrey',
              'other_non_renewable': 'pink',
              'reservoir': 'navy',
              'run_of_river': 'aqua',
              'pumped_storage': 'steelblue',
              'solar': 'yellow',
              'uranium': 'lime',
              'waste': 'sienna',
              'wind': 'blue',
              'wind_onshore': 'skyblue',
              'wind_offshore': 'cornflowerblue',
              'slack': 'pink',
              'load shedding': 'red',
              'nan': 'm',
              'imports': 'salmon',
              '': 'm'}
    return colors


def plot_line_loading(
        network,
        timesteps=range(1,2),
        filename=None,
        boundaries=[],
        arrows=False):
    """
    Plots line loading as a colored heatmap.

    Line loading is displayed as relative to nominal capacity in %.
    
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    timesteps : range 
        Defines which timesteps are considered. If more than one, an 
        average line loading is calculated.
    filename : str
        Specify filename
        If not given, figure will be show directly
    boundaries : list
        If given, the colorbar is fixed to a given min and max value
    arrows : bool
        If True, the direction of the power flows is displayed as 
        arrows.
    """
    # TODO: replace p0 by max(p0,p1) and analogously for q0
    # TODO: implement for all given snapshots

    # calculate relative line loading as S/S_nom
    # with S = sqrt(P^2 + Q^2)
    cmap = plt.cm.jet
    array_line = [['Line'] * len(network.lines), network.lines.index]
    array_link = [['Link'] * len(network.links), network.links.index]
    
    if network.lines_t.q0.empty:

        loading_lines = pd.Series((network.lines_t.p0.mul(
            network.snapshot_weightings, axis=0).loc[network.snapshots[
            timesteps]].abs().sum() / (network.lines.s_nom_opt)).data,
            index=array_line)

    else:
        loading_lines = pd.Series(((network.lines_t.p0.mul(
                network.snapshot_weightings, axis=0)\
                    .loc[network.snapshots[timesteps]].abs().sum() ** 2 +\
                    network.lines_t.q0.mul(
                            network.snapshot_weightings, axis=0)\
                    .loc[network.snapshots[timesteps]].abs().sum() ** 2).\
                    apply(sqrt) / (network.lines.s_nom_opt)).data, index =
                            array_line)

    # Aviod covering of bidirectional links
    network.links['linked_to'] = 0
    for i,  row in network.links.iterrows():
        if not (network.links.index[(network.links.bus0 == row['bus1']) &
                                  (network.links.bus1 == row['bus0']) &
                                  (network.links.length == row['length']
                                  )]).empty:

            l = network.links.index[(network.links.bus0 == row['bus1']) &
                                  (network.links.bus1 == row['bus0']) &
                                  (network.links.length == row['length'])]

            network.links.set_value(i, 'linked_to',l.values[0])

    network.links.linked_to = network.links.linked_to.astype(str)
    link_load = network.links_t.p0[network.links.index[
            network.links.linked_to == '0']]

    for i, row in network.links[network.links.linked_to != '0'].iterrows():
        load = pd.DataFrame(index = network.links_t.p0.index,
                            columns = ['to', 'from'])
        load['to'] = network.links_t.p0[row['linked_to']]
        load['from'] = network.links_t.p0[i]
        link_load[i] = load.abs().max(axis = 1)

    loading_links = pd.Series((link_load.mul(
            network.snapshot_weightings, axis=0).loc[network.snapshots[
            timesteps]].abs().sum()[network.links.index] / (
        network.links.p_nom_opt)).data, index=array_link).dropna()

    load_links_rel = (loading_links/  
                      network.snapshot_weightings\
                            [network.snapshots[timesteps]].sum())* 100
        
    load_lines_rel = (loading_lines / network.snapshot_weightings\
                            [network.snapshots[timesteps]].sum()) * 100

    loading = load_lines_rel.append(load_links_rel)

    ll = network.plot(line_colors=loading, line_cmap=cmap,
                      title="Line loading", line_widths=0.55)
    # add colorbar, note mappable sliced from ll by [1]

    if not boundaries:
        v = np.linspace(min(loading), max(loading), 101)
        boundaries = [min(loading), max(loading)]
        
    else:
        v = np.linspace(boundaries[0], boundaries[1], 101)
        
    cb = plt.colorbar(ll[1], boundaries=v,
                      ticks=v[0:101:10])
    cb_Link = plt.colorbar(ll[2], boundaries=v,
                      ticks=v[0:101:10])

    cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])
    
    cb_Link.set_clim(vmin=boundaries[0], vmax=boundaries[1])
    
    cb_Link.remove()
    
    cb.set_label('Line loading in %')

    if arrows:
        ax = plt.axes()
        path = ll[1].get_segments()
        x_coords_lines = np.zeros([len(path)])
        cmap = cmap
        colors = cmap(ll[1].get_array() / 100)
        for i in range(0, len(path)):
            x_coords_lines[i] = network.buses.loc[str(
                network.lines.iloc[i, 2]), 'x']
            color = colors[i]
            if (x_coords_lines[i] == path[i][0][0] and load_lines_rel[i] >= 0):
                arrowprops = dict(arrowstyle="->", color=color)
            else:
                arrowprops = dict(arrowstyle="<-", color=color)
            ax.annotate(
                "",
                xy=abs(
                    (path[i][0] - path[i][1]) * 0.51 - path[i][0]),
                xytext=abs(
                    (path[i][0] - path[i][1]) * 0.49 - path[i][0]),
                arrowprops=arrowprops,
                size=10)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()



def plot_line_loading_diff(networkA, networkB, timestep=0):
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
    """

    # new colormap to make sure 0% difference has the same color in every plot
    def shiftedColorMap(
            cmap,
            start=0,
            midpoint=0.5,
            stop=1.0,
            name='shiftedcmap'):
        '''
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
        '''
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)

        return newcmap

    # calculate difference in loading between both networks
    loading_switches = abs(
        networkA.lines_t.p0.mul(networkA.snapshot_weightings, axis=0).\
        loc[networkA.snapshots[timestep]].to_frame())
    loading_switches.columns = ['switch']
    loading_noswitches = abs(
        networkB.lines_t.p0.mul(networkB.snapshot_weightings, axis=0).\
        loc[networkB.snapshots[timestep]].to_frame())
    loading_noswitches.columns = ['noswitch']
    diff_network = loading_switches.join(loading_noswitches)
    diff_network['noswitch'] = diff_network['noswitch'].fillna(
        diff_network['switch'])
    diff_network[networkA.snapshots[timestep]] \
        = diff_network['switch'] - diff_network['noswitch']

    # get switches
    new_buses = pd.Series(index=networkA.buses.index.values)
    new_buses.loc[set(networkA.buses.index.values) -
                  set(networkB.buses.index.values)] = 0.1
    new_buses = new_buses.fillna(0)

    # plot network with difference in loading and shifted colormap
    loading = (diff_network.loc[:, networkA.snapshots[timestep]] /
               (networkA.lines.s_nom)) * 100
    midpoint = 1 - max(loading) / (max(loading) + abs(min(loading)))
    shifted_cmap = shiftedColorMap(
        plt.cm.jet, midpoint=midpoint, name='shifted')
    ll = networkA.plot(line_colors=loading, line_cmap=shifted_cmap,
                       title="Line loading", bus_sizes=new_buses,
                       bus_colors='blue', line_widths=0.55)

    cb = plt.colorbar(ll[1])
    cb.set_label('Difference in line loading in % of s_nom')
    


def network_expansion(network, method = 'rel', ext_min=0.1,
                      ext_width=False, filename=None, boundaries=[]):
    """Plot relative or absolute network extension of AC- and DC-lines.
    
    Parameters
    ----------
    network: PyPSA network container
        Holds topology of grid including results from powerflow analysis
    method: str
        Choose 'rel' for extension relative to s_nom and 'abs' for 
        absolute extensions. 
    ext_min: float
        Choose minimum relative line extension shown in plot in p.u..
    ext_width: float or bool
        Choose if line_width respects line extension. Turn off with 'False' or
        set linear factor to decremise extension line_width.
    filename: str or None
        Save figure in this direction
    boundaries: array
       Set boundaries of heatmap axis
    
    """

    cmap = plt.cm.jet

    overlay_network = network.copy()
    overlay_network.lines = overlay_network.lines[
                            overlay_network.lines.s_nom_extendable & ((
                                    overlay_network.lines.s_nom_opt -
                                  overlay_network.lines.s_nom_min) /
                                overlay_network.lines.s_nom >= ext_min)]
    overlay_network.links = overlay_network.links[
                            overlay_network.links.p_nom_extendable & ((
                                    overlay_network.links.p_nom_opt -
                                  overlay_network.links.p_nom_min)/
                                 overlay_network.links.p_nom >= ext_min)]

    for i, row in overlay_network.links.iterrows():
        linked = overlay_network.links[(row['bus1'] ==
                overlay_network.links.bus0) & (
                        row['bus0'] == overlay_network.links.bus1)]
        if not linked.empty:
            if row['p_nom_opt'] < linked.p_nom_opt.values[0]:
                overlay_network.links.p_nom_opt[i] = linked.p_nom_opt.values[0]

    array_line = [['Line'] * len(overlay_network.lines),
                  overlay_network.lines.index]
    
    array_link = [['Link'] * len(overlay_network.links),
                  overlay_network.links.index]
    
    if method == 'rel':

        extension_lines = pd.Series((100 *
                                 (overlay_network.lines.s_nom_opt -
                                  overlay_network.lines.s_nom_min) /
                                overlay_network.lines.s_nom).data,
                                index=array_line)

        extension_links = pd.Series((100 *
                                 (overlay_network.links.p_nom_opt -
                                  overlay_network.links.p_nom_min)/
                                 (overlay_network.links.p_nom)).data,
                                index=array_link)
    if method == 'abs':
        extension_lines = pd.Series(
                                 (overlay_network.lines.s_nom_opt -
                                  overlay_network.lines.s_nom_min).data,
                                index=array_line)

        extension_links = pd.Series(
                                 (overlay_network.links.p_nom_opt -
                                  overlay_network.links.p_nom_min).data,
                                index=array_link)
        

    extension = extension_lines.append(extension_links)

    # Plot whole network in backgroud of plot
    network.plot(
            line_colors=pd.Series("grey", index = [['Line'] * len(
                    network.lines), network.lines.index]).append(
            pd.Series("grey", index = [['Link'] * len(network.links),
                  network.links.index])),
            bus_sizes=0,
            line_widths=pd.Series(0.5, index = [['Line'] * len(network.lines),
                  network.lines.index]).append(
            pd.Series(0.55, index = [['Link'] * len(network.links),
                  network.links.index])))

    if not ext_width:
        line_widths= pd.Series(0.8, index = array_line).append(
                pd.Series(0.8, index = array_link))
        
    else: 
        line_widths= 0.5 + (extension / ext_width)
        
    ll = overlay_network.plot(
        line_colors=extension,
        line_cmap=cmap,
        bus_sizes=0,
        title="Optimized AC- and DC-line expansion",
        line_widths=line_widths) 

    if not boundaries:
        v = np.linspace(min(extension), max(extension), 101)
        boundaries = [min(extension), max(extension)]
        
    else:
        v = np.linspace(boundaries[0], boundaries[1], 101)
        
    if not extension_links.empty:
        cb_Link = plt.colorbar(ll[2], boundaries=v,
                      ticks=v[0:101:10])
        cb_Link.set_clim(vmin=boundaries[0], vmax=boundaries[1])
        
        cb_Link.remove()
        
    cb = plt.colorbar(ll[1], boundaries=v,
                      ticks=v[0:101:10], fraction=0.046, pad=0.04)
    
    cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])
    
    if method == 'rel':
        cb.set_label('line expansion relative to s_nom in %')
    if method == 'abs':
        cb.set_label('line expansion in MW')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

def network_expansion_diff (networkA, networkB, filename=None, boundaries=[]):
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
    
    """

    cmap = plt.cm.jet
    
    array_line = [['Line'] * len(networkA.lines), networkA.lines.index]

    extension_lines = pd.Series(100 *\
                                 ((networkA.lines.s_nom_opt - \
                                    networkB.lines.s_nom_opt)/\
                                    networkA.lines.s_nom_opt  ).values,\
                                index=array_line)

    array_link = [['Link'] * len(networkA.links), networkA.links.index]

    extension_links = pd.Series(100 *
                                 ((networkA.links.p_nom_opt -\
                                    networkB.links.p_nom_opt)/\
                                    networkA.links.p_nom_opt).values,\
                                index=array_link)

    extension = extension_lines.append(extension_links)

    ll = networkA.plot(
        line_colors=extension,
        line_cmap=cmap,
        bus_sizes=0,
        title="Derivation of AC- and DC-line extension",
        line_widths=2)

    if not boundaries:
        v = np.linspace(min(extension), max(extension), 101)
        boundaries = [min(extension).round(0), max(extension).round(0)]
        
    else:
        v = np.linspace(boundaries[0], boundaries[1], 101)
        
    if not extension_links.empty:
        cb_Link = plt.colorbar(ll[2], boundaries=v,
                      ticks=v[0:101:10])
        cb_Link.set_clim(vmin=boundaries[0], vmax=boundaries[1])
        
        cb_Link.remove()
        
    cb = plt.colorbar(ll[1], boundaries=v,
                      ticks=v[0:101:10], fraction=0.046, pad=0.04)
    
    cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])
    cb.set_label('line extension derivation  in %')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def full_load_hours(network, boundaries=[], filename=None, two_cb=False):
    """Plot loading of lines in equivalten full load hours.
    
    Parameters
    ----------
    network: PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename: str or None
        Save figure in this direction
    boundaries: array
       Set boundaries of heatmap axis
    two_cb: bool
        Choose if an extra colorbar for DC-lines is plotted
    
    """
    cmap = plt.cm.jet

    array_line = [['Line'] * len(network.lines), network.lines.index]

    load_lines = pd.Series(abs((network.lines_t.p0.mul(
                                network.snapshot_weightings, axis=0).sum() /
                                (network.lines.s_nom))).data, index=array_line)

    array_link = [['Link'] * len(network.links), network.links.index]

    load_links = pd.Series(abs((network.links_t.p0.mul(
                                network.snapshot_weightings, axis=0).sum() /
                                (network.links.p_nom))).data, index=array_link)

    load_hours = load_lines.append(load_links)

    ll = network.plot(line_colors=load_hours, line_cmap=cmap, bus_sizes=0,
                      title="Full load-hours of lines", line_widths=2)

    if not boundaries:
        cb = plt.colorbar(ll[1])
        cb_Link = plt.colorbar(ll[2])
    elif boundaries:
        v = np.linspace(boundaries[0], boundaries[1], 101)

        cb_Link = plt.colorbar(ll[2], boundaries=v,
                               ticks=v[0:101:10])
        cb_Link.set_clim(vmin=boundaries[0], vmax=boundaries[1])

        cb = plt.colorbar(ll[1], boundaries=v,
                          ticks=v[0:101:10])
        cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])

    if two_cb:
        cb_Link.set_label('Number of full-load hours of DC-lines')
        cb.set_label('Number of full-load hours of AC-lines')

    else:
        cb.set_label('Number of full-load hours')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

def plot_q_flows(network):
    """Plot maximal reactive line load. 
    
    Parameters
    ----------
    network: PyPSA network container
        Holds topology of grid including results from powerflow analysis

    """
    cmap_line = plt.cm.jet
    
    q_flows_max = abs(network.lines_t.q0.abs().max()/(network.lines.s_nom))
    
    ll = network.plot(line_colors = q_flows_max, line_cmap = cmap_line)
    boundaries = [min(q_flows_max), max(q_flows_max)]
    v = np.linspace(boundaries[0], boundaries[1], 101)

    cb = plt.colorbar(ll[1], boundaries=v,
                      ticks=v[0:101:10])

    cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])
    

def max_load(network, boundaries=[], filename=None, two_cb=False):
    
    """Plot maximum loading of each line. 
    
    Parameters
    ----------
    network: PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename: str or None
        Save figure in this direction
    boundaries: array
       Set boundaries of heatmap axis
    two_cb: bool
        Choose if an extra colorbar for DC-lines is plotted
    
    """

    cmap_line = plt.cm.jet
    cmap_link = plt.cm.jet
    array_line = [['Line'] * len(network.lines), network.lines.index]
    array_link = [['Link'] * len(network.links), network.links.index]

    if network.lines_t.q0.empty:
        load_lines = pd.Series((abs(network.lines_t.p0).max(
                ) / (network.lines.s_nom) * 100).data, index=array_line)

    else: load_lines = pd.Series(((network.lines_t.p0**2 +
                           network.lines_t.q0 ** 2).max().apply(sqrt)/
                           (network.lines.s_nom) * 100).data, index=array_line)
        
    load_links = pd.Series((abs(network.links_t.p0.max(
                ) / (network.links.p_nom)) * 100).data, index=array_link)

    max_load = load_lines.append(load_links)

    ll = network.plot(
        line_colors=max_load,
        line_cmap={
            'Line': cmap_line,
            'Link': cmap_link},
        bus_sizes=0,
        title="Maximum of line loading",
        line_widths=2)

    if not boundaries:
        boundaries = [min(max_load), max(max_load)]

    v = np.linspace(boundaries[0], boundaries[1], 101)

    cb = plt.colorbar(ll[1], boundaries=v,
                      ticks=v[0:101:10])

    cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])

    cb_Link = plt.colorbar(ll[2], boundaries=v,
                               ticks=v[0:101:10])
    cb_Link.set_clim(vmin=boundaries[0], vmax=boundaries[1])

    if two_cb:
        # cb_Link.set_label('Maximum load of DC-lines %')
        cb.set_label('Maximum load of AC-lines %')

    else:
        cb.set_label('Maximum load in %')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def load_hours(network, min_load=0.9, max_load=1, boundaries=[0, 8760]):
    
    """Plot number of hours with line loading in selected range. 
    
    Parameters
    ----------
    network: PyPSA network container
        Holds topology of grid including results from powerflow analysis
    min_load: float
        Choose lower bound of relative load 
    max_load: float
        Choose upper bound of relative load
    boundaries: array
       Set boundaries of heatmap axis    
    
    """
    
    cmap_line = plt.cm.jet
    cmap_link = plt.cm.jet
    array_line = [['Line'] * len(network.lines), network.lines.index]

    load_lines = pd.Series(((abs(network.lines_t.p0[(
        abs(network.lines_t.p0.mul(network.snapshot_weightings, axis=0)) / 
        network.lines.s_nom_opt >= min_load) &
                                                    (
        abs(network.lines_t.p0.mul(network.snapshot_weightings, axis=0)) / 
        network.lines.s_nom_opt <= max_load)]) /
                            abs(network.lines_t.p0[(
                                abs(network.lines_t.p0) /
                                network.lines.s_nom_opt >= min_load) &
                                (abs(network.lines_t.p0) /
                                 network.lines.s_nom_opt <= max_load)]))
                            .sum()).data, index=array_line)

    array_link = [['Link'] * len(network.links), network.links.index]

    load_links = pd.Series(((abs(network.links_t.p0[(
        abs(network.links_t.p0.mul(network.snapshot_weightings, axis=0)) / 
        network.links.p_nom_opt >= min_load) &
                                                    (
        abs(network.links_t.p0.mul(network.snapshot_weightings, axis=0)) / 
        network.links.p_nom_opt <= max_load)]) /
                             abs(network.links_t.p0[(
                                abs(network.links_t.p0) /
                                network.links.p_nom_opt >= min_load) &
                                (abs(network.links_t.p0) /
                                 network.links.p_nom_opt <= max_load)]))
                            .sum()).data, index=array_link)

    load_hours = load_lines.append(load_links)

    ll = network.plot(
        line_colors=load_hours,
        line_cmap={
            'Line': cmap_line,
            'Link': cmap_link},
        bus_sizes=0,
        title="Number of hours with more then 90% load",
        line_widths=2)

    v1 = np.linspace(boundaries[0], boundaries[1], 101)
    v = np.linspace(boundaries[0], boundaries[1], 101)
    cb_Link = plt.colorbar(ll[2], boundaries=v1,
                           ticks=v[0:101:10])
    cb_Link.set_clim(vmin=boundaries[0], vmax=boundaries[1])

    cb = plt.colorbar(ll[1], boundaries=v,
                      ticks=v[0:101:10])
    cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])

    cb.set_label('Number of hours')


def plot_residual_load(network):
    """ Plots residual load summed of all exisiting buses.

    Parameters
    ----------
    network : PyPSA network containter
    """

    renewables = network.generators[
        network.generators.carrier.isin(['wind_onshore', 'wind_offshore', 
                                         'solar', 'run_of_river',
                                         'wind'])]
    renewables_t = network.generators.p_nom[renewables.index] * \
        network.generators_t.p_max_pu[renewables.index].mul(
                network.snapshot_weightings, axis=0)
    load = network.loads_t.p_set.mul(network.snapshot_weightings, axis=0).\
    sum(axis=1)
    all_renew = renewables_t.sum(axis=1)
    residual_load = load - all_renew
    plot = residual_load.plot(
        title = 'Residual load',
        drawstyle='steps',
        lw=2,
        color='red',
        legend=False)
    plot.set_ylabel("MW")
    # sorted curve
    sorted_residual_load = residual_load.sort_values(
        ascending=False).reset_index()
    plot1 = sorted_residual_load.plot(
            title='Sorted residual load',
            drawstyle='steps',
            lw=2,
            color='red',
            legend=False)
    plot1.set_ylabel("MW")



def plot_stacked_gen(network, bus=None, resolution='GW', filename=None):
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
    if resolution == 'GW':
        reso_int = 1e3
    elif resolution == 'MW':
        reso_int = 1
    elif resolution == 'KW':
        reso_int = 0.001

    # sum for all buses
    if bus is None:
        p_by_carrier = pd.concat([network.generators_t.p[network.generators
                         [network.generators.control != 'Slack'].index],
                         network.generators_t.p.mul(
                         network.snapshot_weightings, axis=0)
                         [network.generators[network.generators.control == 
                         'Slack'].index]
                         .iloc[:, 0].apply(lambda x: x if x > 0 else 0)],
                         axis=1)\
                         .groupby(network.generators.carrier, axis=1).sum()
        load = network.loads_t.p.sum(axis=1)
        if hasattr(network, 'foreign_trade'):
            trade_sum = network.foreign_trade.sum(axis=1)
            p_by_carrier['imports'] = trade_sum[trade_sum > 0]
            p_by_carrier['imports'] = p_by_carrier['imports'].fillna(0)
    # sum for a single bus
    elif bus is not None:
        filtered_gens = network.generators[network.generators['bus'] == bus]
        p_by_carrier = network.generators_t.p.mul(network.snapshot_weightings,
            axis=0).groupby(filtered_gens.carrier, axis=1).abs().sum()
        filtered_load = network.loads[network.loads['bus'] == bus]
        load = network.loads_t.p.mul(network.snapshot_weightings, axis=0)\
            [filtered_load.index]

    colors = coloring()
#    TODO: column reordering based on available columns

    fig, ax = plt.subplots(1, 1)

    fig.set_size_inches(12, 6)
    colors = [colors[col] for col in p_by_carrier.columns]
    if len(colors) == 1:
        colors = colors[0]
    (p_by_carrier / reso_int).plot(kind="area", ax=ax, linewidth=0,
                                   color=colors)
    (load / reso_int).plot(ax=ax, legend='load', lw=2, color='darkgrey',
                           style='--')
    ax.legend(ncol=4, loc="upper left")

    ax.set_ylabel(resolution)
    ax.set_xlabel("")
    

    matplotlib.rcParams.update({'font.size': 22})



    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def plot_gen_diff(
    networkA,
    networkB,
    leave_out_carriers=[
        'geothermal',
        'oil',
        'other_non_renewable',
        'reservoir',
        'waste']):
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
        gen = pd.concat([network.generators_t.p.mul(
                        network.snapshot_weightings, axis=0)[network.generators
                        [network.generators.control != 'Slack'].index],
                        network.generators_t.p.mul(
                        network.snapshot_weightings, axis=0)[network.generators
                        [network. generators.control == 'Slack'].index]
                        .iloc[:, 0].apply(lambda x: x if x > 0 else 0)],
                        axis=1)\
                        .groupby(network.generators.carrier,axis=1).sum()
        return gen

    gen = gen_by_c(networkB)
    gen_switches = gen_by_c(networkA)
    diff = gen_switches - gen

    colors = coloring()
    diff.drop(leave_out_carriers, axis=1, inplace=True)
    colors = [colors[col] for col in diff.columns]

    plot = diff.plot(kind='line', color=colors, use_index=False)
    plot.legend(loc='upper left', ncol=5, prop={'size': 8})
    x = []
    for i in range(0, len(diff)):
        x.append(i)
    plt.xticks(x, x)
    plot.set_xlabel('Timesteps')
    plot.set_ylabel('Difference in Generation in MW')
    plot.set_title('Difference in Generation')
    plt.tight_layout()


def plot_voltage(network, boundaries=[]):
    """
    Plot voltage at buses as hexbin


    Parameters
    ----------
    network : PyPSA network container
    boundaries: list of 2 values, setting the lower and upper bound of colorbar

    Returns
    -------
    Plot
    """

    x = np.array(network.buses['x'])
    y = np.array(network.buses['y'])

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
    cb.set_label('Voltage Magnitude per unit of v_nom')

    network.plot(
        ax=ax, line_widths=pd.Series(0.5, network.lines.index), bus_sizes=0)
    plt.show()


def curtailment(network, carrier='solar', filename=None):
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
    p_by_carrier = network.generators_t.p.groupby\
        (network.generators.carrier, axis=1).sum()
    capacity = network.generators.groupby("carrier").sum().at[carrier, "p_nom"]
    p_available = network.generators_t.p_max_pu.multiply(
            network.generators["p_nom"])
    p_available_by_carrier = p_available.groupby(
        network.generators.carrier, axis=1).sum()
    p_curtailed_by_carrier = p_available_by_carrier - p_by_carrier
    print(p_curtailed_by_carrier.sum())
    p_df = pd.DataFrame({carrier +
                         " available": p_available_by_carrier[carrier],
                         carrier +
                         " dispatched": p_by_carrier[carrier], carrier +
                         " curtailed": p_curtailed_by_carrier[carrier]})

    p_df[carrier + " capacity"] = capacity
    p_df[carrier + " curtailed"][p_df[carrier + " curtailed"] < 0.] = 0.

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 6)
    p_df[[carrier + " dispatched", carrier + " curtailed"]
         ].plot(kind="area", ax=ax, linewidth=3)
    p_df[[carrier + " available", carrier + " capacity"]
         ].plot(ax=ax, linewidth=3)

    ax.set_xlabel("")
    ax.set_ylabel("Power [MW]")
    ax.set_ylim([0, capacity * 1.1])
    ax.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def storage_distribution(network, scaling=1, filename=None):
    """
    Plot storage distribution as circles on grid nodes

    Displays storage size and distribution in network.
    
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename : str
        Specify filename
        If not given, figure will be show directly
    """

    stores = network.storage_units
    storage_distribution = network.storage_units.p_nom_opt[stores.index]\
            .groupby(network.storage_units.bus)\
            .sum().reindex(network.buses.index, fill_value=0.)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)

    msd_max = storage_distribution.max()
    msd_median = storage_distribution[storage_distribution != 0].median()
    msd_min = storage_distribution[storage_distribution > 1].min()

    if msd_max != 0:
        LabelVal = int(log10(msd_max))
    else:
        LabelVal = 0
    if LabelVal < 0:
        LabelUnit = 'kW'
        msd_max, msd_median, msd_min = msd_max * \
            1000, msd_median * 1000, msd_min * 1000
        storage_distribution = storage_distribution * 1000
    elif LabelVal < 3:
        LabelUnit = 'MW'
    else:
        LabelUnit = 'GW'
        msd_max, msd_median, msd_min = msd_max / \
            1000, msd_median / 1000, msd_min / 1000
        storage_distribution = storage_distribution / 1000

    if sum(storage_distribution) == 0:
        network.plot(bus_sizes=0, ax=ax, title="No storages")
    else:
        network.plot(
            bus_sizes=storage_distribution * scaling,
            ax=ax,
            line_widths=0.3,
            title="Storage distribution")

    # Here we create a legend:
    # we'll plot empty lists with the desired size and label
    for area in [msd_max, msd_median, msd_min]:
        plt.scatter([], [], c='white', s=area * scaling,
                    label='= ' + str(round(area, 0)) + LabelUnit + ' ')
    plt.legend(scatterpoints=1, labelspacing=1, title='Storage size')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()



def storage_expansion(network, basemap=True, scaling=1, filename=None):
    """
    Plot storage distribution as circles on grid nodes
    Displays storage size and distribution in network.
    
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename : str
        Specify filename
        If not given, figure will be show directly
    """

    stores = network.storage_units[network.storage_units.carrier == 
                                   'extendable_storage']
    batteries = stores[stores.max_hours == 6]
    hydrogen = stores[stores.max_hours == 168]
    storage_distribution =\
        network.storage_units.p_nom_opt[stores.index].groupby(
        network.storage_units.bus).sum().reindex(
                network.buses.index, fill_value=0.)
    battery_distribution =\
        network.storage_units.p_nom_opt[batteries.index].groupby(
        network.storage_units.bus).sum().reindex(
                network.buses.index, fill_value=0.)
    hydrogen_distribution =\
        network.storage_units.p_nom_opt[hydrogen.index].groupby(
        network.storage_units.bus).sum().reindex(
                network.buses.index, fill_value=0.)

    sbatt = network.storage_units.index[
        (network.storage_units.p_nom_opt > 1) & (
                network.storage_units.capital_cost > 10) & (
                    network.storage_units.max_hours == 6)]
    shydr = network.storage_units.index[
        (network.storage_units.p_nom_opt > 1) & (
                network.storage_units.capital_cost > 10) & (
                    network.storage_units.max_hours == 168)]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)
    
    msd_max = storage_distribution.max()
    msd_max_bat = battery_distribution.max()
    msd_max_hyd = hydrogen_distribution.max()

    if msd_max != 0:
        LabelVal = int(log10(msd_max))
    else:
        LabelVal = 0
    if LabelVal < 0:
        LabelUnit = 'kW'
        msd_max, msd_max_bat, msd_max_hyd = msd_max * \
            1000, msd_max_bat * 1000, msd_max_hyd * 1000
        battery_distribution = battery_distribution * 1000
        hydrogen_distribution = hydrogen_distribution * 1000
    elif LabelVal < 3:
        LabelUnit = 'MW'
    else:
        LabelUnit = 'GW'
        msd_max, msd_max_bat, msd_max_hyd = msd_max / \
            1000, msd_max_bat / 1000, msd_max_hyd / 1000
        battery_distribution = battery_distribution / 1000
        hydrogen_distribution = hydrogen_distribution / 1000

    if network.storage_units.p_nom_opt[sbatt].sum() < 1 and\
    network.storage_units.p_nom_opt[shydr].sum() < 1:
        print("No storage unit to plot")
    elif network.storage_units.p_nom_opt[sbatt].sum() > 1 and\
    network.storage_units.p_nom_opt[shydr].sum() < 1:
        network.plot(bus_sizes=battery_distribution * scaling, 
                     bus_colors='orangered', ax=ax, line_widths=0.3)
    elif network.storage_units.p_nom_opt[sbatt].sum() < 1 and\
    network.storage_units.p_nom_opt[shydr].sum() > 1:
        network.plot(bus_sizes=hydrogen_distribution * scaling,
                     bus_colors='teal', ax=ax, line_widths=0.3)
    else:
        network.plot(bus_sizes=battery_distribution * scaling,
                     bus_colors='orangered', ax=ax, line_widths=0.3)
        network.plot(bus_sizes=hydrogen_distribution * scaling,
                     bus_colors='teal', ax=ax, line_widths=0.3)

    if basemap and basemap_present:
        x = network.buses["x"]
        y = network.buses["y"]
        x1 = min(x)
        x2 = max(x)
        y1 = min(y)
        y2 = max(y)

        bmap = Basemap(resolution='l', epsg=network.srid, llcrnrlat=y1, urcrnrlat=y2, llcrnrlon=x1, urcrnrlon=x2, ax=ax)
        bmap.drawcountries()
        bmap.drawcoastlines()

    if msd_max_hyd !=0:
        plt.scatter([], [], c='teal', s=msd_max_hyd * scaling,
                label='= ' + str(round(msd_max_hyd, 0)) + LabelUnit + ' hydrogen storage')
    if msd_max_bat !=0:
        plt.scatter([], [], c='orangered', s=msd_max_bat * scaling,
                label='= ' + str(round(msd_max_bat, 0)) + LabelUnit + ' battery storage')
    plt.legend(scatterpoints=1, labelspacing=1, title='Storage size and technology', borderpad=1.3, loc=2)
    ax.set_title("Storage expansion")

    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

    return


def gen_dist(
        network,
        techs=None,
        snapshot=0,
        n_cols=3,
        gen_size=0.2,
        filename=None):
    """
    Generation distribution
    
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
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
        techs = network.generators.carrier.unique()
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

        gens = network.generators[network.generators.carrier == tech]
        gen_distribution = network.generators_t.p.mul(network.
                                    snapshot_weightings, axis=0)[gens.index].\
                                    loc[network.snapshots[snapshot]].groupby(
                                    network.generators.bus).sum().reindex(
                                     network.buses.index, fill_value=0.)

        network.plot(
            ax=ax,
            bus_sizes=gen_size *
            gen_distribution,
            line_widths=0.1)

        ax.set_title(tech)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def gen_dist_diff(
        networkA,
        networkB,
        techs=None,
        snapshot=0,
        n_cols=3,
        gen_size=0.2,
        filename=None,
        buscmap=plt.cm.jet):
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

        gen_distribution =\
            networkA.generators_t.p.mul(networkA.snapshot_weightings, axis=0)\
            [gensA.index].loc[networkA.snapshots[snapshot]].groupby(
            networkA.generators.bus).sum().reindex(
            networkA.buses.index, fill_value=0.) -\
            networkB.generators_t.p.mul(networkB.snapshot_weightings, axis=0)\
            [gensB.index].loc[networkB.snapshots[snapshot]].groupby(
            networkB.generators.bus).sum().reindex(
            networkB.buses.index, fill_value=0.)

        networkA.plot(
            ax=ax,
            bus_sizes=gen_size * abs(gen_distribution),
            bus_colors=gen_distribution,
            line_widths=0.1,
            bus_cmap=buscmap)

        ax.set_title(tech)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def gen_dist(
        network,
        techs=None,
        snapshot=1,
        n_cols=3,
        gen_size=0.2,
        filename=None):
    """
    Generation distribution

    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
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
        techs = network.generators.carrier.unique()
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

        gens = network.generators[network.generators.carrier == tech]
        gen_distribution = network.generators_t.p.mul(network.
                snapshot_weightings, axis=0)\
                [gens.index].loc[network.snapshots[snapshot]].groupby(
                        network.generators.bus).sum().reindex(
                                network.buses.index, fill_value=0.)

        network.plot(
            ax=ax,
            bus_sizes=gen_size * gen_distribution,
            line_widths=0.1)

        ax.set_title(tech)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def nodal_gen_dispatch(
        network,
        networkB=None,
        techs=['wind_onshore', 'solar'],
        item='energy',
        direction=None,
        scaling=1,
        filename=None):
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
    """   
    
    if techs:
        gens = network.generators[network.generators.carrier.isin(techs)]
    elif techs is None:
        gens = network.generators
        techs = gens.carrier.unique()
    if item == 'capacity':
        dispatch = gens.p_nom.groupby([network.generators.bus, 
                                            network.generators.carrier]).sum()
    elif item == 'energy':
        if networkB:
            dispatch_network =\
                    network.generators_t.p[gens.index].mul(
                            network.snapshot_weightings, axis=0).groupby(
                    [network.generators.bus, network.generators.carrier], 
                    axis=1).sum()
            dispatch_networkB =\
                    networkB.generators_t.p[gens.index].mul(
                            networkB.snapshot_weightings, axis=0).groupby(
                    [networkB.generators.bus, networkB.generators.carrier], 
                    axis=1).sum()
            dispatch = dispatch_network - dispatch_networkB
            
            if direction == 'positive':
                dispatch = dispatch[dispatch > 0].fillna(0)
            elif direction == 'negative':
                dispatch = dispatch[dispatch < 0].fillna(0)
            elif direction == 'absolute':
                pass
            else:
                return('No valid direction given.')
            dispatch = dispatch.sum()
            
        elif networkB is None:
            dispatch =\
                    network.generators_t.p[gens.index].mul(
                            network.snapshot_weightings, axis=0).sum().groupby(
                    [network.generators.bus, network.generators.carrier]).sum()
    
    fig, ax = plt.subplots(1, 1)          
    scaling = 1/(max(abs(dispatch.groupby(level=0).sum())))*scaling
    if direction != 'absolute':
        colors = coloring()
        subcolors = {a: colors[a] for a in techs}
        dispatch = dispatch.abs() + 1e-9
    else:
        dispatch = dispatch.sum(level=0)
        colors = {s[0]: 'green' if s[1] > 0 else 'red' 
                  for s in dispatch.iteritems()}
        dispatch = dispatch.abs()
        subcolors = {'negative': 'red', 'positive': 'green'}
    
    network.plot(
            bus_sizes=dispatch * scaling,
            bus_colors=colors,
            line_widths=0.2,
            margin=0.01,
            ax=ax)

    fig.subplots_adjust(right=0.8)
    plt.subplots_adjust(wspace=0, hspace=0.001)

    patchList = []
    for key in subcolors:
        data_key = mpatches.Patch(color=subcolors[key], label=key)
        patchList.append(data_key)

    ax.legend(handles=patchList, loc='upper left')
    ax.autoscale()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

    return

def nodal_production_balance(
        network, 
        snapshot='all', 
        scaling=0.00001, 
        filename=None):
    """
    Plots the nodal difference between generation and consumption.
    
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    snapshot : int or 'all'
        Snapshot to plot.
        default 'all'
    scaling : int
        Scaling to change plot sizes.
        default 0.0001
    filename : path to folder
    
    """
    fig, ax = plt.subplots(1, 1)
    gen = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()
    load = network.loads_t.p.groupby(network.loads.bus, axis=1).sum()
    
    if snapshot == 'all':
        diff = (gen - load).sum()
    else:
        timestep = network.snapshots[snapshot]
        diff = (gen - load).loc[timestep]
    
    colors = {s[0]: 'green' if s[1] > 0 else 'red' 
                  for s in diff.iteritems()}
    subcolors = {'Net Consumer': 'red', 'Net Producer': 'green'}
    diff = diff.abs()
    network.plot(
            bus_sizes=diff * scaling,
            bus_colors=colors,
            line_widths=0.2,
            margin=0.01,
            ax=ax)
    
    patchList = []
    for key in subcolors:
        data_key = mpatches.Patch(color=subcolors[key], label=key)
        patchList.append(data_key)

    ax.legend(handles=patchList, loc='upper left')
    ax.autoscale()
    if filename:
        plt.savefig(filename)
        plt.close()
        
    return 

def storage_p_soc(network, mean='1H', filename = None):
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

    sbatt = network.storage_units.index[(network.storage_units.p_nom_opt > 1)
        & (network.storage_units.capital_cost > 10) &
        (network.storage_units.max_hours == 6)]
    shydr = network.storage_units.index[(network.storage_units.p_nom_opt > 1) 
        & (network.storage_units.capital_cost > 10) & 
        (network.storage_units.max_hours == 168)]

    cap_batt = (network.storage_units.max_hours[sbatt] *
                network.storage_units.p_nom_opt[sbatt]).sum()
    cap_hydr = (network.storage_units.max_hours[shydr] * 
                network.storage_units.p_nom_opt[shydr]).sum()

    fig, ax = plt.subplots(1, 1)

    if network.storage_units.p_nom_opt[sbatt].sum() < 1 and \
        network.storage_units.p_nom_opt[shydr].sum() < 1:
        print("No storage unit to plot")

    elif network.storage_units.p_nom_opt[sbatt].sum() > 1 and \
        network.storage_units.p_nom_opt[shydr].sum() < 1:

        (network.storage_units_t.p[sbatt].resample(mean).mean().sum(axis=1) / \
         network.storage_units.p_nom_opt[sbatt].sum()).plot(
                 ax=ax, label="Battery dispatch", color='orangered')
        # instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx()
        ((network.storage_units_t.state_of_charge[sbatt].resample(mean).\
          mean().sum(axis=1) / cap_batt)*100).plot(ax=ax2,
        label="Battery state of charge", color='blue')
    elif network.storage_units.p_nom_opt[sbatt].sum() < 1 and\
        network.storage_units.p_nom_opt[shydr].sum() > 1:
        (network.storage_units_t.p[shydr].resample(mean).mean().sum(axis=1) /\
         network.storage_units.p_nom_opt[shydr].sum()).plot(
                 ax=ax, label="Hydrogen dispatch", color='teal')
        # instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx()
        ((network.storage_units_t.state_of_charge[shydr].resample(mean).\
          mean().sum(axis=1) / cap_hydr)*100).plot(
        ax=ax2, label="Hydrogen state of charge", color='green')
    else:
        (network.storage_units_t.p[sbatt].resample(mean).mean().sum(axis=1) / \
         network.storage_units.p_nom_opt[sbatt].sum()).plot(
                 ax=ax, label="Battery dispatch", color='orangered')
        
        (network.storage_units_t.p[shydr].resample(mean).mean().sum(axis=1) /\
         network.storage_units.p_nom_opt[shydr].sum()).plot(
                 ax=ax, label="Hydrogen dispatch", color='teal')        
        # instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx()
        ((network.storage_units_t.state_of_charge[shydr].resample(mean).\
          mean().sum(axis=1) / cap_hydr)*100).plot(
        ax=ax2, label="Hydrogen state of charge", color='green')
        
        ((network.storage_units_t.state_of_charge[sbatt].resample(mean).\
          mean().sum(axis=1) / cap_batt)*100).plot(
        ax=ax2, label="Battery state of charge", color='blue')

    ax.set_xlabel("")
    ax.set_ylabel("Storage dispatch in p.u. \n <- charge - discharge ->")
    ax2.set_ylabel("Storage state of charge in % ")
    ax2.set_ylim([0, 100])
    ax.set_ylim([-1,1])
    ax.legend(loc=2)
    ax2.legend(loc=1)
    ax.set_title("Storage dispatch and state of charge")
    

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

    return


def storage_soc_sorted(network, filename = None):
    """
    Plots the soc (state-pf-charge) of extendable storages
    
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis

    filename : path to folder
    
    """
    sbatt = network.storage_units.index[(network.storage_units.p_nom_opt>1) &
                                    (network.storage_units.capital_cost>10) & 
                                    (network.storage_units.max_hours==6)]
    shydr = network.storage_units.index[(network.storage_units.p_nom_opt>1) &
                                    (network.storage_units.capital_cost>10)
                                    & (network.storage_units.max_hours==168)]

    cap_batt = (network.storage_units.max_hours[sbatt] * 
                network.storage_units.p_nom_opt[sbatt]).sum()
    cap_hydr = (network.storage_units.max_hours[shydr] *
                network.storage_units.p_nom_opt[shydr]).sum()

    fig, ax = plt.subplots(1, 1)

    if network.storage_units.p_nom_opt[sbatt].sum() < 1 and \
        network.storage_units.p_nom_opt[shydr].sum() < 1:
        print("No storage unit to plot")
    elif network.storage_units.p_nom_opt[sbatt].sum() > 1 and \
        network.storage_units.p_nom_opt[shydr].sum() < 1:
        (network.storage_units_t.p[sbatt].sum(axis=1).sort_values(
                ascending=False).reset_index() / \
        network.storage_units.p_nom_opt[sbatt].sum())[0].plot(
                ax=ax, label="Battery storage", color='orangered')
    elif network.storage_units.p_nom_opt[sbatt].sum() < 1 and \
        network.storage_units.p_nom_opt[shydr].sum() > 1:
        (network.storage_units_t.p[shydr].sum(axis=1).sort_values(
                ascending=False).reset_index() / \
        network.storage_units.p_nom_opt[shydr].sum())[0].plot(
                ax=ax, label="Hydrogen storage", color='teal')
    else:
        (network.storage_units_t.p[sbatt].sum(axis=1).sort_values(
                ascending=False).reset_index() / \
        network.storage_units.p_nom_opt[sbatt].sum())[0].plot(
                ax=ax, label="Battery storage", color='orangered')
        (network.storage_units_t.p[shydr].sum(axis=1).sort_values(
                ascending=False).reset_index() / \
        network.storage_units.p_nom_opt[shydr].sum())[0].plot(
                ax=ax, label="Hydrogen storage", color='teal')

    ax.set_xlabel("")
    ax.set_ylabel("Storage dispatch in p.u. \n <- charge - discharge ->")
    ax.set_ylim([-1.05,1.05])
    ax.legend()
    ax.set_title("Sorted duration curve of storage dispatch")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename,figsize=(3,4),bbox_inches='tight')
        plt.close()

    return
    
    
if __name__ == '__main__':
    pass
