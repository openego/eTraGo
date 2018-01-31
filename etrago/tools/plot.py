"""
Plot.py defines functions necessary to plot results of eTraGo.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation; either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität Flensburg, Centre for Sustainable Energy Systems, DLR-Institute for Networked Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, MarlonSchlemminger, mariusves, lukasol"


import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
import matplotlib
from math import sqrt
if not 'READTHEDOCS' in os.environ:
    from geoalchemy2.shape import to_shape



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
    
def plot_line_loading(network, timestep=0, filename=None, boundaries=[],
                      arrows=False):
    """
    Plot line loading as color on lines

    Displays line loading relative to nominal capacity
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename : str
        Specify filename
        If not given, figure will be show directly
    """
    # TODO: replace p0 by max(p0,p1) and analogously for q0
    # TODO: implement for all given snapshots

    # calculate relative line loading as S/S_nom
    # with S = sqrt(P^2 + Q^2)
    x = time.time()
    cmap = plt.cm.jet
    if network.lines_t.q0.empty:
        loading_c = (network.lines_t.p0.loc[network.snapshots[timestep]]/ \
                   (network.lines.s_nom)) * 100 
        loading = abs(loading_c)
    else:
         loading = ((network.lines_t.p0.loc[network.snapshots[timestep]] ** 2 +
                   network.lines_t.q0.loc[network.snapshots[timestep]] ** 2).\
                   apply(sqrt) / (network.lines.s_nom)) * 100 

    # do the plotting
    ll = network.plot(line_colors=loading, line_cmap=cmap,
                      title="Line loading", line_widths=0.55)
    
    # add colorbar, note mappable sliced from ll by [1]

    if not boundaries:
        cb = plt.colorbar(ll[1])
    elif boundaries:
        v = np.linspace(boundaries[0], boundaries[1], 101)
        cb = plt.colorbar(ll[1], boundaries=v,
                          ticks=v[0:101:10])
        cb.set_clim(vmin=boundaries[0], vmax=boundaries[1])

    cb.set_label('Line loading in %')
    
    if arrows:
        ax = plt.axes()
        path = ll[1].get_segments()
        x_coords_lines = np.zeros([len(path)])
        cmap = cmap
        colors = cmap(ll[1].get_array()/100)
        for i in range(0, len(path)):
            x_coords_lines[i] = network.buses.loc[str(network.lines.iloc[i, 2]),'x']
            color = colors[i]
            if (x_coords_lines[i] == path[i][0][0] and loading_c[i] >= 0):
                arrowprops = dict(arrowstyle="->", color=color)
            else:
                arrowprops = dict(arrowstyle="<-", color=color)
            ax.annotate("",
                        xy=abs((path[i][0] - path[i][1]) * 0.51 - path[i][0]),
                        xytext=abs((path[i][0] - path[i][1]) * 0.49 - path[i][0]),
                        arrowprops=arrowprops,
                        size=10
                        )
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
    
    y = time.time()
    z = (y-x)/60
    print(z)


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
    def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
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
    loading_switches = abs(networkA.lines_t.p0.loc[networkA.snapshots[timestep]].to_frame())
    loading_switches.columns = ['switch']
    loading_noswitches = abs(networkB.lines_t.p0.loc[networkB.snapshots[timestep]].to_frame())
    loading_noswitches.columns = ['noswitch']
    diff_network = loading_switches.join(loading_noswitches)
    diff_network['noswitch'] = diff_network['noswitch'].fillna(diff_network['switch'])
    diff_network[networkA.snapshots[timestep]] = diff_network['switch']-diff_network['noswitch']
    
    # get switches
    new_buses = pd.Series(index=networkA.buses.index.values)
    new_buses.loc[set(networkA.buses.index.values)-set(networkB.buses.index.values)] = 0.1
    new_buses = new_buses.fillna(0)
    
    # plot network with difference in loading and shifted colormap
    loading = (diff_network.loc[:, networkA.snapshots[timestep]]/ \
                       (networkA.lines.s_nom)) * 100
    midpoint = 1 - max(loading)/(max(loading) + abs(min(loading)))
    shifted_cmap = shiftedColorMap(plt.cm.jet, midpoint=midpoint, name='shifted')             
    ll = networkA.plot(line_colors=loading, line_cmap=shifted_cmap,
                          title="Line loading", bus_sizes=new_buses, 
                          bus_colors='blue', line_widths=0.55)
    
    cb = plt.colorbar(ll[1])
    cb.set_label('Difference in line loading in % of s_nom')


def plot_residual_load(network):
    """ Plots residual load summed of all exisiting buses.

    Parameters
    ----------
    network : PyPSA network containter
    """

    renewables = network.generators[
                    network.generators.dispatch == 'variable']
    renewables_t = network.generators.p_nom[renewables.index] * \
                        network.generators_t.p_max_pu[renewables.index]
    load = network.loads_t.p_set.sum(axis=1)
    all_renew = renewables_t.sum(axis=1)
    residual_load = load - all_renew
    residual_load.plot(drawstyle='steps', lw=2, color='red', legend='residual load')
    # sorted curve
    sorted_residual_load = residual_load.sort_values(
                                ascending=False).reset_index()
    sorted_residual_load.plot(drawstyle='steps', lw=1.4, color='red')


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
    if bus==None:    
        p_by_carrier =  pd.concat([network.generators_t.p
                       [network.generators[network.generators.control!='Slack'].index], 
                       network.generators_t.p[network.generators[network.
                       generators.control=='Slack'].index].iloc[:,0].
                       apply(lambda x: x if x > 0 else 0)], axis=1).\
                       groupby(network.generators.carrier, axis=1).sum()
        load = network.loads_t.p.sum(axis=1)
        if hasattr(network, 'foreign_trade'):
            trade_sum = network.foreign_trade.sum(axis=1)
            p_by_carrier['imports'] = trade_sum[trade_sum > 0]
            p_by_carrier['imports'] = p_by_carrier['imports'].fillna(0)
    # sum for a single bus
    elif bus is not None:
        filtered_gens = network.generators[network.generators['bus'] == bus]
        p_by_carrier = network.generators_t.p.\
                       groupby(filtered_gens.carrier, axis=1).sum()
        filtered_load = network.loads[network.loads['bus'] == bus]
        load = network.loads_t.p[filtered_load.index]

    colors = {'biomass':'green',
              'coal':'k',
              'gas':'orange',
              'eeg_gas':'olive',
              'geothermal':'purple',
              'lignite':'brown',
              'oil':'darkgrey',
              'other_non_renewable':'pink',
              'reservoir':'navy',
              'run_of_river':'aqua',
              'pumped_storage':'steelblue',
              'solar':'yellow',
              'uranium':'lime',
              'waste':'sienna',
              'wind':'skyblue',
              'slack':'pink',
              'load shedding': 'red',
              'nan':'m',
              'imports':'salmon',
              '':'m'}

#    TODO: column reordering based on available columns

    fig,ax = plt.subplots(1,1)

    fig.set_size_inches(12,6)
    colors = [colors[col] for col in p_by_carrier.columns]
    if len(colors) == 1:
        colors = colors[0]
    (p_by_carrier/reso_int).plot(kind="area",ax=ax,linewidth=0,
                            color=colors)
    (load/reso_int).plot(ax=ax, legend='load', lw=2, color='darkgrey', style='--')
    ax.legend(ncol=4,loc="upper left")

    ax.set_ylabel(resolution)
    ax.set_xlabel("")
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def plot_gen_diff(networkA, networkB, leave_out_carriers=['geothermal', 'oil',
                                               'other_non_renewable', 
                                               'reservoir', 'waste']):
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
        gen =  pd.concat([network.generators_t.p
                           [network.generators[network.generators.control!='Slack'].index], 
                           network.generators_t.p[network.generators[network.
                           generators.control=='Slack'].index].iloc[:,0].
                           apply(lambda x: x if x > 0 else 0)], axis=1).\
                           groupby(network.generators.carrier, axis=1).sum()
        return gen

    gen = gen_by_c(networkB)
    gen_switches = gen_by_c(networkA)
    diff = gen_switches-gen
    
    colors = {'biomass':'green',
          'coal':'k',
          'gas':'orange',
          'eeg_gas':'olive',
          'geothermal':'purple',
          'lignite':'brown',
          'oil':'darkgrey',
          'other_non_renewable':'pink',
          'reservoir':'navy',
          'run_of_river':'aqua',
          'pumped_storage':'steelblue',
          'solar':'yellow',
          'uranium':'lime',
          'waste':'sienna',
          'wind':'skyblue',
          'slack':'pink',
          'load shedding': 'red',
          'nan':'m'}
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
    
    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(6,4)
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
    
    network.plot(ax=ax,line_widths=pd.Series(0.5,network.lines.index), bus_sizes=0)
    plt.show()

def curtailment(network, carrier='wind', filename=None):
    
    p_by_carrier = network.generators_t.p.groupby(network.generators.carrier, axis=1).sum()
    capacity = network.generators.groupby("carrier").sum().at[carrier,"p_nom"]
    p_available = network.generators_t.p_max_pu.multiply(network.generators["p_nom"])
    p_available_by_carrier =p_available.groupby(network.generators.carrier, axis=1).sum()
    p_curtailed_by_carrier = p_available_by_carrier - p_by_carrier
    p_df = pd.DataFrame({carrier + " available" : p_available_by_carrier[carrier],
                         carrier + " dispatched" : p_by_carrier[carrier],
                         carrier + " curtailed" : p_curtailed_by_carrier[carrier]})
    
    p_df[carrier + " capacity"] = capacity
    p_df[carrier + " curtailed"][p_df[carrier + " curtailed"] < 0.] = 0.
    
    
    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(12,6)
    p_df[[carrier + " dispatched",carrier + " curtailed"]].plot(kind="area",ax=ax,linewidth=3)
    p_df[[carrier + " available",carrier + " capacity"]].plot(ax=ax,linewidth=3)
    
    ax.set_xlabel("")
    ax.set_ylabel("Power [MW]")
    ax.set_ylim([0,capacity*1.1])
    ax.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
        
def storage_distribution(network, filename=None):
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
    storage_distribution = network.storage_units.p_nom_opt[stores.index].groupby(network.storage_units.bus).sum().reindex(network.buses.index,fill_value=0.)

    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(6,6)
   
    if sum(storage_distribution) == 0:
         network.plot(bus_sizes=0,ax=ax,title="No storages")
    else:
         network.plot(bus_sizes=storage_distribution,ax=ax,line_widths=0.3,title="Storage distribution")
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

def storage_expansion(network, filename=None):
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
    
    stores = network.storage_units[network.storage_units.carrier=='extendable_storage']   
    storage_distribution = network.storage_units.p_nom_opt[stores.index].groupby(network.storage_units.bus).sum().reindex(network.buses.index,fill_value=0.)

    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(6,6)
   
    if sum(storage_distribution) == 0:
         network.plot(bus_sizes=0,ax=ax,title="No extendable storage")
    else:
         network.plot(bus_sizes=storage_distribution,ax=ax,line_widths=0.3,title="Storage expansion distribution")
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

def gen_dist(network, techs=None, snapshot=0, n_cols=3,gen_size=0.2, filename=None):

    """
    Generation distribution
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

    fig.set_size_inches(size*n_cols,size*n_rows)

    for i,tech in enumerate(techs):
        i_row = i // n_cols
        i_col = i % n_cols
    
        ax = axes[i_row,i_col]
    
        gens = network.generators[network.generators.carrier == tech]
        gen_distribution = network.generators_t.p[gens.index].\
        loc[network.snapshots[snapshot]].groupby(network.generators.bus).sum().\
        reindex(network.buses.index,fill_value=0.)
    
    
        network.plot(ax=ax,bus_sizes=gen_size*gen_distribution, line_widths=0.1)
    
        ax.set_title(tech)
    if filename is None:
       plt.show()
    else:
       plt.savefig(filename)
plt.close()

def gen_dist_diff(networkA, networkB, techs=None, snapshot=0, n_cols=3,gen_size=0.2, filename=None, buscmap=plt.cm.jet):

    """
    Difference in generation distribution
    Green/Yellow/Red colors mean that the generation at a location is bigger with switches 
    than without
    Blue colors mean that the generation at a location is smaller with switches
    than without
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

    fig.set_size_inches(size*n_cols,size*n_rows)

    for i,tech in enumerate(techs):
        i_row = i // n_cols
        i_col = i % n_cols
    
        ax = axes[i_row,i_col]
    
        gensA = networkA.generators[networkA.generators.carrier == tech]
        gensB = networkB.generators[networkB.generators.carrier == tech]
        
        gen_distribution = networkA.generators_t.p[gensA.index].\
        loc[networkA.snapshots[snapshot]].groupby(networkA.generators.bus).sum().\
        reindex(networkA.buses.index,fill_value=0.) - networkB.generators_t.p[gensB.index].\
        loc[networkB.snapshots[snapshot]].groupby(networkB.generators.bus).sum().\
        reindex(networkB.buses.index,fill_value=0.)
    
        networkA.plot(ax=ax,bus_sizes=gen_size*abs(gen_distribution), 
                      bus_colors=gen_distribution, line_widths=0.1, bus_cmap=buscmap)
    
        ax.set_title(tech)

        
    if filename is None:
       plt.show()
    else:
       plt.savefig(filename)
plt.close()

def gen_dist(network, techs=None, snapshot=1, n_cols=3,gen_size=0.2, filename=None):

    """
    Generation distribution

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

    fig.set_size_inches(size*n_cols,size*n_rows)

    for i,tech in enumerate(techs):
        i_row = i // n_cols
        i_col = i % n_cols
    
        ax = axes[i_row,i_col]
    
        gens = network.generators[network.generators.carrier == tech]
        gen_distribution = network.generators_t.p[gens.index].\
        loc[network.snapshots[snapshot]].groupby(network.generators.bus).sum().\
        reindex(network.buses.index,fill_value=0.)
        
    
    
        network.plot(ax=ax,bus_sizes=gen_size*gen_distribution, line_widths=0.1)
    
        ax.set_title(tech)
    if filename is None:
       plt.show()
    else:
       plt.savefig(filename)
       plt.close()
        
        
if __name__ == '__main__':
    pass
