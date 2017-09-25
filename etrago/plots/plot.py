"""This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.
Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line."""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "tba"


from math import sqrt
from geoalchemy2.shape import to_shape
from matplotlib import pyplot as plt
import pandas as pd


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
    
def plot_line_loading(network, timestep=0, filename=None):
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

    if network.lines_t.q0.empty:
         loading = abs((network.lines_t.p0.loc[network.snapshots[timestep]]/ \
                   (network.lines.s_nom)) * 100 )
    else:
         loading = ((network.lines_t.p0.loc[network.snapshots[timestep]] ** 2 +
                   network.lines_t.q0.loc[network.snapshots[timestep]] ** 2).\
                   apply(sqrt) / (network.lines.s_nom)) * 100 

    # do the plotting
    ll = network.plot(line_colors=abs(loading), line_cmap=plt.cm.jet,
                      title="Line loading")

    # add colorbar, note mappable sliced from ll by [1]
    cb = plt.colorbar(ll[1])
    cb.set_label('Line loading in %')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

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
              'nan':'m'}

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
         network.plot(bus_sizes=0,ax=ax,title="No extendable storage")
    else:
         network.plot(bus_sizes=storage_distribution,ax=ax,line_widths=0.3,title="Storage distribution")
    
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

def load_dist(network, snapshot=0, filename=None):

    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(6,6)
    load_distribution = network.loads_t.p_set.loc[network.snapshots[snapshot]].groupby(network.loads.bus).sum()
    load_distribution_q = network.loads_t.q_set.loc[network.snapshots[snapshot]].groupby(network.loads.bus).sum()
    network.plot(bus_sizes=load_distribution,ax=ax,legend='active load')
    network.plot(bus_colors='r',bus_sizes=load_distribution_q,ax=ax,legend='reactive load', title="Load distribution")
    ax.legend(ncol=1,loc="upper left")

    if filename is None:
       plt.show()
    else:
       plt.savefig(filename)
       plt.close()

def v_mag_minmax(network, filename=None):
    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(6,6)
    network.buses_t.v_mag_pu.min().plot(ax=ax, legend='minimal')
    network.buses_t.v_mag_pu.max().plot(ax=ax, legend='maximal', title= 'Extreme Voltage Magnitudes (p.u) per bus throughout simulation time' )
    if filename is None:
       plt.show()
    else:
       plt.savefig(filename)
       plt.close()

    
if __name__ == '__main__':
    pass
