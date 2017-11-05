from math import sqrt
from matplotlib import pyplot as plt
            
def plot_max_line_loading(network,filename=None):
    """
    Plot line loading as color on lines
    Displays line loading relative to nominal capacity
    Parameters
    ----------
    network : PyPSA network container
    Holds topology of grid including results from powerflow analysis
    filename : str
    Specify filename
    If not given, figure will be show directly"""
    
    # with S = sqrt(P^2 + Q^2)
    
    loading=[]
    i=0
    while(i<len(network.lines)):
        if network.lines_t.q0.empty:
            p = max(abs(network.lines_t.p0[network.lines_t.p0.keys()[i]]))
            s_nom = network.lines.s_nom[network.lines_t.p0.keys()[i]]
            
            loading.append(p/s_nom*100)
        else:
            p = max(abs(network.lines_t.p0[network.lines_t.p0.keys()[i]]))
            q = max(abs(network.lines_t.q0[network.lines_t.q0.keys()[i]]))
            s_nom = network.lines.s_nom[network.lines_t.p0.keys()[i]]
            
            loading.append(sqrt(p**2+q**2)/s_nom*100)
        i+=1
                
    # do the plotting
    ll = network.plot(line_colors=loading, line_cmap=plt.cm.jet,
                      title="Line maximum loading")

    # add colorbar, note mappable sliced from ll by [1]
    cb = plt.colorbar(ll[1])
    cb.set_label('Line loading in %')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
        
        
def plot_max_opt_line_loading_bench(network,filename=None):
    
    # For Benchmark calculation    
    
    """
    Plot optimal line loading as color on lines
    Displays line loading relative to nominal capacity
    Parameters
    ----------
    network : PyPSA network container
    Holds topology of grid including results from powerflow analysis
    filename : str
    Specify filename
    If not given, figure will be show directly"""
    
    # with S = sqrt(P^2 + Q^2)
    
    loading=[]
    i=0
    while(i<len(network.lines)):
        if network.lines_t.q0.empty:
            p = max(abs(network.lines_t.p0[network.lines_t.p0.keys()[i]]))
            s_nom = network.lines.s_nom_opt[network.lines_t.p0.keys()[i]]
            
            loading.append(p/s_nom*100)
        else:
            p = max(abs(network.lines_t.p0[network.lines_t.p0.keys()[i]]))
            q = max(abs(network.lines_t.q0[network.lines_t.q0.keys()[i]]))
            s_nom = network.lines.s_nom_opt[network.lines_t.p0.keys()[i]]
            
            loading.append(sqrt(p**2+q**2)/s_nom*100)
        i+=1
                
    # do the plotting
    ll = network.plot(line_colors=loading, line_cmap=plt.cm.jet,
                      title="Line maximum loading")

    # add colorbar, note mappable sliced from ll by [1]
    cb = plt.colorbar(ll[1])
    cb.set_label('Line loading in %')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
        
def plot_max_opt_line_loading(network,line_time,filename=None):
    """
    Plot optimal line loading as color on lines
    Displays line loading relative to nominal capacity
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename : str
        Specify filename
        If not given, figure will be show directly
    """
    
    # with S = sqrt(P^2 + Q^2)
    loading=[]
    i=0
    while(i<len(network.lines)):
        p = []
        q = []
        x=0
        while(x<len(line_time)):
            p.append(abs(network.lines_t.p0[network.lines_t.p0.keys()[i]].loc[network.snapshots[line_time[x]]]))
            if network.lines_t.q0.empty:
                q.append(0)
            else:
                q.append(abs(network.lines_t.q0[network.lines_t.q0.keys()[i]].loc[network.snapshots[line_time[x]]]))
                
            x+=1
            
        max_p = max(p)
        max_q = max(q)
            
        s_nom = network.lines.s_nom_opt[network.lines_t.p0.keys()[i]]
        loading.append(sqrt(max_p**2+max_q**2)/s_nom*100)
            
        i+=1

    # do the plotting
    ll = network.plot(line_colors=loading, line_cmap=plt.cm.jet,
                      title="Line maximum loading")

    # add colorbar, note mappable sliced from ll by [1]
    cb = plt.colorbar(ll[1])
    cb.set_label('Line loading in %')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
        

        
def transformers_distribution(network, filename=None):
    """
    Plot transformers distribution as circles on grid nodes
    Displays storage size and distribution in network.
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename : str
        Specify filename
        If not given, figure will be show directly
    """
    
    transformers = network.transformers   
    transformers_distribution = (network.transformers.s_nom_opt-network.transformers.s_nom)[transformers.index].groupby(network.transformers.bus0).sum().reindex(network.buses.index,fill_value=0.)

    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(6,6)
   
    if sum(transformers_distribution) == 0:
         network.plot(bus_sizes=0,ax=ax,title="Transformers distribution")
    else:
         network.plot(bus_sizes=transformers_distribution,ax=ax,line_widths=0.3,title="Transformers distribution")
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
                    
                    
def plot_dif_line_MW(network,filename=None):
    """
    Plot the addition capacity which are calculated for lines as color on lines
    Displays addition capacity
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename : str
        Specify filename
        If not given, figure will be show directly
    """    
    
    
    dif=[]
    i=0
    while(i<len(network.lines)):
        dif.append(network.lines.s_nom_opt[i]-network.lines.s_nom[i])            
        i+=1

    # do the plotting
    ll = network.plot(line_colors=dif, line_cmap=plt.cm.jet,
                      title="Line_distribution")

    # add colorbar, note mappable sliced from ll by [1]
    cb = plt.colorbar(ll[1])
    cb.set_label('Delta S in MW')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()    
        

def plot_dif_line_percent(network,filename=None):
    """
    Plot the addition capacity which are calculated for lines as color on lines
    Displays addition capacity in percent
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    filename : str
        Specify filename
        If not given, figure will be show directly
    """    
    dif=[]
    i=0
    while(i<len(network.lines)):
        dif_ = network.lines.s_nom_opt[i]-network.lines.s_nom[i]
        if dif_ == 0:
            dif.append(0)
        else:
            dif.append(round(dif_/network.lines.s_nom[i]*100))            
        i+=1

    # do the plotting
    ll = network.plot(line_colors=dif, line_cmap=plt.cm.jet,
                      title="Line_distribution in percent")

    # add colorbar, note mappable sliced from ll by [1]
    cb = plt.colorbar(ll[1])
    cb.set_label('Delta S in %')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()    