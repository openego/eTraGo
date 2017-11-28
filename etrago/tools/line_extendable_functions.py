"""line extendables functions"""
from math import sqrt
import numpy as np

def capacity_factor(network,cap_fac):
    """
    This function is for changing the capacities of lines and 
    transformers.
        
    ########################### input parameters ###########################
    
    network: The whole network, which are to calculate
    cap_fac: cap_fac is a variable for the capacitiy factor.
    
    ########################### output parameters ##########################
    network : The function return the network with new capacities of lines
              and transformers
              
    """
    
    
    network.lines.s_nom = network.lines.s_nom * cap_fac
    network.transformers.s_nom = network.transformers.s_nom * cap_fac
    
    return network


def overload_lines(network):
    
    """
    This function is for finding the overload lines.
    First the loadings of all lines would calculate for each timestep.
    Seconde the maximum loadings of all lines are setting.
    After that it will check the value. If the value is over 100% the line
    will be consider. After finding all lines it will be found the timesteps 
    of the maximum loading of the lines which are considered. 
    Last the timesteps which are the same will be counted and sorted by the 
    greatest number.
            
    ########################### input parameters ###########################
    
    network: The whole network, which are to calculate
    
    ########################### output parameters ##########################
    The function return the maximum loadings of each line (max_loading) and 
    the timesteps which are count in (time_list).
              
    """
    
    # List for counting the maximum value for timesteps  
    # 0 : line keys 
    # 1 : maximum loadings of each line
    # 2 : timestep of maximum loading
    max_loading = [[],[],[]]   
    
    # List for counting the maximum loadings of each timestep
    # 0 : timestep index of line
    # 1 : Number of maximum loadings 
    timesteps = [[],[]]
    
    # The same list like timesteps but for sorting by maximum number of loadings
    time_list =[]
    
    i=0
    while (i<len(network.lines_t.p0.keys())):
        # set current line key
        max_loading[0].append(network.lines_t.p0.keys()[i])
        
        # set maximum loading of the current line
        if(network.lines_t.q0.empty):
            s_current = abs(network.lines_t.p0[max_loading[0][i]])
        else:
            p = network.lines_t.p0[max_loading[0][i]]
            q = network.lines_t.q0[max_loading[0][i]]
            s_current = sqrt(p**2+q**2)
            
        max_loading[1].append(max(abs(s_current)/network.lines.s_nom[max_loading[0][i]]*100))
        
        # Find the timestep of maximum loading
        x= 0
        while(x<len(s_current)):
            if(s_current[x]==(max(abs(s_current)))):
                # set the timestep of maximum loading        
                max_loading[2].append(x)
                break
            else:
                x+=1
        
        # filtering the lines which has a loading of over 100%
        loading = max_loading[1][i]
        time_index = max_loading[2][i]
        
        if (((time_index in timesteps[0]) == True) and (loading >= 100)):
            index = timesteps[0].index(time_index)
            timesteps[1][index] +=1
        elif(loading >= 100):
            timesteps[0].append(time_index)
            timesteps[1].append(1) 
                    
        i+=1
        
    # save the Result in a other way    
    x=0
    while(x<len(timesteps[0])):
        time_list.append([timesteps[0][x],timesteps[1][x]])
        x+=1        
    
    # For sorting the List by the item 1
    def getKey(item):
                return item[1]    
                
    time_list = sorted(time_list,key=getKey)
    
    return max_loading,time_list
        
def overload_trafo(network):
    
    """
    This function is for finding the overload transformators.
    First the loadings of all transformators would calculate for each 
    timestep.
    Seconde the maximum loadings of all lines are setting.
    After that it will check the value. If the value is over 100% the line
    will be consider. After finding all lines it will be found the timesteps 
    of the maximum loading of the transformators which are considered. 
    Last the timesteps which are the same will be counted and sorted by the 
    greatest number.
            
    ########################### input parameters ###########################
    
    network: The whole network, which are to calculate
    
    ########################### output parameters ##########################
    The function return the maximum loadings of each transformators 
    (max_loading) and the timesteps which are count in (time_list).
              
    """
    
    # List for counting the maximum value for timesteps  
    # 0 : trafo keys 
    # 1 : maximum loadings of each trafo
    # 2 : timestep of maximum loading
    max_loading = [[],[],[]]   
    
    # List for counting the maximum loadings of each timestep
    # 0 : timestep index of trafo
    # 1 : Number of maximum loadings 
    timesteps = [[],[]]
    
    # The same list like timesteps but for sorting by maximum number of loadings
    time_list =[]
    
    i=0
    while (i<len(network.transformers_t.p0.keys())):
        # set current trafo key
        max_loading[0].append(network.transformers_t.p0.keys()[i])
        
        # set maximum loading of the current trafo
        if(network.transformers_t.q0.empty):
            s_current = abs(network.transformers_t.p0[max_loading[0][i]])
        else:
            p = network.transformers_t.p0[max_loading[0][i]]
            q = network.transformers_t.q0[max_loading[0][i]]
            s_current = sqrt(p**2+q**2)
            
        max_loading[1].append(max(abs(s_current)/network.transformers.s_nom[max_loading[0][i]]*100))
        
        # Find the timestep of maximum loading
        x= 0
        while(x<len(s_current)):
            if(s_current[x]==(max(abs(s_current)))):
                # set the timestep of maximum loading        
                max_loading[2].append(x)
                break
            else:
                x+=1
        
        # filtering the trafo which has a loading of over 100%
        loading = max_loading[1][i]
        time_index = max_loading[2][i]
        
        if (((time_index in timesteps[0]) == True) and (loading >= 100)):
            index = timesteps[0].index(time_index)
            timesteps[1][index] +=1
        elif(loading >= 100):
            timesteps[0].append(time_index)
            timesteps[1].append(1) 
                    
        i+=1
        
    # save the Result in a other way    
    x=0
    while(x<len(timesteps[0])):
        time_list.append([timesteps[0][x],timesteps[1][x]])
        x+=1        
    
    # For sorting the List by the item 1
    def getKey(item):
                return item[1]    
                
    time_list = sorted(time_list,key=getKey)
    
    return max_loading,time_list
    
    
def set_line_cost(network,time_list,max_loading,cost_1,cost_2,cost_3):
    """ 
    This function set the capital cost of lines which ar extendable.
    This function set at the same time that the choosen lines are extendable
    for the calculation.
    
    ########################### input parameters ###########################
    
    network: The whole network, which are to calculate.
    time_list : List whith all timesteps which are considering for the 
                calculation.
    max_loading: List with all maximum loadings of each line.
                 Index 0 : line keys 
                 Index 1 : maximum loadings of each line
                 Index 2 : timestep of maximum loading
                 
    cost_1 : The capital cost for extendable 110kV-lines 
    cost_2 : The capital cost for extendable 220kV-lines 
    cost_3 : The capital cost for extendable 380kV-lines 
    
    ########################### output parameters ##########################
    The function return the network. In this variable the capital costs of 
    the lines which are concidered are setted.
    It return also to extra lists.
    line_time: List with all timesteps which are considered.
    all_time: is the same list, but is for adding another datas.
              
    """
    
    #save the result in differnt variables
    lines_time=[]
    all_time = []
    
    i = 0
    while(i<len(time_list)):
#        if(i > 0.1*(args['end_h']-args['start_h'])):
#                break
        lines_time.append(time_list[len(time_list)-1-i][0]) 
        all_time.append(time_list[len(time_list)-1-i][0])    
        
        index = [a for a,u in enumerate(max_loading[2]) if u==lines_time[i]]
                
        for k in index:
            if(max_loading[1][k]>100):
                network.lines.s_nom_extendable[max_loading[0][k]] = True
                network.lines.s_nom_min[max_loading[0][k]] = network.lines.s_nom[max_loading[0][k]]
                network.lines.s_nom_max[max_loading[0][k]] = np.inf
                        
                name_bus_0 = network.lines.bus0[max_loading[0][k]]
                name_bus_1 = network.lines.bus1[max_loading[0][k]]
                    
                U_bus_0 = network.buses.v_nom[name_bus_0]
                U_bus_1 = network.buses.v_nom[name_bus_1]
                    
                if(U_bus_0 == U_bus_1):
                    if(U_bus_0 == 110):
                        network.lines.capital_cost[max_loading[0][k]] = \
                            (cost_1*network.lines.length[max_loading[0][k]]/network.lines.s_nom[max_loading[0][k]])/\
                            (90*8760)
                    elif(U_bus_0 == 220):
                        network.lines.capital_cost[max_loading[0][k]] = \
                            (cost_2*network.lines.length[max_loading[0][k]]/network.lines.s_nom[max_loading[0][k]])/\
                            (90*8760)
                    else:
                        network.lines.capital_cost[max_loading[0][k]] = \
                            (cost_3*network.lines.length[max_loading[0][k]]/network.lines.s_nom[max_loading[0][k]])/\
                            (90*8760)
                else:
                    print('Error')
        i+=1

    return network,lines_time,all_time


def set_trafo_cost(network,time_list,max_loading,cost_1,cost_2,cost_3):
    
    """ 
    This function set the capital cost of transformators which ar extendable.
    This function set at the same time that the choosen transformators are 
    extendable for the calculation.
    
    ########################### input parameters ###########################
    
    network: The whole network, which are to calculate.
    time_list : List whith all timesteps which are considering for the 
                calculation.
    max_loading: List with all maximum loadings of each transformators.
                 Index 0 : transformators keys 
                 Index 1 : maximum loadings of each transformators
                 Index 2 : timestep of maximum loading
                 
    cost_1 : The capital cost for extendable 110kV-220kV and 110kV-380kV
             Transformators. 
    cost_2 : The capital cost for extendable 220kV-380kV Transformers 
    cost_3 : The capital cost for extendable Transformers (rest) 
    
    ########################### output parameters ##########################
    The function return the network. In this variable the capital costs of 
    the transformators which are concidered are setted.
    It return also to extra lists.
    trafo_time: List with all timesteps which are considered.
              
    """    
    

    # List of choosen timesteps
    trafo_time=[]
        
    i = 0
    while(i<len(time_list)):
#        if(i > 0.1*(args['end_h']-args['start_h'])):
#            break
        trafo_time.append(time_list[len(time_list)-1-i][0])  
            
        index = [a for a,u in enumerate(max_loading[2]) if u==trafo_time[i]]
        for k in index:
            if(max_loading[1][k]>100):
                network.transformers.s_nom_extendable[max_loading[0][k]] = True
                network.transformers.s_nom_min[max_loading[0][k]] = network.transformers.s_nom[max_loading[0][k]]
                network.transformers.s_nom_max[max_loading[0][k]] = np.inf
                    
                name_bus_0 = network.transformers.bus0[max_loading[0][k]]
                name_bus_1 = network.transformers.bus1[max_loading[0][k]]
            
                U_bus_0 = network.buses.v_nom[name_bus_0]
                U_bus_1 = network.buses.v_nom[name_bus_1]
                
                U_OS = max(U_bus_0,U_bus_1)
                U_US = min(U_bus_0,U_bus_1)
                                    
                if((U_OS == 220 and U_US == 110) or (U_OS == 380 and U_US == 110)):
                    network.transformers.capital_cost[max_loading[0][k]] = \
                        ((cost_1)/(40*8760))
                elif(U_OS == 380 and U_US == 220):
                    network.transformers.capital_cost[max_loading[0][k]] = \
                        ((cost_2)/(40*8760))
                else:
                    network.transformers.capital_cost[max_loading[0][k]] = \
                        ((cost_3)/(40*8760))                      
                    print('Other Transformator' + str(k))
            
        i+=1

    return network,trafo_time








































        
        
        