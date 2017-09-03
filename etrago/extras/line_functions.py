from math import sqrt
from egopowerflow.tools.plot import add_coordinates

def set_line_network(scenario):
    
    network = scenario.build_network()
    # add coordinates
    network = add_coordinates(network)

    return network
    
def network_evaluation(network,end_h,start_h):

    # variable for all capital costs    
    capital_cost = []
    
    # variable for all line keys which ar extendable
    line_keys = []
    
    # detection the len of p and q
    timestep = 0        
    p_len = len(network.lines_t.p0.loc[network.snapshots[timestep]])
    q_len = len(network.lines_t.q0.loc[network.snapshots[timestep]])
    max_len = max(p_len,q_len)
                
    # Saving all keys in a varible
    keys = network.lines_t.p0.keys()
        
    #Calculation the number of steps
    steps = end_h-start_h
    
    i = 0
    while(i<=max_len):
        
        timestep = 0
        # Create a variable for each step to find the maximum loading
        percent_max = ['NaN',0]
        
        #Start to find the maximum loading of selected line
        while(timestep <= steps):
            p = network.lines_t.p0.loc[network.snapshots[timestep]]
            q = network.lines_t.q0.loc[network.snapshots[timestep]]
            
            if(i < q_len):
                loading = (sqrt(p[i]**2 + q[i]**2)/network.lines.s_nom[i])*100
            else:
                loading = (sqrt(p[i]**2 + 0**2)/network.lines.s_nom[i])*100
               
            percent_max[0]=keys[i]
            
            if(loading> percent_max[1]):
                percent_max[1]=loading
                   
            if(percent_max[1] > 70):
                network.lines.s_nom_extendable[percent_max[0]] = True
                network.lines.s_nom_min[percent_max[0]]=\
                    network.lines.s_nom[percent_max[0]]
                network.lines.s_nom_max[percent_max[0]]=\
                    network.lines.s_nom[percent_max[0]]*100000000
                
#                U = network.buses.v_nom.length[percent_max[0]]
#                l = network.lines.length[percent_max[0]]
#                x = network.lines.x[percent_max[0]]
#                r = network.lines.r[percent_max[0]]
#                Z = sqrt(x**2 + r**2)
#                S = network.lines.s_nom[percent_max[0]]
                
                # To calculate the capital cost in €/MVA
                capital_cost.append(20000)
                network.lines.capital_cost[percent_max[0]] = 20000
                
                # To save all line keys which are can extended
                line_keys.append(percent_max[0])
                break
            else:
                network.lines.s_nom_extendable[percent_max[0]] = False
                network.lines.s_nom_min[percent_max[0]]=\
                    network.lines.s_nom[percent_max[0]]
                network.lines.s_nom_max[percent_max[0]]=\
                    network.lines.s_nom[percent_max[0]]
                timestep+=1
        i+=1
    
    i = 0
    timestep = 0
    while(i<len(network.generators.p_nom)):
        key = network.generators_t.p.keys()[i]
        p = network.generators_t.p[key]
        k=0
        p_min=[]
        while(k<len(p)):
            p_min.append(p[k]/network.generators.p_nom[key])
            k+=1
        
        network.generators.p_min_pu[key]=min(p_min)
        network.generators.p_max_pu[key]=max(p_min)
        i+=1
        
    return line_keys,capital_cost,network
                
            


                
                
            