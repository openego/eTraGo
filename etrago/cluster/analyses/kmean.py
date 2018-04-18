# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 23:59:44 2018

@author: Kim
"""

import os
from os import path, listdir
from matplotlib import pyplot as plt
import pandas as pd
from numpy import genfromtxt

#prepare the data 
kmean = genfromtxt('C:\eTraGo\etrago\k_mean_parameter.csv')

abs_err = {}
rel_err = {}
abs_time = {}
rel_time = {}
ks=[]

home = os.path.expanduser('C:/eTraGo/etrago/results')
 
#get the information from the saved results 
for i in kmean:
    i =int(i)
    
    resultspath = os.path.join(home, 'snapshot-clustering-results-cyclic-tsam-k'+str(i))
    clustered_path = path.join(resultspath, 'daily')
    original_path = path.join(resultspath, 'original')
    plot_path = resultspath

    network = pd.read_csv(path.join(original_path, 'network.csv'))
    
    
    for c in listdir(clustered_path): # go through the snapsot_parameters
        if c != 'Z.csv': 
            network_c = pd.read_csv(path.join(clustered_path, c, 'network.csv'))
            abs_err[str(c)] = network_c['objective'].values[0]
            abserr=(abs(network_c['objective'].values[0] -
                                   network['objective'].values[0])) * 100
            rel_err[str(c)] = abserr/ network['objective'].values[0]
            abs_time[str(c)] = float(network_c['time'])
            rel_time[str(c)] = ((float(network['time'])-float(network_c['time'])) /
                                float(network['time'])) * 100
        
    #create a dataframe with the needed results for each kmean        
    results = pd.DataFrame({
                        'abs_err': abs_err,
                        'rel_err': rel_err,
                        'abs_time': abs_time,
                        'rel_time': rel_time})
    results.index = [int(i) for i in results.index]
    results.sort_index(inplace=True)
    #save the dataframe for each kmean
    results.to_csv('kmean'+str(i)+'.csv')
            
#definition of 2 axis plots
def plot_2d(variable, name):    
    fig, ax = plt.subplots()
    #get data for the plots
    for i in kmean:
        data = results.from_csv('kmean'+str(int(i))+'.csv', parse_dates=False)
        data = data.iloc[0:len(data)][variable]
        ax.plot(data,'--',label='kmean '+str(int(i)).format(i=i))
    
    ax.plot()
    ax.set_title("Clustering analysis")
    ax.set_xlabel("snapshot parameters")
    ax.set_ylabel(name)
    plt.legend(loc='best')
    plt.show()
    fig.savefig(path.join(home, name + '.png'))

 #plotting time and objective function
plot_2d('abs_time',name='Absolute time in s')
plot_2d('rel_time',name='Relative time deviation in %')

plot_2d('abs_err',name='Objective function')
plot_2d('rel_err',name='Relative objective function deviation in %')
