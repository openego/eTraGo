# -*- coding: utf-8 -*-
"""
"""
from config import clustered_path, original_path, plot_path, root_path
# TODO: Make plot_path import work...don't know why it does not
from os import path, listdir

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
kmean = []

files1 = os.listdir(root_path)

for c in range (1,len(files1)+1):
    #print('c',c)
    path01= root_path + 'ResultsExpansions' + str(c) +'.csv'
    Total = pd.read_csv(path01)
    k= Total.loc[0]['k-mean']
    
    if c == 1:
        results = Total
    else:
        results= results.append(Total)
        print('c=',c)
 
    if (k in kmean) == False:
            kmean.append(k)

print('K-MEAN', kmean)

result1= results.sort_values(['k-mean'], ascending = [1]) 

print('total results', result1)
#result1.to_csv(root_path + 'TotalResults.csv')

for i in range(len(kmean)):
    value = int(kmean[i])
    print('value1', value)
    RS = result1.loc[result1 ['k-mean'] == value]
    print('RS'+ str(i),RS)
    

    ax = RS.plot(x='Unnamed: 0', y='2nd LOPF', title = 'simulation time for k-mean= '+ str(value))
    
    ax.set_ylabel('Simulation Time 2nd LOPF (s)')
    ax.set_xlabel('Number of Snapshots')
    
    #fig = ax.get_figure()
    #fig.savefig(path.join(plot_path, 'simulation time for k-mean= '+ str(value) +'.eps'))
    
