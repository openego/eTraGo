# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:37:49 2018

@author: Kim
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os import path, listdir
from numpy import genfromtxt
import os

kmean = [2,5,10] #genfromtxt('C:\eTraGo\etrago\k_mean_parameter.csv')

abs_obj =[]
rel_obj =[]
abs_time =[]
rel_time =[]
snapshot=[]
k_parameter=[]

home = os.path.expanduser('C:/eTraGo/etrago/results')

#set the time of the highest kmean(without snapshots) as the benchmark 
max_k = int(max(kmean))
resultspath = os.path.join(home, 'snapshot-clustering-results-cyclic-tsam-k'+str(max_k))
original_path = path.join(resultspath, 'original')
network = pd.read_csv(path.join(original_path, 'network.csv'))
benchmark_time =float(network['time'])
#set the ojective function of the highest kmean(without snapshots) as the benchmark
benchmark_objective = network['objective'].values[0]


for i in kmean:
    i =int(i)
    resultspath = os.path.join(home, 'snapshot-clustering-results-cyclic-tsam-k'+str(i))
    clustered_path = path.join(resultspath, 'daily')

    network = pd.read_csv(path.join(original_path, 'network.csv'))

    for c in listdir(clustered_path):      
        if c != 'Z.csv':
            network_c = pd.read_csv(path.join(clustered_path, c, 'network.csv'))

            abs_obj.append(network_c['objective'].values[0])
            rel_obj.append((abs(benchmark_objective - network_c['objective'].values[0])) / 
                           benchmark_objective * 100)
            abs_time.append(float(network_c['time']))
            rel_time.append((benchmark_time -float(network_c['time'])) /
                            benchmark_time * 100)
            snapshot.append(int(c))
            k_parameter.append(i)
                         
#3D plot - surface
def plot_3d_surface (snapshot,k_parameter,variable,name):
    n_radii = 8
    n_angles = 36
    # Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    # Repeat all angles for each radius.
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
   
    x = snapshot
    y = k_parameter
    z = variable
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    
    ax.set_xlabel('snapshot parameters')
    ax.set_ylabel('network parameters')
    ax.set_zlabel(name) 
    plt.title('Clustering_analysis')
    plt.show()
    fig.savefig(path.join(home, 'Analysis_3d_surface, '+ name + '.png'))
   
    
plot_3d_surface (snapshot,k_parameter,abs_time,name='Absolute time in s')
plot_3d_surface (snapshot,k_parameter,rel_time,name='Relative time deviation in %')

plot_3d_surface (snapshot,k_parameter,abs_obj,name='Absolute objective function')
plot_3d_surface (snapshot,k_parameter,rel_obj,name='Relative obejective function deviation in %')
   
#3D plot - points 
def plot_3d_points (snapshot, k_parameter, variable, name):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c='b' #'b'
    m='o' #'^'
    ax.scatter(snapshot, k_parameter, variable, c=c, marker=m)
    
    ax.set_xlabel('snapshot parameters')
    ax.set_ylabel('network parameters')
    ax.set_zlabel(name) 
    plt.title('Clustering_analysis')
    plt.show()
    fig.savefig(path.join(home, 'Analysis_3d_point, '+ name + '.png'))

plot_3d_points (snapshot,k_parameter,abs_time,name='Absolute time in s')
plot_3d_points (snapshot,k_parameter,rel_time,name='Relative time deviation in %')

plot_3d_points (snapshot,k_parameter,abs_obj,name='Absolute objective function')
plot_3d_points (snapshot,k_parameter,rel_obj,name='Relative objective function deviation in %')

