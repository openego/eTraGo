# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:38:17 2017

@author: mario-arbeit
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

lines_bench = [[],[],[],[],[]]

with open('lines_bench.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        lines_bench[0].append(row['line_key']) 
        lines_bench[1].append(row['s_nom']) 
        lines_bench[2].append(float(row['s_nom_opt'])) 
        lines_bench[3].append(row['loading_new']) 
        lines_bench[4].append(row['dif']) 

lines_meth = [[],[],[],[],[]]

with open('lines_meth.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        lines_meth[0].append(row['line_key']) 
        lines_meth[1].append(row['s_nom']) 
        lines_meth[2].append(float(row['s_nom_opt']))
        lines_meth[3].append(row['loading_new']) 
        lines_meth[4].append(row['dif']) 


dif_opt = []

i=0
while(i<len(lines_bench[2])):
    dif_opt.append(lines_bench[2][i]-lines_meth[2][i]/lines_bench[2][i]*100)
    i+=1
    

number = [[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,'>100'],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
i = 0
while(i<len(dif_opt)):
    x=0
    while(x<len(number[0])): 
        if(x==21):
            number[1][x]+=1
            break
        elif(dif_opt[i]<=number[0][x]):
            number[1][x]+=1
            break
        else:
            x+=1
    i+=1
        
#x_name=(number[0][0],number[0][1],number[0][2],number[0][3],number[0][4],number[0][5],number[0][],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0],number[0][0])    
x_name = number[0]
y_pos = np.arange(len(x_name))
performance = number[1]
#y = number[1]
#x = number[0]
width = 5/1
plt.bar(y_pos, performance,align ='center',alpha = 0.5 ,color="blue")
plt.xticks(y_pos,x_name)
plt.xlabel('Abweichung in Prozent')
plt.ylabel('Anzahl der Abweichungen')
plt.title('s_nom_opt vergleich')
fig = plt.gcf()

plt.savefig('Vergleich.jpg')