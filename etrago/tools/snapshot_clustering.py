# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import os


###############################################################################
compare = pd.read_csv('ward-results/med-cost-high-renew/daily365/storage_units-state_of_charge.csv',
                index_col=[0], parse_dates=[0]).sum(axis=1)
lst = ['6', '12', '24', '42']

#compare = pd.DataFrame()
for i in lst:
    x = pd.read_csv('ward-results/med-cost-high-renew/weekly'+i+'/storage_units-state_of_charge.csv',
                    index_col=[0], parse_dates=[0]).sum(axis=1)
    w = pd.read_csv('ward-results/med-cost-high-renew/weekly'+i+'/snapshots.csv',
                    index_col=[0], parse_dates=[0]).weightings
    l = pd.read_csv('ward-results/med-cost-high-renew/weekly'+i+'/loads-p.csv',
                    index_col=[0], parse_dates=[0])
    x = x / w
    compare = pd.concat([compare, x], axis=1)

compare.columns = ['365'] + lst
#compare[compare.isnull()] = 0
compare_new = compare.dropna()
compare_new.reset_index().plot(drawstyle='steps')

###############################################################################
#
storage1 = pd.read_csv('ward-results/med-cost-high-renew-2-nodes/storage_capacity.csv',
                index_col=[0])
storage2 = pd.read_csv('ward-results/med-cost-high-renew/storage_capacity.csv',
                index_col=[0])


daily1 = storage1.ix[:, storage1.columns.str.contains('daily')]
weekly1 = storage1.ix[:, ~storage1.columns.str.contains('weekly')]

daily2 = storage2.ix[:,storage2.columns.str.contains('daily')]
weekly2 = storage2.ix[:,storage2.columns.str.contains('weekly')]

p_d1 =  (daily1.mean() - storage1['365'].mean()) / storage1['365'].mean() * 100
p_w1 =  (weekly1.mean() - storage1['365'].mean()) / storage1['365'].mean() * 100
p_d2 =  (daily2.mean() - storage2['365'].mean()) / storage2['365'].mean() * 100
p_w2 =  (weekly2.mean() - storage2['365'].mean()) / storage2['365'].mean() * 100


ax = p_w1.plot(style='*-', color='blue', label='w-low-renew')
p_d1.plot(ax=ax, style='--', color='blue', label='d-low-renew')
p_w2.plot(ax=ax, style='*-', color='red', label='w-high-renew')
p_d2.plot(ax=ax, style='*--', color='red', label='d-high-renew')
lines = ax.get_lines()
ax.grid(True)
ax.legend(lines, [l.get_label() for l in lines])
ax.set_title('Difference in Storage Capacities')
ax.set_ylabel('Relative Difference of Average Storage Capacities in %')
ax.set_xlabel('Clustered days (in days/weeks)')
ax.set_xticklabels([str(i) for i in [7, 21, 42, 84, 126, 168, 210, 252, 294]])
fig = ax.get_figure()
fig.savefig('storage_capacities.eps')
#storage['365']


###############################################################################

cobj = pd.read_csv('compare-ward-results/med-cost-high-renew/objective.csv',
                index_col=[0])


obj = pd.read_csv('ward-results/med-cost-high-renew/objective.csv',
                index_col=[0])

ref = obj['objective'].loc['365']
d = obj.loc[obj.index.str.contains('daily')]['objective']
#w = obj.loc[obj.index.str.contains('weekly')]['objective']
o = cobj['objective']
d.index = [7, 21, 42, 84, 126, 168, 210, 252, 294]
#w.index = [7, 21, 42, 84, 126, 168, 210, 252, 294]
o.index = [7, 21, 42, 84, 126, 168, 210, 252, 294]

op = (o - ref) / ref * 100
op.name = 'Original clustered storages cap.'
dp = (d - ref) / ref * 100
dp.name = 'Daily Clustering'
#wp = (w - ref) / ref * 100

df = pd.concat([op, dp], axis=1)
ax = df.plot(kind='bar')
ax2 = ax.twinx()
times = obj.loc[obj.index.str.contains('daily')]['time']
relative_time = times / obj['time'].loc['365'] * 100
ax.set_title('Comparison of objective values (110%-Renew), Daily')
ax.set_ylabel('Relative percentage deviation in %')
ax.set_xlabel('Clustered Days')
ax.set_xticklabels([str(i) for i in [7, 21, 42, 84, 126, 168, 210, 252, 294]])
relative_time.plot(ax=ax2, style='*-', color='red')
ax2.set_ylabel('Relative time of clustering compared to original in %')
fig = ax.get_figure()
#fig.savefig("comparison_obj_high.eps")

###############################################################################

objective = pd.read_csv('ward-results/med-cost-low-renew/objective.csv',
                index_col=[0])['time']
objectiveh = pd.read_csv('ward-results/med-cost-high-renew/objective.csv',
                index_col=[0])['time']
d = objective.loc[objective.index.str.contains('daily')]
d.index = [7, 21, 42, 84, 126, 168, 210, 252, 294]
w = objective.loc[objective.index.str.contains('weekly')]
w.index = [7, 21, 42, 84, 126, 168, 210, 252, 294]

dh = objectiveh.loc[objectiveh.index.str.contains('daily')]
dh.index = [7, 21, 42, 84, 126, 168, 210, 252, 294]
wh = objectiveh.loc[objectiveh.index.str.contains('weekly')]
wh.index = [7, 21, 42, 84, 126, 168, 210, 252, 294]

dr =  d / objective.loc['365']
dr.name = 'daily-low'
wr =  w / objective.loc['365']
wr.name = 'weekly-low'


drh =  dh / objectiveh.loc['365']
drh.name = 'daily-high'
wrh =  wh / objectiveh.loc['365']
wrh.name = 'weekly-high'

df = pd.concat([dr, wr, drh, wrh],axis=1)

ax = df.plot(style=['*-', '--', '*-','--'], color=['blue', 'blue', 'red', 'red'])
ax.set_title('Difference in solving time')
ax.grid(True)
ax.set_ylabel('Relative time compared to original %')
ax.set_xlabel('Clustered days (daily/weekly)')

fig = ax.get_figure()
fig.savefig('solving_time.eps')

###############################################################################



df = pd.DataFrame()
path = 'results/med/daily'
original = pd.read_csv('results/med/original/storage_units.csv')#['p_nom_opt']
dirs = [i for i in os.listdir(path) if i !='Z.csv']
lst = [int(i) for i in dirs]
lst.sort()
#df_opt = pd.DataFrame(index=lst, columns=['objective', 'time'])
for d in lst:
    temp_df = pd.read_csv(os.path.join(path, str(d),'storage_units.csv'))
    temp = temp_df.get('p_nom_opt')
    if temp is None:
        temp_df['p_nom_opt'] = 0
    temp = temp_df['p_nom_opt']

    temp.name = str(d)
    df = pd.concat([df, temp], axis=1)

    temp_obj = pd.read_csv(os.path.join(path, str(d),'network.csv'))
    #df_opt.loc[d]['time'] = temp_obj['time'].values[0]

me = pd.DataFrame()
for col in df:
    me[col] = (original['p_nom_opt'] - df[col]) / original['p_nom_opt'] * 100
ax = me.mean().plot()
ax.set_xlabel('Clustered days')
ax.set_ylabel('Average Mean Deveation of Storage capacities in %')
ax.grid(True)


################################################################################

# original
stor = pd.read_csv('results/med/original/storage_units.csv', index_col=0)
gen_t = pd.read_csv('results/med/original/generators-p.csv', index_col=0)
gen = pd.read_csv('results/med/original/generators.csv', index_col=0)


# loop
L1_d = {}
objective_d = {}
clusters = [84, 126, 168, 294]

for c in clusters:
    snapshots = pd.read_csv('results/med/daily/'+str(c)+'/snapshots.csv',
                            parse_dates=[0], index_col=[0])
    stor_7 = pd.read_csv('results/med/daily/'+str(c)+'/storage_units.csv', index_col=0)
    gen_7_t = pd.read_csv('results/med/daily/'+str(c)+'/generators-p.csv', index_col=0)


    L1I = sum(stor.loc[s]['capital_cost'] *
              abs(stor_7.loc[s].get('p_nom_opt', 0) -
                  stor.loc[s]['p_nom_opt'])
              for s in stor.index)

    L1G = sum(gen.loc[g]['marginal_cost'] *
              abs(sum(gen_t.loc[t][str(g)] for t in gen_t.index) -
                  sum(gen_7_t.loc[str(t)][str(g)] for t in snapshots.index))
              for g in gen.index)

    L1 = L1I + L1G

    L1_d[str(c)] = L1

###############################################################################
network = pd.read_csv('results/med/original/network.csv')
objective_d = {}
clusters = [7, 14, 21, 28, 35, 42, 49, 56]
for c in clusters:
    network_c = pd.read_csv('results/med/daily/'+str(c)+'/network.csv')

    objective_d[str(c)] = abs(network_c['objective'].values[0] -
                              network['objective'].values[0])



###############################################################################
# Distance matrix....
###############################################################################

Z = snapshots = pd.read_csv('results/med/daily/Z.csv').values

last = Z[-200:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)
plt.show()

#acceleration = np.diff(last, 2)  # 2nd derivative of the distances
#acceleration_rev = acceleration[::-1]
#plt.plot(idxs[:-2] + 1, acceleration_rev)
#plt.show()
#k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
#print("clusters:", k)
##dendrogram(Z, color_threshold=25)
#
#
#



