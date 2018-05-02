# -*- coding: utf-8 -*-
"""
"""
import os
import pandas as pd

root_path = os.path.join(
    os.path.expanduser('~'),
    'projects', 'pf_results/snapshot_clustering/')

subfolder_prefix  = 'snapshot-clustering-results-cyclic-tsam-k'

kmeans_clusters = [2, 5]

kmeans_paths = {
    k: os.path.join(root_path, subfolder_prefix + str(k))
    for k in kmeans_clusters}

df = pd.DataFrame()
for k, v in kmeans_paths.items():

    # collect the original results without temporal clustering
    snapshot_cluster =  os.path.join(v, 'original')
    _df = pd.read_csv(os.path.join(v, 'original', 'network.csv'))
    _df['s'] = 0 # 0 => no clustering, but original problem!
    _df['k'] = int(k)
    df = pd.concat([df, _df])

    # collect the temporal clustered results
    snapshot_cluster_root_path = os.path.join(v, 'daily')
    for s in os.listdir(snapshot_cluster_root_path):
        snapshot_cluster_path = os.path.join(snapshot_cluster_root_path, s)
        _df = pd.read_csv(os.path.join(snapshot_cluster_path, 'network.csv'))
        _df['s'] = int(s)
        _df['k'] = int(k)
        df = pd.concat([df, _df])

# select columns of interest
df = df[['time', 'objective', 'max_memusage', 's', 'k']]

# set new index s=snapshot, k=kmeans
df.set_index(['s', 'k'], inplace=True)

# sort the index
df.sort_index(level=[1], inplace=True)

# write to csv for analysis
df.to_csv(os.path.join(root_path, 'aggregated.csv'))

for k in df.index.get_level_values('k').unique():

    time_benchmark = df.loc[(0, k), 'time']

    objective_benchmark = df.loc[(0, k), 'objective']

    # absolute time error
    df.loc[(slice(None), k), 'AE-time'] = \
        df.loc[(slice(None), k), 'time'] - time_benchmark

    # absolute objective error
    df.loc[(slice(None), k), 'AE-objective'] = \
        df.loc[(slice(None), k), 'objective'] - objective_benchmark

    # relative time error
    df.loc[(slice(None), k), 'RE-time'] = (
        (df.loc[(slice(None), k), 'time'] - time_benchmark) /
         time_benchmark * 100)

    # relative objective error
    df.loc[(slice(None), k), 'RE-objective'] = (
        (df.loc[(slice(None), k), 'objective'] - objective_benchmark) /
         objective_benchmark * 100)


# plotting with seaborn for testing
%matplotlib inline
import seaborn as sns
sns.set_style("darkgrid")
ax = sns.pointplot(x="s", y="RE-objective", hue='k',
                   data=df.drop(0).reset_index())
ax = sns.pointplot(x="s", y="RE-time", hue='k',
                   data=df.drop(0).reset_index())

df.to_csv('/tmp/data.csv')
