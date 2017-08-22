"""
K-means testing File

see: https://github.com/openego/eTraGo/issues/6

ToDo's:
-------
the remaining work would be:

- [x] implement the [todo](https://github.com/openego/eTraGo/blob/features/k-means-clustering/etrago/k_means_testing.py#L112-L115) so that the x of the lines which are newly defined as 380kV lines are adjusted

- [ ] in the [Hoersch and Brown contribution](https://arxiv.org/pdf/1705.07617.pdf) in Chapter II 2) and follwoing the weighting is defined. the weighting right now is equal over all buses. This should be changed to the assumptions with respect to the contribution or define another senseful weighting

- [ ] add functionality to save the resulting cluster for reproducibility

- [ ] convert it to a function and move it [here](https://github.com/openego/eTraGo/blob/features/k-means-clustering/etrago/cluster/networkclustering.py)


"""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "tba"

from appl import etrago
import json

# import scenario settings
with open('scenario_setting.json') as f:
    scenario_setting = json.load(f)



test = etrago(scenario_setting)



"""
network.buses['v_nom'] = 380.

# TODO adjust the x of the lines which are not 380. problem our lines have no v_nom. this is implicitly defined by the connected buses. Generally it should look something like the following:
#lines_v_nom_b = network.lines.v_nom != 380
#network.lines.loc[lines_v_nom_b, 'x'] *= (380./network.lines.loc[lines_v_nom_b, 'v_nom'])**2
#network.lines.loc[lines_v_nom_b, 'v_nom'] = 380.


trafo_index = network.transformers.index

network.import_components_from_dataframe(
    network.transformers.loc[:,['bus0','bus1','x','s_nom']]
    .assign(x=0.1*380**2/2000)
    .set_index('T' + trafo_index),
    'Line'
)

network.transformers.drop(trafo_index, inplace=True)
for attr in network.transformers_t:
    network.transformers_t[attr] = network.transformers_t[attr].reindex(columns=[])

busmap = busmap_by_kmeans(network, bus_weightings=pd.Series(np.repeat(1, len(network.buses)), index=network.buses.index) , n_clusters= 50)


clustering = get_clustering_from_busmap(network, busmap)
network = clustering.network
#network = cluster_on_extra_high_voltage(network, busmap, with_time=True)

"""
