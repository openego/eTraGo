# -*- coding: utf-8 -*-
# Copyright 2016-2023  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
"""
Defines the market optimization within eTraGo
"""
import os

if "READTHEDOCS" not in os.environ:
    import logging
    import time

    import numpy as np
    import pandas as pd
    
    from etrago.cluster.electrical import (preprocessing, postprocessing,)    
    from etrago.cluster.spatial import (
        strategies_one_ports,
        strategies_generators,)
    
    from pypsa.networkclustering import get_clustering_from_busmap




    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = (
    "ulfmueller, ClaraBuettner, CarlosEpia"
)


def market_optimization(self):
    
    logger.info("Start building market model")
    build_market_model(self)
    logger.info("Start solving market model")
    self.market_model.lopf(
            solver_name=self.args["solver"],
            solver_options=self.args["solver_options"],
            pyomo=True,
            extra_functionality=extra_functionality(),
            formulation=self.args["model_formulation"],        
        )
    #self.market_model.model.write('/home/ulf/file2.lp')
    
def build_market_model(self):
    """Builds market model based on imported network from eTraGo
    
    
    - import market regions from file or database
    - Cluster network to market regions
    -- consider marginal cost incl. generator noise when grouoping electrical 
        generation capacities 

    Returns
    -------
    None.

    """
    
    # use existing preprocessing to get only the electricity system
        
    net, weight, n_clusters, busmap_foreign = preprocessing(self)
    
    df = pd.DataFrame({'country': net.buses.country.unique(), 
                       'marketzone': net.buses.country.unique()}, 
                      columns = ["country", "marketzone"])
        
    df.loc[(df.country == 'DE')| (df.country == 'LU'), 'marketzone'] = 'DE/LU' 
    
    df['cluster'] = df.groupby(df.marketzone).grouper.group_info[0]
    
    
    for i in net.buses.country.unique():
    
        net.buses.loc[net.buses.country== i , 'cluster'] = df.loc[df.country == i, 'cluster'].values[0]
        
    
    busmap = pd.Series(net.buses.cluster.astype(int).astype(str), net.buses.index)
    medoid_idx = pd.Series(dtype=str)

    logger.info("Start market zone specifc clustering")
    
    self.clustering, busmap = postprocessing(
        self, busmap, busmap_foreign, medoid_idx, 
        aggregate_generators_carriers=[], aggregate_links=False)
    
    self.update_busmap(busmap)
 
    
    net = self.clustering.network
    #links_col = net.links.columns
    ac = net.lines[net.lines.carrier == "AC"]
    str1 = "transshipment_"
    ac.index = f"{str1}" + ac.index
    net.import_components_from_dataframe(
        ac.loc[:, ["bus0", "bus1", "capital_cost", "length"]]
        .assign(p_nom=ac.s_nom)
        .assign(p_nom_min=ac.s_nom_min)
        .assign(p_nom_max=ac.s_nom_max)
        .assign(p_nom_extendable=ac.s_nom_extendable)
        .assign(p_max_pu=ac.s_max_pu)
        .assign(p_min_pu=-1)
        .assign(carrier="DC")
        .set_index(ac.index),
        "Link",
    )    
    net.lines.drop(net.lines.loc[net.lines.carrier == 'AC'].index, inplace=True)
    #net.buses.loc[net.buses.carrier == 'AC', 'carrier'] = "DC"

    # delete following unconnected CH4 buses. why are they there?
    net.buses.drop(net.buses[net.buses.index.isin(['37865', '37870'])].index, inplace=True)

    self.market_model = net
    
    # Todo: buses_by_country() geolocation_buses() apply on market_model does not work because no self.network?!
    
    

    

def extra_functionality():
    """Placeholder for extra functionalities within market optimization

    Returns
    -------
    None.

    """
    
    return None