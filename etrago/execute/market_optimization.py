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
    
    from pypsa.clustering.spatial import get_clustering_from_busmap
    from pypsa.components import component_attrs




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
    
    logger.info("Start building pre market model")
    build_market_model(self)
    logger.info("Start solving pre market model")
    self.pre_market_model.lopf(
            solver_name=self.args["solver"],
            solver_options=self.args["solver_options"],
            pyomo=True,
            extra_functionality=extra_functionality(),
            formulation=self.args["model_formulation"],        
        )
   
    logger.info("Preparing short-term UC market model")
    build_shortterm_market_model(self)
    logger.info("Start solving short-term UC market model")
    
    
    self.market_model.optimize.optimize_with_rolling_horizon(
         snapshots=None, horizon=168, overlap=144, solver_name=self.args["solver"])
    
    # quick and dirty csv export of market model results
    path = self.args["csv_export"]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    self.market_model.export_to_csv_folder(path + "/market")
    
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
        .assign(p_min_pu=-1.0)
        .assign(carrier="DC")
        .set_index(ac.index),
        "Link",
    )    
    net.lines.drop(net.lines.loc[net.lines.carrier == 'AC'].index, inplace=True)
    #net.buses.loc[net.buses.carrier == 'AC', 'carrier'] = "DC"
    
    net.generators_t.p_max_pu = self.network.generators_t.p_max_pu

    self.pre_market_model = net
    
    

def build_shortterm_market_model(self):
    
    m = self.pre_market_model
    
    m.storage_units.p_nom_extendable=False
    m.stores.e_nom_extendable=False
    m.links.p_nom_extendable=False
    m.lines.s_nom_extendable=False
    
    m.storage_units.p_nom = m.storage_units.p_nom_opt
    m.stores.e_nom = m.stores.e_nom_opt
    m.links.p_nom = m.links.p_nom_opt
    m.lines.s_nom = m.lines.s_nom_opt
    
    #ToDo maybe ?!
    # somoehow setting seasonal storage (phs, h2 or finding a dynamic definition with respect to the results i.e. the storage behavior)
    # filling level (or similar) for the short-term rolling complicated market problem
    
    # set UC constraints

    unit_commitment = pd.read_csv("./data/unit_commitment.csv", index_col=0)
    unit_commitment.fillna(0, inplace=True)
    committable_attrs = m.generators.carrier.isin(unit_commitment).to_frame("committable")
            
    for attr in unit_commitment.index:
        default = component_attrs["Generator"].default[attr]
        committable_attrs[attr] = m.generators.carrier.map(unit_commitment.loc[attr]).fillna(
            default)
        committable_attrs[attr] = committable_attrs[attr].astype(m.generators.carrier.map(unit_commitment.loc[attr]).dtype)
        
    m.generators[committable_attrs.columns]=committable_attrs
    m.generators.min_up_time = m.generators.min_up_time.astype(int)
    m.generators.min_down_time = m.generators.min_down_time.astype(int)
    
    #Tadress link carriers i.e. OCGT
    committable_links = m.links.carrier.isin(unit_commitment).to_frame("committable")

    for attr in unit_commitment.index:
        default = component_attrs["Link"].default[attr]
        committable_links[attr] = m.links.carrier.map(unit_commitment.loc[attr]).fillna(
            default
        ) 
        committable_links[attr] = committable_links[attr].astype(m.links.carrier.map(unit_commitment.loc[attr]).dtype)

    m.links[committable_links.columns]=committable_links
    m.links.min_up_time = m.links.min_up_time.astype(int)
    m.links.min_down_time = m.links.min_down_time.astype(int)

    self.market_model = m
    

    

def extra_functionality():
    """Placeholder for extra functionalities within market optimization

    Returns
    -------
    None.

    """
    
    return None