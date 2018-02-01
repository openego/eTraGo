# -*- coding: utf-8 -*-

import pandas as pd
import re
from importlib import import_module
from sqlalchemy import and_
from etrago.tools.io import NetworkScenario

def calc_nearest_point(bus1, network):
        
      # bus0 = network.buses[network.buses.index == '39464' or network.buses.index ==  '39465']
        #print(bus1)
        #etwork = kwargs[network]
    bus1_index = network.buses.index[network.buses.index == bus1]
        #print(bus1_index)       
    x0 = network.buses.x[network.buses.index.isin(bus1_index)]
        #print(x0.values)
    y0 = network.buses.y[network.buses.index.isin(bus1_index)]
       # print(y0.values)
    comparable_buses = network.buses[~network.buses.index.isin(bus1_index)]
        #print(comparable_buses)
    x1 = comparable_buses.x
        #print(x1.values)
    y1 = comparable_buses.y
        #print(y1.values)
    distance =(x1.values- x0.values)*(x1.values- x0.values) + (y1.values- y0.values)*(y1.values- y0.values)
        
    min_distance = distance.min()
        #print(min_distance)
        
    bus0 = comparable_buses[((x1.values- x0.values)*(x1.values- x0.values) + (y1.values- y0.values)*(y1.values- y0.values)) == min_distance]
        
    bus0 = bus0.index[bus0.index == bus0.index.max()]
    bus0 = ''.join(bus0.values)
    #print(bus0)
    return bus0

def find_point(bus1, network):
        #global network
        #df = pd.DataFrame({'from_bus': bus1.values}) 
        #print(bus1)
        
        #df['to_bus'] = calc_nearest_point(df['from_bus'], network)
        bus0 = bus1.apply(calc_nearest_point, network = network)
        #bus0 = df.apply(calc_nearest_point,  axis = 1)

        return bus0.values
   
    
def connect_oyerlay_network (network):
    bus1 = network.transformers.bus0[~(network.transformers.bus0.isin(network.buses.index))
                                    & (network.transformers.bus1.isin(network.buses.index))]
    bus_1 = network.transformers.bus1[~network.transformers.bus1.isin(network.buses.index)]
    bus1 = bus1.append(bus_1)
    bus_1 = network.transformers.bus1[~network.transformers.bus1.isin(network.buses.index)]
   
    index = ['trafo_id']
    column = ['trafo_id', 'bus0', 'bus1', 'x', 's_nom']
    
    df =pd.DataFrame(columns=column)
    df.set_index('trafo_id')
    
    df['bus0'] = network.lines.bus0[(~network.lines.bus0.isin(network.lines.bus1)) &
      (network.lines.scn_name == 'extension_nep2035_b2')]
    
   # bus1 = network.transformers.bus1[~network.transformers.bus0.isin(network.buses.index)]
def add_by_scenario (self, df, name,  *args, **kwargs):
    
    if self.add_network != None:
        
        ormclass = getattr(import_module('egoio.db_tables.model_draft'), 'EgoGridPfHvExtension' + name)
    
        query = self.session.query(ormclass).filter(
                        ormclass.scn_name == 'decommissioning_' + self.add_network)
    
        df_decommisionning = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')
        
        if df_decommisionning.empty == False:
            df = df[~df.index.isin(df_decommisionning.index)]
    
        query = self.session.query(ormclass).filter(
                        ormclass.scn_name == 'extension_' + self.add_network)

        df_extension = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')
        df_extension.scn_name = self.scn_name
        
        if name == 'Line':# or 'Transformer':
            df_extension.s_nom_extendable = True  
            
        if name == 'Link':
            df_extension['bus0'] = df_extension.bus0.astype(int)
            df_extension['bus1'] = df_extension.bus1.astype(int)
            df_extension.p_nom_extendable = True      
            
        if df_extension.empty == False:
            df = df.append(df_extension)
    
    
    if self.add_be_no:
        ormclass = getattr(import_module('egoio.db_tables.model_draft'), 'EgoGridPfHvExtension' + name)
        
        query = self.session.query(ormclass).filter(
                        ormclass.scn_name == 'BE_NO_' + self.scn_name)

        df_be_no = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')
        
        if name == 'Line' or 'Transformer':
            df_be_no.s_nom_extendable = True  
            
        if name == 'Link':
            df_be_no['bus0'] = df_be_no.bus0.astype(int)
            df_be_no['bus1'] = df_be_no.bus1.astype(int)
            df_be_no.p_nom_extendable = True 
        

        if df_be_no.empty == False:
            df = df.append(df_be_no)
        
        
    return df            
 
    
def add_series_by_scenario (self, df, name, column,  *args, **kwargs):
    
    if self.add_network != None:
    
        ormclass = getattr(self._pkg, 'EgoGridPfHvExtension' + name)

        # TODO: pls make more robust
        id_column = re.findall(r'[A-Z][^A-Z]*', name)[0] + '_' + 'id'
        id_column = id_column.lower()

        query = self.session.query(
            getattr(ormclass, id_column),
            getattr(ormclass, column)[self.start_snapshot: self.end_snapshot].
            label(column)).filter(and_(
                ormclass.scn_name == self.add_network,
                ormclass.temp_id == self.temp_id))
      

        df_nep = pd.io.sql.read_sql(query.statement,
                                self.session.bind,
                                columns=[column],
                                index_col=id_column)
        
        if df_nep.empty == False:
            df = df.append(df_nep)
        
        
        
    if self.add_be_no == True:
            
            ormclass = getattr(self._pkg, 'EgoGridPfHvExtension' + name)

        # TODO: pls make more robust
            id_column = re.findall(r'[A-Z][^A-Z]*', name)[0] + '_' + 'id'
            id_column = id_column.lower()

            query = self.session.query(
                    getattr(ormclass, id_column),
                    getattr(ormclass, column)[self.start_snapshot: self.end_snapshot].
                    label(column)).filter(and_(
                            ormclass.scn_name == 'BE_NO_' + self.scn_name,
                            ormclass.temp_id == self.temp_id))

            df_be_no = pd.io.sql.read_sql(query.statement,
                                self.session.bind,
                                columns=[column],
                                index_col=id_column)
            
            if df_be_no.empty == False:
               df = df.append(df_be_no)
        
            
    return df