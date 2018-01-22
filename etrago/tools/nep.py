# -*- coding: utf-8 -*-

import pandas as pd
import re
from importlib import import_module
from sqlalchemy import and_
from etrago.tools.io import NetworkScenario


  
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