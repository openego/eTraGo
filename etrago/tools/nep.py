# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:55:19 2017

@author: Clara
"""
import pandas as pd
import re
from importlib import import_module

  
def add_by_scenario (self, df, name,  *args, **kwargs):
    
    if self.add_network != None:
        
        ormclass = getattr(import_module('egoio.db_tables.model_draft'), 'EgoGridPfHvExtension' + name)
    
        query = self.session.query(ormclass).filter(
                        ormclass.scn_name == 'decommissioning_' + self.add_network)
    
        df_decommisionning = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')
        df_decommisionning.scn_name = self.scn_name
    
        if df_decommisionning.empty ==False:
            df = df.drop(df_decommisionning)
    
    
        query = self.session.query(ormclass).filter(
                        ormclass.scn_name == 'extension_' + self.add_network)


        df_extension = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')
    
        df_extension.scn_name = self.scn_name
    
        if df_extension.empty == False:
            df = df.append(df_extension)
            
            
        return df            
    
    if self.add_be_no:
        ormclass = getattr(import_module('egoio.db_tables.model_draft'), 'EgoGridPfHvExtension' + name)
        
        query = self.session.query(ormclass).filter(
                        ormclass.scn_name == 'BE_NO_' + self.scn_name)


        df_be_no = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')
    
        df_be_no.scn_name = self.scn_name
    
        if df_be_no.empty == False:
            df = df.append(df_be_no)
        
        return df
        
    return df            
    """data = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')
    for name, series in data.iteritems():
        data = data.astype((series.dtype) for name, series in df.iteritems() )
    
    df_extension = pd.DataFrame.from_items([
       (name, pd.Series(data = None, dtype=series.dtype))
                         for name, series in df.iteritems()])
    df_extension = df_extension.append (pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id'))"""
    
    
     


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
            df_nep.index = df_nep.index.astype(str)

        # change of format to fit pypsa
            df_nep = df_nep[column].apply(pd.Series).transpose()
            df_nep.scn_name = self.scn_name

            try:
                assert not df.empty
                df_nep.index = self.timeindex
            except AssertionError:
                    print("No data for %s in column %s." % (name, column))
            
           
            df = df.append(df_nep)
        
            return df
        
        if self.add_be_no:
            
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
      

            df_be_no = pd.io.sql.read_sql(query.statement,
                                self.session.bind,
                                columns=[column],
                                index_col=id_column)
            if df_be_no.empty == False:
                df_be_no.index = df_be_no.index.astype(str)

        # change of format to fit pypsa
                df_be_no = df_be_no[column].apply(pd.Series).transpose()
                df_be_no.scn_name = self.scn_name

                try:
                    assert not df.empty
                    df_be_no.index = self.timeindex
                except AssertionError:
                        print("No data for %s in column %s." % (name, column))
            
           
                df = df.append(df_be_no)
        
                return df
            
            return df




    

    
  
