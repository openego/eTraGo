# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
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
#
# File description
"""
io.py

Input/output operations between powerflow schema in the oedb and PyPSA.
Additionally oedb wrapper classes to instantiate PyPSA network objects.


Attributes
----------

packagename: str
    Package containing orm class definitions
temp_ormclass: str
    Orm class name of table with temporal resolution
carr_ormclass: str
    Orm class name of table with carrier id to carrier name datasets

Notes
-----
A configuration file connecting the chosen optimization method with
components to be queried is needed for NetworkScenario class.
"""

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, mariusves"

import pypsa
from importlib import import_module
import pandas as pd
from collections import OrderedDict
import re
import json
import os
import numpy as np
if 'READTHEDOCS' not in os.environ:
    from geoalchemy2.shape import to_shape
    from sqlalchemy.orm.exc import NoResultFound
    from sqlalchemy import and_, func, or_

#from etrago.tools.nep import add_by_scenario, add_series_by_scenario

packagename = 'egoio.db_tables'
temp_ormclass = 'TempResolution'
carr_ormclass = 'Source'


def load_config_file(filename='config.json'):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, filename)
    return json.load(open(path), object_pairs_hook=OrderedDict)


class ScenarioBase():
    """ Base class to address the dynamic provision of orm classes representing
    powerflow components from egoio based on a configuration file


    Parameters
    ----------

    config : OrderedDict
        Dictionary with orm class names that should be accessable via _mapped.
    session : sqla.orm.session.Session
        Handles conversations with the database.
    version : str
        Version number of data version control in grid schema of the oedb.
    prefix : str
        Common prefix of component orm classnames in egoio.
    """

    def __init__(
        self, session, method='lopf', configpath='config.json', version=None,
            prefix='EgoGridPfHv'):

        global packagename
        global temp_ormclass
        global carr_ormclass

        schema = 'grid' if version else 'model_draft'

        self.config = load_config_file(configpath)[method]
        self.session = session
        self.version = version
        self._prefix = prefix
        #: module: Providing orm class definitions to oedb
        self._pkg = import_module(packagename + '.' + schema)
        #: dict: Container for orm classes corresponding to configuration file
        self._mapped = {}

        # Populate _mapped with orm classes according to config
        for k, v in self.config.items():
            self.map_ormclass(k)
            if isinstance(v, dict):
                for kk in v.keys():
                    self.map_ormclass(kk)

        # map temporal resolution table
        self.map_ormclass(temp_ormclass)

        # map carrier id to carrier table
        self.map_ormclass(carr_ormclass)

    def map_ormclass(self, name):
        """ Populate _mapped attribute with orm class

        Parameters
        ----------
        name : str
            Component part of orm class name. Concatenated with _prefix.
        """

        try:
            self._mapped[name] = getattr(self._pkg, self._prefix + name)

        except AttributeError:
            print('Warning: Relation %s does not exist.' % name)


class NetworkScenario(ScenarioBase):
    """ Adapter class between oedb powerflow data and PyPSA. Provides the
    method build_network to generate a pypsa.Network.

    Parameters
    ----------
    scn_name : str
        Scenario name.
    method : str
        Objective function.
    start_snapshot : int
        First snapshot or timestep.
    end_snapshot : int
        Last timestep.
    temp_id : int
        Nummer of temporal resolution.
    """

    def __init__(
        self, session, scn_name='Status Quo', method='lopf',
            start_snapshot=1, end_snapshot=20, temp_id=1, **kwargs):

        self.scn_name = scn_name
        self.method = method
        self.start_snapshot = start_snapshot
        self.end_snapshot = end_snapshot
        self.temp_id = temp_id

        super().__init__(session, **kwargs)

        # network: pypsa.Network
        self.network = None

        self.configure_timeindex()

    def __repr__(self):
        r = ('NetworkScenario: %s' % self.scn_name)

        if not self.network:
            r += "\nTo create a PyPSA network call <NetworkScenario>.build_network()."

        return r

    def configure_timeindex(self):
        """ Construct a DateTimeIndex with the queried temporal resolution,
        start- and end_snapshot. """

        try:

            ormclass = self._mapped['TempResolution']
            if self.version:
                tr = self.session.query(ormclass).filter(
                    ormclass.temp_id == self.temp_id).filter(
                        ormclass.version == self.version).one()
            else:
                tr = self.session.query(ormclass).filter(
                    ormclass.temp_id == self.temp_id).one()

        except (KeyError, NoResultFound):
            print('temp_id %s does not exist.' % self.temp_id)

        timeindex = pd.DatetimeIndex(start=tr.start_time,
                                     periods=tr.timesteps,
                                     freq=tr.resolution)

        self.timeindex = timeindex[self.start_snapshot - 1: self.end_snapshot]
        """ pandas.tseries.index.DateTimeIndex :
                Index of snapshots or timesteps. """

    def id_to_source(self):

        ormclass = self._mapped['Source']
        query = self.session.query(ormclass)
        
        if self.version:
            query = query.filter(ormclass.version == self.version)
            
        # TODO column naming in database
        return {k.source_id: k.name for k in query.all()}

    def fetch_by_relname(self, name):
        """ Construct DataFrame with component data from filtered table data.

        Parameters
        ----------
        name : str
            Component name.

        Returns
        -------
        pd.DataFrame
            Component data.
        """

        ormclass = self._mapped[name]
        query = self.session.query(ormclass)

        if name != carr_ormclass:

            query = query.filter(
                ormclass.scn_name == self.scn_name)

        if self.version:
            query = query.filter(ormclass.version == self.version)

        # TODO: Naming is not consistent. Change in database required.
        if name == 'Transformer':
            name = 'Trafo'

        df = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')
        if name == 'Link':
            df['bus0'] = df.bus0.astype(int)
            df['bus1'] = df.bus1.astype(int)

        if 'source' in df:
            df.source = df.source.map(self.id_to_source())

        return df

    def series_fetch_by_relname(self, name, column):
        """ Construct DataFrame with component timeseries data from filtered
        table data.

        Parameters
        ----------
        name : str
            Component name.
        column : str
            Component field with timevarying data.

        Returns
        -------
        pd.DataFrame
            Component data.
        """

        ormclass = self._mapped[name]

        # TODO: This is implemented in a not very robust way.
        id_column = re.findall(r'[A-Z][^A-Z]*', name)[0] + '_' + 'id'
        id_column = id_column.lower()

        query = self.session.query(
            getattr(ormclass, id_column),
            getattr(ormclass, column)[self.start_snapshot: self.end_snapshot].
            label(column)).filter(and_(
                ormclass.scn_name == self.scn_name,
                ormclass.temp_id == self.temp_id))

        if self.version:
            query = query.filter(ormclass.version == self.version)

        df = pd.io.sql.read_sql(query.statement,
                                self.session.bind,
                                columns=[column],
                                index_col=id_column)

        df.index = df.index.astype(str)

        # change of format to fit pypsa
        df = df[column].apply(pd.Series).transpose()

        try:
            assert not df.empty
            df.index = self.timeindex
        except AssertionError:
            print("No data for %s in column %s." % (name, column))

        return df

    def build_network(self, network=None, *args, **kwargs):
        """  Core method to construct PyPSA Network object.
        """
        # TODO: build_network takes care of divergences in database design and
        # future PyPSA changes from PyPSA's v0.6 on. This concept should be
        # replaced, when the oedb has a revision system in place, because
        # sometime this will break!!!

        if network != None:
            network = network

        else:
            network = pypsa.Network()
            network.set_snapshots(self.timeindex)

        timevarying_override = False

        if pypsa.__version__ == '0.11.0':
            old_to_new_name = {'Generator':
                               {'p_min_pu_fixed': 'p_min_pu',
                                'p_max_pu_fixed': 'p_max_pu',
                                'source': 'carrier',
                                'dispatch': 'former_dispatch'},
                               'Bus':
                               {'current_type': 'carrier'},
                               'Transformer':
                               {'trafo_id': 'transformer_id'},
                               'Storage':
                               {'p_min_pu_fixed': 'p_min_pu',
                                'p_max_pu_fixed': 'p_max_pu',
                                'soc_cyclic': 'cyclic_state_of_charge',
                                'soc_initial': 'state_of_charge_initial',
                                'source': 'carrier'}}

            timevarying_override = True

        else:

            old_to_new_name = {'Storage':
                               {'soc_cyclic': 'cyclic_state_of_charge',
                                'soc_initial': 'state_of_charge_initial'}}

        for comp, comp_t_dict in self.config.items():

            # TODO: This is confusing, should be fixed in db
            pypsa_comp_name = 'StorageUnit' if comp == 'Storage' else comp

            df = self.fetch_by_relname(comp)

            if comp in old_to_new_name:

                tmp = old_to_new_name[comp]
                df.rename(columns=tmp, inplace=True)

            network.import_components_from_dataframe(df, pypsa_comp_name)

            if comp_t_dict:

                for comp_t, columns in comp_t_dict.items():

                    for col in columns:

                        df_series = self.series_fetch_by_relname(comp_t, col)

                        # TODO: VMagPuSet is not implemented.
                        if timevarying_override and comp == 'Generator' \
                        and not df_series.empty:
                            idx = df[df.former_dispatch == 'flexible'].index
                            idx = [i for i in idx if i in df_series.columns]
                            df_series.drop(idx, axis=1, inplace=True)

                        try:

                            pypsa.io.import_series_from_dataframe(
                                network,
                                df_series,
                                pypsa_comp_name,
                                col)

                        except (ValueError, AttributeError):
                            print("Series %s of component %s could not be "
                                  "imported" % (col, pypsa_comp_name))

        # populate carrier attribute in PyPSA network
        network.import_components_from_dataframe(
            self.fetch_by_relname(carr_ormclass), 'Carrier')

        self.network = network

        return network


def clear_results_db(session):
    '''Used to clear the result tables in the OEDB. Caution!
        This deletes EVERY RESULT SET!'''

    from egoio.db_tables.model_draft import EgoGridPfHvResultBus as BusResult,\
        EgoGridPfHvResultBusT as BusTResult,\
        EgoGridPfHvResultStorage as StorageResult,\
        EgoGridPfHvResultStorageT as StorageTResult,\
        EgoGridPfHvResultGenerator as GeneratorResult,\
        EgoGridPfHvResultGeneratorT as GeneratorTResult,\
        EgoGridPfHvResultLine as LineResult,\
        EgoGridPfHvResultLineT as LineTResult,\
        EgoGridPfHvResultLoad as LoadResult,\
        EgoGridPfHvResultLoadT as LoadTResult,\
        EgoGridPfHvResultTransformer as TransformerResult,\
        EgoGridPfHvResultTransformerT as TransformerTResult,\
        EgoGridPfHvResultMeta as ResultMeta
    print('Are you sure that you want to clear all results in the OEDB?')
    choice = ''
    while choice not in ['y', 'n']:
        choice = input('(y/n): ')
    if choice == 'y':
        print('Are you sure?')
        choice2 = ''
        while choice2 not in ['y', 'n']:
            choice2 = input('(y/n): ')
        if choice2 == 'y':
            print('Deleting all results...')
            session.query(BusResult).delete()
            session.query(BusTResult).delete()
            session.query(StorageResult).delete()
            session.query(StorageTResult).delete()
            session.query(GeneratorResult).delete()
            session.query(GeneratorTResult).delete()
            session.query(LoadResult).delete()
            session.query(LoadTResult).delete()
            session.query(LineResult).delete()
            session.query(LineTResult).delete()
            session.query(TransformerResult).delete()
            session.query(TransformerTResult).delete()
            session.query(ResultMeta).delete()
            session.commit()
        else:
            print('Deleting aborted!')
    else:
        print('Deleting aborted!')


def results_to_oedb(session, network, args, grid='hv', safe_results=False):
    """Return results obtained from PyPSA to oedb

    Parameters
    ----------
    session: 
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    args: dict
        Settings from appl.py
    grid: str
        Choose voltage-level, currently only 'hv' implemented
    safe_results: boolean
        If it is set to 'True' the result set will be saved
        to the versioned grid schema eventually apart from
        being saved to the model_draft by a SQL-script.
        ONLY set to True if you know what you are doing.

    """
    # Update generator_ids when k_means clustering to get integer ids
    if args['network_clustering_kmeans'] != False:
        new_index=pd.DataFrame(index = network.generators.index)
        new_index['new']=range(len(network.generators))

        for col in (network.generators_t):
            if not network.generators_t[col].empty:
                network.generators_t[col].columns =\
                    new_index.new[network.generators_t[col].columns]

        network.generators.index = range(len(network.generators))
    
    # moved this here to prevent error when not using the mv-schema
    import datetime
    if grid.lower() == 'mv':
        print('MV currently not implemented')
    elif grid.lower() == 'hv':
        from egoio.db_tables.model_draft import\
            EgoGridPfHvResultBus as BusResult,\
            EgoGridPfHvResultBusT as BusTResult,\
            EgoGridPfHvResultStorage as StorageResult,\
            EgoGridPfHvResultStorageT as StorageTResult,\
            EgoGridPfHvResultGenerator as GeneratorResult,\
            EgoGridPfHvResultGeneratorT as GeneratorTResult,\
            EgoGridPfHvResultLine as LineResult,\
            EgoGridPfHvResultLineT as LineTResult,\
            EgoGridPfHvResultLoad as LoadResult,\
            EgoGridPfHvResultLoadT as LoadTResult,\
            EgoGridPfHvResultTransformer as TransformerResult,\
            EgoGridPfHvResultTransformerT as TransformerTResult,\
            EgoGridPfHvResultMeta as ResultMeta,\
            EgoGridPfHvSource as Source
    else:
        print('Please enter mv or hv!')

    print('Uploading results to db...')
    # get last result id and get new one
    last_res_id = session.query(func.max(ResultMeta.result_id)).scalar()
    if last_res_id == None:
        new_res_id = 1
    else:
        new_res_id = last_res_id + 1

    # result meta data
    res_meta = ResultMeta()
    meta_misc = []
    for arg, value in args.items():
        if arg not in dir(res_meta) and arg not in ['db', 'lpfile',
                                                     'results', 'export']:
            meta_misc.append([arg, str(value)])

    res_meta.result_id = new_res_id
    res_meta.scn_name = args['scn_name']
    res_meta.calc_date = datetime.datetime.now()
    res_meta.user_name = args['user_name']
    res_meta.method = args['method']
    res_meta.start_snapshot = args['start_snapshot']
    res_meta.end_snapshot = args['end_snapshot']
    res_meta.safe_results = safe_results
    res_meta.snapshots = network.snapshots.tolist()
    res_meta.solver = args['solver']
    res_meta.settings = meta_misc

    session.add(res_meta)
    session.commit()

    # get source_id
    sources = pd.read_sql(session.query(Source).statement, session.bind)
    for gen in network.generators.index:
        if network.generators.carrier[gen] not in sources.name.values:
            new_source = Source()
            new_source.source_id = session.query(
                func.max(Source.source_id)).scalar()+1
            new_source.name = network.generators.carrier[gen]
            session.add(new_source)
            session.commit()
            sources = pd.read_sql(
                    session.query(Source).statement, session.bind)
        try:
            old_source_id = int(
                sources.source_id[
                        sources.name == network.generators.carrier[gen]])
            network.generators.set_value(gen, 'source', int(old_source_id))
        except:
            print(
                'Source ' + network.generators.carrier[gen] + 
                ' is not in the source table!')
    for stor in network.storage_units.index:
        if network.storage_units.carrier[stor] not in sources.name.values:
            new_source = Source()
            new_source.source_id = session.query(
                func.max(Source.source_id)).scalar()+1
            new_source.name = network.storage_units.carrier[stor]
            session.add(new_source)
            session.commit()
            sources = pd.read_sql(
                    session.query(Source).statement, session.bind)
        try:
            old_source_id = int(
                sources.source_id[
                        sources.name == network.storage_units.carrier[stor]])
            network.storage_units.set_value(stor, 'source', int(old_source_id))
        except:
            print(
                'Source ' + network.storage_units.carrier[stor] +
                ' is not in the source table!')

    whereismyindex = {BusResult: network.buses.index,
                      LoadResult: network.loads.index,
                      LineResult: network.lines.index,
                      TransformerResult: network.transformers.index,
                      StorageResult: network.storage_units.index,
                      GeneratorResult: network.generators.index,
                      BusTResult: network.buses.index,
                      LoadTResult: network.loads.index,
                      LineTResult: network.lines.index,
                      TransformerTResult: network.transformers.index,
                      StorageTResult: network.storage_units.index,
                      GeneratorTResult: network.generators.index}

    whereismydata = {BusResult: network.buses,
                     LoadResult: network.loads,
                     LineResult: network.lines,
                     TransformerResult: network.transformers,
                     StorageResult: network.storage_units,
                     GeneratorResult: network.generators,
                     BusTResult: network.buses_t,
                     LoadTResult: network.loads_t,
                     LineTResult: network.lines_t,
                     TransformerTResult: network.transformers_t,
                     StorageTResult: network.storage_units_t,
                     GeneratorTResult: network.generators_t}

    new_to_old_name = {'p_min_pu_fixed': 'p_min_pu',
                       'p_max_pu_fixed': 'p_max_pu',
                       'dispatch': 'former_dispatch',
                       'current_type': 'carrier',
                       'soc_cyclic': 'cyclic_state_of_charge',
                       'soc_initial': 'state_of_charge_initial'}

    ormclasses = [BusResult, LoadResult, LineResult, TransformerResult, 
                  GeneratorResult, StorageResult, BusTResult, LoadTResult, 
                  LineTResult, TransformerTResult, GeneratorTResult, 
                  StorageTResult]

    for ormclass in ormclasses:
        for index in whereismyindex[ormclass]:
            myinstance = ormclass()
            columns = ormclass.__table__.columns.keys()
            columns.remove('result_id')
            myinstance.result_id = new_res_id
            for col in columns:
                if '_id' in col:
                    class_id_name = col
                else:
                    continue
            setattr(myinstance, class_id_name, index)
            columns.remove(class_id_name)

            if str(ormclass)[:-2].endswith('T'):
                for col in columns:
                    if col == 'soc_set':
                        try:
                            setattr(myinstance, col, getattr(
                                whereismydata[ormclass], 
                                'state_of_charge_set')[index].tolist())
                        except:
                            pass
                    else:
                        try:
                            setattr(myinstance, col, getattr(
                                whereismydata[ormclass], col)[index].tolist())
                        except:
                            pass
                session.add(myinstance)

            else:
                for col in columns:
                    if col in new_to_old_name:
                        if col == 'soc_cyclic':
                            try:
                                setattr(myinstance, col, bool(
                                    whereismydata[ormclass].loc[index,
                                                 new_to_old_name[col]]))
                            except:
                                pass
                        elif 'Storage' in str(ormclass) and col == 'dispatch':
                            try:
                                setattr(myinstance, col,
                                        whereismydata[ormclass].loc[index, col])
                            except:
                                pass
                        else:
                            try:
                                setattr(
                                    myinstance, col, whereismydata[ormclass].\
                                    loc[index, new_to_old_name[col]])
                            except:
                                pass
                    elif col in ['s_nom_extendable', 'p_nom_extendable']:
                        try:
                            setattr(myinstance, col, bool(
                                whereismydata[ormclass].loc[index, col]))
                        except:
                            pass
                    else:
                        try:
                            setattr(myinstance, col,
                                    whereismydata[ormclass].loc[index, col])
                        except:
                            pass
                session.add(myinstance)

        session.commit()
    print('Upload finished!')

    return


def run_sql_script(conn, scriptname='results_md2grid.sql'):
    """This function runs .sql scripts in the folder 'sql_scripts' """

    script_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'sql_scripts'))
    script_str = open(os.path.join(script_dir, scriptname)).read()
    conn.execution_options(autocommit=True).execute(script_str)

    return



def extension (network, session, version, scn_extension, start_snapshot, 
               end_snapshot, **kwargs):
    """
    Function that adds an additional network to the existing network container. 
    The new network can include every PyPSA-component (e.g. buses, lines, links). 
    To connect it to the existing network, transformers are needed. 
        
    All components and its timeseries of the additional scenario need to be inserted in the fitting 'model_draft.ego_grid_pf_hv_extension_' table. 
    The scn_name in the tables have to be labled with 'extension_' + scn_name (e.g. 'extension_nep2035').

    Until now, the tables include three additional scenarios:
    'nep2035_confirmed': all new lines and needed transformers planed in the 'Netzentwicklungsplan 2035' (NEP2035) that have been confirmed by the Bundesnetzagentur (BNetzA)

    'nep2035_b2': all new lines and needed transformers planned in the NEP 2035 in the scenario 2035 B2

    'BE_NO_NEP 2035': DC-lines and transformers to connect the upcomming electrical-neighbours Belgium and Norway 
     Generation, loads and its timeseries in Belgium and Norway for scenario 'NEP 2035'


     Parameters
     -----
          network : The existing network container (e.g. scenario 'NEP 2035')
          session : session-data
          overlay_scn_name : Name of the additional scenario (WITHOUT 'extension_')
          start_snapshot, end_snapshot: Simulation time

    Returns
    ------
          network : Network container including existing and additional network

    """
    
    if version is None:
       ormcls_prefix = 'EgoGridPfHvExtension'
    else:
        ormcls_prefix = 'EgoPfHvExtension'
               
    # Adding overlay-network to existing network
    scenario = NetworkScenario(session,
                               version = version,
                               prefix=ormcls_prefix,
                               method=kwargs.get('method', 'lopf'),
                               start_snapshot=start_snapshot,
                               end_snapshot=end_snapshot,
                               scn_name='extension_' + scn_extension)

    network = scenario.build_network(network)

    # Allow lossless links to conduct bidirectional
    network.links.loc[network.links.efficiency == 1.0, 'p_min_pu'] = -1

    # Set coordinates for new buses
    extension_buses = network.buses[network.buses.scn_name ==
                                    'extension_' + scn_extension]
    for idx, row in extension_buses.iterrows():
            wkt_geom = to_shape(row['geom'])
            network.buses.loc[idx, 'x'] = wkt_geom.x
            network.buses.loc[idx, 'y'] = wkt_geom.y
               
    return network

def decommissioning(network, session, args, **kwargs):
    """
    Function that removes components in a decommissioning-scenario from
    the existing network container.
    Currently, only lines can be decommissioned.
               
    All components of the decommissioning scenario need to be inserted in
    the fitting 'model_draft.ego_grid_pf_hv_extension_' table. 
    The scn_name in the tables have to be labled with 'decommissioning_' 
    + scn_name (e.g. 'decommissioning_nep2035'). 
        
    
    Parameters
    -----
        network : The existing network container (e.g. scenario 'NEP 2035')
        session : session-data
        overlay_scn_name : Name of the decommissioning scenario


    Returns
    ------
        network : Network container including decommissioning
          
    """  

    if args['gridversion'] == None:   
        ormclass = getattr(import_module('egoio.db_tables.model_draft'),
                           'EgoGridPfHvExtensionLine')
    else:
        ormclass = getattr(import_module('egoio.db_tables.grid'),
                           'EgoPfHvExtensionLine')

    query = session.query(ormclass).filter(
                        ormclass.scn_name == 'decommissioning_' + 
                        args['scn_decommissioning'])

    df_decommisionning = pd.read_sql(query.statement,
                         session.bind,
                         index_col='line_id')
    df_decommisionning.index = df_decommisionning.index.astype(str)

    for idx, row in network.lines.iterrows():
        if (row['s_nom_min'] !=0) & (
            row['scn_name'] =='extension_' + args['scn_decommissioning']):
                v_nom_dec = df_decommisionning['v_nom'][(
                 df_decommisionning.project == row['project']) & (
                         df_decommisionning.project_id == row['project_id'])]

                if (v_nom_dec == 110).any():
                    network.lines.s_nom_min[network.lines.index == idx]\
                    = args['branch_capacity_factor']['HV'] *\
                    network.lines.s_nom_min

                else:
                    network.lines.s_nom_min[network.lines.index == idx] =\
                    args['branch_capacity_factor']['eHV'] *\
                    network.lines.s_nom_min

    ### Drop decommissioning-lines from existing network
    network.lines = network.lines[~network.lines.index.isin(
            df_decommisionning.index)]

    return network


def distance(x0, x1, y0, y1):
    """
    Function that calculates the square of the distance between two points.


    Parameters
    -----
        x0:  x - coordinate of point 0
        x1:  x - coordinate of point 1
        y0:  y - coordinate of point 0
        y1:  y - coordinate of point 1


    Returns
    ------
        distance : float
        square of distance 
          
    """   
    # Calculate square of the distance between two points (Pythagoras)
    distance = (x1.values- x0.values)*(x1.values- x0.values)\
        + (y1.values- y0.values)*(y1.values- y0.values)
    return distance


def calc_nearest_point(bus1, network):
    """
    Function that finds the geographical nearest point in a network from a given bus.


    Parameters
    -----
        bus1:  float
        id of bus
        network: Pypsa network container
        network including the comparable buses


    Returns
    ------
        bus0 : float
        bus_id of nearest point

    """

    bus1_index = network.buses.index[network.buses.index == bus1]

    forbidden_buses = np.append(
        bus1_index.values, network.lines.bus1[
                network.lines.bus0 == bus1].values)

    forbidden_buses = np.append(
        forbidden_buses, network.lines.bus0[network.lines.bus1 == bus1].values)

    forbidden_buses = np.append(
        forbidden_buses, network.links.bus0[network.links.bus1 == bus1].values)

    forbidden_buses = np.append(
        forbidden_buses, network.links.bus1[network.links.bus0 == bus1].values)

    x0 = network.buses.x[network.buses.index.isin(bus1_index)]

    y0 = network.buses.y[network.buses.index.isin(bus1_index)]

    comparable_buses = network.buses[~network.buses.index.isin(
            forbidden_buses)]

    x1 = comparable_buses.x

    y1 = comparable_buses.y

    distance = (x1.values - x0.values)*(x1.values - x0.values) + \
        (y1.values - y0.values)*(y1.values - y0.values)

    min_distance = distance.min()

    bus0 = comparable_buses[(((x1.values - x0.values)*(x1.values - x0.values
        ) + (y1.values - y0.values)*(y1.values - y0.values)) == min_distance)]
    bus0 = bus0.index[bus0.index == bus0.index.max()]
    bus0 = ''.join(bus0.values)

    return bus0


if __name__ == '__main__':
    if pypsa.__version__ not in ['0.6.2', '0.11.0']:
        print('Pypsa version %s not supported.' % pypsa.__version__)
    pass
