""" io.py

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

"""

__copyright__ = ""
__license__ = ""
__author__ = ""

import pypsa
from importlib import import_module
import pandas as pd
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import and_, func
from collections import OrderedDict
import re
import json
import os


packagename = 'egoio.db_tables'
temp_ormclass = 'TempResolution'
carr_ormclass = 'Source'

def loadcfg(path=''):
    if path == '':
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, 'config.json')
    return json.load(open(path), object_pairs_hook=OrderedDict)


class ScenarioBase():
    """ Base class to hide package/db handling
    """

    def __init__(self, session, method, version=None, *args, **kwargs):

        global temp_ormclass
        global carr_ormclass

        schema = 'model_draft' if version is None else 'grid'

        cfgpath = kwargs.get('cfgpath', '')
        self.config = loadcfg(cfgpath)[method]

        self.session = session
        self.version = version
        self._prefix = kwargs.get('prefix', 'EgoGridPfHv')
        self._pkg = import_module(packagename + '.' + schema)
        self._mapped = {}

        # map static and timevarying classes
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

        global packagename

        try:
            self._mapped[name] = getattr(self._pkg, self._prefix + name)

        except AttributeError:
            print('Warning: Relation %s does not exist.' % name)


class NetworkScenario(ScenarioBase):
    """
    """

    def __init__(self, session, *args, **kwargs):
        super().__init__(session, *args, **kwargs)

        self.scn_name = kwargs.get('scn_name', 'Status Quo')
        self.method   = kwargs.get('method', 'lopf')
        self.start_snapshot  = kwargs.get('start_snapshot', 1)
        self.end_snapshot    = kwargs.get('end_snapshot', 20)
        self.temp_id  = kwargs.get('temp_id', 1)
        self.network  = None

        self.configure_timeindex()

    def __repr__(self):
        r = ('NetworkScenario: %s' % self.scn_name)

        if not self.network:
            r += "\nTo create a PyPSA network call <NetworkScenario>.build_network()."

        return r

    def configure_timeindex(self):
        """
        """

        try:

            ormclass = self._mapped['TempResolution']
            tr = self.session.query(ormclass).filter(
                ormclass.temp_id == self.temp_id).one()

        except (KeyError, NoResultFound):
            print('temp_id %s does not exist.' % self.temp_id)

        timeindex = pd.DatetimeIndex(start=tr.start_time,
                                     periods=tr.timesteps,
                                     freq=tr.resolution)

        self.timeindex = timeindex[self.start_snapshot - 1: self.end_snapshot]

    def id_to_source(self):

        ormclass = self._mapped['Source']
        query = self.session.query(ormclass)

        # TODO column naming in database
        return {k.source_id: k.name for k in query.all()}

    def by_scenario(self, name):
        """
        """

        ormclass = self._mapped[name]
        query = self.session.query(ormclass).filter(
            ormclass.scn_name == self.scn_name)

        if self.version:
            query = query.filter(ormclass.version == self.version)

        # TODO: Better handled in db
        if name == 'Transformer':
            name = 'Trafo'

        df = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')

        if 'source' in df:
            df.source = df.source.map(self.id_to_source())

        return df

    def series_by_scenario(self, name, column):
        """
        """

        ormclass = self._mapped[name]

        # TODO: pls make more robust
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

    def build_network(self, *args, **kwargs):
        """
        """
        # TODO: build_network takes care of divergences in database design and
        # future PyPSA changes from PyPSA's v0.6 on. This concept should be
        # replaced, when the oedb has a revision system in place, because
        # sometime this will break!!!

        network = pypsa.Network()
        network.set_snapshots(self.timeindex)

        timevarying_override = False

        if pypsa.__version__ == '0.8.0':

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
                                'soc_initial': 'state_of_charge_initial'}}

            timevarying_override = True

        else:

            old_to_new_name = {'Storage':
                               {'soc_cyclic': 'cyclic_state_of_charge',
                                'soc_initial': 'state_of_charge_initial'}}

        for comp, comp_t_dict in self.config.items():

            # TODO: This is confusing, should be fixed in db
            pypsa_comp_name = 'StorageUnit' if comp == 'Storage' else comp

            df = self.by_scenario(comp)

            if comp in old_to_new_name:

                tmp = old_to_new_name[comp]
                df.rename(columns=tmp, inplace=True)

            network.import_components_from_dataframe(df, pypsa_comp_name)

            if comp_t_dict:

                for comp_t, columns in comp_t_dict.items():

                    for col in columns:

                        df_series = self.series_by_scenario(comp_t, col)

                        # TODO: VMagPuSet?
                        if timevarying_override and comp == 'Generator':
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

        self.network = network

        return network
    
def clear_results_db(session):
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


def results_to_oedb(session, network, grid, args):
    """Return results obtained from PyPSA to oedb"""
    # moved this here to prevent error when not using the mv-schema
    import datetime
    if grid.lower() == 'mv':
        print('MV currently not implemented')
    elif grid.lower() == 'hv':
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
    else:
        print('Please enter mv or hv!')

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
        if arg not in dir(res_meta) and arg not in ['db','lpfile','results','export']:
            meta_misc.append([arg,str(value)])

    res_meta.result_id=new_res_id
    res_meta.scn_name=args['scn_name']
    res_meta.calc_date= datetime.datetime.now()
    res_meta.method=args['method']
    res_meta.gridversion = args['gridversion']
    res_meta.start_snapshot = args['start_snapshot']
    res_meta.end_snapshot = args['end_snapshot']
    res_meta.snapshots = network.snapshots.tolist()
    res_meta.solver = args['solver']
    res_meta.branch_capacity_factor = args['branch_capacity_factor']
    res_meta.pf_post_lopf = args['pf_post_lopf']
    res_meta.network_clustering = args['network_clustering']
    res_meta.storage_extendable = args['storage_extendable']
    res_meta.load_shedding = args['load_shedding']
    res_meta.generator_noise = args['generator_noise']
    res_meta.minimize_loading=args['minimize_loading']
    res_meta.k_mean_clustering=args['k_mean_clustering']
    res_meta.parallelisation=args['parallelisation']
    res_meta.line_grouping=args['line_grouping']
    res_meta.misc=meta_misc
    res_meta.comments=args['comments']
    
    session.add(res_meta)
    session.commit()
    
    # new result bus
    
    for col in network.buses_t.v_mag_pu:
        res_bus = BusResult()
        
        res_bus.result_id = new_res_id
        res_bus.bus_id = col      
        try:
            res_bus.x = network.buses.x[col]
        except:
            res_bus.x = None
        try:
            res_bus.y = network.buses.y[col]
        except:
            res_bus.y = None
        try:
            res_bus.v_nom = network.buses.v_nom[col]
        except: 
            res_bus.v_nom = None
        try:
            res_bus.current_type=network.buses.carrier[col]
        except: 
            res_bus.current_type=None
        try:
            res_bus.v_mag_pu_min = network.buses.v_mag_pu_min[col]
        except:
            res_bus.v_mag_pu_min = None
        try:
            res_bus.v_mag_pu_max = network.buses.v_mag_pu_max[col]
        except:
            res_bus.v_mag_pu_max = None
        try:
            res_bus.geom = network.buses.geom[col]
        except:
            res_bus.geom = None
        session.add(res_bus)
    session.commit()

# not working yet since ego.io classes are not yet iterable
#    for col in network.buses_t.v_mag_pu:
#        res_bus = BusResult()
#        res_bus.result_id = new_res_id
#        res_bus.bus_id = col
#        for var in dir(res_bus):
#            if not var.startswith('_') and var not in ('result_id','bus_id'):
#                try:
#                    res_bus.var = 3 #network.buses.var[col]
#                except:
#                    raise ValueError('WRONG')
#        session.add(res_bus)
#    session.commit()


    # new result bus_t
    for col in network.buses_t.v_mag_pu:
        res_bus_t = BusTResult()
        
        res_bus_t.result_id = new_res_id
        res_bus_t.bus_id = col
        try:
            res_bus_t.p = network.buses_t.p[col].tolist()
        except:
            res_bus_t.p = None
        try:
            res_bus_t.q = network.buses_t.q[col].tolist()
        except:
            res_bus_t.q = None
        try:
            res_bus_t.v_mag_pu = network.buses_t.v_mag_pu[col].tolist()
        except:
            res_bus_t.v_mag_pu = None
        try:
            res_bus_t.v_ang = network.buses_t.v_ang[col].tolist()
        except:
            res_bus_t.v_ang = None
        try:
            res_bus_t.marginal_price = network.buses_t.marginal_price[col].tolist() 
        except:
            res_bus_t.marginal_price = None
            
        session.add(res_bus_t)
    session.commit()


    # generator results
    for col in network.generators_t.p:
        res_gen = GeneratorResult()
        res_gen.result_id = new_res_id
        res_gen.generator_id = col
        res_gen.bus = int(network.generators.bus[col])
        try: 
            res_gen.dispatch = network.generators.former_dispatch[col]
        except:
            res_gen.dispatch = None
        try: 
            res_gen.control = network.generators.control[col]
        except:
            res_gen.control = None
        try: 
            res_gen.p_nom = network.generators.p_nom[col]
        except:
            res_gen.p_nom = None
        try: 
            res_gen.p_nom_extendable = bool(network.generators.p_nom_extendable[col])
        except:
            res_gen.p_nom_extendable = None
        try: 
            res_gen.p_nom_min = network.generators.p_nom_min[col]
        except:
            res_gen.p_nom_min = None
        try: 
            res_gen.p_nom_max = network.generators.p_nom_max[col]
        except:
            res_gen.p_nom_max = None
        try: 
            res_gen.p_min_pu_fixed = network.generators.p_min_pu[col]
        except:
            res_gen.p_min_pu_fixed = None
        try: 
            res_gen.p_max_pu_fixed = network.generators.p_max_pu[col]
        except:
            res_gen.p_max_pu_fixed = None
        try: 
            res_gen.sign = network.generators.sign[col]
        except:
            res_gen.sign = None
#        try: 
#            res_gen.source = network.generators.carrier[col]
#        except:
#            res_gen.source = None
        try: 
            res_gen.marginal_cost = network.generators.marginal_cost[col]
        except:
            res_gen.marginal_cost = None
        try: 
            res_gen.capital_cost = network.generators.capital_cost[col] 
        except:
            res_gen.capital_cost = None
        try: 
            res_gen.efficiency = network.generators.efficiency[col]
        except:
            res_gen.efficiency = None
        try: 
            res_gen.p_nom_opt = network.generators.p_nom_opt[col]
        except:
            res_gen.p_nom_opt = None
        session.add(res_gen)
    session.commit()           

    # generator_t results
    for col in network.generators_t.p:
        res_gen_t = GeneratorTResult()
        res_gen_t.result_id = new_res_id
        res_gen_t.generator_id = col
        try:
            res_gen_t.p_set = network.generators_t.p_set[col].tolist()
        except:
            res_gen_t.p_set = None
        try:
            res_gen_t.q_set = network.generators_t.q_set[col].tolist()
        except:
            res_gen_t.q_set = None
        try:
            res_gen_t.p_min_pu = network.generators_t.p_min_pu[col].tolist()
        except:
            res_gen_t.p_min_pu = None
        try:
            res_gen_t.p_max_pu = network.generators_t.p_max_pu[col].tolist()
        except:
            res_gen_t.p_max_pu = None
        try:
            res_gen_t.p = network.generators_t.p[col].tolist()
        except:
            res_gen_t.p = None
        try:
            res_gen_t.q = network.generators_t.q[col].tolist()
        except:
            res_gen_t.q = None
        try:
            res_gen_t.status = network.generators_t.status[col].tolist()
        except:
            res_gen_t.status = None
        session.add(res_gen_t)
    session.commit()
                
                
    # line results
    for col in network.lines_t.p0:
        res_line = LineResult()
        res_line.result_id=new_res_id,
        res_line.line_id=col
        res_line.bus0=int(network.lines.bus0[col])
        res_line.bus1=int(network.lines.bus1[col])
        try:
            res_line.x = network.lines.x[col]
        except:
            res_line.x = None
        try:
            res_line.r = network.lines.r[col]
        except:
            res_line.r = None
        try:
            res_line.g = network.lines.g[col]
        except:
            res_line.g = None
        try:
            res_line.b = network.lines.b[col]
        except:
            res_line.b = None
        try:
            res_line.s_nom = network.lines.s_nom[col]
        except:
            res_line.s_nom = None
        try:
            res_line.s_nom_extendable = bool(network.lines.s_nom_extendable[col])
        except:
            res_line.s_nom_extendable = None
        try:
            res_line.s_nom_min = network.lines.s_nom_min[col]
        except:
            res_line.s_nom_min = None
        try:
            res_line.s_nom_max = network.lines.s_nom_max[col]
        except:
            res_line.s_nom_max = None
        try:
            res_line.capital_cost = network.lines.capital_cost[col]
        except:
            res_line.capital_cost = None
        try:
            res_line.length = network.lines.length[col]
        except:
            res_line.length = None
        try:
            res_line.cables = int(network.lines.cables[col])
        except:
            res_line.cables = None
        try:
            res_line.frequency = network.lines.frequency[col]
        except:
            res_line.frequency = None
        try:
            res_line.terrain_factor = network.lines.terrain_factor[col]
        except:
            res_line.terrain_factor = None
        try:
            res_line.x_pu = network.lines.x_pu[col]
        except:
            res_line.x_pu = None
        try:
            res_line.r_pu = network.lines.r_pu[col]
        except:
            res_line.r_pu = None
        try:
            res_line.g_pu = network.lines.g_pu[col]
        except:
            res_line.g_pu = None
        try:
            res_line.b_pu = network.lines.b_pu[col]
        except:
            res_line.b_pu = None
        try:
            res_line.s_nom_opt = network.lines.s_nom_opt[col]
        except:
            res_line.s_nom_opt = None
        try:
            res_line.geom = network.lines.geom[col]
        except:
            res_line.geom = None
        try:
            res_line.topo = network.lines.topo[col]
        except:
            res_line.topo = None
        session.add(res_line)
    session.commit()
            

    # line_t results            
    for col in network.lines_t.p0:
        res_line_t = LineTResult()
        res_line_t.result_id=new_res_id,
        res_line_t.line_id=col          
        try:
            res_line_t.p0 = network.lines_t.p0[col].tolist()
        except:
            res_line_t.p0 = None
        try:
            res_line_t.q0 = network.lines_t.q0[col].tolist()
        except:
            res_line_t.q0 = None
        try:
            res_line_t.p1 = network.lines_t.p1[col].tolist()
        except:
            res_line_t.p1 = None
        try:
            res_line_t.q1 = network.lines_t.q1[col].tolist()
        except: 
            res_line_t.q1 = None
        session.add(res_line_t)
    session.commit()
    

    # load results
    for col in network.loads_t.p:
        res_load = LoadResult()
        res_load.result_id=new_res_id,
        res_load.load_id=col   
        res_load.bus = int(network.loads.bus[col])
        try:
            res_load.sign = network.loads.sign[col]
        except:
            res_load.sign = None
        try:
            res_load.e_annual = network.loads.e_annual[col]
        except:
            res_load.e_annual = None
        session.add(res_load)
    session.commit()    
        
    # load_t results
    for col in network.loads_t.p:
        res_load_t = LoadTResult()
        res_load_t.result_id=new_res_id,
        res_load_t.load_id=col 
        try:
            res_load_t.p_set = network.loads_t.p_set[col].tolist()
        except:
            res_load_t.p_set = None
        try:
            res_load_t.q_set = network.loads_t.q_set[col].tolist()
        except:
            res_load_t.q_set = None
        try:
            res_load_t.p = network.loads_t.p[col].tolist()
        except:
            res_load_t.p = None
        try:
            res_load_t.q = network.loads_t.q[col].tolist()
        except:
            res_load_t.q = None
        session.add(res_load_t)
    session.commit()    
            

    # insert results of transformers

    for col in network.transformers_t.p0:
        res_transformer = TransformerResult()
        res_transformer.result_id=new_res_id
        res_transformer.trafo_id=col
        res_transformer.bus0=int(network.transformers.bus0[col])
        res_transformer.bus1=int(network.transformers.bus1[col])
        try:
            res_transformer.x = network.transformers.x[col]
        except:
            res_transformer.x = None
        try:
            res_transformer.r = network.transformers.r[col]
        except:
            res_transformer.r = None
        try:
            res_transformer.g = network.transformers.g[col]
        except:
            res_transformer.g = None
        try:
            res_transformer.b = network.transformers.b[col]
        except:
            res_transformer.b = None
        try:
            res_transformer.s_nom = network.transformers.s_nom[col]
        except:
            res_transformer.s_nom = None
        try:
            res_transformer.s_nom_extendable = bool(network.transformers.s_nom_extendable[col])
        except:
            res_transformer.s_nom_extendable = None
        try:
            res_transformer.s_nom_min = network.transformers.s_nom_min[col]
        except:
            res_transformer.s_nom_min = None
        try:
            res_transformer.s_nom_max = network.transformers.s_nom_max[col]
        except:
            res_transformer.s_nom_max = None
        try:
            res_transformer.tap_ratio = network.transformers.tap_ratio[col]
        except:
            res_transformer.tap_ratio = None
        try:
            res_transformer.phase_shift = network.transformers.phase_shift[col]
        except:
            res_transformer.phase_shift = None
        try:
            res_transformer.capital_cost = network.transformers.capital_cost[col]
        except:
            res_transformer.capital_cost = None
        try:
            res_transformer.x_pu = network.transformers.x_pu[col]
        except:
            res_transformer.x_pu = None
        try:
            res_transformer.r_pu = network.transformers.r_pu[col]
        except:
            res_transformer.r_pu = None
        try:
            res_transformer.g_pu = network.transformers.g_pu[col]
        except:
            res_transformer.g_pu = None
        try:
            res_transformer.b_pu = network.transformers.b_pu[col]
        except:
            res_transformer.b_pu = None
        try:
            res_transformer.s_nom_opt = network.transformers.s_nom_opt[col]
        except:
            res_transformer.s_nom_opt = None
        try:
            res_transformer.geom = network.transformers.geom[col]
        except:
            res_transformer.geom = None
        try:
            res_transformer.topo = network.transformers.topo[col]
        except:
            res_transformer.topo = None
        session.add(res_transformer)
    session.commit()
    
    # insert results of transformers_t   
    for col in network.transformers_t.p0:
        res_transformer_t = TransformerTResult()
        res_transformer_t.result_id=new_res_id
        res_transformer_t.trafo_id=col
        try:
            res_transformer_t.p0 = network.transformers_t.p0[col].tolist()
        except:
            res_transformer_t.p0 = None
        try:
            res_transformer_t.q0 = network.transformers_t.q0[col].tolist()
        except:
            res_transformer_t.q0 = None
        try:
            res_transformer_t.p1 = network.transformers_t.p1[col].tolist()
        except:
            res_transformer_t.p1 = None
        try:
            res_transformer_t.q1 = network.transformers_t.q1[col].tolist()
        except:
            res_transformer_t.q1 = None
        session.add(res_transformer_t)
    session.commit()
        

    
    # storage_units results

    for col in network.storage_units_t.p:
        res_sto = StorageResult()
        res_sto.result_id=new_res_id,
        res_sto.storage_id=col,
        res_sto.bus=int(network.storage_units.bus[col])
        try:
            res_sto.dispatch = network.storage_units.dispatch[col]
        except:
            res_sto.dispatch = None
        try:
            res_sto.control = network.storage_units.control[col]
        except:
            res_sto.control = None
        try:
            res_sto.p_nom = network.storage_units.p_nom[col]
        except:
            res_sto.p_nom = None
        try:
            res_sto.p_nom_extendable = bool(network.storage_units.p_nom_extendable[col])
        except:
            res_sto.p_nom_extendable = None
        try:
            res_sto.p_nom_min = network.storage_units.p_nom_min[col]
        except:
            res_sto.p_nom_min = None
        try:
            res_sto.p_nom_max = network.storage_units.p_nom_max[col]
        except:
            res_sto.p_nom_max = None
        try:
            res_sto.p_min_pu_fixed = network.storage_units.p_min_pu[col]
        except:
            res_sto.p_min_pu_fixed = None
        try:
            res_sto.p_max_pu_fixed = network.storage_units.p_max_pu[col]
        except:
            res_sto.p_max_pu_fixed = None
        try:
            res_sto.sign = network.storage_units.sign[col]
        except:
            res_sto.sign = None
#        try:
#            res_sto.source = network.storage_units.carrier[col]
#        except:
#            res_sto.source = None
        try:
            res_sto.marginal_cost = network.storage_units.marginal_cost[col]
        except:
            res_sto.marginal_cost = None
        try:
            res_sto.capital_cost = network.storage_units.capital_cost[col]
        except:
            res_sto.capital_cost = None
        try:
            res_sto.efficiency = network.storage_units.efficiency[col]
        except:
            res_sto.efficiency = None
        try:
            res_sto.soc_initial = network.storage_units.state_of_charge_initial[col]
        except:
            res_sto.soc_initial = None
        try:
            res_sto.soc_cyclic = bool(network.storage_units.cyclic_state_of_charge[col])
        except:
            res_sto.soc_cyclic = None
        try:
            res_sto.max_hours = network.storage_units.max_hours[col]
        except:
            res_sto.max_hours = None
        try:
            res_sto.efficiency_store = network.storage_units.efficiency_store[col]
        except:
            res_sto.efficiency_store = None
        try:
            res_sto.efficiency_dispatch = network.storage_units.efficiency_dispatch[col]
        except:
            res_sto.efficiency_dispatch = None
        try:
            res_sto.standing_loss = network.storage_units.standing_loss[col]
        except:
            res_sto.standing_loss = None
        try:
            res_sto.p_nom_opt = network.storage_units.p_nom_opt[col]
        except:
            res_sto.p_nom_opt = None   
        session.add(res_sto)
    session.commit()
    
    # storage_units_t results    
    for col in network.storage_units_t.p:
        res_sto_t = StorageTResult()
        res_sto_t.result_id=new_res_id,
        res_sto_t.storage_id=col,
        try:
            res_sto_t.p_set = network.storage_units_t.p_set[col].tolist()
        except:
            res_sto_t.p_set = None
        try:
            res_sto_t.q_set = network.storage_units_t.q_set[col].tolist()
        except:
            res_sto_t.q_set = None
        try:
            res_sto_t.p_min_pu = network.storage_units_t.p_min_pu[col].tolist()
        except:
            res_sto_t.p_min_pu = None
        try:
            res_sto_t.p_max_pu = network.storage_units_t.p_max_pu[col].tolist()
        except:
            res_sto_t.p_max_pu = None
        try:
            res_sto_t.soc_set = network.storage_units_t.state_of_charge_set[col].tolist()
        except:
            res_sto_t.soc_set = None
        try:
            res_sto_t.inflow = network.storage_units_t.inflow[col].tolist()
        except:
              res_sto_t.inflow = None     
        try:
            res_sto_t.p = network.storage_units_t.p[col].tolist()
        except:
            res_sto_t.p = None
        try:
            res_sto_t.q = network.storage_units_t.q[col].tolist()
        except:
            res_sto_t.q = None
        try:
            res_sto_t.state_of_charge = network.storage_units_t.state_of_charge[col].tolist()
        except:
            res_sto_t.state_of_charge = None
        try:
            res_sto_t.spill = network.storage_units_t.spill[col].tolist()
        except:
            res_sto_t.spill = None
        session.add(res_sto_t)
    session.commit()
        

    
if __name__ == '__main__':
    if pypsa.__version__ not in ['0.6.2', '0.8.0']:
        print('Pypsa version %s not supported.' % pypsa.__version__)
    pass
