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

# File description
"""
Define class Etrago
"""

import logging
import json
import pandas as pd
from pypsa.components import Network
from egoio.tools import db
from sqlalchemy.orm import sessionmaker
from etrago import __version__
from etrago.tools.io import (NetworkScenario,
                             extension,
                             decommissioning)
from etrago.tools.utilities import (set_branch_capacity,
                                    add_missing_components,
                                    set_random_noise,
                                    geolocation_buses,
                                    check_args,
                                    load_shedding,
                                    set_q_foreign_loads,
                                    foreign_links,
                                    crossborder_capacity,
                                    set_line_voltages,
                                    convert_capital_costs,
                                    get_args_setting,
                                    export_to_csv)
from etrago.tools.plot import (add_coordinates,
                               plot_grid)
from etrago.tools.extendable import extendable
from etrago.cluster.networkclustering import (run_kmeans_clustering,
                                              ehv_clustering)
from etrago.cluster.snapshot import (skip_snapshots,
                                     snapshot_clustering)
from etrago.cluster.disaggregation import run_disaggregation
from etrago.tools.execute import lopf, run_pf_post_lopf
from etrago.tools.calc_results import calc_etrago_results

logger = logging.getLogger(__name__)

class Etrago():
    """
    Object containing pypsa.Network including the transmission grid,
    input parameters and optimization results.

    Parameters
    ----------
    args : dict
        Dictionary including all inpu parameters.
    csv_folder_name : string
        Name of folder from which to import CSVs of network data.
    name : string, default ""
        Network name.
    ignore_standard_types : boolean, default False
        If True, do not read in PyPSA standard types into standard types
        DataFrames.
    kwargs
        Any remaining attributes to set

    Returns
    -------
    None

    Examples
    --------
    """
    def __init__(self,
                 args=None,
                 json_path=None,
                 csv_folder_name=None,
                 name="",
                 ignore_standard_types=False,
                 **kwargs):

        self.tool_version = __version__

        self.clustering = None

        self.results = pd.DataFrame()

        self.network = Network()

        self.disaggregated_network = Network()

        self.__re_carriers = ['wind_onshore', 'wind_offshore', 'solar',
                              'biomass', 'run_of_river', 'reservoir']
        self.__vre_carriers = ['wind_onshore', 'wind_offshore', 'solar']

        if args is not None:

            self.args = args

            self.get_args_setting(json_path)

            conn = db.connection(section=args['db'])

            session = sessionmaker(bind=conn)

            self.session = session()

            self.check_args()

        elif csv_folder_name is not None:

            self.get_args_setting(csv_folder_name + '/args.json')

            self.network = Network(csv_folder_name,
                                   name,
                                   ignore_standard_types)

            if self.args['disaggregation'] is not None:

                self.disaggregated_network = Network(
                    csv_folder_name + '/disaggregated_network',
                    name,
                    ignore_standard_types)

        else:
            logger.error('Set args or csv_folder_name')


    # Add functions
    get_args_setting = get_args_setting

    check_args = check_args

    add_coordinates = add_coordinates

    geolocation_buses = geolocation_buses

    add_missing_components = add_missing_components

    load_shedding = load_shedding

    set_random_noise = set_random_noise

    set_line_voltages = set_line_voltages

    set_q_foreign_loads = set_q_foreign_loads

    foreign_links = foreign_links

    crossborder_capacity = crossborder_capacity

    convert_capital_costs = convert_capital_costs

    extendable = extendable

    extension = extension

    set_branch_capacity = set_branch_capacity

    decommissioning = decommissioning

    plot_grid = plot_grid

    kmean_clustering = run_kmeans_clustering

    skip_snapshots = skip_snapshots

    ehv_clustering = ehv_clustering

    snapshot_clustering = snapshot_clustering

    lopf = lopf

    pf_post_lopf = run_pf_post_lopf

    disaggregation = run_disaggregation

    calc_results = calc_etrago_results

    export_to_csv = export_to_csv


    def build_network_from_db(self):

        """ Function that imports transmission grid from chosen database

        Returns
        -------
        None.

        """
        self.scenario = NetworkScenario(self.session,
                                        version=self.args['gridversion'],
                                        prefix=('EgoGridPfHv' if
                                                self.args['gridversion'] is None
                                                else 'EgoPfHv'),
                                        method=self.args['method'],
                                        start_snapshot=self.args['start_snapshot'],
                                        end_snapshot=self.args['end_snapshot'],
                                        scn_name=self.args['scn_name'])

        self.network = self.scenario.build_network()

        self.extension()

        self.decommissioning()

        logger.info('Imported network from db')

    def adjust_network(self):
        """
        Function that adjusts the network imported from the database according
        to given input-parameters.

        Returns
        -------
        None.

        """
        self.add_coordinates()

        self.geolocation_buses()

        self.add_missing_components()

        self.load_shedding()

        self.set_random_noise(0.01)

        self.set_line_voltages()

        self.set_q_foreign_loads(cos_phi=1)

        self.foreign_links()

        self.crossborder_capacity()

        self.set_branch_capacity()

        self.extendable(line_max=4)

    def _ts_weighted(self, timeseries):
        return timeseries.mul(self.network.snapshot_weightings, axis=0)

