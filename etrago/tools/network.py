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
Define class Etrago
"""

import logging
import os

from pypsa.components import Network
from sqlalchemy.orm import sessionmaker
import pandas as pd

if "READTHEDOCS" not in os.environ:
    from etrago.tools import db

from etrago import __version__
from etrago.cluster.disaggregation import run_disaggregation
from etrago.cluster.electrical import ehv_clustering, run_spatial_clustering
from etrago.cluster.gas import run_spatial_clustering_gas
from etrago.cluster.snapshot import skip_snapshots, snapshot_clustering
from etrago.execute import (
    dispatch_disaggregation,
    lopf,
    optimize,
    run_pf_post_lopf,
)
from etrago.execute.grid_optimization import (
    add_redispatch_generators,
    grid_optimization,
)
from etrago.execute.market_optimization import (
    build_market_model,
    market_optimization,
)
from etrago.execute.sclopf import (
    iterate_sclopf,
    post_contingency_analysis_lopf,
)
from etrago.tools.calc_results import (
    ac_export,
    ac_export_per_country,
    calc_etrago_results,
    dc_export,
    dc_export_per_country,
    german_network,
    system_costs_germany,
)
from etrago.tools.extendable import extendable
from etrago.tools.io import (
    NetworkScenario,
    add_ch4_h2_correspondence,
    decommissioning,
    extension,
)
from etrago.tools.plot import (
    bev_flexibility_potential,
    demand_side_management,
    flexibility_usage,
    heat_stores,
    hydrogen_stores,
    plot_carrier,
    plot_clusters,
    plot_gas_generation,
    plot_gas_summary,
    plot_grid,
    plot_h2_generation,
    plot_h2_summary,
    plot_heat_loads,
    plot_heat_summary,
    shifted_energy,
)
from etrago.tools.utilities import (
    add_missing_components,
    adjust_CH4_gen_carriers,
    buses_by_country,
    check_args,
    convert_capital_costs,
    crossborder_capacity,
    delete_dispensable_ac_buses,
    delete_irrelevant_oneports,
    drop_sectors,
    export_to_csv,
    filter_links_by_carrier,
    foreign_links,
    geolocation_buses,
    get_args_setting,
    get_clustering_data,
    load_shedding,
    manual_fixes_datamodel,
    set_branch_capacity,
    set_control_strategies,
    set_line_costs,
    set_q_foreign_loads,
    set_q_national_loads,
    set_random_noise,
    set_trafo_costs,
    update_busmap,
)

logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = """AmeliaNadal, CarlosEpia, ClaraBuettner, KathiEsterl, gnn,
 fwitte, ulfmueller, pieterhexen"""


class Etrago:
    """
    Object containing pypsa.Network including the transmission grid, not
    electric sectors, input parameters and optimization results.

    Parameters
    ----------
    args : dict
        Dictionary including all input parameters.
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

    def __init__(
        self,
        args=None,
        csv_folder_name=None,
        ignore_standard_types=False,
        json_path=None,
        name="",
        **kwargs,
    ):
        self.tool_version = __version__

        self.clustering = None

        self.results = pd.DataFrame()

        self.network = Network()

        self.network_tsa = Network()

        self.disaggregated_network = Network()

        self.__re_carriers = [
            "wind_onshore",
            "wind_offshore",
            "solar",
            "biomass",
            "run_of_river",
            "reservoir",
        ]
        self.__vre_carriers = ["wind_onshore", "wind_offshore", "solar"]

        self.busmap = {}

        self.ch4_h2_mapping = {}

        if args is not None:
            self.args = args

            self.get_args_setting(json_path)

            conn = db.connection(section=self.args["db"])

            session = sessionmaker(bind=conn)

            self.engine = conn

            self.session = session()

            self.check_args()

        elif csv_folder_name is not None:
            self.get_args_setting(csv_folder_name + "/args.json")

            self.network = Network(
                csv_folder_name, name, ignore_standard_types
            )

            if self.args["spatial_disaggregation"] is not None:
                self.disaggregated_network = Network(
                    csv_folder_name + "/disaggregated_network",
                    name,
                    ignore_standard_types,
                )

            if self.args["method"]["market_optimization"]:
                try:
                    self.market_model = Network(
                        csv_folder_name + "/market",
                        name,
                        ignore_standard_types,
                    )
                except ValueError:
                    logger.warning(
                        """
                        Could not import a market_model but the selected
                        method in the args indicated that it should be there.
                        This happens when the exported network was not solved
                        yet.Run 'etrago.optimize()' to build and solve the
                        market model.
                        """
                    )

            self.get_clustering_data(csv_folder_name)

        else:
            logger.error("Set args or csv_folder_name")

    # Add functions
    get_args_setting = get_args_setting

    check_args = check_args

    geolocation_buses = geolocation_buses

    add_missing_components = add_missing_components

    load_shedding = load_shedding

    set_random_noise = set_random_noise

    set_q_national_loads = set_q_national_loads

    set_q_foreign_loads = set_q_foreign_loads

    foreign_links = foreign_links

    crossborder_capacity = crossborder_capacity

    convert_capital_costs = convert_capital_costs

    extendable = extendable

    extension = extension

    set_branch_capacity = set_branch_capacity

    decommissioning = decommissioning

    add_ch4_h2_correspondence = add_ch4_h2_correspondence

    spatial_clustering = run_spatial_clustering

    spatial_clustering_gas = run_spatial_clustering_gas

    skip_snapshots = skip_snapshots

    ehv_clustering = ehv_clustering

    snapshot_clustering = snapshot_clustering

    add_redispatch_generators = add_redispatch_generators

    build_market_model = build_market_model

    grid_optimization = grid_optimization

    market_optimization = market_optimization

    lopf = lopf

    optimize = optimize

    temporal_disaggregation = dispatch_disaggregation

    pf_post_lopf = run_pf_post_lopf

    spatial_disaggregation = run_disaggregation

    calc_results = calc_etrago_results

    calc_ac_export = ac_export

    calc_ac_export_per_country = ac_export_per_country

    calc_dc_export = dc_export

    calc_dc_export_per_country = dc_export_per_country

    export_to_csv = export_to_csv

    filter_links_by_carrier = filter_links_by_carrier

    german_network = german_network

    set_line_costs = set_line_costs

    set_trafo_costs = set_trafo_costs

    system_costs_germany = system_costs_germany

    drop_sectors = drop_sectors

    buses_by_country = buses_by_country

    update_busmap = update_busmap

    plot_grid = plot_grid

    plot_clusters = plot_clusters

    plot_carrier = plot_carrier

    plot_gas_generation = plot_gas_generation

    plot_gas_summary = plot_gas_summary

    plot_h2_generation = plot_h2_generation

    plot_h2_summary = plot_h2_summary

    plot_heat_loads = plot_heat_loads

    plot_heat_summary = plot_heat_summary

    plot_flexibility_usage = flexibility_usage

    demand_side_management = demand_side_management

    bev_flexibility_potential = bev_flexibility_potential

    heat_stores = heat_stores

    hydrogen_stores = hydrogen_stores

    delete_dispensable_ac_buses = delete_dispensable_ac_buses

    delete_irrelevant_oneports = delete_irrelevant_oneports

    get_clustering_data = get_clustering_data

    adjust_CH4_gen_carriers = adjust_CH4_gen_carriers

    manual_fixes_datamodel = manual_fixes_datamodel

    shifted_energy = shifted_energy

    post_contingency_analysis = post_contingency_analysis_lopf

    sclopf = iterate_sclopf

    def dc_lines(self):
        return self.filter_links_by_carrier("DC", like=False)

    def build_network_from_db(self):
        """Function that imports transmission grid from chosen database

        Returns
        -------
        None.

        """
        self.scenario = NetworkScenario(
            self.engine,
            self.session,
            version=self.args["gridversion"],
            start_snapshot=self.args["start_snapshot"],
            end_snapshot=self.args["end_snapshot"],
            scn_name=self.args["scn_name"],
        )

        self.network = self.scenario.build_network()

        self.extension()

        self.decommissioning()

        if "H2_grid" in self.network.buses.carrier.unique():
            self.add_ch4_h2_correspondence()

        logger.info("Imported network from db")

    def adjust_network(self):
        """
        Function that adjusts the network imported from the database according
        to given input-parameters.

        Returns
        -------
        None.

        """

        self.manual_fixes_datamodel()

        self.geolocation_buses()

        self.load_shedding()

        self.adjust_CH4_gen_carriers()

        self.set_random_noise(0.01)

        self.set_q_national_loads(cos_phi=0.9)

        self.set_q_foreign_loads(cos_phi=0.9)

        self.foreign_links()

        self.crossborder_capacity()

        self.set_branch_capacity()

        self.extendable(
            grid_max_D=self.args["extendable"]["upper_bounds_grid"][
                "grid_max_D"
            ],
            grid_max_abs_D=self.args["extendable"]["upper_bounds_grid"][
                "grid_max_abs_D"
            ],
            grid_max_foreign=self.args["extendable"]["upper_bounds_grid"][
                "grid_max_foreign"
            ],
            grid_max_abs_foreign=self.args["extendable"]["upper_bounds_grid"][
                "grid_max_abs_foreign"
            ],
        )

        self.convert_capital_costs()

        self.delete_dispensable_ac_buses()

        self.delete_irrelevant_oneports()

        set_control_strategies(self.network)

    def _ts_weighted(self, timeseries):
        return timeseries.mul(self.network.snapshot_weightings, axis=0)
