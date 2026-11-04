# -*- coding: utf-8 -*-
# Copyright 2015-2026
#  Flensburg University of Applied Sciences,
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

import json
import logging
import os

from pypsa.components import Network
from sqlalchemy.orm import sessionmaker
import pypsa
import pandas as pd
import numpy as np

if "READTHEDOCS" not in os.environ:
    from etrago.tools import db

from etrago import __version__
from etrago.analyze.calc_results import (
    ac_export,
    ac_export_per_country,
    calc_etrago_results,
    dc_export,
    dc_export_per_country,
    german_network,
    system_costs_germany,
)
from etrago.analyze.plot import (
    bev_flexibility_potential,
    demand_side_management,
    heat_stores,
    hydrogen_stores,
    plot_carrier,
    plot_clusters,
    plot_curtailment,
    plot_flexibility_usage,
    plot_gas_generation,
    plot_gas_summary,
    plot_grid,
    plot_h2_generation,
    plot_h2_summary,
    plot_heat_loads,
    plot_heat_summary,
    plot_residual_load,
    plot_stacked_gen,
    plot_storage_soc_sorted,
    plot_voltage,
)
from etrago.cluster.electrical import ehv_clustering, run_spatial_clustering
from etrago.cluster.gas import run_spatial_clustering_gas
from etrago.cluster.temporal import skip_snapshots, snapshot_clustering
from etrago.disaggregate.spatial import run_disaggregation
from etrago.disaggregate.temporal import dispatch_disaggregation
from etrago.execute import lopf, optimize, run_pf_post_lopf, run_power_flow
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
from etrago.tools.distribution_grids import add_simplified_distribution_grids
from etrago.tools.extendable import extendable
from etrago.tools.io import (
    NetworkScenario,
    add_ch4_h2_correspondence,
    decommissioning,
    extension,
)
from etrago.tools.utilities import (
    adjust_CH4_gen_carriers,
    adjust_chp_model,
    adjust_PtH2_model,
    buses_by_country,
    check_args,
    convert_capital_costs,
    crossborder_capacity,
    delete_irrelevant_oneports,
    drop_sectors,
    export_to_csv,
    filter_links_by_carrier,
    foreign_links,
    geolocation_buses,
    get_args_setting,
    get_clustering_data,
    levelize_abroad_inland_parameters,
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
        results_folder_name=None,
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

            # Create results folder and store args
            if self.args["export_results_path"]:
                if not os.path.exists(self.args["export_results_path"]):
                    os.makedirs(
                        self.args["export_results_path"], exist_ok=True
                    )

                with open(
                    os.path.join(
                        self.args["export_results_path"], "args.json"
                    ),
                    "w",
                ) as fp:
                    json.dump(self.args, fp, indent=4)

        elif results_folder_name is not None:

            args_file = os.path.join(results_folder_name, "args.json")

            if os.path.isfile(args_file):
                self.get_args_setting(args_file)
            else:
                self.args = None

            if self.args:

                if "csv_export" in self.args:
                    self.network = Network(
                        results_folder_name, name, ignore_standard_types
                    )

                    if self.args["spatial_disaggregation"] is not None:
                        self.disaggregated_network = Network(
                            results_folder_name + "/disaggregated_network",
                            name,
                            ignore_standard_types,
                        )

                    if self.args["method"]["market_optimization"]["active"]:
                        try:
                            self.market_model = Network(
                                results_folder_name + "/market",
                                name,
                                ignore_standard_types,
                            )
                        except ValueError:
                            logger.warning("""
                                Could not import a market_model but the
                                selected method in the args indicated that it
                                should be there.
                                This happens when the exported network was
                                not solved yet.Run 'etrago.optimize()' to build
                                and solve the market model.
                                """)
                else:

                    # Import last network according to args
                    if self.args["pf_post_lopf"]["active"]:
                        try:
                            self.network = Network(
                                results_folder_name + "/non_linear_powerflow"
                            )
                        except FileNotFoundError:
                            self.network = Network(
                                results_folder_name + "/pf_post_lopf"
                            )
                    elif self.args["temporal_disaggregation"]["active"]:
                        self.network = Network(
                            results_folder_name + "/temporal_disaggregation"
                        )
                    else:
                        self.network = Network(
                            results_folder_name + "/grid_optimization"
                        )

                    if self.args["method"]["market_optimization"]["active"]:
                        try:
                            self.market_model = Network(
                                results_folder_name + "/market_optimization",
                                name,
                                ignore_standard_types,
                            )
                        except ValueError:
                            logger.warning("""
                                Could not import a market_model but the
                                selected method in the args indicated that it
                                should be there. This happens when the exported
                                network was not solved yet. Run
                                'etrago.optimize()' to build and solve the
                                market model.
                                """)

                    if self.args["spatial_disaggregation"] is not None:
                        self.disaggregated_network = Network(
                            results_folder_name + "/disaggregated_network.nc",
                            name,
                            ignore_standard_types,
                        )

                    self.get_clustering_data(results_folder_name)

            else:

                logger.warning(f"""
                    No args.json in {results_folder_name} available.
                    """)

                try:
                    self.network = Network(
                        results_folder_name, name, ignore_standard_types
                    )
                except ValueError:
                    logger.info("""
                        No network found.
                        """)

                try:
                    self.disaggregated_network = Network(
                        results_folder_name + "/disaggregated_network",
                        name,
                        ignore_standard_types,
                    )
                except ValueError:
                    logger.info("""
                        No disaggregated network available.
                        """)

                try:
                    self.market_model = Network(
                        results_folder_name + "/market",
                        name,
                        ignore_standard_types,
                    )
                except ValueError:
                    logger.info("""
                            No separate market model available.
                            """)

        else:
            logger.error("Set args or results_folder_name")

    # Add functions
    get_args_setting = get_args_setting

    check_args = check_args

    geolocation_buses = geolocation_buses

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
    
    power_flow = run_power_flow

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

    plot_residual_load = plot_residual_load

    plot_stacked_gen = plot_stacked_gen

    plot_curtailment = plot_curtailment

    plot_voltage = plot_voltage

    plot_storage_soc_sorted = plot_storage_soc_sorted

    plot_grid = plot_grid

    plot_clusters = plot_clusters

    plot_carrier = plot_carrier

    plot_gas_generation = plot_gas_generation

    plot_gas_summary = plot_gas_summary

    plot_h2_generation = plot_h2_generation

    plot_h2_summary = plot_h2_summary

    plot_heat_loads = plot_heat_loads

    plot_heat_summary = plot_heat_summary

    plot_flexibility_usage = plot_flexibility_usage

    demand_side_management = demand_side_management

    bev_flexibility_potential = bev_flexibility_potential

    heat_stores = heat_stores

    hydrogen_stores = hydrogen_stores

    delete_irrelevant_oneports = delete_irrelevant_oneports

    get_clustering_data = get_clustering_data

    adjust_CH4_gen_carriers = adjust_CH4_gen_carriers

    manual_fixes_datamodel = manual_fixes_datamodel

    post_contingency_analysis = post_contingency_analysis_lopf

    sclopf = iterate_sclopf

    adjust_PtH2_model = adjust_PtH2_model

    adjust_chp_model = adjust_chp_model

    levelize_abroad_inland_parameters = levelize_abroad_inland_parameters

    add_simplified_distribution_grids = add_simplified_distribution_grids

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

        if ("H2_grid" in self.network.buses.carrier.unique()) & (
            "H2_grid" not in self.network.links.carrier.unique()
        ):
            self.add_ch4_h2_correspondence()

        logger.info("Imported network from db")

        if self.args["method"]["distribution_grids"]:
            self.add_simplified_distribution_grids()
            logger.info("Added simplified distribution grids")

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

        self.delete_irrelevant_oneports()

        set_control_strategies(self.network)

        if self.args["scn_name"] in [
            "eGon100RE",
            "powerd2025",
            "powerd2030",
            "powerd2035",
        ]:
            self.levelize_abroad_inland_parameters()

    def _ts_weighted(self, timeseries):
        return timeseries.mul(self.network.snapshot_weightings, axis=0)
    
    
def import_egon_results_network(etrago, path):

    # import eGo results
    ego = pypsa.Network(path + "desaggregated/new_lowflex_peak_week.nc")

    # take over the entire network (incl. buses, lines, snapshots, ...)
    etrago.network = ego
    network = etrago.network
    
    # remove DC links within DE 
    # anyhow no components attached
    buses_to_remove = ["13824", "13665", "13478", "12770"]
    links_to_remove = network.links.index[
        network.links["bus0"].isin(buses_to_remove) |
        network.links["bus1"].isin(buses_to_remove)
    ]
    # remove links individually
    for link in links_to_remove:
        network.remove("Link", link)
    
    # remove buses individually
    for bus in buses_to_remove:
        if bus in network.buses.index:
            network.remove("Bus", bus)
    
    # # set country of buses with missing indication to DE
    # missing_country = (
    #     network.buses["country"].isna()
    #         | network.buses["country"].isin(["", "None", "nan"])
    #     )
    # network.buses.loc[missing_country, "country"] = "DE"
    
    # set country of lines based on the countries of their buses
    bus0_country = network.lines["bus0"].map(network.buses["country"])
    bus1_country = network.lines["bus1"].map(network.buses["country"])
    
    network.lines["country"] = np.where(
        bus0_country == bus1_country,
        bus0_country,                  # inside one country, e.g. "DE"
        bus0_country + bus1_country,   # cross-border, e.g. "DEGB"
    )

    # set result timeseries as input for pf: p_set = p (p0 for links)
    result_to_set = {
        "generators_t": "p",
        "storage_units_t": "p",
        "stores_t": "p",
        "links_t": "p0",
    }
    for container_name, result_attr in result_to_set.items():
        container = getattr(network, container_name)
        if result_attr in container:
            container["p_set"] = container[result_attr].copy()

    # set extendable to False
    network.generators["p_nom_extendable"] = False
    network.storage_units["p_nom_extendable"] = False
    network.stores["e_nom_extendable"] = False
    network.links["p_nom_extendable"] = False
    network.lines["s_nom_extendable"] = False
    if len(network.transformers) > 0:
        network.transformers["s_nom_extendable"] = False

    # delete former result timeseries
    keep_attrs = {
        "q_set", "p_set", "p_min_pu", "p_max_pu",
        "e_min_pu", "e_max_pu", "efficiency",
    }
    components_t = [
        network.generators_t,
        network.links_t,
        network.storage_units_t,
        network.stores_t,
        network.loads_t,
        network.buses_t,
        network.lines_t,
        network.transformers_t,
    ]
    for container in components_t:
        for attr in list(container.keys()):
            if attr not in keep_attrs:
                del container[attr]
                
    # dc-links: drop time-variant efficiency
    eff_t = network.links_t.efficiency
    dc_idx = network.links.index[network.links.carrier == "DC"]
    network.links_t.efficiency = eff_t.drop(
        columns=eff_t.columns.intersection(dc_idx)
    )

    # check: other links efficiency
    eff_t = network.links_t.efficiency
    bad = eff_t.columns[(eff_t.isna() | (eff_t <= 0)).any()]
    if len(bad) > 0:
        logger.warning(
            "Links mit Effizienz-Zeitreihe <= 0 oder NaN: "
            f"{network.links.loc[bad, 'carrier'].value_counts().to_dict()}"
        )
        
        
def import_dc_links_results(etrago, path):
    # transfer cross-border link flows from the clustered network into the
    # etrago network as fixed set points (links_t.p_set) for a power flow
    # as the imported disaggregated network does not contains those results
    # therefore sum up flows between all countries
    # and distribute results among resulting DC links in our network

    clus_network = pypsa.Network(path + "clustered")
    target_snapshots = etrago.network.snapshots

    # check snapshots of both network
    missing = target_snapshots.difference(clus_network.snapshots)
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} target snapshots are missing in the source "
            f"network, e.g. {list(missing[:3])}. Resolutions do not match."
        )

    def _pair_and_sign(net, links):
        # identify country pairs and define sign per link
        # +1 if the link's own direction (bus0 -> bus1) points in the pair's reference direction pair[0] -> pair[1]
        # special case: links within one country (reference direction is defined geographically as west -> east)

        c = net.buses["country"]
        x = net.buses["x"]

        # country at each end of the link
        c0, c1 = links.bus0.map(c), links.bus1.map(c)

        # direction-independent pair
        pair = pd.Series(
            [tuple(sorted((a, b))) for a, b in zip(c0, c1)], index=links.index
        )

        # default rule: +1 if the link starts in the first country of the pair
        sign = pd.Series(
            np.where(c0 == pair.str[0], 1.0, -1.0), index=links.index
        )

        # same-country links: +1 if the link points from west to east
        same = c0 == c1
        sign[same] = np.where(
            links.loc[same, "bus0"].map(x) <= links.loc[same, "bus1"].map(x),
            1.0,
            -1.0,
        )

        # buses without a country produce pairs containing NaN; these can
        # never be matched between the networks, so make them visible
        if c0.isna().any() or c1.isna().any():
            logger.warning(
                f"{int((c0.isna() | c1.isna()).sum())} DC links with a bus "
                f"without country: "
                f"{links.index[c0.isna() | c1.isna()].tolist()}"
            )
        return pair, sign

    # source network: net flow per country pair
    clus = clus_network.links[clus_network.links.carrier == "DC"].copy()
    clus["pair"], clus["sign"] = _pair_and_sign(clus_network, clus)

    # p0 = power withdrawn at bus0 = flow from bus0 to bus1 (positive means
    # bus0 -> bus1); only target snapshots
    p0 = clus_network.links_t.p0.reindex(
        index=target_snapshots, columns=clus.index
    )
    if p0.isna().any().any():
        logger.warning(
            f"Source network: p0 missing for "
            f"{int(p0.isna().any().sum())} DC links, set to 0"
        )

    # snapshots x pairs, value = net flow pair[0] -> pair[1] in MW.
    exchange = (
        p0.fillna(0)
        .mul(clus["sign"], axis=1)
        .T.groupby(clus["pair"])
        .sum()
        .T
    )

    # optimised capacity per pair in the source network
    clus_cap = clus.groupby("pair")["p_nom_opt"].sum()

    # target network: same classification of its DC links
    dc_links = etrago.network.links[
        etrago.network.links.carrier == "DC"
    ].copy()
    dc_links["pair"], dc_links["sign"] = _pair_and_sign(
        etrago.network, dc_links
    )

    # pairs that exist in only one of the networks cannot be mapped:
    only_src = set(clus["pair"]) - set(dc_links["pair"])
    only_dst = set(dc_links["pair"]) - set(clus["pair"])
    if only_src:
        logger.warning(
            f"Country pairs only in source network (flow is lost): {only_src}"
        )
    if only_dst:
        logger.warning(
            f"Country pairs only in target network (p_set stays 0): {only_dst}"
        )

    # capacities in the target network
    # distribution key within a pair: each link's share of the pair's
    # existing p_nom; if all links of a pair have p_nom = 0, split equally
    old_sum = dc_links.groupby("pair")["p_nom"].transform("sum")
    n_links = dc_links.groupby("pair")["p_nom"].transform("size")
    key = (dc_links["p_nom"] / old_sum).where(old_sum > 0, 1 / n_links)

    # optimised pair capacity from the source, split by the key
    pair_cap = dc_links["pair"].map(clus_cap)
    new_p_nom = (pair_cap * key).where(pair_cap.notna())

    # never reduce an existing capacity; pairs without source keep p_nom
    dc_links["p_nom"] = np.maximum(new_p_nom, dc_links["p_nom"]).fillna(
        dc_links["p_nom"]
    )

    # write back: fixed capacities, no extension
    etrago.network.links.loc[dc_links.index, "p_nom"] = dc_links["p_nom"]
    etrago.network.links.loc[dc_links.index, "p_nom_extendable"] = False

    # split each pair's flow onto its target links
    pair_p_nom = dc_links.groupby("pair")["p_nom"].transform("sum")
    dc_links["share"] = (dc_links["p_nom"] / pair_p_nom).where(
        pair_p_nom > 0, 1 / n_links
    )

    # technical limits per link in its own direction bus0 -> bus1
    upper = dc_links["p_nom"] * dc_links["p_max_pu"]
    lower = dc_links["p_nom"] * dc_links["p_min_pu"]

    p_set = etrago.network.links_t.p_set
    for link_id, row in dc_links.iterrows():
        pair = row["pair"]

        # pair unknown in the source -> explicitly 0, so that no stale or
        # NaN value from earlier steps remains in p_set
        if pair not in exchange.columns:
            p_set.loc[target_snapshots, link_id] = 0.0
            continue

        # flow of this link in its own direction:
        #   exchange[pair]  ... pair flow in reference direction
        #   * sign          ... turn it into this link's direction
        #   * share         ... this link's part of the pair flow
        desired = exchange[pair].to_numpy() * row["sign"] * row["share"]

        # safety net: respect the link's capacity
        values = np.clip(desired, lower[link_id], upper[link_id])

        # if clipping happened, the pair sum no longer matches the source;
        # report it so the resulting imbalance is traceable
        curtailed = np.abs(desired - values)
        if curtailed.max() > 1e-6:
            logger.warning(
                f"DC link {link_id} ({pair}): p_set limited to capacity in "
                f"{(curtailed > 1e-6).sum()} snapshots, "
                f"max {curtailed.max():.1f} MW"
            )

        p_set.loc[target_snapshots, link_id] = values
    etrago.network.links_t.p_set = p_set

    # check: recompute pair flows from the written p_set
    dst = (
        p_set.loc[target_snapshots, dc_links.index]
        .mul(dc_links["sign"], axis=1)
        .T.groupby(dc_links["pair"])
        .sum()
        .T
    )
    common = exchange.columns.intersection(dst.columns)
    dev = (dst[common] - exchange[common]).abs().max()
    if dev.max() > 1:
        logger.warning(
            f"Pair sums deviate from source (MW):\n{dev[dev > 1].round(1)}"
        )

    return exchange, dst


        
    
    
    
    





