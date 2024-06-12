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
Utilities.py includes a wide range of useful functions.
"""

from collections.abc import Mapping
from copy import deepcopy
import json
import logging
import math
import os

from pyomo.environ import Constraint, PositiveReals, Var
import numpy as np
import pandas as pd
import pypsa
import sqlalchemy.exc

if "READTHEDOCS" not in os.environ:
    from shapely.geometry import Point
    import geopandas as gpd

    from etrago.tools import db

logger = logging.getLogger(__name__)


__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = """ulfmueller, s3pp, wolfbunke, mariusves, lukasol, ClaraBuettner,
CarlosEpia, gnn, pieterhexen, fwitte, KathiEsterl, MGlauer, birgits,
 AmeliaNadal, MarlonSchlemminger, wheitkoetter, jankaeh"""


def filter_links_by_carrier(self, carrier, like=True):
    """

    Parameters
    ----------
    carrier : list or str
        name of the carriers of interest. Can be a list of carriers or single
        sting.
    like : bool, optional
        When like set to True, the links with carrier names that includes the
        carrier(s) supplied are returned, Not just exact matches.
        The default is True.

    Returns
    -------
    df : pandas.DataFrame object
        Dataframe that contains just links with carriers of the types given
        in the argument carrier.

    """
    if isinstance(carrier, str):
        if like:
            df = self.network.links[
                self.network.links.carrier.str.contains(carrier)
            ]
        else:
            df = self.network.links[self.network.links.carrier == carrier]
    elif isinstance(carrier, list):
        df = self.network.links[self.network.links.carrier.isin(carrier)]
    return df


def buses_of_vlvl(network, voltage_level):
    """Get bus-ids of given voltage level(s).

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    voltage_level: list

    Returns
    -------
    list
        List containing bus-ids.
    """

    mask = network.buses.v_nom.isin(voltage_level)
    df = network.buses[mask]

    return df.index


def buses_grid_linked(network, voltage_level):
    """Get bus-ids of a given voltage level connected to the grid.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    voltage_level: list

    Returns
    -------
    list
        List containing bus-ids.
    """

    mask = (
        network.buses.index.isin(network.lines.bus0)
        | (network.buses.index.isin(network.lines.bus1))
        | (
            network.buses.index.isin(
                network.links.loc[network.links.carrier == "DC", "bus0"]
            )
        )
        | (
            network.buses.index.isin(
                network.links.loc[network.links.carrier == "DC", "bus1"]
            )
        )
    ) & (network.buses.v_nom.isin(voltage_level))

    df = network.buses[mask]

    return df.index


def geolocation_buses(self, apply_on="grid_model"):
    """
    If geopandas is installed:
    Use geometries of buses x/y(lon/lat) and polygons
    of countries from RenpassGisParameterRegion
    in order to locate the buses

    Else:
    Use coordinats of buses to locate foreign buses, which is less accurate.

    TODO: Why not alway use geopandas??

    Parameters
    ----------
    etrago : :class:`etrago.Etrago`
       Transmission grid object
    apply_on: str
        State if this function is applied on the grid_model or the
        market_model. The market_model options can only be used if the method
        type is "market_grid".
    """

    if apply_on == "grid_model":
        network = self.network
    elif apply_on == "market_model":
        network = self.market_model
    elif apply_on == "pre_market_model":
        network = self.pre_market_model
    else:
        logger.warning(
            """Parameter apply_on must be either 'grid_model' or 'market_model'
            or 'pre_market_model'."""
        )

    transborder_lines_0 = network.lines[
        network.lines["bus0"].isin(
            network.buses.index[network.buses["country"] != "DE"]
        )
    ].index
    transborder_lines_1 = network.lines[
        network.lines["bus1"].isin(
            network.buses.index[network.buses["country"] != "DE"]
        )
    ].index

    # set country tag for lines
    network.lines.loc[transborder_lines_0, "country"] = network.buses.loc[
        network.lines.loc[transborder_lines_0, "bus0"].values, "country"
    ].values

    network.lines.loc[transborder_lines_1, "country"] = network.buses.loc[
        network.lines.loc[transborder_lines_1, "bus1"].values, "country"
    ].values
    network.lines["country"].fillna("DE", inplace=True)
    doubles = list(set(transborder_lines_0.intersection(transborder_lines_1)))
    for line in doubles:
        c_bus0 = network.buses.loc[network.lines.loc[line, "bus0"], "country"]
        c_bus1 = network.buses.loc[network.lines.loc[line, "bus1"], "country"]
        network.lines.loc[line, "country"] = "{}{}".format(c_bus0, c_bus1)

    transborder_links_0 = network.links[
        network.links["bus0"].isin(
            network.buses.index[network.buses["country"] != "DE"]
        )
    ].index
    transborder_links_1 = network.links[
        network.links["bus1"].isin(
            network.buses.index[network.buses["country"] != "DE"]
        )
    ].index

    # set country tag for links
    network.links.loc[transborder_links_0, "country"] = network.buses.loc[
        network.links.loc[transborder_links_0, "bus0"].values, "country"
    ].values

    network.links.loc[transborder_links_1, "country"] = network.buses.loc[
        network.links.loc[transborder_links_1, "bus1"].values, "country"
    ].values
    network.links["country"].fillna("DE", inplace=True)
    doubles = list(set(transborder_links_0.intersection(transborder_links_1)))
    for link in doubles:
        c_bus0 = network.buses.loc[network.links.loc[link, "bus0"], "country"]
        c_bus1 = network.buses.loc[network.links.loc[link, "bus1"], "country"]
        network.links.loc[link, "country"] = "{}{}".format(c_bus0, c_bus1)

    return network


def buses_by_country(self, apply_on="grid_model"):
    """
    Find buses of foreign countries using coordinates
    and return them as Pandas Series

    Parameters
    ----------
    self : Etrago object
        Overall container of PyPSA
    apply_on: str
        State if this function is applied on the grid_model or the
        market_model. The market_model options can only be used if the method
        type is "market_grid".

    Returns
    -------
    None
    """

    if apply_on == "grid_model":
        network = self.network
    elif apply_on == "market_model":
        network = self.market_model
    elif apply_on == "pre_market_model":
        network = self.pre_market_model
    else:
        logger.warning(
            """Parameter apply_on must be either 'grid_model' or 'market_model'
            or 'pre_market_model'."""
        )

    countries = {
        "Poland": "PL",
        "Czechia": "CZ",
        "Denmark": "DK",
        "Sweden": "SE",
        "Austria": "AT",
        "Switzerland": "CH",
        "Netherlands": "NL",
        "Luxembourg": "LU",
        "France": "FR",
        "Belgium": "BE",
        "United Kingdom": "GB",
        "Norway": "NO",
        "Finland": "FI",
        "Germany": "DE",
        "Russia": "RU",
    }

    # read Germany borders from egon-data
    query = "SELECT * FROM boundaries.vg250_lan"
    con = self.engine
    germany_sh = gpd.read_postgis(query, con, geom_col="geometry")

    path = gpd.datasets.get_path("naturalearth_lowres")
    shapes = gpd.read_file(path)
    shapes = shapes[shapes.name.isin([*countries])].set_index(keys="name")

    # Use Germany borders from egon-data if not using the SH test case
    if len(germany_sh.gen.unique()) > 1:
        shapes.at["Germany", "geometry"] = germany_sh.geometry.unary_union

    geobuses = network.buses.copy()
    geobuses["geom"] = geobuses.apply(
        lambda x: Point([x["x"], x["y"]]), axis=1
    )

    geobuses = gpd.GeoDataFrame(
        data=geobuses, geometry="geom", crs="EPSG:4326"
    )
    geobuses["country"] = np.nan

    for country in countries:
        geobuses["country"][
            network.buses.index.isin(
                geobuses.clip(shapes[shapes.index == country]).index
            )
        ] = countries[country]

    shapes = shapes.to_crs(3035)
    geobuses = geobuses.to_crs(3035)

    for bus in geobuses[geobuses["country"].isna()].index:
        distances = shapes.distance(geobuses.loc[bus, "geom"])
        closest = distances.idxmin()
        geobuses.loc[bus, "country"] = countries[closest]

    network.buses = geobuses.drop(columns="geom")

    return


def clip_foreign(network):
    """
    Delete all components and timelines located outside of Germany.
    If applied after optimization, transborder flows divided by country of
    origin are added as network.foreign_trade.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    """

    # get foreign buses by country

    foreign_buses = network.buses[network.buses.country != "DE"]
    network.buses = network.buses.drop(
        network.buses.loc[foreign_buses.index].index
    )

    if not network.lines_t.p0.empty:
        # identify transborder lines
        # TODO: Add links!
        transborder_lines = network.lines.query("country != 'DE'")
        transborder_lines["bus0"] = network.lines["bus0"]
        transborder_lines["bus1"] = network.lines["bus1"]
        transborder_lines["country"] = network.lines.country

        # identify amount of flows per line and group to get flow per country
        transborder_flows = network.lines_t.p0[transborder_lines.index]
        for i in transborder_flows.columns:
            if network.lines.loc[str(i)]["bus1"] in foreign_buses.index:
                transborder_flows.loc[:, str(i)] = (
                    transborder_flows.loc[:, str(i)] * -1
                )

        network.foreign_trade = transborder_flows.groupby(
            transborder_lines["country"], axis=1
        ).sum()

    # drop foreign components
    network.lines = network.lines.drop(
        network.lines[
            ~(network.lines["bus0"].isin(network.buses.index))
            | ~(network.lines["bus1"].isin(network.buses.index))
        ].index
    )

    network.links = network.links.drop(
        network.links[
            ~(network.links["bus0"].isin(network.buses.index))
            | ~(network.links["bus1"].isin(network.buses.index))
        ].index
    )

    network.transformers = network.transformers.drop(
        network.transformers[
            ~(network.transformers["bus0"].isin(network.buses.index))
            | ~(network.transformers["bus1"].isin(network.buses.index))
        ].index
    )
    network.generators = network.generators.drop(
        network.generators[
            ~(network.generators["bus"].isin(network.buses.index))
        ].index
    )
    network.loads = network.loads.drop(
        network.loads[~(network.loads["bus"].isin(network.buses.index))].index
    )
    network.storage_units = network.storage_units.drop(
        network.storage_units[
            ~(network.storage_units["bus"].isin(network.buses.index))
        ].index
    )

    components = [
        "loads",
        "generators",
        "lines",
        "buses",
        "transformers",
        "links",
    ]
    for g in components:  # loads_t
        h = g + "_t"
        nw = getattr(network, h)  # network.loads_t
        for i in nw.keys():  # network.loads_t.p
            cols = [
                j
                for j in getattr(nw, i).columns
                if j not in getattr(network, g).index
            ]
            for k in cols:
                del getattr(nw, i)[k]

    return network


def foreign_links(self):
    """Change transmission technology of foreign lines from AC to DC (links).

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    """
    if self.args["foreign_lines"]["carrier"] == "DC":
        network = self.network

        foreign_buses = network.buses[
            (network.buses.country != "DE")
            & (network.buses.carrier.isin(["AC", "DC"]))
        ]

        foreign_lines = network.lines[
            network.lines.bus0.astype(str).isin(foreign_buses.index)
            | network.lines.bus1.astype(str).isin(foreign_buses.index)
        ]

        foreign_links = network.links[
            (
                network.links.bus0.astype(str).isin(foreign_buses.index)
                | network.links.bus1.astype(str).isin(foreign_buses.index)
            )
            & (network.links.carrier == "DC")
        ]

        network.links.loc[foreign_links.index, "p_min_pu"] = -1

        network.links.loc[foreign_links.index, "efficiency"] = 1

        network.links.loc[foreign_links.index, "carrier"] = "DC"

        network.import_components_from_dataframe(
            foreign_lines.loc[:, ["bus0", "bus1", "capital_cost", "length"]]
            .assign(p_nom=foreign_lines.s_nom)
            .assign(p_nom_min=foreign_lines.s_nom_min)
            .assign(p_nom_max=foreign_lines.s_nom_max)
            .assign(p_nom_extendable=foreign_lines.s_nom_extendable)
            .assign(p_max_pu=foreign_lines.s_max_pu)
            .assign(p_min_pu=-1)
            .assign(carrier="DC")
            .set_index("N" + foreign_lines.index),
            "Link",
        )

        network.lines = network.lines.drop(foreign_lines.index)

        self.geolocation_buses()


def set_q_national_loads(self, cos_phi):
    """
    Set q component of national loads based on the p component and cos_phi

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    cos_phi : float
        Choose ration of active and reactive power of foreign loads

    Returns
    -------
    network : :class:`pypsa.Network`
        Overall container of PyPSA

    """
    network = self.network

    national_buses = network.buses[
        (network.buses.country == "DE") & (network.buses.carrier == "AC")
    ]

    # Calculate q national loads based on p and cos_phi
    new_q_loads = network.loads_t["p_set"].loc[
        :,
        network.loads.index[
            (network.loads.bus.astype(str).isin(national_buses.index))
            & (network.loads.carrier.astype(str) == "AC")
        ],
    ] * math.tan(math.acos(cos_phi))

    # insert the calculated q in loads_t. Only loads without previous
    # assignment are affected
    network.loads_t.q_set = pd.merge(
        network.loads_t.q_set,
        new_q_loads,
        how="inner",
        right_index=True,
        left_index=True,
        suffixes=("", "delete_"),
    )
    network.loads_t.q_set.drop(
        [i for i in network.loads_t.q_set.columns if "delete" in i],
        axis=1,
        inplace=True,
    )


def set_q_foreign_loads(self, cos_phi):
    """Set reative power timeseries of loads in neighbouring countries

    Parameters
    ----------
    etrago : :class:`etrago.Etrago
        Transmission grid object
    cos_phi: float
        Choose ration of active and reactive power of foreign loads

    Returns
    -------
    None

    """
    network = self.network

    foreign_buses = network.buses[
        (network.buses.country != "DE") & (network.buses.carrier == "AC")
    ]

    network.loads_t["q_set"].loc[
        :,
        network.loads.index[
            (network.loads.bus.astype(str).isin(foreign_buses.index))
            & (network.loads.carrier != "H2_for_industry")
        ].astype(int),
    ] = network.loads_t["p_set"].loc[
        :,
        network.loads.index[
            (network.loads.bus.astype(str).isin(foreign_buses.index))
            & (network.loads.carrier != "H2_for_industry")
        ],
    ].values * math.tan(
        math.acos(cos_phi)
    )

    # To avoid a problem when the index of the load is the weather year,
    # the column names were temporarily set to `int` and changed back to
    # `str`.
    network.loads_t["q_set"].columns = network.loads_t["q_set"].columns.astype(
        str
    )


def connected_grid_lines(network, busids):
    """Get grid lines connected to given buses.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    busids  : list
        List containing bus-ids.

    Returns
    -------
    :class:`pandas.DataFrame
        PyPSA lines.
    """

    mask = network.lines.bus1.isin(busids) | network.lines.bus0.isin(busids)

    return network.lines[mask]


def connected_transformer(network, busids):
    """Get transformer connected to given buses.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    busids  : list
        List containing bus-ids.

    Returns
    -------
    :class:`pandas.DataFrame
        PyPSA transformer.
    """

    mask = network.transformers.bus0.isin(busids)

    return network.transformers[mask]


def load_shedding(self, temporal_disaggregation=False, **kwargs):
    """Implement load shedding in existing network to identify
    feasibility problems

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    marginal_cost : int
        Marginal costs for load shedding
    p_nom : int
        Installed capacity of load shedding generator

    Returns
    -------

    """
    logger.debug("Shedding the load.")
    if self.args["load_shedding"]:
        if temporal_disaggregation:
            network = self.network_tsa
        else:
            network = self.network

        marginal_cost_def = 10000  # network.generators.marginal_cost.max()*2
        p_nom_def = network.loads_t.p_set.max().max()

        marginal_cost = kwargs.get("marginal_cost", marginal_cost_def)
        p_nom = kwargs.get("p_nom", p_nom_def)

        network.add("Carrier", "load")
        start = (
            network.generators.index.to_series()
            .str.rsplit(" ")
            .str[0]
            .astype(int)
            .sort_values()
            .max()
            + 1
        )

        if start != start:
            start = 0

        index = list(range(start, start + len(network.buses.index)))
        network.import_components_from_dataframe(
            pd.DataFrame(
                dict(
                    marginal_cost=marginal_cost,
                    p_nom=p_nom,
                    carrier="load shedding",
                    bus=network.buses.index,
                    control="PQ",
                ),
                index=index,
            ),
            "Generator",
        )


def set_control_strategies(network):
    """Sets control strategies for AC generators and storage units

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    None.

    """
    # Assign generators control strategy
    network.generators.loc[:, "control"] = "PV"

    network.generators.loc[
        network.generators.carrier.isin(
            [
                "load shedding",
                "CH4",
                "CH4_biogas",
                "CH4_NG",
                "central_biomass_CHP_heat",
                "geo_thermal",
                "solar_thermal_collector",
            ]
        ),
        "control",
    ] = "PQ"

    # Assign storage units control strategy
    network.storage_units.loc[:, "control"] = "PV"


def data_manipulation_sh(network):
    """Adds missing components to run calculations with SH scenarios.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    None

    """
    from geoalchemy2.shape import from_shape, to_shape
    from shapely.geometry import LineString, MultiLineString, Point

    # add connection from Luebeck to Siems
    new_bus = str(network.buses.index.astype(np.int64).max() + 1)
    new_trafo = str(network.transformers.index.astype(np.int64).max() + 1)
    new_line = str(network.lines.index.astype(np.int64).max() + 1)
    network.add(
        "Bus", new_bus, carrier="AC", v_nom=220, x=10.760835, y=53.909745
    )
    network.add(
        "Transformer",
        new_trafo,
        bus0="25536",
        bus1=new_bus,
        x=1.29960,
        tap_ratio=1,
        s_nom=1600,
    )
    network.add(
        "Line", new_line, bus0="26387", bus1=new_bus, x=0.0001, s_nom=1600
    )
    network.lines.loc[new_line, "cables"] = 3.0

    # bus geom
    point_bus1 = Point(10.760835, 53.909745)
    network.buses.set_value(new_bus, "geom", from_shape(point_bus1, 4326))

    # line geom/topo
    network.lines.set_value(
        new_line,
        "geom",
        from_shape(
            MultiLineString(
                [
                    LineString(
                        [to_shape(network.buses.geom["26387"]), point_bus1]
                    )
                ]
            ),
            4326,
        ),
    )
    network.lines.set_value(
        new_line,
        "topo",
        from_shape(
            LineString([to_shape(network.buses.geom["26387"]), point_bus1]),
            4326,
        ),
    )

    # trafo geom/topo
    network.transformers.set_value(
        new_trafo,
        "geom",
        from_shape(
            MultiLineString(
                [
                    LineString(
                        [to_shape(network.buses.geom["25536"]), point_bus1]
                    )
                ]
            ),
            4326,
        ),
    )
    network.transformers.set_value(
        new_trafo,
        "topo",
        from_shape(
            LineString([to_shape(network.buses.geom["25536"]), point_bus1]),
            4326,
        ),
    )


def _enumerate_row(row):
    row["name"] = row.name
    return row


def export_to_csv(self, path):
    """Write calculation results to csv-files in `path`.

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    args: dict
        Contains calculation settings of appl.py
    path: str or False or None
        Choose path for csv-files. Specify `""`, `False` or `None` to
        not do anything.

    Returns
    -------
    None

    """
    if not path:
        pass
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    self.network.export_to_csv_folder(path)
    data = pd.read_csv(os.path.join(path, "network.csv"))
    # data['time'] = network.results['Solver'].Time
    data = data.apply(_enumerate_row, axis=1)
    data.to_csv(os.path.join(path, "network.csv"), index=False)

    with open(os.path.join(path, "args.json"), "w") as fp:
        json.dump(self.args, fp, indent=4)

    if hasattr(self.network, "Z"):
        file = [
            i for i in os.listdir(path.strip("0123456789")) if i == "Z.csv"
        ]
        if file:
            print("Z already calculated")
        else:
            self.network.Z.to_csv(
                path.strip("0123456789") + "/Z.csv", index=False
            )

    if bool(self.busmap):
        path_clus = os.path.join(path, "clustering")
        if not os.path.exists(path_clus):
            os.makedirs(path_clus, exist_ok=True)

        with open(os.path.join(path_clus, "busmap.json"), "w") as d:
            json.dump(self.busmap["busmap"], d, indent=4)
        self.busmap["orig_network"].export_to_csv_folder(path_clus)
        data = pd.read_csv(os.path.join(path_clus, "network.csv"))
        data = data.apply(_enumerate_row, axis=1)
        data.to_csv(os.path.join(path_clus, "network.csv"), index=False)

    if isinstance(self.ch4_h2_mapping, pd.Series):
        path_clus = os.path.join(path, "clustering")
        if not os.path.exists(path_clus):
            os.makedirs(path_clus, exist_ok=True)
        with open(os.path.join(path_clus, "ch4_h2_mapping.json"), "w") as d:
            self.ch4_h2_mapping.to_json(d, indent=4)

    return


def loading_minimization(network, snapshots):
    """
    Minimizes the sum of the products of each element in the passive_branches
    of the model.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : 'pandas.core.indexes.datetimes.DatetimeIndex'
        snapshots to perform the minimization

    Returns
    -------
    None

    """
    network.model.number1 = Var(
        network.model.passive_branch_p_index, within=PositiveReals
    )
    network.model.number2 = Var(
        network.model.passive_branch_p_index, within=PositiveReals
    )

    def cRule(model, c, l0, t):
        return (
            model.number1[c, l0, t] - model.number2[c, l0, t]
            == model.passive_branch_p[c, l0, t]
        )

    network.model.cRule = Constraint(
        network.model.passive_branch_p_index, rule=cRule
    )

    network.model.objective.expr += 0.00001 * sum(
        network.model.number1[i] + network.model.number2[i]
        for i in network.model.passive_branch_p_index
    )


def _make_consense(component, attr):
    """
    Returns a function `consense` that will be used to generate a consensus
    value for the attribute `attr` of the given `component`. This consensus
    value is derived from the input DataFrame `x`. If all values in the
    DataFrame are equal, the consensus value will be that common value.
    If all values are missing (NaN), the consensus value will be NaN.
    Otherwise, an assertion error will be raised.

    Parameters
    ----------
    component : str
        specify the name of the component being clustered.
    attr : str
        specify the name of the attribute of the commponent being considered.

    Returns
    -------
    function
        A function that takes a DataFrame as input and returns a single value
        as output when all the elements of the commponent attribute are the
        same.

    """

    def consense(x):
        v = x.iat[0]
        assert (x == v).all() or x.isnull().all(), (
            f"In {component} cluster {x.name} the values"
            f" of attribute {attr} do not agree:\n{x}"
        )
        return v

    return consense


def _normed(s):
    """
    Given a pandas Series `s`, normalizes the series by dividing each element
    by the sum of the series. If the sum of the series is zero, returns 1.0 to
    avoid division by zero errors.

    Parameters
    ----------
    s : pandas.Series
        A pandas Series.

    Returns
    -------
    pandas.Series
        A normalized pandas Series.

    """
    tot = s.sum()
    if tot == 0:
        return 1.0
    else:
        return s / tot


def agg_series_lines(l0, network):
    """
    Given a pandas DataFrame `l0` containing information about lines in a
    network and a network object, aggregates the data in `l0` for all its
    attributes. Returns a pandas Series containing the aggregated data.


    Parameters
    ----------
    l0: pandas.DataFrame
        contain information about lines in a network.
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    pandas.Series
        A pandas Series containing aggregated data for the lines in the
        network.

    """
    attrs = network.components["Line"]["attrs"]
    columns = set(
        attrs.index[attrs.static & attrs.status.str.startswith("Input")]
    ).difference(("name", "bus0", "bus1"))
    consense = {
        attr: _make_consense("Bus", attr)
        for attr in (
            columns
            # | {"sub_network"}
            - {
                "r",
                "x",
                "g",
                "b",
                "terrain_factor",
                "s_nom",
                "s_nom_min",
                "s_nom_max",
                "s_nom_extendable",
                "length",
                "v_ang_min",
                "v_ang_max",
            }
        )
    }

    Line = l0["Line"].iloc[0]
    data = dict(
        r=l0["r"].sum(),
        x=l0["x"].sum(),
        g=1.0 / (1.0 / l0["g"]).sum(),
        b=1.0 / (1.0 / l0["b"]).sum(),
        terrain_factor=l0["terrain_factor"].mean(),
        s_max_pu=(l0["s_max_pu"] * _normed(l0["s_nom"])).sum(),
        s_nom=l0["s_nom"].iloc[0],
        s_nom_min=l0["s_nom_min"].max(),
        s_nom_max=l0["s_nom_max"].min(),
        s_nom_extendable=l0["s_nom_extendable"].any(),
        num_parallel=l0["num_parallel"].max(),
        capital_cost=(_normed(l0["s_nom"]) * l0["capital_cost"]).sum(),
        length=l0["length"].sum(),
        v_ang_min=l0["v_ang_min"].max(),
        v_ang_max=l0["v_ang_max"].min(),
    )
    data.update((f, consense[f](l0[f])) for f in columns.difference(data))
    return pd.Series(
        data, index=[f for f in l0.columns if f in columns], name=Line
    )


def group_parallel_lines(network):
    """
    Function that groups parallel lines of the same voltage level to one
    line component representing all parallel lines

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    None.

    """

    def agg_parallel_lines(l0):
        attrs = network.components["Line"]["attrs"]
        columns = set(
            attrs.index[attrs.static & attrs.status.str.startswith("Input")]
        ).difference(("name", "bus0", "bus1"))
        columns.add("Line")
        columns.add("geom")
        consense = {
            attr: _make_consense("Bus", attr)
            for attr in (
                columns
                | {"sub_network"}
                - {
                    "Line",
                    "r",
                    "x",
                    "g",
                    "b",
                    "terrain_factor",
                    "s_nom",
                    "s_nom_min",
                    "s_nom_max",
                    "s_nom_extendable",
                    "length",
                    "v_ang_min",
                    "v_ang_max",
                    "geom",
                }
            )
        }

        data = dict(
            Line=l0["Line"].iloc[0],
            r=1.0 / (1.0 / l0["r"]).sum(),
            x=1.0 / (1.0 / l0["x"]).sum(),
            g=l0["g"].sum(),
            b=l0["b"].sum(),
            terrain_factor=l0["terrain_factor"].mean(),
            s_max_pu=(l0["s_max_pu"] * _normed(l0["s_nom"])).sum(),
            s_nom=l0["s_nom"].sum(),
            s_nom_min=l0["s_nom_min"].sum(),
            s_nom_max=l0["s_nom_max"].sum(),
            s_nom_extendable=l0["s_nom_extendable"].any(),
            num_parallel=l0["num_parallel"].sum(),
            capital_cost=(_normed(l0["s_nom"]) * l0["capital_cost"]).sum(),
            length=l0["length"].mean(),
            sub_network=consense["sub_network"](l0["sub_network"]),
            v_ang_min=l0["v_ang_min"].max(),
            v_ang_max=l0["v_ang_max"].min(),
            geom=l0["geom"].iloc[0],
        )
        data.update((f, consense[f](l0[f])) for f in columns.difference(data))
        return pd.Series(data, index=[f for f in l0.columns if f in columns])

    # Make bus0 always the greattest to identify repeated lines
    lines_2 = network.lines.copy()
    bus_max = lines_2.apply(lambda x: max(x.bus0, x.bus1), axis=1)
    bus_min = lines_2.apply(lambda x: min(x.bus0, x.bus1), axis=1)
    lines_2["bus0"] = bus_max
    lines_2["bus1"] = bus_min
    lines_2.reset_index(inplace=True)
    lines_2["geom"] = lines_2.apply(
        lambda x: None if x.geom is None else x.geom.wkt, axis=1
    )
    network.lines = (
        lines_2.groupby(["bus0", "bus1"])
        .apply(agg_parallel_lines)
        .reset_index()
        .set_index("Line", drop=True)
    )

    # network.lines["geom"] = gpd.GeoSeries.from_wkt(network.lines["geom"])

    return


def delete_dispensable_ac_buses(etrago):
    """
    Function that identifies and delete AC buses without links, transformers,
    generators, loads, stores or storage_units, which also are connected to
    just one or two other buses

    Parameters
    ----------
    etrago : etrago object

    Returns
    -------
    None.

    """
    if etrago.args["delete_dispensable_ac_buses"] is False:
        return

    def delete_buses(delete_buses, network):
        drop_buses = delete_buses.index.to_list()
        network.buses.drop(labels=drop_buses, inplace=True)
        drop_lines = network.lines.index[
            (network.lines.bus0.isin(drop_buses))
            | (network.lines.bus1.isin(drop_buses))
        ].to_list()
        network.lines.drop(labels=drop_lines, inplace=True)
        drop_storage_units = network.storage_units.index[
            (network.storage_units.bus.isin(drop_buses))
        ].to_list()
        network.storage_units.drop(drop_storage_units, inplace=True)
        drop_generators = network.generators.index[
            (network.generators.bus.isin(drop_buses))
        ].to_list()
        network.generators.drop(drop_generators, inplace=True)
        return (
            network.buses,
            network.lines,
            network.storage_units,
            network.generators,
        )

    def count_lines(lines):
        buses_in_lines = lines[["bus0", "bus1"]].drop_duplicates()

        def count(bus):
            total = (
                (buses_in_lines["bus0"] == bus.name)
                | (buses_in_lines["bus1"] == bus.name)
            ).sum()
            return total

        return count

    network = etrago.network

    # Group the parallel transmission lines to reduce the complexity
    group_parallel_lines(etrago.network)

    # ordering of buses
    bus0_new = network.lines.apply(lambda x: max(x.bus0, x.bus1), axis=1)
    bus1_new = network.lines.apply(lambda x: min(x.bus0, x.bus1), axis=1)
    network.lines["bus0"] = bus0_new
    network.lines["bus1"] = bus1_new

    # Find the buses without any other kind of elements attached to them
    # more than transmission lines.
    ac_buses = network.buses[network.buses.carrier == "AC"][
        ["geom", "country"]
    ]
    b_links = pd.concat([network.links.bus0, network.links.bus1]).unique()
    b_trafo = pd.concat(
        [network.transformers.bus0, network.transformers.bus1]
    ).unique()
    b_gen = network.generators[
        network.generators.carrier != "load shedding"
    ].bus.unique()
    b_load = network.loads.bus.unique()
    b_store = network.stores[network.stores.e_nom > 0].bus.unique()
    b_store_unit = network.storage_units[
        network.storage_units.p_nom > 0
    ].bus.unique()

    ac_buses["links"] = ac_buses.index.isin(b_links)
    ac_buses["trafo"] = ac_buses.index.isin(b_trafo)
    ac_buses["gen"] = ac_buses.index.isin(b_gen)
    ac_buses["load"] = ac_buses.index.isin(b_load)
    ac_buses["store"] = ac_buses.index.isin(b_store)
    ac_buses["storage_unit"] = ac_buses.index.isin(b_store_unit)

    ac_buses = ac_buses[
        ~(ac_buses.links)
        & ~(ac_buses.trafo)
        & ~(ac_buses.gen)
        & ~(ac_buses.load)
        & ~(ac_buses.store)
        & ~(ac_buses.storage_unit)
    ][[]]

    # count how many lines are connected to each bus
    number_of_lines = count_lines(network.lines)
    ac_buses["n_lines"] = 0
    ac_buses["n_lines"] = ac_buses.apply(number_of_lines, axis=1)

    # Keep the buses with two or less transmission lines
    ac_buses = ac_buses[ac_buses["n_lines"] <= 2]

    # Keep only the buses connecting 2 lines with the same capacity
    lines_cap = network.lines[
        (network.lines.bus0.isin(ac_buses.index))
        | (network.lines.bus1.isin(ac_buses.index))
    ][["bus0", "bus1", "s_nom"]]

    delete_bus = []
    for bus in ac_buses[ac_buses["n_lines"] == 2].index:
        l0 = lines_cap[(lines_cap.bus0 == bus) | (lines_cap.bus1 == bus)][
            "s_nom"
        ].unique()
        if len(l0) != 1:
            delete_bus.append(bus)
    ac_buses.drop(delete_bus, inplace=True)

    # create groups of lines to join
    buses_2 = ac_buses[ac_buses["n_lines"] == 2]
    lines = network.lines[
        (network.lines.bus0.isin(buses_2.index))
        | (network.lines.bus1.isin(buses_2.index))
    ][["bus0", "bus1"]].copy()
    lines_index = lines.index
    new_lines = pd.DataFrame(columns=["bus0", "bus1", "lines"])
    group = 0

    for line in lines_index:
        if line not in lines.index:
            continue
        bus0 = lines.at[line, "bus0"]
        bus1 = lines.at[line, "bus1"]
        lines_group = [line]
        lines.drop(line, inplace=True)

        # Determine bus0 new group
        end_search = False

        while not end_search:
            if bus0 not in ac_buses.index:
                end_search = True
                continue
            lines_b = lines[(lines.bus0 == bus0) | (lines.bus1 == bus0)]
            if len(lines_b) > 0:
                lines_group.append(lines_b.index[0])
                if lines_b.iat[0, 0] == bus0:
                    bus0 = lines_b.iat[0, 1]
                else:
                    bus0 = lines_b.iat[0, 0]
                lines.drop(lines_b.index[0], inplace=True)
            else:
                end_search = True

        # Determine bus1 new group
        end_search = False
        while not end_search:
            if bus1 not in ac_buses.index:
                end_search = True
                continue
            lines_b = lines[(lines.bus0 == bus1) | (lines.bus1 == bus1)]
            if len(lines_b) > 0:
                lines_group.append(lines_b.index[0])
                if lines_b.iat[0, 0] == bus1:
                    bus1 = lines_b.iat[0, 1]
                else:
                    bus1 = lines_b.iat[0, 0]
                lines.drop(lines_b.index[0], inplace=True)
            else:
                end_search = True

        # Define the parameters of the new lines to be inserted into
        # `network.lines`.
        new_lines.loc[group] = [bus0, bus1, lines_group]
        group = group + 1

    # Create the new lines as result of aggregating series lines
    lines = network.lines[
        (network.lines.bus0.isin(buses_2.index))
        | (network.lines.bus1.isin(buses_2.index))
    ]

    new_lines_df = pd.DataFrame(columns=lines.columns).rename_axis("Lines")

    for l0 in new_lines.index:
        lines_group = (
            lines[lines.index.isin(new_lines.at[l0, "lines"])]
            .copy()
            .reset_index()
        )
        l_new = agg_series_lines(lines_group, network)
        l_new["bus0"] = new_lines.at[l0, "bus0"]
        l_new["bus1"] = new_lines.at[l0, "bus1"]
        new_lines_df["s_nom_extendable"] = new_lines_df[
            "s_nom_extendable"
        ].astype(bool)
        new_lines_df.loc[l_new.name] = l_new

    # Delete all the dispensable buses
    (
        network.buses,
        network.lines,
        network.storage_units,
        network.generators,
    ) = delete_buses(ac_buses, network)

    # exclude from the new lines the ones connected to deleted buses
    new_lines_df = new_lines_df[
        (~new_lines_df.bus0.isin(ac_buses.index))
        & (~new_lines_df.bus1.isin(ac_buses.index))
    ]

    etrago.network.lines = pd.concat([etrago.network.lines, new_lines_df])

    # Drop s_max_pu timeseries for deleted lines
    etrago.network.lines_t.s_max_pu = (
        etrago.network.lines_t.s_max_pu.transpose()[
            etrago.network.lines_t.s_max_pu.columns.isin(
                etrago.network.lines.index
            )
        ].transpose()
    )

    return


def delete_irrelevant_oneports(etrago):
    network = etrago.network

    network.generators.drop(
        network.generators[
            (network.generators.p_nom == 0)
            & (network.generators.p_nom_extendable is False)
        ].index,
        inplace=True,
    )
    network.storage_units.drop(
        network.storage_units[
            (network.storage_units.p_nom == 0)
            & (network.storage_units.p_nom_extendable is False)
        ].index,
        inplace=True,
    )

    components = ["generators", "storage_units"]
    for g in components:  # loads_t
        h = g + "_t"
        nw = getattr(network, h)  # network.loads_t
        for i in nw.keys():  # network.loads_t.p
            cols = [
                j
                for j in getattr(nw, i).columns
                if j not in getattr(network, g).index
            ]
            for k in cols:
                del getattr(nw, i)[k]

    return


def set_line_costs(self, cost110=230, cost220=290, cost380=85, costDC=375):
    """Set capital costs for extendable lines in respect to PyPSA [€/MVA]

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    args: dict
        containing settings from appl.py
    cost110 :
        capital costs per km for 110kV lines and cables
        default: 230€/MVA/km, source: costs for extra circuit in
        dena Verteilnetzstudie, p. 146)
    cost220 :
        capital costs per km for 220kV lines and cables
        default: 280€/MVA/km, source: costs for extra circuit in
        NEP 2025, capactity from most used 220 kV lines in model
    cost380 :
        capital costs per km for 380kV lines and cables
        default: 85€/MVA/km, source: costs for extra circuit in
        NEP 2025, capactity from most used 380 kV lines in NEP
    costDC :
        capital costs per km for DC-lines
        default: 375€/MVA/km, source: costs for DC transmission line
        in NEP 2035

    """

    network = self.network

    network.lines.loc[(network.lines.v_nom == 110), "capital_cost"] = (
        cost110 * network.lines.length
    )

    network.lines.loc[(network.lines.v_nom == 220), "capital_cost"] = (
        cost220 * network.lines.length
    )

    network.lines.loc[(network.lines.v_nom == 380), "capital_cost"] = (
        cost380 * network.lines.length
    )

    network.links.loc[
        (network.links.p_nom_extendable)
        & (network.links.index.isin(self.dc_lines().index)),
        "capital_cost",
    ] = (
        costDC * network.links.length
    )

    return network


def set_trafo_costs(
    self, cost110_220=7500, cost110_380=17333, cost220_380=14166
):
    """Set capital costs for extendable transformers in respect
    to PyPSA [€/MVA]

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    cost110_220 :
        capital costs for 110/220kV transformer
        default: 7500€/MVA, source: costs for extra trafo in
        dena Verteilnetzstudie, p. 146; S of trafo used in osmTGmod
    cost110_380 :
        capital costs for 110/380kV transformer
        default: 17333€/MVA, source: NEP 2025
    cost220_380 :
        capital costs for 220/380kV transformer
        default: 14166€/MVA, source: NEP 2025

    """

    network = self.network
    network.transformers["v_nom0"] = network.transformers.bus0.map(
        network.buses.v_nom
    )
    network.transformers["v_nom1"] = network.transformers.bus1.map(
        network.buses.v_nom
    )

    network.transformers.loc[
        (network.transformers.v_nom0 == 110)
        & (network.transformers.v_nom1 == 220),
        "capital_cost",
    ] = cost110_220

    network.transformers.loc[
        (network.transformers.v_nom0 == 110)
        & (network.transformers.v_nom1 == 380),
        "capital_cost",
    ] = cost110_380

    network.transformers.loc[
        (network.transformers.v_nom0 == 220)
        & (network.transformers.v_nom1 == 380),
        "capital_cost",
    ] = cost220_380

    return network


def add_missing_components(self):
    """
    Add a missing transformer at Heizkraftwerk Nord in Munich and a missing
    transformer in Stuttgart.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    """

    # Munich
    # TODO: Manually adds lines between hard-coded buses. Has to be
    #       changed for the next dataversion and should be moved to data
    #       processing

    """
    "https://www.swm.de/privatkunden/unternehmen/energieerzeugung"
    + "/heizkraftwerke.html?utm_medium=301"

     to bus 25096:
     25369 (86)
     28232 (24)
     25353 to 25356 (79)
     to bus 23822: (110kV bus  of 380/110-kV-transformer)
     25355 (90)
     28212 (98)

     25357 to 665 (85)
     25354 to 27414 (30)
     27414 to 28212 (33)
     25354 to 28294 (32/63)
     28335 to 28294 (64)
     28335 to 28139 (28)
     Overhead lines:
     16573 to 24182 (part of 4)

     Installierte Leistung der Umspannungsebene Höchst- zu Hochspannung
     (380 kV / 110 kV): 2.750.000 kVA

     "https://www.swm-infrastruktur.de/strom/netzstrukturdaten"
     + "/strukturmerkmale.html
    """
    network = self.network

    new_trafo = str(network.transformers.index.astype(int).max() + 1)

    network.add(
        "Transformer",
        new_trafo,
        bus0="16573",
        bus1="23648",
        x=0.135 / (2750 / 2),
        r=0.0,
        tap_ratio=1,
        s_nom=2750 / 2,
    )

    def add_110kv_line(bus0, bus1, overhead=False):
        new_line = str(network.lines.index.astype(int).max() + 1)
        if not overhead:
            network.add("Line", new_line, bus0=bus0, bus1=bus1, s_nom=280)
        else:
            network.add("Line", new_line, bus0=bus0, bus1=bus1, s_nom=260)
        network.lines.loc[new_line, "scn_name"] = "Status Quo"
        network.lines.loc[new_line, "v_nom"] = 110
        network.lines.loc[new_line, "version"] = "added_manually"
        network.lines.loc[new_line, "frequency"] = 50
        network.lines.loc[new_line, "cables"] = 3.0
        network.lines.loc[new_line, "country"] = "DE"
        network.lines.loc[new_line, "length"] = (
            pypsa.geo.haversine(
                network.buses.loc[bus0, ["x", "y"]],
                network.buses.loc[bus1, ["x", "y"]],
            )[0][0]
            * 1.2
        )
        if not overhead:
            network.lines.loc[new_line, "r"] = (
                network.lines.loc[new_line, "length"] * 0.0177
            )
            network.lines.loc[new_line, "g"] = 0
            # or: (network.lines.loc[new_line, "length"]*78e-9)
            network.lines.loc[new_line, "x"] = (
                network.lines.loc[new_line, "length"] * 0.3e-3
            )
            network.lines.loc[new_line, "b"] = (
                network.lines.loc[new_line, "length"] * 250e-9
            )

        elif overhead:
            network.lines.loc[new_line, "r"] = (
                network.lines.loc[new_line, "length"] * 0.05475
            )
            network.lines.loc[new_line, "g"] = 0
            # or: (network.lines.loc[new_line, "length"]*40e-9)
            network.lines.loc[new_line, "x"] = (
                network.lines.loc[new_line, "length"] * 1.2e-3
            )
            network.lines.loc[new_line, "b"] = (
                network.lines.loc[new_line, "length"] * 9.5e-9
            )

    add_110kv_line("16573", "28353")
    add_110kv_line("16573", "28092")
    add_110kv_line("25096", "25369")
    add_110kv_line("25096", "28232")
    add_110kv_line("25353", "25356")
    add_110kv_line("23822", "25355")
    add_110kv_line("23822", "28212")
    add_110kv_line("25357", "665")
    add_110kv_line("25354", "27414")
    add_110kv_line("27414", "28212")
    add_110kv_line("25354", "28294")
    add_110kv_line("28335", "28294")
    add_110kv_line("28335", "28139")
    add_110kv_line("16573", "24182", overhead=True)

    # Stuttgart
    """
         Stuttgart:
         Missing transformer, because 110-kV-bus is situated outside
         Heizkraftwerk Heilbronn:
    """
    # new_trafo = str(network.transformers.index.astype(int).max()1)
    network.add(
        "Transformer",
        "99999",
        bus0="18967",
        bus1="25766",
        x=0.135 / 300,
        r=0.0,
        tap_ratio=1,
        s_nom=300,
    )
    """
    According to:
    https://assets.ctfassets.net/xytfb1vrn7of/NZO8x4rKesAcYGGcG4SQg/b780d6a3ca4c2600ab51a30b70950bb1/netzschemaplan-110-kv.pdf
    the following lines are missing:
    """
    add_110kv_line("18967", "22449", overhead=True)  # visible in OSM & DSO map
    add_110kv_line("21165", "24068", overhead=True)  # visible in OSM & DSO map
    add_110kv_line("23782", "24089", overhead=True)
    # visible in DSO map & OSM till 1 km from bus1
    """
    Umspannwerk Möhringen (bus 23697)
    https://de.wikipedia.org/wiki/Umspannwerk_M%C3%B6hringen
    there should be two connections:
    to Sindelfingen (2*110kV)
    to Wendingen (former 220kV, now 2*110kV)
    the line to Sindelfingen is connected, but the connection of Sindelfingen
    itself to 380kV is missing:
    """
    add_110kv_line("19962", "27671", overhead=True)  # visible in OSM & DSO map
    add_110kv_line("19962", "27671", overhead=True)
    """
    line to Wendingen is missing, probably because it ends shortly before the
    way of the substation and is connected via cables:
    """
    add_110kv_line("23697", "24090", overhead=True)  # visible in OSM & DSO map
    add_110kv_line("23697", "24090", overhead=True)

    # Lehrte
    """
    Lehrte: 220kV Bus located outsinde way of Betriebszentrtum Lehrte and
    therefore not connected:
    """

    def add_220kv_line(bus0, bus1, overhead=False):
        new_line = str(network.lines.index.astype(int).max() + 1)
        if not overhead:
            network.add("Line", new_line, bus0=bus0, bus1=bus1, s_nom=550)
        else:
            network.add("Line", new_line, bus0=bus0, bus1=bus1, s_nom=520)
        network.lines.loc[new_line, "scn_name"] = "Status Quo"
        network.lines.loc[new_line, "v_nom"] = 220
        network.lines.loc[new_line, "version"] = "added_manually"
        network.lines.loc[new_line, "frequency"] = 50
        network.lines.loc[new_line, "cables"] = 3.0
        network.lines.loc[new_line, "country"] = "DE"
        network.lines.loc[new_line, "length"] = (
            pypsa.geo.haversine(
                network.buses.loc[bus0, ["x", "y"]],
                network.buses.loc[bus1, ["x", "y"]],
            )[0][0]
            * 1.2
        )
        if not overhead:
            network.lines.loc[new_line, "r"] = (
                network.lines.loc[new_line, "length"] * 0.0176
            )
            network.lines.loc[new_line, "g"] = 0
            # or: (network.lines.loc[new_line, "length"]*67e-9)
            network.lines.loc[new_line, "x"] = (
                network.lines.loc[new_line, "length"] * 0.3e-3
            )
            network.lines.loc[new_line, "b"] = (
                network.lines.loc[new_line, "length"] * 210e-9
            )

        elif overhead:
            network.lines.loc[new_line, "r"] = (
                network.lines.loc[new_line, "length"] * 0.05475
            )
            network.lines.loc[new_line, "g"] = 0
            # or: (network.lines.loc[new_line, "length"]*30e-9)
            network.lines.loc[new_line, "x"] = (
                network.lines.loc[new_line, "length"] * 1e-3
            )
            network.lines.loc[new_line, "b"] = (
                network.lines.loc[new_line, "length"] * 11e-9
            )

    add_220kv_line("266", "24633", overhead=True)

    # temporary turn buses of transformers
    network.transformers["v_nom0"] = network.transformers.bus0.map(
        network.buses.v_nom
    )
    network.transformers["v_nom1"] = network.transformers.bus1.map(
        network.buses.v_nom
    )
    new_bus0 = network.transformers.bus1[
        network.transformers.v_nom0 > network.transformers.v_nom1
    ]
    new_bus1 = network.transformers.bus0[
        network.transformers.v_nom0 > network.transformers.v_nom1
    ]
    network.transformers.bus0[
        network.transformers.v_nom0 > network.transformers.v_nom1
    ] = new_bus0.values
    network.transformers.bus1[
        network.transformers.v_nom0 > network.transformers.v_nom1
    ] = new_bus1.values

    return network


def convert_capital_costs(self):
    """Convert capital_costs to fit to considered timesteps

    Parameters
    ----------
    etrago : :class:`etrago.Etrago
        Transmission grid object

    """

    network = self.network
    n_snapshots = self.args["end_snapshot"] - self.args["start_snapshot"] + 1

    # Costs are already annuized yearly in the datamodel
    # adjust to number of considered snapshots

    network.lines.loc[network.lines.s_nom_extendable, "capital_cost"] *= (
        n_snapshots / 8760
    )

    network.links.loc[network.links.p_nom_extendable, "capital_cost"] *= (
        n_snapshots / 8760
    )

    network.transformers.loc[
        network.transformers.s_nom_extendable, "capital_cost"
    ] *= (n_snapshots / 8760)

    network.storage_units.loc[
        network.storage_units.p_nom_extendable, "capital_cost"
    ] *= (n_snapshots / 8760)

    network.stores.loc[network.stores.e_nom_extendable, "capital_cost"] *= (
        n_snapshots / 8760
    )


def find_snapshots(network, carrier, maximum=True, minimum=True, n=3):
    """
    Function that returns snapshots with maximum and/or minimum feed-in of
    selected carrier.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    carrier: str
        Selected carrier of generators
    maximum: bool
        Choose if timestep of maximal feed-in is returned.
    minimum: bool
        Choose if timestep of minimal feed-in is returned.
    n: int
        Number of maximal/minimal snapshots

    Returns
    -------
    calc_snapshots : 'pandas.core.indexes.datetimes.DatetimeIndex'
        List containing snapshots
    """

    if carrier == "residual load":
        power_plants = network.generators[
            network.generators.carrier.isin(["solar", "wind", "wind_onshore"])
        ]
        power_plants_t = (
            network.generators.p_nom[power_plants.index]
            * network.generators_t.p_max_pu[power_plants.index]
        )
        load = network.loads_t.p_set.sum(axis=1)
        all_renew = power_plants_t.sum(axis=1)
        all_carrier = load - all_renew

    if carrier in (
        "solar",
        "wind",
        "wind_onshore",
        "wind_offshore",
        "run_of_river",
    ):
        power_plants = network.generators[
            network.generators.carrier == carrier
        ]

        power_plants_t = (
            network.generators.p_nom[power_plants.index]
            * network.generators_t.p_max_pu[power_plants.index]
        )
        all_carrier = power_plants_t.sum(axis=1)

    if maximum and not minimum:
        times = all_carrier.sort_values().head(n=n)

    if minimum and not maximum:
        times = all_carrier.sort_values().tail(n=n)

    if maximum and minimum:
        times = all_carrier.sort_values().head(n=n)
        times = pd.concat([times, all_carrier.sort_values().tail(n=n)])

    calc_snapshots = all_carrier.index[all_carrier.index.isin(times.index)]

    return calc_snapshots


def ramp_limits(network):
    """Add ramping constraints to thermal power plants.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------

    """
    carrier = [
        "coal",
        "biomass",
        "gas",
        "oil",
        "waste",
        "lignite",
        "uranium",
        "geothermal",
    ]
    data = {
        "start_up_cost": [77, 57, 42, 57, 57, 77, 50, 57],  # €/MW
        "start_up_fuel": [4.3, 2.8, 1.45, 2.8, 2.8, 4.3, 16.7, 2.8],  # MWh/MW
        "min_up_time": [5, 2, 3, 2, 2, 5, 12, 2],
        "min_down_time": [7, 2, 2, 2, 2, 7, 17, 2],
        # ===================================================================
        #   'ramp_limit_start_up':[0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.5, 0.4],
        #   'ramp_limit_shut_down':[0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.5, 0.4]
        # ===================================================================
        "p_min_pu": [0.33, 0.38, 0.4, 0.38, 0.38, 0.5, 0.45, 0.38],
    }
    df = pd.DataFrame(data, index=carrier)
    fuel_costs = network.generators.marginal_cost.groupby(
        network.generators.carrier
    ).mean()[carrier]
    df["start_up_fuel"] = df["start_up_fuel"] * fuel_costs
    df["start_up_cost"] = df["start_up_cost"] + df["start_up_fuel"]
    df.drop("start_up_fuel", axis=1, inplace=True)
    for tech in df.index:
        for limit in df.columns:
            network.generators.loc[
                network.generators.carrier == tech, limit
            ] = df.loc[tech, limit]
    network.generators.start_up_cost = (
        network.generators.start_up_cost * network.generators.p_nom
    )
    network.generators.committable = True


def get_args_setting(self, jsonpath="scenario_setting.json"):
    """
    Get and open json file with scenaio settings of eTraGo ``args``.
    The settings incluedes all eTraGo specific settings of arguments and
    parameters for a reproducible calculation.

    Parameters
    ----------
    json_file : str
        Default: ``scenario_setting.json``
        Name of scenario setting json file

    Returns
    -------
    args : dict
        Dictionary of json file
    """

    if jsonpath is not None:
        with open(jsonpath) as f:
            if "args" in locals():
                self.args = merge_dicts(self.args, json.load(f))
            else:
                self.args = json.load(f)


def merge_dicts(dict1, dict2):
    """
    Return a new dictionary by merging two dictionaries recursively.

    Parameters
    ----------
    dict1 : dict
        dictionary 1.
    dict2 : dict
        dictionary 2.

    Returns
    -------
    result : dict
        Union of dict1 and dict2

    """

    result = deepcopy(dict1)

    for key, value in dict2.items():
        if isinstance(value, Mapping):
            result[key] = merge_dicts(result.get(key, {}), value)
        else:
            result[key] = deepcopy(dict2[key])

    return result


def get_clustering_data(self, path):
    """
    Import the final busmap and the initial buses, lines and links

    Parameters
    ----------
    path : str
        Name of folder from which to import CSVs of network data.

    Returns
    -------
    None

    """

    if (self.args["network_clustering_ehv"]["active"]) | (
        self.args["network_clustering"]["active"]
    ):
        path_clus = os.path.join(path, "clustering")
        if os.path.exists(path_clus):
            ch4_h2_mapping_path = os.path.join(
                path_clus, "ch4_h2_mapping.json"
            )
            if os.path.exists(ch4_h2_mapping_path):
                with open(ch4_h2_mapping_path) as f:
                    self.ch4_h2_mapping = pd.read_json(f, typ="series").astype(
                        str
                    )
                    self.ch4_h2_mapping.index.name = "CH4_bus"
                    self.ch4_h2_mapping.index = (
                        self.ch4_h2_mapping.index.astype(str)
                    )
            else:
                logger.info(
                    """There is no CH4 to H2 bus mapping data
                    available in the loaded object."""
                )

            busmap_path = os.path.join(path_clus, "busmap.json")
            if os.path.exists(busmap_path):
                with open(busmap_path) as f:
                    self.busmap["busmap"] = json.load(f)
                self.busmap["orig_network"] = pypsa.Network(
                    path_clus, name="orig"
                )
            else:
                logger.info(
                    "There is no busmap data available in the loaded object."
                )

        else:
            logger.info(
                "There is no clustering data available in the loaded object."
            )


def set_random_noise(self, sigma=0.01):
    """
    Sets random noise to marginal cost of each generator.

    Parameters
    ----------
    etrago : :class:`etrago.Etrago
        Transmission grid object
    seed: int
        seed number, needed to reproduce results
    sigma: float
        Default: 0.01
        standard deviation, small values reduce impact on dispatch
        but might lead to numerical instability
    """

    if self.args["generator_noise"]:
        network = self.network
        seed = self.args["generator_noise"]
        s = np.random.RandomState(seed)
        network.generators.marginal_cost[
            network.generators.bus.isin(
                network.buses.index[network.buses.country == "DE"]
            )
        ] += abs(
            s.normal(
                0,
                sigma,
                len(
                    network.generators.marginal_cost[
                        network.generators.bus.isin(
                            network.buses.index[network.buses.country == "DE"]
                        )
                    ]
                ),
            )
        )

        network.generators.marginal_cost[
            network.generators.bus.isin(
                network.buses.index[network.buses.country != "DE"]
            )
        ] += abs(
            s.normal(
                0,
                sigma,
                len(
                    network.generators.marginal_cost[
                        network.generators.bus.isin(
                            network.buses.index[network.buses.country == "DE"]
                        )
                    ]
                ),
            )
        ).max()


def set_line_country_tags(network):
    """
    Set country tag for AC- and DC-lines.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    """

    transborder_lines_0 = network.lines[
        network.lines["bus0"].isin(
            network.buses.index[network.buses["country"] != "DE"]
        )
    ].index
    transborder_lines_1 = network.lines[
        network.lines["bus1"].isin(
            network.buses.index[network.buses["country"] != "DE"]
        )
    ].index
    # set country tag for lines
    network.lines.loc[transborder_lines_0, "country"] = network.buses.loc[
        network.lines.loc[transborder_lines_0, "bus0"].values, "country"
    ].values

    network.lines.loc[transborder_lines_1, "country"] = network.buses.loc[
        network.lines.loc[transborder_lines_1, "bus1"].values, "country"
    ].values
    network.lines["country"].fillna("DE", inplace=True)
    doubles = list(set(transborder_lines_0.intersection(transborder_lines_1)))
    for line in doubles:
        c_bus0 = network.buses.loc[network.lines.loc[line, "bus0"], "country"]
        c_bus1 = network.buses.loc[network.lines.loc[line, "bus1"], "country"]
        network.lines.loc[line, "country"] = "{}{}".format(c_bus0, c_bus1)

    transborder_links_0 = network.links[
        network.links["bus0"].isin(
            network.buses.index[network.buses["country"] != "DE"]
        )
    ].index
    transborder_links_1 = network.links[
        network.links["bus1"].isin(
            network.buses.index[network.buses["country"] != "DE"]
        )
    ].index

    # set country tag for links
    network.links.loc[transborder_links_0, "country"] = network.buses.loc[
        network.links.loc[transborder_links_0, "bus0"].values, "country"
    ].values

    network.links.loc[transborder_links_1, "country"] = network.buses.loc[
        network.links.loc[transborder_links_1, "bus1"].values, "country"
    ].values
    network.links["country"].fillna("DE", inplace=True)
    doubles = list(set(transborder_links_0.intersection(transborder_links_1)))
    for link in doubles:
        c_bus0 = network.buses.loc[network.links.loc[link, "bus0"], "country"]
        c_bus1 = network.buses.loc[network.links.loc[link, "bus1"], "country"]
        network.links.loc[link, "country"] = "{}{}".format(c_bus0, c_bus1)


def crossborder_capacity_tyndp2020():
    """
    This function downloads and extracts a scenario datafile for the TYNDP 2020
    (Ten-Year Network Development Plan), reads a specific sheet from the file,
    filters it based on certain criteria, and then calculates the minimum
    cross-border capacities for a list of European countries. The minimum
    cross-border capacity is the minimum of the export and import capacities
    between two countries.

    Returns
    -------
    dict
        Dictionary with cossborder capacities.

    """
    from urllib.request import urlretrieve
    import zipfile

    path = "TYNDP-2020-Scenario-Datafile.xlsx"

    urlretrieve(
        "https://www.entsos-tyndp2020-scenarios.eu/wp-content/uploads"
        "/2020/06/TYNDP-2020-Scenario-Datafile.xlsx.zip",
        path,
    )

    file = zipfile.ZipFile(path)

    df = pd.read_excel(
        file.open("TYNDP-2020-Scenario-Datafile.xlsx").read(),
        sheet_name="Line",
    )

    df = df[
        (df.Scenario == "Distributed Energy")
        & (df.Case == "Reference Grid")
        & (df.Year == 2040)
        & (df["Climate Year"] == 1984)
        & (
            (df.Parameter == "Import Capacity")
            | (df.Parameter == "Export Capacity")
        )
    ]

    df["country0"] = df["Node/Line"].str[:2]

    df["country1"] = df["Node/Line"].str[5:7]

    c_export = (
        df[df.Parameter == "Export Capacity"]
        .groupby(["country0", "country1"])
        .Value.sum()
    )

    c_import = (
        df[df.Parameter == "Import Capacity"]
        .groupby(["country0", "country1"])
        .Value.sum()
    )

    capacities = pd.DataFrame(
        index=c_export.index,
        data={"export": c_export.abs(), "import": c_import.abs()},
    ).reset_index()

    with_de = capacities[
        (capacities.country0 == "DE") & (capacities.country1 != "DE")
    ].set_index("country1")[["export", "import"]]

    with_de = pd.concat(
        [
            with_de,
            capacities[
                (capacities.country0 != "DE") & (capacities.country1 == "DE")
            ].set_index("country0")[["export", "import"]],
        ]
    )

    countries = [
        "DE",
        "DK",
        "NL",
        "CZ",
        "PL",
        "AT",
        "CH",
        "FR",
        "LU",
        "BE",
        "GB",
        "NO",
        "SE",
    ]

    without_de = capacities[
        (capacities.country0 != "DE")
        & (capacities.country1 != "DE")
        & (capacities.country0.isin(countries))
        & (capacities.country1.isin(countries))
        & (capacities.country1 != capacities.country0)
    ]

    without_de["country"] = without_de.country0 + without_de.country1

    without_de.set_index("country", inplace=True)

    without_de = without_de[["export", "import"]].fillna(0.0)

    return {
        **without_de.min(axis=1).to_dict(),
        **with_de.min(axis=1).to_dict(),
    }


def crossborder_capacity(self):
    """
    Adjust interconnector capacties.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    method : string
        Method of correction. Options are 'ntc_acer' and 'thermal_acer'.
        'ntc_acer' corrects all capacities according to values published by
        the ACER in 2016.
        'thermal_acer' corrects certain capacities where our dataset most
        likely overestimates the thermal capacity.

    """
    if self.args["foreign_lines"]["capacity"] != "osmTGmod":
        network = self.network

        if self.args["foreign_lines"]["capacity"] == "ntc_acer":
            cap_per_country = {
                "AT": 4900,
                "CH": 2695,
                "CZ": 1301,
                "DK": 913,
                "FR": 3593,
                "LU": 2912,
                "NL": 2811,
                "PL": 280,
                "SE": 217,
                "CZAT": 574,
                "ATCZ": 574,
                "CZPL": 312,
                "PLCZ": 312,
                "ATCH": 979,
                "CHAT": 979,
                "CHFR": 2087,
                "FRCH": 2087,
                "FRLU": 364,
                "LUFR": 364,
                "SEDK": 1928,
                "DKSE": 1928,
            }

        elif self.args["foreign_lines"]["capacity"] == "thermal_acer":
            cap_per_country = {
                "CH": 12000,
                "DK": 4000,
                "SEDK": 3500,
                "DKSE": 3500,
            }

        elif self.args["foreign_lines"]["capacity"] == "tyndp2020":
            cap_per_country = crossborder_capacity_tyndp2020()

        else:
            logger.info(
                "args['foreign_lines']['capacity'] has to be "
                "in ['osmTGmod', 'ntc_acer', 'thermal_acer', 'tyndp2020']"
            )

        if not network.lines[network.lines.country != "DE"].empty:
            weighting = (
                network.lines.loc[network.lines.country != "DE", "s_nom"]
                .groupby(network.lines.country)
                .transform(lambda x: x / x.sum())
            )

        dc_lines = self.dc_lines()

        weighting_links = (
            dc_lines.loc[dc_lines.country != "DE", "p_nom"]
            .groupby(dc_lines.country)
            .transform(lambda x: x / x.sum())
            .fillna(0.0)
        )

        for country in cap_per_country:
            index_HV = network.lines[
                (network.lines.country == country)
                & (network.lines.v_nom == 110)
            ].index
            index_eHV = network.lines[
                (network.lines.country == country)
                & (network.lines.v_nom > 110)
            ].index
            index_links = dc_lines[dc_lines.country == country].index

            if not network.lines[network.lines.country == country].empty:
                network.lines.loc[index_HV, "s_nom"] = (
                    weighting[index_HV] * cap_per_country[country]
                )

                network.lines.loc[index_eHV, "s_nom"] = (
                    weighting[index_eHV] * cap_per_country[country]
                )

            if not dc_lines[dc_lines.country == country].empty:
                network.links.loc[index_links, "p_nom"] = (
                    weighting_links[index_links] * cap_per_country[country]
                )
            if country == "SE":
                network.links.loc[
                    dc_lines[dc_lines.country == country].index, "p_nom"
                ] = cap_per_country[country]

            if not network.lines[
                network.lines.country == (country + country)
            ].empty:
                i_HV = network.lines[
                    (network.lines.v_nom == 110)
                    & (network.lines.country == country + country)
                ].index

                i_eHV = network.lines[
                    (network.lines.v_nom == 110)
                    & (network.lines.country == country + country)
                ].index

                network.lines.loc[i_HV, "s_nom"] = (
                    weighting[i_HV] * cap_per_country[country]
                )
                network.lines.loc[i_eHV, "s_nom"] = (
                    weighting[i_eHV] * cap_per_country[country]
                )

            if not dc_lines[dc_lines.country == (country + country)].empty:
                i_links = dc_lines[
                    dc_lines.country == (country + country)
                ].index
                network.links.loc[i_links, "p_nom"] = (
                    weighting_links[i_links] * cap_per_country[country]
                )


def set_branch_capacity(etrago):
    """
    Set branch capacity factor of lines and transformers, different factors for
    HV (110kV) and eHV (220kV, 380kV).

    Parameters
    ----------
    etrago : :class:`etrago.Etrago
        Transmission grid object

    """
    network = etrago.network
    args = etrago.args

    network.transformers["v_nom0"] = network.transformers.bus0.map(
        network.buses.v_nom
    )

    # If any line has a time dependend s_max_pu, use the time dependend
    # factor for all lines, to avoid problems in the clustering
    if not network.lines_t.s_max_pu.empty:
        # Set time dependend s_max_pu for
        # lines without dynamic line rating to 1.0
        network.lines_t.s_max_pu[
            network.lines[
                ~network.lines.index.isin(network.lines_t.s_max_pu.columns)
            ].index
        ] = 1.0

        # Multiply time dependend s_max_pu with static branch capacitiy fator
        network.lines_t.s_max_pu[
            network.lines[network.lines.v_nom == 110].index
        ] *= args["branch_capacity_factor"]["HV"]

        network.lines_t.s_max_pu[
            network.lines[network.lines.v_nom > 110].index
        ] *= args["branch_capacity_factor"]["eHV"]
    else:
        network.lines.s_max_pu[network.lines.v_nom == 110] = args[
            "branch_capacity_factor"
        ]["HV"]

        network.lines.s_max_pu[network.lines.v_nom > 110] = args[
            "branch_capacity_factor"
        ]["eHV"]

    network.transformers.s_max_pu[network.transformers.v_nom0 == 110] = args[
        "branch_capacity_factor"
    ]["HV"]

    network.transformers.s_max_pu[network.transformers.v_nom0 > 110] = args[
        "branch_capacity_factor"
    ]["eHV"]


def check_args(etrago):
    """
    Function that checks the consistency of etragos input parameters.

    Parameters
    ----------
    etrago : :class:`etrago.Etrago
        Overall container of eTraGo

    Returns
    -------
    None.

    """

    names = [
        "eGon2035",
        "eGon100RE",
        "eGon2035_lowflex",
        "eGon100RE_lowflex",
        "status2019",
    ]

    assert (
        etrago.args["scn_name"] in names
    ), f"'scn_name' has to be in {names} but is {etrago.args['scn_name']}."

    assert (
        etrago.args["start_snapshot"] <= etrago.args["end_snapshot"]
    ), "start_snapshot after end_snapshot"

    if etrago.args["gridversion"] is not None:
        from saio.grid import egon_etrago_bus

        assert (
            etrago.args["gridversion"]
            in pd.read_sql(
                etrago.session.query(egon_etrago_bus).statement,
                etrago.session.bind,
            ).version.unique()
        ), "gridversion does not exist"

    if etrago.args["snapshot_clustering"]["active"]:
        # Assert that skip_snapshots and snapshot_clustering are not combined
        # more information: https://github.com/openego/eTraGo/issues/691
        assert etrago.args["skip_snapshots"] is False, (
            "eTraGo does not support combining snapshot_clustering and"
            " skip_snapshots. Please update your settings and choose either"
            " snapshot_clustering or skip_snapshots."
        )
        # typical periods
        if etrago.args["snapshot_clustering"]["method"] == "typical_periods":
            # typical days

            if etrago.args["snapshot_clustering"]["how"] == "daily":
                assert (
                    etrago.args["end_snapshot"]
                    / etrago.args["start_snapshot"]
                    % 24
                    == 0
                ), (
                    "Please select snapshots covering whole days when"
                    " choosing clustering to typical days."
                )

                if (
                    etrago.args["snapshot_clustering"]["method"]
                    == "typical_periods"
                ):
                    assert etrago.args["end_snapshot"] - etrago.args[
                        "start_snapshot"
                    ] + 1 >= (
                        24 * etrago.args["snapshot_clustering"]["n_clusters"]
                    ), (
                        "The umber of selected snapshots is is too small"
                        " for the chosen number of typical days."
                    )

            # typical weeks

            if etrago.args["snapshot_clustering"]["how"] == "weekly":
                assert (
                    etrago.args["end_snapshot"]
                    / etrago.args["start_snapshot"]
                    % 168
                    == 0
                ), (
                    "Please select snapshots covering whole weeks when"
                    " choosing clustering to typical weeks."
                )

                if (
                    etrago.args["snapshot_clustering"]["method"]
                    == "typical_periods"
                ):
                    assert etrago.args["end_snapshot"] - etrago.args[
                        "start_snapshot"
                    ] + 1 >= (
                        168 * etrago.args["snapshot_clustering"]["n_clusters"]
                    ), (
                        "The number of selected snapshots is too small"
                        " for the chosen number of typical weeks."
                    )
            # typical months

            if etrago.args["snapshot_clustering"]["how"] == "monthly":
                assert (
                    etrago.args["end_snapshot"]
                    / etrago.args["start_snapshot"]
                    % 720
                    == 0
                ), (
                    "Please select snapshots covering whole months when"
                    " choosing clustering to typical months."
                )

                if (
                    etrago.args["snapshot_clustering"]["method"]
                    == "typical_periods"
                ):
                    assert etrago.args["end_snapshot"] - etrago.args[
                        "start_snapshot"
                    ] + 1 >= (
                        720 * etrago.args["snapshot_clustering"]["n_clusters"]
                    ), (
                        "The number of selected snapshots is too small"
                        " for the chosen number of typical months."
                    )

        # segmentation

        elif etrago.args["snapshot_clustering"]["method"] == "segmentation":
            assert etrago.args["end_snapshot"] - etrago.args[
                "start_snapshot"
            ] + 1 >= (
                etrago.args["snapshot_clustering"]["n_segments"]
            ), "Number of segments is higher than number of snapshots"

        if not etrago.args["method"]["pyomo"]:
            logger.warning(
                "Snapshot clustering constraints are"
                " not yet correctly implemented without pyomo."
                " Setting `args['method']['pyomo']` to `True`."
            )
            etrago.args["method"]["pyomo"] = True

    if etrago.args["method"]["formulation"] != "pyomo":
        try:
            # The import isn't used, but just here to test for Gurobi.
            # So we can make `flake8` stop complaining about the "unused
            # import" via the appropriate `noqa` comment.
            import gurobipy  # noqa: F401
        except ModuleNotFoundError:
            print(
                "If you want to use nomopyomo you need to use the"
                " solver gurobi and the package gurobipy."
                " You can find more information and installation"
                " instructions for gurobi here:"
                " https://support.gurobi.com/hc/en-us/articles"
                "/360044290292-How-do-I-install-Gurobi-for-Python-"
                " For installation of gurobipy use pip."
            )
            raise


def drop_sectors(self, drop_carriers):
    """
    Manually drop secors from network.
    Makes sure the network can be calculated without the dropped sectors.

    Parameters
    ----------
    drop_carriers : array
        List of sectors that will be dropped.
        e.g. ['dsm', 'CH4', 'H2_saltcavern', 'H2_grid',
        'central_heat', 'rural_heat', 'central_heat_store',
        'rural_heat_store', 'Li ion'] means everything but AC

    Returns
    -------
    None.

    """

    if self.scenario.scn_name == "eGon2035":
        if "CH4" in drop_carriers:
            # create gas generators from links
            # in order to not lose them when dropping non-electric carriers
            gas_to_add = ["central_gas_CHP", "industrial_gas_CHP", "OCGT"]
            gen = self.network.generators

            for i in gas_to_add:
                gen_empty = gen.drop(gen.index)
                gen_empty.bus = self.network.links[
                    self.network.links.carrier == i
                ].bus1
                gen_empty.p_nom = (
                    self.network.links[self.network.links.carrier == i].p_nom
                    * self.network.links[
                        self.network.links.carrier == i
                    ].efficiency
                )
                gen_empty.marginal_cost = (
                    self.network.links[
                        self.network.links.carrier == i
                    ].marginal_cost
                    + 35.851
                )  # add fuel costs (source: NEP)
                gen_empty.efficiency = 1
                gen_empty.carrier = i
                gen_empty.scn_name = "eGon2035"
                gen_empty.p_nom_extendable = False
                gen_empty.sign = 1
                gen_empty.p_min_pu = 0
                gen_empty.p_max_pu = 1
                gen_empty.control = "PV"
                gen_empty.fillna(0, inplace=True)
                self.network.import_components_from_dataframe(
                    gen_empty, "Generator"
                )

    self.network.mremove(
        "Bus",
        self.network.buses[
            self.network.buses.carrier.isin(drop_carriers)
        ].index,
    )

    for one_port in self.network.iterate_components(
        ["Load", "Generator", "Store", "StorageUnit"]
    ):
        self.network.mremove(
            one_port.name,
            one_port.df[~one_port.df.bus.isin(self.network.buses.index)].index,
        )

    for two_port in self.network.iterate_components(
        ["Line", "Link", "Transformer"]
    ):
        self.network.mremove(
            two_port.name,
            two_port.df[
                ~two_port.df.bus0.isin(self.network.buses.index)
            ].index,
        )

        self.network.mremove(
            two_port.name,
            two_port.df[
                ~two_port.df.bus1.isin(self.network.buses.index)
            ].index,
        )

    logger.info("The following sectors are dropped: " + str(drop_carriers))


def update_busmap(self, new_busmap):
    """
    Update busmap after any clustering process

    Parameters
    ----------
    new_busmap : dictionary
        busmap used to clusted the network.

    Returns
    -------
    None.
    """
    if "busmap" not in self.busmap.keys():
        self.busmap["busmap"] = new_busmap
        self.busmap["orig_network"] = pypsa.Network()
        pypsa.io.import_components_from_dataframe(
            self.busmap["orig_network"], self.network.buses, "Bus"
        )
        pypsa.io.import_components_from_dataframe(
            self.busmap["orig_network"], self.network.lines, "Line"
        )
        pypsa.io.import_components_from_dataframe(
            self.busmap["orig_network"], self.network.links, "Link"
        )

    else:
        self.busmap["busmap"] = (
            pd.Series(self.busmap["busmap"]).map(new_busmap).to_dict()
        )


def adjust_CH4_gen_carriers(self):
    """Precise the carrier for the generators with CH4 carrier

    For the eGon2035 scenario, the generators with carrier CH4
    represent the prodution od biogas and methan. In the data model,
    these two differents types are differenciated only by the
    marginal cost of the generator. This function introduces a
    carrier distion (CH4_biogas and CH4_NG) in order to avoid the
    clustering of these two types of generator together and facilitate
    the contraint applying differently to each of them.
    """

    if "eGon2035" in self.args["scn_name"]:
        # Define marginal cost
        marginal_cost_def = {"CH4": 40.9765, "biogas": 25.6}

        engine = db.connection(section=self.args["db"])
        try:
            sql = f"""
            SELECT gas_parameters
            FROM scenario.egon_scenario_parameters
            WHERE name = '{self.args["scn_name"].split("_")[0]}';"""
            df = pd.read_sql(sql, engine)
            marginal_cost = df["gas_parameters"][0]["marginal_cost"]
        except sqlalchemy.exc.ProgrammingError:
            marginal_cost = marginal_cost_def

        self.network.generators.loc[
            self.network.generators[
                (self.network.generators.carrier == "CH4")
                & (
                    self.network.generators.marginal_cost
                    == marginal_cost["CH4"]
                )
                & (
                    self.network.generators.bus.astype(str).isin(
                        self.network.buses.index[
                            self.network.buses.country == "DE"
                        ]
                    )
                )
            ].index,
            "carrier",
        ] = "CH4_NG"

        self.network.generators.loc[
            self.network.generators[
                (self.network.generators.carrier == "CH4")
                & (
                    self.network.generators.marginal_cost
                    == marginal_cost["biogas"]
                )
                & (
                    self.network.generators.bus.astype(str).isin(
                        self.network.buses.index[
                            self.network.buses.country == "DE"
                        ]
                    )
                )
            ].index,
            "carrier",
        ] = "CH4_biogas"


def residual_load(network, sector="electricity"):
    """
    Calculate the residual load for the specified sector.

    In case of the electricity sector residual load is calculated using
    all AC loads and all renewable generators with carriers
    'wind_onshore', 'wind_offshore', 'solar', 'solar_rooftop',
    'biomass', 'run_of_river', and 'reservoir'.

    In case of the central heat sector residual load is calculated using
    all central heat loads and all renewable generators with carriers
    'solar_thermal_collector' and 'geo_thermal'.

    Parameters
    -----------
    network : PyPSA network
        Network to retrieve load and generation time series from, needed
        to determine residual load.
    sector : str
        Sector to determine residual load for. Possible options are
        'electricity' and 'central_heat'. Default: 'electricity'.

    Returns
    --------
    pd.DataFrame
        Dataframe with residual load for each bus in the network.
        Columns of the dataframe contain the corresponding bus name and
        index of the dataframe is a datetime index with the
        corresponding time step.

    """

    if sector == "electricity":
        carrier_gen = [
            "wind_onshore",
            "wind_offshore",
            "solar",
            "solar_rooftop",
            "biomass",
            "run_of_river",
            "reservoir",
        ]
        carrier_load = ["AC"]
    elif sector == "central_heat":
        carrier_gen = ["solar_thermal_collector", "geo_thermal"]
        carrier_load = ["central_heat"]
    else:
        raise ValueError(
            f"Specified sector {sector} is not a valid option."
            " Valid options are 'electricity' and 'central_heat'."
        )
    # Calculate loads per bus and timestep
    loads = network.loads[network.loads.carrier.isin(carrier_load)]
    loads_per_bus = (
        network.loads_t.p_set[loads.index].groupby(loads.bus, axis=1).sum()
    )

    # Calculate dispatch of renewable generators per bus of loads and timesteps
    renewable_dispatch = pd.DataFrame(
        index=loads_per_bus.index, columns=loads_per_bus.columns, data=0
    )

    renewable_generators = network.generators[
        network.generators.carrier.isin(carrier_gen)
    ]

    renewable_dispatch[renewable_generators.bus.unique()] = (
        network.generators_t.p[renewable_generators.index]
        .groupby(renewable_generators.bus, axis=1)
        .sum()
    )

    return loads_per_bus - renewable_dispatch


def manual_fixes_datamodel(etrago):
    """Apply temporal fixes to the data model until a new egon-data run
    is there

    Parameters
    ----------
    etrago : :class:`Etrago
        Overall container of Etrago

    Returns
    -------
    None.

    """
    # Set line type
    etrago.network.lines.type = ""

    # Set life time of storage_units, transformers and lines
    etrago.network.storage_units.lifetime = 27.5
    etrago.network.transformers.lifetime = 40
    etrago.network.lines.lifetime = 40

    # Set efficiences of CHP
    etrago.network.links.loc[
        etrago.network.links[
            etrago.network.links.carrier.str.contains("CHP")
        ].index,
        "efficiency",
    ] = 0.43

    # Enlarge gas boilers as backup heat supply
    etrago.network.links.loc[
        etrago.network.links[
            etrago.network.links.carrier.str.contains("gas_boiler")
        ].index,
        "p_nom",
    ] *= 1000

    # Set p_max_pu for run of river and reservoir
    etrago.network.generators.loc[
        etrago.network.generators[
            etrago.network.generators.carrier.isin(
                ["run_of_river", "reservoir"]
            )
        ].index,
        "p_max_pu",
    ] = 0.65

    # Set costs for CO2 from DAC for needed for methanation
    etrago.network.links.loc[
        etrago.network.links.carrier == "H2_to_CH4", "marginal_cost"
    ] = 25

    # Set r value if missing
    if not etrago.network.lines.loc[etrago.network.lines.r == 0, "r"].empty:
        logger.info(
            f"""
            There are {len(
                etrago.network.lines.loc[etrago.network.lines.r == 0, "r"]
                )} lines without a resistance (r) in the data model.
            The resistance of these lines will be automatically set to 0.0001.
            """
        )

    etrago.network.lines.loc[etrago.network.lines.r == 0, "r"] = 0.0001

    if not etrago.network.transformers.loc[
        etrago.network.transformers.r == 0, "r"
    ].empty:
        logger.info(
            f"""There are {len(etrago.network.transformers.loc[
                etrago.network.transformers.r == 0, "r"]
                )} trafos without a resistance (r) in the data model.
            The resistance of these trafos will be automatically set to 0.0001.
            """
        )
    etrago.network.transformers.loc[
        etrago.network.transformers.r == 0, "r"
    ] = 0.0001

    # Set vnom of transformers
    etrago.network.transformers["v_nom"] = etrago.network.buses.loc[
        etrago.network.transformers.bus0.values, "v_nom"
    ].values

    # Drop methanation option in lowflex sceanrio
    if etrago.args["scn_name"] == "eGon2035_lowflex":
        etrago.network.links.drop(
            etrago.network.links[
                etrago.network.links.carrier == "H2_to_CH4"
            ].index,
            inplace=True,
        )
