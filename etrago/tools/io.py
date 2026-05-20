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
-----------
packagename: str
    Package containing orm class definitions
temp_ormclass: str
    Orm class name of table with temporal resolution
carr_ormclass: str
    Orm class name of table with carrier id to carrier name datasets

Notes
-------
A configuration file connecting the chosen optimization method with
components to be queried is needed for NetworkScenario class.
"""

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, mariusves, pieterhexen, ClaraBuettner"

import logging
import os

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

if "READTHEDOCS" not in os.environ:

    from sqlalchemy import literal_column
    from sqlalchemy.orm.exc import NoResultFound
    import saio


carr_ormclass = "Source"


class ScenarioBase:
    """Base class to address the dynamic provision of orm classes representing
    powerflow components from egoio

    Parameters
    ----------
    session : sqla.orm.session.Session
        Handles conversations with the database.
    version : str
        Version number of data version control in grid schema of the oedb.
    """

    def __init__(self, engine, session, version=None):
        global carr_ormclass  # noqa: F824

        saio.register_schema("grid", engine)
        self.session = session
        self.version = version


class NetworkScenario(ScenarioBase):
    """Adapter class between oedb powerflow data and PyPSA. Provides the
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
        self,
        engine,
        session,
        scn_name="Status Quo",
        start_snapshot=1,
        end_snapshot=20,
        temp_id=1,
        scenario_extension=False,
        **kwargs,
    ):
        self.scn_name = scn_name
        self.start_snapshot = start_snapshot
        self.end_snapshot = end_snapshot
        self.temp_id = temp_id
        self.scenario_extension = scenario_extension

        super().__init__(engine, session, **kwargs)

        # network: pypsa.Network
        self.network = None

        if "oep.iks.cs.ovgu.de" in str(engine.url):
            self.oep_active = True
        else:
            self.oep_active = False

        if self.oep_active:
            saio.register_schema("tables", engine)
        else:
            saio.register_schema("grid", engine)

        self.configure_timeindex()

    def __repr__(self):
        r = "NetworkScenario: %s" % self.scn_name

        if not self.network:
            r += """
            \nTo create a PyPSA network call <NetworkScenario>.build_network().
            """

        return r

    def configure_timeindex(self):
        """Construct a DateTimeIndex with the queried temporal resolution,
        start- and end_snapshot."""

        if self.oep_active:
            from saio.tables import edut_00_073 as egon_etrago_temp_resolution
        else:
            from saio.grid import egon_etrago_temp_resolution

        try:
            if self.version:
                tr = saio.as_pandas(
                    self.session.query(egon_etrago_temp_resolution)
                    .filter(
                        egon_etrago_temp_resolution.version == self.version
                    )
                    .filter(
                        egon_etrago_temp_resolution.temp_id == self.temp_id
                    )
                ).squeeze()
            else:
                tr = saio.as_pandas(
                    self.session.query(egon_etrago_temp_resolution).filter(
                        egon_etrago_temp_resolution.temp_id == self.temp_id
                    )
                ).squeeze()

        except (KeyError, NoResultFound):
            print("temp_id %s does not exist." % self.temp_id)

        timeindex = pd.DatetimeIndex(
            data=pd.date_range(
                start=tr.start_time, periods=tr.timesteps, freq=tr.resolution
            )
        )

        self.timeindex = timeindex[self.start_snapshot - 1 : self.end_snapshot]

    def id_to_source(self):
        ormclass = self._mapped["Source"]
        query = self.session.query(ormclass)

        if self.version:
            query = query.filter(ormclass.version == self.version)

        # TODO column naming in database
        return {k.source_id: k.name for k in query.all()}

    def fetch_by_relname(self, name):
        """Construct DataFrame with component data from filtered table data.

        Parameters
        ----------
        name : str
            Component name.

        Returns
        -------
        pd.DataFrame
            Component data.
        """

        if self.oep_active:
            from saio.tables import (  # noqa: F401
                edut_00_056 as egon_etrago_bus,
                edut_00_060 as egon_etrago_generator,
                edut_00_063 as egon_etrago_line,
                edut_00_065 as egon_etrago_link,
                edut_00_067 as egon_etrago_load,
                edut_00_069 as egon_etrago_storage,
                edut_00_071 as egon_etrago_store,
                edut_00_074 as egon_etrago_transformer,
            )

        else:
            from saio.grid import (  # noqa: F401
                egon_etrago_bus,
                egon_etrago_generator,
                egon_etrago_line,
                egon_etrago_link,
                egon_etrago_load,
                egon_etrago_storage,
                egon_etrago_store,
                egon_etrago_transformer,
            )

        if self.scenario_extension:
            from saio.grid import (  # noqa: F401,F811
                egon_etrago_extension_bus as egon_etrago_bus,
                egon_etrago_extension_line as egon_etrago_line,
                egon_etrago_extension_link as egon_etrago_link,
                egon_etrago_extension_transformer as egon_etrago_transformer,
            )

        index = f"{name.lower()}_id"

        if name == "Transformer":
            index = "trafo_id"

        query = self.session.query(
            vars()[f"egon_etrago_{name.lower()}"]
        ).filter(
            vars()[f"egon_etrago_{name.lower()}"].scn_name == self.scn_name
        )

        if self.version:
            query = query.filter(
                vars()[f"egon_etrago_{name.lower()}"].version == self.version
            )

        df_saio = saio.as_pandas(query, crs=4326, geometry=None).set_index(
            index
        )

        # Drop internal id column from test oep
        df_saio.drop("id", axis="columns", inplace=True, errors="ignore")

        # Copy data into new dataframe which a´has column names with type 'str'
        # When using saio, the data type of column names is 'quoted_name',
        # which leads to errors when using pandas.groupby()
        df = pd.DataFrame()
        for col in df_saio.columns:
            df.loc[:, str(col)] = df_saio.loc[:, col]

        if name == "Transformer":
            df.tap_side = 0
            df.tap_position = 0
            df = df.drop_duplicates()

        if "source" in df:
            df.source = df.source.map(self.id_to_source())

        return df

    def series_fetch_by_relname(self, network, name, pypsa_name):
        """Construct DataFrame with component timeseries data from filtered
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

        if self.oep_active:
            from saio.tables import (  # noqa: F401
                edut_00_057 as egon_etrago_bus_timeseries,
                edut_00_061 as egon_etrago_generator_timeseries,
                edut_00_064 as egon_etrago_line_timeseries,
                edut_00_066 as egon_etrago_link_timeseries,
                edut_00_068 as egon_etrago_load_timeseries,
                edut_00_070 as egon_etrago_storage_timeseries,
                edut_00_072 as egon_etrago_store_timeseries,
                edut_00_075 as egon_etrago_transformer_timeseries,
            )

        else:

            from saio.grid import (  # noqa: F401
                egon_etrago_bus_timeseries,
                egon_etrago_generator_timeseries,
                egon_etrago_line_timeseries,
                egon_etrago_link_timeseries,
                egon_etrago_load_timeseries,
                egon_etrago_storage_timeseries,
                egon_etrago_store_timeseries,
                egon_etrago_transformer_timeseries,
            )

        # Select index column
        if name == "Transformer":
            index_col = "trafo_id"

        else:
            index_col = f"{name.lower()}_id"

        # Select columns with time series data
        query_columns = self.session.query(
            vars()[f"egon_etrago_{name.lower()}_timeseries"]
        ).limit(1)

        key_columns = ["scn_name", index_col, "temp_id", "id"]

        if self.version:
            key_columns.append(["version"])

        columns = saio.as_pandas(query_columns).columns.drop(
            key_columns, errors="ignore"
        )

        # Query and import time series data
        for col in columns:
            tbl = vars()[f"egon_etrago_{name.lower()}_timeseries"]

            query = self.session.query(
                getattr(tbl, index_col),
                literal_column(
                    f"{col}[{self.start_snapshot}:{self.end_snapshot}]"
                ).label(col),
            ).filter(tbl.scn_name == self.scn_name)

            if self.version:
                query = query.filter(
                    vars()[f"egon_etrago_{name.lower()}_timeseries"].version
                    == self.version
                )

            df_all = saio.as_pandas(query)

            # Rename index
            df_all.set_index(index_col, inplace=True)

            df_all.index = df_all.index.astype(str)

            if not df_all.isnull().all().all():

                # Drop empty series
                df_all = df_all[~df_all[col].isnull()]

                df = df_all[col].apply(pd.Series).transpose()

                df.index = self.timeindex

                pypsa.io.import_series_from_dataframe(
                    network, df, pypsa_name, col
                )

        return network

    def build_network(self, network=None, *args, **kwargs):
        """Core method to construct PyPSA Network object."""
        if network is not None:
            network = network

        else:
            network = pypsa.Network()
            network.set_snapshots(self.timeindex)

        for comp in [
            "Bus",
            "Line",
            "Transformer",
            "Link",
            "Load",
            "Generator",
            "Storage",
            "Store",
        ]:
            pypsa_comp = "StorageUnit" if comp == "Storage" else comp

            if comp[-1] == "s":
                logger.info(f"Importing {comp}es from database")
            else:
                logger.info(f"Importing {comp}s from database")

            df = self.fetch_by_relname(comp)

            # Drop columns with only NaN values
            df = df.drop(df.isnull().all()[df.isnull().all()].index, axis=1)

            # Replace NaN values with defailt values from pypsa
            for c in df.columns:
                if c in network.component_attrs[pypsa_comp].index:
                    df[c].fillna(
                        network.component_attrs[pypsa_comp].default[c],
                        inplace=True,
                    )

            if pypsa_comp == "Generator":
                df.sign = 1

            network.import_components_from_dataframe(df, pypsa_comp)

            network = self.series_fetch_by_relname(network, comp, pypsa_comp)

        # populate carrier attribute in PyPSA network
        # network.import_components_from_dataframe(
        #     self.fetch_by_relname(carr_ormclass), 'Carrier')

        self.network = network

        return network


def run_sql_script(conn, scriptname="results_md2grid.sql"):
    """This function runs .sql scripts in the folder 'sql_scripts'"""

    script_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "sql_scripts")
    )
    script_str = open(os.path.join(script_dir, scriptname)).read()
    conn.execution_options(autocommit=True).execute(script_str)

    return


def extension(self, **kwargs):
    """
    Function that adds an additional network to the existing network container.
    The new network can include every PyPSA-component (e.g. buses, lines,
    links).
    To connect it to the existing network, transformers are needed.

    All components and its timeseries of the additional scenario need to be
    inserted in the fitting 'model_draft.ego_grid_pf_hv_extension\_' table.
    The scn_name in the tables have to be labled with 'extension\_' + scn_name
    (e.g. 'extension_nep2035').

    Until now, the tables include three additional scenarios:
    'nep2035_confirmed': all new lines and needed transformers planed in the
    'Netzentwicklungsplan 2035' (NEP2035) that have been confirmed by the
    Bundesnetzagentur (BNetzA)

    'nep2035_b2': all new lines and needed transformers planned in the NEP 2035
    in the scenario 2035 B2

    'BE_NO_NEP 2035': DC-lines and transformers to connect the upcomming
    electrical-neighbours Belgium and Norway
    Generation, loads and its timeseries in Belgium and Norway for scenario
    'NEP 2035'

    Parameters
    -----------
    network :
        The existing network container (e.g. scenario 'NEP 2035')
    session :
        session-data
    overlay_scn_name :
        Name of the additional scenario (WITHOUT 'extension\_')
    start_snapshot :
    end_snapshot:
        Simulation time

    Returns
    -------
    Network container including existing and additional network

    """
    if self.args["scn_extension"] is not None:
        for i in range(len(self.args["scn_extension"])):
            scn_extension = self.args["scn_extension"][i]
            # Adding overlay-network to existing network
            scenario = NetworkScenario(
                self.session,
                version=self.args["gridversion"],
                start_snapshot=self.args["start_snapshot"],
                end_snapshot=self.args["end_snapshot"],
                scn_name=scn_extension,
                scenario_extension=True,
            )

            self.network = scenario.build_network(self.network)

            # Allow lossless links to conduct bidirectional
            self.network.links.loc[
                self.network.links.efficiency == 1.0, "p_min_pu"
            ] = -1


def decommissioning(self, **kwargs):
    """
    Function that removes components in a decommissioning-scenario from
    the existing network container.
    Currently, only lines can be decommissioned.

    All components of the decommissioning scenario need to be inserted in
    the fitting 'model_draft.ego_grid_pf_hv_extension\_' table.
    The scn_name in the tables have to be labled with 'decommissioning\_'
    + scn_name (e.g. 'decommissioning_nep2035').

    Parameters
    -----------
    network :
        The existing network container (e.g. scenario 'NEP 2035')
    session :
        session-data
    overlay_scn_name :
        Name of the decommissioning scenario

    Returns
    ------
    Network container including decommissioning

    """
    if self.args["scn_extension"] is not None:
        for i in range(len(self.args["scn_extension"])):
            scn_decom = self.args["scn_extension"][i]

            df_decommisionning = pd.read_sql(
                f"""
                SELECT * FROM
                grid.egon_etrago_extension_line
                WHERE scn_name = 'decomissioining_{scn_decom}'
                """,
                self.session.bind,
            )

            self.network.mremove(
                "Line",
                df_decommisionning.line_id.astype(str).values,
            )

            # buses only between removed lines
            candidates = pd.concat(
                [df_decommisionning.bus0, df_decommisionning.bus1]
            )
            candidates.drop_duplicates(inplace=True, keep="first")
            candidates = candidates.astype(str)

            # Drop buses that are connecting other lines
            candidates = candidates[~candidates.isin(self.network.lines.bus0)]
            candidates = candidates[~candidates.isin(self.network.lines.bus1)]

            # Drop buses that are connection other DC-lines
            candidates = candidates[~candidates.isin(self.dc_lines().bus0)]
            candidates = candidates[~candidates.isin(self.dc_lines().bus1)]

            drop_buses = self.network.buses[
                (self.network.buses.index.isin(candidates.values))
                & (self.network.buses.country == "DE")
            ]

            drop_links = self.network.links[
                (self.network.links.bus0.isin(drop_buses.index))
                | (self.network.links.bus1.isin(drop_buses.index))
            ].index

            drop_trafos = self.network.transformers[
                (self.network.transformers.bus0.isin(drop_buses.index))
                | (self.network.transformers.bus1.isin(drop_buses.index))
            ].index

            drop_su = self.network.storage_units[
                self.network.storage_units.bus.isin(candidates.values)
            ].index

            self.network.mremove("StorageUnit", drop_su)

            self.network.mremove("Transformer", drop_trafos)

            self.network.mremove("Link", drop_links)

            self.network.mremove("Bus", drop_buses.index)


def distance(x0, x1, y0, y1):
    """
    Function that calculates the square of the distance between two points.

    Parameters
    ---------
    x0 :
        x - coordinate of point 0
    x1 :
        x - coordinate of point 1
    y0 :
        y - coordinate of point 0
    y1 :
        y - coordinate of point 1

    Returns
    --------
    distance : float
        square of distance

    """
    # Calculate square of the distance between two points (Pythagoras)
    distance = (x1.values - x0.values) * (x1.values - x0.values) + (
        y1.values - y0.values
    ) * (y1.values - y0.values)
    return distance


def calc_nearest_point(bus1, network):
    """
    Function that finds the geographical nearest point in a network from a
    given bus.

    Parameters
    -----------
    bus1 : float
        id of bus
    network : Pypsa network container
        network including the comparable buses

    Returns
    -------
    bus0 : float
        bus_id of nearest point

    """

    bus1_index = network.buses.index[network.buses.index == bus1]

    forbidden_buses = np.append(
        bus1_index.values,
        network.lines.bus1[network.lines.bus0 == bus1].values,
    )

    forbidden_buses = np.append(
        forbidden_buses, network.lines.bus0[network.lines.bus1 == bus1].values
    )

    forbidden_buses = np.append(
        forbidden_buses, network.links.bus0[network.links.bus1 == bus1].values
    )

    forbidden_buses = np.append(
        forbidden_buses, network.links.bus1[network.links.bus0 == bus1].values
    )

    x0 = network.buses.x[network.buses.index.isin(bus1_index)]

    y0 = network.buses.y[network.buses.index.isin(bus1_index)]

    comparable_buses = network.buses[
        ~network.buses.index.isin(forbidden_buses)
    ]

    x1 = comparable_buses.x

    y1 = comparable_buses.y

    distance = (x1.values - x0.values) * (x1.values - x0.values) + (
        y1.values - y0.values
    ) * (y1.values - y0.values)

    min_distance = distance.min()

    bus0 = comparable_buses[
        (
            (
                (x1.values - x0.values) * (x1.values - x0.values)
                + (y1.values - y0.values) * (y1.values - y0.values)
            )
            == min_distance
        )
    ]
    bus0 = bus0.index[bus0.index == bus0.index.max()]
    bus0 = "".join(bus0.values)

    return bus0


def add_ch4_h2_correspondence(self):
    """
    Method adding the database table grid.egon_etrago_ch4_h2 to self.
    It contains the mapping from H2 buses to their corresponding CH4 buses.

    """
    h2_buses = self.network.buses[self.network.buses.carrier == "H2_grid"]
    self.ch4_h2_mapping = pd.Series()
    for h2_bus in h2_buses.index:
        x = h2_buses.loc[h2_bus, "x"]
        y = h2_buses.loc[h2_bus, "y"]
        self.ch4_h2_mapping.loc[
            self.network.buses[
                (self.network.buses.carrier == "CH4")
                & (self.network.buses.x == x)
                & (self.network.buses.y == y)
            ].index[0]
        ] = h2_bus
        self.ch4_h2_mapping.index.name = "CH4_bus"


if __name__ == "__main__":
    if pypsa.__version__ not in ["0.6.2", "0.11.0"]:
        print("Pypsa version %s not supported." % pypsa.__version__)
    pass
