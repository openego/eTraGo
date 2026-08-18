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
Defines the market optimization within eTraGo
"""

import os
import numpy as np
from pyomo.core import TransformationFactory, Var
import pyomo.core.plugins.transform

if "READTHEDOCS" not in os.environ:
    import logging

    from pypsa.components import component_attrs
    from shapely.geometry import Point
    import pandas as pd
    import geopandas as gpd
    import requests
    import os

    from etrago.cluster.electrical import postprocessing, preprocessing
    from etrago.tools.constraints import Constraints
    from etrago.tools.market_zones import create_market_zone_busmap
    from etrago.cluster.spatial import group_links

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, ClaraBuettner, CarlosEpia"


def log_committable_components(net):
    for component_name in ["generators", "links"]:
        df = getattr(net, component_name)

        if "committable" not in df.columns:
            print(
                f"\n{component_name}: no committable column",
                flush=True,
            )
            continue

        mask = df["committable"].fillna(False).astype(bool)
        selected = df.loc[mask].copy()

        print(
            f"\n{component_name}: " f"{len(selected)} committable components",
            flush=True,
        )

        columns = [
            col
            for col in [
                "carrier",
                "bus",
                "bus0",
                "bus1",
                "p_nom",
                "p_nom_extendable",
                "p_min_pu",
                "min_up_time",
                "min_down_time",
            ]
            if col in selected.columns
        ]

        if selected.empty:
            print("None", flush=True)
        else:
            print(
                selected[columns].to_string(),
                flush=True,
            )


def _unit_commitment_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "unit_commitment.csv",
    )


def _apply_unit_commitment_attributes(net):
    """
    Apply unit-commitment attributes to generators and links.

    UC parameters are read from data/unit_commitment.csv and assigned
    according to the component carrier.

    The diagnostic logging reports:
      - carriers defined in the UC CSV,
      - carriers present in the clustered network,
      - carrier overlap,
      - final number of committable generators and links.

    These diagnostics do not change the optimization formulation.
    """
    unit_commitment = pd.read_csv(
        _unit_commitment_path(),
        index_col=0,
    )

    unit_commitment.fillna(0, inplace=True)

    # ==============================================================
    # Diagnostic: compare network carriers with UC CSV carriers
    # ==============================================================

    uc_carriers = list(unit_commitment.columns.astype(str))

    if not net.generators.empty:
        gen_carriers = sorted(
            net.generators.carrier.dropna().astype(str).unique().tolist()
        )
    else:
        gen_carriers = []

    if not net.links.empty:
        link_carriers = sorted(
            net.links.carrier.dropna().astype(str).unique().tolist()
        )
    else:
        link_carriers = []

    gen_overlap = sorted(set(gen_carriers).intersection(uc_carriers))

    link_overlap = sorted(set(link_carriers).intersection(uc_carriers))

    logger.info(
        "UC DEBUG: UC carriers = %s",
        uc_carriers,
    )

    logger.info(
        "UC DEBUG: generator carriers after clustering = %s",
        gen_carriers,
    )

    logger.info(
        "UC DEBUG: generator carrier overlap = %s",
        gen_overlap,
    )

    logger.info(
        "UC DEBUG: link carriers after clustering = %s",
        link_carriers,
    )

    logger.info(
        "UC DEBUG: link carrier overlap = %s",
        link_overlap,
    )

    # ==============================================================
    # Generators
    # ==============================================================

    if not net.generators.empty:

        # Generator is committable when its carrier occurs as a
        # column in unit_commitment.csv.
        gen_attrs = net.generators.carrier.isin(
            unit_commitment.columns
        ).to_frame("committable")

        # Assign all UC attributes from unit_commitment.csv.
        for attr in unit_commitment.index:

            default = component_attrs["Generator"].default[attr]

            mapped = net.generators.carrier.map(
                unit_commitment.loc[attr]
            ).fillna(default)

            try:
                mapped = mapped.astype(
                    net.generators.carrier.map(unit_commitment.loc[attr]).dtype
                )
            except Exception:
                pass

            gen_attrs[attr] = mapped

        # Write attributes back to network.
        net.generators[gen_attrs.columns] = gen_attrs

        # ----------------------------------------------------------
        # Make minimum up/down times integer
        # ----------------------------------------------------------

        if "min_up_time" in net.generators.columns:
            net.generators["min_up_time"] = (
                net.generators["min_up_time"].fillna(0).astype(int)
            )

        if "min_down_time" in net.generators.columns:
            net.generators["min_down_time"] = (
                net.generators["min_down_time"].fillna(0).astype(int)
            )

        # ----------------------------------------------------------
        # Ensure ramp_limit_down exists for committable generators
        # ----------------------------------------------------------

        if "ramp_limit_down" in net.generators.columns:

            mask = net.generators["committable"].fillna(False).astype(bool)

            net.generators.loc[
                mask,
                "ramp_limit_down",
            ] = net.generators.loc[
                mask,
                "ramp_limit_down",
            ].fillna(1.0)

        # ----------------------------------------------------------
        # Preserve existing treatment of start/shutdown costs
        # ----------------------------------------------------------

        if "start_up_cost" in net.generators.columns:

            mask = net.generators["committable"].fillna(False).astype(bool)

            net.generators.loc[
                mask,
                "start_up_cost",
            ] = 0.0

        if "shut_down_cost" in net.generators.columns:

            mask = net.generators["committable"].fillna(False).astype(bool)

            net.generators.loc[
                mask,
                "shut_down_cost",
            ] = 0.0

    # ==============================================================
    # Links
    # ==============================================================

    if not net.links.empty:

        # Link is committable when its carrier occurs as a
        # column in unit_commitment.csv.
        link_attrs = net.links.carrier.isin(unit_commitment.columns).to_frame(
            "committable"
        )

        # Assign all UC attributes from unit_commitment.csv.
        for attr in unit_commitment.index:

            default = component_attrs["Link"].default[attr]

            mapped = net.links.carrier.map(unit_commitment.loc[attr]).fillna(
                default
            )

            try:
                mapped = mapped.astype(
                    net.links.carrier.map(unit_commitment.loc[attr]).dtype
                )
            except Exception:
                pass

            link_attrs[attr] = mapped

        # Write attributes back to network.
        net.links[link_attrs.columns] = link_attrs

        # ----------------------------------------------------------
        # Make minimum up/down times integer
        # ----------------------------------------------------------

        if "min_up_time" in net.links.columns:
            net.links["min_up_time"] = (
                net.links["min_up_time"].fillna(0).astype(int)
            )

        if "min_down_time" in net.links.columns:
            net.links["min_down_time"] = (
                net.links["min_down_time"].fillna(0).astype(int)
            )

        # ----------------------------------------------------------
        # Ensure ramp_limit_down exists for committable links
        # ----------------------------------------------------------

        if "ramp_limit_down" in net.links.columns:

            mask = net.links["committable"].fillna(False).astype(bool)

            net.links.loc[
                mask,
                "ramp_limit_down",
            ] = net.links.loc[
                mask,
                "ramp_limit_down",
            ].fillna(1.0)

        # ----------------------------------------------------------
        # Preserve reversible-link treatment
        # ----------------------------------------------------------

        if "p_min_pu" in net.links.columns:

            reversible_carriers = [
                "CH4",
                "DC",
                "AC",
                "H2_grid",
                "H2_saltcavern",
            ]

            net.links.loc[
                net.links.carrier.isin(reversible_carriers),
                "p_min_pu",
            ] = -1.0

        # ----------------------------------------------------------
        # Preserve existing treatment of start/shutdown costs
        # ----------------------------------------------------------

        if "start_up_cost" in net.links.columns:

            mask = net.links["committable"].fillna(False).astype(bool)

            net.links.loc[
                mask,
                "start_up_cost",
            ] = 0.0

        if "shut_down_cost" in net.links.columns:

            mask = net.links["committable"].fillna(False).astype(bool)

            net.links.loc[
                mask,
                "shut_down_cost",
            ] = 0.0

    # ==============================================================
    # Final diagnostic
    # ==============================================================

    if not net.generators.empty and "committable" in net.generators.columns:
        n_gen_uc = int(
            net.generators["committable"].fillna(False).astype(bool).sum()
        )
    else:
        n_gen_uc = 0

    if not net.links.empty and "committable" in net.links.columns:
        n_link_uc = int(
            net.links["committable"].fillna(False).astype(bool).sum()
        )
    else:
        n_link_uc = 0

    logger.info(
        "UC DEBUG: assigned committable generators = %d",
        n_gen_uc,
    )

    logger.info(
        "UC DEBUG: assigned committable links = %d",
        n_link_uc,
    )


def _disable_unit_commitment(net):
    """Ensure that the annual pre-market model is a continuous LP."""
    for component_name in ["generators", "links"]:
        df = getattr(net, component_name)

        if "committable" in df.columns:
            df.loc[:, "committable"] = False


def _effective_nominal(df, base):
    """
    Return *_nom_opt where available and positive, otherwise *_nom.

    base examples:
        p -> p_nom / p_nom_opt
        e -> e_nom / e_nom_opt
        s -> s_nom / s_nom_opt
    """
    nom_col = f"{base}_nom"
    opt_col = f"{base}_nom_opt"

    if df is None or df.empty:
        return pd.Series(dtype=float)

    nom = pd.Series(0.0, index=df.index, dtype=float)
    opt = pd.Series(pd.NA, index=df.index, dtype="Float64")

    if nom_col in df.columns:
        nom = pd.to_numeric(df[nom_col], errors="coerce").fillna(0.0)

    if opt_col in df.columns:
        opt = pd.to_numeric(df[opt_col], errors="coerce")

    out = opt.where(opt.notna(), nom)
    return out.clip(lower=0.0).astype(float).fillna(0.0)


def _freeze_nominal(df, base):
    """
    Freeze extendable nominal capacity after the pre-market model.

    Uses *_nom_opt if available and positive; otherwise falls back to *_nom.
    """
    if df is None or df.empty:
        return

    nom_col = f"{base}_nom"
    ext_col = f"{base}_nom_extendable"

    if nom_col in df.columns:
        df[nom_col] = _effective_nominal(df, base)

    if ext_col in df.columns:
        df[ext_col] = False


def _ensure_time_series_columns(ts_df, component_index, fill_value=0.0):
    """
    Ensure that a PyPSA time-series dataframe has all component columns.

    PyPSA usually stores time series as:
        index   = snapshots
        columns = component names

    With higher clustering resolution, static components may exist without
    corresponding time-series columns. This function prevents strict .loc
    calls from crashing.
    """
    if ts_df is None:
        return ts_df

    if component_index is None or len(component_index) == 0:
        return ts_df

    missing = pd.Index(component_index).difference(ts_df.columns)

    for comp in missing:
        ts_df[comp] = fill_value

    return ts_df


def _get_timeseries_row(ts_df, snapshot, components, label):
    """
    Safely return one snapshot row for selected components.

    Normal PyPSA orientation:
        ts_df.index   = snapshots
        ts_df.columns = component names

    The function also guards against missing columns and logs them instead
    of raising a KeyError.
    """
    components = pd.Index(components)

    if ts_df is None or ts_df.empty or len(components) == 0:
        return pd.Series(dtype=float)

    # Normal orientation: rows are snapshots, columns are components.
    if snapshot in ts_df.index:
        present = components.intersection(ts_df.columns)
        missing = components.difference(ts_df.columns)

        if len(missing) > 0:
            logger.warning(
                "%s: %s of %s requested components are missing from "
                "time-series columns. Existing initial values are kept. "
                "Example missing: %s",
                label,
                len(missing),
                len(components),
                list(missing[:5]),
            )

        if len(present) == 0:
            return pd.Series(dtype=float)

        return ts_df.loc[snapshot, present].astype(float).reindex(present)

    # Defensive fallback for accidentally transposed dataframes.
    if snapshot in ts_df.columns:
        present = components.intersection(ts_df.index)
        missing = components.difference(ts_df.index)

        if len(missing) > 0:
            logger.warning(
                "%s appears transposed: %s of %s requested components "
                "are missing from time-series index. Existing initial "
                "values are kept. Example missing: %s",
                label,
                len(missing),
                len(components),
                list(missing[:5]),
            )

        if len(present) == 0:
            return pd.Series(dtype=float)

        return ts_df.loc[present, snapshot].astype(float).reindex(present)

    logger.warning(
        "%s: snapshot %s not found in time-series index or columns.",
        label,
        snapshot,
    )
    return pd.Series(dtype=float)


def gas_clustering_market_model(self):
    from etrago.cluster.gas import (
        gas_postprocessing,
        preprocessing as gas_preprocessing,
    )

    if self.network.links[self.network.links.carrier == "H2_grid"].empty:
        logger.warning("H2 grid not clustered for market in this scenario")
        return

    ch4_network, weight_ch4, n_clusters_ch4 = gas_preprocessing(
        self, "CH4", apply_on="market_model"
    )

    df = pd.DataFrame(
        {
            "country": ch4_network.buses.country.unique(),
            "marketzone": ch4_network.buses.country.unique(),
        },
        columns=["country", "marketzone"],
    )

    df.loc[(df.country == "DE") | (df.country == "LU"), "marketzone"] = "DE/LU"

    df["cluster"] = df.groupby(df.marketzone).grouper.group_info[0]

    for i in ch4_network.buses.country.unique():
        ch4_network.buses.loc[ch4_network.buses.country == i, "cluster"] = (
            df.loc[df.country == i, "cluster"].values[0]
        )

    busmap = pd.Series(
        ch4_network.buses.cluster.astype(int).astype(str),
        ch4_network.buses.index,
    )

    if "H2_grid" in self.network.links.carrier.unique():
        h2_network, weight_h2, n_clusters_h2 = gas_preprocessing(
            self, "H2_grid", apply_on="market_model"
        )

        df_h2 = pd.DataFrame(
            {
                "country": h2_network.buses.country.unique(),
                "marketzone": h2_network.buses.country.unique(),
            },
            columns=["country", "marketzone"],
        )

        df_h2.loc[
            (df.country == "DE") | (df_h2.country == "LU"), "marketzone"
        ] = "DE/LU"

        df_h2["cluster"] = df_h2.groupby(df_h2.marketzone).grouper.group_info[
            0
        ] + len(df)

        for i in h2_network.buses.country.unique():
            h2_network.buses.loc[h2_network.buses.country == i, "cluster"] = (
                df_h2.loc[df_h2.country == i, "cluster"].values[0]
            )

        busmap = pd.concat(
            [
                busmap,
                pd.Series(
                    h2_network.buses.cluster.astype(int).astype(str),
                    h2_network.buses.index,
                ),
            ]
        )

    medoid_idx = pd.Series()
    # Set country tags for market model
    self.buses_by_country(apply_on="pre_market_model")
    self.geolocation_buses(apply_on="pre_market_model")

    self.pre_market_model, busmap_new = gas_postprocessing(
        self,
        busmap,
        medoid_idx=medoid_idx,
        apply_on="market_model",
        aggregate_generators_carriers=[],
    )


def market_optimization(self):
    logger.info("Start building pre market model")

    build_market_model(self)
    self.pre_market_model.determine_network_topology()

    # Diagnostic only:
    # list components that create binaries in the pre-market model.
    log_committable_components(self.pre_market_model)

    pm_gen_uc = int(
        self.pre_market_model.generators.get(
            "committable",
            pd.Series(False, index=self.pre_market_model.generators.index),
        )
        .fillna(False)
        .astype(bool)
        .sum()
    )

    pm_link_uc = int(
        self.pre_market_model.links.get(
            "committable",
            pd.Series(False, index=self.pre_market_model.links.index),
        )
        .fillna(False)
        .astype(bool)
        .sum()
    )

    logger.info(
        "Pre-market model: %d snapshots, %d generators, %d links, "
        "%d committable generators, %d committable links",
        len(self.pre_market_model.snapshots),
        len(self.pre_market_model.generators),
        len(self.pre_market_model.links),
        pm_gen_uc,
        pm_link_uc,
    )

    logger.info("Start solving pre market model")

    if self.args["method"]["formulation"] == "pyomo":
        standard_extra_functionality = Constraints(
            self.args,
            False,
        ).functionality

        def pyomo_linear_uc_extra_functionality(network, snapshots):
            standard_extra_functionality(network, snapshots)

            discrete_before = sum(
                variable.is_binary() or variable.is_integer()
                for variable in network.model.component_data_objects(
                    Var,
                    active=True,
                    descend_into=True,
                )
            )

            transformation = TransformationFactory("core.relax_integer_vars")

            if transformation is None:
                raise RuntimeError(
                    "Pyomo transformation core.relax_integer_vars "
                    "is unavailable."
                )

            transformation.apply_to(network.model)

            discrete_after = sum(
                variable.is_binary() or variable.is_integer()
                for variable in network.model.component_data_objects(
                    Var,
                    active=True,
                    descend_into=True,
                )
            )

            logger.info(
                "Pyomo integer relaxation: %d discrete variables before, "
                "%d after",
                discrete_before,
                discrete_after,
            )

        status, condition = self.pre_market_model.lopf(
            solver_name=self.args["solver"],
            solver_options=self.args["solver_options"],
            pyomo=True,
            extra_functionality=pyomo_linear_uc_extra_functionality,
            formulation=self.args["model_formulation"],
        )

    elif self.args["method"]["formulation"] == "linopy":
        status, condition = self.pre_market_model.optimize(
            solver_name=self.args["solver"],
            solver_options=self.args["solver_options"],
            extra_functionality=Constraints(
                self.args,
                False,
                apply_on="pre_market_model",
            ).functionality,
            linearized_unit_commitment=True,
        )

    else:
        raise ValueError("Method type must be either 'pyomo' or 'linopy'.")

    logger.info(
        "Pre-market optimization finished with " "status=%s and condition=%s",
        status,
        condition,
    )

    # Solver status can be returned as a string or enum-like object.
    status_text = str(status).strip().lower().split(".")[-1]
    condition_text = str(condition).strip().lower().split(".")[-1]

    # Do not export or continue with an unsolved pre-market model.
    if status_text != "ok" or condition_text != "optimal":
        raise RuntimeError(
            "Pre-market optimization did not terminate optimally. "
            f"status={status}, condition={condition}. "
            "Stopping before export and short-term market optimization."
        )

    # Export results of pre-market model.
    if self.args["csv_export"]:
        path = self.args["csv_export"]

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        self.pre_market_model.export_to_csv_folder(path + "/pre_market")

    logger.info("Preparing short-term UC market model")

    build_shortterm_market_model(self)
    self.market_model.determine_network_topology()

    logger.info("Start solving short-term UC market model")

    # Diagnostic only:
    # verify that UC attributes are still present after construction
    # of the short-term market model.
    log_committable_components(self.market_model)

    old_formulation = self.args["method"]["formulation"]
    self.args["method"]["formulation"] = "linopy"

    try:
        optimize_with_rolling_horizon(
            self.market_model,
            self.pre_market_model,
            snapshots=None,
            horizon=self.args["method"]["market_optimization"][
                "rolling_horizon"
            ]["planning_horizon"],
            overlap=self.args["method"]["market_optimization"][
                "rolling_horizon"
            ]["overlap"],
            solver_name=self.args["solver"],
            extra_functionality=Constraints(
                self.args,
                False,
                apply_on="market_model",
            ).functionality,
            args=self.args,
        )

    finally:
        self.args["method"]["formulation"] = old_formulation

    # Export results of market model.
    if self.args["csv_export"]:
        path = self.args["csv_export"]

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        self.market_model.export_to_csv_folder(path + "/market")


def build_market_model(self):
    """
    Build the pre-market model based on the imported eTraGo network.

    Workflow:
        1. Preprocess the network.
        2. Create market-zone clustering.
        3. Convert clustered AC lines to market transshipment links.
        4. Restore generator availability time series.
        5. Configure cyclic stores/storage units.
        6. Apply gas-network reduction.
        7. Aggregate parallel sector-coupling links.
        8. Apply unit-commitment attributes to the FINAL pre-market model.
        9. Add country/geolocation information.

    Unit commitment is deliberately applied after all clustering and
    aggregation operations so that the network passed to the solver
    contains the correct committable generators and links.
    """

    # ==============================================================
    # Preprocessing
    # ==============================================================

    net, weight, n_clusters, busmap_foreign = preprocessing(
        self,
        apply_on="market_model",
    )

    market_zones = self.args["method"]["market_optimization"]["market_zones"]

    # ==============================================================
    # Build market-zone busmap
    # ==============================================================

    if market_zones == "status_quo":

        df = pd.DataFrame(
            {
                "country": net.buses.country.unique(),
                "marketzone": net.buses.country.unique(),
            },
            columns=[
                "country",
                "marketzone",
            ],
        )

        # Germany and Luxembourg form one bidding zone.
        df.loc[
            (df.country == "DE") | (df.country == "LU"),
            "marketzone",
        ] = "DE/LU"

        df["cluster"] = df.groupby(df.marketzone).grouper.group_info[0]

        for country in net.buses.country.unique():
            net.buses.loc[
                net.buses.country == country,
                "cluster",
            ] = df.loc[
                df.country == country,
                "cluster",
            ].values[0]

        busmap = pd.Series(
            net.buses.cluster.astype(int).astype(str),
            net.buses.index,
        )

        medoid_idx = pd.Series(dtype=str)

    elif market_zones in [
        "DE2",
        "DE3",
        "DE4",
        "DE5",
    ]:

        busmap, medoid_idx = create_market_zone_busmap(
            net,
            market_zones,
        )

    else:
        raise ValueError(
            f"Market zone setting {market_zones} is not available. "
            "Please use one of "
            "['status_quo', 'DE2', 'DE3', 'DE4', 'DE5']."
        )

    # ==============================================================
    # Market-zone-specific clustering
    # ==============================================================

    logger.info("Start market zone specific clustering")

    clustering, busmap = postprocessing(
        self,
        busmap,
        busmap_foreign,
        medoid_idx,
        aggregate_generators_carriers=[],
        aggregate_links=False,
        apply_on="market_model",
    )

    net = clustering.network

    # ==============================================================
    # Convert clustered AC lines to abstract market links
    # ==============================================================

    if not net.lines.empty and "carrier" in net.lines.columns:

        ac = net.lines[net.lines.carrier == "AC"].copy()

        if not ac.empty:

            ac.index = "transshipment_" + ac.index.astype(str)

            link_df = (
                ac.loc[
                    :,
                    [
                        "bus0",
                        "bus1",
                        "capital_cost",
                        "length",
                    ],
                ]
                .assign(p_nom=ac.s_nom)
                .assign(p_nom_min=ac.s_nom_min)
                .assign(p_nom_max=ac.s_nom_max)
                .assign(p_nom_extendable=ac.s_nom_extendable)
                .assign(p_max_pu=ac.s_max_pu)
                .assign(p_min_pu=-1.0)
                .assign(carrier="DC")
                .set_index(ac.index)
            )

            net.import_components_from_dataframe(
                link_df,
                "Link",
            )

            net.lines.drop(
                net.lines.loc[net.lines.carrier == "AC"].index,
                inplace=True,
            )

    # ==============================================================
    # Restore generator availability time series
    # ==============================================================

    if hasattr(self, "network_tsa"):

        try:
            net.generators_t.p_max_pu = self.network_tsa.generators_t.p_max_pu

        except Exception as exc:
            logger.warning(
                "Could not assign " "network_tsa.generators_t.p_max_pu: %s",
                exc,
            )

    # ==============================================================
    # Configure cyclic storage behavior for pre-market model
    # ==============================================================

    if not net.stores.empty and "carrier" in net.stores.columns:

        net.stores.loc[
            net.stores.carrier != "battery_storage",
            "e_cyclic",
        ] = True

    if not net.storage_units.empty:

        net.storage_units["cyclic_state_of_charge"] = True

    # ==============================================================
    # Assign network as pre-market model
    # ==============================================================

    self.pre_market_model = net

    # ==============================================================
    # Gas-network reduction
    # ==============================================================

    logger.info("Start gas clustering for pre-market model")

    gas_clustering_market_model(self)

    # ==============================================================
    # Aggregate parallel sector-coupling links
    # ==============================================================

    logger.info(
        "Aggregate parallel sector-coupling links " "in pre-market model"
    )

    (
        self.pre_market_model.links,
        self.pre_market_model.links_t,
    ) = group_links(
        self.pre_market_model,
        carriers=[
            "central_heat_pump",
            "central_resistive_heater",
            "rural_heat_pump",
            "rural_resistive_heater",
            "BEV_charger",
            "dsm",
            "central_gas_boiler",
            "rural_gas_boiler",
        ],
    )

    # ==============================================================
    # Apply unit commitment
    # ==============================================================
    #
    # IMPORTANT:
    # Apply UC only AFTER all clustering and aggregation operations.
    # This ensures that committable attributes are assigned to exactly
    # the components that will be sent to the optimizer.
    #
    # Do NOT call _disable_unit_commitment() after this point.
    # ==============================================================

    logger.info(
        "Apply unit-commitment attributes " "to final pre-market model"
    )

    _apply_unit_commitment_attributes(self.pre_market_model)

    # ==============================================================
    # Final diagnostics
    # ==============================================================

    n_gen_uc = 0
    n_link_uc = 0

    if (
        not self.pre_market_model.generators.empty
        and "committable" in self.pre_market_model.generators.columns
    ):

        n_gen_uc = int(
            self.pre_market_model.generators["committable"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    if (
        not self.pre_market_model.links.empty
        and "committable" in self.pre_market_model.links.columns
    ):

        n_link_uc = int(
            self.pre_market_model.links["committable"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    logger.info(
        "Final pre-market UC configuration: "
        "%d committable generators, "
        "%d committable links",
        n_gen_uc,
        n_link_uc,
    )

    # ==============================================================
    # Country/geolocation information
    # ==============================================================

    self.buses_by_country(apply_on="pre_market_model")

    self.geolocation_buses(apply_on="pre_market_model")


def optimize_with_rolling_horizon(
    n,
    pre_market,
    snapshots,
    horizon,
    overlap,
    solver_name,
    extra_functionality,
    args,
):
    """
    Optimize the short-term market model in rolling-horizon mode.

    This version is robust against higher spatial clustering resolutions,
    where static stores/storage units can differ from the available
    time-series columns.
    """
    if snapshots is None:
        snapshots = n.snapshots

    snapshots = pd.Index(snapshots)

    if len(snapshots) == 0:
        logger.warning(
            "No snapshots supplied to rolling-horizon optimization."
        )
        return n

    if horizon <= overlap:
        raise ValueError("overlap must be smaller than horizon")

    if not n.links.empty:
        n.links["marginal_cost_quadratic"] = 0.0

    # Make sure the most important time-series tables contain all
    # current components as columns. Missing columns are filled with
    # neutral values.
    if not n.stores.empty:
        n.stores_t.e = _ensure_time_series_columns(
            n.stores_t.e,
            n.stores.index,
            fill_value=0.0,
        )

        n.stores_t.e_min_pu = _ensure_time_series_columns(
            n.stores_t.e_min_pu,
            n.stores.index,
            fill_value=0.0,
        )

        n.stores_t.e_max_pu = _ensure_time_series_columns(
            n.stores_t.e_max_pu,
            n.stores.index,
            fill_value=1.0,
        )

    if not n.storage_units.empty:
        n.storage_units_t.state_of_charge = _ensure_time_series_columns(
            n.storage_units_t.state_of_charge,
            n.storage_units.index,
            fill_value=0.0,
        )

    if not pre_market.stores.empty:
        pre_market.stores_t.e = _ensure_time_series_columns(
            pre_market.stores_t.e,
            pre_market.stores.index,
            fill_value=0.0,
        )

    if not pre_market.storage_units.empty:
        pre_market.storage_units_t.state_of_charge = (
            _ensure_time_series_columns(
                pre_market.storage_units_t.state_of_charge,
                pre_market.storage_units.index,
                fill_value=0.0,
            )
        )

    starting_points = list(
        range(
            0,
            len(snapshots),
            horizon - overlap,
        )
    )

    for i, start in enumerate(starting_points):
        end = min(
            len(snapshots),
            start + horizon,
        )

        sns = snapshots[start:end]

        logger.info(
            "Optimizing network for snapshot horizon [%s:%s] (%s/%s).",
            sns[0],
            sns[-1],
            i + 1,
            len(starting_points),
        )

        previous_snapshot = snapshots[start - 1]
        end_snapshot = snapshots[end - 1]

        # --------------------------------------------------------------
        # Store state handover
        # --------------------------------------------------------------
        if not n.stores.empty:
            carrier = n.stores.carrier.astype(str)

            stores_no_dsm = n.stores.index[
                ~carrier.isin(
                    [
                        "dsm",
                        "battery_storage",
                    ]
                )
            ]

            store_initial = _get_timeseries_row(
                n.stores_t.e,
                previous_snapshot,
                stores_no_dsm,
                "n.stores_t.e initial store handover",
            )

            if not store_initial.empty:
                n.stores.loc[
                    store_initial.index,
                    "e_initial",
                ] = store_initial

            # Seasonal stores follow the pre-market trajectory.
            seasonal_stores = n.stores.index[
                carrier.isin(
                    [
                        "central_heat_store",
                        "H2_overground",
                        "CH4",
                    ]
                )
            ]

            seasonal_initial = _get_timeseries_row(
                pre_market.stores_t.e,
                previous_snapshot,
                seasonal_stores,
                "pre_market.stores_t.e seasonal initial",
            )

            if not seasonal_initial.empty:
                n.stores.loc[
                    seasonal_initial.index,
                    "e_initial",
                ] = seasonal_initial

            # End-of-window seasonal store target from pre-market model.
            seasonal_end = _get_timeseries_row(
                pre_market.stores_t.e,
                end_snapshot,
                seasonal_stores,
                "pre_market.stores_t.e seasonal end",
            )

            if not seasonal_end.empty:
                available = seasonal_end.index.intersection(
                    pre_market.stores.index
                )

                available = available.intersection(n.stores.index)

                if len(available) > 0:
                    e_nom = _effective_nominal(
                        pre_market.stores.loc[available],
                        "e",
                    )

                    e_nom = pd.to_numeric(
                        e_nom,
                        errors="coerce",
                    ).astype("float64")

                    # Avoid division by zero.
                    e_nom = e_nom.mask(
                        e_nom == 0.0,
                        np.nan,
                    )

                    seasonal_values = pd.to_numeric(
                        seasonal_end.reindex(available),
                        errors="coerce",
                    ).astype("float64")

                    ratio = seasonal_values.div(e_nom)

                    # Remove NaN and infinite ratios.
                    ratio = ratio.where(np.isfinite(ratio)).dropna()

                    if len(ratio) > 0:
                        for store in ratio.index:
                            if store not in n.stores_t.e_max_pu.columns:
                                n.stores_t.e_max_pu[store] = 1.0

                            if store not in n.stores_t.e_min_pu.columns:
                                n.stores_t.e_min_pu[store] = 0.0

                        n.stores_t.e_max_pu.loc[
                            end_snapshot,
                            ratio.index,
                        ] = (
                            ratio * 1.01
                        )

                        n.stores_t.e_min_pu.loc[
                            end_snapshot,
                            ratio.index,
                        ] = (
                            ratio * 0.99
                        )

            n.stores_t.e_min_pu.fillna(
                0.0,
                inplace=True,
            )

            n.stores_t.e_max_pu.fillna(
                1.0,
                inplace=True,
            )

        # --------------------------------------------------------------
        # Storage-unit state-of-charge handover
        # --------------------------------------------------------------
        if not n.storage_units.empty:
            storage_units = n.storage_units.index

            if i == 0:
                # Preserve the original logic:
                # the first short-term window starts from the
                # end-of-period pre-market SOC.
                if not pre_market.storage_units_t.state_of_charge.empty:
                    first_soc_snapshot = (
                        pre_market.storage_units_t.state_of_charge.index[-1]
                    )

                    soc_initial = _get_timeseries_row(
                        pre_market.storage_units_t.state_of_charge,
                        first_soc_snapshot,
                        storage_units,
                        (
                            "pre_market.storage_units_t."
                            "state_of_charge first window"
                        ),
                    )

                else:
                    soc_initial = pd.Series(dtype=float)

            else:
                soc_initial = _get_timeseries_row(
                    n.storage_units_t.state_of_charge,
                    previous_snapshot,
                    storage_units,
                    ("n.storage_units_t.state_of_charge " "rolling handover"),
                )

            if not soc_initial.empty:
                n.storage_units.loc[
                    soc_initial.index,
                    "state_of_charge_initial",
                ] = soc_initial

            if i == len(starting_points) - 1:
                extra_functionality = Constraints(
                    args,
                    False,
                    apply_on="last_market_model",
                ).functionality

        # --------------------------------------------------------------
        # Diagnostic: verify UC components in every rolling window
        # --------------------------------------------------------------
        n_gen_uc = int(
            n.generators.get(
                "committable",
                pd.Series(
                    False,
                    index=n.generators.index,
                ),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        )

        n_link_uc = int(
            n.links.get(
                "committable",
                pd.Series(
                    False,
                    index=n.links.index,
                ),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        )

        logger.info(
            "Rolling window %s/%s: %d committable generators, "
            "%d committable links",
            i + 1,
            len(starting_points),
            n_gen_uc,
            n_link_uc,
        )

        status, condition = n.optimize(
            sns,
            solver_name=solver_name,
            solver_options=args["solver_options"],
            extra_functionality=extra_functionality,
            linearized_unit_commitment=True,
        )

        logger.info(
            "Rolling window %s/%s finished with " "status=%s and condition=%s",
            i + 1,
            len(starting_points),
            status,
            condition,
        )

        if status != "ok":
            logger.warning(
                "Optimization failed with status %s and condition %s",
                status,
                condition,
            )

            try:
                n.model.print_infeasibilities()

            except Exception as exc:
                logger.warning(
                    "Could not print infeasibilities: %s",
                    exc,
                )

    return n


def build_shortterm_market_model(self):
    """
    Build the short-term UC market model from the solved pre-market model.

    Capacities are fixed to pre-market optimized capacities, but safely fall
    back to original nominal capacities if *_nom_opt is missing or zero.
    """
    m = self.pre_market_model.copy()

    _freeze_nominal(m.storage_units, "p")
    _freeze_nominal(m.stores, "e")
    _freeze_nominal(m.links, "p")
    _freeze_nominal(m.lines, "s")

    if not m.stores.empty:
        m.stores["e_cyclic"] = False

        m.stores_t.e = _ensure_time_series_columns(
            m.stores_t.e,
            m.stores.index,
            fill_value=0.0,
        )
        m.stores_t.e_min_pu = _ensure_time_series_columns(
            m.stores_t.e_min_pu,
            m.stores.index,
            fill_value=0.0,
        )
        m.stores_t.e_max_pu = _ensure_time_series_columns(
            m.stores_t.e_max_pu,
            m.stores.index,
            fill_value=1.0,
        )

    if not m.storage_units.empty:
        m.storage_units["cyclic_state_of_charge"] = False

        m.storage_units_t.state_of_charge = _ensure_time_series_columns(
            m.storage_units_t.state_of_charge,
            m.storage_units.index,
            fill_value=0.0,
        )

    _apply_unit_commitment_attributes(m)

    self.market_model = m

    # Set country tags for market model.
    self.buses_by_country(apply_on="market_model")
    self.geolocation_buses(apply_on="market_model")
