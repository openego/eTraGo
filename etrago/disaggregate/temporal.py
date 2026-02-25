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
temporal.py defines the methods to run temporal disaggregation on networks.
"""

from datetime import timedelta
import logging
import math
import os
import time

import pandas as pd

logger = logging.getLogger(__name__)

if "READTHEDOCS" not in os.environ:

    from etrago.execute import optimize_with_rolling_horizon
    from etrago.tools.constraints import Constraints


__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = (
    "ulfmueller, s3pp, wolfbunke, mariusves, lukasol, KathiEsterl, "
    "ClaraBuettner, CarlosEpia, AmeliaNadal"
)


def temp_disagg_soc(n, n_tsa):
    """Interpolate state of charges for not-optimized snapshots

    Parameters
    ----------
    n : pypsa.Network
        Network in full temporal resolution without optimization results
    n_tsa : pypsa.Network
        Network in reduced temporal resolution with optimization results

    Returns
    -------
    stores_e_full_ts : pd.DataFrame
        Interpolated store-e timeseries for each snapshot
    su_soc_full_ts : pd.DataFrame
        Interpolated storage units-soc timeseries for each snapshot

    """

    stores_e_full_ts = pd.DataFrame(index=n.snapshots, columns=n.stores.index)
    stores_e_full_ts.loc[n_tsa.snapshots] = n_tsa.stores_t.e

    su_soc_full_ts = pd.DataFrame(
        index=n.snapshots, columns=n.storage_units.index
    )
    su_soc_full_ts.loc[n_tsa.snapshots] = n_tsa.storage_units_t.state_of_charge

    n_skip = int(n_tsa.snapshot_weightings.iloc[0, 0])

    for calc_sns in n_tsa.snapshots:
        next_calc = calc_sns + timedelta(hours=int(n_skip))

        if next_calc in n_tsa.snapshots:
            next_calc = next_calc
        else:
            next_calc = n_tsa.snapshots[0]
        e_now = stores_e_full_ts.loc[calc_sns]
        e_next = stores_e_full_ts.loc[next_calc]

        soc_now = su_soc_full_ts.loc[calc_sns]
        soc_next = su_soc_full_ts.loc[next_calc]

        diff_e = (e_next - e_now) / n_skip
        diff_soc = (soc_next - soc_now) / n_skip

        for i in range(n_skip - 1):
            hour = calc_sns + timedelta(hours=1 + i)
            if hour in stores_e_full_ts.index:
                stores_e_full_ts.loc[hour] = (
                    stores_e_full_ts.loc[hour - timedelta(hours=1)] + diff_e
                )
                su_soc_full_ts.loc[hour] = (
                    su_soc_full_ts.loc[hour - timedelta(hours=1)] + diff_soc
                )

    return stores_e_full_ts, su_soc_full_ts


def dispatch_disaggregation(self):
    """
    Function running the tempral disaggregation meaning the optimization
    of dispatch in the temporally fully resolved network; therfore, the problem
    is reduced to smaller subproblems by slicing the whole considered time span
    while keeping inforation on the state of charge of storage units and stores
    to ensure compatibility and to reproduce saisonality.

    Returns
    -------
    None.

    """

    if self.args["temporal_disaggregation"]["active"]:

        horizon = math.ceil(
            len(self.network_tsa.snapshots)
            / self.args["temporal_disaggregation"]["no_slices"]
        )

        n = self.network_tsa.copy()

        n_tsa = self.network.copy()

        if self.args["skip_snapshots"]:
            n_tsa.stores_t.e, n_tsa.storage_units_t.state_of_charge = (
                temp_disagg_soc(n, n_tsa)
            )

        # Avoid infeasibilies in e-Mobility stores
        e_mob_stores = n.stores[n.stores.carrier == "battery_storage"]
        max_e_first_hour = (
            n.stores_t.e_max_pu.iloc[0]
            .loc[e_mob_stores.index]
            .mul(e_mob_stores.e_nom)
        )

        index = e_mob_stores[e_mob_stores.e_initial > max_e_first_hour].index
        n.stores.loc[index, "e_initial"] = max_e_first_hour.loc[index]

        # Copy extension results
        n.lines["s_nom"] = n_tsa.lines["s_nom_opt"]
        n.lines["s_nom_extendable"] = False

        n.links["p_nom"] = n_tsa.links["p_nom_opt"]
        n.links["p_nom_extendable"] = False

        n.transformers["s_nom"] = n_tsa.transformers["s_nom_opt"]
        n.transformers.s_nom_extendable = False

        n.storage_units["p_nom"] = n_tsa.storage_units["p_nom_opt"]
        n.storage_units["p_nom_extendable"] = False

        n.stores["e_nom"] = n_tsa.stores["e_nom_opt"]
        n.stores["e_nom_extendable"] = False

        n.storage_units.cyclic_state_of_charge = False
        n.stores.e_cyclic = False

        n = optimize_with_rolling_horizon(
            n,
            n_tsa,
            snapshots=None,
            horizon=horizon,
            overlap=2,
            solver_name=self.args["solver"],
            extra_functionality=Constraints(
                self.args, False, apply_on="market_model"
            ).functionality,
            args=self.args,
            temporal_disaggregation=True,
        )

        # switch to temporally fully resolved network as standard network,
        # temporally reduced network is stored in network_tsa
        self.network_tsa = self.network.copy()
        self.network = n

        # keep original settings
        self.network.lines["s_nom_extendable"] = self.network_tsa.lines[
            "s_nom_extendable"
        ]
        self.network.links["p_nom_extendable"] = self.network_tsa.links[
            "p_nom_extendable"
        ]
        self.network.transformers.s_nom_extendable = (
            self.network_tsa.transformers.s_nom_extendable
        )
        self.network.storage_units["p_nom_extendable"] = (
            self.network_tsa.storage_units["p_nom_extendable"]
        )
        self.network.stores["e_nom_extendable"] = self.network_tsa.stores[
            "e_nom_extendable"
        ]
        self.network.storage_units.cyclic_state_of_charge = (
            self.network_tsa.storage_units.cyclic_state_of_charge
        )
        self.network.stores.e_cyclic = self.network_tsa.stores.e_cyclic

        if self.args["csv_export"]:
            path = self.args["csv_export"]
            self.export_to_csv(path)
            self.export_to_csv(path + "/temporal_disaggregaton")
