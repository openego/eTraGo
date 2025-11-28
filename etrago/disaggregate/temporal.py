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
import logging
import os
import time

import pandas as pd

logger = logging.getLogger(__name__)

if "READTHEDOCS" not in os.environ:

    from etrago.execute import iterate_lopf
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
        x = time.time()

        if self.args["temporal_disaggregation"]["no_slices"]:
            # split dispatch_disaggregation into subproblems
            # keep some information on soc in beginning and end of slices
            # to ensure compatibility and to reproduce saisonality

            # define number of slices and corresponding slice length
            no_slices = self.args["temporal_disaggregation"]["no_slices"]
            slice_len = int(len(self.network.snapshots) / no_slices)

            # transition snapshots defining start and end of slices
            transits = self.network.snapshots[0::slice_len]
            if len(transits) > 1:
                transits = transits[1:]
            if transits[-1] != self.network.snapshots[-1]:
                transits = transits.insert(
                    (len(transits)), self.network.snapshots[-1]
                )
            # for stores, exclude emob and dsm because of their special
            # constraints
            sto = self.network.stores[
                ~self.network.stores.carrier.isin(
                    ["battery_storage", "battery storage", "dsm"]
                )
            ]

            # save state of charge of storage units and stores at those
            # transition snapshots
            self.conduct_dispatch_disaggregation = pd.DataFrame(
                columns=self.network.storage_units.index.append(sto.index),
                index=transits,
            )
            for storage in self.network.storage_units.index:
                self.conduct_dispatch_disaggregation[storage] = (
                    self.network.storage_units_t.state_of_charge[storage]
                )
            for store in sto.index:
                self.conduct_dispatch_disaggregation[store] = (
                    self.network.stores_t.e[store]
                )

            extra_func = self.args["extra_functionality"]
            self.args["extra_functionality"] = {}

        load_shedding = self.args["load_shedding"]
        if not load_shedding:
            self.args["load_shedding"] = True
            self.load_shedding(temporal_disaggregation=True)

        iterate_lopf(
            self,
            Constraints(
                self.args, self.conduct_dispatch_disaggregation
            ).functionality,
            method=self.args["method"],
        )

        # switch to temporally fully resolved network as standard network,
        # temporally reduced network is stored in network_tsa
        network1 = self.network.copy()
        self.network = self.network_tsa.copy()
        self.network_tsa = network1.copy()
        network1 = 0

        # keep original settings

        if self.args["temporal_disaggregation"]["no_slices"]:
            self.args["extra_functionality"] = extra_func
        self.args["load_shedding"] = load_shedding

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

        y = time.time()
        z = (y - x) / 60
        logger.info("Time for LOPF [min]: {}".format(round(z, 2)))
