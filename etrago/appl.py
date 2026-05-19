# -*- coding: utf-8 -*-
# Copyright 2015-2026
#  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
"""
This is the application file for the tool eTraGo.
Define your connection parameters and power flow settings before executing
the function run_etrago.
"""

import datetime
import os
import os.path
import resource

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = (
    "ulfmueller, lukasol, wolfbunke, mariusves, s3pp, ClaraBuettner, "
    "CarlosEpia, KathiEsterl, fwitte, gnn, pieterhexen, AmeliaNadal"
)

if "READTHEDOCS" not in os.environ:
    # Sphinx does not run this code.
    # Do not import internal packages directly

    from etrago import Etrago
    from etrago.parameters import EtragoArgs


args = EtragoArgs(
    scn_name="eGon2035",
    start_snapshot=1,
    end_snapshot=24,
    method={"market_optimization": {"active": False}},
    network_clustering={"electricity_grid":{"n_clusters":30}},
    )

args = args.model_dump()

def run_etrago(args, config_path=None):
    """Conduct a full eTraGo optimisation run.

    Parameters
    ----------
    args : dict or EtragoArgs
        Scenario settings and optimisation parameters.
    config_path : str, optional
        Path to a JSON or YAML config file whose settings are merged
        into args. Default: None.

    Returns
    -------
    etrago : Etrago
        eTraGo object containing all network information and a PyPSA
        network <https://www.pypsa.org/doc/components.html#network>`_.
    """
    etrago = Etrago(args=args, config_path=config_path)

    # import network from database
    etrago.build_network_from_db()

    # adjust network regarding eTraGo setting
    etrago.adjust_network()

    # ehv network clustering
    etrago.ehv_clustering()

    # spatial clustering
    etrago.spatial_clustering()
    etrago.spatial_clustering_gas()

    # snapshot clustering
    etrago.snapshot_clustering()

    # skip snapshots
    etrago.skip_snapshots()

    # start linear optimal powerflow calculations
    etrago.optimize()

    # conduct lopf with full complex timeseries for dispatch disaggregation
    etrago.temporal_disaggregation()

    # start power flow based on lopf results
    etrago.pf_post_lopf()

    # spatial disaggregation
    etrago.spatial_disaggregation()

    # calculate central etrago results
    etrago.calc_results()

    return etrago


if __name__ == "__main__":
    # execute etrago function
    print(datetime.datetime.now())

    etrago = run_etrago(args)

    # RAM tracking
    self_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    total_peak = self_peak + children_peak
    print(f"Peak RAM usage:    {total_peak / 10**6:.2f} GB")

    print(datetime.datetime.now())
    etrago.session.close()
    # plots: more in tools/plot.py
    # make a line loading plot
    # etrago.plot_grid(
    # line_colors='line_loading', bus_sizes=0.0001, timesteps=range(2))
    # network and storage
    # etrago.plot_grid(
    # line_colors='expansion_abs',
    # bus_colors='storage_expansion',
    # bus_sizes=0.0001)
    # flexibility usage
    # etrago.flexibility_usage('DSM')
