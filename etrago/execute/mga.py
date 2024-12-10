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

if "READTHEDOCS" not in os.environ:
    import logging

    import pandas as pd

    from etrago.tools.constraints import Constraints
    from linopy import LinearExpression, QuadraticExpression, merge

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ClaraBuettner"


def mga_electrolysis_expansion(
    etrago,
    snapshots=None,
    sense="max",
    slack=0.05,
    model_kwargs={},
    **kwargs,
):
    """
    Run modelling-to-generate-alternatives (MGA) on network to find near-
    optimal solutions.

    Parameters
    ----------
    n : pypsa.Network snapshots : list-like
        Set of snapshots to consider in the optimization. The default is None.
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimize in
        multiple investment periods. Then, snapshots should be a
        ``pd.MultiIndex``.
    sense : str|int
        Optimization sense of alternate objective function. Defaults to 'min'.
        Can also be 'max'.
    slack : float
        Cost slack for budget constraint. Defaults to 0.05.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or
        `chunk`.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

    Returns
    -------
    None
    """
    
    n = etrago.network

    if snapshots is None:
        snapshots = n.snapshots

    # check that network has been solved
    assert hasattr(
        n, "objective"
    ), "Network needs to be solved with `n.optimize()` before running MGA."


    n.links.marginal_cost_quadratic = 0.0
    n.links.loc[n.links.carrier=="power_to_H2", "p_nom_max"] = 25000
    # create basic model
    m = n.optimize.create_model(
        snapshots=snapshots,
    )
    
    # Add extra functionalities
    Constraints(etrago.args, conduct_dispatch_disaggregation=False).functionality(n, n.snapshots)

    # build budget constraint
    # Wenn Komponenten nur CAPEX haben, kommt es hier zu fehlern!
    #optimal_cost = (n.statistics.capex() + n.statistics.opex()).sum()
    optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
    fixed_cost = 0 #n.statistics.installed_capex().sum()
    objective = m.objective
    if not isinstance(objective, (LinearExpression, QuadraticExpression)):
        objective = objective.expression
        
    m.add_constraints(
        objective + fixed_cost <= (1 + slack) * optimal_cost, name="budget"
    )

    # parse optimization sense
    if (
        isinstance(sense, str)
        and sense.startswith("min")
        or isinstance(sense, int)
        and sense > 0
    ):
        sense = 1
    elif (
        isinstance(sense, str)
        and sense.startswith("max")
        or isinstance(sense, int)
        and sense < 0
    ):
        sense = -1
    else:
        raise ValueError(f"Could not parse optimization sense {sense}")

    # build alternate objective
    objective = []
    
    objective.append(m["Link-p_nom"].loc[n.links[n.links.carrier=="power_to_H2"].index] * 1 * sense)
    
    m.objective = merge(objective)

    n.optimize.solve_model(
        solver_name = etrago.args["solver"],
        solver_options = etrago.args["solver_options"],
)

    # write MGA coefficients into metadata
    n.meta["slack"] = slack
    n.meta["sense"] = sense

    def convert_to_dict(obj):
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        else:
            return obj
