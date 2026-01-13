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
import pandas as pd

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

args = {
    # Setup and Configuration:
    "db": "oep",  # database session: oep or local database
    "gridversion": None,  # None for model_draft or Version number
    "method": {  # Choose method and settings for optimization
        "type": "lopf",  # type of optimization, 'lopf' or 'sclopf'
        "n_iter": 4,  # abort criterion of iterative optimization, 'n_iter' or 'threshold'
        "formulation": "linopy",
        "market_optimization":
            {
                "active": True,
                "market_zones": "status_quo", # only used if type='market_grid'
                "rolling_horizon": {# Define parameter of market optimization
                    "planning_horizon": 168, # number of snapshots in each optimization
                    "overlap": 120, # number of overlapping hours
                 },
                "redispatch": True,
             }
    },
    "pf_post_lopf": {
        "active": False,  # choose if perform a pf after lopf
        "add_foreign_lopf": True,  # keep results of lopf for foreign DC-links
        "q_allocation": "p_nom",  # allocate reactive power via 'p_nom' or 'p'
    },
    "start_snapshot": 1,
    "end_snapshot": 168,
    "solver": "gurobi",  # glpk, cplex or gurobi
    "solver_options": {
        "BarConvTol": 1.0e-5,
        "FeasibilityTol": 1.0e-5,
        "method": 2,
        "crossover": 0,
        "logFile": "solver_etrago.log",
        "threads": 4,
        "BarHomogeneous": 1,
    },
    "model_formulation": "kirchhoff",  # angles or kirchhoff
    "scn_name": "eGon2035",  # scenario, e.g. eGon2035, eGon2035_lowflex or status2019
    # Scenario variations:
    "scn_extension": None,  # None or array of extension scenarios
    # Export options:
    "lpfile": False,  # save pyomo's lp file: False or /path/to/lpfile.lp
    "csv_export": "results",  # save results as csv: False or /path/tofolder
    # Settings:
    "extendable": {
        "extendable_components": [
            "as_in_db"
        ],  # Array of components to optimize
        "upper_bounds_grid": {  # Set upper bounds for grid expansion
            # lines in Germany
            "grid_max_D": None,  # relative to existing capacity
            "grid_max_abs_D": {  # absolute capacity per voltage level
                "380": {"i": 1020, "wires": 4, "circuits": 4},
                "220": {"i": 1020, "wires": 4, "circuits": 4},
                "110": {"i": 1020, "wires": 4, "circuits": 2},
                "dc": 0,
            },
            # border crossing lines
            "grid_max_foreign": 4,  # relative to existing capacity
            "grid_max_abs_foreign": None,  # absolute capacity per voltage level
        },
    },
    "generator_noise": 789456,  # apply generator noise, False or seed number
    "extra_functionality": {},  # Choose function name or {}
    # Spatial Complexity:
    "network_clustering_ehv": {
        "active": False,  # choose if clustering of HV buses to EHV buses is activated
        "busmap": False,  # False or path to stored busmap
    },
    "network_clustering": {
        "active": True,  # choose if clustering is activated
        "method": "kmedoids-dijkstra",  # choose clustering method: kmeans or kmedoids-dijkstra
        "n_clusters_AC": 30,  # total number of resulting AC nodes (DE+foreign)
        "cluster_foreign_AC": False,  # take foreign AC buses into account, True or False
        "method_gas": "kmedoids-dijkstra",  # choose clustering method: kmeans or kmedoids-dijkstra
        "n_clusters_gas": 15,  # total number of resulting CH4 nodes (DE+foreign)
        "n_clusters_h2": 15,  # total number of resulting H2 nodes (DE+foreign)
        "cluster_foreign_gas": False,  # take foreign CH4 buses into account, True or False
        "k_elec_busmap": False,  # False or path/to/busmap.csv
        "k_gas_busmap": False,  # False or path/to/ch4_busmap.csv
        "bus_weight_tocsv": None,  # None or path/to/bus_weight.csv
        "bus_weight_fromcsv": None,  # None or path/to/bus_weight.csv
        "gas_weight_tocsv": None,  # None or path/to/gas_bus_weight.csv
        "gas_weight_fromcsv": None,  # None or path/to/gas_bus_weight.csv
        "line_length_factor": 1,  # Factor to multiply distance between new buses for new line lengths
        "remove_stubs": False,  # remove stubs bevore kmeans clustering
        "use_reduced_coordinates": False,  # If True, do not average cluster coordinates
        "random_state": 42,  # random state for replicability of clustering results
        "n_init": 10,  # affects clustering algorithm, only change when neccesary
        "max_iter": 100,  # affects clustering algorithm, only change when neccesary
        "tol": 1e-6,  # affects clustering algorithm, only change when neccesary
        "CPU_cores": 4,  # number of cores used during clustering, "max" for all cores available.
    },
    "spatial_disaggregation": None,  # None or 'uniform'
    # Temporal Complexity:
    "snapshot_clustering": {
        "active": False,  # choose if clustering is activated
        "method": "segmentation",  # 'typical_periods' or 'segmentation'
        "extreme_periods": None,  # consideration of extreme timesteps; e.g. 'append'
        "how": "daily",  # type of period - only relevant for 'typical_periods'
        "storage_constraints": "soc_constraints",  # additional constraints for storages  - only relevant for 'typical_periods'
        "n_clusters": 5,  # number of periods - only relevant for 'typical_periods'
        "n_segments": 5,  # number of segments - only relevant for segmentation
    },
    "skip_snapshots": 5,  # False or number of snapshots to skip
    "temporal_disaggregation": {
        "active": False,  # choose if temporally full complex dispatch optimization should be conducted
        "no_slices": 8,  # number of subproblems optimization is divided into
    },
    # Simplifications:
    "branch_capacity_factor": {"HV": 0.5, "eHV": 0.7},  # p.u. branch derating
    "load_shedding": True,  # meet the demand at value of loss load cost
    "foreign_lines": {
        "carrier": "AC",  # 'DC' for modeling foreign lines as links
        "capacity": "osmTGmod",  # 'osmTGmod', 'tyndp2020', 'ntc_acer' or 'thermal_acer'
    },
    "comments": None,
}


def run_etrago(args, json_path):
    """Function to conduct optimization considering the following arguments.

    Parameters
    ----------
    db : str
        Name of Database session setting stored in *config.ini* within 
        *.etrago_database/* in case of local database,
        or  ``'test-oep'`` or ``'oedb'`` to load model from OEP.
    gridversion : None or str
        Name of the data version number of oedb: state ``'None'`` for
        model_draft (sand-box) or an explicit version number
        (e.g. 'v0.4.6') for the grid schema.
    method : dict
        Choose method and settings for optimization.
        The provided dictionary can have the following entries:

        * "type" : str
            Choose the type of optimization. Current options: "lopf", "sclopf"
            or "market_grid". Default: "market_grid".
        * "n_iter" : int
            In case of extendable lines, several LOPFs have to be performed.
            You can either set "n_iter" and specify a fixed number of
            iterations or set "threshold" and specify a threshold of the
            objective function as abort criteria of the iterative optimization.
            Default: 4.
        * "threshold" : int
            In case of extendable lines, several LOPFs have to be performed.
            You can either set "n_iter" and specify a fixed number of
            iterations or set "threshold" and specify a threshold of the
            objective function as abort criteria of the iterative optimization.
            Per default, "n_iter" of 4 is set.
        * "pyomo" : bool
            Set to True, if pyomo is used for model building.
            Set to False for big optimization problems - currently only
            possible when solver is "gurobi".

    pf_post_lopf : dict
        Settings for option to run a non-linear power flow (PF) directly after
        the linear optimal power flow (LOPF), and thus the dispatch
        optimisation, has finished.
        The provided dictionary can have the following entries:

        * "active" : bool
            If True, a PF is performed after the LOPF. Default: True.
        * "add_foreign_lopf" : bool
            If foreign lines are modeled as DC-links (see parameter
            `foreign_lines`), results of the LOPF can be added by setting
            "add_foreign_lopf" to True. Default: True.
        * "q_allocation" : bool
            Allocate reactive power to all generators at the same bus either
            by "p_nom" or "p".
            Default: "p_nom".

    start_snapshot : int
        Start hour of the scenario year to be calculated. Default: 1.
    end_snapshot : int
        End hour of the scenario year to be calculated. If snapshot clustering
        is used (see parameter `snapshot_clustering`), the selected snapshots
        should cover the number of periods / segments. Default: 2.
    solver : str
        Choose your preferred solver. Current options: "glpk" (open-source),
        "cplex" or "gurobi". Default: "gurobi".
    solver_options : dict
        Choose settings of solver to improve simulation time and result.
        Options are described in documentation of chosen solver. Per default,
        the following dictionary is set:

        {
            "BarConvTol": 1.0e-5,
            "FeasibilityTol": 1.0e-5,
            "method": 2,
            "crossover": 0,
            "logFile": "solver_etrago.log",
            "threads": 4,
        }

        Make sure to reset or adapt these settings when using another solver!
        Otherwise, you may run into errors.
    model_formulation : str
        Choose formulation of pyomo-model.
        Current options are: "angles", "cycles", "kirchhoff", "ptdf".
        "angels" works best for small networks, while "kirchhoff" works best
        for larger networks.
        Default: "kirchhoff".
    scn_name : str
         Choose your scenario. For an overview of available scenarios, see the 
         documentation on Read the Docs.
    scn_extension : None or list of str

        Choose extension-scenarios which will be added to the existing
        network container. In case new lines replace existing ones, these are
        dropped from the network. Data of the extension scenarios is located in
        extension-tables (e.g. grid.egon_etrago_extension_line)
        There are two overlay networks:

        * 'nep2021_confirmed' includes all planed new lines confirmed by the
          Bundesnetzagentur included in the NEP version 2021
        * 'nep2021_c2035' includes all new lines planned by the
          Netzentwicklungsplan 2021 in scenario 2035 C
        Default: None.
    lpfile : bool or str
        State if and where you want to save pyomo's lp file. Options:
        False or '/path/tofile.lp'. Default: False.
    csv_export : bool or str
        State if and where you want to save results as csv files. Options:
        False or '/path/tofolder'. Default: False.

    extendable : dict
        Choose components you want to optimize and set upper bounds for grid
        expansion. The provided dictionary can have the following entries:

        * "extendable_components" : list(str)
            The list defines a set of components to optimize.
            Settings can be added in /tools/extendable.py.
            The most important possibilities:

            * 'as_in_db'
                leaves everything as it is defined in the data coming from the
                database
            * 'network'
                set all lines, links and transformers in electrical grid
                extendable
            * 'german_network'
                set lines and transformers in German electrical grid extendable
            * 'foreign_network'
                set foreign lines and transformers in electrical grid
                extendable
            * 'transformers'
                set all transformers extendable
            * 'storages' / 'stores'
                allow to install extendable storages (unlimited in size) at
                each grid node in order to meet the flexibility demand

            Default: "as_in_db".

        * "upper_bounds_grid" : dict
            Dictionary can have the following entries:

            * 'grid_max_D'
                Upper bounds for electrical grid expansion can be defined for
                lines in Germany relative to the existing capacity.
                Alternatively, 'grid_max_abs_D' can be used. Per default, this
                is set to None and 'grid_max_abs_D' is set.

            * 'grid_max_abs_D'
                Upper bounds for electrical grid expansion can be defined for
                lines in Germany as absolute maximum capacities between two
                electrical buses per voltage level. Per default the following
                dictionary is set:

                {
                    "380": {"i": 1020, "wires": 4, "circuits": 4},
                    "220": {"i": 1020, "wires": 4, "circuits": 4},
                    "110": {"i": 1020, "wires": 4, "circuits": 2},
                    "dc": 0,
                }
            * 'grid_max_foreign'
                Upper bounds for border-crossing electrical lines can be
                defined relative to the existing capacity. Alternatively,
                'grid_max_abs_foreign' can be set.
                Default: 4.
            * 'grid_max_abs_foreign'
                Upper bounds for border-crossing electrical lines can be
                defined equally to 'grid_max_abs_D' as absolute capacity per
                voltage level.
                Default: None.

    generator_noise : bool or int
        State if you want to apply a small random noise to the marginal costs
        of each generator in order to prevent an optima plateau. To reproduce
        a noise, choose the same integer (seed number). Default: 789456.
    extra_functionality : dict or None
        Choose extra functionalities and their parameters.
        Settings can be added in /tools/constraints.py.
        Current options are:

        * 'max_line_ext' : float
            Maximal share of network extension in p.u.
        * 'min_renewable_share' : float
            Minimal share of renewable generation in p.u.
        * 'cross_border_flow' : array of two floats
            Limit AC cross-border-flows between Germany and its neighbouring
            countries. Set values in MWh for all snapshots, e.g. [-x, y]
            (with x Import, y Export, positive: export from Germany).
        * 'cross_border_flows_per_country' : dict of cntr and array of floats
            Limit AC cross-border-flows between Germany and its neighbouring
            countries. Set values in MWh for each country, e.g. [-x, y]
            (with x Import, y Export, positive: export from Germany).
        * 'capacity_factor' : dict of arrays
            Limit overall energy production for each carrier.
            Set upper/lower limit in p.u.
        * 'capacity_factor_per_gen' : dict of arrays
            Limit overall energy production for each generator by carrier.
            Set upper/lower limit in p.u.
        * 'capacity_factor_per_cntr': dict of dict of arrays
            Limit overall energy production country-wise for each carrier.
            Set upper/lower limit in p.u.
        * 'capacity_factor_per_gen_cntr': dict of dict of arrays
            Limit overall energy production country-wise for each generator
            by carrier. Set upper/lower limit in p.u.

    network_clustering_ehv : dict
        Choose if you want to apply an extra high voltage clustering to the
        electrical network.
        The provided dictionary can have the following entries:

        * "active" : bool
        Choose if you want to cluster the full HV/EHV dataset down to only the
        EHV buses. In that case, all HV buses are assigned to their closest EHV
        substation, taking into account the shortest distance on power lines.
        Default: False.
        * "busmap" : str
        Choose if an stored busmap can be used to make the process quicker, or
        a new busmap must be calculated. False or path to the busmap in csv
        format should be given.
        Default: False

    network_clustering : dict
        Choose if you want to apply a clustering of all network buses and
        specify settings.
        The provided dictionary can have the following entries:

        * "active" : bool
            If True, the AC buses are clustered down to ``'n_clusters_AC'``
            and the gas buses are clustered down to``'n_clusters_gas'``.
            Default: True.
        * "method" : str
            Method used for AC clustering. You can choose between two
            clustering methods:
            * "kmeans": considers geographical locations of buses
            * "kmedoids-dijkstra":  considers electrical distances between
            buses

            Default: "kmedoids-dijkstra".
        * "n_clusters_AC" : int
            Defines total number of resulting AC nodes including DE and foreign
            nodes if `cluster_foreign_AC` is set to True, otherwise only DE
            nodes.
            Default: 30.
        * "cluster_foreign_AC" : bool
            If set to False, the AC buses outside Germany are not clustered
            and the buses inside Germany are clustered to complete
            ``'n_clusters_AC'``. If set to True, foreign AC buses are clustered
            as well and included in number of clusters specified through
            ``'n_clusters_AC'``.
            Default: False.
        * "method_gas" : str
            Method used for gas clustering. You can choose between two
            clustering methods:
            * "kmeans": considers geographical locations of buses
            * "kmedoids-dijkstra":  considers 'electrical' distances between
            buses

            Default: "kmedoids-dijkstra".
        * "n_clusters_gas" : int
            Defines total number of resulting CH4 nodes including DE and
            foreign nodes if `cluster_foreign_gas` is set to True, otherwise
            only DE nodes.
            Default: 14.
        * "n_clusters_h2" : int
            Defines total number of resulting H2 nodes including DE and
            foreign nodes if `cluster_foreign_gas` is set to True, otherwise
            only DE nodes.
            Default: 14.
        * "cluster_foreign_gas" : bool
            If set to False, the gas buses outside Germany are not clustered
            and the buses inside Germany are clustered to complete
            ``'n_clusters_gas'``. If set to True, foreign gas buses are
            clustered as well and included in number of clusters specified
            through ``'n_clusters_gas'``.
            Default: False.
        * "k_elec_busmap" : bool or str
            With this option you can load cluster coordinates from a previous
            AC clustering run. Options are False, in which case no previous
            busmap is loaded, and path/to/busmap.csv in which case the busmap
            is loaded from the specified file. Please note, that when a path is
            provided, the set number of clusters will be ignored.
            Default: False.
        * "k_gas_busmap" : bool or str
            With this option you can load cluster coordinates from a previous
            gas clustering run. Options are False, in which case no previous
            busmap is loaded, and path/to/busmap.csv in which case the busmap
            is loaded from the specified file. Please note, that when a path is
            provided, the set number of clusters will be ignored.
            Default: False.
        * "bus_weight_fromcsv" : None or str
            In general, the weighting of AC buses takes place considering
            generation and load at each node. With this option, you can load an
            own weighting for the AC buses by providing a path to a csv file.
            If None, weighting is conducted as described above.
            Default: None.
        * "bus_weight_tocsv" : None or str
            Specifies whether to store the weighting of AC buses to csv or not.
            If None, it is not stored. Otherwise, it is stored to the provided
            path/to/bus_weight.csv.
            Default: None.
        * "gas_weight_fromcsv" : None or str
            In general, the weighting of CH4 nodes takes place considering
            generation and load at each node, as well as non-transport
            capacities at each node. With this option, you can load an own
            weighting for the CH4 buses by providing a path to a csv file. If
            None, weighting is conducted as described above.
            Default: None.
        * "gas_weight_tocsv" : None or str
            Specifies whether to store the weighting of gas buses to csv or
            not. If None, it is not stored. Otherwise, it is stored to the
            provided path/to/gas_bus_weight.csv.
            Default: None.
        * "line_length_factor" : float
            Defines the factor to multiply the crow-flies distance
            between new buses by, in order to get new line lengths.
            Default: 1.
        * "remove_stubs" : bool
            If True, remove stubs before k-means clustering, which reduces the
            overestimating of line meshes.
            This option is only used within the k-means clustering.
            Default: False.
        * "use_reduced_coordinates" : bool
            If True, do not average cluster coordinates, but take from busmap.
            This option is only used within the k-means clustering.
            Default: False.
        * "random_state" : int
            Random state for replicability of clustering results. Default: 42.
        * "n_init" : int
            Affects clustering algorithm, only change when necessary!
            Documentation and possible settings are described in
            sklearn-package (sklearn/cluster/kmeans.py).
            Default: 10.
        * "max_iter" : int
            Affects clustering algorithm, only change when necessary!
            Documentation and possible settings are described in
            sklearn-package (sklearn/cluster/kmeans.py).
            Default: 100.
        * "tol" : float
            Affects clustering algorithm, only change when necessary!
            Documentation and possible settings are described in
            sklearn-package (sklearn/cluster/kmeans.py).
            Default: 1e-6.
        * "CPU_cores" : int or str
            Number of cores used in clustering. Specify a concrete number or
            "max" to use all cores available.
            Default: 4.

    disaggregation : None or str
        Specify None, in order to not perform a spatial disaggregation, or the
        method you want to use for the spatial disaggregation. Only possible
        option is currently "uniform".
    snapshot_clustering : dict
        State if you want to apply a temporal clustering and run the
        optimization only on a subset of snapshot periods, and specify
        settings. The provided dictionary can have the following entries:

        * "active" : bool
            Choose, if clustering is activated or not. If True, it is
            activated.
            Default: False.
        * "method" : str
            Method to apply. Possible options are "typical_periods" and
            "segmentation".
            Default: "segmentation".
        * "extreme_periods" : None or str
            Method used to consider extreme snapshots (time steps with extreme
            residual load) in reduced timeseries.
            Possible options are None, "append", "new_cluster_center", and
            "replace_cluster_center". The default is None, in which case
            extreme periods are not considered.
        * "how" : str
            Definition of period in case `method` is set to "typical_periods".
            Possible options are "daily", "weekly", and "monthly".
            Default: "daily".
        * "storage_constraints" : str
            Defines additional constraints for storage units in case `method`
            is set to "typical_periods". Possible options are "daily_bounds",
            "soc_constraints" and "soc_constraints_simplified".
            Default: "soc_constraints".
        * "n_clusters" : int
            Number of clusters in case `method` is set to "typical_periods".
            Default: 5.
        * "n_segments" : int
            Number of segments in case `method` is set to "segmentation".
            Default: 5.

    skip_snapshots : bool or int
        State None, if you want to use all time steps, or provide a number,
        if you only want to consider every n-th timestep to reduce
        temporal complexity. Default: 5.
    temporal_disaggregation : dict
        State if you want to apply a second LOPF considering dispatch only
        (no capacity optimization) to disaggregate the dispatch to the whole
        temporal complexity. Be aware that a load shedding will be applied in
        this optimization. The provided dictionary must have the following
        entries:

        * "active" : bool
            Choose, if temporal disaggregation is activated or not. If True,
            it is activated.
            Default: False.
        * "no_slices" : int
            With "no_slices" the optimization problem will be calculated as a
            given number of sub-problems while using some information on the
            state of charge of storage units and stores from the former
            optimization (at the moment only possible with skip_snapshots and
            extra_functionalities are disregarded).
            Default: 8.

    branch_capacity_factor : dict[str, float]
        Add a factor here if you want to globally change line capacities
        (e.g. to "consider" an (n-1) criterion or for debugging purposes).
        The factor specifies the p.u. branch rating, e.g. 0.5 to allow half the
        line capacity. Per default, it is set to {'HV': 0.5, 'eHV' : 0.7}.
    load_shedding : bool
        State here if you want to make use of the load shedding function which
        is helpful when debugging: a very expensive generator is set to each
        bus and meets the demand when regular generators cannot do so.
        Default: False.
    foreign_lines : dict
        Choose transmission technology and capacity of foreign lines:

        * 'carrier': 'AC' or 'DC'
        * 'capacity': 'osmTGmod', 'tyndp2020', 'ntc_acer' or 'thermal_acer'

        Per default, it is set to {'carrier':'AC', 'capacity': 'osmTGmod'}.

    comments : str
        Can be any comment you wish to make.

    Returns
    -------
    etrago : etrago object
        eTraGo containing all network information and a PyPSA network
        <https://www.pypsa.org/doc/components.html#network>`_

    """
    etrago = Etrago(args, json_path=json_path)

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
    etrago = run_etrago(args, json_path=None)

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
