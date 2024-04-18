# -*- coding: utf-8 -*-
# Copyright 2016-2023  Flensburg University of Applied Sciences,
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
    "db": "egon-data",  # database session
    "gridversion": None,  # None for model_draft or Version number
    "method": {  # Choose method and settings for optimization
        "type": "lopf",  # type of optimization, currently only 'lopf'
        "n_iter": 4,  # abort criterion of iterative optimization, 'n_iter' or 'threshold'
        "pyomo": True,  # set if pyomo is used for model building
    },
    "pf_post_lopf": {
        "active": True,  # choose if perform a pf after lopf
        "add_foreign_lopf": True,  # keep results of lopf for foreign DC-links
        "q_allocation": "p_nom",  # allocate reactive power via 'p_nom' or 'p'
    },
    "start_snapshot": 1,
    "end_snapshot": 10,
    "solver": "gurobi",  # glpk, cplex or gurobi
    "solver_options": {
        "BarConvTol": 1.0e-5,
        "FeasibilityTol": 1.0e-5,
        "method": 2,
        "crossover": 0,
        "logFile": "solver_etrago.log",
        "threads": 4,
    },
    "model_formulation": "kirchhoff",  # angles or kirchhoff
    "scn_name": "eGon2035",  # scenario: eGon2035, eGon100RE or status2019
    # Scenario variations:
    "scn_extension": None,  # None or array of extension scenarios
    "scn_decommissioning": None,  # None or decommissioning scenario
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
    "delete_dispensable_ac_buses": True, # bool. Find and delete expendable buses
    "network_clustering_ehv": {
        "active": False,  # choose if clustering of HV buses to EHV buses is activated
        "busmap": False, # False or path to stored busmap
    },
    "network_clustering": {
        "active": True,  # choose if clustering is activated
        "method": "kmedoids-dijkstra",  # choose clustering method: kmeans or kmedoids-dijkstra
        "n_clusters_AC": 30,  # total number of resulting AC nodes (DE+foreign)
        "cluster_foreign_AC": False,  # take foreign AC buses into account, True or False
      
       "interest_area": ["Nordfriesland","Flensburg","Kiel","Lübeck","Neumünster","Dithmarschen","Herzogtum Lauenburg",
                          "Nordfriesland","Ostholstein","Pinneberg","Plön","Rendsburg-Eckernförde","Schleswig-Flensburg"
                          "Segeberg","Steinburg","Stormarn"],  # False, path to shapefile or list of nuts names of not cluster area
        "cluster_interest_area": False, # False or number of buses.
        "method_gas": "kmedoids-dijkstra",  # choose clustering method: kmeans or kmedoids-dijkstra
        "n_clusters_gas": 17,  # total number of resulting CH4 nodes (DE+foreign)
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
    "sector_coupled_clustering": {
        "active": True,  # choose if clustering is activated
        "carrier_data": {  # select carriers affected by sector coupling
            "central_heat": {
                "base": ["CH4", "AC"],
                "strategy": "simultaneous",  # select strategy to cluster other sectors
            },
        },
    },
    "disaggregation": None,  # None or 'uniform'
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
        Name of Database session setting stored in *config.ini* of *.egoio*,
        e.g. ``'oedb'``.
    gridversion : None or str
        Name of the data version number of oedb: state ``'None'`` for
        model_draft (sand-box) or an explicit version number
        (e.g. 'v0.4.6') for the grid schema.
    method : dict
        Choose method and settings for optimization.
        The provided dictionary can have the following entries:

        * "lopf" : str
            Type of optimization, currently only "lopf". Default: "lopf".
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
         Choose your scenario. Currently, there are two different
         scenarios: "eGon2035", "eGon100RE". Default: "eGon2035".
    scn_extension : None or list of str

        Choose extension-scenarios which will be added to the existing
        network container. Data of the extension scenarios are located in
        extension-tables (e.g. grid.egon_etrago_extension_line)
        There are two overlay networks:

        * 'nep2021_confirmed' includes all planed new lines confirmed by the
          Bundesnetzagentur included in the NEP version 2021
        * 'nep2021_c2035' includes all new lines planned by the
          Netzentwicklungsplan 2021 in scenario 2035 C
        Default: None.
    scn_decommissioning : NoneType or str
        This option does currently not work!

        Choose an extra scenario which includes lines you want to decommission
        from the existing network. Data of the decommissioning scenarios are
        located in extension-tables
        (e.g. model_draft.ego_grid_pf_hv_extension_bus) with the prefix
        'decommissioning\_'.
        Currently, there are two decommissioning_scenarios which are linked to
        extension-scenarios:

        * 'nep2035_confirmed' includes all lines that will be replaced in
          confirmed projects
        * 'nep2035_b2' includes all lines that will be replaced in
          NEP-scenario 2035 B2

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

    delete_dispensable_ac_buses: bool
        Choose if electrical buses that are only connecting two lines should be
        removed. These buses have no other components attached to them. The
        connected lines are merged. This reduces the spatial complexity without
        losing any accuracy.
        Default: True.
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
        * "interest_area": False, list, string
            Area of especial interest that will be not clustered. It is by
            default set to false. When an exclusion area is provided, the given
            value for n_clusters_AC will mean the total of AC buses outside the
            area.The areas can be provided in two ways: list of
            nuts names e.G. ["Cuxhaven", "Bremerhaven", "Bremen"] or a string
            with a path to a shape file.
            Default: False.
        * "cluster_interest_area": False, int
            Number of buses to cluster all the electrical buses in the
            exclusion area. Method provided in the arg "method" is used.
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
            Default: 17.
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

    sector_coupled_clustering : dict
        Choose if you want to apply a clustering of sector coupled carriers,
        such as central_heat, and specify settings.
        The provided dictionary can have the following entries:

        * "active" : bool
            State if you want to apply clustering of sector coupled carriers,
            such as central_heat.
            Default: True.
        * "carrier_data" : dict[str, dict]
            Keys of the dictionary specify carriers affected by sector
            coupling, e.g. "central_heat". The corresponding dictionaries
            specify, how the carrier should be clustered. This dictionary must
            contain the following entries:

            * "base" : list(str)
                The approach bases on already clustered buses (AC and CH4) and
                builds clusters around the topology of those buses. With this
                option, you can specify the carriers to use as base. See
                `strategy` for more information.
            * "strategy" :  str
                Strategy to use in the clustering. Possible options are:

                * "consecutive"
                    This strategy clusters around the buses of the first
                    carrier in the `'base'`` list. The links to other buses are
                    preserved. All buses, that have no connection to the first
                    carrier will then be clustered around the buses of the
                    second carrier in the list.
                * "simultaneous"
                    This strategy looks for links connecting the buses of the
                    carriers in the ``'base'`` list and aggregates buses in
                    case they have the same set of links connected. For
                    example, a heat bus connected to CH4 via gas boiler and to
                    AC via heat pump will only form a cluster with other buses,
                    if these have the same links to the same clusters of CH4
                    and AC.

            Per default, the following dictionary is set:
            {
                "central_heat": {
                    "base": ["CH4", "AC"],
                    "strategy": "simultaneous",
                },
            }

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

    #add Energy Community
    etrago.add_energy_community_to_network

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
    etrago.lopf()

    # conduct lopf with full complex timeseries for dispatch disaggregation
    etrago.dispatch_disaggregation()

    # start power flow based on lopf results
    etrago.pf_post_lopf()

    # spatial disaggregation
    # needs to be adjusted for new sectors
    etrago.disaggregation()

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
    etrago.plot_grid(
    line_colors='line_loading', bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,
    boundaries=None,
    filename=None,
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10})
    # Example boundaries - you need to adjust these to Schleswig-Holstein's actual boundaries
schleswig_holstein_boundaries = [8.5, 11.5, 53.5, 55]

etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,
    boundaries=schleswig_holstein_boundaries,
    filename=None,
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
)

# Example osm settings - adjust 'x' and 'y' to Schleswig-Holstein's boundaries
# Example osm settings - adjust 'x' and 'y' to Schleswig-Holstein's boundaries
# Define OSM settings to attempt zooming via function parameters

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Create a GeoAxesSubplot with a specific projection
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Define the boundaries (e.g., Schleswig-Holstein)
boundaries = [8.5, 11.5, 53.5, 55]

# Set the extent of the map to the boundaries
ax.set_extent(boundaries, crs=ccrs.PlateCarree())

# Call the customized plot_grid function with the pre-configured axis
etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,  # Assume OSM is not needed
    boundaries=boundaries,
    filename='etrago_plot_schleswig_holstein.png',
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
    ax=ax  # Pass the pre-configured GeoAxesSubplot
)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Create a GeoAxesSubplot with a specific projection
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Define the boundaries
boundaries = [8.5, 11.5, 53.5, 55]

# Set the extent of the map to the boundaries
ax.set_extent(boundaries, crs=ccrs.PlateCarree())

# Test: plot a marker at a known location within the boundaries
ax.plot(9.8, 54.8, marker='o', color='red', transform=ccrs.Geodetic())

# Assuming eTraGo has a method to get coordinates of elements
# Plot all network buses to see if they fall within the view
x = [bus.x for bus in etrago.network.buses.itertuples()]
y = [bus.y for bus in etrago.network.buses.itertuples()]
ax.scatter(x, y, color='blue', transform=ccrs.Geodetic(), label='Buses')

plt.legend()
plt.show()




# Example osm settings - adjust 'x' and 'y' to Schleswig-Holstein's boundaries
osm_settings = {'x': [8.5, 11.5], 'y': [53.5, 55], 'zoom': 6}


etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=osm_settings,
    boundaries=None,
    filename='etrago_plot_schleswig_holstein.png',
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
)

# Approximate boundaries for Schleswig-Holstein
# You might need to adjust these values based on more precise boundaries
import matplotlib.pyplot as plt

# Your existing code to plot
boundaries = [8.5, 11.5, 53.5, 55]  # [x1, x2, y1, y2]

etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,
    boundaries=boundaries,
    filename=None,
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
)


import matplotlib.pyplot as plt

etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,  # Not using OSM background
    boundaries=None,
    filename='output.png',  # Save the output to a file
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
)

plt.show()  # Display the plot



# Define the boundaries for Schleswig-Holstein
# You might need to adjust these values to better fit the region of interest
import matplotlib.pyplot as plt

# Call the plot function without specifying boundaries


# Define the boundaries for Schleswig-Holstein
# You might need to adjust these values to better fit the region of interest
boundaries = [8.0, 10.0, 53.0, 54]  # [x1, x2, y1, y2]

etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,  # Not using OSM background
    boundaries=False,  # Focus on Schleswig-Holstein
    filename='SH.png',  # Save the output to a file
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
)

plt.show()  # Display the plot

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Define the projection
projection = ccrs.PlateCarree()

# Create a GeoAxesSubplot object with the defined projection
fig, ax = plt.subplots(figsize=(10, 8), dpi=300, subplot_kw={'projection': projection})

# Define the boundaries for Schleswig-Holstein
boundaries = [8.5, 11.5, 53.5, 55]  # [x_min, x_max, y_min, y_max]

# Set the extent of the map to the boundaries
ax.set_extent(boundaries, crs=projection)

# Now call the plot_grid function without the osm and boundaries arguments
etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,  # Not using OSM since we are manually setting the boundaries
    boundaries=boundaries,  # Pass the adjusted boundaries
    filename='etrago_plot_schleswig_holstein.png',
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
)

plt.show()  # Display the plot


# Display the plot
plt.show()

# Save the figure
fig.savefig('etrago_plot_schleswig_holstein.png', dpi=300)


import matplotlib.pyplot as plt

# Generate the plot without specifying boundaries
etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,
    boundaries=None,
    filename=None,
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
)

# Retrieve the current axis
ax = plt.gca()

# Set the x and y limits to zoom in on the desired area
# Adjust these values based on the geographical location you want to focus on
ax.set_xlim([8.5, 11.5])
ax.set_ylim([53.5, 55])

# Display the plot
plt.show()

# If you want to save the figure, use savefig after setting the limits
plt.savefig('etrago_plot_schleswig_holstein_zoomed.png', dpi=300)

# Since the plot is directly generated by the function, we don't use plt.show() here
# The output should be saved to 'etrago_plot_schleswig_holstein.png'




# After the plot is created, save the current figure with a higher DPI
plt.savefig("high_res_plot.png", dpi=300)  # Adjust the filename and DPI as needed
plt.show()  # Show the plot



    # network and storage
    #etrago.plot_grid(
    #line_colors='expansion_abs',
   # bus_colors='storage_expansion',
   # bus_sizes=0.0001)
    # flexibility usage
   # etrago.flexibility_usage('DSM')
# Define OSM settings to attempt zooming via function parameters
osm_settings = {'x': [8.5, 11.5], 'y': [53.5, 55], 'zoom': 6}

# Call the etrago.plot_grid function with adjusted settings for Schleswig-Holstein
etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=osm_settings,  # Assuming this sets background and zoom level
    boundaries=None,  # Ensure this is the correct way to use this parameter
    filename='etrago_plot_schleswig_holstein.png',  # This will save the plot
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
)

plt.savefig("high_res_plot.png", dpi=300) 

# Create a GeoAxesSubplot with the PlateCarree projection
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Now, instead of calling the `etrago.plot_grid()` directly, you will need to modify the function to accept the `ax` parameter or use the `ax` object inside the function for plotting.
# If `plot_grid` is your own custom function, add `ax` as an argument and replace plt with ax in your function definition.

# Assuming `plot_grid` can now accept `ax` as an argument:
etrago.plot_grid(
    line_colors='expansion_abs',
    bus_sizes=0.001,
    bus_colors="grey",
    timesteps=range(2),
    osm=False,  # Not using OSM since we are using Cartopy
    boundaries=boundaries,  # Define your boundaries if needed
    filename='etrago_plot_schleswig_holstein.png',  # This will save the plot
    disaggregated=False,
    ext_min=0.1,
    ext_width=False,
    legend_entries="all",
    scaling_store_expansion={
        "H2": 50,
        "heat": 0.1,
        "battery": 10,
    },
    ax=ax  # Pass the GeoAxesSubplot to the function
)
plt.show()  # Display the plot if not saving to file


import folium
import geopandas as gpd
from shapely.geometry import Point

# Assuming 'network' is your PyPSA Network object and it has geographical coordinates
# Convert buses and lines to GeoDataFrames
gdf_buses = gpd.GeoDataFrame(etrago.network.buses, geometry=gpd.points_from_xy(etrago.network.buses.x, etrago.network.buses.y))
gdf_lines = gpd.GeoDataFrame(etrago.network.lines)

# Create a map centered around the mean coordinates of the buses
m = folium.Map(location=[gdf_buses.geometry.y.mean(), gdf_buses.geometry.x.mean()], zoom_start=12)

# Add buses to the map
for idx, row in gdf_buses.iterrows():
    folium.CircleMarker(location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        popup=row.name,
                        color='blue',
                        fill=True).add_to(m)

# Assuming line geometries can be derived (simplified example)
for idx, row in gdf_lines.iterrows():
    folium.PolyLine(locations=[(etrago.network.buses.loc[row.bus0].y, etrago.network.buses.loc[row.bus0].x),
                               (etrago.network.buses.loc[row.bus1].y, etrago.network.buses.loc[row.bus1].x)],
                    color='green').add_to(m)

# Save to HTML
m.save('network_map_2.html')

# Assuming you have run the function to add components to the network
# data_manipulation_sh(network)

# Convert bus index to list of strings
bus_ids = etrago.network.buses.index.tolist()

# Convert list of strings to integers and find the max
new_bus_id = max(int(bus_id) for bus_id in bus_ids if bus_id.isdigit())

# Check if the new bus ID is in the network bus dataframe index
if str(new_bus_id) in etrago.network.buses.index:
    print(f"New bus with ID {new_bus_id} has been added to the network.")
    # Print out the details of the new bus
    print(etrago.network.buses.loc[str(new_bus_id)])
else:
    print(f"New bus with ID {new_bus_id} was not added to the network.")

# Alternatively, you can simply print out the last few rows of the bus dataframe
print(etrago.network.buses.tail())


bus_ids = etrago.network.buses.index.tolist()

# Convert list of strings to integers where possible and find the max
new_bus_id = max(int(bus_id) for bus_id in bus_ids if bus_id.isdigit())

# Check if the new bus ID is in the network bus dataframe index
if str(new_bus_id) in etrago.network.buses.index:
    print(f"New bus with ID {new_bus_id} has been added to the network.")
    
    # Print out the details of the new bus
    print("Details of the new bus:")
    print(etrago.network.buses.loc[str(new_bus_id)])
    
    # Find generators associated with this new bus
    associated_generators = etrago.network.generators[etrago.network.generators.bus == str(new_bus_id)]
    if not associated_generators.empty:
        print("\nGenerators associated with this new bus:")
        print(associated_generators)
    else:
        print("No generators are associated with this new bus.")
else:
    print(f"New bus with ID {new_bus_id} was not added to the network.")

# Optionally, print out the last few rows of the bus dataframe
print("\nLast few rows of the buses dataframe:")
print(etrago.network.buses.tail())

import numpy as np
Add_EC_to_network(etrago.network)