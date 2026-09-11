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

from pathlib import Path

from scenario_config import (
    apply_network_price_scenario,
    load_and_apply_config,
    scenario_summary,
    write_resolved_config,
)

CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")

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
    # Do not import internal packages directly.
    from etrago import Etrago

    from etrago.tools.utilities import (
        restore_load_shedding_after_clustering,
    )

    from etrago.tools.swfl_real_system import (
        apply_swfl_real_system,
        purge_legacy_swfl_heat_pumps,
        remove_known_legacy_swfl_heat_pump_before_clustering,
    )

    from etrago.tools.biogas_sh import (
        apply_biogas_sh_assets,
        apply_biogas_sh_transport_route,
        validate_biogas_sh_storage_topology,
    )

from etrago.tools.import_data import (
    biogas_sh_csv,
    fix_custom_component_scn_names,
    get_data_paths,
)


import warnings
import logging

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed.*",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="linopy.expressions",
)

# ---------------------------------------------------------------------------
# Input/output paths
# ---------------------------------------------------------------------------

DATA_PATHS = get_data_paths(validate=False)

ETRAGO_DATA_DIR = DATA_PATHS.ETRAGO_DATA_DIR

BIOGAS_SH_DIR = DATA_PATHS.BIOGAS_SH_DIR

# Set to None for the default CSV.
# Set to a filename in data/biogas-sh/ for sensitivities/debug runs.
BIOGAS_SH_CSV_NAME = "biogas_sh_force_grid_injection_debug.csv"
# BIOGAS_SH_CSV_NAME = None

BIOGAS_SH_CSV = biogas_sh_csv(BIOGAS_SH_CSV_NAME)

BIOGAS_SH_MAPPED_CSV = DATA_PATHS.BIOGAS_SH_MAPPED_CSV
DING0_MV_GPKG = DATA_PATHS.DING0_MV_GPKG
BIOGAS_SH_FOCUS_REGION = DATA_PATHS.BIOGAS_SH_FOCUS_REGION

SWFL_DIR = DATA_PATHS.SWFL_DIR
SWFL_HEAT_CSV = DATA_PATHS.SWFL_HEAT_CSV

DEBUG_LOG_PATH = DATA_PATHS.DEBUG_LOG_PATH

args = {
    # Setup and Configuration:
    "db": "local_egon2035",  # database session: oep or local database
    "gridversion": None,  # None for model_draft or version number

    "method": {  # choose method and settings for optimization
        "type": "lopf",  # type of optimization, 'lopf' or 'sclopf'
        "n_iter": 4,  # abort criterion of iterative optimization, 'n_iter' or 'threshold'
        "formulation": "linopy",  # pyomo or linopy
        "market_optimization": {
            "active": True,
            "market_zones": "status_quo",  # only used if type='market_grid'
            "rolling_horizon": {  # define parameter of market optimization
                "planning_horizon": 168,  # number of snapshots in each optimization
                "overlap": 120,  # number of overlapping hours
            },
            "redispatch": True,
        },
        "distribution_grids": False,  # False or path to file with edisgo results
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
        "FeasibilityTol": 1e-5,
        "Method": 2,
        "BarConvTol": 1e-5,
        "BarHomogeneous": 1,
        "Crossover": 0,
        "Threads": 4,
    },

    "model_formulation": "kirchhoff",  # angles or kirchhoff
    "scn_name": "eGon2035",  # scenario, e.g. eGon2035, eGon2035_lowflex or status2019

    # Scenario variations:
    "scn_extension": None,  # None or array of extension scenarios

    # Export options:
    "lpfile": False,  # save pyomo's lp file: False or /path/to/lpfile.lp
    "csv_export": "biogas_sh_hybrid_10h_50ac_hybrid",  # save results as csv: False or /path/tofolder

    # Settings:
    "extendable": {
        "extendable_components": [
            "as_in_db",
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

    "debug_pre_market_slacks": False,
    "debug_market_model_slacks": False,

    "extra_functionality": {
        "biogas_sh_resource": {
            "csv_path": str(BIOGAS_SH_CSV),
            "eta_el": 0.38,
            "eta_heat": 0.45,
            "eta_upgrade": 0.96,
            "ignore_missing_components": False,
        },
     },

    #"extra_functionality": {},

    # Spatial Complexity:
    "network_clustering_ehv": {
        "active": False,  # choose if clustering of HV buses to EHV buses is activated
        "busmap": False,  # False or path to stored busmap
        "cpu_cores": 4,  # number of cores used during clustering, "max" for all cores available.
    },

    "network_clustering": {
        "method": {
            #"focus_region": str(BIOGAS_SH_FOCUS_REGION),  # None, shape-file or list with string for Kreise
            "focus_region": [
                "Flensburg",
                #"Kiel",
                #"Lübeck",
                #"Neumünster",
                #"Dithmarschen",
                #"Herzogtum Lauenburg",
                "Nordfriesland",
                #"Ostholstein",
                #"Pinneberg",
                #"Plön",
                #"Rendsburg-Eckernförde",
                "Schleswig-Flensburg",
                #"Segeberg",
                #"Steinburg",
                #"Stormarn",
            ],
            "per_country": True,  # if True, buses are restricted to one cluster per foreign country
            "algorithm": "kmedoids-dijkstra",  # choose clustering method: kmeans or kmedoids-dijkstra
            "remove_stubs": False,  # remove stubs before kmeans clustering
            "use_reduced_coordinates": False,  # if True, do not average cluster coordinates (in remove stubs)
            "line_length_factor": 1,  # Factor to multiply distance between new buses for new line lengths
            "random_state": 42,  # random state for replicability of clustering results
            "n_init": 10,  # affects clustering algorithm, only change when necessary
            "max_iter": 100,  # affects clustering algorithm, only change when necessary
            "tol": 1e-6,  # affects clustering algorithm, only change when necessary
            "cpu_cores": 4,  # number of cores used during clustering, "max" for all cores available.
        },
        "electricity_grid": {
            "active": True,  # choose if clustering is activated
            "cluster_within_focus": False,  # False for very low clustering within focus region
            "n_clusters": 50,  # total number of resulting AC nodes
            "k_elec_busmap": False,  # False or path/to/busmap.csv
        },
       "gas_grids": {
           "active": True,
           "cluster_within_focus": False,
           "n_clusters_ch4": 15,
           "n_clusters_h2": 15,
           "k_ch4_busmap": False,

            "protect_custom_ch4_buses": True,

            "custom_ch4_buses": [
                "47538",
                "biogas_sh_swfl_ch4_bus",
                "biogas_sh_storage_ch4_bus",
                "swfl_real_biomethane_ch4_bus",
            ],

            "custom_ch4_bus_prefixes": [
                "biogas_sh_ch4_bus_",
            ],

            "custom_ch4_link_carriers": [
                "biogas_sh_swfl_direct",
                "biogas_sh_gas_grid_injection",
                "biogas_sh_swfl_grid_supply",

                # Plant -> central Biogas.SH storage
                "biogas_sh_collection_to_storage",

                # Central storage output routes
                "biogas_sh_storage_to_grid",
                "biogas_sh_storage_to_swfl",

                # SWFL detailed fuel routes
                "swfl_real_natural_gas_to_boiler",
                "swfl_real_biomethane_to_boiler",
                "swfl_real_gas_to_power",
            ],
        },
    },

    "spatial_disaggregation": None,  # None or 'uniform'

    # Temporal Complexity:
    "snapshot_clustering": {
        "active": False,  # choose if clustering is activated
        "method": "segmentation",  # 'typical_periods' or 'segmentation'
        "extreme_periods": None,  # consideration of extreme timesteps; e.g. 'append'
        "how": "daily",  # type of period - only relevant for 'typical_periods'
        "storage_constraints": "soc_constraints",  # additional constraints for storages - only relevant for 'typical_periods'
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

    "swfl_real_system": {
        "active": True,

        # ------------------------------------------------------------------
        # Area definition: selected DING0 MV grid districts
        # ------------------------------------------------------------------
        # We do NOT use radius-based selection.
        # Flensburg/SWFL assets are selected from these DING0 MV grid districts:
        #   33935, 33543, 35906
        "area_mode": "ding0_mv_grid_districts",

        # Same DING0 MV grid district file usFalseed for Biogas-SH demand mapping.
        "mv_grid_districts_gpkg": str(DING0_MV_GPKG),
        "mv_grid_layer": None,

        # Column in the DING0 MV grid district file containing the district ID.
        "mv_grid_id_column": "name",

        "selected_mv_grid_district_ids": [
            "33935",
            "33543",
            "35906",
        ],

        # Important:
        # The selected MV districts are mainly AC buses.
        # This expands the area to directly connected local heat/CHP/boiler buses.
        "expand_area_through_local_links": True,

        "area_expansion_link_carrier_patterns": [
            "central_gas",
            "central_heat",
            "rural_heat",
            "heat_pump",
            "CHP",
            "boiler",
        ],

        # Remove old generic eGon Flensburg/SWFL assets inside the selected area.
        "remove_existing_flensburg_assets": True,

        # Do not delete Biogas-SH or newly created SWFL components.
        "protected_prefixes": [
            "biogas_sh_",
            "swfl_real_",
            "swfl_gwp_",
        ],

        # Old eGon conversion links to remove inside the selected MV districts.
        "remove_link_carrier_patterns": [
            "central_gas",
            "central_heat",
            "rural_heat",
            "heat_pump",
            "CHP",
            "boiler",
        ],

        # Old eGon heat pumps to remove inside the selected MV districts.
        "remove_heat_pump_carrier_patterns": [
            "central_heat_pump",
            "rural_heat_pump",
            "heat_pump",
        ],

        # ------------------------------------------------------------------
        # Main replacement buses
        # ------------------------------------------------------------------
        "swfl_ac_bus": "33935",
        "swfl_heat_bus": "swfl_real_central_heat_bus",
        "swfl_ch4_bus": "biogas_sh_swfl_ch4_bus",

        # Stadtwerke Flensburg / SWFL approximate location.
        "swfl_ch4_bus_x": 9.436502119171873,
        "swfl_ch4_bus_y": 54.79233181101448,

        # ------------------------------------------------------------------
        # Real SWFL heat load
        # ------------------------------------------------------------------
        # Uses one complete 8760-hour year from Stadtwerke Flensburg.
        "heat_load": {
            "active": True,
            "csv_path": str(SWFL_HEAT_CSV),
            # Use a non-leap year if possible.
            "year": 2023,
            "datetime_column": "timestamp",
            "column": "hkw_heat_mw",
            "unit": "MW",
            "carrier": "central_heat",
            "name": "swfl_real_heat_load",

            # If automatic timestamp detection fails, uncomment and adjust:
            # "datetime_column": "Zeitstempel",

            # Handles missing DST hours.
            "fill_method": "time_interpolate",

            # Avoid negative load values.
            "clip_negative": True,
        },

        # ------------------------------------------------------------------
        # Real-scaled Flensburg/SWFL AC load
        # ------------------------------------------------------------------
        # Temporal shape comes from existing eGon AC loads in the selected
        # MV grid districts. Annual energy is scaled to real electricity demand.
        "ac_load": {
            "active": True,
            "target_is_annual": True,
            "scale_annual_target_to_snapshot_hours": True,

            "use_existing_egon_profile_shape": True,

            # Stadtwerke / Flensburg real annual electricity demand:
            # 381.516 GWh/a = 381,516 MWh/a
            "target_annual_demand_mwh": 381516.0,

            "carrier": "AC",
            "source_carrier": "AC",

            "name": "swfl_real_ac_load",

            "clip_negative": True,

            # Optional fallback if automatic MV-district selection misses loads:
            # "source_load_ids": ["load_id_1", "load_id_2"],
        },

        # ------------------------------------------------------------------
        # Real SWFL central gas CHP
        # ------------------------------------------------------------------
        # Represented as two simple eGon-style links:
        #
        #   biogas_sh_swfl_ch4_bus -> swfl_ac_bus
        #   biogas_sh_swfl_ch4_bus -> swfl_central_heat_bus
        #
        # This version does not yet force a fixed heat-to-power coupling.
        "central_gas_chp": {
            "active": True,

            # Keep only the temporary aggregate 241 MWel electricity side.
            "add_electric_link": True,
            "add_heat_link": False,

            "electric_link_name": "swfl_real_gas_to_power",
            "heat_link_name": "swfl_real_gas_to_heat",

            "electric_capacity_mw": 241.0,
            "heat_capacity_mw": 370.0,

            "gas_bus": "biogas_sh_swfl_ch4_bus",
            "ac_bus": "33935",
            "heat_bus": "swfl_real_central_heat_bus",

            "carrier_el": "swfl_real_gas_to_power",
            "carrier_heat": "swfl_real_gas_to_heat",

            "p_nom_is_output_capacity": True,

            # Provisional assumption until SWFL provides verified data.
            "electric_efficiency": 0.40,

            # Not used while add_heat_link=False.
            "heat_efficiency": 0.90,

            "extendable": False,
            "p_min_pu": 0.0,
            "p_max_pu": 1.0,

            # Fuel cost is represented upstream.
            "marginal_cost": 0.0,
            "capital_cost": 0.0,
        },

        "central_heat_units": {
            "active": True,

            "natural_gas_bus": "biogas_sh_swfl_ch4_bus",
            "biomethane_bus": "swfl_real_biomethane_ch4_bus",
            
            # Dedicated non-upgraded raw-biogas bus.
            "raw_biogas_bus": "swfl_real_raw_biogas_bus",

            "ac_bus": "33935",
            "heat_bus": "swfl_real_central_heat_bus",

            # --------------------------------------------------------------
            # Biomethane scenario
            # --------------------------------------------------------------
            # "off"             -> no biomethane use
            # "k12_k13_only"    -> K12 + K13 = 170 MWth
            # "all_gas_units"   -> K5 + K11 + K12 + K13 = 300 MWth
            # "custom"          -> use custom_biomethane_units
            "biomethane_mode": "k12_k13_only",

            "planned_biomethane_units": [
                "swfl_real_k12",
                "swfl_real_k13",
            ],

            "custom_biomethane_units": [
                "swfl_real_k12",
                "swfl_real_k13",
            ],
            
            # ------------------------------------------------------------------
            # Direct raw-biogas eligibility
            # ------------------------------------------------------------------
            #
            # scenario_config.py will overwrite this according to
            # technical.biogas_sh.direct_raw_biogas_to_swfl.eligible_units.
            #
            "raw_biogas_units": [],

            # Normal eGon2035 case: no HEL.
            "allow_hel_backup": False,

            "hel_bus": "swfl_real_hel_bus",
            "hel_supply_generator": "swfl_real_hel_supply",
            "hel_supply_p_nom_mw": 1000.0,

            # Only used when HEL backup is active.
            "hel_marginal_cost": 0.0,

            # Provisional assumptions until SWFL supplies efficiency data.
            "default_boiler_efficiency": 0.90,
            "default_resistive_efficiency": 0.99,

            "expected_total_heat_capacity_mw": 370.0,

            "boilers": [
                {
                    "name": "swfl_real_k5",
                    "active": True,
                    "heat_capacity_mw": 60.0,

                    "base_fuels": [
                        "natural_gas",
                    ],

                    "optional_fuels": [
                        "hel",
                    ],

                    "efficiency": 0.90,
                    "carrier": "central_gas_boiler",

                    "extendable": False,
                    "p_min_pu": 0.0,
                    "p_max_pu": 1.0,

                    "marginal_cost": 0.0,
                    "capital_cost": 0.0,
                },
                {
                    "name": "swfl_real_k11",
                    "active": True,
                    "heat_capacity_mw": 70.0,

                    # eGon2035 model assumption:
                    # K11 is represented as converted from coal to gas.
                    "base_fuels": [
                        "natural_gas",
                    ],

                    "optional_fuels": [],

                    "efficiency": 0.90,
                    "carrier": "central_gas_boiler",

                    "extendable": False,
                    "p_min_pu": 0.0,
                    "p_max_pu": 1.0,

                    "marginal_cost": 0.0,
                    "capital_cost": 0.0,
                },
                {
                    "name": "swfl_real_k12",
                    "active": True,
                    "heat_capacity_mw": 80.0,

                    "base_fuels": [
                        "natural_gas",
                    ],

                    "optional_fuels": [],

                    "efficiency": 0.90,
                    "carrier": "central_gas_boiler",

                    "extendable": False,
                    "p_min_pu": 0.0,
                    "p_max_pu": 1.0,

                    "marginal_cost": 0.0,
                    "capital_cost": 0.0,
                },
                {
                    "name": "swfl_real_k13",
                    "active": True,
                    "heat_capacity_mw": 90.0,

                    "base_fuels": [
                        "natural_gas",
                    ],

                    "optional_fuels": [],

                    "efficiency": 0.90,
                    "carrier": "central_gas_boiler",

                    "extendable": False,
                    "p_min_pu": 0.0,
                    "p_max_pu": 1.0,

                    "marginal_cost": 0.0,
                    "capital_cost": 0.0,
                },
            ],

            "resistive_heaters": [
                {
                    "name": "swfl_real_ehk1",
                    "active": True,
                    "heat_capacity_mw": 30.0,
                    "efficiency": 0.99,
                    "carrier": "central_resistive_heater",

                    "ac_bus": "33935",
                    "heat_bus": "swfl_real_central_heat_bus",

                    "extendable": False,
                    "p_min_pu": 0.0,
                    "p_max_pu": 1.0,

                    "marginal_cost": 0.0,
                    "capital_cost": 0.0,
                },
                {
                    "name": "swfl_real_ehk2",
                    "active": True,
                    "heat_capacity_mw": 40.0,
                    "efficiency": 0.99,
                    "carrier": "central_resistive_heater",

                    "ac_bus": "33935",
                    "heat_bus": "swfl_real_central_heat_bus",

                    "extendable": False,
                    "p_min_pu": 0.0,
                    "p_max_pu": 1.0,

                    "marginal_cost": 0.0,
                    "capital_cost": 0.0,
                },
            ],
        },

        # ------------------------------------------------------------------
        # Optional SWFL reserve gas boiler
        # ------------------------------------------------------------------
        # Use this if we want to include the 203 MWth reserve plant.
        "reserve_gas_boiler": {
            "active": False,

            "name": "swfl_real_reserve_gas_boiler",

            "heat_capacity_mw": 203.0,

            "gas_bus": "biogas_sh_swfl_ch4_bus",
            "heat_bus": "swfl_real_central_heat_bus",

            "carrier": "central_gas_boiler",

            "p_nom_is_output_capacity": True,
            "efficiency": 1.0,

            "extendable": False,

            "p_min_pu": 0.0,
            "p_max_pu": 1.0,

            "marginal_cost": 0.0,
            "capital_cost": 0.0,
        },

        # ------------------------------------------------------------------
        # Future SWFL large heat pumps
        # ------------------------------------------------------------------
        # First remove existing eGon heat pumps inside the selected MV districts.
        # Then optionally add none, one, or both planned SWFL heat pumps.
        "future_heat_pumps": {
            "remove_existing_central_heat_pumps": True,

            # Main heat-pump scenario switch.
            "active": True,

            # Two-heat-pump scenario:
            "active_units": [
            "swfl_gwp_1",
            "swfl_gwp_2",
            ],

            # Do not define a shared central_heat_pump carrier here.
            "default_cop": 3.0,

            "extendable": False,

            "p_min_pu": 0.0,
            "p_max_pu": 1.0,

            "marginal_cost": 0.0,
            "capital_cost": 0.0,

            "units": [
                {
                    "name": "swfl_gwp_1",

                    # Unique carrier prevents clustering with GWP 2
                    # and generic eGon heat pumps.
                    "carrier": "swfl_gwp_1_heat_pump",

                    # This is ignored when active_units is defined,
                    # but it can remain for compatibility.
                    "active": True,

                    # Useful heat-output capacity.
                    "heat_capacity_mw": 60.0,

                    # p_nom = 60 MWth / 3 = 20 MWel.
                    "cop": 3.0,

                    "planned_year": 2028,

                    "ac_bus": "33935",
                    "heat_bus": "swfl_real_central_heat_bus",
                },
                {
                    "name": "swfl_gwp_2",

                    # Separate carrier for GWP 2.
                    "carrier": "swfl_gwp_2_heat_pump",

                    "active": True,

                    "heat_capacity_mw": 60.0,
                    "cop": 3.0,

                    "planned_year": None,

                    "ac_bus": "33935",
                    "heat_bus": "swfl_real_central_heat_bus",
                },
            ],
        },
    },

    "biogas_sh": {
        "active": True,
        "csv_path": biogas_sh_csv(BIOGAS_SH_CSV_NAME),

        # ------------------------------------------------------------------
        # Flexible scenario routing
        # ------------------------------------------------------------------
        # Options:
        #   "onsite"   -> local electricity/heat only
        #   "gas_grid" -> biomethane to public CH4 grid only
        #   "swfl"     -> biomethane direct to artificial SWFL CH4 bus only
        #   "hybrid"   -> all three routes active
        #   "custom"   -> use the booleans below independently
        "scenario_mode": "hybrid",

        "add_local_generation": True, #use Biogas.SH → supply local heat and electricty
        "add_gas_grid_generation": True, #add Biogas.SH → public gas grid injection route.
        "add_swfl_direct_supply": True, #add Biogas.SH → Stadtwerke Flensburg
        
        # config.yaml activates it for hybrid_raw_swfl.
        "add_swfl_raw_biogas_supply": False,

        # SWFL still needs access to fossil natural gas even when the
        # upgraded-biomethane -> SWFL route is disabled.
        "ensure_swfl_gas_access": True,

        # Legacy field kept for compatibility. If add_gas_grid_generation is absent,
        # this is interpreted as gas-grid generation.
        # "add_gas_generation": True,
        
        "add_biomethane_transport_supply": False,

        # Public gas grid injection route.
        "gas_connection_target": "gas_grid",
        "target_ch4_bus": "47538",
        "ch4_link_carrier": "biogas_sh_gas_grid_injection",
        "gas_topology": "producer_bus_link",
        "ch4_link_p_min_pu": 0.0,
        "ch4_link_efficiency": 1.0,
        "ch4_link_p_nom_factor": 1.0,
        "ch4_link_extendable": False,
        "ch4_link_marginal_cost": 0.0,
        "ch4_link_capital_cost": 0.0,

        # ------------------------------------------------------------------
        # New SWFL direct route
        # ------------------------------------------------------------------
        "swfl_direct": {
            "active": True,

            "swfl_ch4_bus": "biogas_sh_swfl_ch4_bus",
            "swfl_ch4_bus_x": 9.436502119171873,
            "swfl_ch4_bus_y": 54.79233181101448,

            "public_ch4_bus": "47538",

            # Do not redirect old eGon SWFL links anymore.
            # They are removed/replaced by swfl_real_system.
            "consumer_link_ids": [],
            "redirect_consumer_links": False,

            # Keep public-grid backup to the SWFL CH4 bus.
            "grid_supply_active": True,
            "grid_supply_link": "biogas_sh_swfl_grid_supply_47538_to_swfl",
            "grid_supply_extendable": False,
            "grid_supply_p_nom_factor": 1.0,
            "grid_supply_efficiency": 1.0,
            "grid_supply_marginal_cost": 0.0, # Fossil-gas commodity and CO2 costs are applied upstream to CH4_NG generators.
            "grid_supply_capital_cost": 0.0,
            "grid_supply_p_nom": 1500.0,

            # Direct Biogas-SH plant CH4 buses -> SWFL CH4 bus.
            "direct_link_p_nom_factor": 1.0,
            "direct_link_extendable": False,
            "direct_link_efficiency": 1.0,
            "direct_link_marginal_cost": 0.0,
            "direct_link_capital_cost": 0.0,

            # Do not add a separate gas load here.
            # SWFL demand now comes from the real CHP/heat links.
            "add_swfl_gas_load": False,
            "swfl_demand_mwh_a": 0.0,
        },
        
        "raw_biogas_to_swfl": {

            # Safe default. The route case in config.yaml overwrites this.
            "active": False,

            "target_bus": "swfl_real_raw_biogas_bus",
        
            # Raw-biogas commodity cost.
            #
            # scenario_config.py overwrites this from:
                #
                #   price_cases.onsite.raw_biogas_cost_eur_per_mwh_hs
                #
            "raw_biogas_cost_eur_per_mwh_hs": 75.0,

            # Project-derived collection / transport adder.
            #
            # scenario_config.py overwrites this from:
                #
                # technical.biogas_sh.direct_raw_biogas_to_swfl
                #
            "transport_cost_eur_per_mwh_hs": 8.78,

            # Final delivered cost:
                #
                #   75.00 + 8.78
                #   = 83.78 EUR/MWh_Hs
                #
            "marginal_cost_eur_per_mwh_hs": 83.78,

            # None means:
            #
            #   total annual regional raw-biogas potential / 8760
            #
            "power_capacity_mw": None,

            "generator_name": "biogas_sh_raw_biogas_swfl_supply",
        
            "generator_carrier": "biogas_sh_raw_biogas_swfl",

            "bus_carrier": "raw_biogas",

            # Units allowed to receive direct raw biogas.
            "eligible_units": [
                "swfl_real_k12",
                "swfl_real_k13",
            ],
        },
            
        "transport_biomethane": {

            "active": True,

            "h2_transport_load_carrier":
                "H2_hgv_load",

            "eligible_mv_grid_ids": [
                "34966",
            ],

            "source_store":
                "biogas_sh_ch4_store",

            "transport_bus_prefix":
                "biogas_sh_hgv_transport_energy_",

            "transport_bus_carrier":
                "biogas_sh_hgv_transport_energy",

            "h2_link_prefix":
                "biogas_sh_h2_to_hgv_transport_",

            "h2_link_carrier":
                "biogas_sh_h2_to_hgv_transport",

            "biomethane_link_prefix":
                "biogas_sh_biomethane_to_hgv_transport_",

            "biomethane_link_carrier":
                "biogas_sh_biomethane_to_hgv_transport",

            "h2_to_transport_efficiency":
                1.0,

            "biomethane_to_transport_efficiency":
                1.0,

            "h2_link_p_nom_factor":
                1.0,

            "biomethane_link_p_nom_factor":
                1.0,

            "delivery_cost_eur_per_mwh_hs":
                0.0,

            "thg_quota_active":
                False,

            "thg_quota_price_eur_per_tco2":
                280.0,

            "ghg_saving_tco2_per_mwh_hs":
                None,

            "thg_credit_override_eur_per_mwh_hs":
                None,
        },

        "gas_storage": {
            "active": True,

            "bus": "biogas_sh_storage_ch4_bus",
            "store": "biogas_sh_ch4_store",

            "country": "DE",
            "x": 9.436502119171873,
            "y": 54.79233181101448,

            # Plant -> storage
            "input_link_name_prefix": "biogas_sh_storage_input",
            "input_link_carrier": "biogas_sh_collection_to_storage",
            "input_link_p_nom_factor": 1.0,
            "input_link_extendable": False,
            "input_link_p_nom_min": 0.0,
            "input_link_p_min_pu": 0.0,
            "input_link_p_max_pu": 1.0,
            "input_link_efficiency": 1.0,
            "input_link_marginal_cost": 0.0,
            "input_link_capital_cost": 0.0,

            # Storage energy capacity
            "e_nom_mwh": 500.0,
            "e_nom_extendable": False,
            "e_nom_min": 0.0,
            "e_initial": 0.0,
            "e_cyclic": True,
            "standing_loss": 0.0,
            "marginal_cost": 0.0,
            "capital_cost": 0.0,

            # Storage -> public CH4 grid
            "grid_link": "biogas_sh_storage_to_grid_47538",
            "grid_link_carrier": "biogas_sh_storage_to_grid",
            "grid_link_p_nom_mw": 50.0,
            "grid_link_extendable": False,
            "grid_link_p_nom_min": 0.0,
            "grid_link_p_min_pu": 0.0,
            "grid_link_p_max_pu": 1.0,
            "grid_link_efficiency": 1.0,
            "grid_link_marginal_cost": 0.0,
            "grid_link_capital_cost": 0.0,

            # Storage -> dedicated SWFL biomethane bus
            "swfl_target_bus": "swfl_real_biomethane_ch4_bus",
            "swfl_target_bus_x": 9.436502119171873,
            "swfl_target_bus_y": 54.79233181101448,

            "swfl_link": "biogas_sh_storage_to_swfl_biomethane",
            "swfl_link_carrier": "biogas_sh_storage_to_swfl",

            # Keep 50 MWgas for the constrained test.
            # Use 188.8889 MWgas to enable 170 MWth at eta=0.90.
            "swfl_link_p_nom_mw": 50.0,

            "swfl_link_extendable": False,
            "swfl_link_p_nom_min": 0.0,
            "swfl_link_p_min_pu": 0.0,
            "swfl_link_p_max_pu": 1.0,
            "swfl_link_efficiency": 1.0,
            "swfl_link_marginal_cost": 0.0,
            "swfl_link_capital_cost": 0.0,
        },

        # ------------------------------------------------------------------
        # Demand-side mapping
        # ------------------------------------------------------------------
        "auto_map_demand_buses": True,
        "mv_grid_districts_gpkg": str(DING0_MV_GPKG),
        "mv_grid_layer": None,
        "mv_grid_id_column": "name",
        "mv_grid_bus_column": "name",
        "fallback_to_neighbor_area": True,
        "ac_load_carrier": "AC",
        "heat_demand_carrier": "rural_heat",
        "rural_heat_pump_carrier": "rural_heat_pump",

        "write_mapped_csv": str(BIOGAS_SH_MAPPED_CSV),
        "print_skipped_components": True,

        # Plant carriers for onsite route.
        "ac_only_carrier": "biogas_sh_onsite_el",
        "chp_el_carrier": "biogas_sh_onsite_chp_el",

        "supported_ac_only_carrier": "biogas_sh_onsite_el_supported",
        "supported_chp_el_carrier": "biogas_sh_onsite_chp_el_supported",

        "chp_heat_carrier": "biogas_sh_onsite_chp_heat",

        # Capacities.
        "electric_capacity_column": "hbl_95_mwel",
        "heat_capacity_method": "annual_heat_div_8760",
        "hours_for_heat_capacity": 8760.0,
        "hours_for_gas_capacity": 8760.0,

        # ------------------------------------------------------------------
        # Final Biogas.SH economic assumptions
        # ------------------------------------------------------------------
        # Raw biogas:
        #   7.5 ct/kWh_Hs = 75.0 €/MWh_Hs
        #
        # Merchant onsite electricity:
        #   75 / 0.38 = 197.37 €/MWh_el
        #
        # Onsite heat:
        #   75 / 0.45 = 166.67 €/MWh_th
        #
        # Central biomethane:
        #   75 + 17.9 = 92.9 €/MWh_Hs
        #
        # The 92.9 €/MWh_Hs is an all-in fixed cost representation.
        # PSA electricity, compression electricity, pipeline CAPEX and
        # storage CAPEX are therefore NOT added separately.

        "raw_biogas_cost_eur_per_mwh_hs": 75.0,

        "electricity_marginal_cost": 197.37,
        "heat_marginal_cost": 166.67,

        "biomethane_price_override_eur_per_mwh": 92.9,
        "default_biomethane_cost": 92.9,

        "support": {
            # Safe default if config.yaml has not overridden the run.
            "case": "post_eeg",

            "eeg_active": False,

            "market_electricity_marginal_cost": 197.37,
            "supported_electricity_marginal_cost": 110.31,

            "supported_hours_per_year": 0.0,

            "flexibility_active": False,
            "chp_capacity_multiplier": 1.0,

            "flex_capex_eur_per_kw": 800.0,
            "flex_lifetime_years": 15,
            "flex_discount_rate": 0.05,
            "flex_fixed_om_fraction": 0.02,
            "flexibility_payment_eur_per_kw_year": 0.0,
        },
        
        },      
}



def run_etrago(args, json_path):
    """Function to conduct optimization considering the following arguments.

    Parameters
    ----------
    db : str
        Name of Database session setting stored in *config.ini* within
        *.etrago_database/* in case of local database,
        or ``'oep'`` to load model from OEP.
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
        * "formulation" : str
            Select formulation used for model building.
            You can either choose "pyomo" or "linopy".
            Default: "linopy".
        * "market_optimization" : dict
            Select if a seperate market optimization should be performed before the
            grid optimization. Otherwise, an integrated optimization is performed.
            Per default, the following dictionary is set:

            {
                "active": True,
                "market_zones": "status_quo",
                "rolling_horizon": {
                    "planning_horizon": 168,
                    "overlap": 120,
                },
                "redispatch": True,
            }

        * "distribution_grids" : str
            If you want to consider simplyfied distribution grids within the
            transmission grid optimization, provide a path to the parameters for each
            distribution grid here.
            Default: False.

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
"add_gas_grid_generation": True,
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
        * 'cross_border etrago.spatial_clustering_gas()_flows_per_country' : dict of cntr and array of floats
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
        * "CPU_cores" : int or str
            Number of cores used in clustering. Specify a concrete number or
            "max" to use all cores available.
            Default: 4.

    network_clustering : dict
        Choose if you want to apply a clustering of the network buses and
        specify settings.
        The provided dictionary can have the following entries:

        * "method" : dict
            Choose general settings for network clusterings:

            * "focus_region": None or str or list(str)
                Defines a focus region for clustering. A higher spatial resolution
                will be applied inside and around this region.
                Enter a path to a shape-file or add a list of strings with Kreisnamen.
                Needs to be one connected region with defined CRS.
                Default: None.
            * "per_country": bool
                If True, the clusters are constrained to one cluster per foreign
                country. If set to False, the AC buses outside and inside Germany
                are clustered in one process.
                Default: True.
            * "algortihm": dict
                Algorithm used for clustering. You can choose between two
                clustering methods:
                    * "kmeans": considers geographical locations of buses
                    * "kmedoids-dijkstra":  considers electrical distances between
                        buses
                Default: "kmedoids-dijkstra".
            * "remove_stubs" : bool
                If True, remove stubs before k-means clustering, which reduces the
                overestimating of line meshes.
                This option is only used within the k-means clustering.
                Default: False.
            * "use_reduced_coordinates" : bool
                If True, do not average cluster coordinates, but take from busmap.
                This option is only used within the k-means clustering.
                Default: False.
            * "line_length_factor" : float
                Defines the factor to multiply the crow-flies distance
                between new buses by, in order to get new line lengths.
                Default: 1.
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

        * "electricity_grid" : dict
            Choose clustering settings for electricity grid:

            * "active": bool
                If True, the AC buses are clustered down to ``'n_clusters'``.
                Default: True.
            * "cluster_within_focus": bool
                If False, the AC buses within the focus region will not be clustered.
            apply_biogas_sh_assets = apply_biogas_sh_assets    Default: True.
            * "n_clusters" : int
                Defines total number of resulting AC nodes including DE and foreign
                nodes if `cluster_foreign_AC` is set to True, otherwise only DE
                nodes.
                Default: 30.
            * "k_elec_busmap" : bool or str
                With this option you can load cluster coordinates from a previous
                AC clustering run. Options are False, in which case no previous
                busmap is loaded, and path/to/busmap.csv in which case the busmap
                is loaded from the specified file. Please note, that when a path is
                provided, the set number of clusters will be ignored.
                Default: False.

        * "gas_grids" : dict
            Choose clustering settings for CH4 and H2 grids:

            * "active": bool
                If True, the AC buses are clustered down to ``'n_clusters'``.
                Default: True.
            * "cluster_within_focus": bool
                If False, the gas grid buses within the focus region will barely be clustered.
                Default: True.
            * "n_clusters_ch4" : int
                Defines total number of resulting CH4 nodes including DE and
                foreign nodes if `cluster_foreign_gas` is set to True, otherwise
                only DE nodes.
                Default: 15.
            * "n_clusters_h2" : int
                Defines total number of resulting H2 nodes including DE and
                foreign nodes if `cluster_foreign_gas` is set to True, otherwise
                only DE nodes.
                Default: 15.
            * "k_ch4_busmap" : bool or str
                With this option you can load cluster coordinates from a previous
                gas clustering run. Options are False, in which case no previous
                busmap is loaded, and path/to/busmap.csv in which case the busmap
                is loaded from the specified file. Please note, that when a path is
                provided, the set number of clusters will be ignored.
                Default: False.
            * "sector_coupled_clustering" : bool
                Choose if you want to apply a clustering of sector coupled carriers,
                such as central_heat. You finde the specified settings in cluster/gas.py.
                Default: True.

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
    config_path = Path(__file__).with_name(
        "config.yaml"
    )

    args, resolved_scenario = load_and_apply_config(
        args,
        config_path,
    )

    print(
        scenario_summary(
            resolved_scenario
        )
    )

    etrago = Etrago(args, json_path=json_path)

    # import network from database
    etrago.build_network_from_db()

    # adjust network regarding eTraGo setting
    etrago.adjust_network()

    # Add the detailed SWFL assets
    apply_swfl_real_system(
        etrago.network,
        args.get("swfl_real_system", {}),
    )

    biogas_sh_active = bool(
        args.get("biogas_sh", {}).get("active", False)
    )

    swfl_active = bool(
        args.get("swfl_real_system", {}).get("active", False)
    )

    if not biogas_sh_active:

        extra = args.setdefault(
            "extra_functionality",
            {},
        )

        # --------------------------------------------------
        # Disable EEG/flex constraints belonging exclusively
        # to the Biogas.SH plant fleet.
        # --------------------------------------------------
        support = extra.get(
            "biogas_sh_support"
        )

        if isinstance(support, dict):
            support["active"] = False

        # --------------------------------------------------
        # Disable the regional Biogas.SH resource constraint.
        # No Biogas.SH plants exist in this control run.
        # --------------------------------------------------
        resource = extra.get(
            "biogas_sh_resource"
        )

        if isinstance(resource, dict):
            resource["active"] = False
            resource["ignore_missing_components"] = True

        logger.info(
            "Biogas.SH inactive: disabled "
            "Biogas.SH support/resource constraints."
        )

    # Biogas.SH connects to buses created by the SWFL setup
    apply_biogas_sh_assets(etrago)

    # ------------------------------------------------------------
    # Preserve ordinary public-grid CH4 supply to SWFL even when
    # Biogas.SH assets are disabled.
    # ------------------------------------------------------------
    if (not biogas_sh_active) and swfl_active:

        n = etrago.network

        public_ch4_bus = "47538"
        swfl_ch4_bus = "biogas_sh_swfl_ch4_bus"

        grid_supply_link = (
            "biogas_sh_swfl_grid_supply_47538_to_swfl"
        )

        if public_ch4_bus not in n.buses.index:
            raise RuntimeError(
                f"Public CH4 bus {public_ch4_bus} is missing."
            )

        if swfl_ch4_bus not in n.buses.index:
            raise RuntimeError(
                f"SWFL CH4 bus {swfl_ch4_bus} is missing."
            )

        if grid_supply_link not in n.links.index:

            if "biogas_sh_swfl_grid_supply" not in n.carriers.index:
                n.add(
                    "Carrier",
                    "biogas_sh_swfl_grid_supply",
                )

            n.add(
                "Link",
                grid_supply_link,
                bus0=public_ch4_bus,
                bus1=swfl_ch4_bus,
                carrier="biogas_sh_swfl_grid_supply",
                p_nom=1500.0,
                p_nom_extendable=False,
                efficiency=1.0,
                marginal_cost=0.0,
                capital_cost=0.0,
                p_min_pu=0.0,
                p_max_pu=1.0,
                scn_name="eGon2035",
            )

            logger.info(
                "Biogas.SH inactive: retained ordinary "
                "public-grid CH4 supply %s -> %s.",
                public_ch4_bus,
                swfl_ch4_bus,
            )

    if (not biogas_sh_active) and swfl_active:

        n = etrago.network

        biomethane_bus = "swfl_real_biomethane_ch4_bus"

        if biomethane_bus in n.buses.index:

            # Find every Link connected to the biomethane-only bus.
            attached_links = n.links.index[
                n.links["bus0"].astype(str).eq(biomethane_bus)
                | n.links["bus1"].astype(str).eq(biomethane_bus)
            ].tolist()

            logger.info(
                "Biogas.SH inactive: removing biomethane-only "
                "SWFL links: %s",
                attached_links,
            )

            for link_id in attached_links:
                n.remove("Link", link_id)

            # The bus itself has no role in the no-Biogas baseline.
            n.remove("Bus", biomethane_bus)

            logger.info(
                "Biogas.SH inactive: removed unused SWFL "
                "biomethane bus '%s'.",
                biomethane_bus,
            )


    # Assign the selected scenario name to custom components
    fix_custom_component_scn_names(
        etrago.network,
        scn_name=args.get("scn_name", "eGon2035"),
    )

    apply_network_price_scenario(
        network=etrago.network,
        resolved=resolved_scenario,
        biogas_sh_active=biogas_sh_active,
    )


    # Validate the Biogas.SH storage topology before clustering.
    if biogas_sh_active:
        validate_biogas_sh_storage_topology(
            network=etrago.network,
            args=args,
        )
    else:
        logger.info(
            "Biogas.SH inactive: "
            "storage topology validation skipped."
        )

    # remove the original eGon heat pump

    if swfl_active:

        remove_known_legacy_swfl_heat_pump_before_clustering(
            etrago.network,
        )

    etrago.spatial_clustering()

    # Defensive post-clustering cleanup.
    future_hp_cfg = (
        args.get("swfl_real_system", {})
        .get("future_heat_pumps", {})
    )

    if swfl_active and future_hp_cfg.get("active", False):
        purge_legacy_swfl_heat_pumps(
            etrago.network,
            stage="after spatial clustering",
        )


    if (not biogas_sh_active) and swfl_active:

        n = etrago.network
        b = "biogas_sh_swfl_ch4_bus"

        print("\nNO-BIOGAS SWFL CH4 CHECK")
        print("----------------------------------")
        print(
            "SWFL bus carrier:",
            n.buses.loc[b, "carrier"],
        )

        mask = (
            n.links.bus0.astype(str).eq(b)
            | n.links.bus1.astype(str).eq(b)
        )

        print(
            n.links.loc[
                mask,
                [
                    "bus0",
                    "bus1",
                    "carrier",
                    "p_nom",
                ],
            ].to_string()
        )

        print("----------------------------------\n")


    etrago.spatial_clustering_gas()


    apply_biogas_sh_transport_route(
        etrago
    )


    # snapshot clustering
    etrago.snapshot_clustering()

    # skip snapshots
    etrago.skip_snapshots()


    n = etrago.network

    print("Buses:", n.buses.shape)
    print("Links:", n.links.shape)
    print("Generators:", n.generators.shape)
    print("Stores:", n.stores.shape)

    # Consistency check
    n.consistency_check()

    restore_load_shedding_after_clustering(
        etrago,
        negative_load_shedding=("Li_ion",),
    )

    # Network was modified by the restoration, so check again.
    n.consistency_check()

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

    result_directory = args.get("csv_export")

    if result_directory:
        result_directory = Path(result_directory)
        result_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_resolved_config(
            resolved_scenario,
            result_directory / "resolved_config.yaml",
        )

        network_path = result_directory / "network.nc"

        etrago.network.export_to_netcdf(
            str(network_path)
        )

        print(
            f"Saved PyPSA network: {network_path}"
        )

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
