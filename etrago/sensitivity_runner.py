
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

args = {
    "nuts_3_map" : "germany-de-nuts-3-regions.geojson",
    # Setup and Configuration:
    "db": "egon-data-wam02",  # database session # "egon-data-wam02"
    "gridversion": None,  # None for model_draft or Version number
    "method": {  # Choose method and settings for optimization
        "type": "lopf",  # type of optimization, 'lopf' or 'sclopf'
        "n_iter": 4,  # abort criterion of iterative optimization, 'n_iter' or 'threshold'
        "formulation": "linopy",
        "market_optimization":
            {
                "active": False,
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
    "end_snapshot": 8760,
    "solver": "gurobi",  # glpk, cplex or gurobi
    "solver_options": {
        "BarConvTol": 1.0e-9,
        "FeasibilityTol": 1.0e-9,
        "method": 2,
        "crossover": 1,
        "logFile": "solver_etrago.log",
        "threads": 4,
        "NumericFocus": 0,
        "BarHomogeneous": 1,
    },
    "model_formulation": "kirchhoff",  # angles or kirchhoff
    "scn_name": "eGon2035",  # scenario: eGon2035, eGon100RE or status2019
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
    "extra_functionality": {
        "add_chp_ratio_constraint": {}
    },  # Choose function name or {}
    # Spatial Complexity:
    "delete_dispensable_ac_buses": True,  # bool. Find and delete expendable buses
    "interest_area": ["Ingolstadt"],  # False, path to shapefile or list of nuts names of the area that is excluded from the clustering. By default the buses inside remain the same, but the parameter "n_cluster_interest_area" inside "network clustering" defines if it should be clustered to a certain number of buses.
    "network_clustering_ehv": {
        "active": False,  # choose if clustering of HV buses to EHV buses is activated
        "busmap": False,  # False or path to stored busmap
    },
    "network_clustering": {
        "active": True,  # choose if clustering is activated
        "method": "kmedoids-dijkstra",  # choose clustering method: kmeans or kmedoids-dijkstra
        "n_clusters_AC": 30,  # total number of resulting AC nodes (DE+foreign-interest_area)
        "cluster_foreign_AC": False,  # take foreign AC buses into account, True or False
        "n_cluster_interest_area": 1, # False or number of buses.
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
        "CPU_cores": 7,  # number of cores used during clustering, "max" for all cores available.
    },
    "sector_coupled_clustering": {
        "active": True,  # choose if clustering is activated
        "carrier_data": {  # select carriers affected by sector coupling
            "central_heat": {
                "base": ["CH4", "AC"],
                "strategy": "simultaneous",  # select strategy to cluster other sectors
            },
            "rural_heat": {
                "base": ["CH4", "AC"],
                "strategy": "simultaneous",  # select strategy to cluster other sectors
            },
            "H2": {
                "base": ["CH4"],
                "strategy": "consecutive",  # select strategy to cluster other sectors
            },
            "H2_saltcavern": {
                "base": ["H2_grid"],
                "strategy": "consecutive",  # select strategy to cluster other sectors
            },
            "Li_ion": {
                "base": ["AC"],
                "strategy": "consecutive",  # select strategy to cluster other sectors
            },
        },
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
    "skip_snapshots": 3,  # False or number of snapshots to skip
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

import pypsa
from etrago.network import Etrago, find_interest_buses
import os

class SensitivityEtrago(Etrago):
    def __init__(self, nc_path="base_network_Scenario_1a.nc", args=None):
        # Initialisiere NICHT den vollen eTraGo-Workflow, sondern lade nur das gespeicherte Netzwerk
        self.network = pypsa.Network(nc_path)
        self.args = args  # Default-Pfad für Ergebnisexport

        # 🔧 Wichtige Initialisierungen für Kompatibilität
        self.busmap = {}
        self.ch4_h2_mapping = {}
        self.tool_version = "manual_sensitivity"

    def update_capital_cost_of_solar_ingolstadt(self, new_capital_cost):
        """
        Setzt den capital_cost für alle solar_rooftop Generatoren in der Interest Area Ingolstadt.
        """
        buses_ingolstadt = find_interest_buses(self)
        bus_list = buses_ingolstadt.index.to_list()

        gens = self.network.generators
        is_solar_in_ingolstadt = (gens.carrier == "solar_rooftop") & (gens.bus.isin(bus_list))
        solar_generators = gens[is_solar_in_ingolstadt]

        if solar_generators.empty:
            print("⚠️ Keine passenden Solar-Generatoren in Ingolstadt gefunden.")
            return

        self.network.generators.loc[solar_generators.index, "capital_cost"] = new_capital_cost
        print(f"✅ capital_cost auf {new_capital_cost:.2f} €/MW/a für {len(solar_generators)} Solar-Generator(en) gesetzt.")

    def update_marginal_cost_of_CH4_generators(self, new_marginal_cost):
        gens = self.network.generators
        is_ch4 = gens.carrier == "CH4_NG"
        if is_ch4.sum() == 0:
            print("⚠️ Keine CH4_NG-Generatoren gefunden.")
            return

        self.network.generators.loc[is_ch4, "marginal_cost"] = new_marginal_cost
        print(f"✅ marginal_cost auf {new_marginal_cost:.2f} €/MWh gesetzt ({is_ch4.sum()} CH₄-Generatoren).")

    def update_marginal_cost_due_to_CO2_price(self, CO2_new):
        """
        Passt die marginal costs von CH4_NG- und waste-Generatoren an,
        um den veränderten CO2-Preis zu berücksichtigen.

        Formel:
            marginal_cost += (CO2_new - CO2_default) * emissions_factor

        Args:
            CO2_new (float): Neuer CO2-Preis in €/tCO2.
        """

        gens = self.network.generators

        # Fester Default-Preis
        CO2_default = 76.5  # €/tCO2

        # Delta CO2
        delta_CO2 = CO2_new - CO2_default

        # Emissionsfaktoren in tCO2/MWh
        emissions_factors = {
            "CH4_NG": 0.201,
            "waste": 0.165
        }

        # Selektiere Generatoren mit Carrier CH4_NG oder waste
        mask = gens.carrier.isin(emissions_factors.keys())

        if mask.sum() == 0:
            print("⚠️ Keine relevanten Generatoren mit CO2-Emissionen gefunden.")
            return

        # Iteriere über die betroffenen Carrier
        for carrier, ef in emissions_factors.items():
            carrier_mask = gens.carrier == carrier
            n = carrier_mask.sum()
            if n == 0:
                continue

            # Berechne den Zuschlag
            delta_marginal = delta_CO2 * ef

            # Addiere zum bestehenden marginal_cost
            self.network.generators.loc[carrier_mask, "marginal_cost"] += delta_marginal

            print(
                f"✅ {n} {carrier}-Generator(en): marginal_cost um {delta_marginal:.2f} €/MWh angepasst (neuer CO2-Preis: {CO2_new} €/tCO2, alt: {CO2_default} €/tCO2).")


# === Sensitivitäten ===

def run_solar_cost_sensitivity():
    cost_values = [15352.2938, 16083.35546, 16814.41707, 17545.47868] # 210,220,230,240
    for cost in cost_values:
        print(f" Starte Solar-Sensitivität mit capital_cost = {cost:.2f} €/MW/a")

        etrago = SensitivityEtrago(args=args)

        etrago.update_capital_cost_of_solar_ingolstadt(cost)

        export_dir = f"results/sensitivity_solar_cost_{cost:.5f}".replace(".", "_")
        os.makedirs(export_dir, exist_ok=True)
        etrago.args["csv_export"] = export_dir

        etrago.optimize()
        print(f"✅ Ergebnisse gespeichert unter: {export_dir}")

def run_ch4_cost_sensitivity():
    ch4_prices = [20, 30, 60, 80, 100]  # in €/MWh
    for price in ch4_prices:
        print(f" Starte CH₄-Sensitivität mit marginal_cost = {price:.2f} €/MWh_th")

        etrago = SensitivityEtrago(args=args)

        etrago.update_marginal_cost_of_CH4_generators(price)

        export_dir = f"results/sensitivity_CH4_price_{price:.2f}".replace(".", "_")
        os.makedirs(export_dir, exist_ok=True)
        etrago.args["csv_export"] = export_dir

        etrago.optimize()
        print(f"✅ Ergebnisse gespeichert unter: {export_dir}")

def run_co2_price_sensitivity():
    """
    Führt eine Sensitivitätsanalyse über verschiedene CO2-Preise durch.
    Für jeden Preis wird das Modell mit angepassten marginal_costs gerechnet.
    Ergebnisse werden in separate Ordner exportiert.
    """

    CO2_prices = [50, 100, 130, 160, 200]  # in €/tCO2

    for price in CO2_prices:
        print(f"🔄 Starte CO₂-Sensitivität mit CO2-Preis = {price:.2f} €/tCO2")

        # Neues Modell initialisieren
        etrago = SensitivityEtrago(args=args)

        # CO2-bedingte Marginalkosten anpassen
        etrago.update_marginal_cost_due_to_CO2_price(price)

        # Export-Verzeichnis anlegen
        export_dir = f"results/sensitivity_CO2_price_{price:.2f}".replace(".", "_")
        os.makedirs(export_dir, exist_ok=True)
        etrago.args["csv_export"] = export_dir

        # Optimierung starten
        etrago.optimize()

        print(f"✅ Ergebnisse gespeichert unter: {export_dir}")



if __name__ == "__main__":
    #run_solar_cost_sensitivity()
    run_ch4_cost_sensitivity()
    run_co2_price_sensitivity()
