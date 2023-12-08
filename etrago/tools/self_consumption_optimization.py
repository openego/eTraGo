import pypsa
import os
import numpy as np
import pandas as pd
import configparser as cp
import multiprocessing as mp
from sqlalchemy import create_engine, select
import saio
from etrago.tools import db
from sqlalchemy.orm import sessionmaker

filepath = os.path.join(
    os.path.expanduser("~"), ".etrago_database", "config.ini"
)
section = "egon-data"
pw = "data"


# Assumptions

max_hours = 1

# Create Connection
cfg = db.readcfg(filepath, section)

conn = create_engine(
    "postgresql+{dialect}://{user}:{password}@{host}:{port}/{db}".format(
        dialect=cfg.get(section, "dialect", fallback="psycopg2"),
        user=cfg.get(section, "username"),
        password=pw,
        host=cfg.get(section, "host"),
        port=cfg.get(section, "port"),
        db=cfg.get(section, "database"),
    )
)
session = sessionmaker(bind=conn)()

# Import battery data
sql = """
    SELECT * FROM
    supply.egon_home_batteries
    WHERE scenario = 'eGon2035' Limit 5

"""
home_battery = pd.read_sql(sql, conn)

# MW in W
home_battery.p_nom = home_battery.p_nom * 1000000

n_process=2
battery_set={}
length = int(len(home_battery) / n_process)
for i in range(n_process):
    battery_set[str(i+1)]=home_battery.index[i*length : (i+1)*length].values
manager = mp.Manager()
d = manager.dict()
battery_set[str(n_process)] = home_battery.index[i*length :].values


def self_consumption_optimization(battery_set, home_battery):

    for idx in battery_set[0]:


        # Extract building ID
        building_id = home_battery.building_id.loc[idx]

        # Import PV rooftop data
        sql = f"""
            SELECT * FROM
            supply.egon_power_plants_pv_roof_building
            WHERE scenario = 'eGon2035'
            AND building_id={building_id};
        """
        pv_rooftop = pd.read_sql(sql, conn)
        pv_capacity = pv_rooftop.capacity.iloc[0] * 1000000

        # Import PV feedin
        sql = f"""
            SELECT feedin FROM
            supply.egon_era5_renewable_feedin
            WHERE w_id = {pv_rooftop.weather_cell_id.iloc[0]}
            AND carrier = 'pv'
        """

        pv_feedin = pd.read_sql(sql, conn)
        pv_feedin_normalized = pd.Series(pv_feedin.feedin.iloc[0])


        # Import demand data

        sql = f"""
            SELECT * FROM
            demand.egon_household_electricity_profile_of_buildings
            WHERE building_id={building_id}
        """
        hh_idp_match = pd.read_sql(sql, conn)
        profile_id = hh_idp_match.profile_id.iloc[0]

        # Select the demand profile by its id
        sql = f"""
            SELECT * FROM
            demand.iee_household_load_profiles
            WHERE type = '{profile_id}'
        """
        idp_household = pd.read_sql(sql, conn)

        # Select peak load for the chosen building
        building_id = hh_idp_match.building_id.iloc[0]
        sql = f"""
            SELECT peak_load_in_w FROM
            demand.egon_building_electricity_peak_loads
            WHERE scenario = 'eGon2035'
            AND sector= 'residential'
            AND building_id = {building_id}
        """
        peak_load = pd.read_sql(sql, conn).iloc[0, 0]

        peak_idp = max(idp_household.load_in_wh.iloc[0])

        household_profile = pd.Series(idp_household.load_in_wh.iloc[0]) * (
            peak_load / peak_idp
        )


        # Create network
        network_sco = pypsa.Network()
        network_sco.snapshots=range(8760)

        # Add an electrical bus
        network_sco.add("Bus", "AC_bus", carrier="AC")

        # Add PV rooftop plant
        network_sco.add(
            "Generator",
            "PV_rooftop",
            bus="AC_bus",
            carrier="solar",
            p_nom=pv_capacity,
            p_max_pu=pv_feedin_normalized,
            marginal_cost=0,
        )

        # Add electrical load
        network_sco.add(
            "Load", "AC_load", bus="AC_bus", carrier="AC", p_set=household_profile
        )

        # Add battery storage
        network_sco.add(
            "StorageUnit",
            "Home_Battery",
            bus="AC_bus",
            p_nom=home_battery.p_nom.loc[idx],
            carrier="AC",
            max_hours=max_hours,
            efficiency_store=0.98,
            efficiency_dispatch=0.98,
            standing_loss=0,
        )

        # Add a generator to represent electricity purchase from the grid
        network_sco.add(
            "Generator",
            "Grid_purchase",
            bus="AC_bus",
            carrier="AC",
            marginal_cost=300,
            p_nom=10000000,
        )

        # Add bus, link and store to represent electricity feed into the grid

        network_sco.add("Bus", "Grid_bus", carrier="AC")

        network_sco.add(
            "Link",
            "Grid_feedin",
            bus0="AC_bus",
            bus1="Grid_bus",
            carrier="AC",
            efficiency=1,
            p_nom=pv_capacity,
        )

        network_sco.add(
            "Store",
            "Grid_store",
            bus="Grid_bus",
            carrier="AC",
            e_nom=10000000000,
            e_initial=0,
        )

        network_sco.lopf(solver_name="gurobi", pyomo=False)
        network_sco.export_to_csv_folder(f"results_building_{idx}")

processes=[mp.Process(target=self_consumption_optimization, args=([battery_set[i]], home_battery)) for i in battery_set.keys()]

# Run processes
for p in processes:
    p.start()


# Exit the completed processes
for p in processes:
    p.join()
for p in processes:
    p.terminate()
