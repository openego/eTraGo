#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:17:53 2024

@author: student
"""

def add_EC_to_network(self):
    """Adds Energy Community to the network."""
    from geoalchemy2.shape import from_shape, to_shape
    from shapely.geometry import LineString, MultiLineString, Point
    
        
    print("Buses before addition:", len(self.network.buses))
    print("Transformers before addition:", len(self.network.transformers))
    print("Lines before addition:", len(self.network.lines))
    print("Generators before addition:", len(self.network.generators))
    print("Loads before addition:", len(self.network.loads))

    
    # Generate new component IDs
    new_bus = str(self.network.buses.index.astype(int).max() + 1)
    new_trafo = str(self.network.transformers.index.astype(int).max() + 1)

    # Add new bus with additional attributes
    self.network.add("Bus", new_bus, carrier="AC", v_nom=220, x=8.998612, y=54.646649)
    self.network.buses.loc[new_bus, "scn_name"] = "eGon2035"
    self.network.buses.loc[new_bus, "country"] = "DE"
    
    # Set geometry for new bus
    point_bus1 = Point(8.998612, 54.646649)
    self.network.buses.at[new_bus, "geom"] = from_shape(point_bus1, srid=4326)
    
    # Function to add a 110 kV line
    def add_110kv_line(bus0, bus1, overhead=False):
        new_line = str(self.network.lines.index.astype(int).max() + 1)
        line_length = pypsa.geo.haversine(self.network.buses.loc[bus0, ["x", "y"]], self.network.buses.loc[bus1, ["x", "y"]])[0][0] * 1.2
        if not overhead:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=line_length, x=0.3e-3, s_nom=280, r=0.0177, b=250e-9, cables=3, carrier='AC')
            capital_cost = 230 * line_length
        else:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=line_length, x=1.2e-3, s_nom=260, r=0.05475, b=9.5e-9, cables=3, carrier='AC')
            capital_cost = 230 * line_length

        # Set additional attributes
        self.network.lines.loc[new_line, "scn_name"] = "eGon2035"
        self.network.lines.loc[new_line, "v_nom"] = 110
        self.network.lines.loc[new_line, "country"] = "DE"
        self.network.lines.loc[new_line, "version"] = "added_manually"
        self.network.lines.loc[new_line, "frequency"] = 50
        self.network.lines.loc[new_line, "length"] = (
            pypsa.geo.haversine(
                self.network.buses.loc[bus0, ["x", "y"]],
                self.network.buses.loc[bus1, ["x", "y"]],
            )[0][0]
            * 1.2
        )
        
        self.network.lines.loc[new_line, "capital_cost"] = capital_cost


    # Function to add a 220 kV line
    def add_220kv_line(bus0, bus1, overhead=False):
        new_line = str(self.network.lines.index.astype(int).max() + 1)
        line_length = pypsa.geo.haversine(self.network.buses.loc[bus0, ["x", "y"]], self.network.buses.loc[bus1, ["x", "y"]])[0][0] * 1.2
        if not overhead:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=line_length, x=0.3e-3, s_nom=550, r=0.0176, b=210e-9, cables=3, carrier='AC')
            capital_cost = 290 * line_length
        else:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=line_length, x=1e-3, s_nom=520, r=0.05475, b=11e-9, cables=3, carrier='AC')
            capital_cost = 290 * line_length

  

        # Set additional attributes
        self.network.lines.loc[new_line, "scn_name"] = "eGon2035"
        self.network.lines.loc[new_line, "v_nom"] = 220
        self.network.lines.loc[new_line, "country"] = "DE"
        self.network.lines.loc[new_line, "version"] = "added_manually"
        self.network.lines.loc[new_line, "frequency"] = 50
        self.network.lines.loc[new_line, "length"] = (
            pypsa.geo.haversine(
                self.network.buses.loc[bus0, ["x", "y"]],
                self.network.buses.loc[bus1, ["x", "y"]],
            )[0][0]
            * 1.2
        )
        
        self.network.lines.loc[new_line, "capital_cost"] = capital_cost


    # Add new transformer and line with additional attributes
    self.network.add("Transformer", new_trafo, bus0="32941", bus1=new_bus, x=1.29960, tap_ratio=1, s_nom=1600)
    add_110kv_line("32941", new_bus, overhead=False)

    

    print("New components added. Network now contains:")
    print(f"Buses: {len(self.network.buses)}")
    print(f"Transformers: {len(self.network.transformers)}")
    print(f"Lines: {len(self.network.lines)}")
    print(f"Generators: {len(self.network.generators)}")
    print(f"Loads: {len(self.network.loads)}")


      
    # Load generation time series from CSV
    time_series_data = pd.read_csv('data/generators1-p_max_pu.csv')
    pv_time_series = time_series_data['PV']
    biogas_time_series = time_series_data['KWK']

    # Determine the attributes for new generators by copying from similar existing generators
    default_attrs = ['start_up_cost', 'shut_down_cost', 'min_up_time', 'min_down_time', 'up_time_before', 'down_time_before', 'ramp_limit_up', 'ramp_limit_down', 'ramp_limit_start_up', 'ramp_limit_shut_down', 'e_nom_max']
    existing_solar = self.network.generators[self.network.generators.carrier == 'solar'].iloc[0]
    solar_attrs = {attr: existing_solar.get(attr, 0) for attr in default_attrs}

    existing_biogas = self.network.generators[self.network.generators.carrier == 'central_biomass_CHP_heat'].iloc[0]
    biogas_attrs = {attr: existing_biogas.get(attr, 0) for attr in default_attrs}

    # Add the solar and biogas generators with the new ID
    # Determine the next generator ID
    if not self.network.generators.empty:
        max_id = max(self.network.generators.index, key=lambda x: int(x) if x.isdigit() else -1)
        gen_id = str(int(max_id) + 1 if max_id.isdigit() else 1)
    else:
        gen_id = "1"

    # Add the solar generator with the new ID
    solar_gen_id = gen_id
    self.network.add("Generator", solar_gen_id, bus=new_bus, p_nom=2.0, carrier="solar", marginal_cost=0, 
                     capital_cost=1200, p_max_pu=1, **solar_attrs)

    # Add the biogas generator with the new ID
    biogas_gen_id = str(int(solar_gen_id) + 1)
    self.network.add("Generator", biogas_gen_id, bus=new_bus, p_nom=1.5, carrier="central_biomass_CHP_heat", marginal_cost=50, 
                     capital_cost=1000, p_max_pu=1, **biogas_attrs)
    
  
    self.network.generators.loc[solar_gen_id, "scn_name"] = "eGon2035"
    self.network.generators.loc[biogas_gen_id, "scn_name"] = "eGon2035"
    
    
    # Print updated p_max_pu time series
    print(f"Time series for Solar generator {solar_gen_id} added successfully.")
    print(f"Time series for Biogas generator {biogas_gen_id} added successfully.")
    print("Updated p_max_pu time series:")
    print(self.network.generators_t['p_max_pu'].head())
    
    
    # Initialize and populate time series dataframe for p_max_pu if not already
    if 'p_max_pu' not in self.network.generators_t:
        self.network.generators_t['p_max_pu'] = pd.DataFrame(0, index=self.network.snapshots, columns=self.network.generators.index)
    self.network.generators_t['p_max_pu'].loc[:, solar_gen_id] = pv_time_series.values[:len(self.network.snapshots)]
    self.network.generators_t['p_max_pu'].loc[:, biogas_gen_id] = biogas_time_series.values[:len(self.network.snapshots)]

    
   
    print(f"Time series for Solar generator {solar_gen_id} added successfully.")
    print(f"Time series for Biogas generator {biogas_gen_id} added successfully.")
    print("Updated p_max_pu time series:")
    print(self.network.generators_t['p_max_pu'].head())
    

 # Determine new load IDs
    load_ac_id = str(self.network.loads.index.astype(int).max() + 1)
    load_ev_id = str(int(load_ac_id) + 1)

    # Add loads
    self.network.add("Load", load_ac_id, bus=new_bus, carrier="AC", p_set=0, q_set=0, sign=-1)
    self.network.add("Load", load_ev_id, bus=new_bus, carrier="land transport EV", p_set=0, q_set=0, sign=-1)
    
    self.network.loads.loc[load_ac_id, "scn_name"] = "eGon2035"   
    self.network.loads.loc[load_ev_id, "scn_name"] = "eGon2035"

    # Load time series data from CSV file
    load_time_series = pd.read_csv('data/loads.csv')
    ac_load_series = load_time_series['AC load']
    ev_load_series = load_time_series['EV load']

    # Initialize time series data frame for loads if not already present
    if 'p_set' not in self.network.loads_t:
        self.network.loads_t['p_set'] = pd.DataFrame(0, index=self.network.snapshots, columns=self.network.loads.index)
    if 'q_set' not in self.network.loads_t:
        self.network.loads_t['q_set'] = pd.DataFrame(0, index=self.network.snapshots, columns=self.network.loads.index)

    # Add the time series data for the new loads
    self.network.loads_t['p_set'][load_ac_id] = ac_load_series.values[:len(self.network.snapshots)]
    self.network.loads_t['p_set'][load_ev_id] = ev_load_series.values[:len(self.network.snapshots)]

    print(f"AC Load ID: {load_ac_id} and EV Load ID: {load_ev_id} added to Bus {new_bus}")
    print("Time series data for loads has been successfully updated.")
    
    
    print("Buses after addition:", len(self.network.buses))
    print("Transformers after addition:", len(self.network.transformers))
    print("Lines after addition:", len(self.network.lines))
    print("Generators after addition:", len(self.network.generators))
    print("Loads after addition:", len(self.network.loads))



#This works:

def add_EC_to_network(self):
    """Adds Energy Community to the network."""
    from geoalchemy2.shape import from_shape, to_shape
    from shapely.geometry import LineString, MultiLineString, Point
 
    def add_110kv_line(bus0, bus1, overhead=False):
        new_line = str(self.network.lines.index.astype(int).max() + 1)
        if not overhead:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=10, x=0.3e-3, s_nom=280, r=0.0177, b=250e-9, cables=3, carrier='AC')
        else:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=10, x=1.2e-3, s_nom=260, r=0.05475, b=9.5e-9, cables=3, carrier='AC')

        # Set additional attributes
        self.network.lines.loc[new_line, "scn_name"] = "eGon2035"
        self.network.lines.loc[new_line, "v_nom"] = 110
        self.network.lines.loc[new_line, "country"] = "DE"
        self.network.lines.loc[new_line, "version"] = "added_manually"
        self.network.lines.loc[new_line, "frequency"] = 50
        self.network.lines.loc[new_line, "length"] = (
            pypsa.geo.haversine(
                self.network.buses.loc[bus0, ["x", "y"]],
                self.network.buses.loc[bus1, ["x", "y"]],
            )[0][0]
            * 1.2
        )
    
    
    def add_220kv_line(bus0, bus1, overhead=False):
        new_line = str(self.network.lines.index.astype(int).max() + 1)
        if not overhead:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=10, x=0.3e-3, s_nom=550, r=0.0176, b=210e-9, cables=3, carrier='AC')
        else:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=10, x=1e-3, s_nom=520, r=0.05475, b=11e-9, cables=3, carrier='AC')

        # Set additional attributes
        self.network.lines.loc[new_line, "scn_name"] = "eGon2035"
        self.network.lines.loc[new_line, "v_nom"] = 220
        self.network.lines.loc[new_line, "country"] = "DE"
        self.network.lines.loc[new_line, "version"] = "added_manually"
        self.network.lines.loc[new_line, "frequency"] = 50
        self.network.lines.loc[new_line, "length"] = (
            pypsa.geo.haversine(
                self.network.buses.loc[bus0, ["x", "y"]],
                self.network.buses.loc[bus1, ["x", "y"]],
            )[0][0]
            * 1.2
        )
    
    
    print("Buses before addition:", len(self.network.buses))
    print("Transformers before addition:", len(self.network.transformers))
    print("Lines before addition:", len(self.network.lines))
    print("Generators before addition:", len(self.network.generators))
    print("Loads before addition:", len(self.network.loads))

    
    # Generate new component IDs
    new_bus = str(self.network.buses.index.astype(np.int64).max() + 1)
    new_trafo = str(self.network.transformers.index.astype(np.int64).max() + 1)
    new_line = str(self.network.lines.index.astype(np.int64).max() + 1)

    # Add new bus with additional attributes
    self.network.add("Bus", new_bus, carrier="AC", v_nom=220, x=8.998612, y=54.646649)
    self.network.buses.loc[new_bus, "scn_name"] = "eGon2035"
    self.network.buses.loc[new_bus, "country"] = "DE"

    # Print the added bus details
    print(f"New Bus Added: ID={new_bus}, SCN Name={self.network.buses.at[new_bus, 'scn_name']}, Country={self.network.buses.at[new_bus, 'country']}")

    # Add new transformer and line with additional attributes
    self.network.add("Transformer", new_trafo, bus0="32941", bus1=new_bus, x=1.29960, tap_ratio=1, s_nom=1600)
    #self.network.add("Line", new_line, bus0="32941", bus1=new_bus, x=0.0001, s_nom=1600)
    add_220kv_line("32941", new_bus, overhead=False)
    #self.network.lines.loc[new_line, "cables"] = 3.0
    #self.network.lines.loc[new_line, "scn_name"] = "eGon2035"
    #self.network.lines.loc[new_line, "country"] = "DE"

    # Print the added transformer and line details
    print(f"New Transformer Added: ID={new_trafo}, Bus0=32941, Bus1={new_bus}")
    print(f"New Line Added: ID={new_line}, Bus0=32941, Bus1={new_bus}, Cables={self.network.lines.at[new_line, 'cables']}, SCN Name={self.network.lines.at[new_line, 'scn_name']}, Country={self.network.lines.at[new_line, 'country']}")

    # check the geometries 
    point_bus1 = Point(8.998612, 54.646649)
    self.network.buses.at[new_bus, "geom"] = from_shape(point_bus1, srid=4326)
    self.network.lines.at[new_line, "geom"] = from_shape(MultiLineString([LineString([to_shape(self.network.buses.at["32941", "geom"]), point_bus1])]), srid=4326)
    
    print(f"Geometry for new bus set. Bus {new_bus} location: {point_bus1}")


   
    # Load generation time series from CSV
    time_series_data = pd.read_csv('data/generators1-p_max_pu.csv')
    pv_time_series = time_series_data['PV']
    biogas_time_series = time_series_data['KWK']

    # Determine the attributes for new generators by copying from similar existing generators
    default_attrs = ['start_up_cost', 'shut_down_cost', 'min_up_time', 'min_down_time', 'up_time_before', 'down_time_before', 'ramp_limit_up', 'ramp_limit_down', 'ramp_limit_start_up', 'ramp_limit_shut_down', 'e_nom_max']
    existing_solar = self.network.generators[self.network.generators.carrier == 'solar'].iloc[0]
    solar_attrs = {attr: existing_solar.get(attr, 0) for attr in default_attrs}

    existing_biogas = self.network.generators[self.network.generators.carrier == 'central_biomass_CHP_heat'].iloc[0]
    biogas_attrs = {attr: existing_biogas.get(attr, 0) for attr in default_attrs}

    # Add the solar and biogas generators with the new ID
    # Determine the next generator ID
    if not self.network.generators.empty:
        max_id = max(self.network.generators.index, key=lambda x: int(x) if x.isdigit() else -1)
        gen_id = str(int(max_id) + 1 if max_id.isdigit() else 1)
    else:
        gen_id = "1"

    # Add the solar generator with the new ID
    solar_gen_id = gen_id
    self.network.add("Generator", solar_gen_id, bus=new_bus, p_nom=2.0, carrier="solar", marginal_cost=0, 
                     capital_cost=1200, p_max_pu=1, **solar_attrs)

    # Add the biogas generator with the new ID
    biogas_gen_id = str(int(solar_gen_id) + 1)
    self.network.add("Generator", biogas_gen_id, bus=new_bus, p_nom=1.5, carrier="central_biomass_CHP_heat", marginal_cost=50, 
                     capital_cost=1000, p_max_pu=1, **biogas_attrs)
    
  
    self.network.generators.loc[solar_gen_id, "scn_name"] = "eGon2035"
    self.network.generators.loc[biogas_gen_id, "scn_name"] = "eGon2035"
    
    
    # Print updated p_max_pu time series
    print(f"Time series for Solar generator {solar_gen_id} added successfully.")
    print(f"Time series for Biogas generator {biogas_gen_id} added successfully.")
    print("Updated p_max_pu time series:")
    print(self.network.generators_t['p_max_pu'].head())
    
    
    # Initialize and populate time series dataframe for p_max_pu if not already
    if 'p_max_pu' not in self.network.generators_t:
        self.network.generators_t['p_max_pu'] = pd.DataFrame(0, index=self.network.snapshots, columns=self.network.generators.index)
    self.network.generators_t['p_max_pu'].loc[:, solar_gen_id] = pv_time_series.values[:len(self.network.snapshots)]
    self.network.generators_t['p_max_pu'].loc[:, biogas_gen_id] = biogas_time_series.values[:len(self.network.snapshots)]

    
   
    print(f"Time series for Solar generator {solar_gen_id} added successfully.")
    print(f"Time series for Biogas generator {biogas_gen_id} added successfully.")
    print("Updated p_max_pu time series:")
    print(self.network.generators_t['p_max_pu'].head())
    

 # Determine new load IDs
    load_ac_id = str(self.network.loads.index.astype(int).max() + 1)
    load_ev_id = str(int(load_ac_id) + 1)

    # Add loads
    self.network.add("Load", load_ac_id, bus=new_bus, carrier="AC", p_set=0, q_set=0, sign=-1)
    self.network.add("Load", load_ev_id, bus=new_bus, carrier="land transport EV", p_set=0, q_set=0, sign=-1)
    
    self.network.loads.loc[load_ac_id, "scn_name"] = "eGon2035"   
    self.network.loads.loc[load_ev_id, "scn_name"] = "eGon2035"

    # Load time series data from CSV file
    load_time_series = pd.read_csv('data/loads.csv')
    ac_load_series = load_time_series['AC load']
    ev_load_series = load_time_series['EV load']

    # Initialize time series data frame for loads if not already present
    if 'p_set' not in self.network.loads_t:
        self.network.loads_t['p_set'] = pd.DataFrame(0, index=self.network.snapshots, columns=self.network.loads.index)
    if 'q_set' not in self.network.loads_t:
        self.network.loads_t['q_set'] = pd.DataFrame(0, index=self.network.snapshots, columns=self.network.loads.index)

    # Add the time series data for the new loads
    self.network.loads_t['p_set'][load_ac_id] = ac_load_series.values[:len(self.network.snapshots)]
    self.network.loads_t['p_set'][load_ev_id] = ev_load_series.values[:len(self.network.snapshots)]

    print(f"AC Load ID: {load_ac_id} and EV Load ID: {load_ev_id} added to Bus {new_bus}")
    print("Time series data for loads has been successfully updated.")
    
    
    print("Buses after addition:", len(self.network.buses))
    print("Transformers after addition:", len(self.network.transformers))
    print("Lines after addition:", len(self.network.lines))
    print("Generators after addition:", len(self.network.generators))
    print("Loads after addition:", len(self.network.loads))



def add_EC_to_network(self):
    """Adds Energy Community to the network."""
    from geoalchemy2.shape import from_shape, to_shape
    from shapely.geometry import LineString, MultiLineString, Point
 
     
        
    # Function to add a 110 kV line
    def add_110kv_line(bus0, bus1, overhead=False):
        new_line = str(self.network.lines.index.astype(int).max() + 1)
        line_length = pypsa.geo.haversine(self.network.buses.loc[bus0, ["x", "y"]], self.network.buses.loc[bus1, ["x", "y"]])[0][0] * 1.2
        if not overhead:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=line_length, x=0.3e-3, s_nom=280, r=0.0177, b=250e-9, cables=3, carrier='AC')
            capital_cost = 230 * line_length
        else:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=line_length, x=1.2e-3, s_nom=260, r=0.05475, b=9.5e-9, cables=3, carrier='AC')
            capital_cost = 230 * line_length

        # Set additional attributes
        self.network.lines.loc[new_line, "scn_name"] = "eGon2035"
        self.network.lines.loc[new_line, "v_nom"] = 110
        self.network.lines.loc[new_line, "country"] = "DE"
        self.network.lines.loc[new_line, "version"] = "added_manually"
        self.network.lines.loc[new_line, "frequency"] = 50
        self.network.lines.loc[new_line, "length"] = (
            pypsa.geo.haversine(
                self.network.buses.loc[bus0, ["x", "y"]],
                self.network.buses.loc[bus1, ["x", "y"]],
            )[0][0]
            * 1.2
        )
        
        self.network.lines.loc[new_line, "capital_cost"] = capital_cost


    # Function to add a 220 kV line
    def add_220kv_line(bus0, bus1, overhead=False):
        new_line = str(self.network.lines.index.astype(int).max() + 1)
        line_length = pypsa.geo.haversine(self.network.buses.loc[bus0, ["x", "y"]], self.network.buses.loc[bus1, ["x", "y"]])[0][0] * 1.2
        if not overhead:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=line_length, x=0.3e-3, s_nom=550, r=0.0176, b=210e-9, cables=3, carrier='AC')
            capital_cost = 290 * line_length
        else:
            self.network.add("Line", new_line, bus0=bus0, bus1=bus1, length=line_length, x=1e-3, s_nom=520, r=0.05475, b=11e-9, cables=3, carrier='AC')
            capital_cost = 290 * line_length

  

        # Set additional attributes
        self.network.lines.loc[new_line, "scn_name"] = "eGon2035"
        self.network.lines.loc[new_line, "v_nom"] = 220
        self.network.lines.loc[new_line, "country"] = "DE"
        self.network.lines.loc[new_line, "version"] = "added_manually"
        self.network.lines.loc[new_line, "frequency"] = 50
        self.network.lines.loc[new_line, "length"] = (
            pypsa.geo.haversine(
                self.network.buses.loc[bus0, ["x", "y"]],
                self.network.buses.loc[bus1, ["x", "y"]],
            )[0][0]
            * 1.2
        )
        
        self.network.lines.loc[new_line, "capital_cost"] = capital_cost
    
    
    print("Buses before addition:", len(self.network.buses))
    print("Transformers before addition:", len(self.network.transformers))
    print("Lines before addition:", len(self.network.lines))
    print("Generators before addition:", len(self.network.generators))
    print("Loads before addition:", len(self.network.loads))

    
    # Generate new component IDs
    new_bus = str(self.network.buses.index.astype(np.int64).max() + 1)
    new_trafo = str(self.network.transformers.index.astype(np.int64).max() + 1)
    new_line = str(self.network.lines.index.astype(np.int64).max() + 1)

    # Add new bus with additional attributes
    self.network.add("Bus", new_bus, carrier="AC", v_nom=220, x=8.998612, y=54.646649)
    self.network.buses.loc[new_bus, "scn_name"] = "eGon2035"
    self.network.buses.loc[new_bus, "country"] = "DE"

    # Print the added bus details
    print(f"New Bus Added: ID={new_bus}, SCN Name={self.network.buses.at[new_bus, 'scn_name']}, Country={self.network.buses.at[new_bus, 'country']}")

    # Add new transformer and line with additional attributes
    self.network.add("Transformer", new_trafo, bus0="32941", bus1=new_bus, x=1.29960, tap_ratio=1, s_nom=1600)
    #self.network.add("Line", new_line, bus0="32941", bus1=new_bus, x=0.0001, s_nom=1600)
    add_220kv_line("32941", new_bus, overhead=False)
    #self.network.lines.loc[new_line, "cables"] = 3.0
    #self.network.lines.loc[new_line, "scn_name"] = "eGon2035"
    #self.network.lines.loc[new_line, "country"] = "DE"

    # Print the added transformer and line details
    print(f"New Transformer Added: ID={new_trafo}, Bus0=32941, Bus1={new_bus}")
    print(f"New Line Added: ID={new_line}, Bus0=32941, Bus1={new_bus}, Cables={self.network.lines.at[new_line, 'cables']}, SCN Name={self.network.lines.at[new_line, 'scn_name']}, Country={self.network.lines.at[new_line, 'country']}")

    # check the geometries 
    point_bus1 = Point(8.998612, 54.646649)
    self.network.buses.at[new_bus, "geom"] = from_shape(point_bus1, srid=4326)
    self.network.lines.at[new_line, "geom"] = from_shape(MultiLineString([LineString([to_shape(self.network.buses.at["32941", "geom"]), point_bus1])]), srid=4326)
    
    print(f"Geometry for new bus set. Bus {new_bus} location: {point_bus1}")


   
    # Load generation time series from CSV
    time_series_data = pd.read_csv('data/generators1-p_max_pu.csv')
    pv_time_series = time_series_data['PV']
    biogas_time_series = time_series_data['KWK']

    # Determine the attributes for new generators by copying from similar existing generators
    default_attrs = ['start_up_cost', 'shut_down_cost', 'min_up_time', 'min_down_time', 'up_time_before', 'down_time_before', 'ramp_limit_up', 'ramp_limit_down', 'ramp_limit_start_up', 'ramp_limit_shut_down', 'e_nom_max']
    existing_solar = self.network.generators[self.network.generators.carrier == 'solar'].iloc[0]
    solar_attrs = {attr: existing_solar.get(attr, 0) for attr in default_attrs}

    existing_biogas = self.network.generators[self.network.generators.carrier == 'central_biomass_CHP_heat'].iloc[0]
    biogas_attrs = {attr: existing_biogas.get(attr, 0) for attr in default_attrs}

    # Add the solar and biogas generators with the new ID
    # Determine the next generator ID
    if not self.network.generators.empty:
        max_id = max(self.network.generators.index, key=lambda x: int(x) if x.isdigit() else -1)
        gen_id = str(int(max_id) + 1 if max_id.isdigit() else 1)
    else:
        gen_id = "1"

    # Add the solar generator with the new ID
    solar_gen_id = gen_id
    self.network.add("Generator", solar_gen_id, bus=new_bus, p_nom=2.0, carrier="solar", marginal_cost=0, 
                     capital_cost=1200, p_max_pu=1, **solar_attrs)

    # Add the biogas generator with the new ID
    biogas_gen_id = str(int(solar_gen_id) + 1)
    self.network.add("Generator", biogas_gen_id, bus=new_bus, p_nom=1.5, carrier="central_biomass_CHP_heat", marginal_cost=50, 
                     capital_cost=1000, p_max_pu=1, **biogas_attrs)
    
  
    self.network.generators.loc[solar_gen_id, "scn_name"] = "eGon2035"
    self.network.generators.loc[biogas_gen_id, "scn_name"] = "eGon2035"
    
    
    # Print updated p_max_pu time series
    print(f"Time series for Solar generator {solar_gen_id} added successfully.")
    print(f"Time series for Biogas generator {biogas_gen_id} added successfully.")
    print("Updated p_max_pu time series:")
    print(self.network.generators_t['p_max_pu'].head())
    
    
    # Initialize and populate time series dataframe for p_max_pu if not already
    if 'p_max_pu' not in self.network.generators_t:
        self.network.generators_t['p_max_pu'] = pd.DataFrame(0, index=self.network.snapshots, columns=self.network.generators.index)
    self.network.generators_t['p_max_pu'].loc[:, solar_gen_id] = pv_time_series.values[:len(self.network.snapshots)]
    self.network.generators_t['p_max_pu'].loc[:, biogas_gen_id] = biogas_time_series.values[:len(self.network.snapshots)]

    
   
    print(f"Time series for Solar generator {solar_gen_id} added successfully.")
    print(f"Time series for Biogas generator {biogas_gen_id} added successfully.")
    print("Updated p_max_pu time series:")
    print(self.network.generators_t['p_max_pu'].head())
    

 # Determine new load IDs
    load_ac_id = str(self.network.loads.index.astype(int).max() + 1)
    load_ev_id = str(int(load_ac_id) + 1)

    # Add loads
    self.network.add("Load", load_ac_id, bus=new_bus, carrier="AC", p_set=0, q_set=0, sign=-1)
    self.network.add("Load", load_ev_id, bus=new_bus, carrier="land transport EV", p_set=0, q_set=0, sign=-1)
    
    self.network.loads.loc[load_ac_id, "scn_name"] = "eGon2035"   
    self.network.loads.loc[load_ev_id, "scn_name"] = "eGon2035"

    # Load time series data from CSV file
    load_time_series = pd.read_csv('data/loads.csv')
    ac_load_series = load_time_series['AC load']
    ev_load_series = load_time_series['EV load']

    # Initialize time series data frame for loads if not already present
    if 'p_set' not in self.network.loads_t:
        self.network.loads_t['p_set'] = pd.DataFrame(0, index=self.network.snapshots, columns=self.network.loads.index)
    if 'q_set' not in self.network.loads_t:
        self.network.loads_t['q_set'] = pd.DataFrame(0, index=self.network.snapshots, columns=self.network.loads.index)

    # Add the time series data for the new loads
    self.network.loads_t['p_set'][load_ac_id] = ac_load_series.values[:len(self.network.snapshots)]
    self.network.loads_t['p_set'][load_ev_id] = ev_load_series.values[:len(self.network.snapshots)]

    print(f"AC Load ID: {load_ac_id} and EV Load ID: {load_ev_id} added to Bus {new_bus}")
    print("Time series data for loads has been successfully updated.")
    
    
    print("Buses after addition:", len(self.network.buses))
    print("Transformers after addition:", len(self.network.transformers))
    print("Lines after addition:", len(self.network.lines))
    print("Generators after addition:", len(self.network.generators))
    print("Loads after addition:", len(self.network.loads))
