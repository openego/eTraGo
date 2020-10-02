import pandas as pd

def ptg_links_clustering(n_clusters):
    filename_1 = 'kmeans_busmap_'+str(n_clusters)+'_result.csv'
    filename_2 = 'PtG_implementation/max_capacities_subst_h2_MW.csv'
    # filename_correspondances = "PtG_implementation/correspondances.csv"
    
    list_capacities=[]
    list_names=[]
    df_correspondance = pd.read_csv(filename_1, index_col ='bus_id')
    df_orginal_capacities = pd.read_csv(filename_2)
    new_column_list = []
    
    for index, row in df_orginal_capacities.iterrows():
        cell_content = row["otg_id"]
        new_cell = df_correspondance.at[int(cell_content),"clustered_bus_ID"]
        new_column_list.append(new_cell)
    
    df_orginal_capacities["clustered_bus_ID"] = new_column_list
    # df_orginal_capacities.to_csv(filename_correspondances)
    
    for i in range(n_clusters):
        df_tmp = df_orginal_capacities[df_orginal_capacities["clustered_bus_ID"] == i]
        capacity = df_tmp['PtH2_max_inst_capacity_MWel'].sum()  
        list_capacities.append(capacity)
        list_names.append("Link_"+str(i))
          
    df = pd.DataFrame({'name':list_names,
                       'bus0': range(n_clusters),
                       'p_nom_max': list_capacities}) 
    df.set_index('name')
    df['bus1']="Gas_Bus"
    df['efficiency']=0.8 # Brown et al. 2018 "SynergSynergies of sector coupling and transmission reinforcement in a cost-optimised, highly renewable European energy system", p.4
    df['capital_cost']=350000 # Brown et al. 2018 "SynergSynergies of sector coupling and transmission reinforcement in a cost-optimised, highly renewable European energy system", p.4
    df['p_nom']=0
    df['p_nom_min']=0
    df['p_nom_extendable']="True"
    df['marginal_cost']=0
    df['p_min_pu']=0
    df['p_max_pu']=1
    df['version']="0.4.5"
    df['scn_name']="NEP 2035"
    
    return df

def ptg_links_ST_pu_clustering(n_clusters):
    filename_1 = 'kmeans_busmap_'+str(n_clusters)+'_result.csv'
    filename_2 = "PtG_implementation/time_series_sum_up_h2_MW.csv"
    
    df_correspondance = pd.read_csv(filename_1, index_col ='bus_id')
    df_orginal_ST = pd.read_csv(filename_2, index_col ='time')
    
    original_names = df_orginal_ST.columns.tolist()
    new_names = []
    
    for i in original_names:
        new_cell = df_correspondance.at[int(i),"clustered_bus_ID"]
        new_names.append(new_cell)
    
    df_orginal_ST.loc["new_names"] = new_names
    df_T = df_orginal_ST.T
    df = pd.DataFrame()
    
    for i in range(n_clusters): 
        df_tmp = df_T[df_T["new_names"] == i]
        ST = df_tmp.sum()
        ST = ST.drop('new_names')
        C_max = ST.max()
        if C_max == 0:
            df['Link_'+str(i)] = ST
        else:
            df['Link_'+str(i)] = ST/C_max

    return df

def ptg_addition(network, n_clusters):
    # add gas bus
    network.add("Bus",
                  "Gas_Bus", 
                  carrier="AC",
                  v_nom=380)
        
    # add links
    df = ptg_links_clustering(n_clusters)
    df = df.set_index('name')
    network.import_components_from_dataframe(df, "Link")
    
    # add p_max_pu TS to gas links    
    df_t = ptg_links_ST_pu_clustering(n_clusters)
    df_t.index = pd.DatetimeIndex(df_t.index)
    network.import_series_from_dataframe(df_t, 
                                            "Link",
                                            "p_max_pu")
    
    # add gas store
    network.add("Store",
                "Gas_Store",
                bus = "Gas_Bus",
                e_cyclic = True)
    
    # add H2 load
    P_grid_oriented_installations_H2 = 3000 # [MWel] NEP_2035_v_2021_Szenariorahmen_2035_Entwurf , p.52, Scenario C, netzdienliche Power-to-Hydrogen Anlagen
    n_Full_load_hours_H2 = 1500 # NEP_2035_v_2021_Szenariorahmen_2035_Entwurf , p.52, Scenario C, netzdienliche Power-to-Hydrogen Anlagen
    n_year_hours = 8760
    E_year = P_grid_oriented_installations_H2*n_Full_load_hours_H2
    network.add("Load",
                "Gas_Load",
                bus = "Gas_Bus",
                p_set = E_year/n_year_hours)  
    