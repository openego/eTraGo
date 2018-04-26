DO $$
DECLARE
    rec RECORD;
    new_grid_res INTEGER;
    
BEGIN
	
	FOR rec in SELECT result_id from model_draft.ego_grid_pf_hv_result_meta WHERE safe_results = TRUE 
	
	LOOP
		new_grid_res = 	CASE WHEN (SELECT min(result_id) from grid.ego_pf_hv_result_meta) IS NULL
						THEN 1
						ELSE (SELECT max(result_id)+1 from grid.ego_pf_hv_result_meta)
						END;
	
	
		INSERT INTO grid.ego_pf_hv_result_meta
			(result_id,
			modeldraft_id ,
			scn_name,
			calc_date,
			user_name,
			method,
			start_snapshot,
			end_snapshot,
			snapshots,
			solver,
			settings)
		SELECT 
			new_grid_res,
			rec.result_id,
			scn_name,
			calc_date,
			user_name,
			method,
			start_snapshot,
			end_snapshot,
			snapshots,
			solver,
			settings
		FROM model_draft.ego_grid_pf_hv_result_meta
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_bus
			(result_id,
			bus_id ,
			x,
			y,
			v_nom,
			current_type,
			v_mag_pu_min,
			v_mag_pu_max,
			geom)
		SELECT 
			new_grid_res,
			bus_id ,
			x,
			y,
			v_nom,
			current_type,
			v_mag_pu_min,
			v_mag_pu_max,
			geom
		FROM model_draft.ego_grid_pf_hv_result_bus
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_bus_t
			(result_id,
			bus_id,
			v_mag_pu_set,
			p,
			q,
			v_mag_pu,
			v_ang,
			marginal_price)
		SELECT 
			new_grid_res,
			bus_id,
			v_mag_pu_set,
			p,
			q,
			v_mag_pu,
			v_ang,
			marginal_price
		FROM model_draft.ego_grid_pf_hv_result_bus_t
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_generator
			(result_id,
			generator_id,
			bus,
			dispatch,
			control,
			p_nom,
			p_nom_extendable,
			p_nom_min,
			p_nom_max,
			p_min_pu_fixed,
			p_max_pu_fixed,
			sign,
			source,
			marginal_cost,
			capital_cost,
			efficiency,
			p_nom_opt)
		SELECT 
			new_grid_res,
			generator_id,
			bus,
			dispatch,
			control,
			p_nom,
			p_nom_extendable,
			p_nom_min,
			p_nom_max,
			p_min_pu_fixed,
			p_max_pu_fixed,
			sign,
			source,
			marginal_cost,
			capital_cost,
			efficiency,
			p_nom_opt
		FROM model_draft.ego_grid_pf_hv_result_generator
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_generator_t
			(result_id,
			generator_id,
			p_set,
			q_set,
			p_min_pu,
			p_max_pu,
			p,
			q,
			status)
		SELECT
			new_grid_res,
			generator_id,
			p_set,
			q_set,
			p_min_pu,
			p_max_pu,
			p,
			q,
			status
		FROM model_draft.ego_grid_pf_hv_result_generator_t
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_line
			(result_id,
			line_id,
			bus0,
			bus1,
			x,
			r,
			g,
			b,
			s_nom,
			s_nom_extendable,
			s_nom_min,
			s_nom_max,
			capital_cost,
			length,
			cables,
			frequency,
			terrain_factor,
			x_pu,
			r_pu,
			g_pu,
			b_pu,
			s_nom_opt,
			geom,
			topo)
		SELECT 
			new_grid_res,
			line_id,
			bus0,
			bus1,
			x,
			r,
			g,
			b,
			s_nom,
			s_nom_extendable,
			s_nom_min,
			s_nom_max,
			capital_cost,
			length,
			cables,
			frequency,
			terrain_factor,
			x_pu,
			r_pu,
			g_pu,
			b_pu,
			s_nom_opt,
			geom,
			topo
		FROM model_draft.ego_grid_pf_hv_result_line
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_line_t
			(result_id,
			line_id,
			p0,
			q0,
			p1,
			q1)
		SELECT 
			new_grid_res,
			line_id,
			p0,
			q0,
			p1,
			q1
		FROM model_draft.ego_grid_pf_hv_result_line_t
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_load
			(result_id,
			load_id,
			bus,
			sign,
			e_annual)
		SELECT 
			new_grid_res,
			load_id,
			bus,
			sign,
			e_annual
		FROM model_draft.ego_grid_pf_hv_result_load
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_load_t
			(result_id,
			load_id,
			p_set,
			q_set,
			p,
			q)
		SELECT 
			new_grid_res,
			load_id,
			p_set,
			q_set,
			p,
			q
		FROM model_draft.ego_grid_pf_hv_result_load_t
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_storage
			(result_id,
			storage_id,
			bus,
			dispatch,
			control,
			p_nom,
			p_nom_extendable,
			p_nom_min,
			p_nom_max,
			p_min_pu_fixed,
			p_max_pu_fixed,
			sign,
			source,
			marginal_cost,
			capital_cost,
			efficiency,
			soc_initial,
			soc_cyclic,
			max_hours,
			efficiency_store,
			efficiency_dispatch,
			standing_loss,
			p_nom_opt)
		SELECT 
			new_grid_res,
			storage_id,
			bus,
			dispatch,
			control,
			p_nom,
			p_nom_extendable,
			p_nom_min,
			p_nom_max,
			p_min_pu_fixed,
			p_max_pu_fixed,
			sign,
			source,
			marginal_cost,
			capital_cost,
			efficiency,
			soc_initial,
			soc_cyclic,
			max_hours,
			efficiency_store,
			efficiency_dispatch,
			standing_loss,
			p_nom_opt
		FROM model_draft.ego_grid_pf_hv_result_storage
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_storage_t
			(result_id,
			storage_id,
			p_set,
			q_set,
			p_min_pu,
			p_max_pu,
			soc_set,
			inflow,
			p,
			q,
			state_of_charge,
			spill)
		SELECT 
			new_grid_res,
			storage_id,
			p_set,
			q_set,
			p_min_pu,
			p_max_pu,
			soc_set,
			inflow,
			p,
			q,
			state_of_charge,
			spill
		FROM model_draft.ego_grid_pf_hv_result_storage_t
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_transformer
			(result_id,
			trafo_id,
			bus0,
			bus1,
			x,
			r,
			g,
			b,
			s_nom,
			s_nom_extendable,
			s_nom_min,
			s_nom_max,
			tap_ratio,
			phase_shift,
			capital_cost,
			x_pu,
			r_pu,
			g_pu,
			b_pu,
			s_nom_opt,
			geom,
			topo)
		SELECT 
			new_grid_res,
			trafo_id,
			bus0,
			bus1,
			x,
			r,
			g,
			b,
			s_nom,
			s_nom_extendable,
			s_nom_min,
			s_nom_max,
			tap_ratio,
			phase_shift,
			capital_cost,
			x_pu,
			r_pu,
			g_pu,
			b_pu,
			s_nom_opt,
			geom,
			topo
		FROM model_draft.ego_grid_pf_hv_result_transformer
		WHERE result_id = rec.result_id;

		INSERT INTO grid.ego_pf_hv_result_transformer_t
			(result_id,
			trafo_id,
			p0,
			q0,
			p1,
			q1)
		SELECT 
			new_grid_res,
			trafo_id,
			p0,
			q0,
			p1,
			q1
		FROM model_draft.ego_grid_pf_hv_result_transformer_t
		WHERE result_id = rec.result_id;
 
		UPDATE model_draft.ego_grid_pf_hv_result_meta
		SET safe_results = FALSE
		WHERE result_id = rec.result_id; 

	END LOOP;
	
END; $$

