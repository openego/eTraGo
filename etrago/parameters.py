"""
Pydantic models for eTraGo grid calculation arguments.

"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from copy import deepcopy

from collections.abc import Mapping

import os
# ---------------------------------------------------------------------------
# Sub-models: Method
# ---------------------------------------------------------------------------


class RollingHorizon(BaseModel):
    """Parameters controlling the rolling-horizon market optimisation window."""

    planning_horizon: int = Field(
        168,
        description="Number of snapshots included in each optimisation window.",
    )
    overlap: int = Field(
        120,
        description="Number of overlapping hours between consecutive windows.",
    )


class MarketOptimization(BaseModel):
    """
    Settings for the optional separate market optimisation performed before
    the grid optimisation.  When inactive, an integrated optimisation is run
    instead.
    """

    active: bool = Field(
        True,
        description=(
            "If True, a separate market optimisation is performed before the "
            "grid optimisation."
        ),
    )
    market_zones: str = Field(
        "status_quo",
        description="Market zone definition; only used when method type is 'market_grid'.",
    )
    rolling_horizon: RollingHorizon = Field(default_factory=RollingHorizon)
    redispatch: bool = Field(
        True,
        description="If True, redispatch is performed after the market optimisation.",
    )


class Method(BaseModel):
    """
    Method and settings for the network optimisation.

    Attributes
    ----------
    type : str
        Type of optimisation to perform.  Options: ``'lopf'``, ``'sclopf'``,
        or ``'market_grid'``.  Default: ``'lopf'``.
    n_iter : int
        In case of extendable lines, several LOPFs must be performed.  Set
        ``n_iter`` to specify a fixed number of iterations, or use
        ``'threshold'`` as the abort criterion instead.  Default: ``4``.
    formulation : str
        Model-building formulation.  Options: ``'pyomo'`` or ``'linopy'``.
        Default: ``'linopy'``.
    market_optimization : MarketOptimization
        Settings for an optional separate market optimisation.  When active,
        it is run before the grid optimisation; otherwise an integrated
        optimisation is performed.
    distribution_grids : bool or str
        If simplified distribution grids should be considered within the
        transmission grid optimisation, provide a path to the parameters for
        each distribution grid here.  Set to ``False`` to disable.
        Default: ``False``.
    """

    type: Literal["lopf", "sclopf", "market_grid"] = Field(
        "lopf",
        description="Type of optimisation: 'lopf', 'sclopf', or 'market_grid'.",
    )
    n_iter: int = Field(
        4,
        ge=1,
        description=(
            "Fixed number of LOPF iterations used as the abort criterion "
            "for extendable-line optimisation."
        ),
    )
    formulation: Literal["pyomo", "linopy"] = Field(
        "linopy",
        description="Model-building formulation backend: 'pyomo' or 'linopy'.",
    )
    market_optimization: MarketOptimization = Field(default_factory=MarketOptimization)
    distribution_grids: Union[bool, str] = Field(
        False,
        description=(
            "False, or path to the distribution-grid parameter file used to "
            "include simplified distribution grids in the transmission optimisation."
        ),
    )


# ---------------------------------------------------------------------------
# Sub-models: Power Flow post LOPF
# ---------------------------------------------------------------------------


class PfPostLopf(BaseModel):
    """
    Settings for running a non-linear power flow (PF) directly after the
    linear optimal power flow (LOPF) dispatch optimisation.

    Attributes
    ----------
    active : bool
        If True, a PF is performed after the LOPF.  Default: ``True``.
    add_foreign_lopf : bool
        If foreign lines are modelled as DC-links (see ``foreign_lines``),
        setting this to True retains the LOPF results for those links.
        Default: ``True``.
    q_allocation : str
        Strategy for allocating reactive power to all generators at the same
        bus.  Options: ``'p_nom'`` or ``'p'``.  Default: ``'p_nom'``.
    """

    active: bool = Field(
        False,
        description="If True, a non-linear power flow is run after the LOPF.",
    )
    add_foreign_lopf: bool = Field(
        True,
        description=(
            "Retain LOPF results for foreign DC-links when foreign lines are "
            "modelled as DC-links."
        ),
    )
    q_allocation: Literal["p_nom", "p"] = Field(
        "p_nom",
        description=(
            "Reactive-power allocation strategy for generators at the same bus: "
            "'p_nom' (by nominal power) or 'p' (by active power dispatch)."
        ),
    )


# ---------------------------------------------------------------------------
# Sub-models: Extendable / Grid expansion
# ---------------------------------------------------------------------------


class VoltageLevel(BaseModel):
    """Absolute capacity specification for a single voltage level."""

    i: int = Field(..., description="Current rating [A].")
    wires: int = Field(..., description="Number of wires per circuit.")
    circuits: int = Field(..., description="Maximum number of circuits.")


class UpperBoundsGrid(BaseModel):
    """
    Upper bounds for electrical grid expansion, separately for domestic lines
    and border-crossing lines.

    Attributes
    ----------
    grid_max_D : float or None
        Upper bound for domestic (German) grid expansion relative to existing
        capacity [p.u.].  Mutually exclusive with ``grid_max_abs_D``.
        Default: ``None`` (``grid_max_abs_D`` is used instead).
    grid_max_abs_D : dict or None
        Absolute maximum capacity between two electrical buses per voltage
        level [kV] for lines in Germany.  Keys are voltage levels
        (``'380'``, ``'220'``, ``'110'``, ``'dc'``); values are
        ``VoltageLevel`` objects or integers.  Default::

            {
                "380": {"i": 1020, "wires": 4, "circuits": 4},
                "220": {"i": 1020, "wires": 4, "circuits": 4},
                "110": {"i": 1020, "wires": 4, "circuits": 2},
                "dc": 0,
            }

    grid_max_foreign : float or None
        Upper bound for border-crossing line expansion relative to existing
        capacity.  Mutually exclusive with ``grid_max_abs_foreign``.
        Default: ``4``.
    grid_max_abs_foreign : dict or None
        Absolute capacity limits per voltage level for border-crossing lines,
        defined in the same way as ``grid_max_abs_D``.  Default: ``None``.
    """

    grid_max_D: Optional[float] = Field(
        None,
        description=(
            "Relative upper bound for domestic grid expansion [p.u. of existing "
            "capacity].  Use instead of grid_max_abs_D."
        ),
    )
    grid_max_abs_D: Optional[dict[str, Union[VoltageLevel, int]]] = Field(
        None,
        description=(
            "Absolute maximum capacity per voltage level for domestic lines. "
            "Keys: '380', '220', '110', 'dc'."
        ),
    )
    grid_max_foreign: Optional[float] = Field(
        None,
        description=(
            "Relative upper bound for border-crossing line expansion [p.u. of "
            "existing capacity].  Use instead of grid_max_abs_foreign."
        ),
    )
    grid_max_abs_foreign: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Absolute capacity limits per voltage level for border-crossing lines, "
            "defined analogously to grid_max_abs_D."
        ),
    )


class Extendable(BaseModel):
    """
    Configuration for component extendability and grid-expansion upper bounds.

    Attributes
    ----------
    extendable_components : list of str
        Defines which components are included in the capacity optimisation.
        Possible entries:

        * ``'as_in_db'`` – leave everything as defined in the database.
        * ``'network'`` – set all lines, links, and transformers extendable.
        * ``'german_network'`` – set German lines and transformers extendable.
        * ``'foreign_network'`` – set foreign lines and transformers extendable.
        * ``'transformers'`` – set all transformers extendable.
        * ``'storages'`` / ``'stores'`` – allow unlimited extendable storage
          at each grid node.

        Default: ``['as_in_db']``.
    upper_bounds_grid : UpperBoundsGrid
        Upper bounds for grid expansion (domestic and border-crossing lines).
    """

    extendable_components: list[str] = Field(
        default_factory=lambda: ["as_in_db"],
        description=(
            "Components included in capacity optimisation. Options: 'as_in_db', "
            "'network', 'german_network', 'foreign_network', 'transformers', "
            "'storages', 'stores'."
        ),
    )
    upper_bounds_grid: UpperBoundsGrid = Field(default_factory=UpperBoundsGrid)


# ---------------------------------------------------------------------------
# Sub-models: Network clustering
# ---------------------------------------------------------------------------


class NetworkClusteringEhv(BaseModel):
    """
    Settings for optional extra-high-voltage (EHV) clustering of the
    electrical network.

    Attributes
    ----------
    active : bool
        If True, the full HV/EHV dataset is clustered down to EHV buses only.
        All HV buses are assigned to their closest EHV substation based on
        shortest electrical distance.  Default: ``False``.
    busmap : bool or str
        ``False`` to compute a new busmap, or a path to a stored busmap CSV to
        skip recomputation.  Default: ``False``.
    cpu_cores : int or str
        Number of CPU cores used during clustering.  Use ``'max'`` to utilise
        all available cores.  Default: ``4``.
    """

    active: bool = Field(
        False,
        description=(
            "If True, cluster the full HV/EHV dataset down to EHV buses, "
            "assigning each HV bus to its closest EHV substation."
        ),
    )
    busmap: Union[bool, str] = Field(
        False,
        description="False to recompute, or path to a stored busmap CSV.",
    )
    cpu_cores: Union[int, Literal["max"]] = Field(
        4,
        description="CPU cores for clustering; 'max' uses all available cores.",
    )


class ClusteringMethod(BaseModel):
    """
    General algorithm settings shared by all network clustering steps.

    Attributes
    ----------
    focus_region : None, str, or list of str
        Defines a focus region for clustering.  A higher spatial resolution
        is applied inside and around this region.  Provide a path to a
        shape-file or a list of Kreisnamen (one connected region with defined
        CRS).  Default: ``None``.
    per_country : bool
        If True, clusters are constrained to one cluster per foreign country.
        If False, AC buses inside and outside Germany are clustered together.
        Default: ``True``.
    algorithm : str
        Clustering algorithm.  Options:

        * ``'kmeans'`` – considers geographical bus locations.
        * ``'kmedoids-dijkstra'`` – considers electrical distances between buses.

        Default: ``'kmedoids-dijkstra'``.
    remove_stubs : bool
        If True, remove stub branches before k-means clustering to reduce
        overestimation of line meshes.  Only used with k-means.
        Default: ``False``.
    use_reduced_coordinates : bool
        If True, take cluster coordinates from the busmap rather than
        averaging them.  Only used with k-means.  Default: ``False``.
    line_length_factor : float
        Factor applied to the crow-flies distance between new buses to obtain
        new line lengths.  Default: ``1``.
    random_state : int
        Random seed for reproducible clustering results.  Default: ``42``.
    n_init : int
        Number of initialisations for the clustering algorithm.  Only change
        when necessary (see sklearn documentation).  Default: ``10``.
    max_iter : int
        Maximum iterations for the clustering algorithm.  Only change when
        necessary (see sklearn documentation).  Default: ``100``.
    tol : float
        Convergence tolerance for the clustering algorithm.  Only change when
        necessary (see sklearn documentation).  Default: ``1e-6``.
    cpu_cores : int or str
        Number of CPU cores used during clustering.  Use ``'max'`` for all
        available cores.  Default: ``4``.
    """

    focus_region: Optional[Union[str, list[str]]] = Field(
        None,
        description=(
            "Focus region for clustering (higher spatial resolution applied). "
            "Provide a shape-file path or a list of Kreisnamen."
        ),
    )
    per_country: bool = Field(
        True,
        description=(
            "If True, restrict clusters so that each foreign country forms at "
            "most one cluster."
        ),
    )
    algorithm: Literal["kmeans", "kmedoids-dijkstra"] = Field(
        "kmedoids-dijkstra",
        description=(
            "Clustering algorithm: 'kmeans' (geographic) or "
            "'kmedoids-dijkstra' (electrical distance)."
        ),
    )
    remove_stubs: bool = Field(
        False,
        description="Remove stubs before k-means clustering (k-means only).",
    )
    use_reduced_coordinates: bool = Field(
        False,
        description=(
            "Use busmap coordinates instead of averaged coordinates "
            "(k-means only)."
        ),
    )
    line_length_factor: float = Field(
        1.0,
        description=(
            "Multiplier applied to the crow-flies distance between new buses "
            "to derive new line lengths."
        ),
    )
    random_state: int = Field(42, description="Random seed for reproducibility.")
    n_init: int = Field(
        10,
        description="Number of algorithm initialisations (see sklearn docs).",
    )
    max_iter: int = Field(
        100,
        description="Maximum clustering iterations (see sklearn docs).",
    )
    tol: float = Field(
        1e-6,
        description="Convergence tolerance (see sklearn docs).",
    )
    cpu_cores: Union[int, Literal["max"]] = Field(
        4,
        description="CPU cores for clustering; 'max' uses all available.",
    )


class ElectricityGrid(BaseModel):
    """
    Clustering settings for the AC electricity grid.

    Attributes
    ----------
    active : bool
        If True, AC buses are clustered down to ``n_clusters`` nodes.
        Default: ``True``.
    cluster_within_focus : bool
        If False, AC buses within the focus region will not be clustered.
        Default: ``False``.
    n_clusters : int
        Total number of resulting AC nodes.  Includes foreign nodes when
        foreign AC clustering is enabled.  Default: ``30``.
    k_elec_busmap : bool or str
        ``False`` to recompute the clustering, or a path to a busmap CSV from
        a previous AC clustering run.  When a path is provided, ``n_clusters``
        is ignored.  Default: ``False``.
    """

    active: bool = Field(
        True,
        description="If True, cluster AC buses to n_clusters nodes.",
    )
    cluster_within_focus: bool = Field(
        False,
        description="If False, AC buses within the focus region are not clustered.",
    )
    n_clusters: int = Field(
        30,
        ge=1,
        description="Target number of AC nodes after clustering.",
    )
    k_elec_busmap: Union[bool, str] = Field(
        False,
        description=(
            "False to recompute, or path to a busmap CSV from a previous "
            "AC clustering run (n_clusters is ignored when a path is given)."
        ),
    )


class GasGrids(BaseModel):
    """
    Clustering settings for the CH4 and H2 gas grids.

    Attributes
    ----------
    active : bool
        If True, gas grid buses are clustered.  Default: ``True``.
    cluster_within_focus : bool
        If False, gas grid buses within the focus region are barely clustered.
        Default: ``False``.
    n_clusters_ch4 : int
        Total number of resulting CH4 nodes.  Default: ``15``.
    n_clusters_h2 : int
        Total number of resulting H2 nodes.  Default: ``15``.
    k_ch4_busmap : bool or str
        ``False`` to recompute, or path to a CH4 busmap CSV from a previous
        run (``n_clusters_ch4`` is ignored when a path is provided).
        Default: ``False``.
    k_h2_busmap : bool or str
        ``False`` to recompute, or path to an H2 busmap CSV.
        Default: ``False``.
    sector_coupled_clustering : bool
        If True, apply clustering to sector-coupled carriers such as
        ``central_heat`` (settings in ``cluster/gas.py``).  Default: ``True``.
    """

    active: bool = Field(True, description="If True, cluster gas grid buses.")
    cluster_within_focus: bool = Field(
        False,
        description="If False, gas grid buses in the focus region are barely clustered.",
    )
    n_clusters_ch4: int = Field(15, ge=1, description="Target number of CH4 nodes.")
    n_clusters_h2: int = Field(15, ge=1, description="Target number of H2 nodes.")
    k_ch4_busmap: Union[bool, str] = Field(
        False,
        description=(
            "False to recompute, or path to a CH4 busmap CSV "
            "(n_clusters_ch4 is ignored when a path is given)."
        ),
    )
    k_h2_busmap: Union[bool, str] = Field(
        False,
        description="False to recompute, or path to an H2 busmap CSV.",
    )
    sector_coupled_clustering: bool = Field(
        True,
        description=(
            "If True, apply clustering to sector-coupled carriers such as "
            "central_heat (see cluster/gas.py)."
        ),
    )


class NetworkClustering(BaseModel):
    """
    Top-level container for all network-clustering settings (electricity and
    gas grids).
    """

    method: ClusteringMethod = Field(default_factory=ClusteringMethod)
    electricity_grid: ElectricityGrid = Field(default_factory=ElectricityGrid)
    gas_grids: GasGrids = Field(default_factory=GasGrids)


# ---------------------------------------------------------------------------
# Sub-models: Snapshot / temporal clustering
# ---------------------------------------------------------------------------


class SnapshotClustering(BaseModel):
    """
    Settings for temporal clustering: run the optimisation on a reduced
    subset of representative snapshots.

    Attributes
    ----------
    active : bool
        If True, snapshot clustering is applied.  Default: ``False``.
    method : str
        Clustering method.  Options: ``'typical_periods'`` or
        ``'segmentation'``.  Default: ``'segmentation'``.
    extreme_periods : None or str
        Method for incorporating extreme snapshots (time steps with extreme
        residual load) in the reduced time series.  Options: ``None``,
        ``'append'``, ``'new_cluster_center'``, ``'replace_cluster_center'``.
        Default: ``None`` (extreme periods are not considered).
    how : str
        Period definition used when ``method`` is ``'typical_periods'``.
        Options: ``'daily'``, ``'weekly'``, ``'monthly'``.  Default: ``'daily'``.
    storage_constraints : str
        Additional constraints for storage units when ``method`` is
        ``'typical_periods'``.  Options: ``'daily_bounds'``,
        ``'soc_constraints'``, ``'soc_constraints_simplified'``.
        Default: ``'soc_constraints'``.
    n_clusters : int
        Number of typical periods (only for ``method='typical_periods'``).
        Default: ``5``.
    n_segments : int
        Number of segments (only for ``method='segmentation'``).
        Default: ``5``.
    """

    active: bool = Field(False, description="If True, activate snapshot clustering.")
    method: Literal["typical_periods", "segmentation"] = Field(
        "segmentation",
        description="Clustering method: 'typical_periods' or 'segmentation'.",
    )
    extreme_periods: Optional[
        Literal["append", "new_cluster_center", "replace_cluster_center"]
    ] = Field(
        None,
        description=(
            "Method for handling extreme snapshots in the reduced time series. "
            "Options: None, 'append', 'new_cluster_center', 'replace_cluster_center'."
        ),
    )
    how: Literal["daily", "weekly", "monthly"] = Field(
        "daily",
        description="Period type for 'typical_periods' method.",
    )
    storage_constraints: Literal[
        "daily_bounds", "soc_constraints", "soc_constraints_simplified"
    ] = Field(
        "soc_constraints",
        description="Additional storage constraints for 'typical_periods' method.",
    )
    n_clusters: int = Field(
        5,
        ge=1,
        description="Number of typical periods (only for method='typical_periods').",
    )
    n_segments: int = Field(
        5,
        ge=1,
        description="Number of segments (only for method='segmentation').",
    )


class TemporalDisaggregation(BaseModel):
    """
    Settings for an optional second LOPF (dispatch only, no capacity
    optimisation) that disaggregates dispatch to the full temporal resolution.

    Note: load shedding is applied during this optimisation.  At present,
    ``skip_snapshots`` must be active and extra functionalities are
    disregarded.

    Attributes
    ----------
    active : bool
        If True, temporal disaggregation is performed.  Default: ``False``.
    no_slices : int
        Number of sub-problems the optimisation is divided into.  State-of-
        charge information for storage units and stores from the preceding
        optimisation is passed between slices.  Default: ``8``.
    """

    active: bool = Field(
        False,
        description="If True, run a dispatch-only LOPF to disaggregate to full temporal resolution.",
    )
    no_slices: int = Field(
        8,
        ge=1,
        description=(
            "Number of sub-problems the optimisation is split into; storage "
            "SoC information is passed between slices."
        ),
    )


# ---------------------------------------------------------------------------
# Sub-models: Foreign lines
# ---------------------------------------------------------------------------


class ForeignLines(BaseModel):
    """
    Transmission technology and capacity settings for border-crossing lines.

    Attributes
    ----------
    carrier : str
        Model foreign lines as ``'AC'`` lines or ``'DC'`` links.
        Default: ``'AC'``.
    capacity : str
        Data source for foreign line capacities.  Options: ``'osmTGmod'``,
        ``'tyndp2020'``, ``'ntc_acer'``, ``'thermal_acer'``.
        Default: ``'osmTGmod'``.
    """

    carrier: Literal["AC", "DC"] = Field(
        "AC",
        description="Model foreign lines as 'AC' or as 'DC' links.",
    )
    capacity: Literal["osmTGmod", "tyndp2020", "ntc_acer", "thermal_acer"] = Field(
        "osmTGmod",
        description=(
            "Capacity data source for foreign lines: 'osmTGmod', 'tyndp2020', "
            "'ntc_acer', or 'thermal_acer'."
        ),
    )


# ---------------------------------------------------------------------------
# Sub-models: Solver options
# ---------------------------------------------------------------------------


class SolverOptions(BaseModel):
    """
    Solver-specific settings to improve simulation time and solution quality.

    The defaults below are tuned for Gurobi.  Reset or adapt all values when
    switching to a different solver to avoid errors.

    ``model_config = {"extra": "allow"}`` permits arbitrary additional
    solver-specific keys.
    """

    BarConvTol: float = Field(1e-5, description="Barrier convergence tolerance.")
    FeasibilityTol: float = Field(1e-5, description="Feasibility tolerance.")
    method: int = Field(2, description="Algorithm method (2 = barrier).")
    crossover: int = Field(0, description="Crossover strategy (0 = disabled).")
    logFile: str = Field("solver_etrago.log", description="Path to solver log file.")
    threads: int = Field(4, ge=1, description="Number of solver threads.")
    BarHomogeneous: int = Field(
        1,
        description="Enable homogeneous barrier algorithm (1 = enabled).",
    )

    model_config = {"extra": "allow"}  # allow solver-specific extra options


# ---------------------------------------------------------------------------
# Sub-models: Branch capacity factor
# ---------------------------------------------------------------------------


class BranchCapacityFactor(BaseModel):
    """
    Global derating factors applied to line capacities, e.g. to approximate
    an (n-1) security criterion or for debugging purposes.

    The factor specifies the p.u. branch rating: e.g. ``0.5`` allows half the
    nominal line capacity.

    Attributes
    ----------
    HV : float
        Derating factor for HV lines.  Default: ``0.5``.
    eHV : float
        Derating factor for eHV lines.  Default: ``0.7``.
    """

    HV: float = Field(0.5, ge=0.0, le=1.0, description="Branch derating factor for HV lines [p.u.].")
    eHV: float = Field(0.7, ge=0.0, le=1.0, description="Branch derating factor for eHV lines [p.u.].")


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class EtragoArgs(BaseModel):
    """
    Top-level configuration for an eTraGo grid calculation run.

    Attributes
    ----------
    db : str
        Name of the database session setting stored in *config.ini* within
        *.etrago_database/* for a local database, or ``'oep'`` to load the
        model from the Open Energy Platform.
    gridversion : None or str
        Version number of the oedb data: ``None`` for *model_draft* (sandbox)
        or an explicit version string (e.g. ``'v0.4.6'``) for the grid schema.
    method : Method
        Method and settings for the network optimisation.
    pf_post_lopf : PfPostLopf
        Settings for an optional non-linear power flow after the LOPF.
    start_snapshot : int
        First hour of the scenario year to be calculated.  Default: ``1``.
    end_snapshot : int
        Last hour of the scenario year to be calculated.  When snapshot
        clustering is active, the selected range should cover the required
        number of periods / segments.  Default: ``2``.
    solver : str
        Preferred solver.  Options: ``'glpk'`` (open-source), ``'cplex'``,
        ``'gurobi'``.  Default: ``'gurobi'``.
    solver_options : SolverOptions
        Solver settings.  Defaults are tuned for Gurobi; reset when switching
        solvers.
    model_formulation : str
        PyPSA model formulation.  Options: ``'angles'``, ``'cycles'``,
        ``'kirchhoff'``, ``'ptdf'``.  ``'angles'`` works best for small
        networks; ``'kirchhoff'`` for larger ones.  Default: ``'kirchhoff'``.
    scn_name : str
        Scenario name.  See the Read the Docs documentation for available
        scenarios.
    scn_extension : None or list of str
        Extension scenarios added on top of the base network.  Existing lines
        replaced by new ones are dropped.  Extension data lives in
        ``extension``-tables (e.g. ``grid.egon_etrago_extension_line``).
        Available overlays:

        * ``'nep2021_confirmed'`` – new lines confirmed by the
          Bundesnetzagentur (NEP 2021).
        * ``'nep2021_c2035'`` – new lines planned in NEP 2021 scenario 2035 C.

        Default: ``None``.
    lpfile : bool or str
        ``False`` or a file path (``'/path/to/file.lp'``) to save the LP file.
        Default: ``False``.
    csv_export : bool or str
        ``False`` or a directory path to save results as CSV files.
        Default: ``False``.
    extendable : Extendable
        Component extendability settings and grid-expansion upper bounds.
    generator_noise : bool or int
        Apply a small random noise to generator marginal costs to avoid
        optima plateaus.  ``False`` to disable, or an integer seed for
        reproducibility.  Default: ``789456``.
    extra_functionality : dict
        Extra constraint functions and their parameters (see
        ``/tools/constraints.py``).  Available options:

        * ``'max_line_ext'`` (*float*) – maximum network extension share [p.u.].
        * ``'min_renewable_share'`` (*float*) – minimum renewable generation
          share [p.u.].
        * ``'cross_border_flow'`` (*[float, float]*) – AC cross-border flow
          limits for Germany [MWh]; ``[-x, y]`` with x = import, y = export.
        * ``'cross_border_flows_per_country'`` (*dict*) – per-country AC
          cross-border flow limits.
        * ``'capacity_factor'`` (*dict*) – overall energy production limits
          per carrier [p.u.].
        * ``'capacity_factor_per_gen'`` (*dict*) – per-generator energy
          production limits by carrier [p.u.].
        * ``'capacity_factor_per_cntr'`` (*dict*) – country-wise production
          limits per carrier [p.u.].
        * ``'capacity_factor_per_gen_cntr'`` (*dict*) – country-wise,
          per-generator production limits by carrier [p.u.].

    network_clustering_ehv : NetworkClusteringEhv
        Settings for optional HV->EHV bus clustering.
    network_clustering : NetworkClustering
        Settings for spatial clustering of the electricity and gas grids.
    spatial_disaggregation : None or str
        ``None`` to skip, or ``'uniform'`` for uniform spatial disaggregation.
    snapshot_clustering : SnapshotClustering
        Settings for temporal (snapshot) clustering.
    skip_snapshots : bool or int
        ``False`` to use all time steps, or an integer *n* to consider only
        every *n*-th time step.  Default: ``5``.
    temporal_disaggregation : TemporalDisaggregation
        Settings for a second dispatch-only LOPF that disaggregates results to
        full temporal resolution.
    branch_capacity_factor : BranchCapacityFactor
        Global p.u. derating factors for HV and eHV line capacities.
    load_shedding : bool
        If True, a very expensive generator is attached to every bus so that
        demand can always be met (useful for debugging).  Default: ``False``.
    foreign_lines : ForeignLines
        Transmission technology and capacity data source for border-crossing lines.
    comments : str or None
        Free-text comments for the run.
    """

    model_config = {"validate_assignment": True}

    # --- Setup & configuration ---
    db: Literal["oep", "local"] = Field(
        "oep",
        description=(
            "Database session: 'oep' to load from the Open Energy Platform, "
            "or the name of a local database session in config.ini."
        ),
    )
    gridversion: Optional[str] = Field(
        None,
        description=(
            "oedb data version: None for model_draft (sandbox) or an explicit "
            "version string, e.g. 'v0.4.6'."
        ),
    )

    # --- Optimisation method ---
    method: Method = Field(default_factory=Method)
    pf_post_lopf: PfPostLopf = Field(default_factory=PfPostLopf)

    # --- Time range ---
    start_snapshot: int = Field(
        1, ge=1, description="First hour of the scenario year to calculate."
    )
    end_snapshot: int = Field(
        168,
        ge=1,
        description=(
            "Last hour of the scenario year to calculate.  Must cover the "
            "required periods/segments when snapshot clustering is active."
        ),
    )

    @model_validator(mode="after")
    def _check_snapshot_order(self) -> "EtragoArgs":
        if self.end_snapshot < self.start_snapshot:
            raise ValueError("end_snapshot must be >= start_snapshot.")
        return self

    # --- Solver ---
    solver: Literal["glpk", "cplex", "gurobi"] = Field(
        "gurobi",
        description="Solver: 'glpk' (open-source), 'cplex', or 'gurobi'.",
    )
    solver_options: SolverOptions = Field(default_factory=SolverOptions)
    model_formulation: Literal["angles", "cycles", "kirchhoff", "ptdf"] = Field(
        "kirchhoff",
        description=(
            "PyPSA model formulation: 'angles', 'cycles', 'kirchhoff', or 'ptdf'. "
            "'kirchhoff' is recommended for large networks."
        ),
    )

    # --- Scenario ---
    scn_name: Literal[
        "eGon2035",
        "eGon2035_lowflex",
        "eGon100RE",
        "eGon100RE_lowflex",
        "status2019",
    ] = Field(
        "eGon2035",
        description="Scenario name (see Read the Docs for available options).",
    )
    scn_extension: Optional[list[str]] = Field(
        None,
        description=(
            "Extension scenario names added to the base network. "
            "Options: 'nep2021_confirmed', 'nep2021_c2035'."
        ),
    )

    # --- Export ---
    lpfile: Union[bool, str] = Field(
        False,
        description="False, or path to save the solver LP file.",
    )
    csv_export: Union[bool, str] = Field(
        "results",
        description="False, or directory path for CSV result export.",
    )

    # --- Grid expansion ---
    extendable: Extendable = Field(default_factory=Extendable)

    # --- Noise ---
    generator_noise: Union[bool, int] = Field(
        789456,
        description=(
            "False to disable generator noise, or an integer seed to apply "
            "reproducible marginal-cost noise and avoid optima plateaus."
        ),
    )

    # --- Extra functionality ---
    extra_functionality: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra constraint functions and parameters (see /tools/constraints.py). "
            "Keys: 'max_line_ext', 'min_renewable_share', 'cross_border_flow', etc."
        ),
    )

    # --- Spatial complexity ---
    network_clustering_ehv: NetworkClusteringEhv = Field(
        default_factory=NetworkClusteringEhv
    )
    network_clustering: NetworkClustering = Field(default_factory=NetworkClustering)
    spatial_disaggregation: Optional[Literal["uniform"]] = Field(
        None,
        description="None to skip spatial disaggregation, or 'uniform'.",
    )

    # --- Temporal complexity ---
    snapshot_clustering: SnapshotClustering = Field(default_factory=SnapshotClustering)
    skip_snapshots: Union[bool, int] = Field(
        5,
        description=(
            "False to use all time steps, or integer n to use every n-th step."
        ),
    )
    temporal_disaggregation: TemporalDisaggregation = Field(
        default_factory=TemporalDisaggregation
    )

    # --- Simplifications ---
    branch_capacity_factor: BranchCapacityFactor = Field(
        default_factory=BranchCapacityFactor
    )
    load_shedding: bool = Field(
        True,
        description=(
            "If True, attach a very expensive generator to every bus so that "
            "demand can always be met (useful for debugging)."
        ),
    )
    foreign_lines: ForeignLines = Field(default_factory=ForeignLines)

    # --- Misc ---
    comments: Optional[str] = Field(None, description="Free-text run comments.")


# ---------------------------------------------------------------------------
# Convenience: load args from YAML and validate
# ---------------------------------------------------------------------------


def load_etrago_config(path: str) -> EtragoArgs:
    """Load and validate an :class:`EtragoArgs` configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    EtragoArgs
        Validated configuration object.
    """
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return EtragoArgs(**raw)



def get_args_setting(self, path="scenario_setting.json"):
    """
    Load scenario settings for eTraGo ``args`` from a JSON or YAML file.

    The settings include all eTraGo specific arguments and parameters for
    a reproducible calculation. The file format is detected automatically
    from the file extension (``.json``, ``.yml``, ``.yaml``).

    Parameters
    ----------
    path : str
        Path to the scenario settings file (JSON or YAML).
        Default: ``'scenario_setting.json'``

    Returns
    -------
    None
        Sets ``self.args`` to the loaded (and optionally merged) settings,
        or ``None`` if the file does not exist.
    """
    if path is None:
        return

    ext = os.path.splitext(path)[-1].lower()

    with open(path) as f:
        if ext == ".json":
            import json
            data = json.load(f)
        elif ext in (".yml", ".yaml"):
            import yaml
            data = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. Use '.json', '.yml', or '.yaml'."
            )

    if hasattr(self, "args") and self.args is not None:
        self.args = merge_dicts(self.args, data)
    else:
        self.args = data


def merge_dicts(dict1, dict2):
    """
    Return a new dictionary by merging two dictionaries recursively.

    Parameters
    ----------
    dict1 : dict
        dictionary 1.
    dict2 : dict
        dictionary 2.

    Returns
    -------
    result : dict
        Union of dict1 and dict2

    """

    result = deepcopy(dict1)

    for key, value in dict2.items():
        if isinstance(value, Mapping):
            result[key] = merge_dicts(result.get(key, {}), value)
        else:
            result[key] = deepcopy(dict2[key])

    return result


def find_args_file(folder: str, stem: str = "args") -> str | None:
    """Return the first matching args file in *folder*, or None if not found."""
    for ext in (".json", ".yml", ".yaml"):
        candidate = os.path.join(folder, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None

if __name__ == "__main__":
    config = load_etrago_config("args_default.yml")
    print(config.model_dump_json(indent=2))