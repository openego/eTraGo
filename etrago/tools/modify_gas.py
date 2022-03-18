# -*- coding: utf-8 -*-
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
import numpy as np
import pandas as pd


def H2_feedin_modeling(n, gas_network_carrier, H2_share):
    """Generate H2 feedin model

    Parameters
    ----------
    n : pypsa.Network
        Network to be modified.
    gas_network_carrier : str
        Name of the pipeline network carrier.
    """
    mix_carrier = "CH4_H2_mix"
    _feed_and_return_pipelines(n, gas_network_carrier)
    _generate_gas_mix_buses(n, gas_network_carrier, mix_carrier)
    _generate_gas_mix_pipelines(n, gas_network_carrier, mix_carrier)
    _feedin_links(n, gas_network_carrier, mix_carrier)
    _reconnect_links_loads(n, gas_network_carrier, mix_carrier)
    _reconnect_stores(n, gas_network_carrier, mix_carrier)


def _feed_and_return_pipelines(n, base_carrier):
    """Duplicate pipeline infrastructure for feed and return pipelines.

    Parameters
    ----------
    n : pypsa.Network
        Network to be modified.
    base_carrier : str
        Name of the pipeline network carrier.
    """
    # duplicate links, change order, index and rewrite to network
    links = n.links[n.links["carrier"] == base_carrier].copy()
    links.rename(columns={'bus0': 'bus1', 'bus1': 'bus0'})
    links.index = links.index + '_return'
    n.import_components_from_dataframe(links, "Link")


def _generate_gas_mix_buses(n, base_carrier, mix_carrier):
    """Generate gas mixture buses for every gas bus.

    Parameters
    ----------
    n : pypsa.Network
        Network to be modified.
    base_carrier : str
        Name of the pipeline network carrier.
    mix_carrier : str
        Name of the carrier for gasmixture.
    """
    # copy gas buses, adjust carrier, rename index and write to network
    buses = n.buses[n.buses["carrier"] == base_carrier].copy()
    buses["carrier"] = mix_carrier
    buses.index = buses.index + "_" + mix_carrier
    n.import_components_from_dataframe(buses, "Bus")


def _generate_gas_mix_pipelines(n, base_carrier, mix_carrier):
    """Create pipelines for gasmixture.

    Parameters
    ----------
    n : pypsa.Network
        Network to be modified.
    base_carrier : str
        Name of the pipeline network carrier.
    mix_carrier : str
        Name of the carrier for gasmixture.
    """
    # duplicate existing pipelines, adjust carrier, index and buses
    pipelines = n.links[n.links["carrier"] == base_carrier].copy()
    pipelines["carrier"] = mix_carrier
    pipelines.index = pipelines.index + "_" + mix_carrier
    pipelines["bus0"] = pipelines["bus0"] + "_" + mix_carrier
    pipelines["bus1"] = pipelines["bus1"] + "_" + mix_carrier
    n.import_components_from_dataframe(pipelines, "Link")


def _feedin_links(n, base_carrier, mix_carrier):
    """Create links for feedin of both gases into mixture bus.

    Parameters
    ----------
    n : pypsa.Network
        Network to be modified.
    base_carrier : str
        Name of the pipeline network carrier.
    mix_carrier : str
        Name of the carrier for gasmixture.
    """
    # select gas buses, generate link from gas bus to respective mix bus
    buses = n.buses[n.buses["carrier"] == base_carrier]
    links = pd.DataFrame()
    links["bus0"] = buses.index
    links["bus1"] = buses.index + "_" + mix_carrier
    links["p_nom"] = np.inf
    links["link_id"] = buses.index + "_" + base_carrier + "_input"
    links["carrier"] = base_carrier + "_feedin"
    n.import_components_from_dataframe(links.set_index("link_id"), "Link")

    # change the original H2 feedin target buses to mixture buses
    H2_feedin_filter = n.links["carrier"] == "H2_feedin"
    n.links.loc[H2_feedin_filter, "bus1"] += mix_carrier


def _reconnect_links_loads(n, base_carrier, mix_carrier, no_mixture=["CH4_to_H2"]):
    """Reconnect all outgoing links and loads from pipeline network.

    Parameters
    ----------
    n : pypsa.Network
        Network to be modified.
    base_carrier : str
        Name of the pipeline network carrier.
    mix_carrier : str
        Name of the carrier for gasmixture.
    no_mixture : list, optional
        Links, that do not use mixture, by default ["CH4_to_H2"]
    """
    # create mixture (base carrier + mixture carrier) output buses
    buses_base = n.buses[n.buses["carrier"] == base_carrier]
    buses = buses_base.copy()
    buses["carrier"] = mix_carrier + "_output"
    buses.index = buses.index + "_output"
    n.import_components_from_dataframe(buses, "Bus")

    # reconnect links
    # include all links with bus0 in base gas carrier
    m1 = n.links["bus0"].isin(n.buses[n.buses["carrier"] == base_carrier].index)
    # exclude pipelines
    m2 = n.links["carrier"] != base_carrier
    # exclude links to mix buses
    m3 = ~n.links["bus1"].isin(n.buses[n.buses["carrier"] == mix_carrier].index)
    # exlucde links which require pure base carrier
    m4 = ~n.links["bus1"].isin(n.buses[n.buses["carrier"] == base_carrier].index)
    m5 = ~n.links["carrier"].isin(no_mixture)
    n.links.loc[m1 & m2 & m3 & m4 & m5, "bus0"] += "_" + mix_carrier + "_output"

    # reconnect loads
    m1 = n.loads["carrier"] == base_carrier
    n.loads.loc[m1, "bus"] += "_" + mix_carrier + "_output"

    # create links to output buses
    links = pd.DataFrame()
    links["labels"] = buses.index + "_" + base_carrier + "_input"
    # source bus is the base carrier
    links["bus0"] = buses_base.index
    # target bus is the output bus
    links["bus1"] = buses.index
    links["p_nom"] = np.inf
    links["carrier"] = base_carrier
    n.import_components_from_dataframe(links.set_index("labels"), "Link")

    # copy data with mixture_carrier as source bus
    links = links.copy()
    links["labels"] = buses.index + "_" + mix_carrier + "_input"
    links["bus0"] += "_" + mix_carrier
    links["carrier"] = mix_carrier
    n.import_components_from_dataframe(links.set_index("labels"), "Link")


def _reconnect_stores(n, base_carrier, mix_carrier):
    """Reconnect all stores.

    Parameters
    ----------
    n : pypsa.Network
        Network to be modified.
    base_carrier : str
        Name of the pipeline network carrier.
    mix_carrier : str
        Name of the carrier for gasmixture.
    """
    # select and duplicate all CH4 stores
    # reconnect copied stores to mix_buses
    # recalculate capacity (???)
    pass


def compressor_modeling():
    """
    """
    pass

def H2_pipeline_modeling():
    """
    """
    pass

def H2_retrofit_modeling():
    """
    """
    pass

def generate_extra_functionalities():
    """
    """
    pass
