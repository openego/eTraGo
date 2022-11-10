import os

from setuptools import find_packages, setup

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = (
    "ulfmueller, wolfbunke, BartelsJ, ClaraBuettner, gnn, "
    "simnh, lukasol, s3pp, MGlauer, kimvk, MarlonSchlemminger, "
    "mariusves"
)


setup(
    name="eTraGo",
    author="DLR VE, ZNES Flensburg",
    author_email="",
    description="electric transmission grid optimization",
    version="0.8.0",
    url="https://github.com/openego/eTraGo",
    license="GNU Affero General Public License Version 3 (AGPL-3.0)",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "egoio == 0.4.7",
        "saio",
        "scikit-learn",
        "sqlalchemy == 1.3.16",
        "geoalchemy2 >= 0.3.0",
        "matplotlib >= 3.0.3",
        "pyomo < 6.4.1",
        "tsam == 1.1.0",
        "shapely",
        "oedialect",
        "tilemapbase == 0.4.5",
        "pypsa == 0.19.1",
        "setuptools >= 54.2.0",
        "geopandas",
        "rtree",
    ],
    extras_require={
        "docs": ["sphinx >= 1.4", "sphinx_rtd_theme"],
        "gurobipy": ["gurobipy"],
    },
    package_data={"etrago": [os.path.join("tools", "*.json")]},
)
