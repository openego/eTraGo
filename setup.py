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
        "geoalchemy2 >= 0.3.0",
        "geopandas",
        "matplotlib >= 3.0.3",
        "oedialect",
        "pypsa == 0.20.1",
        "rtree",
        "saio",
        "scikit-learn",
        "setuptools >= 54.2.0",
        "shapely",
        "sqlalchemy",
        "tilemapbase == 0.4.5",
        "tsam == 1.1.0",
    ],
    extras_require={
        "docs": ["sphinx >= 1.4", "sphinx_rtd_theme"],
        "gurobipy": ["gurobipy"],
        "cartopy": ["cartopy", "requests"],
    },
    package_data={"etrago": [os.path.join("tools", "*.json")]},
)
