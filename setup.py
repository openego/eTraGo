from os.path import dirname, join
import io
import re

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
    "mariusves",
    "CarlosEpia",
    "KathiEsterl",
    "pieterhexen",
    "fwitte",
    "AmeliaNadal",
)


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fh:
        return fh.read()


setup(
    name="eTraGo",
    author="DLR VE, ZNES Flensburg",
    author_email="",
    description="electric transmission grid optimization",
    long_description="{}".format(
        re.compile("^.. start-badges.*^.. end-header", re.M | re.S).sub(
            "", read("README.rst")
        )
    ),
    long_description_content_type="text/x-rst",
    version="0.9.0",
    url="https://github.com/openego/eTraGo",
    license="GNU Affero General Public License Version 3 (AGPL-3.0)",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geoalchemy2 >= 0.3.0",
        "geopandas",
        "keyring",
        "loguru",
        "matplotlib >= 3.0.3",
        "oedialect",
        # Fix upper version limits for pyomo and pandas
        # Related to problems with old pypsa version
        "pandas < 2",
        "pyomo>6.4, <6.6, !=6.4.3",
        "pypsa == 0.20.1",
        "rtree",
        "saio",
        "scikit-learn",
        "setuptools >= 54.2.0",
        "shapely",
        "sqlalchemy < 2",
        "tables < 3.9",
        "tilemapbase == 0.4.5",
        "tsam",
    ],
    extras_require={
        "docs": [
            "nbsphinx",
            "numpydoc",
            "sphinx >= 1.4",
            "sphinx_rtd_theme",
        ],
        "gurobipy": ["gurobipy"],
        "cartopy": ["cartopy", "requests"],
    },
    package_data={"etrago": [join("tools", "*.json")]},
)
