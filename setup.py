from os.path import dirname, join
import io
import re

from setuptools import find_packages, setup

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Center for Sustainable Energy Systems Flensburg, "
    "DLR-Institute of Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Center for Sustainable Energy Systems Flensburg, "
    "DLR-Institute of Networked Energy Systems"
)


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fh:
        return fh.read()


setup(
    name="eTraGo",
    author="Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Center for Sustainable Energy Systems Flensburg, "
    "DLR - Institute of Networked Energy Systems",
    author_email="",
    description="electric transmission grid optimization",
    long_description="{}".format(
        re.compile("^.. start-badges.*^.. end-header", re.M | re.S).sub(
            "", read("README.rst")
        )
    ),
    long_description_content_type="text/x-rst",
    version="0.10.1",
    url="https://github.com/openego/eTraGo",
    license="GNU Affero General Public License Version 3 (AGPL-3.0)",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geoalchemy2 >= 0.3.0",
        "geopandas",
        "keyring",
        "linopy==0.3.2",
        "loguru",
        "matplotlib >= 3.0.3, <3.9",
        "oedialect",
        "pandas < 2.2",
        "pyomo < 6.6",
        "pypsa == 0.26.2",
        "rtree",
        "saio",
        "scikit-learn",
        "setuptools >= 54.2.0",
        "shapely",
        "sqlalchemy < 1.4",
        "tables",
        "tilemapbase == 0.4.5",
        "tsam",
        "xarray<2024.07.2",
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
