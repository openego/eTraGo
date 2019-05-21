import os
from setuptools import find_packages, setup

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = ("ulfmueller, wolfbunke, BartelsJ, ClaraBuettner, gnn, " 
              "simnh, lukasol, s3pp, MGlauer, kimvk, MarlonSchlemminger, " 
              "mariusves")


setup(
    name='eTraGo',
    author='DLR VE, ZNES Flensburg',
    author_email='',
    description="electric transmission grid optimization",
    version='0.7.1',
    url='https://github.com/openego/eTraGo',
    license="GNU Affero General Public License Version 3 (AGPL-3.0)",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['egoio == 0.4.5',
                      'scikit-learn == 0.19.0',
                      'pandas >= 0.19.0, <=0.20.3',
                      'pypsa==0.11.0fork',
                      'sqlalchemy >= 1.0.15, <= 1.1.4',
                      'geoalchemy2 >= 0.3.0, <=0.4.0',
                      'matplotlib >= 1.5.3, <=1.5.3',
                      'tsam==0.9.9',
                      'shapely',
                      'oedialect'],
    dependency_links=[
        ('git+https://git@github.com/openego/PyPSA.git'
         '@master#egg=pypsa-0.11.0fork')],
    extras_require={
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme']},
    package_data={
        'etrago': [
            os.path.join('tools', '*.json')]
    }
)
