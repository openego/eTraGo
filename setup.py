__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität Flensburg, Centre for Sustainable Energy Systems, DLR-Institute for Networked Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "mariusves"


from setuptools import find_packages, setup

setup(name='eTraGo',
      author='DLR VE, ZNES Flensburg',
      author_email='',
      description='electrical Transmission Grid Optimization of flexibility options for transmission grids based on PyPSA',
      version='0.3',
	  url='https://github.com/openego/eTraGo',
      license="GNU Affero General Public License Version 3 (AGPL-3.0)",
      packages=find_packages(),
      install_requires=['egoio == 0.2.11',
                        'egopowerflow == 0.0.5',
                        'scikit-learn == 0.19.0'],
	  dependency_links=['git+ssh://git@github.com/openego/PyPSA.git@dev#egg=PyPSA'],
      extras_require={
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme']}
     )
