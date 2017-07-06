__copyright__ = "Reiner Lemoine Institut, ZNES, Next Energy, IKS Uni Magdeburg"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "mariusves"


from setuptools import find_packages, setup

setup(name='etrago',
      author='openego development group',
      author_email='',
      description='Powerflow analysis based on PyPSA',
      version='0.1',
      license="GNU Affero General Public License Version 3 (AGPL-3.0)",
      packages=find_packages(),
      install_requires=['egoio = 0.2.0',
                        'egopowerflow = 0.0.4'],
	  dependency_links=['git+ssh://git@github.com/openego/PyPSA.git@dev#egg=PyPSA']
     )
