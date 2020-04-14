#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:12:09 2019

@author: clara
"""

# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
"""
This is the application file for the tool eTraGo.
Define your connection parameters and power flow settings before executing
the function etrago.
"""


import datetime
import os
import os.path
import glob
from etrago.tools.utilities import get_args_setting


from etrago.appl import etrago

if __name__ == '__main__':
    # execute etrago function
    paths = glob.glob("/home/student/Clara/SCLOPF/args_lopf/sensitivitaeten/*.json")
    paths.sort()
    args= {}
    for json_path in paths:
        print('Setting ' + json_path )
        args = get_args_setting(args, jsonpath=json_path)
        print(datetime.datetime.now())
        network, disaggregated_network = etrago(args)
        print(datetime.datetime.now())
