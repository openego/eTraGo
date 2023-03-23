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

import configparser as cp
import getpass
import os

from sqlalchemy import create_engine
import keyring
import oedialect


def readcfg(filepath, section):
    """
    Reads the configuration file. If section is not available, calls
    create_oedb_config_file to add the new section to an existing config.ini.

    Parameters
    ----------
    filepath : str
        Absolute path of config file including the filename itself
    section : str
        Section in config file which contains connection details
    Returns
    -------
    cfg : configparser.ConfigParser
        Used for configuration file parser language.
    """

    cfg = cp.ConfigParser()
    cfg.read(filepath)

    if not cfg.has_section(section):
        print(
            'The section "{sec}" is not in the config file {file}.'.format(
                sec=section, file=filepath
            )
        )
        cfg = create_oedb_config_file(filepath, section)

    return cfg


def get_connection_details(section):
    """
    Asks the user for the database connection details and returns them as a
    ConfigParser-object.

    Parameters
    ----------
    None

    Returns
    -------
    cfg : configparser.ConfigParser
        Used for configuration file parser language.
    """
    print("Please enter your connection details:")
    dialect = (
        input("Enter input value for `dialect` (default: psycopg2): ")
        or "psycopg2"
    )
    username = input("Enter value for `username`: ")
    database = input("Enter value for `database`: ")
    host = input("Enter value for `host`: ")
    port = input("Enter value for `port` (default: 5432): ") or "5432"

    cfg = cp.ConfigParser()
    cfg.add_section(section)
    cfg.set(section, "dialect", dialect)
    cfg.set(section, "username", username)
    cfg.set(section, "host", host)
    cfg.set(section, "port", port)
    cfg.set(section, "database", database)
    pw = getpass.getpass(
        prompt="Enter your password/token to "
        "store it in "
        "keyring: ".format(database=section)
    )
    keyring.set_password(section, cfg.get(section, "username"), pw)

    return cfg


def create_oedb_config_file(filepath, section="oep"):
    """

    Parameters
    ----------
    filepath : str
        Absolute path of config file including the filename itself
    section : str
        Section in config file which contains connection details

    Returns
    -------
    cfg : configparser.ConfigParser
        Used for configuration file parser language.
    """

    cfg = get_connection_details(section)

    print(
        "Do you want to store the connection details in the config file {file} ?".format(
            file=filepath
        )
    )
    choice = ""
    while choice not in ["y", "n"]:
        choice = input("(y/n): ")

    if choice == "y":
        # create egoio dir if not existent
        base_path = os.path.split(filepath)[0]
        if not os.path.isdir(base_path):
            os.mkdir(base_path)
            print("The directory {path} was created.".format(path=base_path))

        with open(filepath, "a") as configfile:
            cfg.write(configfile)
            pass

        print(
            'Template {0} with section "{1}" created.\nYou can manually edit'
            " the config file.".format(filepath, section)
        )
    else:
        pass

    return cfg


def connection(filepath=None, section="oep", readonly=False):
    """
    Instantiate a database connection (for the use with SQLAlchemy).

    The keyword argument `filepath` specifies the location of the config file
    that contains database connection information. If not given, the default
    of `~/.etrago_database/config.ini` applies.

    Parameters
    ----------
    filepath : str
        Absolute path of config file including the filename itself
    section : str
        Section in config file containing database connection parameters.
        Default: 'oep'.
    readonly : bool
        Set this option to True for creating a read-only and passwordless
        engine for accessing the open energy platform.
        Default: False.

    Returns
    -------
    conn : sqlalchemy.engine
        SQLalchemy engine object containing the connection details
    """

    if readonly:
        conn = create_engine("postgresql+oedialect://openenergy-platform.org")
    else:
        # define default filepath if not provided
        if filepath is None:
            filepath = os.path.join(
                os.path.expanduser("~"), ".etrago_database", "config.ini"
            )

        # does the file exist?
        if not os.path.isfile(filepath):
            print(
                "DB config file {file} not found. "
                "This might be the first run of the tool. ".format(
                    file=filepath
                )
            )
            cfg = create_oedb_config_file(filepath, section=section)
        else:
            cfg = readcfg(filepath, section)

        try:
            pw = cfg.get(section, "password")
        except:
            pw = keyring.get_password(section, cfg.get(section, "username"))
            if pw is None:
                pw = getpass.getpass(
                    prompt='No password found for database "{db}". '
                    "Enter your password to "
                    "store it in keyring: ".format(
                        db=cfg.get(section, "database")
                    )
                )
                keyring.set_password(section, cfg.get(section, "username"), pw)

        # establish connection and return it
        conn = create_engine(
            "postgresql+{dialect}://{user}:{password}@{host}:{port}/{db}".format(
                dialect=cfg.get(section, "dialect", fallback="psycopg2"),
                user=cfg.get(section, "username"),
                password=pw,
                host=cfg.get(section, "host"),
                port=cfg.get(section, "port"),
                db=cfg.get(section, "database"),
            )
        )

    return conn
