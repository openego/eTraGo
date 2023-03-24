"""Multi purpose tools that don't fit anywhere else in eTraGo.
"""

__copyright__ = (
    "Copyright (C) 2023"
    " Otto-von-Guericke-University Magdeburg,"
    " Research group for theoretical computer science"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "gnn <gnn.code@gmail.com>"


def noop(*ignored_arguments, **ignored_keyword_arguments):
    """Do nothing.

    Accept all kinds of arguments, ignore them and do nothing.
    """
    pass


class Noops:
    """Provide arbitrarily named methods that do nothing.

    Any attribute access will return a method that does nothing, i.e.
    all methods of this object are :py:func:`noop`s. Normally you don't
    need to instantiate this class. All instances behave the same, so
    the containing module provides one called :py:obj:`noops` which you
    can import and use.
    """

    @classmethod
    def __getattribute__(cls, ignored_name):
        return noop


noops = Noops()
"""A default :py:class:`Noops` instance so you don't have to create one.
"""
