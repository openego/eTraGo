from pathlib import Path
from pprint import pformat

import nox

cleaned = [
    "etrago/cluster/disaggregation.py",
    "etrago/cluster/electrical.py",
    "etrago/cluster/gas.py",
    "etrago/cluster/snapshot.py",
    "etrago/cluster/spatial.py",
    "etrago/execute/__init__.py",
    "etrago/execute/grid_optimization.py",
    "etrago/execute/market_optimization.py",
    "etrago/tools/calc_results.py",
    "etrago/tools/extendable.py",
    "etrago/tools/io.py",
    "etrago/tools/network.py",
    "etrago/tools/utilities.py",
    "noxfile.py",
    "setup.py",
]


def setdefaults(session):
    session.env["PYTHONUNBUFFERED"] = "yes"


@nox.session(python="3")
def check(session):
    """Run custom checks."""
    setdefaults(session)
    assert cleaned == sorted(set(cleaned)), (
        "The list of cleaned files contains duplicates and/or isn't sorted"
        " alphabetically."
        f"\nExpected:\n{pformat(sorted(set(cleaned)))}"
        f"\nGot:\n{pformat(cleaned)}"
    )


@nox.session(python="3")
def black(session):
    """Check for happy little style accidents with `black`."""
    setdefaults(session)
    session.install("black")
    session.run("black", "--check", "--diff", *cleaned)


@nox.session(python="3")
def isort(session):
    """Check import ordering with `isort`."""
    setdefaults(session)
    session.install("isort >= 5")
    session.run("isort", "--check-only", "--diff", *cleaned)


@nox.session(python="3")
def flake8(session):
    """Check for happy little style accidents with `flake8`."""
    setdefaults(session)
    session.install("Flake8-pyproject", "flake8")
    session.run("flake8", "--ignore=E722, W605, W503, E203", *cleaned)


@nox.session(python=["3", "3.9", "3.10", "3.11"])
def build(session):
    """Build the package and check for packaging errors."""
    setdefaults(session)
    session.install("twine")
    session.run("python", "setup.py", "bdist", "bdist_wheel")
    session.run("twine", "check", "dist/eTraGo*")


@nox.session(python=["3", "3.9", "3.10", "3.11"])
def install(session):
    """Install the package."""
    setdefaults(session)
    session.env["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "False"
    session.run("python", "-mpip", "install", "--upgrade", "pip")
    session.run("python", "-mpip", "install", *Path("dist").glob("*.whl"))
