from pathlib import Path
from pprint import pformat
import platform

import nox

cleaned = [
    "etrago/analyze/__init__.py",
    "etrago/analyze/calc_results.py",
    "etrago/analyze/plot.py",
    "etrago/cluster/electrical.py",
    "etrago/cluster/gas.py",
    "etrago/cluster/snapshot.py",
    "etrago/cluster/spatial.py",
    "etrago/disaggregate/spatial.py",
    "etrago/disaggregate/temporal.py",
    "etrago/execute/__init__.py",
    "etrago/execute/grid_optimization.py",
    "etrago/execute/market_optimization.py",
    "etrago/network.py",
    "etrago/tools/extendable.py",
    "etrago/tools/io.py",
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
    # Get the current Python version and OS
    current_version = session.python if session.python else "unknown"
    current_os = platform.system()
    print(f"Running install on Python {current_version} and OS {current_os}")

    # Check if the current session is Python 3.9 on macOS and skip
    if current_version == "3.9" and current_os == "Darwin":
        session.skip("Skipping tests for Python 3.9 on macOS")

    setdefaults(session)
    session.install("twine")
    session.run("python", "setup.py", "bdist", "bdist_wheel")
    session.run("twine", "check", "dist/eTraGo*")


@nox.session(python=["3", "3.9", "3.10", "3.11"])
def install(session):
    """Install the package."""
    # Get the current Python version and OS
    current_version = session.python if session.python else "unknown"
    current_os = platform.system()
    print(f"Running install on Python {current_version} and OS {current_os}")

    # Check if the current session is Python 3.9 on macOS and skip
    if current_version == "3.9" and current_os == "Darwin":
        session.skip("Skipping tests for Python 3.9 on macOS")

    setdefaults(session)
    session.env["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "False"
    session.run("python", "-mpip", "install", "--upgrade", "pip")
    session.run("python", "-mpip", "install", *Path("dist").glob("*.whl"))
