from pprint import pformat

import nox

cleaned = [
    "etrago/cluster/disaggregation.py",
    "etrago/tools/network.py",
    "etrago/tools/utilities.py",
    "noxfile.py",
    "setup.py",
]


def setdefaults(session):
    session.env["PYTHONUNBUFFERED"] = "yes"


@nox.session(python="3")
def check(session):
    setdefaults(session)
    packages = ["Flake8-pyproject"]
    packages.extend(["black", "flake8", "isort >= 5", "twine"])
    session.install(*packages)
    assert cleaned == sorted(set(cleaned)), (
        "The list of cleaned files contains duplicates and/or isn't sorted"
        " alphabetically."
        f"\nExpected:\n{pformat(sorted(set(cleaned)))}"
        f"\nGot:\n{pformat(cleaned)}"
    )
    session.run("black", "--check", "--diff", *cleaned)
    session.run("isort", "--check-only", "--diff", *cleaned)
    session.run("flake8", *cleaned)
    session.run("python", "setup.py", "bdist", "bdist_wheel")
    session.run("twine", "check", "dist/eTraGo*")
