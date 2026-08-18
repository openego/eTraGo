---
name: Release
about: Checklist and workflow for preparing and publishing a new release
title: "[Release] vX.Y.Z"
labels: release
---

## Overview

This issue tracks the full release process, from milestone creation to publishing on PyPI. Use the checklists below and check off items as they are completed.

---

## 1. Road to Release

Initial setup and issue tracking for the upcoming release.

- [ ] Create a milestone for the release
- [ ] Add relevant issues to the milestone
- [ ] Work through and close the milestone issues
- [ ] Update documentation (Read the Docs)

---

## 2. Shortly Before Release

Final checks and testing before cutting the release branch.

- [ ] Re-check installation, ideally on different operating systems
- [ ] Run test calculations on the server:
  - [ ] Include larger test runs
  - [ ] Test different settings/configurations with specific note on any new features and their combinations
- [ ] Review existing minimal examples

### Create release branch

- [ ] Create a specific branch for the release
- [ ] Update version numbers and release date/year in:
  - [ ] `doc/conf.py`
  - [ ] `setup.py`
  - [ ] `etrago/__init__.py`
- [ ] Update "What's New" section in Read the Docs (RTD)

---

## 3. Release

Merging the release branch through to production.

- [ ] Create virtual environment `venv-release` and activate it, use it for the further process
- [ ] Install eTraGo one more time from the release branch
- [ ] Merge release branch into `dev`
- [ ] Switch to `master`
- [ ] Merge `dev` into `master` (locally) and push

---

## 4. GitHub Release

Creating the official GitHub release.

- [ ] Create new release on GitHub
- [ ] Set branch: `master`
- [ ] Set tag: version number
- [ ] Add description: copy from "What's New" (RTD)
- [ ] Run `git fetch` afterward to fetch the new tag

---

## 5. Publish to PyPI

All steps performed inside `venv-release`.

- [ ] Upgrade build tools:

      pip install --upgrade setuptools wheel twine build

- [ ] Build the package:

      python -m build .

- [ ] Check built files:

      twine check dist/*

- [ ] Ensure TestPyPI and PyPI tokens are available (see `.pypirc`)

### TestPyPI upload

- [ ] Upload to TestPyPI:

      twine upload --repository-url https://test.pypi.org/legacy/ dist/*

- [ ] Test installation from TestPyPI:

      pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple etrago

### PyPI upload

- [ ] Upload to PyPI:

      twine upload dist/*

---

## Reference

- See also: [How-To-Release from oemof](https://github.com/oemof/oemof/wiki/How-to-release)
