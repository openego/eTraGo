============
Contributing
============

ANPASSEN FÜR ETRAGO

The research project eGo_n and egon-data are collaborative projects with
several people contributing to it. The following section gives an
overview of applicable guidelines and rules to enable a prospering
collaboration.
Any external contributions are welcome as well, and they are greatly
appreciated! Every little bit helps, and credit will always be given.


Bug reports and feature requests
================================

The best way to report bugs, inform about intended developments, send
feedback or propose a feature
is to file an issue at
https://github.com/openego/eGon-data/issues.

Please tag your issue with one of the predefined labels as it helps
others to keep track of unsolved bugs, open tasks and questions.

To inform others about intended developments please include:
* a describtion of the purpose and the value it adds
* outline the required steps for implementation
* list open questions

When reporting a bug please include all information needed to reproduce
the bug you found.
This may include information on

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.


Contribution guidelines
=======================


Development
-----------

Adding changes to the egon-data repository should follow some guidelines:

1. Create an `issue`_ in our `repository`_ to describe the intended
   developments briefly

   .. _issue: https://github.com/openego/eGon-data/issues
   .. _repository: https://github.com/openego/eGon-data

2. Create a branch for your issue related development from the
   dev-branch following our branch naming convention::

    git checkout -b `<prefix>/#<issue-id>-very-brief-description`

   where `issue-id` is the issue number on GitHub and `prefix` is one of

    - features
    - fixes
    - refactorings

   depending on which one is appropriate. This command creates a new
   branch in your local repository, in which you can now make your
   changes. Be sure to check out our `style conventions`_ so that your
   code is in line with them.
   If you don't have push rights to our `repository`_, you need to fork
   it via the "Fork" button in the upper right of the `repository`_
   page and work on the fork.

   .. _style conventions: `Code and Commit Style`_

3. Make sure to update the documentation along with your code changes

4. When you're done making changes run all the checks and docs builder
   with `tox <https://tox.readthedocs.io/en/latest/install.html>`_ one
   command:

   .. code-block:: bash

      tox

5. Commit your changes and push your branch to GitHub::

    git add -p
    git commit
    git push origin features/#<issue-id>-very-brief-description

  Note that the :code:`-p` switch will make :code:`git add` iterate
  through your changes and prompt for each one on whether you want to
  include it in the upcoming commit. This is useful if you made multiple
  changes which should conceptually be grouped into different commits,
  like e.g. fixing the documentation of one function and changing the
  implementation of an unrelated one in parallel, because it allows you
  to still make separate commits for these changes. It has the drawback
  of not picking up new files though, so if you added files and want to
  put them under version control, you have to add them explicitly by
  running :code:`git add FILE1 FILE2 ...` instead.

6. Submit a pull request through the GitHub website.


Code and Commit Style
---------------------

We try the adhere to the `PEP 8 Style Guide <PEP8_>`_ wherever possible.
In addition to that, we use `a code formatter` to have a consistent
style, even in cases where PEP 8 leaves multiple degrees of freedom. So
please run your code through :code:`black` before committing it. [#black]_
PEP 8 also specifies a way to group imports, onto which we put the
additional constraint that the imports within each group are ordered
alphabetically. Once again, you don't have to keep track of this
manually, but you can use `isort`_ to have imports sorted automatically.
Note that `pre-commit` hooks are configured for this repository, so you
can just :code:`pip install pre-commit` followed by :code:`pre-commit
install` in the repository, and every commit will automatically be
checked for style violations.

Unfortunately these tools don't catch everything, so here's a short list
of things you have to keep track of manually:

  - :code:`Black` can't automatically break up overly long strings, so
    make use of Python's automatic string concatenation feature by e.g.
    converting

    .. code-block:: python

      something = "A really really long string"

    into the equivalent:

    .. code-block:: python

      something = (
          "A really really"
          " long string"
      )

  - :code:`Black` also can't check whether you're using readable names
    for your variables. So please don't use abbreviations. Use `readable
    names`_.

  - :code:`Black` also can't reformat your comments. So please keep in
    mind that PEP 8 specifies a line length of 72 for free flowing text
    like comments and docstrings. This also extends to the documentation
    in reStructuredText files.

Last but not least, commit messages are a kind of documentation, too,
which should adhere to a certain style. There are quite a few documents
detailing this style, but the shortest and easiest to find is probably
https://commit.style. If you have 15 minutes instead of only five to
spare, there's also a very good and only `slightly longer article`_ on
this subject, containing references to other style guides, and also
explaining why commit messages are important.

At the very least, try to only commit small, related changes. If you
have to use an "and" when trying to summarize your changes, they should
probably be grouped into separate commits.

.. _a code formatter: https://pypi.org/project/black/
.. _slightly longer article: https://chris.beams.io/posts/git-commit/
.. _isort: https://pypi.org/project/isort/
.. _pre-commit: https://pre-commit.com
.. _readable names: https://chrisdone.com/posts/german-naming-convention/
.. [#black]
    If you want to be really nice, run any file you touch through
    :code:`black` before making changes, and commit the result
    separately from other changes.. The repository may contain wrongly
    formatted legacy code, and this way you commit eventually necessary
    style fixes separated from your actually meaningful changes, which
    makes the reviewers job a lot easier.

Pull Request Guidelines
-----------------------

We use pull requests (PR) to integrate code changes from branches.
PRs always need to be reviewed (exception proves the rule!). Therefore, ask
one of the other developers for reviewing your changes. Once approved, the PR
can be merged. Please delete the branch after merging.

Before requesting a review, please

1. Include passing tests (run ``tox``). [#tox-note]_
2. Let the workflow run in :ref:`Test mode` once from scratch to verify
   successful execution
3. Make sure that your changes are tested in integration with other
   tasks and on a complete run at least once by merging them into the
   `continuous-integration/run-everything-over-the-weekend`_ branch.
   This branch will regularly be checked out and tested on a complete
   workflow run on friday evening.
4. Update documentation when there's new API, functionality etc.
5. Add a note to ``CHANGELOG.rst`` about the changes and refer to the
   corresponding Github issue.
6. Add yourself to ``AUTHORS.rst``.

.. [#tox-note]
    If you don't have all the necessary Python versions available locally
    you can rely on CI via GitHub actions -
    it will `run the tests`_ for each change you add in the pull request.

    It will be slower though ...

.. _run the tests: https://github.com/openego/eGon-data/actions?query=workflow%3A%22Tests%2C+code+style+%26+coverage%22
.. _continuous-integration/run-everything-over-the-weekend: https://github.com/openego/eGon-data/tree/continuous-integration/run-everything-over-the-weekend


When requesting reviews, please keep in mind it might be a significant effort
to review the PR. Try to make it easier for them and keep the overall effort
as low as possible. Therefore,

* asking for reviewing specific aspects helps reviewers a lot to focus on the
  relevant parts
* when multiple people are asked for a review it should be avoided that they
  check/test the same things. Be even more specific what you expect from
  someone in particular.


What needs to be reviewed?
--------------------------

Things that definitely should be checked during a review of a PR:

* *Is the code working?* The contributor should already have made sure that
  this is the case. Either by automated test or manual execution.
* *Is the data correct?* Verifying that newly integrated and processed data
  is correct is usually not possible during reviewing a PR. If it is necessary,
  please ask the reviewer specifically for this.
* *Do tests pass?* See automatic checks.
* *Is the documentation up-to-date?* Please check this.
* *Was* ``CHANGELOG.rst`` *updated accordingly?* Should be the case, please
  verify.
* *Is metadata complete and correct (in case of data integration)?* Please
  verify. In case of a pending metadata creation make sure an appropriate
  issue is filed.


Extending the data workflow
===========================

The egon-data workflow uses Apache Airflow which organizes the order of
different processing steps and their execution.


How to add Python scripts
-------------------------

To integrate a new Python function to the egon-data workflow follow the
steps listed:

1. Add your well documented script to the egon-data repository
2. Integrate functions which need to be called within the workflow to
   pipeline.py, which organzies and calls the different tasks within the
   workflow
3. Define the interdependencies between the scripts by setting the task
   downstream to another required task
4. The workflow can now be triggered via Apache Airflow


Where to save (downloaded) data?
--------------------------------

If a task requires to retrieve some data from external sources which needs to
be saved locally, please use `CWD` to store the data. This is achieved by using

.. code-block:: python

  from pathlib import Path
  from urllib.request import urlretrieve

  filepath = Path(".") / "filename.csv"
  urlretrieve("https://url/to/file", filepath)

Add metadata
------------

Add a metadata for every dataset you create for describing data with
machine-readable information. Adhere to the OEP Metadata v1.4.1, you can
follow
`the example <https://github.com/OpenEnergyPlatform/oemetadata/blob/develop/metadata/latest/example.json>`_
to understand how the fields are used. Field are described in detail in the
`Open Energy Metadata Description`_.

You can obtain the metadata string from a table you created in SQL via

.. code-block:: sql

  SELECT obj_description('<SCHEMA>.<TABLE>'::regclass);

Alternatively, you can write the table comment directly to a JSON file by

.. code-block:: bash

  psql -h <HOST> -p <PORT> -d <DB> -U <USER> -c "\COPY (SELECT obj_description('<SCHEMA>.<TABLE>'::regclass)) TO '/PATH/TO/FILE.json';"

For bulk export of all DB's table comments you can use `this script
<https://gist.github.com/nesnoj/86145999eca8182f43c2bca36bcc984f>`_.
Please verify that your metadata string is in compliance with the OEP Metadata
standard version 1.4.1 using the `OMI tool
<https://github.com/OpenEnergyPlatform/omi>`_ (tool is shipped with eGon-data):

.. code-block:: bash

  omi translate -f oep-v1.4 -t oep-v1.4 metadata_file.json

If your metadata string is correct, OMI puts the keys in the correct order and
prints the full string (use `-o` option for export).

You may omit the fields `id` and `publicationDate` in your string as it will be
automatically set at the end of the pipeline but you're required to set them to
some value for a complete validation with OMI. For datasets published on the
OEP `id` will be the URL which points to the table, it will follow the pattern
`https://openenergy-platform.org/dataedit/view/SCHEMA/TABLE`.

For previous discussions on metadata, you may want to check
`PR 176 <https://github.com/openego/eGon-data/pull/176>`_.

Helpers
^^^^^^^

You can use the
`Metadata creator <https://meta.rl-institut.de/meta_creator/141>`_ **GUI**.
Fill the fields and hit `Edit JSON` to get the metadata string. Vice versa,
you can paste a metadata string into this box and the fields will be filled
automatically which may be helpful if you want to amend existing strings.

There are some **licence templates** provided in :py:mod:`egon.data.metadata`
you can make use of for fields 11.4 and 12 of the
`Open Energy Metadata Description`_. Also, there's a template for the
**metaMetadata** (field 16).

There are some functions to quickly generate a template for the
**resource fields** (field 14.6.1 in `Open Energy Metadata Description`_) from
a SQLA table class or a DB table. This might be especially helpful if your
table has plenty of columns.

* From SQLA table class:
  :py:func:`egon.data.metadata.generate_resource_fields_from_sqla_model`
* From database table:
  :py:func:`egon.data.metadata.generate_resource_fields_from_db_table`

Sources
^^^^^^^

The **sources** (field 11) are the most important parts of the metadata which
need to be filled manually. You may also add references to tables in eGon-data
(e.g. from an upstream task) so you don't have to list all original sources
again. Make sure you include all upstream attribution requirements.

The following example uses various input datasets whose attribution must be
retained:

.. code-block:: python

  "sources": [
      {
          "title": "eGo^n - Medium voltage grid districts",
          "description": (
              "Medium-voltage grid districts describe the area supplied by "
              "one MV grid. Medium-voltage grid districts are defined by one "
              "polygon that represents the supply area. Each MV grid district "
              "is connected to the HV grid via a single substation."
          ),
          "path": "https://openenergy-platform.org/dataedit/view/"
                  "grid/egon_mv_grid_district", # "id" in the source dataset
          "licenses": [
              license_odbl(attribution=
                  "© OpenStreetMap contributors, 2021; "
                  "© Statistische Ämter des Bundes und der Länder, 2014; "
                  "© Statistisches Bundesamt, Wiesbaden 2015; "
                  "(Daten verändert)"
              )
          ]
      },
      # more sources...
  ]

.. _Open Energy Metadata Description: https://github.com/OpenEnergyPlatform/oemetadata/blob/develop/metadata/v141/metadata_key_description.md

Adjusting test mode data
------------------------

When integrating new data or data processing scripts, make sure the
:ref:`Test mode` still works correctly on a limited subset of data.
In particular, if a new external data sources gets integrated make sure the
data gets cut to the region of the test mode.


Documentation
=============

eGon-data could always use more documentation, whether as part of the
official eGon-data docs, in docstrings, or even in articles, blog posts
or similar resources. Always keep in mind to update the documentation
along with your code changes though.

The changes of the documentation in a feature branch get visible once a
pull request is opened.

How to document Python scripts
------------------------------

Use docstrings to document your Python code. Note that PEP 8 also
contains a `section <PEP8-docstrings_>`_ on docstrings and that there is
a whole `PEP <PEP257_>`_ dedicated to docstring conventions. Try to
adhere to both of them.
Additionally every Python script needs to contain a header describing
the general functionality and objective and including information on
copyright, license and authors.

.. code-block:: python

   """ Provide an example of the first line of a module docstring.

   This is an example header describing the functionalities of a Python
   script to give the user a general overview of what's happening here.
   """

   __copyright__ = "Example Institut"
   __license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
   __url__ = "https://github.com/openego/eGon-data/blob/main/LICENSE"
   __author__ = "github_alias1, github_alias2"


How to document SQL scripts
---------------------------

Please also add a similar header to your SQL scripts to give users and
fellow developers an insight into your scripts and the methodologies
applied. Please describe the content and objectives of the script
briefly but as detailed as needed to allow other to comprehend how it
works.

.. code-block:: SQL

   /*
   This is an example header describing the functionalities of a SQL
   script to give the user a general overview what's happening here

   __copyright__ = "Example Institut"
   __license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
   __url__ = "https://github.com/openego/eGon-data/blob/main/LICENSE"
   __author__ = "github_alias1, github_alias2"
   */



You can build the documentation locally with (executed in the repos root
directory)

.. code-block:: bash

   sphinx-build -E -a docs docs/_build/

Eventually, you might need to install additional dependencies for building the
documenmtation:

.. code-block:: bash

   pip install -r docs/requirements.txt


Tips
----

To run a subset of tests::

    tox -e envname -- pytest -k test_myfeature

To run all the test environments in *parallel*::

    tox -p auto


.. _PEP8: https://www.python.org/dev/peps/pep-0008
.. _PEP8-docstrings: https://www.python.org/dev/peps/pep-0008/#documentation-strings
.. _PEP257: https://www.python.org/dev/peps/pep-0257/
