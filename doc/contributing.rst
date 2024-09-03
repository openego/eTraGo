.. _Contributing_ref:
============
Contributing
============

ANPASSEN FÜR ETRAGO

The package *eTraGo* is a collaborative projects with several people
contributing to it. The following section gives an overview of
applicable guidelines and rules to enable a prospering collaboration.
Any external contributions are welcome as well, and they are greatly
appreciated! Every little bit helps, and credit will always be given.


Bug reports and feature requests
================================

The best way to report bugs, inform about intended developments, send
feedback or propose a feature
is to file an issue at
https://github.com/openego/eTraGo/issues.

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

Adding changes to the *eTraGo* repository should follow some guidelines:

1. Create an `issue`_ in our `repository`_ to describe the intended
   developments briefly

   .. _issue: https://github.com/openego/eTraGo/issues
   .. _repository: https://github.com/openego/eTraGo

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

When requesting reviews, please keep in mind it might be a significant effort
to review the PR. Try to make it easier for them and keep the overall effort
as low as possible. Therefore,

* asking for reviewing specific aspects helps reviewers a lot to focus on the
  relevant parts
* when multiple people are asked for a review it should be avoided that they
  check/test the same things. Be even more specific what you expect from
  someone in particular.



Documentation
=============

*eTraGo* could always use more documentation, whether as part of the
official *eTraGo* docs, in docstrings, or even in articles, blog posts
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
   __url__ = "https://github.com/openego/eTraGo/blob/main/LICENSE"
   __author__ = "github_alias1, github_alias2"



You can build the documentation locally with (executed in the repos root
directory)

.. code-block:: bash

   sphinx-build -E -a docs docs/_build/

Eventually, you might need to install additional dependencies for building the
documenmtation:

.. code-block:: bash

   pip install -r docs/requirements.txt


.. _PEP8: https://www.python.org/dev/peps/pep-0008
.. _PEP8-docstrings: https://www.python.org/dev/peps/pep-0008/#documentation-strings
.. _PEP257: https://www.python.org/dev/peps/pep-0257/
