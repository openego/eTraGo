"""This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.
Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line."""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "tba"


from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


def oedb_session(section='oedb'):
    """Get SQLAlchemy session object with valid connection to OEDB"""

    # get session object by oemof.db tools (requires .oemof/config.ini
    try:
        from oemof import db
        conn = db.connection(section=section)

    except:
        print('Please provide connection parameters to database:')

        host = input('host (default 127.0.0.1): ') or '127.0.0.1'
        port = input('port (default 5432): ') or '5432'
        user = input('user (default postgres): ') or 'postgres'
        database = input('database name: ')
        password = input('password: ')

        conn = create_engine(
            'postgresql://' + '%s:%s@%s:%s/%s' % (user,
                                                  password,
                                                  host,
                                                  port,
                                                  database))

    Session = sessionmaker(bind=conn)
    session = Session()
    return session