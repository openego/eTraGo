import etrago


def test_etrago_exposes_a_version():
    assert isinstance(etrago.__version__, str)
