import os
import json
import datetime
from etrago.appl import run_etrago


def main(args=None, json_path=None):
    # execute etrago function
    print(datetime.datetime.now())
    etrago = run_etrago(args=args, json_path=json_path)

    print(datetime.datetime.now())
    etrago.session.close()
    if hasattr(etrago, "ssh_server"):
        etrago.ssh_server.close()

# plots: more in tools/plot.py
    # make a line loading plot
    # etrago.plot_grid(
    # line_colors='line_loading', bus_sizes=0.0001, timesteps=range(2))
    # network and storage
    # etrago.plot_grid(
    # line_colors='expansion_abs',
    # bus_colors='storage_expansion',
    # bus_sizes=0.0001)
    # flexibility usage
    # etrago.flexibility_usage('DSM')


if __name__ == "__main__":

    _proxy = "path_to_local_proxy_conf.json"
    if _proxy:
        with open(_proxy) as f:
            _proxy = json.load(f)[os.name]
        os.environ["HTTP_PROXY"] = _proxy["HTTP_PROXY"]
        os.environ["HTTPS_PROXY"] = _proxy["HTTPS_PROXY"]

    # example for _ssh_tunnel config
    _ssh_tunnel = {
        "ip_remote_server": "ip_remote_server",
        "username_server": "username_remote_server",
        "pw_server": "password_remote_server",
        "username_db": "username_database_on_remote_server",
        "pw_db": "password_database_on_remote_server",
        "ip_PostgreSQL": "ip_PostgreSQL",  # e.g. "127.0.0.1"
        "port_PostgreSQL": "int('port_PostgreSQL')",
        "database_table": "database_table"  # e.g. "status2019"
    }

    _ssh_tunnel_path = r"path_to_local_ssh_tunnel_conf.json"
    if _ssh_tunnel_path:
        with open(_ssh_tunnel_path) as f:
            _ssh_tunnel = json.load(f)
    else:
        _ssh_tunnel = {}

    args = {}
    if not args:
        if _ssh_tunnel:  # only add db_ssh to args if your intention is to connect to a remote database
            args = {"db_ssh": _ssh_tunnel}
    # else: # merge dct e.g. args = args | _ssh_tunnel

    # get path of eTraGo dir and join with etrago, args_carlos_ehv.json
    json_path = os.path.join(os.path.dirname(os.getcwd()), "etrago", "args.json")

    main(args=args, json_path=json_path)
