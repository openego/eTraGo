import os
import logging
from etrago import Etrago


logger = logging.getLogger(__name__)


def get_some_results(df):
    # todo: compare something
    return df


def main(dir=None, export_p=None):
    """iterate dir for grids; compare components and export results to export_p
    """
    if not dir:
        logger.info("Theres no directory given, thus, no grids to compare.")
        return

    # init results df
    res_components_dct = {
        "buses": {}, "lines": {}, "links": {}, "generators": {},
        "loads": {}, "storage_units": {}, "transformers": {}
    }

    for grid_name in os.listdir(dir):
        import_path = os.path.join(dir, grid_name)
        etrago = Etrago(csv_folder_name=import_path)
        n = etrago.network

        for component in res_components_dct.keys():
            df = getattr(n, component)
            df = get_some_results(df)
            res_components_dct[component][grid_name] = df

    if export_p is not None:
        msg = ("TODO: do something simple with res_components_dct, "
               "e.g. v_nom.value_counts, line type, -length, s_nom, ...")
        print(msg)
        # export if export path is provided
        # df.to_csv(export_p)


if __name__ == "__main__":

    grids_dir = r"D:\grid_planning\powerd_data_validation\grids\pre_clustered"
    main(dir=grids_dir, export_p=grids_dir)
