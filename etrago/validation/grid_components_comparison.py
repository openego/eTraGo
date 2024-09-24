import os
import logging
import pandas as pd
from etrago import Etrago

logger = logging.getLogger(__name__)


def main(dir=None, export_p=None):
    """iterate dir for grids; compare components and export results to export_p
    """
    if not dir:
        logger.info("Theres no directory given, thus, no grids to compare.")
        return

    # init results df
    df = pd.DataFrame()

    for grid_name in os.listdir(dir):
        import_path = os.path.join(dir, grid_name)
        etrago = Etrago(csv_folder_name=import_path)

    if export_p is not None:
        # export if export path is provided
        df.to_csv(export_p)


if __name__ == "__main__":

    grids_dir = r"D:\grid_planning\powerd_data_validation\grids\pre_clustered"
    main(dir=grids_dir, export_p=grids_dir)
