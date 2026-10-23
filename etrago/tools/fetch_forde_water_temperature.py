"""Download Flensburg Förde water temperature from Copernicus Marine.

Baltic Sea Physics Reanalysis (BALTICSEA_MULTIYEAR_PHY_003_011), daily
mean potential temperature (thetao), about 1 nmi grid. Writes a NetCDF of
the Förde box and a CSV with the daily temperature at the water cell
nearest to the SWFL heat-pump intake, at the chosen intake depth.

Needs a Copernicus Marine account; run `copernicusmarine login` once.

Usage:
    python fetch_forde_water_temperature.py --years 2011 --out-dir forde_temperature
"""

import argparse
from pathlib import Path

import copernicusmarine
import numpy as np
import xarray as xr

DATASET = "cmems_mod_bal_phy_my_P1D-m"
# Box around the Flensburg Förde (inner and outer part).
BOX = {"lat": (54.76, 54.96), "lon": (9.38, 9.98)}
# SWFL power plant / heat-pump intake at the Flensburg harbour.
INTAKE = (54.797, 9.437)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2011])
    parser.add_argument("--depth-m", type=float, default=5.0)
    parser.add_argument("--out-dir", default="forde_temperature")
    a = parser.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nc = out / f"forde_thetao_{min(a.years)}_{max(a.years)}.nc"

    if not nc.exists():
        copernicusmarine.subset(
            dataset_id=DATASET,
            variables=["thetao", "siconc"],
            minimum_latitude=BOX["lat"][0],
            maximum_latitude=BOX["lat"][1],
            minimum_longitude=BOX["lon"][0],
            maximum_longitude=BOX["lon"][1],
            minimum_depth=0.0,
            maximum_depth=20.0,
            start_datetime=f"{min(a.years)}-01-01T00:00:00",
            end_datetime=f"{max(a.years)}-12-31T23:59:59",
            output_directory=str(out),
            output_filename=nc.name,
        )

    ds = xr.open_dataset(nc)
    temp = ds["thetao"].sel(depth=a.depth_m, method="nearest")

    # Nearest cell with water on every day.
    water = temp.notnull().all("time")
    lat2, lon2 = xr.broadcast(temp.latitude, temp.longitude)
    dist = np.hypot(
        (lat2 - INTAKE[0]) * 111.0,
        (lon2 - INTAKE[1]) * 111.0 * np.cos(np.radians(INTAKE[0])),
    ).where(water)
    i, j = np.unravel_index(int(dist.argmin(skipna=True)), dist.shape)
    cell = temp.isel(latitude=i, longitude=j)
    lat, lon = float(cell.latitude), float(cell.longitude)

    df = cell.to_dataframe()[["thetao"]].rename(columns={"thetao": "water_temp_c"})
    df.index = df.index.normalize()
    df.index.name = "date"
    csv = out / f"forde_water_temp_{min(a.years)}_{max(a.years)}_{a.depth_m:g}m.csv"
    df.to_csv(csv, float_format="%.3f")

    print(f"cell {lat:.4f} N {lon:.4f} E, {float(dist.min()):.1f} km from the intake, "
          f"depth {float(cell.depth):.1f} m")
    for limit in (3.0, 4.0, 5.0):
        print(f"days below {limit:.0f} °C: {int((df.water_temp_c < limit).sum())}")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
