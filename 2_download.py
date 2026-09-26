"""
Stage 1 - File 2 (ROBUST): Download CH4 + wind stack from Earth Engine.

Fixes the "export succeeds but file is empty/metadata-only" problem by:
  1. Widening the AOI (a 1x1 degree box at 5km is only ~22x22 px).
  2. Averaging Sentinel-5P CH4 over a +/- window per day so days with no
     satellite overpass don't produce empty images.
  3. CHECKING data availability and sampling a pixel before exporting.
  4. Printing the band count so you can confirm before exporting.

Run 1_auth.py first. After this finishes, download stage1_stack.tif from the
Drive folder 'stage1_exports' and place it next to 3_build_dataset.py.
"""

import ee

PROJECT_ID = "methaneplume"

# Slightly larger AOI so the raster isn't microscopic. [west, south, east, north]
AOI_COORDS = [-104.2, 30.0, -102.4, 31.6]

START_DATE = "2024-06-17"
N_DAYS = 13

SCALE_M = 5000
CRS = "EPSG:4326"

# CH4 has sparse daily coverage. Average over a +/- window around each day
# so every day-slot has real pixels instead of an empty (masked) image.
CH4_WINDOW_DAYS = 5


def main():
    ee.Initialize(project=PROJECT_ID)

    aoi = ee.Geometry.Rectangle(AOI_COORDS)
    start = ee.Date(START_DATE)

    ch4_ic = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
        .select("CH4_column_volume_mixing_ratio_dry_air")
        .filterBounds(aoi)
    )
    era5_ic = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
        .filterBounds(aoi)
    )

    # Availability report so you can SEE there's data.
    ch4_count = ch4_ic.filterDate(start, start.advance(N_DAYS, "day")).size().getInfo()
    print(f"CH4 images intersecting AOI in window: {ch4_count}")
    if ch4_count == 0:
        raise SystemExit(
            "No Sentinel-5P CH4 images intersect this AOI/date range.\n"
            "Try a wider AOI, a different date range, or a longer window."
        )

    def build_day(i):
        day = start.advance(i, "day")

        lo = day.advance(-CH4_WINDOW_DAYS, "day")
        hi = day.advance(CH4_WINDOW_DAYS + 1, "day")
        ch4 = (
            ch4_ic.filterDate(lo, hi)
            .mean()
            .unmask(0)
            .toFloat()
            .rename("ch4")
        )

        wind = era5_ic.filterDate(day, day.advance(1, "day")).mean()
        u = wind.select("u_component_of_wind_10m").unmask(0).toFloat().rename("u")
        v = wind.select("v_component_of_wind_10m").unmask(0).toFloat().rename("v")

        img = ee.Image.cat([ch4, u, v]).toFloat().clip(aoi)
        img = img.reproject(crs=CRS, scale=SCALE_M)
        return img

    daily = [build_day(i) for i in range(N_DAYS)]
    stack = ee.Image.cat(daily).toFloat()

    n_bands = stack.bandNames().size().getInfo()
    print(f"Stack band count: {n_bands}  (expected {3 * N_DAYS})")

    sample = stack.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi.centroid(1),
        scale=SCALE_M,
        maxPixels=int(1e9),
    ).getInfo()
    non_null = sum(1 for v in sample.values() if v not in (None, 0))
    print(f"Non-zero sampled bands at centroid: {non_null}/{n_bands}")
    if non_null == 0:
        print("WARNING: sampled center pixel is all zero/null. "
              "Export may still be empty - widen AOI or window.")

    task = ee.batch.Export.image.toDrive(
        image=stack,
        description="stage1_full_stack",
        folder="stage1_exports",
        fileNamePrefix="stage1_stack",
        region=aoi,
        scale=SCALE_M,
        crs=CRS,
        maxPixels=int(1e13),
        fileFormat="GeoTIFF",
    )
    task.start()

    print("\nExport started.")
    print("Watch status at https://code.earthengine.google.com/tasks")
    print("When COMPLETED, download stage1_stack.tif from Drive/stage1_exports.")
    print(f"It should have {n_bands} bands.")


if __name__ == "__main__":
    main()
